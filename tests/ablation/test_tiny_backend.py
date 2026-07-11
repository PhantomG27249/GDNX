from __future__ import annotations

import inspect
import math
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from typing import get_args, get_type_hints

import pytest
import torch
import torch.nn.functional as F

import research.kmd2_ablation.tiny_backend as tiny_backend_module
from research.kmd2_ablation.config import CacheConfig
from research.kmd2_ablation.tasks import generate_task
from research.kmd2_ablation.tiny_backend import (
    TINY_BACKEND_SCHEMA_VERSION,
    TinyCellOutput,
    TinyFactorProjector,
    TinyFactors,
    TinyKMD2Cell,
    TinyKMD2Config,
    TinyKMD2Model,
    TinyModelOutput,
    future_query_relevance,
    tiny_factors_from_episode,
)


def _config(**overrides: object) -> TinyKMD2Config:
    values: dict[str, object] = {
        "d_model": 12,
        "heads": 1,
        "dk": 2,
        "dv": 3,
        "layers": 1,
        "vocab_size": 16,
        "d_ff": 24,
        "r_out": 1,
        "mimo_rank": 1,
        "continuous_input_dim": 3,
        "output_dim": 1,
        "conv_kernel": 3,
        "dtype": torch.float32,
        "eps": 1.0e-6,
        "rotation_mode": "none",
        "convolution_gate_init": 0.0,
        "rotation_gate_init": 0.0,
        "channel_decay_gate_init": 0.0,
        "write_offset_gate_init": 0.0,
        "cache": None,
    }
    values.update(overrides)
    return TinyKMD2Config(**values)  # type: ignore[arg-type]


def _factors(
    *,
    batch: int = 1,
    steps: int = 3,
    heads: int = 1,
    q_slots: int = 1,
    write_slots: int = 1,
    dk: int = 2,
    dv: int = 1,
    requires_grad: bool = False,
) -> TinyFactors:
    q = torch.zeros(batch, steps, heads, q_slots, dk)
    k = torch.zeros(batch, steps, heads, write_slots, dk)
    q[..., 0] = 1.0
    k[..., 0] = 1.0
    mimo = write_slots > 1
    tensors = {
        "q": q.requires_grad_(requires_grad),
        "k": k.requires_grad_(requires_grad),
        "v": torch.arange(1, steps + 1, dtype=torch.float32)
        .view(1, steps, 1, 1, 1)
        .expand(batch, steps, heads, write_slots, dv)
        .clone()
        .requires_grad_(requires_grad),
        "decay": torch.ones(batch, steps, heads, dk).requires_grad_(requires_grad),
        "beta_e": torch.zeros(batch, steps, heads, write_slots).requires_grad_(
            requires_grad
        ),
        "beta_w": torch.ones(batch, steps, heads, write_slots).requires_grad_(
            requires_grad
        ),
        "out_mix": torch.full(
            (
                (batch, steps, heads, q_slots, dv)
                if mimo
                else (batch, steps, heads, q_slots)
            ),
            1.0 / q_slots,
        ).requires_grad_(requires_grad),
    }
    return TinyFactors(
        **tensors,
        valid=torch.ones(batch, steps, dtype=torch.bool),
        positions=torch.arange(steps, dtype=torch.int64).repeat(batch, 1),
        read_gate=(
            torch.ones(batch, steps, heads, q_slots, dv).requires_grad_(
                requires_grad
            )
            if mimo
            else None
        ),
    )


def test_tiny_api_shapes_and_validation() -> None:
    assert TINY_BACKEND_SCHEMA_VERSION == "1.1.0"
    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.dk = 4  # type: ignore[misc]
    with pytest.raises(ValueError, match="r_out.*mimo_rank"):
        _config(r_out=4, mimo_rank=2)
    with pytest.raises(ValueError, match="even"):
        _config(dk=3, rotation_mode="current")
    with pytest.raises(TypeError, match="dtype"):
        _config(dtype=torch.int64)

    factors = _factors(dv=3)
    assert factors.q.shape == (1, 3, 1, 1, 2)
    with pytest.raises(FrozenInstanceError):
        factors.q = factors.q.clone()  # type: ignore[misc]
    with pytest.raises(ValueError, match="beta_w"):
        TinyFactors(
            q=factors.q,
            k=factors.k,
            v=factors.v,
            decay=factors.decay,
            beta_e=factors.beta_e,
            beta_w=factors.beta_w[..., :0],
            out_mix=factors.out_mix,
            valid=factors.valid,
            positions=factors.positions,
        )

    cell = TinyKMD2Cell(config)
    output = cell(factors)
    assert isinstance(output, TinyCellOutput)
    assert output.read.shape == (1, 3, 1, 3)
    assert output.final_state.shape == (1, 1, 2, 3)
    assert output.scores.shape == (1, 3, 1)
    assert output.state_read.shape == output.cache_read.shape == output.read.shape
    assert output.selected_positions.shape == (1, 1, 0)
    assert output.sink_mass.shape == (1, 3, 1)
    assert output.state_bytes == 1 * 1 * 2 * 3 * 4
    assert output.cache_persistent_bytes == output.cache_block_bytes == 0
    with pytest.raises(FrozenInstanceError):
        output.read = output.read.clone()  # type: ignore[misc]

    shared_query = _factors(q_slots=4, dv=3)
    shared_output = TinyKMD2Cell(_config(r_out=4))(shared_query)
    assert shared_output.read.shape == (1, 3, 1, 3)

    mimo_config = _config(mimo_rank=2)
    mimo_factors = _factors(q_slots=2, write_slots=2, dv=3)
    assert mimo_factors.q.shape[3] == mimo_factors.k.shape[3] == 2
    mimo_output = TinyKMD2Cell(mimo_config)(mimo_factors)
    assert mimo_output.read.shape == (1, 3, 1, 3)
    assert torch.isfinite(mimo_output.read).all()


@pytest.mark.parametrize(
    "score",
    [
        "coupled_paper",
        "residual_only",
        "write_value",
        "recency",
        "reservoir",
        "future_query_oracle",
    ],
)
def test_tiny_cache_score_modes_execute_without_changing_recurrence(score: str) -> None:
    factors = _factors(steps=6, dk=2, dv=2)
    native = TinyKMD2Cell(_config(dv=2))(factors)
    cell = TinyKMD2Cell(
        _config(
            dv=2,
            selector_seed=317,
            cache=CacheConfig(
                width=2,
                block_size=2,
                score=score,
                storage_dtype="fp32",
            ),
        )
    )
    future_relevance = None
    if score == "future_query_oracle":
        future_relevance = torch.tensor([[5.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    output = cell(factors, future_relevance=future_relevance)
    assert torch.equal(output.state_read, native.state_read)
    assert torch.equal(output.final_state, native.final_state)
    assert torch.isfinite(output.scores).all()
    assert output.scores.dtype == torch.float32
    assert output.selected_positions.shape == (1, 1, 2)
    if score == "recency":
        assert set(output.selected_positions.flatten().tolist()) == {4, 5}
    if score == "future_query_oracle":
        assert 0 in output.selected_positions.flatten().tolist()


def test_tiny_reservoir_selector_is_seeded_and_call_order_independent() -> None:
    factors = _factors(steps=12, dk=2, dv=1)

    def selected(seed: int) -> torch.Tensor:
        config = _config(
            dv=1,
            selector_seed=seed,
            cache=CacheConfig(
                width=3, block_size=2, score="reservoir", storage_dtype="fp32"
            ),
        )
        return TinyKMD2Cell(config)(factors).selected_positions

    first = selected(919)
    selected(7)
    assert torch.equal(first, selected(919))
    assert not torch.equal(first, selected(920))


def test_tiny_unbounded_cache_retains_every_entry_without_eviction() -> None:
    factors = _factors(steps=9, dk=2, dv=1)
    unbounded = TinyKMD2Cell(
        _config(
            dv=1,
            unbounded_cache=True,
            selector_seed=1,
            cache=CacheConfig(
                width=1, block_size=2, score="exact_outer", storage_dtype="fp32"
            ),
        )
    )(factors)
    bounded = TinyKMD2Cell(
        _config(
            dv=1,
            selector_seed=1,
            cache=CacheConfig(
                width=9, block_size=2, score="exact_outer", storage_dtype="fp32"
            ),
        )
    )(factors)
    assert unbounded.selected_positions.shape == (1, 1, 9)
    assert set(unbounded.selected_positions.flatten().tolist()) == set(range(9))
    assert unbounded.eviction_count == 0
    assert torch.equal(unbounded.cache_read, bounded.cache_read)
    assert unbounded.cache_persistent_bytes == bounded.cache_persistent_bytes


@pytest.mark.parametrize("unbounded_cache", [False, True])
def test_cache_diagnostic_budget_is_checked_before_scan_for_every_cache(
    unbounded_cache: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = tiny_backend_module._cache_diagnostic_allocation_bytes
    assert {"cache_width", "layers"} <= set(inspect.signature(estimator).parameters)
    steps = 4
    declared_width = 2
    cache_width = steps if unbounded_cache else declared_width
    estimated_bytes = estimator(
        batch=1,
        steps=steps,
        heads=1,
        query_slots=1,
        value_dim=1,
        cache_width=cache_width,
        block_size=2,
        per_slot_read=False,
        layers=3,
    )
    budget = estimated_bytes - 1
    assert hasattr(tiny_backend_module, "_MAX_CACHE_DIAGNOSTIC_BYTES")
    monkeypatch.setattr(tiny_backend_module, "_MAX_CACHE_DIAGNOSTIC_BYTES", budget)
    cell = TinyKMD2Cell(
        _config(
            dv=1,
            layers=3,
            unbounded_cache=unbounded_cache,
            cache=CacheConfig(
                width=declared_width, block_size=2, storage_dtype="fp32"
            ),
        )
    )
    factors = _factors(steps=steps, dk=2, dv=1)

    def unexpected_work(*args, **kwargs):
        raise AssertionError("recurrent allocation/matmul began before the budget check")

    monkeypatch.setattr(tiny_backend_module.torch, "zeros", unexpected_work)
    monkeypatch.setattr(tiny_backend_module.torch, "matmul", unexpected_work)
    with pytest.raises(tiny_backend_module.CacheDiagnosticBudgetError) as caught:
        cell(factors)
    assert caught.value.code == "diagnostic_budget_exceeded"
    assert "cache diagnostics require" in str(caught.value)
    assert "unbounded cache diagnostics" not in str(caught.value)
    assert caught.value.context == {
        "estimated_bytes": estimated_bytes,
        "budget_bytes": budget,
        "batch": 1,
        "steps": steps,
        "heads": 1,
        "query_slots": 1,
        "value_dim": 1,
        "cache_width": cache_width,
        "block_size": 2,
        "per_slot_read": False,
        "layers": 3,
        "unbounded_cache": unbounded_cache,
    }


def test_cache_diagnostic_estimate_includes_layers_without_blocking_normal_profile() -> None:
    estimator = tiny_backend_module._cache_diagnostic_allocation_bytes
    assert {"cache_width", "layers"} <= set(inspect.signature(estimator).parameters)
    common = {
        "batch": 8,
        "steps": 256,
        "heads": 4,
        "query_slots": 1,
        "value_dim": 64,
        "cache_width": 32,
        "block_size": 64,
        "per_slot_read": False,
    }
    one_layer = estimator(layers=1, **common)
    four_layers = estimator(layers=4, **common)
    assert four_layers == 4 * one_layer
    assert four_layers < tiny_backend_module._MAX_CACHE_DIAGNOSTIC_BYTES


def test_cache_diagnostic_estimator_preserves_unbounded_compatibility_defaults() -> None:
    estimator = tiny_backend_module._cache_diagnostic_allocation_bytes
    legacy = estimator(
        batch=1,
        steps=9,
        heads=1,
        query_slots=1,
        value_dim=1,
        block_size=2,
        per_slot_read=False,
    )
    explicit = estimator(
        batch=1,
        steps=9,
        heads=1,
        query_slots=1,
        value_dim=1,
        cache_width=9,
        block_size=2,
        per_slot_read=False,
        layers=1,
    )
    assert legacy == explicit


def test_cache_forward_return_annotation_matches_runtime_arity() -> None:
    annotation = get_type_hints(TinyKMD2Cell._cache_forward)["return"]
    fields = get_args(annotation)
    assert len(fields) == 20
    assert fields[:16] == (torch.Tensor,) * 16
    assert fields[16:] == (int,) * 4


def test_future_query_oracle_derives_relevance_from_episode_source_spans() -> None:
    episode = generate_task("mqar", 1, 4, 421, "train", {"width": 4})
    relevance = future_query_relevance(episode)
    assert relevance.shape == episode.valid.shape
    assert relevance.dtype == torch.float32
    expected = torch.zeros_like(relevance)
    for query in torch.nonzero(episode.query_mask[0], as_tuple=False).flatten():
        start, stop = episode.source_spans[0, query].tolist()
        expected[0, start:stop] += 1.0
    assert torch.equal(relevance, expected)

    vocab_size = max(
        int(episode.input_ids.max()), int(episode.targets[episode.loss_mask].max())
    ) + 1
    model = TinyKMD2Model(
        TinyKMD2Config(
            d_model=8,
            heads=1,
            dk=2,
            dv=2,
            layers=1,
            vocab_size=vocab_size,
            d_ff=16,
            rotation_mode="none",
            selector_seed=421,
            cache=CacheConfig(
                width=2,
                block_size=2,
                score="future_query_oracle",
                storage_dtype="fp32",
            ),
        ),
        init_seed=421,
    )
    output = model.forward_episode(episode)
    selected = set(output.cell_outputs[0].selected_positions.flatten().tolist())
    relevant_positions = set(torch.nonzero(relevance[0], as_tuple=False).flatten().tolist())
    assert selected & relevant_positions


def test_future_query_oracle_rejects_forward_without_annotations() -> None:
    factors = _factors(steps=4, dk=2, dv=1)
    cell = TinyKMD2Cell(
        _config(
            dv=1,
            selector_seed=9,
            cache=CacheConfig(
                width=2,
                block_size=2,
                score="future_query_oracle",
                storage_dtype="fp32",
            ),
        )
    )
    with pytest.raises(ValueError, match="future_relevance_required"):
        cell(factors)


def test_tiny_pre_rotation_cache_changes_only_cache_coordinates_and_read() -> None:
    rotated_config = _config(
        dk=4,
        dv=2,
        rotation_mode="current",
        rotation_gate_init=1.0,
        cache=CacheConfig(
            width=2,
            block_size=2,
            coordinate_frame="rotated_recurrence",
            storage_dtype="fp32",
        ),
    )
    pre_config = replace(
        rotated_config,
        cache=CacheConfig(
            width=2,
            block_size=2,
            coordinate_frame="pre_rotation",
            pre_rotation_diagnostic=True,
            storage_dtype="fp32",
        ),
    )
    rotated_projector = TinyFactorProjector(rotated_config)
    pre_projector = TinyFactorProjector(pre_config)
    pre_projector.load_state_dict(rotated_projector.state_dict(), strict=True)
    hidden = torch.randn(
        1,
        6,
        rotated_config.d_model,
        generator=torch.Generator().manual_seed(89),
        requires_grad=True,
    )
    valid = torch.ones(1, 6, dtype=torch.bool)
    positions = torch.arange(6).view(1, 6)
    rotated_factors = rotated_projector(hidden, valid, positions)
    pre_factors = pre_projector(hidden, valid, positions)
    assert pre_factors.cache_q is not None and pre_factors.cache_k is not None
    assert not torch.equal(pre_factors.cache_q, pre_factors.q)
    assert not torch.equal(pre_factors.cache_k, pre_factors.k)
    rotated_cell = TinyKMD2Cell(rotated_config)
    pre_cell = TinyKMD2Cell(pre_config)
    pre_cell.load_state_dict(rotated_cell.state_dict(), strict=True)
    with torch.no_grad():
        rotated_cell.cache_amplitude.fill_(1.0)
        pre_cell.cache_amplitude.fill_(1.0)
    rotated = rotated_cell(rotated_factors)
    pre = pre_cell(pre_factors)
    assert torch.equal(rotated.state_read, pre.state_read)
    assert torch.equal(rotated.final_state, pre.final_state)
    assert not torch.equal(rotated.cache_read, pre.cache_read)
    loss = pre.cache_read.square().sum()
    gradients = torch.autograd.grad(loss, (hidden, pre_projector.q_proj.weight), allow_unused=True)
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_tiny_per_slot_cache_read_matches_r_out_one_and_combines_real_slots() -> None:
    single = _factors(steps=5, dk=2, dv=2)
    cache = CacheConfig(width=2, block_size=2, storage_dtype="fp32")
    shared_one = TinyKMD2Cell(_config(dv=2, cache=cache))(single)
    slot_one = TinyKMD2Cell(
        _config(dv=2, cache=cache, per_slot_cache_read=True)
    )(single)
    assert torch.equal(slot_one.cache_read, shared_one.cache_read)
    assert slot_one.slot_cache_read.shape == (1, 5, 1, 1, 2)
    assert torch.equal(slot_one.slot_cache_read.squeeze(3), slot_one.cache_read)

    widened = TinyFactors(
        q=torch.cat(
            (
                single.q,
                torch.flip(single.q, dims=(-1,)),
                -single.q,
                torch.roll(single.q, 1, dims=-1),
            ),
            dim=3,
        ),
        k=single.k,
        v=single.v,
        decay=single.decay,
        beta_e=single.beta_e,
        beta_w=single.beta_w,
        out_mix=torch.tensor([0.1, 0.2, 0.3, 0.4]).view(1, 1, 1, 4).expand(1, 5, 1, 4),
        valid=single.valid,
        positions=single.positions,
    )
    slot_cell = TinyKMD2Cell(
        _config(dv=2, r_out=4, cache=cache, per_slot_cache_read=True)
    )
    with torch.no_grad():
        slot_cell.cache_amplitude.fill_(1.0)
    output = slot_cell(widened)
    assert output.slot_cache_read.shape == (1, 5, 1, 4, 2)
    expected = (
        output.slot_cache_read * widened.out_mix.unsqueeze(-1)
    ).sum(dim=3)
    assert torch.allclose(output.cache_read, expected, atol=0, rtol=0)
    assert output.slot_attention_weights.shape[:4] == (1, 5, 1, 4)
    assert torch.allclose(
        output.slot_attention_weights.sum(dim=-1),
        torch.ones(1, 5, 1, 4),
    )


def test_tiny_api_post_update_read_boundaries_valid_and_initial_state() -> None:
    config = _config(d_model=4, dk=2, dv=1, d_ff=8)
    factors = _factors(dv=1)
    boundaries = torch.tensor([[True, False, True]])
    segmented_factors = TinyFactors(
        q=factors.q,
        k=factors.k,
        v=factors.v,
        decay=factors.decay,
        beta_e=factors.beta_e,
        beta_w=factors.beta_w,
        out_mix=factors.out_mix,
        valid=factors.valid,
        positions=torch.tensor([[0, 1, 0]]),
    )
    output = TinyKMD2Cell(config)(segmented_factors, boundaries=boundaries)
    assert torch.equal(output.read[0, :, 0, 0], torch.tensor([1.0, 3.0, 3.0]))
    assert output.final_state[0, 0, 0, 0] == 3.0

    initial = torch.zeros(1, 1, 2, 1, dtype=torch.float32)
    initial[..., 0, 0] = 5.0
    no_reset = TinyKMD2Cell(config)(factors, state=initial)
    assert torch.equal(no_reset.read[0, :, 0, 0], torch.tensor([6.0, 8.0, 11.0]))

    valid = torch.tensor([[True, False, True]])
    masked = TinyFactors(
        q=factors.q,
        k=factors.k,
        v=factors.v,
        decay=factors.decay,
        beta_e=factors.beta_e,
        beta_w=factors.beta_w,
        out_mix=factors.out_mix,
        valid=valid,
        positions=torch.tensor([[0, -1, 1]], dtype=torch.int64),
    )
    masked_output = TinyKMD2Cell(config)(masked)
    assert torch.equal(masked_output.read[0, :, 0, 0], torch.tensor([1.0, 0.0, 4.0]))
    with pytest.raises(ValueError, match="boundaries.*valid"):
        TinyKMD2Cell(config)(masked, boundaries=torch.tensor([[False, True, False]]))
    with pytest.raises(TypeError, match="state.*float32"):
        TinyKMD2Cell(config)(factors, state=initial.double())
    with pytest.raises(ValueError, match="state.*shape"):
        TinyKMD2Cell(config)(factors, state=torch.zeros(1, 1, 3, 1))
    with pytest.raises(ValueError, match="boundaries.*shape"):
        TinyKMD2Cell(config)(factors, boundaries=torch.zeros(1, 2, dtype=torch.bool))

    with pytest.raises(TypeError, match="q must be floating"):
        TinyFactors(
            q=factors.q.to(torch.int64),
            k=factors.k,
            v=factors.v,
            decay=factors.decay,
            beta_e=factors.beta_e,
            beta_w=factors.beta_w,
            out_mix=factors.out_mix,
            valid=factors.valid,
            positions=factors.positions,
        )
    bad_q = factors.q.clone()
    bad_q[0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="q.*finite"):
        TinyFactors(
            q=bad_q,
            k=factors.k,
            v=factors.v,
            decay=factors.decay,
            beta_e=factors.beta_e,
            beta_w=factors.beta_w,
            out_mix=factors.out_mix,
            valid=factors.valid,
            positions=factors.positions,
        )


@pytest.mark.parametrize(
    "boundaries",
    [None, torch.tensor([[True, False, False]])],
)
def test_tiny_api_cell_rejects_undeclared_position_reset(
    boundaries: torch.Tensor | None,
) -> None:
    source = _factors(steps=3, dk=2, dv=1)
    reset_positions = TinyFactors(
        q=source.q,
        k=source.k,
        v=source.v,
        decay=source.decay,
        beta_e=source.beta_e,
        beta_w=source.beta_w,
        out_mix=source.out_mix,
        valid=source.valid,
        positions=torch.tensor([[0, 1, 0]]),
    )
    with pytest.raises(ValueError, match="positions|boundaries"):
        TinyKMD2Cell(_config(dv=1))(reset_positions, boundaries=boundaries)


def test_tiny_api_model_runs_task6_token_continuous_and_direct_factors() -> None:
    token_batch = generate_task("parity", 2, 4, 157, "train", {})
    token_config = _config(vocab_size=8, continuous_input_dim=None, output_dim=None)
    token_model = TinyKMD2Model(token_config, init_seed=11)
    token_output = token_model.forward_episode(token_batch)
    assert isinstance(token_output, TinyModelOutput)
    assert token_output.logits.shape == (*token_batch.targets.shape, 8)
    assert token_output.loss is not None and token_output.loss.ndim == 0
    assert len(token_output.final_states) == len(token_output.cell_outputs) == 1

    continuous_batch = generate_task(
        "irregular_integration", 2, 4, 163, "train", {"components": 1}
    )
    continuous_model = TinyKMD2Model(_config(), init_seed=13)
    continuous_output = continuous_model.forward_episode(continuous_batch)
    assert continuous_output.logits.shape == (2, 4, 1)
    assert continuous_output.loss is not None

    affine_batch = generate_task(
        "affine_associative_regression",
        2,
        3,
        167,
        "train",
        {"input_dim": 3, "output_dim": 2},
    )
    affine_factors = tiny_factors_from_episode(affine_batch)
    assert affine_factors.q.shape == (2, 7, 1, 1, 3)
    affine_config = _config(
        d_model=8,
        dk=3,
        dv=2,
        d_ff=16,
        vocab_size=8,
        continuous_input_dim=None,
        output_dim=2,
    )
    affine_model = TinyKMD2Model(affine_config, init_seed=17)
    affine_output = affine_model.forward_episode(affine_batch)
    assert affine_output.logits.shape == (2, 7, 2)
    assert affine_output.loss is not None
    direct_output = affine_model(
        factors=affine_factors,
        targets=affine_batch.targets,
        loss_mask=affine_batch.loss_mask,
        boundaries=affine_batch.boundaries,
    )
    assert torch.equal(affine_output.logits, direct_output.logits)

    explicit_continuous = continuous_model(
        continuous_inputs=continuous_batch.continuous_inputs,
        targets=continuous_batch.targets,
        loss_mask=continuous_batch.loss_mask,
        boundaries=continuous_batch.boundaries,
        valid=continuous_batch.valid,
        positions=continuous_batch.positions,
    )
    assert torch.equal(continuous_output.logits, explicit_continuous.logits)
    continuous_inference = continuous_model(
        continuous_inputs=continuous_batch.continuous_inputs,
        boundaries=continuous_batch.boundaries,
        valid=continuous_batch.valid,
        positions=continuous_batch.positions,
    )
    assert continuous_inference.logits.shape == (2, 4, 1)
    assert continuous_inference.loss is None

    no_targets = token_model(
        input_ids=token_batch.input_ids,
        boundaries=token_batch.boundaries,
        valid=token_batch.valid,
        positions=token_batch.positions,
    )
    assert no_targets.loss is None

    with pytest.raises(ValueError, match="exactly one input modality"):
        token_model()

    with pytest.raises(ValueError, match="exactly one input modality"):
        token_model(
            input_ids=token_batch.input_ids,
            continuous_inputs=torch.zeros(2, token_batch.input_ids.shape[1], 3),
        )


def test_tiny_api_exact_masked_losses_and_outside_mask_invariance() -> None:
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [8.0, -8.0]]])
    targets = torch.tensor([[0, 1, 1]])
    mask = torch.tensor([[True, True, False]])
    ce = TinyKMD2Model._compute_loss(logits, targets, mask)
    changed = targets.clone()
    changed[0, 2] = 0
    assert torch.equal(ce, TinyKMD2Model._compute_loss(logits, changed, mask))
    assert torch.allclose(
        ce,
        F.cross_entropy(logits[mask], targets[mask]),
        atol=0,
        rtol=0,
    )

    regression_logits = torch.tensor([[[1.0], [3.0], [99.0]]])
    regression_targets = torch.tensor([[[2.0], [1.0], [-99.0]]])
    mse = TinyKMD2Model._compute_loss(regression_logits, regression_targets, mask)
    altered = regression_targets.clone()
    altered[0, 2] = 0.0
    assert torch.equal(
        mse, TinyKMD2Model._compute_loss(regression_logits, altered, mask)
    )
    assert mse == pytest.approx(((1.0 - 2.0) ** 2 + (3.0 - 1.0) ** 2) / 2)


def test_tiny_api_rejects_loss_mask_outside_valid_for_every_modality() -> None:
    invalid = torch.tensor([[True, False, True]])
    positions = torch.tensor([[0, -1, 0]])
    boundaries = torch.tensor([[True, False, True]])
    bad_loss = torch.tensor([[False, True, False]])
    token_model = TinyKMD2Model(
        _config(vocab_size=8, continuous_input_dim=None, output_dim=None),
        init_seed=23,
    )
    with pytest.raises(ValueError, match="loss_mask.*valid"):
        token_model(
            input_ids=torch.tensor([[1, 0, 2]]),
            targets=torch.tensor([[-100, 1, -100]]),
            loss_mask=bad_loss,
            valid=invalid,
            positions=positions,
            boundaries=boundaries,
        )

    continuous_model = TinyKMD2Model(_config(), init_seed=29)
    with pytest.raises(ValueError, match="loss_mask.*valid"):
        continuous_model(
            continuous_inputs=torch.zeros(1, 3, 3),
            targets=torch.zeros(1, 3, 1),
            loss_mask=bad_loss,
            valid=invalid,
            positions=positions,
            boundaries=boundaries,
        )

    factors = _factors(steps=3, dk=2, dv=1)
    factors = TinyFactors(
        q=factors.q,
        k=factors.k,
        v=factors.v,
        decay=factors.decay,
        beta_e=factors.beta_e,
        beta_w=factors.beta_w,
        out_mix=factors.out_mix,
        valid=invalid,
        positions=positions,
    )
    direct_model = TinyKMD2Model(_config(dv=1), init_seed=31)
    with pytest.raises(ValueError, match="loss_mask.*valid"):
        direct_model(
            factors=factors,
            targets=torch.zeros(1, 3, 1),
            loss_mask=bad_loss,
            boundaries=boundaries,
        )


def test_tiny_api_multi_head_merge_and_multi_layer_residual_flow() -> None:
    config = _config(
        d_model=8,
        heads=2,
        dk=2,
        dv=2,
        layers=2,
        d_ff=16,
        vocab_size=8,
        continuous_input_dim=None,
        output_dim=None,
    )
    model = TinyKMD2Model(config, init_seed=19)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[-100, -100, -100, 1]])
    mask = torch.tensor([[False, False, False, True]])
    output = model(input_ids=input_ids, targets=targets, loss_mask=mask)
    assert output.logits.shape == (1, 4, 8)
    assert len(output.cell_outputs) == len(output.final_states) == 2
    assert all(cell.read.shape == (1, 4, 2, 2) for cell in output.cell_outputs)
    assert output.loss is not None
    output.loss.backward()
    for block in model.blocks:
        gradient = block.out_proj.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()


def test_tiny_api_invalid_and_boundary_inputs_cannot_leak_through_convolution() -> None:
    config = _config(
        d_model=8,
        dk=2,
        dv=2,
        d_ff=16,
        vocab_size=32,
        continuous_input_dim=None,
        output_dim=None,
        convolution_gate_init=1.0,
    )
    model = TinyKMD2Model(config, init_seed=37)
    valid = torch.tensor([[True, True, False, True, True]])
    positions = torch.tensor([[0, 1, -1, 0, 1]])
    boundaries = torch.tensor([[True, False, False, True, False]])
    left = model(
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        valid=valid,
        positions=positions,
        boundaries=boundaries,
    )
    changed_invalid = model(
        input_ids=torch.tensor([[1, 2, 31, 4, 5]]),
        valid=valid,
        positions=positions,
        boundaries=boundaries,
    )
    changed_prior_segment = model(
        input_ids=torch.tensor([[20, 21, 3, 4, 5]]),
        valid=valid,
        positions=positions,
        boundaries=boundaries,
    )
    assert torch.equal(left.logits[:, 3:], changed_invalid.logits[:, 3:])
    assert torch.equal(left.logits[:, 3:], changed_prior_segment.logits[:, 3:])


def test_tiny_api_rejects_non_segmentwise_positions_and_boundary_mismatch() -> None:
    model = TinyKMD2Model(
        _config(vocab_size=8, continuous_input_dim=None, output_dim=None),
        init_seed=39,
    )
    input_ids = torch.tensor([[1, 2, 3]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="positions.*increase"):
        model(
            input_ids=input_ids,
            valid=valid,
            positions=torch.tensor([[0, 2, 3]]),
            boundaries=torch.tensor([[True, False, False]]),
        )
    with pytest.raises(ValueError, match="boundaries.*position zero"):
        model(
            input_ids=input_ids,
            valid=valid,
            positions=torch.tensor([[0, 1, 0]]),
            boundaries=torch.tensor([[True, False, False]]),
        )
    with pytest.raises(ValueError, match="positions.*increase"):
        model(
            input_ids=input_ids,
            valid=valid,
            positions=torch.tensor([[0, 2, 0]]),
            boundaries=None,
        )


@pytest.mark.parametrize(
    ("valid", "positions", "boundaries", "message"),
    [
        (
            torch.tensor([[True, False, True]]),
            torch.tensor([[0, -1, 2]]),
            torch.tensor([[True, False, False]]),
            "positions.*increase",
        ),
        (
            torch.tensor([[True, True, True]]),
            torch.tensor([[0, 0, 1]]),
            torch.tensor([[True, False, False]]),
            "boundaries.*position zero",
        ),
        (
            torch.tensor([[True, True, True]]),
            torch.tensor([[0, 1, 2]]),
            torch.tensor([[True, True, False]]),
            "boundary tokens.*position zero",
        ),
        (
            torch.tensor([[True, True, True]]),
            torch.tensor([[0, 2, 0]]),
            None,
            "positions.*increase",
        ),
    ],
)
def test_tiny_api_direct_factors_require_declared_segment_layout(
    valid: torch.Tensor,
    positions: torch.Tensor,
    boundaries: torch.Tensor | None,
    message: str,
) -> None:
    source = _factors(steps=3, dk=2, dv=1)
    factors = TinyFactors(
        q=source.q,
        k=source.k,
        v=source.v,
        decay=source.decay,
        beta_e=source.beta_e,
        beta_w=source.beta_w,
        out_mix=source.out_mix,
        valid=valid,
        positions=positions,
    )
    model = TinyKMD2Model(_config(dv=1), init_seed=40)
    with pytest.raises(ValueError, match=message):
        model(factors=factors, boundaries=boundaries)


def test_tiny_api_model_init_preserves_global_rng_and_optional_import_isolation() -> None:
    torch.manual_seed(41)
    before = torch.random.get_rng_state().clone()
    TinyKMD2Model(_config(), init_seed=43)
    assert torch.equal(before, torch.random.get_rng_state())

    script = r"""
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'transformers' or fullname.startswith('transformers.'):
            raise ModuleNotFoundError(fullname)
        if fullname == 'triton' or fullname.startswith('triton.'):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, Blocker())
from research.kmd2_ablation.tiny_backend import TinyKMD2Config
print(TinyKMD2Config.__name__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.stdout.strip() == "TinyKMD2Config"


def test_tiny_api_rejects_nonfinite_selected_regression_values() -> None:
    mask = torch.tensor([[True, False]])
    logits = torch.tensor([[[float("nan")], [0.0]]])
    targets = torch.zeros(1, 2, 1)
    with pytest.raises(ValueError, match="regression logits.*finite"):
        TinyKMD2Model._compute_loss(logits, targets, mask)
    with pytest.raises(ValueError, match="regression targets.*finite"):
        TinyKMD2Model._compute_loss(
            torch.zeros_like(logits),
            torch.tensor([[[float("inf")], [0.0]]]),
            mask,
        )

    outside_only = torch.tensor([[[0.0], [float("nan")]]])
    assert torch.isfinite(TinyKMD2Model._compute_loss(outside_only, targets, mask))


def _independent_native_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta_e: torch.Tensor,
    beta_w: torch.Tensor,
    out_mix: torch.Tensor,
    valid: torch.Tensor,
    boundaries: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state
    outputs = []
    for token in range(q.shape[1]):
        state = torch.where(
            boundaries[:, token, None, None, None],
            torch.zeros((), dtype=state.dtype, device=state.device),
            state,
        )
        state_bar = decay[:, token].unsqueeze(-1) * state
        key = k[:, token, :, 0]
        value = v[:, token, :, 0]
        memory = torch.matmul(key.unsqueeze(-2), state_bar).squeeze(-2)
        update = (
            beta_w[:, token, :, 0].unsqueeze(-1) * value
            - beta_e[:, token, :, 0].unsqueeze(-1) * memory
        )
        candidate = state_bar + key.unsqueeze(-1) * update.unsqueeze(-2)
        state = torch.where(valid[:, token, None, None, None], candidate, state)
        slots = torch.matmul(q[:, token], state)
        read = (slots * out_mix[:, token].unsqueeze(-1)).sum(dim=-2)
        read = torch.where(
            valid[:, token, None, None], read, torch.zeros_like(read)
        )
        outputs.append(read)
    return torch.stack(outputs, dim=1), state


@pytest.mark.parametrize("r_out", [1, 4])
def test_tiny_native_parity_independent_forward_and_all_gradients(r_out: int) -> None:
    generator = torch.Generator().manual_seed(173 + r_out)
    batch, steps, heads, dk, dv = 2, 5, 2, 4, 3

    def leaf(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator).requires_grad_()

    q = leaf((batch, steps, heads, r_out, dk))
    k = leaf((batch, steps, heads, 1, dk))
    v = leaf((batch, steps, heads, 1, dv))
    decay_raw = leaf((batch, steps, heads, dk))
    beta_e_raw = leaf((batch, steps, heads, 1))
    beta_w_raw = leaf((batch, steps, heads, 1))
    mix_raw = leaf((batch, steps, heads, r_out))
    initial = leaf((batch, heads, dk, dv))
    decay = torch.sigmoid(decay_raw)
    beta_e = torch.sigmoid(beta_e_raw)
    beta_w = torch.sigmoid(beta_w_raw)
    out_mix = torch.softmax(mix_raw, dim=-1)
    valid = torch.tensor(
        [[True, True, True, True, True], [True, True, False, True, True]]
    )
    boundaries = torch.tensor(
        [[True, False, False, True, False], [True, False, False, False, False]]
    )
    positions = torch.tensor([[0, 1, 2, 0, 1], [0, 1, -1, 2, 3]])
    factors = TinyFactors(
        q=q,
        k=k,
        v=v,
        decay=decay,
        beta_e=beta_e,
        beta_w=beta_w,
        out_mix=out_mix,
        valid=valid,
        positions=positions,
    )
    cell = TinyKMD2Cell(
        _config(
            d_model=16,
            heads=heads,
            dk=dk,
            dv=dv,
            d_ff=32,
            r_out=r_out,
        )
    )
    actual = cell(factors, state=initial, boundaries=boundaries)
    expected_read, expected_state = _independent_native_oracle(
        q,
        k,
        v,
        decay,
        beta_e,
        beta_w,
        out_mix,
        valid,
        boundaries,
        initial,
    )
    assert torch.allclose(actual.read, expected_read, rtol=1e-5, atol=1e-6)
    assert torch.allclose(actual.final_state, expected_state, rtol=1e-5, atol=1e-6)

    leaves = (q, k, v, decay_raw, beta_e_raw, beta_w_raw, mix_raw, initial)
    actual_gradients = torch.autograd.grad(
        actual.read.square().sum() + actual.final_state.square().sum(),
        leaves,
        retain_graph=True,
    )
    expected_gradients = torch.autograd.grad(
        expected_read.square().sum() + expected_state.square().sum(), leaves
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients
    ):
        assert torch.allclose(
            actual_gradient, expected_gradient, rtol=2e-5, atol=2e-6
        )

    expected64, state64 = _independent_native_oracle(
        q.detach().double(),
        k.detach().double(),
        v.detach().double(),
        decay.detach().double(),
        beta_e.detach().double(),
        beta_w.detach().double(),
        out_mix.detach().double(),
        valid,
        boundaries,
        initial.detach().double(),
    )
    assert torch.allclose(actual.read.double(), expected64, rtol=2e-5, atol=2e-6)
    assert torch.allclose(actual.final_state.double(), state64, rtol=2e-5, atol=2e-6)


def test_tiny_native_parity_is_exact_under_cpu_autocast_with_cache() -> None:
    def clone_factors(source: TinyFactors) -> TinyFactors:
        return TinyFactors(
            q=source.q.detach().clone().requires_grad_(),
            k=source.k.detach().clone().requires_grad_(),
            v=source.v.detach().clone().requires_grad_(),
            decay=source.decay.detach().clone().requires_grad_(),
            beta_e=source.beta_e.detach().clone().requires_grad_(),
            beta_w=source.beta_w.detach().clone().requires_grad_(),
            out_mix=source.out_mix.detach().clone().requires_grad_(),
            valid=source.valid,
            positions=source.positions,
        )

    source = _factors(steps=4, dk=2, dv=2)
    ordinary_factors = clone_factors(source)
    autocast_factors = clone_factors(source)
    cell = TinyKMD2Cell(
        _config(
            dv=2,
            cache=CacheConfig(
                width=2,
                block_size=2,
                read="rmsnorm",
                storage_dtype="fp32",
            ),
        )
    )
    with torch.no_grad():
        cell.cache_amplitude.fill_(0.75)
    boundaries = torch.tensor([[True, False, False, False]])
    ordinary = cell(ordinary_factors, boundaries=boundaries)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast = cell(autocast_factors, boundaries=boundaries)

    for name in ("read", "state_read", "cache_read", "final_state", "sink_mass"):
        ordinary_tensor = getattr(ordinary, name)
        autocast_tensor = getattr(autocast, name)
        assert autocast_tensor.dtype == torch.float32
        assert torch.equal(autocast_tensor, ordinary_tensor), name
    assert torch.equal(autocast.selected_positions, ordinary.selected_positions)

    ordinary_variables = (
        ordinary_factors.q,
        ordinary_factors.k,
        ordinary_factors.v,
        ordinary_factors.decay,
        ordinary_factors.beta_e,
        ordinary_factors.beta_w,
        ordinary_factors.out_mix,
        *tuple(cell.parameters()),
    )
    autocast_variables = (
        autocast_factors.q,
        autocast_factors.k,
        autocast_factors.v,
        autocast_factors.decay,
        autocast_factors.beta_e,
        autocast_factors.beta_w,
        autocast_factors.out_mix,
        *tuple(cell.parameters()),
    )
    ordinary_gradients = torch.autograd.grad(
        ordinary.read.square().sum() + ordinary.sink_mass.sum(),
        ordinary_variables,
        retain_graph=True,
    )
    autocast_gradients = torch.autograd.grad(
        autocast.read.square().sum() + autocast.sink_mass.sum(),
        autocast_variables,
    )
    for ordinary_gradient, autocast_gradient in zip(
        ordinary_gradients, autocast_gradients, strict=True
    ):
        assert torch.equal(autocast_gradient, ordinary_gradient)


@pytest.mark.parametrize("r_out", [1, 4])
def test_tiny_native_parity_optional_production_scan(
    r_out: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        from types import SimpleNamespace

        from gdn3 import kmd2_native as native_module
        from gdn3.kmd2_native import KMD2NativeAttn
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("transformers"):
            pytest.skip("optional Transformers dependency is not installed")
        raise

    monkeypatch.setenv("GDN3_KMD2_ROUT", str(r_out))
    monkeypatch.setattr(native_module, "_FAST_SCAN", False)
    production = KMD2NativeAttn(
        SimpleNamespace(
            hidden_size=4,
            linear_num_value_heads=1,
            linear_num_key_heads=1,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_conv_kernel_dim=3,
            rms_norm_eps=1.0e-6,
        )
    )
    generator = torch.Generator().manual_seed(181 + r_out)
    q = torch.randn(1, 4, 1, r_out, 2, generator=generator, requires_grad=True)
    k = torch.randn(1, 4, 1, 2, generator=generator, requires_grad=True)
    v = torch.randn(1, 4, 1, 2, generator=generator, requires_grad=True)
    decay_raw = torch.randn(1, 4, 1, 2, generator=generator, requires_grad=True)
    beta_e_raw = torch.randn(1, 4, 1, generator=generator, requires_grad=True)
    beta_w_raw = torch.randn(1, 4, 1, generator=generator, requires_grad=True)
    decay = torch.sigmoid(decay_raw)
    beta_e = torch.sigmoid(beta_e_raw)
    beta_w = torch.sigmoid(beta_w_raw)
    if r_out > 1:
        with torch.no_grad():
            production.out_mix.copy_(torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
        mix = production.out_mix[None, None].expand(1, 4, 1, r_out)
    else:
        mix = torch.ones(1, 4, 1, 1)

    production_output = production._scan(q, k, v, decay, beta_e, beta_w)
    tiny = TinyKMD2Cell(
        _config(d_model=4, heads=1, dk=2, dv=2, d_ff=8, r_out=r_out)
    )
    tiny_output = tiny(
        TinyFactors(
            q=q,
            k=k.unsqueeze(3),
            v=v.unsqueeze(3),
            decay=decay,
            beta_e=beta_e.unsqueeze(-1),
            beta_w=beta_w.unsqueeze(-1),
            out_mix=mix,
            valid=torch.ones(1, 4, dtype=torch.bool),
            positions=torch.arange(4).view(1, 4),
        )
    ).read
    assert torch.allclose(tiny_output, production_output, rtol=1e-5, atol=1e-6)

    compared = [q, k, v, decay_raw, beta_e_raw, beta_w_raw]
    if r_out > 1:
        compared.append(production.out_mix)
    production_gradients = torch.autograd.grad(
        production_output.square().sum(), compared, retain_graph=True
    )
    tiny_gradients = torch.autograd.grad(tiny_output.square().sum(), compared)
    for tiny_gradient, production_gradient in zip(
        tiny_gradients, production_gradients
    ):
        assert torch.allclose(
            tiny_gradient, production_gradient, rtol=2e-5, atol=2e-6
        )


def _projector(config: TinyKMD2Config, seed: int = 191) -> TinyFactorProjector:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TinyFactorProjector(config)


def _projector_inputs(config: TinyKMD2Config) -> tuple[torch.Tensor, ...]:
    hidden = torch.randn(2, 5, config.d_model, generator=torch.Generator().manual_seed(193))
    valid = torch.ones(2, 5, dtype=torch.bool)
    positions = torch.arange(5).repeat(2, 1)
    return hidden, valid, positions


@pytest.mark.parametrize(
    "rotation_mode",
    ["current", "constant_rate", "non_cumulative", "fixed_rope", "moving_frame"],
)
def test_tiny_disabled_identity_rotation_controls(rotation_mode: str) -> None:
    config = _config(rotation_mode=rotation_mode, rotation_gate_init=0.0)
    projector = _projector(config)
    hidden, valid, positions = _projector_inputs(config)
    before = projector(hidden, valid, positions)
    with torch.no_grad():
        projector.rot_proj.weight.normal_()
        projector.rot_proj.bias.fill_(3.0)
        projector.rotation_rate.fill_(0.7)
    after = projector(hidden, valid, positions)
    assert torch.equal(before.q, after.q)
    assert torch.equal(before.k, after.k)


def test_tiny_disabled_identity_native_mechanism_gates_and_gradients() -> None:
    config = _config(rotation_mode="current")
    projector = _projector(config)
    hidden, valid, positions = _projector_inputs(config)
    hidden = hidden.requires_grad_()
    baseline = projector(hidden, valid, positions)
    baseline_loss = (
        baseline.q.sum()
        + baseline.k.sum()
        + baseline.v.sum()
        + baseline.decay.sum()
        + baseline.beta_w.sum()
    )
    baseline_gradient = torch.autograd.grad(
        baseline_loss, hidden, retain_graph=False
    )[0]
    with torch.no_grad():
        projector.conv.weight.normal_(mean=4.0, std=2.0)
        projector.decay_chan.fill_(1.5)
        projector.bw_off.fill_(2.0)
    hidden_again = hidden.detach().clone().requires_grad_()
    disabled = projector(hidden_again, valid, positions)
    disabled_loss = (
        disabled.q.sum()
        + disabled.k.sum()
        + disabled.v.sum()
        + disabled.decay.sum()
        + disabled.beta_w.sum()
    )
    disabled_gradient = torch.autograd.grad(disabled_loss, hidden_again)[0]
    for name in ("q", "k", "v", "decay", "beta_e", "beta_w", "out_mix"):
        assert torch.equal(getattr(baseline, name), getattr(disabled, name)), name
    assert torch.equal(baseline_gradient, disabled_gradient)


def test_tiny_disabled_identity_convolution_uses_logical_valid_positions() -> None:
    config = _config(
        rotation_mode="none", convolution_gate_init=1.0, conv_kernel=3
    )
    projector = _projector(config, seed=195)
    compact_hidden = torch.randn(
        1, 4, config.d_model, generator=torch.Generator().manual_seed(196)
    )
    compact_valid = torch.ones(1, 4, dtype=torch.bool)
    compact_positions = torch.arange(4).view(1, 4)
    compact = projector(compact_hidden, compact_valid, compact_positions)

    selected = torch.tensor([0, 2, 3, 4])
    hole_hidden = torch.zeros(1, 5, config.d_model)
    hole_hidden[:, selected] = compact_hidden
    hole_hidden[:, 1] = 999.0
    hole_valid = torch.tensor([[True, False, True, True, True]])
    hole_positions = torch.tensor([[0, -1, 1, 2, 3]])
    hole = projector(hole_hidden, hole_valid, hole_positions)
    for name in ("q", "k", "v", "decay", "beta_e", "beta_w", "out_mix", "read_gate"):
        torch.testing.assert_close(
            getattr(hole, name)[:, selected],
            getattr(compact, name),
            atol=1.0e-7,
            rtol=1.0e-7,
            msg=name,
        )


def test_tiny_disabled_identity_shared_query_slots_match_siso() -> None:
    single = _factors(q_slots=1, dv=2)
    widened = TinyFactors(
        q=single.q.expand(-1, -1, -1, 4, -1).clone(),
        k=single.k,
        v=single.v,
        decay=single.decay,
        beta_e=single.beta_e,
        beta_w=single.beta_w,
        out_mix=torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 1, 4).expand(1, 3, 1, 4),
        valid=single.valid,
        positions=single.positions,
    )
    siso = TinyKMD2Cell(_config(dv=2))(single)
    shared = TinyKMD2Cell(_config(dv=2, r_out=4))(widened)
    assert torch.equal(siso.read, shared.read)
    assert torch.equal(siso.final_state, shared.final_state)
    assert torch.equal(siso.scores, shared.scores)


def test_tiny_disabled_identity_exact_cache_is_branch_local() -> None:
    factors = _factors(steps=4, dk=2, dv=2, requires_grad=True)
    cache_config = CacheConfig(
        width=2,
        block_size=2,
        read="rmsnorm",
        storage_dtype="fp32",
    )
    native = TinyKMD2Cell(_config(dv=2))(factors)
    cached_cell = TinyKMD2Cell(_config(dv=2, cache=cache_config))
    cached = cached_cell(factors)
    assert torch.equal(native.read, cached.read)
    assert torch.equal(native.state_read, cached.state_read)
    assert torch.equal(native.final_state, cached.final_state)
    assert torch.equal(native.scores, cached.scores)
    assert torch.count_nonzero(cached.cache_read) > 0
    assert cached.selected_positions.shape == (1, 1, 2)
    assert cached.cache_persistent_bytes > 0
    assert cached.cache_block_bytes > 0
    assert torch.all((cached.sink_mass >= 0) & (cached.sink_mass <= 1))
    assert cached.hit_ready_positions.shape == (1, 4, 1, 4)
    assert cached.persistent_selected_positions.shape == (1, 4, 1, 2)
    assert torch.equal(
        cached.persistent_selected_positions[0, :2],
        torch.full((2, 1, 2), -1, dtype=torch.int64),
    )
    assert torch.all(cached.persistent_selected_positions[0, 2:] >= 0)
    assert cached.candidate_valid.shape == (1, 4, 1, 4)
    assert cached.attention_weights.shape == (1, 4, 1, 5)
    assert cached.top1_positions.shape == (1, 4, 1)
    assert cached.attention_entropy.shape == (1, 4, 1)
    assert cached.top1_mass.shape == (1, 4, 1)
    assert torch.equal(
        cached.hit_ready_positions >= 0, cached.candidate_valid
    )
    assert torch.allclose(
        cached.attention_weights.sum(dim=-1), torch.ones(1, 4, 1)
    )
    assert cached.retention_count == 4
    assert cached.eviction_count == 2
    assert torch.equal(
        cached.scores[0, :, 0],
        math.sqrt(2.0) * torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )

    native_gradient = torch.autograd.grad(
        native.read.square().sum(),
        (factors.q, factors.k, factors.v),
        retain_graph=True,
    )
    cached_gradient = torch.autograd.grad(
        cached.read.square().sum(),
        (factors.q, factors.k, factors.v),
    )
    for left, right in zip(native_gradient, cached_gradient):
        assert torch.equal(left, right)


def test_tiny_disabled_identity_cache_invalid_hole_preserves_persistent_state() -> None:
    source = _factors(steps=5, dk=2, dv=1)
    valid = torch.tensor([[True, False, True, True, True]])
    hole = TinyFactors(
        q=source.q,
        k=source.k,
        v=source.v,
        decay=source.decay,
        beta_e=source.beta_e,
        beta_w=source.beta_w,
        out_mix=source.out_mix,
        valid=valid,
        positions=torch.tensor([[0, -1, 1, 2, 3]]),
    )
    selected = torch.tensor([0, 2, 3, 4])
    compact = TinyFactors(
        q=source.q[:, selected],
        k=source.k[:, selected],
        v=source.v[:, selected],
        decay=source.decay[:, selected],
        beta_e=source.beta_e[:, selected],
        beta_w=source.beta_w[:, selected],
        out_mix=source.out_mix[:, selected],
        valid=torch.ones(1, 4, dtype=torch.bool),
        positions=torch.tensor([[0, 1, 2, 3]]),
    )
    config = _config(
        dv=1,
        cache=CacheConfig(
            width=2, block_size=2, read="unit_l2", storage_dtype="fp32"
        ),
    )
    hole_cell = TinyKMD2Cell(config)
    compact_cell = TinyKMD2Cell(config)
    with torch.no_grad():
        hole_cell.cache_amplitude.fill_(1.0)
        compact_cell.cache_amplitude.fill_(1.0)
    hole_output = hole_cell(
        hole, boundaries=torch.tensor([[True, False, False, False, False]])
    )
    compact_output = compact_cell(
        compact, boundaries=torch.tensor([[True, False, False, False]])
    )
    assert torch.allclose(
        hole_output.cache_read[:, selected], compact_output.cache_read, atol=0, rtol=0
    )
    assert torch.equal(
        hole_output.selected_positions, compact_output.selected_positions
    )
    assert torch.allclose(
        hole_output.final_state, compact_output.final_state, atol=0, rtol=0
    )


def test_tiny_active_effect_convolution_channel_decay_and_write_offset() -> None:
    config = _config(rotation_mode="none")
    projector = _projector(config, seed=197)
    hidden, valid, positions = _projector_inputs(config)
    baseline = projector(hidden, valid, positions)
    with torch.no_grad():
        projector.convolution_gate.fill_(1.0)
        projector.channel_decay_gate.fill_(1.0)
        projector.write_offset_gate.fill_(1.0)
        projector.decay_chan.fill_(0.25)
        projector.bw_off.fill_(0.5)
    active = projector(hidden, valid, positions)
    assert not torch.equal(baseline.v, active.v)
    assert not torch.equal(baseline.decay, active.decay)
    assert not torch.equal(baseline.beta_w, active.beta_w)
    loss = active.v.square().sum() + active.decay.sum() + active.beta_w.sum()
    gradients = torch.autograd.grad(
        loss,
        (
            projector.convolution_gate,
            projector.channel_decay_gate,
            projector.write_offset_gate,
            projector.conv.weight,
            projector.decay_chan,
            projector.bw_off,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize(
    "rotation_mode",
    ["current", "constant_rate", "non_cumulative", "fixed_rope"],
)
def test_tiny_active_effect_rotation_controls_have_finite_gradients(
    rotation_mode: str,
) -> None:
    config = _config(rotation_mode=rotation_mode, rotation_gate_init=0.0)
    projector = _projector(config, seed=199)
    hidden, valid, positions = _projector_inputs(config)
    baseline = projector(hidden, valid, positions)
    with torch.no_grad():
        projector.rotation_gate.fill_(1.0)
    active = projector(hidden, valid, positions)
    assert not torch.equal(baseline.q, active.q)
    assert not torch.equal(baseline.k, active.k)
    q_weights = torch.linspace(-1.0, 1.0, active.q.numel()).reshape_as(active.q)
    k_weights = torch.linspace(1.0, -0.5, active.k.numel()).reshape_as(active.k)
    gradient = torch.autograd.grad(
        (active.q * q_weights).sum() + (active.k * k_weights).sum(),
        projector.rotation_gate,
    )[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().item() > 1.0e-6


def test_tiny_active_moving_frame_emits_phase_with_finite_gate_gradient() -> None:
    config = _config(rotation_mode="moving_frame", rotation_gate_init=0.0)
    projector = _projector(config, seed=201)
    hidden, valid, positions = _projector_inputs(config)
    baseline = projector(hidden, valid, positions)
    assert baseline.moving_frame_phase is not None
    assert baseline.moving_frame_phase.count_nonzero() == 0
    with torch.no_grad():
        projector.rotation_gate.fill_(0.7)
    active = projector(hidden, valid, positions)
    assert active.moving_frame_phase is not None
    assert active.moving_frame_phase.count_nonzero() > 0
    assert torch.equal(active.q, baseline.q)
    assert torch.equal(active.k, baseline.k)
    gradient = torch.autograd.grad(
        active.moving_frame_phase.square().sum(), projector.rotation_gate
    )[0]
    assert torch.isfinite(gradient).all() and gradient.abs() > 0


def test_tiny_active_effect_shared_query_slots_change_read() -> None:
    factors = _factors(q_slots=4, dv=2)
    baseline = TinyKMD2Cell(_config(dv=2, r_out=4))(factors).read
    changed_q = factors.q.clone()
    changed_q[..., 1, 1] = 0.5
    changed_mix = factors.out_mix.clone()
    changed_mix[..., 0] = 0.25
    changed_mix[..., 1] = 0.75
    active_factors = TinyFactors(
        q=changed_q.requires_grad_(),
        k=factors.k,
        v=factors.v,
        decay=factors.decay,
        beta_e=factors.beta_e,
        beta_w=factors.beta_w,
        out_mix=changed_mix.requires_grad_(),
        valid=factors.valid,
        positions=factors.positions,
    )
    active = TinyKMD2Cell(_config(dv=2, r_out=4))(active_factors).read
    assert not torch.equal(baseline, active)
    gradients = torch.autograd.grad(
        active.square().sum(), (active_factors.q, active_factors.out_mix)
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_tiny_active_effect_cache_opening_and_read_parameter_gradients() -> None:
    factors = _factors(steps=4, dk=2, dv=2, requires_grad=True)
    config = _config(
        dv=2,
        cache=CacheConfig(
            width=2,
            block_size=2,
            read="rmsnorm",
            storage_dtype="fp32",
        ),
    )
    cell = TinyKMD2Cell(config)
    closed = cell(factors)
    opening_gradient = torch.autograd.grad(
        closed.read.square().sum(), cell.cache_amplitude
    )[0]
    assert torch.isfinite(opening_gradient).all()
    assert torch.count_nonzero(opening_gradient) > 0

    with torch.no_grad():
        cell.cache_amplitude.fill_(0.5)
    opened = cell(factors)
    assert not torch.equal(opened.read, opened.state_read)
    gradients = torch.autograd.grad(
        opened.read.square().sum(),
        (
            cell.cache_gamma_q,
            cell.cache_gamma_k,
            cell.cache_sink_logit,
            cell.cache_amplitude,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
