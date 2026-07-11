from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from types import SimpleNamespace

import pytest
import torch


DECLARED_ARM_IDS = {
    "native",
    "rotation.current",
    "rotation.off",
    "rotation.constant_rate",
    "rotation.non_cumulative",
    "rotation.fixed_rope",
    "rotation.moving_frame_oracle",
    "convolution.on",
    "convolution.off",
    "trapezoid",
    "bc_bias",
    "bc_bias.diagonal_rescale",
    "bc_bias.constant_coordinate_oracle",
    "corrected_momentum",
    "causal_lookahead",
    "state_size.sweep",
    "true_mimo.sweep",
    "gdn2_decoupled.channelwise",
    "exact_cache.off",
    "exact_cache.current_block_only",
    "exact_cache.selector.exact_outer",
    "exact_cache.selector.coupled_paper",
    "exact_cache.selector.residual_only",
    "exact_cache.selector.write_value",
    "exact_cache.selector.recency",
    "exact_cache.selector.reservoir",
    "exact_cache.selector.future_query_oracle",
    "exact_cache.read.unit_l2",
    "exact_cache.read.fixed_temperature",
    "exact_cache.read.rmsnorm",
    "exact_cache.storage.bf16",
    "exact_cache.storage.fp32",
    "exact_cache.pre_rotation_diagnostic",
    "exact_cache.per_slot_read",
    "exact_cache.unbounded_oracle",
    "exact_cache.width.0",
    "exact_cache.width.8",
    "exact_cache.width.16",
    "exact_cache.width.32",
    "exact_cache.width.64",
    "exact_cache.width.128",
    "exact_cache.block.64",
    "exact_cache.block.128",
    "exact_cache.block.256",
    "exact_cache.rotation_factorial",
    "exact_cache.r_out_factorial",
    "exact_cache.rotation_factorial.M00",
    "exact_cache.rotation_factorial.M10",
    "exact_cache.rotation_factorial.M01",
    "exact_cache.rotation_factorial.M11",
    "exact_cache.r_out_factorial.M00",
    "exact_cache.r_out_factorial.M10",
    "exact_cache.r_out_factorial.M01",
    "exact_cache.r_out_factorial.M11",
}


def test_registry_has_every_declared_arm() -> None:
    from research.kmd2_ablation.variants import VARIANT_REGISTRY, all_variants

    records = all_variants()
    assert isinstance(records, tuple)
    assert {record.arm_id for record in records} == DECLARED_ARM_IDS
    assert tuple(VARIANT_REGISTRY) == tuple(sorted(DECLARED_ARM_IDS))
    assert len(records) == len(VARIANT_REGISTRY)

    valid_evidence = {"baseline", "addition", "reliance", "diagnostic"}
    valid_comparisons = {
        "baseline",
        "incremental",
        "replacement",
        "reliance",
        "diagnostic",
        "factorial",
    }
    valid_stages = {
        "local_correctness",
        "mechanism_screen",
        "tiny_promotion",
        "qwen_reliance",
        "qwen_heal",
        "selector_replay",
        "read_screen",
        "capacity_screen",
        "native_interaction",
    }
    for record in records:
        assert record.evidence_kind in valid_evidence
        assert record.comparison in valid_comparisons
        assert record.compatible_backends
        assert record.compatible_backends <= frozenset({"tiny", "qwen"})
        assert record.compatible_tasks
        assert record.changed_parameters or record.changed_state or record.arm_id == "native"
        assert record.required_stage in valid_stages


def test_registry_lookup_is_strict_and_records_are_deeply_immutable() -> None:
    from research.kmd2_ablation.variants import VARIANT_REGISTRY, get_variant

    trapezoid = get_variant("trapezoid")
    assert trapezoid is VARIANT_REGISTRY["trapezoid"]
    assert trapezoid.evidence_kind == "addition"
    assert trapezoid.comparison == "incremental"
    assert trapezoid.compatible_backends == frozenset({"tiny", "qwen"})
    assert trapezoid.compatible_tasks == frozenset({"irregular_integration"})
    assert trapezoid.changed_parameters == ("rho_head", "rho_proj.weight")
    assert trapezoid.changed_state == ("k_prev", "u_prev")
    assert trapezoid.required_stage == "mechanism_screen"

    with pytest.raises(KeyError, match="unknown variant arm"):
        get_variant("Trapezoid")
    with pytest.raises(TypeError, match="arm_id"):
        get_variant(1)  # type: ignore[arg-type]
    with pytest.raises(FrozenInstanceError):
        trapezoid.arm_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        VARIANT_REGISTRY["changed"] = trapezoid  # type: ignore[index]


@pytest.mark.parametrize(
    ("arm_id", "evidence_kind", "comparison", "stage"),
    [
        ("rotation.off", "reliance", "reliance", "qwen_reliance"),
        ("convolution.off", "reliance", "reliance", "qwen_reliance"),
        ("bc_bias.diagonal_rescale", "diagnostic", "diagnostic", "mechanism_screen"),
        (
            "bc_bias.constant_coordinate_oracle",
            "diagnostic",
            "diagnostic",
            "mechanism_screen",
        ),
        (
            "exact_cache.selector.future_query_oracle",
            "diagnostic",
            "diagnostic",
            "selector_replay",
        ),
        (
            "exact_cache.pre_rotation_diagnostic",
            "diagnostic",
            "diagnostic",
            "tiny_promotion",
        ),
        ("exact_cache.unbounded_oracle", "diagnostic", "diagnostic", "tiny_promotion"),
        (
            "exact_cache.rotation_factorial",
            "addition",
            "factorial",
            "native_interaction",
        ),
        (
            "exact_cache.r_out_factorial",
            "addition",
            "factorial",
            "native_interaction",
        ),
    ],
)
def test_registry_preserves_scientific_role(
    arm_id: str, evidence_kind: str, comparison: str, stage: str
) -> None:
    from research.kmd2_ablation.variants import get_variant

    record = get_variant(arm_id)
    assert (record.evidence_kind, record.comparison, record.required_stage) == (
        evidence_kind,
        comparison,
        stage,
    )


def test_registry_keeps_qwen_incompatible_redesigns_tiny_only() -> None:
    from research.kmd2_ablation.variants import get_variant

    state_size = get_variant("state_size.sweep")
    true_mimo = get_variant("true_mimo.sweep")
    gdn2_decoupled = get_variant("gdn2_decoupled.channelwise")
    assert state_size.compatible_backends == frozenset({"tiny"})
    assert true_mimo.compatible_backends == frozenset({"tiny"})
    assert gdn2_decoupled.compatible_backends == frozenset({"tiny"})
    assert state_size.experiment_kind == true_mimo.experiment_kind == "cold_redesign"
    assert state_size.native_warm_start is true_mimo.native_warm_start is False
    assert gdn2_decoupled.experiment_kind == "cold_redesign"
    assert gdn2_decoupled.native_warm_start is False
    assert gdn2_decoupled.changed_parameters == (
        "erase_proj.weight",
        "write_proj.weight",
    )
    assert get_variant("rotation.moving_frame_oracle").compatible_backends == frozenset(
        {"tiny"}
    )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("backend", "qwen"),
        ("task", "trajectory"),
        ("stage", "qwen_heal"),
        ("experiment_kind", "native_warm_start"),
    ],
)
def test_true_mimo_compatibility_request_rejects_each_contract_violation(
    field: str, wrong_value: str
) -> None:
    from research.kmd2_ablation.variants import (
        VariantCompatibilityError,
        validate_variant_compatibility,
    )

    request = {
        "backend": "tiny",
        "task": "mqar",
        "stage": "mechanism_screen",
        "experiment_kind": "cold_redesign",
    }
    accepted = validate_variant_compatibility("true_mimo.sweep", **request)
    assert accepted.arm_id == "true_mimo.sweep"

    request[field] = wrong_value
    with pytest.raises(VariantCompatibilityError) as caught:
        validate_variant_compatibility("true_mimo.sweep", **request)
    assert caught.value.arm_id == "true_mimo.sweep"
    assert caught.value.violations == (field,)


def test_true_mimo_qwen_heal_request_reports_all_compatibility_violations() -> None:
    from research.kmd2_ablation.variants import (
        VariantCompatibilityError,
        validate_variant_compatibility,
    )

    with pytest.raises(VariantCompatibilityError) as caught:
        validate_variant_compatibility(
            "true_mimo.sweep",
            backend="qwen",
            task="mqar",
            stage="qwen_heal",
            experiment_kind="native_warm_start",
        )
    assert caught.value.violations == ("backend", "stage", "experiment_kind")
    assert "true_mimo.sweep" in str(caught.value)


def test_registry_experiment_kind_and_warm_start_metadata_are_consistent() -> None:
    from research.kmd2_ablation.variants import all_variants, get_variant

    assert get_variant("trapezoid").experiment_kind == "native_warm_start"
    assert get_variant("trapezoid").native_warm_start is True
    assert get_variant("rotation.off").experiment_kind == "reliance"
    assert get_variant("rotation.off").native_warm_start is False
    for record in all_variants():
        assert record.experiment_kind in {
            "baseline",
            "native_warm_start",
            "cold_redesign",
            "reliance",
            "diagnostic",
        }
        assert record.native_warm_start is (
            record.experiment_kind == "native_warm_start"
        )


def _tiny_config(**overrides: object):
    from research.kmd2_ablation.tiny_backend import TinyKMD2Config

    values: dict[str, object] = {
        "d_model": 8,
        "heads": 1,
        "dk": 2,
        "dv": 2,
        "layers": 1,
        "vocab_size": 11,
        "d_ff": 16,
        "rotation_mode": "none",
        "trapezoid": False,
        "trapezoid_gate_init": 0.0,
        "corrected_momentum": False,
        "momentum_gamma_init": 0.0,
        "causal_lookahead": False,
        "lookahead_rho_init": 0.0,
    }
    values.update(overrides)
    return TinyKMD2Config(**values)


def _tiny_factors(
    *,
    steps: int = 4,
    positions: torch.Tensor | None = None,
    trapezoid_rho: torch.Tensor | None = None,
    momentum_gamma: torch.Tensor | None = None,
    lookahead_rho: torch.Tensor | None = None,
    requires_grad: bool = False,
):
    from research.kmd2_ablation.tiny_backend import TinyFactors

    generator = torch.Generator().manual_seed(9107)
    q = torch.randn(1, steps, 1, 1, 2, generator=generator)
    k = torch.randn(1, steps, 1, 1, 2, generator=generator)
    v = torch.randn(1, steps, 1, 1, 2, generator=generator)
    decay = torch.sigmoid(torch.randn(1, steps, 1, 2, generator=generator))
    beta_e = torch.sigmoid(torch.randn(1, steps, 1, 1, generator=generator))
    beta_w = torch.sigmoid(torch.randn(1, steps, 1, 1, generator=generator))
    out_mix = torch.ones(1, steps, 1, 1)
    tensors = [q, k, v, decay, beta_e, beta_w, out_mix]
    if requires_grad:
        for tensor in tensors:
            tensor.requires_grad_()
        if trapezoid_rho is not None:
            trapezoid_rho.requires_grad_()
        if momentum_gamma is not None:
            momentum_gamma.requires_grad_()
        if lookahead_rho is not None:
            lookahead_rho.requires_grad_()
    if positions is None:
        positions = torch.arange(steps, dtype=torch.int64).view(1, steps)
    optional: dict[str, torch.Tensor] = {}
    if momentum_gamma is not None:
        optional["momentum_gamma"] = momentum_gamma
    if lookahead_rho is not None:
        optional["lookahead_rho"] = lookahead_rho
    return TinyFactors(
        q=q,
        k=k,
        v=v,
        decay=decay,
        beta_e=beta_e,
        beta_w=beta_w,
        out_mix=out_mix,
        valid=torch.ones(1, steps, dtype=torch.bool),
        positions=positions,
        trapezoid_rho=trapezoid_rho,
        **optional,
    )


def _factor_grads(factors) -> tuple[torch.Tensor, ...]:
    return tuple(
        getattr(factors, name).grad.detach().clone()
        for name in ("q", "k", "v", "decay", "beta_e", "beta_w", "out_mix")
    )


def _pre_task9_native_oracle(factors, boundaries, initial_state):
    q = factors.q.float()
    k = factors.k.float()
    v = factors.v.float()
    decay = factors.decay.float()
    beta_e = factors.beta_e.float()
    beta_w = factors.beta_w.float()
    out_mix = factors.out_mix.float()
    state = initial_state
    reads = []
    scores = []
    for token in range(q.shape[1]):
        state = torch.where(
            boundaries[:, token, None, None, None],
            torch.zeros((), dtype=torch.float32),
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
        state = torch.where(
            factors.valid[:, token, None, None, None], candidate, state
        )
        slots = torch.matmul(q[:, token], state)
        read = (slots * out_mix[:, token].unsqueeze(-1)).sum(dim=-2)
        reads.append(
            torch.where(
                factors.valid[:, token, None, None], read, torch.zeros_like(read)
            )
        )
        score = torch.linalg.vector_norm(key, dim=-1) * torch.linalg.vector_norm(
            update, dim=-1
        )
        scores.append(
            torch.where(
                factors.valid[:, token, None], score, torch.zeros_like(score)
            )
        )
    return torch.stack(reads, dim=1), state, torch.stack(scores, dim=1)


def test_trapezoid_zero_gate_is_bit_exact_pre_task9_arithmetic_oracle() -> None:
    from research.kmd2_ablation.tiny_backend import TinyFactors, TinyKMD2Cell

    generator = torch.Generator().manual_seed(19021)

    def leaf(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator).requires_grad_()

    batch, steps, heads, dk, dv = 2, 5, 2, 4, 3
    q = leaf((batch, steps, heads, 1, dk))
    k = leaf((batch, steps, heads, 1, dk))
    v = leaf((batch, steps, heads, 1, dv))
    decay_raw = leaf((batch, steps, heads, dk))
    beta_e_raw = leaf((batch, steps, heads, 1))
    beta_w_raw = leaf((batch, steps, heads, 1))
    out_mix = leaf((batch, steps, heads, 1))
    initial = leaf((batch, heads, dk, dv))
    rho = torch.zeros(batch, steps, heads, requires_grad=True)
    valid = torch.tensor(
        [[True, True, True, True, True], [True, True, False, True, True]]
    )
    positions = torch.tensor([[0, 1, 2, 0, 1], [0, 1, -1, 2, 3]])
    boundaries = torch.tensor(
        [[True, False, False, True, False], [True, False, False, False, False]]
    )
    factors = TinyFactors(
        q=q,
        k=k,
        v=v,
        decay=torch.sigmoid(decay_raw),
        beta_e=torch.sigmoid(beta_e_raw),
        beta_w=torch.sigmoid(beta_w_raw),
        out_mix=out_mix,
        valid=valid,
        positions=positions,
        trapezoid_rho=rho,
    )
    actual = TinyKMD2Cell(
        _tiny_config(heads=heads, dk=dk, dv=dv, trapezoid=True)
    )(factors, state=initial, boundaries=boundaries)
    expected_read, expected_state, expected_scores = _pre_task9_native_oracle(
        factors, boundaries, initial
    )
    assert torch.equal(actual.read, expected_read)
    assert torch.equal(actual.final_state, expected_state)
    assert torch.equal(actual.scores, expected_scores.detach())

    leaves = (q, k, v, decay_raw, beta_e_raw, beta_w_raw, out_mix, initial)
    actual_gradients = torch.autograd.grad(
        actual.read.square().sum() + actual.final_state.square().sum(),
        leaves,
        retain_graph=True,
    )
    expected_gradients = torch.autograd.grad(
        expected_read.square().sum() + expected_state.square().sum(), leaves
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.equal(actual_gradient, expected_gradient)


def test_trapezoid_zero_gate_preserves_exact_cache_scores_and_diagnostics() -> None:
    from research.kmd2_ablation.config import CacheConfig
    from research.kmd2_ablation.tiny_backend import TinyKMD2Cell

    cache = CacheConfig(
        width=2,
        block_size=2,
        read="rmsnorm",
        storage_dtype="fp32",
    )
    source = _tiny_factors(steps=5)
    native_factors = _tiny_factors_from(source, trapezoid_rho=None)
    trapezoid_factors = _tiny_factors_from(
        source, trapezoid_rho=torch.zeros(1, 5, 1)
    )
    factor_names = ("q", "k", "v", "decay", "beta_e", "beta_w", "out_mix")
    for factors in (native_factors, trapezoid_factors):
        for name in factor_names:
            getattr(factors, name).requires_grad_()
    native_cell = TinyKMD2Cell(_tiny_config(cache=cache))
    trapezoid_cell = TinyKMD2Cell(_tiny_config(cache=cache, trapezoid=True))
    trapezoid_cell.load_state_dict(native_cell.state_dict(), strict=False)
    with torch.no_grad():
        native_cell.cache_amplitude.fill_(0.4)
        trapezoid_cell.cache_amplitude.fill_(0.4)
    native = native_cell(native_factors)
    trapezoid = trapezoid_cell(trapezoid_factors)
    expected_read, expected_state, expected_scores = _pre_task9_native_oracle(
        trapezoid_factors,
        torch.zeros(1, 5, dtype=torch.bool),
        torch.zeros(1, 1, 2, 2),
    )
    assert torch.equal(trapezoid.state_read, expected_read)
    assert torch.equal(trapezoid.final_state, expected_state)
    assert torch.equal(trapezoid.scores, expected_scores.detach())
    for field in (
        "read",
        "state_read",
        "cache_read",
        "final_state",
        "scores",
        "selected_positions",
        "sink_mass",
    ):
        assert torch.equal(getattr(trapezoid, field), getattr(native, field)), field
    assert trapezoid.cache_persistent_bytes == native.cache_persistent_bytes
    assert trapezoid.cache_block_bytes == native.cache_block_bytes
    native_leaves = tuple(getattr(native_factors, name) for name in factor_names) + tuple(
        native_cell.parameters()
    )
    trapezoid_leaves = tuple(
        getattr(trapezoid_factors, name) for name in factor_names
    ) + tuple(trapezoid_cell.parameters())
    native_gradients = torch.autograd.grad(
        native.read.square().sum() + native.final_state.square().sum(), native_leaves
    )
    trapezoid_gradients = torch.autograd.grad(
        trapezoid.read.square().sum() + trapezoid.final_state.square().sum(),
        trapezoid_leaves,
    )
    for actual, expected in zip(
        trapezoid_gradients, native_gradients, strict=True
    ):
        assert torch.equal(actual, expected)


def test_trapezoid_tiny_zero_gate_is_exact_native_forward_and_backward() -> None:
    from research.kmd2_ablation.tiny_backend import TinyKMD2Cell

    native_factors = _tiny_factors(requires_grad=True)
    trap_factors = _tiny_factors(
        trapezoid_rho=torch.zeros(1, 4, 1), requires_grad=True
    )
    native = TinyKMD2Cell(_tiny_config())(native_factors)
    trapezoid = TinyKMD2Cell(_tiny_config(trapezoid=True))(trap_factors)

    assert torch.equal(trapezoid.read, native.read)
    assert torch.equal(trapezoid.final_state, native.final_state)
    native_loss = native.read.square().sum() + native.final_state.square().sum()
    trap_loss = trapezoid.read.square().sum() + trapezoid.final_state.square().sum()
    native_loss.backward()
    trap_loss.backward()
    for actual, expected in zip(_factor_grads(trap_factors), _factor_grads(native_factors)):
        assert torch.equal(actual, expected)
    assert trap_factors.trapezoid_rho.grad is not None
    assert torch.isfinite(trap_factors.trapezoid_rho.grad).all()
    assert trap_factors.trapezoid_rho.grad[:, 0].count_nonzero() == 0


def test_trapezoid_tiny_equation_active_effect_boundary_and_gate_gradient() -> None:
    from research.kmd2_ablation.tiny_backend import TinyFactors, TinyKMD2Cell

    def factors(*, boundary: bool, gate: float) -> TinyFactors:
        rho = torch.tensor([[[gate], [gate]]], requires_grad=True)
        return TinyFactors(
            q=torch.tensor([[[[[1.0, 0.0]]], [[[1.0, 0.0]]]]]),
            k=torch.tensor([[[[[1.0, 0.0]]], [[[1.0, 0.0]]]]]),
            v=torch.tensor([[[[[1.0, 0.0]]], [[[3.0, 0.0]]]]]),
            decay=torch.full((1, 2, 1, 2), 0.5),
            beta_e=torch.zeros(1, 2, 1, 1),
            beta_w=torch.ones(1, 2, 1, 1),
            out_mix=torch.ones(1, 2, 1, 1),
            valid=torch.ones(1, 2, dtype=torch.bool),
            positions=torch.tensor([[0, 0 if boundary else 1]], dtype=torch.int64),
            trapezoid_rho=rho,
        )

    active_factors = factors(boundary=False, gate=1.0)
    active = TinyKMD2Cell(_tiny_config(trapezoid=True))(active_factors)
    native = TinyKMD2Cell(_tiny_config())(
        _tiny_factors_from(active_factors, trapezoid_rho=None)
    )
    # At t=1: S_bar=.5, current write is suppressed, and D_t U_prev=.5.
    assert active.read[0, 1, 0, 0].item() == pytest.approx(1.0)
    assert active.read[0, 1, 0, 0] != native.read[0, 1, 0, 0]

    mixed_factors = factors(boundary=False, gate=0.4)
    mixed = TinyKMD2Cell(_tiny_config(trapezoid=True))(mixed_factors)
    mixed.read.sum().backward()
    assert mixed_factors.trapezoid_rho.grad is not None
    assert mixed_factors.trapezoid_rho.grad[0, 0, 0] == 0
    assert mixed_factors.trapezoid_rho.grad[0, 1, 0].abs() > 0

    boundary_factors = factors(boundary=True, gate=1.0)
    boundary = TinyKMD2Cell(_tiny_config(trapezoid=True))(
        boundary_factors, boundaries=torch.tensor([[True, True]])
    )
    assert boundary.read[0, :, 0, 0].tolist() == pytest.approx([1.0, 3.0])


def _tiny_factors_from(factors, *, trapezoid_rho):
    from research.kmd2_ablation.tiny_backend import TinyFactors

    return TinyFactors(
        q=factors.q.detach().clone(),
        k=factors.k.detach().clone(),
        v=factors.v.detach().clone(),
        decay=factors.decay.detach().clone(),
        beta_e=factors.beta_e.detach().clone(),
        beta_w=factors.beta_w.detach().clone(),
        out_mix=factors.out_mix.detach().clone(),
        valid=factors.valid.detach().clone(),
        positions=factors.positions.detach().clone(),
        trapezoid_rho=trapezoid_rho,
    )


def test_trapezoid_tiny_projector_parameters_are_active_and_projectable() -> None:
    from research.kmd2_ablation.tiny_backend import (
        TinyFactorProjector,
        TinyKMD2Cell,
        project_trapezoid_gates_,
    )

    config = _tiny_config(trapezoid=True, trapezoid_gate_init=0.6)
    projector = TinyFactorProjector(config)
    hidden = torch.randn(1, 4, config.d_model, generator=torch.Generator().manual_seed(14))
    valid = torch.ones(1, 4, dtype=torch.bool)
    positions = torch.arange(4, dtype=torch.int64).view(1, 4)
    factors = projector(hidden, valid, positions)
    assert factors.trapezoid_rho is not None
    assert torch.all((factors.trapezoid_rho >= 0) & (factors.trapezoid_rho <= 1))
    output = TinyKMD2Cell(config)(factors)
    output.read.square().sum().backward()
    assert projector.rho_head.grad is not None
    assert projector.rho_head.grad.abs().sum() > 0
    assert projector.rho_proj.weight.grad is not None
    assert projector.rho_proj.weight.grad.abs().sum() > 0

    with torch.no_grad():
        projector.rho_head.copy_(torch.tensor([-0.5]))
    project_trapezoid_gates_(projector)
    assert projector.rho_head.item() == 0.0
    with torch.no_grad():
        projector.rho_head.copy_(torch.tensor([1.5]))
    project_trapezoid_gates_(projector)
    assert projector.rho_head.item() == 1.0


def _tiny_training_config(job_id: str):
    from research.kmd2_ablation.tiny_training import TinyTrainingConfig

    return TinyTrainingConfig(
        job_id=job_id,
        seed=71,
        updates=2,
        max_tokens=128,
        learning_rate=1.0e-3,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        weight_decay=0.0,
        warmup_updates=0,
        max_grad_norm=1.0,
    )


def test_trapezoid_rho_head_is_strict_in_post_step_and_checkpoint_resume(
    tmp_path,
) -> None:
    from research.kmd2_ablation.tiny_backend import TinyKMD2Model
    from research.kmd2_ablation.tiny_training import TinyTrainer

    config = _tiny_config(trapezoid=True)
    source = TinyTrainer(
        TinyKMD2Model(config, init_seed=401), _tiny_training_config("rho-checkpoint")
    )
    rho_name = "blocks.0.projector.rho_head"
    rho = dict(source.model.named_parameters())[rho_name]
    with torch.no_grad():
        rho.fill_(0.375)
    checkpoint = tmp_path / "rho.pt"
    source.save_checkpoint(checkpoint)

    resumed = TinyTrainer(
        TinyKMD2Model(config, init_seed=999), _tiny_training_config("rho-checkpoint")
    )
    resumed.load_checkpoint(checkpoint)
    assert torch.equal(
        resumed.model.state_dict()[rho_name], source.model.state_dict()[rho_name]
    )

    with torch.no_grad():
        dict(resumed.model.named_parameters())[rho_name].fill_(1.01)
    with pytest.raises(FloatingPointError, match=r"rho_head.*\[0,1\]"):
        resumed._validate_post_step_state()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model_state"][rho_name].fill_(-0.01)
    corrupt = tmp_path / "rho-corrupt.pt"
    torch.save(payload, corrupt)
    before = resumed.model.state_dict()[rho_name].clone()
    with pytest.raises(ValueError, match=r"rho_head.*\[0,1\]"):
        resumed.load_checkpoint(corrupt)
    assert torch.equal(resumed.model.state_dict()[rho_name], before)


def test_tiny_checkpoint_accepts_pre_variant_default_config_signature(tmp_path) -> None:
    import hashlib
    import json
    from dataclasses import fields, is_dataclass

    from research.kmd2_ablation.tiny_backend import TinyKMD2Model
    from research.kmd2_ablation.tiny_training import TinyTrainer

    def canonical(value):
        if is_dataclass(value) and not isinstance(value, type):
            return {field.name: canonical(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, dict):
            return {str(key): canonical(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        return value

    config = _tiny_config()
    source = TinyTrainer(
        TinyKMD2Model(config, init_seed=700),
        _tiny_training_config("pre-variant-defaults"),
    )
    current_path = tmp_path / "current.pt"
    source.save_checkpoint(current_path)
    payload = torch.load(current_path, map_location="cpu", weights_only=False)
    current_signature = payload["model_config_signature"]
    historical_config = canonical(config)
    for name in (
        "corrected_momentum",
        "momentum_gamma_init",
        "causal_lookahead",
        "lookahead_rho_init",
        "bc_bias_mode",
    ):
        del historical_config[name]
    historical_signature = hashlib.sha256(
        json.dumps(
            historical_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert historical_signature != current_signature
    payload["model_config_signature"] = historical_signature
    historical_path = tmp_path / "historical.pt"
    torch.save(payload, historical_path)

    resumed = TinyTrainer(
        TinyKMD2Model(config, init_seed=701),
        _tiny_training_config("pre-variant-defaults"),
    )
    resumed.load_checkpoint(historical_path)
    for name, expected in source.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[name], expected), name


def test_tiny_checkpoint_accepts_pre_bc_bias_default_config_signature(tmp_path) -> None:
    import hashlib
    import json
    from dataclasses import fields, is_dataclass

    from research.kmd2_ablation.tiny_backend import TinyKMD2Model
    from research.kmd2_ablation.tiny_training import TinyTrainer

    def canonical(value):
        if is_dataclass(value) and not isinstance(value, type):
            return {field.name: canonical(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, dict):
            return {str(key): canonical(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        return value

    config = _tiny_config()
    source = TinyTrainer(
        TinyKMD2Model(config, init_seed=702),
        _tiny_training_config("pre-bc-bias-defaults"),
    )
    current_path = tmp_path / "current.pt"
    source.save_checkpoint(current_path)
    payload = torch.load(current_path, map_location="cpu", weights_only=False)
    historical_config = canonical(config)
    assert all(
        name in historical_config
        for name in (
            "corrected_momentum",
            "momentum_gamma_init",
            "causal_lookahead",
            "lookahead_rho_init",
        )
    )
    del historical_config["bc_bias_mode"]
    payload["model_config_signature"] = hashlib.sha256(
        json.dumps(
            historical_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    historical_path = tmp_path / "pre-bc-bias.pt"
    torch.save(payload, historical_path)

    resumed = TinyTrainer(
        TinyKMD2Model(config, init_seed=703),
        _tiny_training_config("pre-bc-bias-defaults"),
    )
    resumed.load_checkpoint(historical_path)
    for name, expected in source.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[name], expected), name


def test_trapezoid_interaction_requires_individual_promotion() -> None:
    from research.kmd2_ablation.variants import trapezoid_convolution_interaction_allowed

    assert not trapezoid_convolution_interaction_allowed(trapezoid_promoted=False)
    assert trapezoid_convolution_interaction_allowed(trapezoid_promoted=True)
    with pytest.raises(TypeError, match="trapezoid_promoted"):
        trapezoid_convolution_interaction_allowed(trapezoid_promoted=1)  # type: ignore[arg-type]


def _qwen_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=12,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=3,
        rms_norm_eps=1.0e-6,
    )


def test_trapezoid_qwen_subclass_strictly_clones_native_and_adds_only_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("transformers")
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_variants import KMD2TrapezoidAttn

    monkeypatch.setenv("GDN3_KMD2_ROUT", "4")
    torch.manual_seed(88)
    native = KMD2NativeAttn(_qwen_config(), layer_idx=5)
    native.register_buffer("transfer_probe", torch.arange(3, dtype=torch.float64))
    native.rot_proj.bias.requires_grad_(False)
    native.eval()
    native.conv1d.train()
    native.transfer_metadata = {"nested": ["preserved"]}
    inherited = {
        name: value.detach().clone() for name, value in native.state_dict().items()
    }

    trapezoid = KMD2TrapezoidAttn.from_native(native)
    assert issubclass(KMD2TrapezoidAttn, KMD2NativeAttn)
    assert trapezoid is not native
    assert trapezoid.r_out == 4
    assert trapezoid.layer_idx == 5
    assert trapezoid.training is False
    assert trapezoid.conv1d.training is True
    assert trapezoid.transfer_metadata == native.transfer_metadata
    assert trapezoid.rho_head.dtype == torch.float32
    assert tuple(trapezoid.rho_head.shape) == (native.H,)
    assert tuple(trapezoid.rho_proj.weight.shape) == (
        native.H,
        native.in_proj_qkv.in_features,
    )
    assert set(trapezoid.state_dict()) - set(inherited) == {
        "rho_head",
        "rho_proj.weight",
    }
    for name, expected in inherited.items():
        assert torch.equal(trapezoid.state_dict()[name], expected), name
    assert trapezoid.rot_proj.bias.requires_grad is False
    assert trapezoid.transfer_probe.data_ptr() != native.transfer_probe.data_ptr()

    with pytest.raises(ValueError, match="already"):
        KMD2TrapezoidAttn.from_native(trapezoid)
    with pytest.raises(TypeError, match="KMD2NativeAttn"):
        KMD2TrapezoidAttn.from_native(torch.nn.Linear(2, 2))  # type: ignore[arg-type]


def test_trapezoid_qwen_zero_gate_identity_active_gradient_and_forces_python_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("transformers")
    import gdn3.kmd2_native as native_module
    import gdn3.kmd2_fast_scan as fast_scan
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_variants import KMD2TrapezoidAttn

    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    monkeypatch.setattr(native_module, "_FAST_SCAN", False)
    torch.manual_seed(507)
    native = KMD2NativeAttn(_qwen_config(), layer_idx=1)
    trapezoid = KMD2TrapezoidAttn.from_native(native)
    x_native = torch.randn(2, 5, 12, requires_grad=True)
    x_trapezoid = x_native.detach().clone().requires_grad_()
    y_native = native(x_native)
    y_trapezoid = trapezoid(x_trapezoid)
    assert torch.equal(y_trapezoid, y_native)
    y_native.square().sum().backward()
    y_trapezoid.square().sum().backward()
    assert torch.equal(x_trapezoid.grad, x_native.grad)
    for name, parameter in native.named_parameters():
        expected = parameter.grad
        actual = dict(trapezoid.named_parameters())[name].grad
        assert expected is not None and actual is not None, name
        assert torch.equal(actual, expected), name
    assert trapezoid.rho_head.grad is not None
    assert torch.isfinite(trapezoid.rho_head.grad).all()

    trapezoid.zero_grad(set_to_none=True)
    with torch.no_grad():
        trapezoid.rho_head.fill_(0.7)
        trapezoid.rho_proj.weight.fill_(0.05)
    active_input = x_native.detach().clone().requires_grad_()
    active = trapezoid(active_input)
    assert not torch.equal(active, native(x_native.detach()))
    active.square().sum().backward()
    assert trapezoid.rho_head.grad is not None
    assert trapezoid.rho_head.grad.abs().sum() > 0
    assert trapezoid.rho_proj.weight.grad is not None
    assert trapezoid.rho_proj.weight.grad.abs().sum() > 0

    called = False

    def forbidden_fast_scan(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("trapezoid must not dispatch to the native fast scan")

    monkeypatch.setattr(fast_scan, "scan", forbidden_fast_scan)
    monkeypatch.setattr(native_module, "_FAST_SCAN", True)
    forced = trapezoid(active_input.detach())
    assert torch.isfinite(forced).all()
    assert called is False


def test_trapezoid_qwen_module_plumbs_boundaries_and_rejects_packing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("transformers")
    import research.kmd2_ablation.qwen_variants as qwen_variants
    from gdn3.kmd2_native import KMD2NativeAttn

    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    native = KMD2NativeAttn(_qwen_config(), layer_idx=3)
    trapezoid = qwen_variants.KMD2TrapezoidAttn.from_native(native)
    hidden = torch.randn(2, 4, 12, generator=torch.Generator().manual_seed(55))
    boundaries = torch.tensor(
        [[True, False, True, False], [True, False, False, True]]
    )
    seen: list[torch.Tensor | None] = []
    reference = qwen_variants.trapezoid_reference_scan

    def recording_scan(*args, **kwargs):
        value = kwargs.get("boundaries")
        seen.append(None if value is None else value.detach().clone())
        return reference(*args, **kwargs)

    monkeypatch.setattr(qwen_variants, "trapezoid_reference_scan", recording_scan)
    output = trapezoid(hidden, boundaries=boundaries)
    assert torch.isfinite(output).all()
    assert len(seen) == 1 and torch.equal(seen[0], boundaries)

    with pytest.raises(ValueError, match=r"boundaries.*bool.*\[B,T\]"):
        trapezoid(hidden, boundaries=boundaries.float())
    with pytest.raises(ValueError, match="packed|segment_ids"):
        trapezoid(hidden, segment_ids=torch.zeros(2, 4, dtype=torch.int64))


def test_trapezoid_qwen_reference_loop_resets_carry_at_boundaries() -> None:
    pytest.importorskip("transformers")
    from research.kmd2_ablation.qwen_variants import trapezoid_reference_scan

    q = torch.tensor([[[[[1.0, 0.0]]], [[[1.0, 0.0]]]]])
    k = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
    v = torch.tensor([[[[1.0]], [[3.0]]]])
    decay = torch.full((1, 2, 1, 2), 0.5)
    beta_e = torch.zeros(1, 2, 1)
    beta_w = torch.ones(1, 2, 1)
    rho = torch.ones(1, 2, 1)
    no_boundary = trapezoid_reference_scan(q, k, v, decay, beta_e, beta_w, rho)
    reset = trapezoid_reference_scan(
        q,
        k,
        v,
        decay,
        beta_e,
        beta_w,
        rho,
        boundaries=torch.tensor([[True, True]]),
    )
    assert no_boundary[0, :, 0, 0].tolist() == pytest.approx([1.0, 1.0])
    assert reset[0, :, 0, 0].tolist() == pytest.approx([1.0, 3.0])


def _scalar_momentum_factors(*, reset_second: bool, requires_grad: bool = False):
    from research.kmd2_ablation.tiny_backend import TinyFactors

    positions = torch.tensor([[0, 0 if reset_second else 1]], dtype=torch.int64)
    gamma = torch.full((1, 2, 1), 0.5)
    if requires_grad:
        gamma.requires_grad_()
    return TinyFactors(
        q=torch.ones(1, 2, 1, 1, 1),
        k=torch.ones(1, 2, 1, 1, 1),
        v=torch.tensor([1.0, 2.0]).view(1, 2, 1, 1, 1),
        decay=torch.full((1, 2, 1, 1), 0.5),
        beta_e=torch.full((1, 2, 1, 1), 0.4),
        beta_w=torch.ones(1, 2, 1, 1),
        out_mix=torch.ones(1, 2, 1, 1),
        valid=torch.ones(1, 2, dtype=torch.bool),
        positions=positions,
        momentum_gamma=gamma,
    )


def test_momentum_tiny_matches_corrected_equations_resets_and_doubles_state() -> None:
    from research.kmd2_ablation.tiny_backend import TinyKMD2Cell

    cell = TinyKMD2Cell(
        _tiny_config(heads=1, dk=1, dv=1, corrected_momentum=True)
    )
    factors = _scalar_momentum_factors(reset_second=False, requires_grad=True)
    output = cell(factors, boundaries=torch.tensor([[True, False]]))

    # S_bar=.5, M_bar=.5, S_look=.75, G=2-.4*.75=1.7,
    # M=.5*.5+1.7=1.95, and S=.5+1.95=2.45.
    assert output.read[0, :, 0, 0].tolist() == pytest.approx([1.0, 2.45])
    assert output.final_state[0, 0, 0, 0].item() == pytest.approx(2.45)
    assert output.scores[0, :, 0].tolist() == pytest.approx([1.0, 1.7])
    assert output.state_bytes == 2 * 1 * 1 * 1 * 1 * 4

    output.read.square().sum().backward()
    assert factors.momentum_gamma.grad is not None
    assert torch.isfinite(factors.momentum_gamma.grad).all()
    assert factors.momentum_gamma.grad[0, 1, 0].abs() > 0

    reset_factors = _scalar_momentum_factors(reset_second=True)
    reset = cell(reset_factors, boundaries=torch.tensor([[True, True]]))
    assert reset.read[0, :, 0, 0].tolist() == pytest.approx([1.0, 2.0])


def test_momentum_tiny_zero_gamma_is_exact_native_and_active_gradients_are_finite() -> None:
    from research.kmd2_ablation.tiny_backend import TinyKMD2Cell

    native_factors = _tiny_factors(requires_grad=True)
    momentum_factors = _tiny_factors(
        momentum_gamma=torch.zeros(1, 4, 1), requires_grad=True
    )
    native = TinyKMD2Cell(_tiny_config())(native_factors)
    momentum = TinyKMD2Cell(_tiny_config(corrected_momentum=True))(momentum_factors)
    assert torch.equal(momentum.read, native.read)
    assert torch.equal(momentum.final_state, native.final_state)
    assert torch.equal(momentum.scores, native.scores)
    assert momentum.state_bytes == 2 * native.state_bytes

    native_loss = native.read.square().sum() + native.final_state.square().sum()
    momentum_loss = (
        momentum.read.square().sum() + momentum.final_state.square().sum()
    )
    native_loss.backward()
    momentum_loss.backward()
    for expected, actual in zip(
        _factor_grads(native_factors), _factor_grads(momentum_factors), strict=True
    ):
        assert torch.equal(actual, expected)
    assert momentum_factors.momentum_gamma.grad is not None
    assert torch.isfinite(momentum_factors.momentum_gamma.grad).all()

    active_factors = _tiny_factors(
        momentum_gamma=torch.full((1, 4, 1), 0.4), requires_grad=True
    )
    active = TinyKMD2Cell(_tiny_config(corrected_momentum=True))(active_factors)
    assert not torch.equal(active.read, native.read.detach())
    active.read.square().sum().backward()
    assert active_factors.momentum_gamma.grad is not None
    assert torch.isfinite(active_factors.momentum_gamma.grad).all()
    assert active_factors.momentum_gamma.grad.abs().sum() > 0


def test_momentum_decay_erase_interaction_requires_individual_promotion() -> None:
    from research.kmd2_ablation.variants import (
        momentum_decay_erase_interaction_allowed,
    )

    assert not momentum_decay_erase_interaction_allowed(momentum_promoted=False)
    assert momentum_decay_erase_interaction_allowed(momentum_promoted=True)
    with pytest.raises(TypeError, match="momentum_promoted"):
        momentum_decay_erase_interaction_allowed(momentum_promoted=1)  # type: ignore[arg-type]


def test_momentum_qwen_reference_equation_and_boundary_reset() -> None:
    pytest.importorskip("transformers")
    from research.kmd2_ablation.qwen_variants import momentum_reference_scan

    q = torch.ones(1, 2, 1, 1, 1)
    k = torch.ones(1, 2, 1, 1)
    v = torch.tensor([1.0, 2.0]).view(1, 2, 1, 1)
    decay = torch.full((1, 2, 1, 1), 0.5)
    beta_e = torch.full((1, 2, 1), 0.4)
    beta_w = torch.ones(1, 2, 1)
    gamma = torch.full((1, 2, 1), 0.5)
    active = momentum_reference_scan(q, k, v, decay, beta_e, beta_w, gamma)
    reset = momentum_reference_scan(
        q,
        k,
        v,
        decay,
        beta_e,
        beta_w,
        gamma,
        boundaries=torch.tensor([[True, True]]),
    )
    assert active[0, :, 0, 0].tolist() == pytest.approx([1.0, 2.45])
    assert reset[0, :, 0, 0].tolist() == pytest.approx([1.0, 2.0])


def test_momentum_qwen_zero_gate_gradient_matches_positive_boundary_derivative() -> None:
    pytest.importorskip("transformers")
    from research.kmd2_ablation.qwen_variants import momentum_reference_scan

    q = torch.ones(1, 2, 1, 1, 1)
    k = torch.ones(1, 2, 1, 1)
    v = torch.tensor([1.0, 2.0]).view(1, 2, 1, 1)
    decay = torch.full((1, 2, 1, 1), 0.5)
    beta_e = torch.full((1, 2, 1), 0.4)
    beta_w = torch.ones(1, 2, 1)
    gamma = torch.zeros(1, 2, 1, requires_grad=True)

    zero = momentum_reference_scan(q, k, v, decay, beta_e, beta_w, gamma)
    gradient = torch.autograd.grad(zero[0, 1, 0, 0], gamma)[0][0, 1, 0]
    epsilon = 1.0e-3
    positive_gamma = gamma.detach().clone()
    positive_gamma[0, 1, 0] = epsilon
    positive = momentum_reference_scan(
        q, k, v, decay, beta_e, beta_w, positive_gamma
    )
    one_sided = (positive[0, 1, 0, 0] - zero[0, 1, 0, 0].detach()) / epsilon

    assert one_sided.item() == pytest.approx(0.3, abs=2.0e-4)
    assert gradient.item() == pytest.approx(one_sided.item(), abs=2.0e-4)


def test_momentum_qwen_zero_identity_active_gradient_and_forces_python_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("transformers")
    import gdn3.kmd2_fast_scan as fast_scan
    import gdn3.kmd2_native as native_module
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_variants import KMD2MomentumAttn

    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    monkeypatch.setattr(native_module, "_FAST_SCAN", False)
    torch.manual_seed(911)
    native = KMD2NativeAttn(_qwen_config(), layer_idx=7)
    inherited = {
        name: value.detach().clone() for name, value in native.state_dict().items()
    }
    momentum = KMD2MomentumAttn.from_native(native)
    assert issubclass(KMD2MomentumAttn, KMD2NativeAttn)
    assert set(momentum.state_dict()) - set(inherited) == {"momentum_gamma"}
    assert momentum.dynamic_state_multiplier == 2
    for name, expected in inherited.items():
        assert torch.equal(momentum.state_dict()[name], expected), name

    x_native = torch.randn(2, 5, 12, requires_grad=True)
    x_momentum = x_native.detach().clone().requires_grad_()
    y_native = native(x_native)
    y_momentum = momentum(x_momentum)
    assert torch.equal(y_momentum, y_native)
    y_native.square().sum().backward()
    y_momentum.square().sum().backward()
    assert torch.equal(x_momentum.grad, x_native.grad)
    for name, parameter in native.named_parameters():
        expected = parameter.grad
        actual = dict(momentum.named_parameters())[name].grad
        assert expected is not None and actual is not None, name
        assert torch.equal(actual, expected), name
    assert momentum.momentum_gamma.grad is not None
    assert torch.isfinite(momentum.momentum_gamma.grad).all()

    momentum.zero_grad(set_to_none=True)
    with torch.no_grad():
        momentum.momentum_gamma.fill_(0.5)
    active_input = x_native.detach().clone().requires_grad_()
    active = momentum(active_input)
    assert not torch.equal(active, native(x_native.detach()))
    active.square().sum().backward()
    assert momentum.momentum_gamma.grad is not None
    assert torch.isfinite(momentum.momentum_gamma.grad).all()
    assert momentum.momentum_gamma.grad.abs().sum() > 0

    called = False

    def forbidden_fast_scan(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("momentum must not dispatch to the native fast scan")

    monkeypatch.setattr(fast_scan, "scan", forbidden_fast_scan)
    monkeypatch.setattr(native_module, "_FAST_SCAN", True)
    forced = momentum(active_input.detach())
    assert torch.isfinite(forced).all()
    assert called is False


def _scalar_lookahead_factors(*, reset_second: bool, requires_grad: bool = False):
    from research.kmd2_ablation.tiny_backend import TinyFactors

    positions = torch.tensor([[0, 0 if reset_second else 1]], dtype=torch.int64)
    rho = torch.full((1, 2, 1), 0.5)
    if requires_grad:
        rho.requires_grad_()
    return TinyFactors(
        q=torch.ones(1, 2, 1, 1, 1),
        k=torch.ones(1, 2, 1, 1, 1),
        v=torch.tensor([1.0, 3.0]).view(1, 2, 1, 1, 1),
        decay=torch.full((1, 2, 1, 1), 0.5),
        beta_e=torch.full((1, 2, 1, 1), 0.4),
        beta_w=torch.ones(1, 2, 1, 1),
        out_mix=torch.ones(1, 2, 1, 1),
        valid=torch.ones(1, 2, dtype=torch.bool),
        positions=positions,
        lookahead_rho=rho,
    )


def test_lookahead_tiny_matches_value_target_resets_and_tracks_previous_value() -> None:
    from research.kmd2_ablation.tiny_backend import TinyKMD2Cell

    cell = TinyKMD2Cell(_tiny_config(heads=1, dk=1, dv=1, causal_lookahead=True))
    assert torch.equal(cell.lookahead_projection.weight, torch.ones(1, 1))
    factors = _scalar_lookahead_factors(reset_second=False, requires_grad=True)
    output = cell(factors, boundaries=torch.tensor([[True, False]]))

    # At t=1: target=3+.5*(3-1)=4, S_bar=.5, error=4-.4*.5=3.8.
    assert output.read[0, :, 0, 0].tolist() == pytest.approx([1.0, 4.3])
    assert output.final_state[0, 0, 0, 0].item() == pytest.approx(4.3)
    assert output.scores[0, :, 0].tolist() == pytest.approx([1.0, 3.8])
    assert output.state_bytes == 1 * 1 * 1 * 1 * 4 + 1 * 1 * 1 * 4

    output.read.square().sum().backward()
    assert factors.lookahead_rho.grad is not None
    assert torch.isfinite(factors.lookahead_rho.grad).all()
    assert factors.lookahead_rho.grad[0, 1, 0].abs() > 0
    assert cell.lookahead_projection.weight.grad is not None
    assert torch.isfinite(cell.lookahead_projection.weight.grad).all()
    assert cell.lookahead_projection.weight.grad.abs().sum() > 0

    reset_factors = _scalar_lookahead_factors(reset_second=True)
    reset = cell(reset_factors, boundaries=torch.tensor([[True, True]]))
    assert reset.read[0, :, 0, 0].tolist() == pytest.approx([1.0, 3.0])


def test_lookahead_tiny_zero_rho_is_exact_native_and_active_gradients_are_finite() -> None:
    from research.kmd2_ablation.tiny_backend import TinyKMD2Cell

    native_factors = _tiny_factors(requires_grad=True)
    lookahead_factors = _tiny_factors(
        lookahead_rho=torch.zeros(1, 4, 1), requires_grad=True
    )
    native = TinyKMD2Cell(_tiny_config())(native_factors)
    lookahead_cell = TinyKMD2Cell(_tiny_config(causal_lookahead=True))
    lookahead = lookahead_cell(lookahead_factors)
    assert torch.equal(lookahead.read, native.read)
    assert torch.equal(lookahead.final_state, native.final_state)
    assert torch.equal(lookahead.scores, native.scores)

    native_loss = native.read.square().sum() + native.final_state.square().sum()
    lookahead_loss = (
        lookahead.read.square().sum() + lookahead.final_state.square().sum()
    )
    native_loss.backward()
    lookahead_loss.backward()
    for expected, actual in zip(
        _factor_grads(native_factors), _factor_grads(lookahead_factors), strict=True
    ):
        assert torch.equal(actual, expected)
    assert lookahead_factors.lookahead_rho.grad is not None
    assert torch.isfinite(lookahead_factors.lookahead_rho.grad).all()

    active_factors = _tiny_factors(
        lookahead_rho=torch.full((1, 4, 1), 0.4), requires_grad=True
    )
    active_cell = TinyKMD2Cell(_tiny_config(causal_lookahead=True))
    active = active_cell(active_factors)
    assert not torch.equal(active.read, native.read.detach())
    active.read.square().sum().backward()
    assert active_factors.lookahead_rho.grad is not None
    assert torch.isfinite(active_factors.lookahead_rho.grad).all()
    assert active_factors.lookahead_rho.grad.abs().sum() > 0
    assert active_cell.lookahead_projection.weight.grad is not None
    assert torch.isfinite(active_cell.lookahead_projection.weight.grad).all()
    assert active_cell.lookahead_projection.weight.grad.abs().sum() > 0


def test_lookahead_interactions_require_individual_promotions() -> None:
    from research.kmd2_ablation.variants import (
        lookahead_convolution_interaction_allowed,
        lookahead_trapezoid_interaction_allowed,
    )

    assert not lookahead_convolution_interaction_allowed(lookahead_promoted=False)
    assert lookahead_convolution_interaction_allowed(lookahead_promoted=True)
    assert not lookahead_trapezoid_interaction_allowed(
        lookahead_promoted=False, trapezoid_promoted=True
    )
    assert not lookahead_trapezoid_interaction_allowed(
        lookahead_promoted=True, trapezoid_promoted=False
    )
    assert lookahead_trapezoid_interaction_allowed(
        lookahead_promoted=True, trapezoid_promoted=True
    )
    with pytest.raises(TypeError, match="lookahead_promoted"):
        lookahead_convolution_interaction_allowed(lookahead_promoted=1)  # type: ignore[arg-type]


def test_lookahead_qwen_reference_target_and_boundary_reset() -> None:
    pytest.importorskip("transformers")
    from research.kmd2_ablation.qwen_variants import lookahead_reference_scan

    q = torch.ones(1, 2, 1, 1, 1)
    k = torch.ones(1, 2, 1, 1)
    v = torch.tensor([1.0, 3.0]).view(1, 2, 1, 1)
    decay = torch.full((1, 2, 1, 1), 0.5)
    beta_e = torch.full((1, 2, 1), 0.4)
    beta_w = torch.ones(1, 2, 1)
    rho = torch.full((1, 2, 1), 0.5)
    projection = torch.full((1, 1), 2.0)
    active = lookahead_reference_scan(
        q, k, v, decay, beta_e, beta_w, rho, projection
    )
    reset = lookahead_reference_scan(
        q,
        k,
        v,
        decay,
        beta_e,
        beta_w,
        rho,
        projection,
        boundaries=torch.tensor([[True, True]]),
    )
    # P=2 makes the second target 3+.5*2*(3-1)=5.
    assert active[0, :, 0, 0].tolist() == pytest.approx([1.0, 5.3])
    assert reset[0, :, 0, 0].tolist() == pytest.approx([1.0, 3.0])


def test_lookahead_qwen_zero_identity_active_gradient_and_forces_python_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("transformers")
    import gdn3.kmd2_fast_scan as fast_scan
    import gdn3.kmd2_native as native_module
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_variants import KMD2LookaheadAttn

    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    monkeypatch.setattr(native_module, "_FAST_SCAN", False)
    torch.manual_seed(1911)
    native = KMD2NativeAttn(_qwen_config(), layer_idx=8)
    inherited = {
        name: value.detach().clone() for name, value in native.state_dict().items()
    }
    lookahead = KMD2LookaheadAttn.from_native(native)
    assert issubclass(KMD2LookaheadAttn, KMD2NativeAttn)
    assert set(lookahead.state_dict()) - set(inherited) == {
        "lookahead_rho",
        "lookahead_projection.weight",
    }
    for name, expected in inherited.items():
        assert torch.equal(lookahead.state_dict()[name], expected), name

    x_native = torch.randn(2, 5, 12, requires_grad=True)
    x_lookahead = x_native.detach().clone().requires_grad_()
    y_native = native(x_native)
    y_lookahead = lookahead(x_lookahead)
    assert torch.equal(y_lookahead, y_native)
    y_native.square().sum().backward()
    y_lookahead.square().sum().backward()
    assert torch.equal(x_lookahead.grad, x_native.grad)
    for name, parameter in native.named_parameters():
        expected = parameter.grad
        actual = dict(lookahead.named_parameters())[name].grad
        assert expected is not None and actual is not None, name
        assert torch.equal(actual, expected), name
    assert lookahead.lookahead_rho.grad is not None
    assert torch.isfinite(lookahead.lookahead_rho.grad).all()

    lookahead.zero_grad(set_to_none=True)
    with torch.no_grad():
        lookahead.lookahead_rho.fill_(0.5)
    active_input = x_native.detach().clone().requires_grad_()
    active = lookahead(active_input)
    assert not torch.equal(active, native(x_native.detach()))
    active.square().sum().backward()
    assert lookahead.lookahead_rho.grad is not None
    assert torch.isfinite(lookahead.lookahead_rho.grad).all()
    assert lookahead.lookahead_rho.grad.abs().sum() > 0
    assert lookahead.lookahead_projection.weight.grad is not None
    assert torch.isfinite(lookahead.lookahead_projection.weight.grad).all()
    assert lookahead.lookahead_projection.weight.grad.abs().sum() > 0

    called = False

    def forbidden_fast_scan(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("lookahead must not dispatch to the native fast scan")

    monkeypatch.setattr(fast_scan, "scan", forbidden_fast_scan)
    monkeypatch.setattr(native_module, "_FAST_SCAN", True)
    forced = lookahead(active_input.detach())
    assert torch.isfinite(forced).all()
    assert called is False


class _VariantGateHealModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.memory_weight = torch.nn.Parameter(torch.eye(3))
        self.momentum_gamma = torch.nn.Parameter(torch.tensor([0.5]))
        self.lookahead_rho = torch.nn.Parameter(torch.tensor([0.5]))

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool,
        use_cache: bool,
    ) -> SimpleNamespace:
        assert output_hidden_states is True and use_cache is False
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=3).float()
        gate = self.momentum_gamma + self.lookahead_rho
        return SimpleNamespace(logits=one_hot @ self.memory_weight + gate * one_hot)


def test_qwen_shared_variant_gate_projector_clamps_both_coefficients() -> None:
    from research.kmd2_ablation.qwen_variants import project_variant_gates_

    model = _VariantGateHealModel()
    with torch.no_grad():
        model.momentum_gamma.fill_(1.5)
        model.lookahead_rho.fill_(-0.5)
    assert project_variant_gates_(model) == (
        "momentum_gamma",
        "lookahead_rho",
    )
    assert model.momentum_gamma.item() == 1.0
    assert model.lookahead_rho.item() == 0.0


def test_qwen_heal_post_step_projects_variant_gates_after_crossing_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.kmd2_ablation.qwen_training import (
        QwenHealTrainer,
        QwenHealTrainingConfig,
    )

    model = _VariantGateHealModel()
    optimizer = torch.optim.SGD(
        [
            {
                "name": "memory",
                "params": list(model.parameters()),
                "lr": 0.1,
            }
        ]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    original_step = optimizer.step

    def crossing_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        with torch.no_grad():
            model.momentum_gamma.fill_(1.5)
            model.lookahead_rho.fill_(-0.5)
        return result

    monkeypatch.setattr(optimizer, "step", crossing_step)
    trainer = QwenHealTrainer(
        model=model,
        teacher=None,
        optimizer=optimizer,
        scheduler=scheduler,
        config=QwenHealTrainingConfig(
            objective="synthetic_only",
            ce_weight=1.0,
            kl_weight=0.0,
            layerwise_weight=0.0,
            temperature=1.0,
            accumulation_steps=1,
            max_updates=1,
            max_tokens=3,
            gradient_checkpointing=False,
        ),
        job_id="variant-gate-projection",
        pairing_id="a" * 64,
        arm="native",
        expected_example_windows=(("e0",),),
    )
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    trainer.train_update(
        (
            {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "example_ids": ("e0",),
            },
        )
    )

    assert model.momentum_gamma.item() == 1.0
    assert model.lookahead_rho.item() == 0.0


def test_bc_bias_tiny_zero_identity_active_affine_effect_and_equal_control() -> None:
    from research.kmd2_ablation.tiny_backend import (
        TinyKMD2Cell,
        apply_bc_additive,
        apply_bc_diagonal_rescale,
    )

    native_factors = _tiny_factors(requires_grad=True)
    biased_factors = _tiny_factors(requires_grad=True)
    native = TinyKMD2Cell(_tiny_config())(native_factors)
    biased_cell = TinyKMD2Cell(_tiny_config(bc_bias_mode="additive"))
    biased = biased_cell(biased_factors)
    assert torch.equal(biased.read, native.read)
    assert torch.equal(biased.final_state, native.final_state)
    assert torch.equal(biased.scores, native.scores)

    diagonal_cell = TinyKMD2Cell(_tiny_config(bc_bias_mode="diagonal_rescale"))
    additive_count = sum(parameter.numel() for parameter in biased_cell.parameters())
    diagonal_count = sum(parameter.numel() for parameter in diagonal_cell.parameters())
    assert additive_count == diagonal_count == 2 * 1 * (2 + 1)

    zero_q = torch.zeros(1, 1, 1, 1, 2)
    zero_k = torch.zeros(1, 1, 1, 1, 2)
    amplitude = torch.ones(1)
    vectors = torch.tensor([[1.5, -0.5]])
    additive_q, additive_k = apply_bc_additive(
        zero_q, zero_k, amplitude, amplitude, vectors, vectors
    )
    diagonal_q, diagonal_k = apply_bc_diagonal_rescale(
        zero_q, zero_k, amplitude, amplitude, vectors, vectors
    )
    assert additive_q.count_nonzero() and additive_k.count_nonzero()
    assert diagonal_q.count_nonzero() == diagonal_k.count_nonzero() == 0

    with torch.no_grad():
        biased_cell.bc_q_amplitude.fill_(0.6)
        biased_cell.bc_k_amplitude.fill_(0.4)
        biased_cell.bc_q_bias.copy_(vectors)
        biased_cell.bc_k_bias.copy_(-vectors)
    active_factors = _tiny_factors(requires_grad=True)
    active = biased_cell(active_factors)
    assert not torch.equal(active.read, native.read.detach())
    active.read.square().sum().backward()
    for name in (
        "bc_q_amplitude",
        "bc_k_amplitude",
        "bc_q_bias",
        "bc_k_bias",
    ):
        gradient = dict(biased_cell.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name


def test_bc_bias_constant_coordinate_oracle_appends_exact_constant_basis() -> None:
    from research.kmd2_ablation.tiny_backend import append_constant_coordinate

    q = torch.randn(2, 3, 1, 1, 4, generator=torch.Generator().manual_seed(81))
    k = torch.randn(2, 3, 1, 1, 4, generator=torch.Generator().manual_seed(82))
    q_oracle, k_oracle = append_constant_coordinate(q, k)
    assert q_oracle.shape[-1] == k_oracle.shape[-1] == 5
    assert torch.equal(q_oracle[..., :-1], q)
    assert torch.equal(k_oracle[..., :-1], k)
    assert torch.equal(q_oracle[..., -1], torch.ones_like(q_oracle[..., -1]))
    assert torch.equal(k_oracle[..., -1], torch.ones_like(k_oracle[..., -1]))


def test_bc_bias_constant_coordinate_oracle_runs_projector_and_cell_honestly() -> None:
    from research.kmd2_ablation.tiny_backend import (
        TinyFactorProjector,
        TinyKMD2Cell,
    )
    from research.kmd2_ablation.variants import get_variant

    arm = get_variant("bc_bias.constant_coordinate_oracle")
    config = _tiny_config(dk=4, bc_bias_mode=arm.variant)
    native_config = _tiny_config(dk=4)
    assert arm.changed_parameters == (
        "q_proj.weight",
        "k_proj.weight",
        "conv.weight",
        "q_slot_scale",
        "decay_chan",
    )
    assert arm.changed_state == ("constant_coordinate",)

    projector = TinyFactorProjector(config)
    native_projector = TinyFactorProjector(native_config)
    hidden = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(1456))
    valid = torch.ones(2, 3, dtype=torch.bool)
    positions = torch.arange(3, dtype=torch.int64).repeat(2, 1)

    factors = projector(hidden, valid, positions)
    assert factors.q.shape == factors.k.shape == (2, 3, 1, 1, config.dk)
    assert factors.decay.shape == (2, 3, 1, config.dk)
    assert torch.equal(factors.q[..., -1], torch.ones_like(factors.q[..., -1]))
    assert torch.equal(factors.k[..., -1], torch.ones_like(factors.k[..., -1]))

    cell = TinyKMD2Cell(config)
    output = cell(factors)
    assert output.final_state.shape == (2, config.heads, config.dk, config.dv)
    assert output.state_bytes == 2 * config.heads * config.dk * config.dv * 4
    assert not tuple(cell.parameters())
    assert torch.isfinite(output.read).all()

    removed_parameters = (
        2 * config.heads * config.mimo_rank * config.d_model
        + 2 * config.heads * config.mimo_rank * config.conv_kernel
        + config.heads * config.r_out
        + config.heads
    )
    native_count = sum(parameter.numel() for parameter in native_projector.parameters())
    oracle_count = sum(parameter.numel() for parameter in projector.parameters())
    assert native_count - oracle_count == removed_parameters


def test_bc_bias_constant_coordinate_oracle_runs_declared_affine_episode() -> None:
    from research.kmd2_ablation.tasks import generate_task
    from research.kmd2_ablation.tiny_backend import TinyKMD2Model

    episode = generate_task(
        "affine_associative_regression",
        batch_size=2,
        length=3,
        seed=1457,
        split="train",
        params={"input_dim": 3, "output_dim": 2},
    )
    config = _tiny_config(
        dk=4,
        dv=2,
        output_dim=2,
        bc_bias_mode="constant_coordinate_oracle",
    )
    output = TinyKMD2Model(config, init_seed=1457).forward_episode(episode)
    cell_output = output.cell_outputs[0]
    assert output.loss is not None and torch.isfinite(output.loss)
    assert cell_output.final_state.shape == (2, config.heads, config.dk, config.dv)
    assert cell_output.state_bytes == 2 * config.heads * config.dk * config.dv * 4


def test_bc_bias_constant_coordinate_direct_and_projected_decay_semantics_match() -> None:
    from research.kmd2_ablation.tiny_backend import (
        TinyFactorProjector,
        TinyFactors,
        TinyKMD2Cell,
    )

    config = _tiny_config(dk=4, bc_bias_mode="constant_coordinate_oracle")
    projector = TinyFactorProjector(config)
    hidden = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(1458))
    valid = torch.ones(1, 4, dtype=torch.bool)
    positions = torch.arange(4, dtype=torch.int64).view(1, 4)
    projected = projector(hidden, valid, positions)
    assert torch.equal(projected.decay[..., -1], projected.decay[..., 0])
    assert bool((projected.decay[..., -1] < 1.0).all())

    direct = TinyFactors(
        q=projected.q[..., :-1],
        k=projected.k[..., :-1],
        v=projected.v,
        decay=projected.decay[..., :-1],
        beta_e=projected.beta_e,
        beta_w=projected.beta_w,
        out_mix=projected.out_mix,
        valid=projected.valid,
        positions=projected.positions,
        read_gate=projected.read_gate,
    )
    cell = TinyKMD2Cell(config)
    projected_output = cell(projected)
    direct_output = cell(direct)
    assert torch.equal(direct_output.final_state, projected_output.final_state)
    assert torch.equal(direct_output.read, projected_output.read)
    assert torch.equal(direct_output.scores, projected_output.scores)


def test_bc_bias_constant_coordinate_raw_factors_reject_ambiguous_decay() -> None:
    from research.kmd2_ablation.tiny_backend import TinyFactors, TinyKMD2Cell

    source = _tiny_factors(steps=2)
    direct = TinyFactors(
        q=torch.cat((source.q, source.q), dim=-1),
        k=torch.cat((source.k, source.k), dim=-1),
        v=source.v,
        decay=torch.tensor([[[[0.8, 0.7, 0.6, 0.5]], [[0.7, 0.6, 0.5, 0.4]]]]),
        beta_e=source.beta_e,
        beta_w=source.beta_w,
        out_mix=source.out_mix,
        valid=source.valid,
        positions=source.positions,
    )
    with pytest.raises(ValueError, match="channel-tied decay"):
        TinyKMD2Cell(
            _tiny_config(dk=5, bc_bias_mode="constant_coordinate_oracle")
        )(direct)


def test_bc_bias_interactions_require_individual_wins() -> None:
    from research.kmd2_ablation.variants import (
        bc_bias_pair_convolution_interaction_allowed,
        bc_bias_trapezoid_interaction_allowed,
    )

    assert not bc_bias_trapezoid_interaction_allowed(
        bc_bias_promoted=False, trapezoid_promoted=True
    )
    assert not bc_bias_trapezoid_interaction_allowed(
        bc_bias_promoted=True, trapezoid_promoted=False
    )
    assert bc_bias_trapezoid_interaction_allowed(
        bc_bias_promoted=True, trapezoid_promoted=True
    )
    assert not bc_bias_pair_convolution_interaction_allowed(
        bc_bias_promoted=True,
        trapezoid_promoted=True,
        pair_promoted=False,
    )
    assert bc_bias_pair_convolution_interaction_allowed(
        bc_bias_promoted=True,
        trapezoid_promoted=True,
        pair_promoted=True,
    )


def test_bc_bias_qwen_zero_identity_active_effect_and_finite_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("transformers")
    import gdn3.kmd2_native as native_module
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_variants import KMD2BCBiasAttn

    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    monkeypatch.setattr(native_module, "_FAST_SCAN", False)
    torch.manual_seed(712)
    native = KMD2NativeAttn(_qwen_config(), layer_idx=9)
    inherited = {
        name: value.detach().clone() for name, value in native.state_dict().items()
    }
    biased = KMD2BCBiasAttn.from_native(native)
    assert set(biased.state_dict()) - set(inherited) == {
        "bc_q_amplitude",
        "bc_k_amplitude",
        "bc_q_bias",
        "bc_k_bias",
    }

    x_native = torch.randn(2, 5, 12, requires_grad=True)
    x_biased = x_native.detach().clone().requires_grad_()
    y_native = native(x_native)
    y_biased = biased(x_biased)
    assert torch.equal(y_biased, y_native)
    y_native.square().sum().backward()
    y_biased.square().sum().backward()
    assert torch.equal(x_biased.grad, x_native.grad)
    for name, parameter in native.named_parameters():
        actual = dict(biased.named_parameters())[name].grad
        assert parameter.grad is not None and actual is not None, name
        assert torch.equal(actual, parameter.grad), name

    biased.zero_grad(set_to_none=True)
    with torch.no_grad():
        biased.bc_q_amplitude.fill_(0.5)
        biased.bc_k_amplitude.fill_(0.4)
        biased.bc_q_bias.fill_(0.25)
        biased.bc_k_bias.fill_(-0.2)
    active_input = x_native.detach().clone().requires_grad_()
    active = biased(active_input)
    assert not torch.equal(active, native(x_native.detach()))
    active.square().sum().backward()
    for name in (
        "bc_q_amplitude",
        "bc_k_amplitude",
        "bc_q_bias",
        "bc_k_bias",
    ):
        gradient = dict(biased.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name


def test_true_mimo_rank_one_update_is_exact_siso_forward_and_gradient() -> None:
    from research.kmd2_ablation.tiny_backend import true_mimo_update

    generator = torch.Generator().manual_seed(3301)

    def leaves():
        return tuple(
            tensor.requires_grad_()
            for tensor in (
                torch.randn(2, 2, 3, 2, generator=generator),
                torch.randn(2, 2, 1, 3, generator=generator),
                torch.randn(2, 2, 1, 2, generator=generator),
                torch.sigmoid(torch.randn(2, 2, 1, generator=generator)),
                torch.sigmoid(torch.randn(2, 2, 1, generator=generator)),
            )
        )

    state, key, value, beta_e, beta_w = leaves()
    expected_state = state.detach().clone().requires_grad_()
    expected_key = key.detach().clone().requires_grad_()
    expected_value = value.detach().clone().requires_grad_()
    expected_beta_e = beta_e.detach().clone().requires_grad_()
    expected_beta_w = beta_w.detach().clone().requires_grad_()

    actual = true_mimo_update(state, key, value, beta_e, beta_w)
    key_one = expected_key[:, :, 0]
    value_one = expected_value[:, :, 0]
    memory = torch.matmul(key_one.unsqueeze(-2), expected_state).squeeze(-2)
    update = (
        expected_beta_w[:, :, 0].unsqueeze(-1) * value_one
        - expected_beta_e[:, :, 0].unsqueeze(-1) * memory
    )
    expected = expected_state + key_one.unsqueeze(-1) * update.unsqueeze(-2)
    assert torch.equal(actual, expected)

    actual_gradients = torch.autograd.grad(
        actual.square().sum(), (state, key, value, beta_e, beta_w)
    )
    expected_gradients = torch.autograd.grad(
        expected.square().sum(),
        (
            expected_state,
            expected_key,
            expected_value,
            expected_beta_e,
            expected_beta_w,
        ),
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.equal(actual_gradient, expected_gradient)


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("integer", TypeError, "floating"),
        ("mixed_dtype", ValueError, "dtype"),
        ("nonfinite", ValueError, "finite"),
        ("negative_beta_e", ValueError, "beta_e.*nonnegative"),
        ("device", ValueError, "device"),
    ],
)
def test_true_mimo_update_rejects_invalid_operand_contracts(
    case: str, error: type[Exception], message: str
) -> None:
    from research.kmd2_ablation.tiny_backend import true_mimo_update

    operands = [
        torch.zeros(1, 1, 2, 2),
        torch.ones(1, 1, 2, 2),
        torch.ones(1, 1, 2, 2),
        torch.full((1, 1, 2), 0.5),
        torch.full((1, 1, 2), 0.5),
    ]
    if case == "integer":
        operands[1] = operands[1].to(torch.int64)
    elif case == "mixed_dtype":
        operands[2] = operands[2].to(torch.float64)
    elif case == "nonfinite":
        operands[2][0, 0, 0, 0] = float("nan")
    elif case == "negative_beta_e":
        operands[3][0, 0, 0] = -0.1
    else:
        operands[4] = operands[4].to("meta")

    with pytest.raises(error, match=message):
        true_mimo_update(*operands)


@pytest.mark.parametrize(
    ("feature", "overrides"),
    [
        ("trapezoid", {"trapezoid": True}),
        ("corrected_momentum", {"corrected_momentum": True}),
        ("causal_lookahead", {"causal_lookahead": True}),
    ],
)
def test_true_mimo_config_rejects_siso_only_recurrence_features(
    feature: str, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match=rf"mimo_rank.*{feature}"):
        _tiny_config(mimo_rank=2, **overrides)


def _true_mimo_factors(*, rank: int = 3, requires_grad: bool = False):
    from research.kmd2_ablation.tiny_backend import TinyFactors

    generator = torch.Generator().manual_seed(4410 + rank)
    tensors = {
        "q": torch.randn(1, 3, 1, rank, 2, generator=generator),
        "k": torch.randn(1, 3, 1, rank, 2, generator=generator),
        "v": torch.randn(1, 3, 1, rank, 2, generator=generator),
        "decay": torch.sigmoid(torch.randn(1, 3, 1, 2, generator=generator)),
        "beta_e": torch.sigmoid(torch.randn(1, 3, 1, rank, generator=generator)),
        "beta_w": torch.sigmoid(torch.randn(1, 3, 1, rank, generator=generator)),
        "out_mix": torch.randn(1, 3, 1, rank, 2, generator=generator),
        "read_gate": torch.randn(1, 3, 1, rank, 2, generator=generator),
    }
    if requires_grad:
        for tensor in tensors.values():
            tensor.requires_grad_()
    return TinyFactors(
        **tensors,
        valid=torch.ones(1, 3, dtype=torch.bool),
        positions=torch.arange(3, dtype=torch.int64).view(1, 3),
    )


def test_true_mimo_cell_matches_simultaneous_equations_and_finite_gradients() -> None:
    from research.kmd2_ablation.tiny_backend import TinyKMD2Cell

    rank = 2
    factors = _true_mimo_factors(rank=rank, requires_grad=True)
    initial = torch.randn(1, 1, 2, 2, generator=torch.Generator().manual_seed(49))
    output = TinyKMD2Cell(_tiny_config(mimo_rank=rank))(factors, state=initial)

    state = initial
    reads = []
    for token in range(3):
        state_bar = factors.decay[:, token].unsqueeze(-1) * state
        key = factors.k[:, token]
        value = factors.v[:, token]
        erase_key = torch.sqrt(factors.beta_e[:, token] / rank).unsqueeze(-1) * key
        state = state_bar - torch.einsum(
            "bhrd,bhrv->bhdv", erase_key, torch.matmul(erase_key, state_bar)
        )
        state = state + torch.einsum(
            "bhrd,bhrv->bhdv",
            key,
            factors.beta_w[:, token].unsqueeze(-1) * value,
        )
        slots = torch.matmul(factors.q[:, token], state)
        gated = slots * torch.nn.functional.silu(factors.read_gate[:, token])
        reads.append((gated * factors.out_mix[:, token]).sum(-2))
    expected_read = torch.stack(reads, dim=1)
    assert torch.allclose(output.final_state, state, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(output.read, expected_read, atol=1.0e-6, rtol=1.0e-6)

    output.read.square().sum().backward()
    for name in (
        "q",
        "k",
        "v",
        "decay",
        "beta_e",
        "beta_w",
        "out_mix",
        "read_gate",
    ):
        gradient = getattr(factors, name).grad
        assert gradient is not None and torch.isfinite(gradient).all(), name


def test_true_mimo_common_slot_permutation_is_forward_invariant() -> None:
    from research.kmd2_ablation.tiny_backend import TinyFactors, TinyKMD2Cell

    rank = 3
    factors = _true_mimo_factors(rank=rank)
    permutation = torch.tensor([2, 0, 1])
    permuted = TinyFactors(
        q=factors.q.index_select(3, permutation),
        k=factors.k.index_select(3, permutation),
        v=factors.v.index_select(3, permutation),
        decay=factors.decay,
        beta_e=factors.beta_e.index_select(3, permutation),
        beta_w=factors.beta_w.index_select(3, permutation),
        out_mix=factors.out_mix.index_select(3, permutation),
        valid=factors.valid,
        positions=factors.positions,
        read_gate=factors.read_gate.index_select(3, permutation),
    )
    cell = TinyKMD2Cell(_tiny_config(mimo_rank=rank))
    original = cell(factors)
    reordered = cell(permuted)
    assert torch.allclose(reordered.final_state, original.final_state, atol=1.0e-6)
    assert torch.allclose(reordered.read, original.read, atol=1.0e-6)
    assert torch.allclose(reordered.scores, original.scores, atol=1.0e-6)


def test_true_mimo_uses_mamba3_channelwise_rank_scalings_and_gates() -> None:
    from research.kmd2_ablation.tiny_backend import TinyFactorProjector
    from research.kmd2_ablation.variants import get_variant

    rank = 3
    projector = TinyFactorProjector(_tiny_config(mimo_rank=rank))
    hidden = torch.randn(1, 2, 8, generator=torch.Generator().manual_seed(51))
    factors = projector(
        hidden,
        torch.ones(1, 2, dtype=torch.bool),
        torch.arange(2, dtype=torch.int64).view(1, 2),
    )
    assert factors.out_mix.shape == (1, 2, 1, rank, 2)
    assert factors.read_gate.shape == (1, 2, 1, rank, 2)
    assert torch.equal(projector.mimo_v, torch.full_like(projector.mimo_v, 1 / rank))
    assert torch.equal(projector.mimo_z, torch.ones_like(projector.mimo_z))
    assert torch.equal(
        projector.mimo_out, torch.full_like(projector.mimo_out, 1 / rank)
    )
    assert not hasattr(projector, "q_slot_scale")
    assert not hasattr(projector, "out_mix")
    assert factors.k.shape[3] == factors.q.shape[3] == rank
    assert get_variant("true_mimo.sweep").compatible_backends == frozenset({"tiny"})
    with pytest.raises(ValueError, match="r_out.*mimo_rank"):
        _tiny_config(r_out=4, mimo_rank=rank)


def test_moving_frame_transport_equivalence_requires_pair_tied_decay() -> None:
    from research.kmd2_ablation.tiny_backend import (
        moving_frame_transport_diagnostic,
    )

    generator = torch.Generator().manual_seed(601)
    state = torch.randn(2, 1, 4, 3, generator=generator, requires_grad=True)
    previous_phase = torch.randn(2, 1, 2, generator=generator)
    current_phase = torch.randn(2, 1, 2, generator=generator)
    tied_decay = torch.tensor([0.8, 0.6, 0.8, 0.6]).view(1, 1, 4).expand(2, -1, -1)
    exact, moving = moving_frame_transport_diagnostic(
        state, tied_decay, previous_phase, current_phase
    )
    assert torch.allclose(moving, exact, atol=1.0e-6, rtol=1.0e-6)
    gradients = torch.autograd.grad((moving.square().sum() + exact.square().sum()), state)
    assert torch.isfinite(gradients[0]).all()

    independent_decay = torch.tensor([0.8, 0.6, 0.5, 0.9]).view(1, 1, 4)
    exact_independent, moving_independent = moving_frame_transport_diagnostic(
        state.detach()[:1], independent_decay, previous_phase[:1], current_phase[:1]
    )
    assert not torch.allclose(
        moving_independent, exact_independent, atol=1.0e-6, rtol=1.0e-6
    )


def test_moving_frame_recurrence_matches_current_only_for_pair_tied_decay() -> None:
    from research.kmd2_ablation.tiny_backend import TinyFactorProjector, TinyKMD2Cell

    current_config = _tiny_config(
        dk=4,
        rotation_mode="current",
        rotation_gate_init=1.0,
        channel_decay_gate_init=1.0,
    )
    moving_config = _tiny_config(
        dk=4,
        rotation_mode="moving_frame",
        rotation_gate_init=1.0,
        channel_decay_gate_init=1.0,
    )
    torch.manual_seed(1602)
    current_projector = TinyFactorProjector(current_config)
    moving_projector = TinyFactorProjector(moving_config)
    moving_projector.load_state_dict(current_projector.state_dict(), strict=True)
    hidden = torch.randn(1, 5, 8, generator=torch.Generator().manual_seed(1603))
    valid = torch.ones(1, 5, dtype=torch.bool)
    positions = torch.arange(5, dtype=torch.int64).view(1, 5)
    boundaries = torch.tensor([[True, False, False, False, False]])

    pair_tied = torch.tensor([[0.15, -0.2, 0.15, -0.2]])
    with torch.no_grad():
        current_projector.decay_chan.copy_(pair_tied)
        moving_projector.decay_chan.copy_(pair_tied)
    current_factors = current_projector(hidden, valid, positions)
    moving_factors = moving_projector(hidden, valid, positions)
    assert moving_factors.moving_frame_phase is not None
    fixed = TinyKMD2Cell(current_config)(current_factors, boundaries=boundaries)
    moving = TinyKMD2Cell(moving_config)(moving_factors, boundaries=boundaries)
    assert torch.allclose(moving.read, fixed.read, atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(moving.scores, fixed.scores, atol=2.0e-5, rtol=2.0e-5)

    independent = torch.tensor([[0.15, -0.2, 0.35, -0.45]])
    with torch.no_grad():
        current_projector.decay_chan.copy_(independent)
        moving_projector.decay_chan.copy_(independent)
    fixed_independent = TinyKMD2Cell(current_config)(
        current_projector(hidden, valid, positions), boundaries=boundaries
    )
    moving_independent = TinyKMD2Cell(moving_config)(
        moving_projector(hidden, valid, positions), boundaries=boundaries
    )
    assert not torch.allclose(
        moving_independent.read,
        fixed_independent.read,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


@pytest.mark.parametrize(
    ("comparison", "arm_overrides"),
    [
        ("state_size", {"dk": 4, "dv": 3}),
        ("mimo_rank", {"mimo_rank": 2}),
        ("factorial", {"dk": 4, "dv": 3, "mimo_rank": 2}),
    ],
)
def test_parameter_match_uses_exact_instantiated_counts_and_state_bytes(
    comparison: str, arm_overrides: dict[str, object]
) -> None:
    from dataclasses import replace

    from research.kmd2_ablation.tiny_backend import TinyKMD2Model
    from research.kmd2_ablation.variants import match_tiny_parameter_count

    target = _tiny_config(d_ff=32)
    arm = _tiny_config(d_ff=32, **arm_overrides)
    result = match_tiny_parameter_count(
        target,
        arm,
        comparison=comparison,
        d_ff_match_min=8,
        d_ff_match_max=128,
    )

    def trainable(config) -> int:
        return sum(
            parameter.numel()
            for parameter in TinyKMD2Model(config, init_seed=0).parameters()
            if parameter.requires_grad
        )

    target_count = trainable(target)
    candidates = tuple(range(8, 129, 8))
    expected_d_ff = min(
        candidates,
        key=lambda d_ff: (abs(trainable(replace(arm, d_ff=d_ff)) - target_count), d_ff),
    )
    expected_config = replace(arm, d_ff=expected_d_ff)
    assert result.comparison == comparison
    assert result.bounds == (8, 128)
    assert result.target.trainable_parameters == target_count
    assert result.raw.d_ff == target.d_ff == arm.d_ff
    assert result.raw.trainable_parameters == trainable(arm)
    assert result.matched.d_ff == expected_d_ff
    assert result.matched.trainable_parameters == trainable(expected_config)
    assert result.matched.config == expected_config
    assert result.target.recurrent_state_elements == (
        target.layers * target.heads * target.dk * target.dv
    )
    assert result.raw.recurrent_state_elements == (
        arm.layers * arm.heads * arm.dk * arm.dv
    )
    assert result.raw.recurrent_state_bytes == 4 * result.raw.recurrent_state_elements
    assert result.matched.recurrent_state_bytes == result.raw.recurrent_state_bytes
    assert result.residual_mismatch == (
        result.matched.trainable_parameters - result.target.trainable_parameters
    )
    assert result.tolerance == max(0.005 * target_count, 1024.0)
    assert abs(result.residual_mismatch) <= result.tolerance


def test_parameter_match_rejects_when_finite_divisible_bounds_have_no_legal_match() -> None:
    from research.kmd2_ablation.variants import match_tiny_parameter_count

    target = _tiny_config(d_ff=8)
    oversized = _tiny_config(d_ff=8, dk=64, dv=64, mimo_rank=8)
    with pytest.raises(ValueError, match="no legal parameter match"):
        match_tiny_parameter_count(
            target,
            oversized,
            comparison="factorial",
            d_ff_match_min=8,
            d_ff_match_max=8,
        )


@pytest.mark.parametrize("width", (0, 8, 16, 32, 64, 128))
def test_equal_state_bytes_control_instantiates_closest_recurrent_increase(
    width: int,
) -> None:
    from research.kmd2_ablation.tiny_backend import TinyKMD2Model
    from research.kmd2_ablation.variants import construct_equal_state_byte_control

    base = _tiny_config(heads=2, dk=4, dv=3, layers=2, d_ff=32)
    result = construct_equal_state_byte_control(
        base,
        cache_width=width,
        storage_dtype="fp32",
    )
    cache_slot_bytes = (base.dk + base.dv) * 4 + 4 + 8 + 1
    expected_cache_bytes = base.layers * base.heads * width * cache_slot_bytes
    row_bytes = base.layers * base.heads * base.dv * 4
    minimum_rows = 0 if width == 0 else 1
    expected_rows = min(
        range(minimum_rows, max(minimum_rows + 1, expected_cache_bytes // row_bytes + 3)),
        key=lambda rows: (abs(rows * row_bytes - expected_cache_bytes), rows),
    )

    assert result.cache_width == width
    assert result.storage_dtype == "fp32"
    assert result.cache_persistent_bytes == expected_cache_bytes
    assert result.base.config == base
    assert result.control.config.dk == base.dk + expected_rows
    assert result.control.config.dv == base.dv
    assert result.control.config.d_ff == base.d_ff
    assert result.base.recurrent_state_bytes == base.layers * base.heads * base.dk * base.dv * 4
    assert result.recurrent_increase_bytes == expected_rows * row_bytes
    assert result.control.recurrent_state_bytes == (
        result.base.recurrent_state_bytes + result.recurrent_increase_bytes
    )
    assert result.byte_mismatch == result.recurrent_increase_bytes - expected_cache_bytes
    assert result.absolute_byte_mismatch == abs(result.byte_mismatch)
    instantiated = TinyKMD2Model(result.control.config, init_seed=0)
    assert sum(
        parameter.numel()
        for parameter in instantiated.parameters()
        if parameter.requires_grad
    ) == result.control.trainable_parameters


def _cache_stage_controls():
    from research.kmd2_ablation.variants import CacheControlProfile

    return CacheControlProfile(
        cache_width=64,
        cache_block_size=128,
        read_arm="exact_cache.read.unit_l2",
        gate_initialization="gamma_one_sink_zero_amplitude_zero",
        token_budget=4096,
        update_budget=32,
    )


def _passing_cache_stage_evidence():
    from research.kmd2_ablation.variants import CacheStageEvidence

    return CacheStageEvidence(
        selector_primary_lcb=0.05,
        read_primary_lcb=0.05,
        tiny_primary_lcb=0.05,
        short_accuracy_vs_native_lcb=-0.02,
        freshness_latest_vs_native_lcb=-0.02,
        freshness_stale_vs_native_lcb=-0.02,
        freshness_latest_vs_recency_lcb=-0.02,
        freshness_stale_vs_recency_lcb=-0.02,
        interactions_complete=True,
    )


def _expand_cache_stage_fixture(*, evidence=None, **overrides):
    from research.kmd2_ablation.variants import expand_exact_cache_stages

    arguments = {
        "canonical_config": {
            "schema_version": "test-v1",
            "experiment": "serial-cache-screen",
        },
        "screen_seeds": (11, 22, 33),
        "promotion_seeds": (11, 22, 33, 44, 55),
        "heal_seeds": (101, 202, 303),
        "winner_selector_arm": "exact_cache.selector.exact_outer",
        "winner_read_arm": "exact_cache.read.unit_l2",
        "winner_width": 64,
        "winner_block_size": 128,
        "recency_controls": _cache_stage_controls(),
        "surprise_controls": _cache_stage_controls(),
        "evidence": evidence or _passing_cache_stage_evidence(),
        "ruler_episodes_per_cell": 64,
    }
    arguments.update(overrides)
    return expand_exact_cache_stages(**arguments)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("cache_width", 32),
        ("cache_block_size", 64),
        ("read_arm", "exact_cache.read.rmsnorm"),
        ("gate_initialization", "different_gate"),
        ("token_budget", 2048),
        ("update_budget", 16),
    ],
)
def test_cache_compat_requires_exact_capacity_read_gate_and_budget_controls(
    field: str, replacement: object
) -> None:
    from dataclasses import replace

    from research.kmd2_ablation.variants import validate_matched_cache_controls

    reference = _cache_stage_controls()
    assert validate_matched_cache_controls(reference, reference) == reference
    with pytest.raises(ValueError, match=field):
        validate_matched_cache_controls(
            reference,
            replace(reference, **{field: replacement}),
        )


def test_cache_compat_enforces_geometry_claim_and_noop_gates() -> None:
    from research.kmd2_ablation.variants import validate_cache_compatibility

    common = {
        "block_size": 64,
        "max_sequence_length": 128,
        "claimed_evidence_kind": "addition",
        "disabled_identity": True,
        "active_output_changed": True,
        "native_feature_present": False,
    }
    selected = validate_cache_compatibility(
        "exact_cache.selector.exact_outer",
        width=32,
        **common,
    )
    assert selected.arm_id == "exact_cache.selector.exact_outer"
    chunk_only = validate_cache_compatibility(
        "exact_cache.width.0",
        width=0,
        **{**common, "claimed_evidence_kind": "diagnostic"},
    )
    assert chunk_only.mechanism == "current_block_only"

    invalid_requests = (
        (
            "exact_cache.selector.exact_outer",
            {"width": 0},
            "width=0.*chunk-only",
        ),
        (
            "exact_cache.width.0",
            {"width": 8, "claimed_evidence_kind": "diagnostic"},
            "chunk-only.*width=0",
        ),
        (
            "exact_cache.selector.exact_outer",
            {"width": 32, "max_sequence_length": 127},
            "two blocks",
        ),
        (
            "exact_cache.selector.exact_outer",
            {"width": 128},
            "eviction",
        ),
        (
            "exact_cache.selector.future_query_oracle",
            {"width": 32},
            "diagnostic",
        ),
        (
            "exact_cache.pre_rotation_diagnostic",
            {"width": 32},
            "diagnostic",
        ),
        (
            "exact_cache.selector.exact_outer",
            {"width": 32, "claimed_evidence_kind": "reliance"},
            "post-hoc reliance",
        ),
        (
            "exact_cache.selector.exact_outer",
            {"width": 32, "disabled_identity": False},
            "disabled identity",
        ),
        (
            "exact_cache.selector.exact_outer",
            {"width": 32, "active_output_changed": False},
            "active effect",
        ),
        (
            "exact_cache.selector.exact_outer",
            {"width": 32, "native_feature_present": True},
            "native-present",
        ),
    )
    for arm_id, changes, message in invalid_requests:
        request = {**common, **changes}
        with pytest.raises(ValueError, match=message):
            validate_cache_compatibility(arm_id, **request)


def test_stage_expansion_has_exact_serial_counts_pairings_and_job_ids() -> None:
    import hashlib

    from research.kmd2_ablation.results import canonical_json_bytes, semantic_job_id

    batches = _expand_cache_stage_fixture()
    assert tuple(batch.lane for batch in batches) == (
        "selector_replay",
        "read_screen",
        "capacity_width",
        "capacity_block",
        "tiny_promotion",
        "native_interaction",
        "qwen_heal",
    )
    assert tuple(len(batch.jobs) for batch in batches) == (21, 9, 18, 9, 60, 40, 9)
    all_jobs = tuple(job for batch in batches for job in batch.jobs)
    assert len(all_jobs) == 166
    assert len({job.job_id for job in all_jobs}) == len(all_jobs)

    selectors = batches[0].jobs
    assert tuple(job.arm_id for job in selectors[:7]) == (
        "exact_cache.selector.exact_outer",
        "exact_cache.selector.coupled_paper",
        "exact_cache.selector.residual_only",
        "exact_cache.selector.write_value",
        "exact_cache.selector.recency",
        "exact_cache.selector.reservoir",
        "exact_cache.selector.future_query_oracle",
    )
    assert {job.seed for job in selectors} == {11, 22, 33}
    assert len({job.pairing_id for job in selectors if job.seed == 11}) == 1
    canonical = {
        "schema_version": "test-v1",
        "experiment": "serial-cache-screen",
    }
    expected_pairing = hashlib.sha256(
        canonical_json_bytes(
            {
                "canonical_config": canonical,
                "lane": "selector_replay",
                "stage": "selector_replay",
                "backend": "tiny",
                "seed": 11,
                "task": "far_surprise",
                "comparison_semantics": {
                    "comparison_key": "selectors",
                    "selector_arm": None,
                    "read_arm": None,
                    "cache_width": 64,
                    "cache_block_size": 128,
                    "controls": None,
                    "ruler_episodes_per_cell": None,
                },
            }
        )
    ).hexdigest()
    assert selectors[0].pairing_id == expected_pairing
    assert selectors[0].job_id == semantic_job_id(
        canonical,
        backend="tiny",
        arm_id="exact_cache.selector.exact_outer",
        seed=11,
        stage="selector_replay",
        pairing_id=expected_pairing,
    )

    reads = batches[1].jobs
    assert {job.arm_id for job in reads} == {
        "exact_cache.read.unit_l2",
        "exact_cache.read.fixed_temperature",
        "exact_cache.read.rmsnorm",
    }
    assert {job.selector_arm for job in reads} == {
        "exact_cache.selector.exact_outer"
    }

    widths = batches[2].jobs
    assert {job.cache_width for job in widths} == {0, 8, 16, 32, 64, 128}
    assert {job.cache_block_size for job in widths} == {128}
    blocks = batches[3].jobs
    assert {job.cache_width for job in blocks} == {64}
    assert {job.cache_block_size for job in blocks} == {64, 128, 256}
    assert len({(job.cache_width, job.cache_block_size) for job in blocks}) == 3

    promotion = batches[4].jobs
    assert {job.arm_id for job in promotion} == {
        "native",
        "exact_cache.selector.recency",
        "exact_cache.selector.exact_outer",
    }
    assert {job.task for job in promotion} == {
        "structured_exceptions",
        "mqar",
        "far_surprise",
        "freshness",
    }
    assert {job.seed for job in promotion} == {11, 22, 33, 44, 55}
    assert {job.controls for job in promotion} == {_cache_stage_controls()}

    interactions = batches[5].jobs
    assert {job.cell for job in interactions} == {"M00", "M10", "M01", "M11"}
    assert {job.declared_arm_id for job in interactions} == {
        "exact_cache.rotation_factorial",
        "exact_cache.r_out_factorial",
    }
    assert {job.arm_id for job in interactions} == {
        f"{arm}.{cell}"
        for arm in (
            "exact_cache.rotation_factorial",
            "exact_cache.r_out_factorial",
        )
        for cell in ("M00", "M10", "M01", "M11")
    }
    for declared_arm in {
        "exact_cache.rotation_factorial",
        "exact_cache.r_out_factorial",
    }:
        paired = [
            job
            for job in interactions
            if job.declared_arm_id == declared_arm and job.seed == 11
        ]
        assert len(paired) == 4
        assert len({job.pairing_id for job in paired}) == 1

    qwen = batches[6].jobs
    assert {job.backend for job in qwen} == {"qwen"}
    assert {job.arm_id for job in qwen} == {
        "native",
        "exact_cache.selector.recency",
        "exact_cache.selector.exact_outer",
    }
    assert {job.seed for job in qwen} == {101, 202, 303}
    assert {job.ruler_episodes_per_cell for job in qwen} == {64}
    assert tuple(
        tuple(job.job_id for job in batch)
        for batch in _expand_cache_stage_fixture()
    ) == tuple(tuple(job.job_id for job in batch) for batch in batches)


@pytest.mark.parametrize(
    ("evidence_overrides", "argument_overrides", "expected_lanes"),
    [
        (
            {"selector_primary_lcb": 0.049},
            {},
            ("selector_replay",),
        ),
        (
            {"read_primary_lcb": 0.049},
            {},
            ("selector_replay", "read_screen"),
        ),
        (
            {},
            {"winner_width": None},
            ("selector_replay", "read_screen", "capacity_width"),
        ),
        (
            {},
            {"winner_block_size": None},
            (
                "selector_replay",
                "read_screen",
                "capacity_width",
                "capacity_block",
            ),
        ),
        (
            {"tiny_primary_lcb": 0.049},
            {},
            (
                "selector_replay",
                "read_screen",
                "capacity_width",
                "capacity_block",
                "tiny_promotion",
            ),
        ),
        (
            {"short_accuracy_vs_native_lcb": -0.021},
            {},
            (
                "selector_replay",
                "read_screen",
                "capacity_width",
                "capacity_block",
                "tiny_promotion",
            ),
        ),
        (
            {"freshness_latest_vs_native_lcb": -0.021},
            {},
            (
                "selector_replay",
                "read_screen",
                "capacity_width",
                "capacity_block",
                "tiny_promotion",
            ),
        ),
        (
            {"freshness_stale_vs_recency_lcb": -0.021},
            {},
            (
                "selector_replay",
                "read_screen",
                "capacity_width",
                "capacity_block",
                "tiny_promotion",
            ),
        ),
        (
            {"interactions_complete": False},
            {},
            (
                "selector_replay",
                "read_screen",
                "capacity_width",
                "capacity_block",
                "tiny_promotion",
                "native_interaction",
            ),
        ),
    ],
)
def test_stage_expansion_failed_gate_emits_no_downstream_cartesian_jobs(
    evidence_overrides: dict[str, object],
    argument_overrides: dict[str, object],
    expected_lanes: tuple[str, ...],
) -> None:
    from dataclasses import replace

    evidence = replace(_passing_cache_stage_evidence(), **evidence_overrides)
    batches = _expand_cache_stage_fixture(evidence=evidence, **argument_overrides)
    assert tuple(batch.lane for batch in batches) == expected_lanes


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"screen_seeds": (11, 22)}, "screen_seeds.*exactly 3"),
        ({"promotion_seeds": (11, 22, 33, 44)}, "promotion_seeds.*exactly 5"),
        ({"heal_seeds": (101, 202)}, "heal_seeds.*exactly 3"),
        ({"ruler_episodes_per_cell": 63}, "at least 64"),
    ],
)
def test_stage_expansion_pins_seed_counts_and_matched_ruler_episodes(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _expand_cache_stage_fixture(**overrides)


def test_stage_expansion_ids_include_fixed_selector_geometry() -> None:
    first = _expand_cache_stage_fixture(fixed_cache_width=64, fixed_block_size=128)
    second = _expand_cache_stage_fixture(fixed_cache_width=128, fixed_block_size=256)
    first_selector = first[0].jobs[0]
    second_selector = second[0].jobs[0]

    assert first_selector.pairing_id != second_selector.pairing_id
    assert first_selector.job_id != second_selector.job_id
    assert len({job.pairing_id for job in first[0].jobs if job.seed == 11}) == 1
    assert len({job.pairing_id for job in second[0].jobs if job.seed == 11}) == 1


@pytest.mark.parametrize(
    ("budget_field", "changed_value"),
    (("token_budget", 8192), ("update_budget", 64)),
)
def test_stage_expansion_ids_include_promotion_and_qwen_budgets(
    budget_field: str, changed_value: int
) -> None:
    from dataclasses import replace

    baseline = _expand_cache_stage_fixture()
    changed_controls = replace(
        _cache_stage_controls(), **{budget_field: changed_value}
    )
    changed = _expand_cache_stage_fixture(
        recency_controls=changed_controls,
        surprise_controls=changed_controls,
    )
    for lane_index in (4, 6):
        baseline_job = baseline[lane_index].jobs[0]
        changed_job = changed[lane_index].jobs[0]
        assert baseline_job.pairing_id != changed_job.pairing_id
        assert baseline_job.job_id != changed_job.job_id

    changed_promotion_pair = [
        job
        for job in changed[4].jobs
        if job.task == "structured_exceptions" and job.seed == 11
    ]
    changed_qwen_pair = [job for job in changed[6].jobs if job.seed == 101]
    assert len(changed_promotion_pair) == len(changed_qwen_pair) == 3
    assert len({job.pairing_id for job in changed_promotion_pair}) == 1
    assert len({job.pairing_id for job in changed_qwen_pair}) == 1


def test_stage_expansion_jobs_have_executable_registry_compatibility() -> None:
    from research.kmd2_ablation.variants import (
        get_variant,
        validate_variant_compatibility,
    )

    for batch in _expand_cache_stage_fixture():
        for job in batch.jobs:
            spec = get_variant(job.arm_id)
            assert job.stage in spec.compatible_stages
            accepted = validate_variant_compatibility(
                job.arm_id,
                backend=job.backend,
                task=job.task,
                stage=job.stage,
                experiment_kind=spec.experiment_kind,
            )
            assert accepted is spec


def test_registry_keeps_mechanism_variant_resolution_unambiguous() -> None:
    from research.kmd2_ablation.variants import all_variants

    identities = tuple((spec.mechanism, spec.variant) for spec in all_variants())
    assert len(identities) == len(set(identities))


def test_registry_parameter_metadata_matches_instantiated_tiny_tensors() -> None:
    from research.kmd2_ablation.tiny_backend import TinyKMD2Model
    from research.kmd2_ablation.variants import get_variant

    native = TinyKMD2Model(_tiny_config(rotation_mode="current"), init_seed=0)
    state_size = TinyKMD2Model(
        _tiny_config(rotation_mode="current", dk=4, dv=3), init_seed=0
    )
    true_mimo = TinyKMD2Model(
        _tiny_config(rotation_mode="current", mimo_rank=2), init_seed=0
    )
    diagonal = TinyKMD2Model(
        _tiny_config(rotation_mode="current", bc_bias_mode="diagonal_rescale"),
        init_seed=0,
    )

    expected_state_size = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "z_proj.weight",
        "conv.weight",
        "decay_chan",
        "q_slot_scale",
        "rot_proj.weight",
        "rot_proj.bias",
        "rotation_rate",
        "out_proj.weight",
    )
    expected_true_mimo = (
        "q_proj.weight",
        "k_proj.weight",
        "b_proj.weight",
        "conv.weight",
        "bw_off",
        "mimo_v",
        "mimo_z",
        "mimo_out",
    )
    expected_diagonal = (
        "bc_q_amplitude",
        "bc_k_amplitude",
        "bc_q_scale",
        "bc_k_scale",
    )
    assert get_variant("state_size.sweep").changed_parameters == expected_state_size
    assert get_variant("true_mimo.sweep").changed_parameters == expected_true_mimo
    assert get_variant("bc_bias.diagonal_rescale").changed_parameters == expected_diagonal
    assert {"mimo_v", "mimo_z", "mimo_out"}.issubset(expected_true_mimo)

    def shapes(model: TinyKMD2Model) -> dict[str, tuple[int, ...]]:
        return {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}

    def local_name(name: str) -> str:
        return (
            name.removeprefix("blocks.0.")
            .removeprefix("projector.")
            .removeprefix("cell.")
        )

    native_shapes = shapes(native)
    for model, expected in (
        (state_size, expected_state_size),
        (true_mimo, expected_true_mimo),
    ):
        variant_shapes = shapes(model)
        changed = {
            local_name(name)
            for name in native_shapes.keys() | variant_shapes.keys()
            if native_shapes.get(name) != variant_shapes.get(name)
            and name in variant_shapes
        }
        assert changed == set(expected)
        assert all(
            any(name.endswith(metadata_name) for name in variant_shapes)
            for metadata_name in expected
        )
    diagonal_names = tuple(name for name, _ in diagonal.named_parameters())
    assert all(
        any(name.endswith(metadata_name) for name in diagonal_names)
        for metadata_name in expected_diagonal
    )


@pytest.mark.parametrize(
    ("arm_id", "width", "block_size", "message"),
    [
        ("exact_cache.width.8", 16, 64, "declares width=8"),
        ("exact_cache.block.64", 32, 128, "declares block_size=64"),
    ],
)
def test_cache_compat_rejects_runtime_geometry_that_conflicts_with_arm_label(
    arm_id: str, width: int, block_size: int, message: str
) -> None:
    from research.kmd2_ablation.variants import validate_cache_compatibility

    with pytest.raises(ValueError, match=message):
        validate_cache_compatibility(
            arm_id,
            width=width,
            block_size=block_size,
            max_sequence_length=512,
            claimed_evidence_kind="addition",
            disabled_identity=True,
            active_output_changed=True,
            native_feature_present=False,
        )


def test_true_mimo_zero_erase_gate_has_finite_correct_boundary_gradient() -> None:
    from research.kmd2_ablation.tiny_backend import true_mimo_update

    generator = torch.Generator().manual_seed(7781)
    state = torch.randn(2, 2, 4, 3, generator=generator, requires_grad=True)
    key = torch.randn(2, 2, 3, 4, generator=generator, requires_grad=True)
    value = torch.randn(2, 2, 3, 3, generator=generator, requires_grad=True)
    beta_e = torch.rand(2, 2, 3, generator=generator)
    beta_e[..., 0] = 0.0
    beta_e.requires_grad_()
    beta_w = torch.rand(2, 2, 3, generator=generator, requires_grad=True)
    operands = (state, key, value, beta_e, beta_w)
    reference_operands = tuple(
        operand.detach().clone().requires_grad_() for operand in operands
    )

    actual = true_mimo_update(*operands)
    ref_state, ref_key, ref_value, ref_beta_e, ref_beta_w = reference_operands
    rank = ref_key.shape[2]
    memory = torch.matmul(ref_key, ref_state)
    erase = torch.einsum(
        "bhrd,bhrv->bhdv",
        ref_key,
        (ref_beta_e / rank).unsqueeze(-1) * memory,
    )
    write = torch.einsum(
        "bhrd,bhrv->bhdv",
        ref_key,
        ref_beta_w.unsqueeze(-1) * ref_value,
    )
    expected = ref_state - erase + write
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()

    probe = torch.randn(actual.shape, generator=generator)
    actual_gradients = torch.autograd.grad((actual * probe).sum(), operands)
    expected_gradients = torch.autograd.grad(
        (expected * probe).sum(), reference_operands
    )
    assert expected_gradients[3][..., 0].abs().sum() > 0
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_trapezoid_state_bytes_include_previous_key_and_write_carries() -> None:
    from dataclasses import replace

    from research.kmd2_ablation.tiny_backend import TinyFactors, TinyKMD2Cell

    batch, steps, heads, dk, dv = 2, 3, 3, 4, 5
    generator = torch.Generator().manual_seed(7782)
    trapezoid_factors = TinyFactors(
        q=torch.randn(batch, steps, heads, 1, dk, generator=generator),
        k=torch.randn(batch, steps, heads, 1, dk, generator=generator),
        v=torch.randn(batch, steps, heads, 1, dv, generator=generator),
        decay=torch.rand(batch, steps, heads, dk, generator=generator),
        beta_e=torch.rand(batch, steps, heads, 1, generator=generator),
        beta_w=torch.rand(batch, steps, heads, 1, generator=generator),
        out_mix=torch.ones(batch, steps, heads, 1),
        valid=torch.ones(batch, steps, dtype=torch.bool),
        positions=torch.arange(steps, dtype=torch.int64).repeat(batch, 1),
        trapezoid_rho=torch.full((batch, steps, heads), 0.5),
    )
    native_factors = replace(trapezoid_factors, trapezoid_rho=None)
    native = TinyKMD2Cell(
        _tiny_config(heads=heads, dk=dk, dv=dv, rotation_mode="none")
    )(native_factors)
    trapezoid = TinyKMD2Cell(
        _tiny_config(
            heads=heads,
            dk=dk,
            dv=dv,
            rotation_mode="none",
            trapezoid=True,
        )
    )(trapezoid_factors)

    recurrent_bytes = batch * heads * dk * dv * 4
    trapezoid_carry_bytes = batch * heads * (dk + dv) * 4
    assert native.state_bytes == recurrent_bytes
    assert trapezoid.state_bytes == recurrent_bytes + trapezoid_carry_bytes
    assert trapezoid.state_bytes - native.state_bytes == trapezoid_carry_bytes


def test_tiny_config_preserves_legacy_positional_cache_field() -> None:
    from research.kmd2_ablation.config import CacheConfig
    from research.kmd2_ablation.tiny_backend import TinyKMD2Config

    cache = CacheConfig(width=8)
    config = TinyKMD2Config(
        8,
        1,
        2,
        2,
        1,
        11,
        16,
        1,
        1,
        None,
        None,
        3,
        torch.float32,
        1.0e-6,
        "none",
        0.0,
        0.0,
        0.0,
        0.0,
        False,
        0.0,
        cache,
    )
    assert config.cache is cache
    assert config.corrected_momentum is False
    assert config.momentum_gamma_init == 0.0
    assert config.causal_lookahead is False
    assert config.lookahead_rho_init == 0.0
    assert config.bc_bias_mode == "none"


def test_registry_builder_rejects_duplicate_mechanism_variant_identity() -> None:
    from dataclasses import replace

    from research.kmd2_ablation.variants import _build_registry, get_variant

    native = get_variant("native")
    duplicate_identity = replace(native, arm_id="native.alias")
    with pytest.raises(RuntimeError, match="duplicate mechanism/variant"):
        _build_registry((native, duplicate_identity))
