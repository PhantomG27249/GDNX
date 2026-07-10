from __future__ import annotations

from dataclasses import FrozenInstanceError
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
    assert state_size.compatible_backends == frozenset({"tiny"})
    assert true_mimo.compatible_backends == frozenset({"tiny"})
    assert state_size.experiment_kind == true_mimo.experiment_kind == "cold_redesign"
    assert state_size.native_warm_start is true_mimo.native_warm_start is False
    assert get_variant("rotation.moving_frame_oracle").compatible_backends == frozenset(
        {"tiny"}
    )


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
    }
    values.update(overrides)
    return TinyKMD2Config(**values)


def _tiny_factors(
    *,
    steps: int = 4,
    positions: torch.Tensor | None = None,
    trapezoid_rho: torch.Tensor | None = None,
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
    if positions is None:
        positions = torch.arange(steps, dtype=torch.int64).view(1, steps)
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
