"""Typed registry for every preregistered KMD-2 ablation arm.

The registry is deliberately data-only.  Execution and staged expansion build on
these records without inferring scientific roles from names.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


EvidenceKind = Literal["baseline", "addition", "reliance", "diagnostic"]
ComparisonKind = Literal[
    "baseline", "incremental", "replacement", "reliance", "diagnostic", "factorial"
]
BackendName = Literal["tiny", "qwen"]
ExperimentKind = Literal[
    "baseline", "native_warm_start", "cold_redesign", "reliance", "diagnostic"
]


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """Immutable scientific and execution metadata for one declared arm."""

    arm_id: str
    mechanism: str
    variant: str
    evidence_kind: EvidenceKind
    comparison: ComparisonKind
    experiment_kind: ExperimentKind
    native_warm_start: bool
    compatible_backends: frozenset[BackendName]
    compatible_tasks: frozenset[str]
    changed_parameters: tuple[str, ...]
    changed_state: tuple[str, ...]
    required_stage: str

    def __post_init__(self) -> None:
        for name in ("arm_id", "mechanism", "variant", "required_stage"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty str")
        if self.evidence_kind not in {
            "baseline",
            "addition",
            "reliance",
            "diagnostic",
        }:
            raise ValueError("invalid evidence_kind")
        if self.comparison not in {
            "baseline",
            "incremental",
            "replacement",
            "reliance",
            "diagnostic",
            "factorial",
        }:
            raise ValueError("invalid comparison")
        if self.experiment_kind not in {
            "baseline",
            "native_warm_start",
            "cold_redesign",
            "reliance",
            "diagnostic",
        }:
            raise ValueError("invalid experiment_kind")
        if type(self.native_warm_start) is not bool:
            raise TypeError("native_warm_start must be a bool")
        if self.native_warm_start is not (
            self.experiment_kind == "native_warm_start"
        ):
            raise ValueError(
                "native_warm_start must be true exactly for native_warm_start experiments"
            )
        if not self.compatible_backends or not self.compatible_backends <= {
            "tiny",
            "qwen",
        }:
            raise ValueError("compatible_backends must contain tiny and/or qwen")
        if not self.compatible_tasks or any(
            type(task) is not str or not task for task in self.compatible_tasks
        ):
            raise ValueError("compatible_tasks must contain nonempty strings")
        for field in ("changed_parameters", "changed_state"):
            values = getattr(self, field)
            if any(type(value) is not str or not value for value in values):
                raise ValueError(f"{field} must contain nonempty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{field} must not contain duplicates")


_ALL_TASKS = frozenset(
    {
        "affine_associative_regression",
        "drift_reversal",
        "far_surprise",
        "freshness",
        "irregular_integration",
        "local_binding",
        "modular_counter",
        "mqar",
        "parity",
        "ruler",
        "state_tracking",
        "structured_exceptions",
        "toggle_fsm",
        "trajectory",
    }
)
_STATE_TASKS = frozenset({"state_tracking", "parity", "modular_counter", "toggle_fsm"})
_CACHE_TASKS = frozenset(
    {"structured_exceptions", "mqar", "far_surprise", "freshness", "ruler"}
)
_TINY = frozenset({"tiny"})
_BOTH = frozenset({"tiny", "qwen"})


def _spec(
    arm_id: str,
    mechanism: str,
    variant: str,
    evidence_kind: EvidenceKind,
    comparison: ComparisonKind,
    backends: frozenset[BackendName],
    tasks: frozenset[str],
    *,
    parameters: tuple[str, ...] = (),
    state: tuple[str, ...] = (),
    stage: str,
    experiment_kind: ExperimentKind | None = None,
    native_warm_start: bool | None = None,
) -> VariantSpec:
    if experiment_kind is None:
        experiment_kind = {
            "baseline": "baseline",
            "addition": "native_warm_start",
            "reliance": "reliance",
            "diagnostic": "diagnostic",
        }[evidence_kind]
    if native_warm_start is None:
        native_warm_start = experiment_kind == "native_warm_start"
    return VariantSpec(
        arm_id=arm_id,
        mechanism=mechanism,
        variant=variant,
        evidence_kind=evidence_kind,
        comparison=comparison,
        experiment_kind=experiment_kind,
        native_warm_start=native_warm_start,
        compatible_backends=backends,
        compatible_tasks=tasks,
        changed_parameters=parameters,
        changed_state=state,
        required_stage=stage,
    )


_RECORDS = (
    _spec("native", "native", "native", "baseline", "baseline", _BOTH, _ALL_TASKS, stage="local_correctness"),
    _spec("rotation.current", "rotation", "current_rotation", "reliance", "reliance", _BOTH, _STATE_TASKS, parameters=("rot_proj.weight", "rot_proj.bias"), state=("phase",), stage="qwen_reliance"),
    _spec("rotation.off", "rotation", "rotation_off", "reliance", "reliance", _BOTH, _STATE_TASKS, parameters=("rotation_gate",), state=("phase",), stage="qwen_reliance"),
    _spec("rotation.constant_rate", "rotation", "constant_rate_rotation", "diagnostic", "diagnostic", _TINY, _STATE_TASKS, parameters=("rotation_rate",), state=("phase",), stage="mechanism_screen"),
    _spec("rotation.non_cumulative", "rotation", "non_cumulative_rotation", "diagnostic", "diagnostic", _TINY, _STATE_TASKS, parameters=("rot_proj.weight", "rot_proj.bias"), stage="mechanism_screen"),
    _spec("rotation.fixed_rope", "rotation", "fixed_rope", "diagnostic", "diagnostic", _TINY, _STATE_TASKS, state=("fixed_phase",), stage="mechanism_screen"),
    _spec("rotation.moving_frame_oracle", "rotation", "moving_frame_oracle", "diagnostic", "diagnostic", _TINY, _STATE_TASKS, state=("moving_frame_state", "phase"), stage="mechanism_screen"),
    _spec("convolution.on", "convolution", "convolution_on", "reliance", "reliance", _BOTH, frozenset({"local_binding", "mqar"}), parameters=("conv1d.weight",), state=("conv_tail",), stage="qwen_reliance"),
    _spec("convolution.off", "convolution", "convolution_off", "reliance", "reliance", _BOTH, frozenset({"local_binding", "mqar"}), parameters=("convolution_gate",), state=("conv_tail",), stage="qwen_reliance"),
    _spec("trapezoid", "trapezoid", "trapezoid", "addition", "incremental", _BOTH, frozenset({"irregular_integration"}), parameters=("rho_head", "rho_proj.weight"), state=("k_prev", "u_prev"), stage="mechanism_screen"),
    _spec("bc_bias", "bc_bias", "bc_bias", "addition", "incremental", _BOTH, frozenset({"affine_associative_regression"}), parameters=("bc_q_amplitude", "bc_k_amplitude", "bc_q_bias", "bc_k_bias"), stage="mechanism_screen"),
    _spec("bc_bias.diagonal_rescale", "bc_bias", "diagonal_rescale", "diagnostic", "diagnostic", _TINY, frozenset({"affine_associative_regression"}), parameters=("bc_q_scale", "bc_k_scale"), stage="mechanism_screen"),
    _spec("bc_bias.constant_coordinate_oracle", "bc_bias", "constant_coordinate_oracle", "diagnostic", "diagnostic", _TINY, frozenset({"affine_associative_regression"}), state=("constant_coordinate",), stage="mechanism_screen"),
    _spec("corrected_momentum", "corrected_momentum", "corrected_momentum", "addition", "incremental", _BOTH, frozenset({"drift_reversal"}), parameters=("momentum_gamma",), state=("velocity",), stage="mechanism_screen"),
    _spec("causal_lookahead", "causal_lookahead", "causal_lookahead", "addition", "incremental", _BOTH, frozenset({"trajectory"}), parameters=("lookahead_rho", "lookahead_projection.weight"), state=("v_prev",), stage="mechanism_screen"),
    _spec("state_size.sweep", "state_size", "state_size_sweep", "addition", "incremental", _TINY, frozenset({"mqar"}), parameters=("q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight"), state=("state_shape",), stage="mechanism_screen", experiment_kind="cold_redesign", native_warm_start=False),
    _spec("true_mimo.sweep", "true_mimo", "true_mimo_sweep", "addition", "incremental", _TINY, frozenset({"mqar"}), parameters=("mimo_q_proj.weight", "mimo_k_proj.weight", "mimo_v_proj.weight", "mimo_out_mix"), state=("simultaneous_slots",), stage="mechanism_screen", experiment_kind="cold_redesign", native_warm_start=False),
    _spec("exact_cache.off", "exact_cache", "cache_off", "baseline", "baseline", _BOTH, _CACHE_TASKS, state=("cache_disabled",), stage="local_correctness"),
    _spec("exact_cache.current_block_only", "current_block_only", "chunk_only", "diagnostic", "diagnostic", _BOTH, _CACHE_TASKS, state=("current_block",), stage="capacity_screen"),
    _spec("exact_cache.selector.exact_outer", "exact_cache", "top_surprise", "addition", "incremental", _BOTH, _CACHE_TASKS, state=("persistent_cache", "selection_scores"), stage="selector_replay"),
    _spec("exact_cache.selector.coupled_paper", "exact_cache", "coupled_surprise", "addition", "incremental", _BOTH, _CACHE_TASKS, state=("persistent_cache", "selection_scores"), stage="selector_replay"),
    _spec("exact_cache.selector.residual_only", "exact_cache", "residual_only", "addition", "incremental", _BOTH, _CACHE_TASKS, state=("persistent_cache", "selection_scores"), stage="selector_replay"),
    _spec("exact_cache.selector.write_value", "exact_cache", "write_value_only", "addition", "incremental", _BOTH, _CACHE_TASKS, state=("persistent_cache", "selection_scores"), stage="selector_replay"),
    _spec("exact_cache.selector.recency", "exact_cache", "recency", "addition", "incremental", _BOTH, _CACHE_TASKS, state=("persistent_cache", "positions"), stage="selector_replay"),
    _spec("exact_cache.selector.reservoir", "exact_cache", "reservoir", "addition", "incremental", _BOTH, _CACHE_TASKS, state=("persistent_cache", "reservoir_rng"), stage="selector_replay"),
    _spec("exact_cache.selector.future_query_oracle", "exact_cache", "future_query_oracle", "diagnostic", "diagnostic", _TINY, _CACHE_TASKS, state=("oracle_persistent_cache",), stage="selector_replay"),
    _spec("exact_cache.read.unit_l2", "exact_cache", "unit_l2", "addition", "incremental", _BOTH, _CACHE_TASKS, parameters=("cache_amplitude", "cache_sink_logit"), state=("persistent_cache",), stage="read_screen"),
    _spec("exact_cache.read.fixed_temperature", "exact_cache", "fixed_temperature", "addition", "incremental", _BOTH, _CACHE_TASKS, parameters=("cache_amplitude", "cache_sink_logit"), state=("persistent_cache",), stage="read_screen"),
    _spec("exact_cache.read.rmsnorm", "exact_cache", "rmsnorm", "addition", "incremental", _BOTH, _CACHE_TASKS, parameters=("cache_gamma_q", "cache_gamma_k", "cache_sink_logit", "cache_amplitude"), state=("persistent_cache",), stage="read_screen"),
    _spec("exact_cache.storage.bf16", "exact_cache", "storage_bf16", "addition", "incremental", _BOTH, _CACHE_TASKS, state=("persistent_cache_bf16",), stage="capacity_screen"),
    _spec("exact_cache.storage.fp32", "exact_cache", "storage_fp32", "diagnostic", "diagnostic", _BOTH, _CACHE_TASKS, state=("persistent_cache_fp32",), stage="capacity_screen"),
    _spec("exact_cache.pre_rotation_diagnostic", "exact_cache", "pre_rotation", "diagnostic", "diagnostic", _TINY, _CACHE_TASKS, state=("pre_rotation_cache",), stage="tiny_promotion"),
    _spec("exact_cache.per_slot_read", "exact_cache", "per_slot_read", "diagnostic", "diagnostic", _TINY, _CACHE_TASKS, parameters=("per_slot_cache_mix",), state=("per_slot_cache_read",), stage="native_interaction"),
    _spec("exact_cache.unbounded_oracle", "exact_cache", "unbounded_oracle", "diagnostic", "diagnostic", _TINY, _CACHE_TASKS, state=("unbounded_exact_memory",), stage="tiny_promotion"),
    *(
        _spec(
            f"exact_cache.width.{width}",
            "exact_cache" if width else "current_block_only",
            f"width_{width}",
            "addition" if width else "diagnostic",
            "incremental" if width else "diagnostic",
            _BOTH,
            _CACHE_TASKS,
            state=(f"persistent_width_{width}",),
            stage="capacity_screen",
        )
        for width in (0, 8, 16, 32, 64, 128)
    ),
    *(
        _spec(
            f"exact_cache.block.{block_size}",
            "exact_cache",
            f"block_{block_size}",
            "addition",
            "incremental",
            _BOTH,
            _CACHE_TASKS,
            state=(f"block_size_{block_size}",),
            stage="capacity_screen",
        )
        for block_size in (64, 128, 256)
    ),
    _spec("exact_cache.rotation_factorial", "exact_cache", "cache_rotation_factorial", "addition", "factorial", _BOTH, _CACHE_TASKS, parameters=("cache_amplitude", "rot_proj.weight", "rot_proj.bias"), state=("persistent_cache", "phase"), stage="native_interaction"),
    _spec("exact_cache.r_out_factorial", "exact_cache", "cache_r_out_factorial", "addition", "factorial", _BOTH, _CACHE_TASKS, parameters=("cache_amplitude", "q_slot_scale", "out_mix"), state=("persistent_cache",), stage="native_interaction"),
)


def _build_registry(records: tuple[VariantSpec, ...]) -> Mapping[str, VariantSpec]:
    by_id: dict[str, VariantSpec] = {}
    for record in records:
        if record.arm_id in by_id:
            raise RuntimeError(f"duplicate variant arm_id: {record.arm_id}")
        by_id[record.arm_id] = record
    return MappingProxyType(dict(sorted(by_id.items())))


VARIANT_REGISTRY: Mapping[str, VariantSpec] = _build_registry(_RECORDS)


def all_variants() -> tuple[VariantSpec, ...]:
    """Return every declared arm in stable arm-id order."""

    return tuple(VARIANT_REGISTRY.values())


def get_variant(arm_id: str) -> VariantSpec:
    """Look up one arm without aliases or case folding."""

    if type(arm_id) is not str:
        raise TypeError("arm_id must be a str")
    try:
        return VARIANT_REGISTRY[arm_id]
    except KeyError:
        raise KeyError(f"unknown variant arm: {arm_id!r}") from None


lookup_variant = get_variant


def trapezoid_convolution_interaction_allowed(*, trapezoid_promoted: bool) -> bool:
    """Gate the replacement interaction on a completed individual screen."""

    if type(trapezoid_promoted) is not bool:
        raise TypeError("trapezoid_promoted must be a bool")
    return trapezoid_promoted


__all__ = [
    "VARIANT_REGISTRY",
    "VariantSpec",
    "all_variants",
    "get_variant",
    "lookup_variant",
    "trapezoid_convolution_interaction_allowed",
]
