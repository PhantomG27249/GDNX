"""Typed registry for every preregistered KMD-2 ablation arm.

The registry is deliberately data-only.  Execution and staged expansion build on
these records without inferring scientific roles from names.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Any, Literal, Mapping


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
    compatible_stages: frozenset[str]

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
        if not self.compatible_stages or any(
            type(stage) is not str or not stage for stage in self.compatible_stages
        ):
            raise ValueError("compatible_stages must contain nonempty strings")
        if self.required_stage not in self.compatible_stages:
            raise ValueError("compatible_stages must include required_stage")


class VariantCompatibilityError(ValueError):
    """A declared arm cannot run under one or more requested conditions."""

    def __init__(self, arm_id: str, violations: tuple[str, ...]):
        self.arm_id = arm_id
        self.violations = violations
        joined = ", ".join(violations)
        super().__init__(f"variant {arm_id!r} is incompatible with: {joined}")


@dataclass(frozen=True, slots=True)
class TinyArmAccounting:
    """Exact instantiated parameter and recurrent-state accounting."""

    config: Any
    d_ff: int
    trainable_parameters: int
    recurrent_state_elements: int
    recurrent_state_bytes: int


@dataclass(frozen=True, slots=True)
class ParameterMatchResult:
    """Raw fixed-FFN and feed-forward-only matched arm accounting."""

    comparison: str
    bounds: tuple[int, int]
    target: TinyArmAccounting
    raw: TinyArmAccounting
    matched: TinyArmAccounting
    tolerance: float
    residual_mismatch: int


@dataclass(frozen=True, slots=True)
class EqualStateByteControl:
    """Closest legal recurrent-state increase for persistent cache bytes."""

    cache_width: int
    storage_dtype: str
    cache_persistent_bytes: int
    base: TinyArmAccounting
    control: TinyArmAccounting
    recurrent_increase_bytes: int
    byte_mismatch: int
    absolute_byte_mismatch: int


@dataclass(frozen=True, slots=True)
class CacheControlProfile:
    """Exact cache capacity/read/gate/budget controls for paired arms."""

    cache_width: int
    cache_block_size: int
    read_arm: str
    gate_initialization: str
    token_budget: int
    update_budget: int

    def __post_init__(self) -> None:
        for name, minimum in (
            ("cache_width", 0),
            ("cache_block_size", 1),
            ("token_budget", 1),
            ("update_budget", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an int")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        for name in ("read_arm", "gate_initialization"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty str")


@dataclass(frozen=True, slots=True)
class CacheStageEvidence:
    """Observed lower confidence bounds that unlock serial cache stages."""

    selector_primary_lcb: float | None = None
    read_primary_lcb: float | None = None
    tiny_primary_lcb: float | None = None
    short_accuracy_vs_native_lcb: float | None = None
    freshness_latest_vs_native_lcb: float | None = None
    freshness_stale_vs_native_lcb: float | None = None
    freshness_latest_vs_recency_lcb: float | None = None
    freshness_stale_vs_recency_lcb: float | None = None
    interactions_complete: bool = False

    def __post_init__(self) -> None:
        import math

        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "interactions_complete":
                if type(value) is not bool:
                    raise TypeError("interactions_complete must be a bool")
                continue
            if value is None:
                continue
            if type(value) not in {int, float} or type(value) is bool:
                raise TypeError(f"{field.name} must be a finite real or None")
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            object.__setattr__(self, field.name, float(value))


@dataclass(frozen=True, slots=True)
class CacheStageJob:
    """One immutable job emitted by the exact-cache serial stage graph."""

    job_id: str
    pairing_id: str
    lane: str
    stage: str
    backend: BackendName
    arm_id: str
    declared_arm_id: str
    seed: int
    task: str
    cell: str | None = None
    selector_arm: str | None = None
    read_arm: str | None = None
    cache_width: int | None = None
    cache_block_size: int | None = None
    controls: CacheControlProfile | None = None
    ruler_episodes_per_cell: int | None = None


@dataclass(frozen=True, slots=True)
class CacheStageBatch:
    """Ordered jobs for one serial lane."""

    lane: str
    jobs: tuple[CacheStageJob, ...]

    def __post_init__(self) -> None:
        if type(self.lane) is not str or not self.lane:
            raise ValueError("lane must be a nonempty str")
        object.__setattr__(self, "jobs", tuple(self.jobs))
        if not self.jobs:
            raise ValueError("stage batches must contain at least one job")
        if any(job.lane != self.lane for job in self.jobs):
            raise ValueError("every job in a stage batch must share its lane")

    def __iter__(self):
        return iter(self.jobs)

    def __len__(self) -> int:
        return len(self.jobs)


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
    additional_stages: frozenset[str] = frozenset(),
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
        compatible_stages=frozenset({stage}) | additional_stages,
    )


_RECORDS = (
    _spec("native", "native", "native", "baseline", "baseline", _BOTH, _ALL_TASKS, stage="local_correctness", additional_stages=frozenset({"tiny_promotion", "qwen_heal"})),
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
    _spec("bc_bias.diagonal_rescale", "bc_bias", "diagonal_rescale", "diagnostic", "diagnostic", _TINY, frozenset({"affine_associative_regression"}), parameters=("bc_q_amplitude", "bc_k_amplitude", "bc_q_scale", "bc_k_scale"), stage="mechanism_screen"),
    _spec("bc_bias.constant_coordinate_oracle", "bc_bias", "constant_coordinate_oracle", "diagnostic", "diagnostic", _TINY, frozenset({"affine_associative_regression"}), parameters=("q_proj.weight", "k_proj.weight", "conv.weight", "q_slot_scale", "decay_chan"), state=("constant_coordinate",), stage="mechanism_screen"),
    _spec("corrected_momentum", "corrected_momentum", "corrected_momentum", "addition", "incremental", _BOTH, frozenset({"drift_reversal"}), parameters=("momentum_gamma",), state=("velocity",), stage="mechanism_screen"),
    _spec("causal_lookahead", "causal_lookahead", "causal_lookahead", "addition", "incremental", _BOTH, frozenset({"trajectory"}), parameters=("lookahead_rho", "lookahead_projection.weight"), state=("v_prev",), stage="mechanism_screen"),
    _spec("state_size.sweep", "state_size", "state_size_sweep", "addition", "incremental", _TINY, frozenset({"mqar"}), parameters=("q_proj.weight", "k_proj.weight", "v_proj.weight", "z_proj.weight", "conv.weight", "decay_chan", "q_slot_scale", "rot_proj.weight", "rot_proj.bias", "rotation_rate", "out_proj.weight"), state=("state_shape",), stage="mechanism_screen", experiment_kind="cold_redesign", native_warm_start=False),
    _spec("true_mimo.sweep", "true_mimo", "true_mimo_sweep", "addition", "incremental", _TINY, frozenset({"mqar"}), parameters=("q_proj.weight", "k_proj.weight", "b_proj.weight", "conv.weight", "bw_off", "mimo_v", "mimo_z", "mimo_out"), state=("simultaneous_slots",), stage="mechanism_screen", experiment_kind="cold_redesign", native_warm_start=False),
    _spec("exact_cache.off", "exact_cache", "cache_off", "baseline", "baseline", _BOTH, _CACHE_TASKS, state=("cache_disabled",), stage="local_correctness"),
    _spec("exact_cache.current_block_only", "current_block_only", "chunk_only", "diagnostic", "diagnostic", _BOTH, _CACHE_TASKS, state=("current_block",), stage="capacity_screen"),
    _spec("exact_cache.selector.exact_outer", "exact_cache", "top_surprise", "addition", "incremental", _BOTH, _CACHE_TASKS, state=("persistent_cache", "selection_scores"), stage="selector_replay", additional_stages=frozenset({"tiny_promotion", "qwen_heal"})),
    _spec("exact_cache.selector.coupled_paper", "exact_cache", "coupled_surprise", "addition", "incremental", _BOTH, _CACHE_TASKS, state=("persistent_cache", "selection_scores"), stage="selector_replay", additional_stages=frozenset({"tiny_promotion", "qwen_heal"})),
    _spec("exact_cache.selector.residual_only", "exact_cache", "residual_only", "addition", "incremental", _BOTH, _CACHE_TASKS, state=("persistent_cache", "selection_scores"), stage="selector_replay", additional_stages=frozenset({"tiny_promotion", "qwen_heal"})),
    _spec("exact_cache.selector.write_value", "exact_cache", "write_value_only", "addition", "incremental", _BOTH, _CACHE_TASKS, state=("persistent_cache", "selection_scores"), stage="selector_replay", additional_stages=frozenset({"tiny_promotion", "qwen_heal"})),
    _spec("exact_cache.selector.recency", "exact_cache", "recency", "addition", "incremental", _BOTH, _CACHE_TASKS, state=("persistent_cache", "positions"), stage="selector_replay", additional_stages=frozenset({"tiny_promotion", "qwen_heal"})),
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
    *(
        _spec(
            f"exact_cache.rotation_factorial.{cell}",
            "exact_cache",
            f"cache_rotation_{cell.lower()}",
            "addition",
            "factorial",
            _TINY,
            frozenset({"far_surprise"}),
            parameters=("cache_amplitude", "rot_proj.weight", "rot_proj.bias"),
            state=("persistent_cache", "phase"),
            stage="native_interaction",
        )
        for cell in ("M00", "M10", "M01", "M11")
    ),
    *(
        _spec(
            f"exact_cache.r_out_factorial.{cell}",
            "exact_cache",
            f"cache_r_out_{cell.lower()}",
            "addition",
            "factorial",
            _TINY,
            frozenset({"far_surprise"}),
            parameters=("cache_amplitude", "q_slot_scale", "out_mix"),
            state=("persistent_cache",),
            stage="native_interaction",
        )
        for cell in ("M00", "M10", "M01", "M11")
    ),
)


def _build_registry(records: tuple[VariantSpec, ...]) -> Mapping[str, VariantSpec]:
    by_id: dict[str, VariantSpec] = {}
    by_identity: dict[tuple[str, str], str] = {}
    for record in records:
        if record.arm_id in by_id:
            raise RuntimeError(f"duplicate variant arm_id: {record.arm_id}")
        identity = (record.mechanism, record.variant)
        if identity in by_identity:
            raise RuntimeError(
                "duplicate mechanism/variant identity: "
                f"{record.mechanism}/{record.variant}"
            )
        by_id[record.arm_id] = record
        by_identity[identity] = record.arm_id
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


def validate_variant_compatibility(
    arm_id: str,
    *,
    backend: BackendName,
    task: str,
    stage: str,
    experiment_kind: ExperimentKind,
) -> VariantSpec:
    """Validate one complete execution request and return its declared arm."""

    spec = get_variant(arm_id)
    request = {
        "backend": backend,
        "task": task,
        "stage": stage,
        "experiment_kind": experiment_kind,
    }
    for name, value in request.items():
        if type(value) is not str:
            raise TypeError(f"{name} must be a str")
        if not value:
            raise ValueError(f"{name} must be nonempty")

    violations: list[str] = []
    if backend not in spec.compatible_backends:
        violations.append("backend")
    if task not in spec.compatible_tasks:
        violations.append("task")
    if stage not in spec.compatible_stages:
        violations.append("stage")
    if experiment_kind != spec.experiment_kind:
        violations.append("experiment_kind")
    if violations:
        raise VariantCompatibilityError(spec.arm_id, tuple(violations))
    return spec


def _tiny_arm_accounting(config: Any) -> TinyArmAccounting:
    from .tiny_backend import TinyKMD2Config, TinyKMD2Model

    if not isinstance(config, TinyKMD2Config):
        raise TypeError("config must be a TinyKMD2Config")
    if config.cache is not None:
        raise ValueError("state-size parameter matching cannot include exact cache")
    if config.trapezoid or config.corrected_momentum or config.causal_lookahead:
        raise ValueError("state-size parameter matching requires the base recurrence")
    model = TinyKMD2Model(config, init_seed=0)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    state_elements = config.layers * config.heads * config.dk * config.dv
    return TinyArmAccounting(
        config=config,
        d_ff=config.d_ff,
        trainable_parameters=trainable,
        recurrent_state_elements=state_elements,
        recurrent_state_bytes=4 * state_elements,
    )


def match_tiny_parameter_count(
    target_config: Any,
    arm_config: Any,
    *,
    comparison: str,
    d_ff_match_min: int,
    d_ff_match_max: int,
) -> ParameterMatchResult:
    """Select the exact instantiated feed-forward-only parameter match."""

    from .tiny_backend import TinyKMD2Config

    if not isinstance(target_config, TinyKMD2Config) or not isinstance(
        arm_config, TinyKMD2Config
    ):
        raise TypeError("target_config and arm_config must be TinyKMD2Config records")
    if comparison not in {"state_size", "mimo_rank", "factorial"}:
        raise ValueError("comparison must be state_size, mimo_rank, or factorial")
    for name, value in (
        ("d_ff_match_min", d_ff_match_min),
        ("d_ff_match_max", d_ff_match_max),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an int")
        if value < 8 or value % 8:
            raise ValueError(f"{name} must be at least 8 and divisible by 8")
    if d_ff_match_min > d_ff_match_max:
        raise ValueError("d_ff_match_min must not exceed d_ff_match_max")
    if target_config.d_ff != arm_config.d_ff:
        raise ValueError("raw fixed-FFN target and arm must share d_ff")

    variable = {"dk", "dv", "mimo_rank", "d_ff"}
    for field in fields(TinyKMD2Config):
        if field.name not in variable and getattr(target_config, field.name) != getattr(
            arm_config, field.name
        ):
            raise ValueError(
                f"parameter matching may not change {field.name}; only state/MIMO and d_ff vary"
            )
    if target_config.mimo_rank != 1:
        raise ValueError("parameter-match target must use mimo_rank=1")
    dimensions_changed = (target_config.dk, target_config.dv) != (
        arm_config.dk,
        arm_config.dv,
    )
    if comparison == "state_size":
        if arm_config.mimo_rank != 1 or not dimensions_changed:
            raise ValueError("state_size matching requires changed dk/dv at mimo_rank=1")
    elif comparison == "mimo_rank":
        if dimensions_changed or arm_config.mimo_rank <= 1:
            raise ValueError("mimo_rank matching fixes dk/dv and requires arm rank>1")
    elif not dimensions_changed and arm_config.mimo_rank == 1:
        raise ValueError("factorial matching requires a changed state size or MIMO rank")

    target = _tiny_arm_accounting(target_config)
    raw = _tiny_arm_accounting(arm_config)
    candidates = tuple(
        _tiny_arm_accounting(replace(arm_config, d_ff=d_ff))
        for d_ff in range(d_ff_match_min, d_ff_match_max + 1, 8)
    )
    matched = min(
        candidates,
        key=lambda item: (
            abs(item.trainable_parameters - target.trainable_parameters),
            item.d_ff,
        ),
    )
    residual = matched.trainable_parameters - target.trainable_parameters
    tolerance = max(0.005 * target.trainable_parameters, 1024.0)
    if abs(residual) > tolerance:
        raise ValueError(
            "no legal parameter match within tolerance for configured finite d_ff bounds"
        )
    return ParameterMatchResult(
        comparison=comparison,
        bounds=(d_ff_match_min, d_ff_match_max),
        target=target,
        raw=raw,
        matched=matched,
        tolerance=tolerance,
        residual_mismatch=residual,
    )


def construct_equal_state_byte_control(
    base_config: Any,
    *,
    cache_width: int,
    storage_dtype: str,
) -> EqualStateByteControl:
    """Instantiate the closest dk-only state-byte control without FFN matching."""

    from .tiny_backend import TinyKMD2Config

    if not isinstance(base_config, TinyKMD2Config):
        raise TypeError("base_config must be a TinyKMD2Config")
    if base_config.cache is not None:
        raise ValueError("base_config for the recurrent control must have cache disabled")
    if type(cache_width) is not int:
        raise TypeError("cache_width must be an int")
    if cache_width < 0:
        raise ValueError("cache_width must be nonnegative")
    if storage_dtype not in {"bf16", "fp32"}:
        raise ValueError("storage_dtype must be bf16 or fp32")

    storage_bytes = 2 if storage_dtype == "bf16" else 4
    slot_bytes = (base_config.dk + base_config.dv) * storage_bytes + 4 + 8 + 1
    cache_bytes = (
        base_config.layers * base_config.heads * cache_width * slot_bytes
    )
    base = _tiny_arm_accounting(base_config)
    row_bytes = base_config.layers * base_config.heads * base_config.dv * 4
    minimum_rows = 0 if cache_width == 0 else 1
    maximum_rows = max(minimum_rows, cache_bytes // row_bytes) + 2
    candidates: list[tuple[int, TinyArmAccounting]] = []
    for rows in range(minimum_rows, maximum_rows + 1):
        try:
            accounting = _tiny_arm_accounting(
                replace(base_config, dk=base_config.dk + rows)
            )
        except ValueError:
            continue
        candidates.append((rows, accounting))
    if not candidates:
        raise ValueError("no legal recurrent-state byte control")
    rows, control = min(
        candidates,
        key=lambda item: (
            abs(item[0] * row_bytes - cache_bytes),
            item[0],
        ),
    )
    increase = rows * row_bytes
    mismatch = increase - cache_bytes
    return EqualStateByteControl(
        cache_width=cache_width,
        storage_dtype=storage_dtype,
        cache_persistent_bytes=cache_bytes,
        base=base,
        control=control,
        recurrent_increase_bytes=increase,
        byte_mismatch=mismatch,
        absolute_byte_mismatch=abs(mismatch),
    )


def validate_matched_cache_controls(
    reference: CacheControlProfile,
    candidate: CacheControlProfile,
) -> CacheControlProfile:
    """Require exact capacity/read/gate/budget equality for paired caches."""

    if not isinstance(reference, CacheControlProfile):
        raise TypeError("reference must be a CacheControlProfile")
    if not isinstance(candidate, CacheControlProfile):
        raise TypeError("candidate must be a CacheControlProfile")
    mismatches = tuple(
        field.name
        for field in fields(CacheControlProfile)
        if getattr(reference, field.name) != getattr(candidate, field.name)
    )
    if mismatches:
        raise ValueError(
            "cache controls are not exactly matched: " + ", ".join(mismatches)
        )
    return candidate


def validate_cache_compatibility(
    arm_id: str,
    *,
    width: int,
    block_size: int,
    max_sequence_length: int,
    claimed_evidence_kind: EvidenceKind,
    disabled_identity: bool,
    active_output_changed: bool,
    native_feature_present: bool,
) -> VariantSpec:
    """Enforce exact-cache geometry, claim semantics, and anti-no-op gates."""

    spec = get_variant(arm_id)
    if not arm_id.startswith("exact_cache."):
        raise ValueError("validate_cache_compatibility requires an exact-cache arm")
    for name, value, minimum in (
        ("width", width, 0),
        ("block_size", block_size, 1),
        ("max_sequence_length", max_sequence_length, 1),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an int")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    if claimed_evidence_kind not in {
        "baseline",
        "addition",
        "reliance",
        "diagnostic",
    }:
        raise ValueError("claimed_evidence_kind is invalid")
    for name, value in (
        ("disabled_identity", disabled_identity),
        ("active_output_changed", active_output_changed),
        ("native_feature_present", native_feature_present),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be a bool")

    chunk_only_arms = {
        "exact_cache.current_block_only",
        "exact_cache.width.0",
    }
    if arm_id in chunk_only_arms and width != 0:
        raise ValueError("chunk-only cache controls require width=0")
    if arm_id.startswith("exact_cache.width."):
        declared_width = int(arm_id.rsplit(".", 1)[1])
        if width != declared_width:
            raise ValueError(
                f"{arm_id} declares width={declared_width}, received width={width}"
            )
    if arm_id.startswith("exact_cache.block."):
        declared_block_size = int(arm_id.rsplit(".", 1)[1])
        if block_size != declared_block_size:
            raise ValueError(
                f"{arm_id} declares block_size={declared_block_size}, "
                f"received block_size={block_size}"
            )
    if width == 0 and arm_id not in chunk_only_arms:
        raise ValueError("width=0 is reserved for a chunk-only cache control")
    if max_sequence_length < 2 * block_size:
        raise ValueError("cache screens require at least two blocks")
    if width > 0 and max_sequence_length <= width:
        raise ValueError("cache screens must contain an actual eviction")

    if claimed_evidence_kind == "reliance":
        raise ValueError("exact cache cannot support a post-hoc reliance claim")
    if spec.evidence_kind == "diagnostic" and claimed_evidence_kind != "diagnostic":
        raise ValueError(f"{arm_id} is diagnostic-only")
    if spec.evidence_kind != "diagnostic" and claimed_evidence_kind != spec.evidence_kind:
        raise ValueError(
            f"claimed evidence {claimed_evidence_kind!r} does not match "
            f"declared {spec.evidence_kind!r}"
        )
    if native_feature_present and spec.evidence_kind != "baseline":
        raise ValueError("native-present cache variants would be no-op additions")
    if spec.evidence_kind != "baseline" and not disabled_identity:
        raise ValueError("cache addition failed the disabled identity gate")
    if spec.evidence_kind != "baseline" and not active_output_changed:
        raise ValueError("cache addition failed the active effect gate")
    return spec


_CACHE_SELECTOR_ARMS = (
    "exact_cache.selector.exact_outer",
    "exact_cache.selector.coupled_paper",
    "exact_cache.selector.residual_only",
    "exact_cache.selector.write_value",
    "exact_cache.selector.recency",
    "exact_cache.selector.reservoir",
    "exact_cache.selector.future_query_oracle",
)
_CACHE_PROMOTABLE_SELECTORS = frozenset(_CACHE_SELECTOR_ARMS[:4])
_CACHE_READ_ARMS = (
    "exact_cache.read.unit_l2",
    "exact_cache.read.fixed_temperature",
    "exact_cache.read.rmsnorm",
)
_CACHE_WIDTHS = (0, 8, 16, 32, 64, 128)
_CACHE_PROMOTABLE_WIDTHS = frozenset(_CACHE_WIDTHS[1:])
_CACHE_BLOCK_SIZES = (64, 128, 256)
_CACHE_PROMOTION_TASKS = (
    "structured_exceptions",
    "mqar",
    "far_surprise",
    "freshness",
)
_CACHE_FACTORIAL_ARMS = (
    "exact_cache.rotation_factorial",
    "exact_cache.r_out_factorial",
)
_CACHE_FACTORIAL_CELLS = ("M00", "M10", "M01", "M11")


def _validated_stage_seeds(
    name: str,
    values: tuple[int, ...],
    *,
    count: int,
) -> tuple[int, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list")
    seeds = tuple(values)
    if len(seeds) != count:
        raise ValueError(f"{name} must contain exactly {count} paired seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} must not contain duplicate seeds")
    for seed in seeds:
        if type(seed) is not int:
            raise TypeError(f"{name} seeds must be ints")
        if seed < 0:
            raise ValueError(f"{name} seeds must be nonnegative")
    return seeds


def _cache_control_semantics(
    controls: CacheControlProfile | None,
) -> dict[str, Any] | None:
    if controls is None:
        return None
    return {
        field.name: getattr(controls, field.name)
        for field in fields(CacheControlProfile)
    }


def _cache_comparison_semantics(
    *,
    comparison_key: str,
    selector_arm: str | None = None,
    read_arm: str | None = None,
    cache_width: int | None = None,
    cache_block_size: int | None = None,
    controls: CacheControlProfile | None = None,
    ruler_episodes_per_cell: int | None = None,
) -> dict[str, Any]:
    return {
        "comparison_key": comparison_key,
        "selector_arm": selector_arm,
        "read_arm": read_arm,
        "cache_width": cache_width,
        "cache_block_size": cache_block_size,
        "controls": _cache_control_semantics(controls),
        "ruler_episodes_per_cell": ruler_episodes_per_cell,
    }


def _cache_stage_pairing_id(
    canonical_config: Mapping[str, Any],
    *,
    lane: str,
    stage: str,
    backend: BackendName,
    seed: int,
    task: str,
    comparison_semantics: Mapping[str, Any],
) -> str:
    import hashlib

    from .results import canonical_json_bytes

    basis = {
        "canonical_config": canonical_config,
        "lane": lane,
        "stage": stage,
        "backend": backend,
        "seed": seed,
        "task": task,
        "comparison_semantics": comparison_semantics,
    }
    return hashlib.sha256(canonical_json_bytes(basis)).hexdigest()


def _cache_stage_job(
    canonical_config: Mapping[str, Any],
    *,
    lane: str,
    stage: str,
    backend: BackendName,
    arm_id: str,
    seed: int,
    task: str,
    comparison_semantics: Mapping[str, Any],
    declared_arm_id: str | None = None,
    cell: str | None = None,
    selector_arm: str | None = None,
    read_arm: str | None = None,
    cache_width: int | None = None,
    cache_block_size: int | None = None,
    controls: CacheControlProfile | None = None,
    ruler_episodes_per_cell: int | None = None,
) -> CacheStageJob:
    from .results import semantic_job_id

    pairing_id = _cache_stage_pairing_id(
        canonical_config,
        lane=lane,
        stage=stage,
        backend=backend,
        seed=seed,
        task=task,
        comparison_semantics=comparison_semantics,
    )
    job_id = semantic_job_id(
        canonical_config,
        backend=backend,
        arm_id=arm_id,
        seed=seed,
        stage=stage,
        pairing_id=pairing_id,
    )
    return CacheStageJob(
        job_id=job_id,
        pairing_id=pairing_id,
        lane=lane,
        stage=stage,
        backend=backend,
        arm_id=arm_id,
        declared_arm_id=declared_arm_id or arm_id,
        seed=seed,
        task=task,
        cell=cell,
        selector_arm=selector_arm,
        read_arm=read_arm,
        cache_width=cache_width,
        cache_block_size=cache_block_size,
        controls=controls,
        ruler_episodes_per_cell=ruler_episodes_per_cell,
    )


def _cache_stage_batch(lane: str, jobs: list[CacheStageJob]) -> CacheStageBatch:
    return CacheStageBatch(lane=lane, jobs=tuple(jobs))


def _lcb_passes(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def expand_exact_cache_stages(
    canonical_config: Mapping[str, Any],
    *,
    screen_seeds: tuple[int, ...],
    promotion_seeds: tuple[int, ...],
    heal_seeds: tuple[int, ...],
    evidence: CacheStageEvidence,
    winner_selector_arm: str | None = None,
    winner_read_arm: str | None = None,
    winner_width: int | None = None,
    winner_block_size: int | None = None,
    recency_controls: CacheControlProfile | None = None,
    surprise_controls: CacheControlProfile | None = None,
    ruler_episodes_per_cell: int = 64,
    fixed_cache_width: int = 64,
    fixed_block_size: int = 128,
) -> tuple[CacheStageBatch, ...]:
    """Expand only the prefix unlocked by the preregistered serial cache gates."""

    from .results import canonical_json_bytes

    if not isinstance(canonical_config, Mapping):
        raise TypeError("canonical_config must be a mapping")
    canonical_json_bytes(canonical_config)
    if not isinstance(evidence, CacheStageEvidence):
        raise TypeError("evidence must be CacheStageEvidence")
    screen_seeds = _validated_stage_seeds("screen_seeds", screen_seeds, count=3)
    promotion_seeds = _validated_stage_seeds(
        "promotion_seeds", promotion_seeds, count=5
    )
    heal_seeds = _validated_stage_seeds("heal_seeds", heal_seeds, count=3)
    for name, value in (
        ("fixed_cache_width", fixed_cache_width),
        ("fixed_block_size", fixed_block_size),
        ("ruler_episodes_per_cell", ruler_episodes_per_cell),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an int")
    if fixed_cache_width <= 0:
        raise ValueError("fixed_cache_width must be positive")
    if fixed_block_size <= 0:
        raise ValueError("fixed_block_size must be positive")
    if ruler_episodes_per_cell < 64:
        raise ValueError("Qwen heal requires at least 64 matched RULER episodes per cell")
    if recency_controls is not None and not isinstance(
        recency_controls, CacheControlProfile
    ):
        raise TypeError("recency_controls must be CacheControlProfile or None")
    if surprise_controls is not None and not isinstance(
        surprise_controls, CacheControlProfile
    ):
        raise TypeError("surprise_controls must be CacheControlProfile or None")

    batches: list[CacheStageBatch] = []
    selector_jobs = [
        _cache_stage_job(
            canonical_config,
            lane="selector_replay",
            stage="selector_replay",
            backend="tiny",
            arm_id=arm_id,
            seed=seed,
            task="far_surprise",
            comparison_semantics=_cache_comparison_semantics(
                comparison_key="selectors",
                cache_width=fixed_cache_width,
                cache_block_size=fixed_block_size,
            ),
            selector_arm=arm_id,
            cache_width=fixed_cache_width,
            cache_block_size=fixed_block_size,
        )
        for seed in screen_seeds
        for arm_id in _CACHE_SELECTOR_ARMS
    ]
    batches.append(_cache_stage_batch("selector_replay", selector_jobs))

    if winner_selector_arm is not None and winner_selector_arm not in _CACHE_SELECTOR_ARMS:
        raise ValueError("winner_selector_arm is not a declared selector replay arm")
    if winner_selector_arm == "exact_cache.selector.future_query_oracle":
        raise ValueError("the future-query oracle is diagnostic-only and cannot promote")
    if (
        winner_selector_arm not in _CACHE_PROMOTABLE_SELECTORS
        or not _lcb_passes(evidence.selector_primary_lcb, 0.05)
    ):
        return tuple(batches)

    read_jobs = [
        _cache_stage_job(
            canonical_config,
            lane="read_screen",
            stage="read_screen",
            backend="tiny",
            arm_id=arm_id,
            seed=seed,
            task="far_surprise",
            comparison_semantics=_cache_comparison_semantics(
                comparison_key="reads",
                selector_arm=winner_selector_arm,
                cache_width=fixed_cache_width,
                cache_block_size=fixed_block_size,
            ),
            selector_arm=winner_selector_arm,
            read_arm=arm_id,
            cache_width=fixed_cache_width,
            cache_block_size=fixed_block_size,
        )
        for seed in screen_seeds
        for arm_id in _CACHE_READ_ARMS
    ]
    batches.append(_cache_stage_batch("read_screen", read_jobs))

    if winner_read_arm is not None and winner_read_arm not in _CACHE_READ_ARMS:
        raise ValueError("winner_read_arm is not a declared cache read arm")
    if winner_read_arm is None or not _lcb_passes(evidence.read_primary_lcb, 0.05):
        return tuple(batches)

    width_jobs = [
        _cache_stage_job(
            canonical_config,
            lane="capacity_width",
            stage="capacity_screen",
            backend="tiny",
            arm_id=f"exact_cache.width.{width}",
            seed=seed,
            task="mqar",
            comparison_semantics=_cache_comparison_semantics(
                comparison_key="widths",
                selector_arm=winner_selector_arm,
                read_arm=winner_read_arm,
                cache_block_size=fixed_block_size,
            ),
            selector_arm=winner_selector_arm,
            read_arm=winner_read_arm,
            cache_width=width,
            cache_block_size=fixed_block_size,
        )
        for seed in screen_seeds
        for width in _CACHE_WIDTHS
    ]
    batches.append(_cache_stage_batch("capacity_width", width_jobs))

    if winner_width is not None and winner_width not in _CACHE_WIDTHS:
        raise ValueError("winner_width is not a declared cache width")
    if winner_width not in _CACHE_PROMOTABLE_WIDTHS:
        return tuple(batches)

    block_jobs = [
        _cache_stage_job(
            canonical_config,
            lane="capacity_block",
            stage="capacity_screen",
            backend="tiny",
            arm_id=f"exact_cache.block.{block_size}",
            seed=seed,
            task="mqar",
            comparison_semantics=_cache_comparison_semantics(
                comparison_key="blocks",
                selector_arm=winner_selector_arm,
                read_arm=winner_read_arm,
                cache_width=winner_width,
            ),
            selector_arm=winner_selector_arm,
            read_arm=winner_read_arm,
            cache_width=winner_width,
            cache_block_size=block_size,
        )
        for seed in screen_seeds
        for block_size in _CACHE_BLOCK_SIZES
    ]
    batches.append(_cache_stage_batch("capacity_block", block_jobs))

    if winner_block_size is not None and winner_block_size not in _CACHE_BLOCK_SIZES:
        raise ValueError("winner_block_size is not a declared cache block size")
    if winner_block_size is None:
        return tuple(batches)
    if recency_controls is None or surprise_controls is None:
        raise ValueError("tiny promotion requires recency and surprise cache controls")
    controls = validate_matched_cache_controls(recency_controls, surprise_controls)
    if controls.cache_width != winner_width:
        raise ValueError("cache controls cache_width must equal winner_width")
    if controls.cache_block_size != winner_block_size:
        raise ValueError("cache controls cache_block_size must equal winner_block_size")
    if controls.read_arm != winner_read_arm:
        raise ValueError("cache controls read_arm must equal winner_read_arm")

    promotion_arms = (
        "native",
        "exact_cache.selector.recency",
        winner_selector_arm,
    )
    promotion_jobs = [
        _cache_stage_job(
            canonical_config,
            lane="tiny_promotion",
            stage="tiny_promotion",
            backend="tiny",
            arm_id=arm_id,
            seed=seed,
            task=task,
            comparison_semantics=_cache_comparison_semantics(
                comparison_key="promotion",
                selector_arm=winner_selector_arm,
                read_arm=winner_read_arm,
                cache_width=winner_width,
                cache_block_size=winner_block_size,
                controls=controls,
            ),
            selector_arm=(None if arm_id == "native" else arm_id),
            read_arm=(None if arm_id == "native" else winner_read_arm),
            cache_width=(None if arm_id == "native" else winner_width),
            cache_block_size=(None if arm_id == "native" else winner_block_size),
            controls=controls,
        )
        for task in _CACHE_PROMOTION_TASKS
        for seed in promotion_seeds
        for arm_id in promotion_arms
    ]
    batches.append(_cache_stage_batch("tiny_promotion", promotion_jobs))

    tiny_gates = (
        _lcb_passes(evidence.tiny_primary_lcb, 0.05),
        _lcb_passes(evidence.short_accuracy_vs_native_lcb, -0.02),
        _lcb_passes(evidence.freshness_latest_vs_native_lcb, -0.02),
        _lcb_passes(evidence.freshness_stale_vs_native_lcb, -0.02),
        _lcb_passes(evidence.freshness_latest_vs_recency_lcb, -0.02),
        _lcb_passes(evidence.freshness_stale_vs_recency_lcb, -0.02),
    )
    if not all(tiny_gates):
        return tuple(batches)

    interaction_jobs: list[CacheStageJob] = []
    for declared_arm in _CACHE_FACTORIAL_ARMS:
        for seed in promotion_seeds:
            for cell in _CACHE_FACTORIAL_CELLS:
                arm_id = f"{declared_arm}.{cell}"
                interaction_jobs.append(
                    _cache_stage_job(
                        canonical_config,
                        lane="native_interaction",
                        stage="native_interaction",
                        backend="tiny",
                        arm_id=arm_id,
                        declared_arm_id=declared_arm,
                        seed=seed,
                        task="far_surprise",
                        comparison_semantics=_cache_comparison_semantics(
                            comparison_key=f"factorial:{declared_arm}",
                            selector_arm=winner_selector_arm,
                            read_arm=winner_read_arm,
                            cache_width=winner_width,
                            cache_block_size=winner_block_size,
                            controls=controls,
                        ),
                        cell=cell,
                        selector_arm=winner_selector_arm,
                        read_arm=winner_read_arm,
                        cache_width=winner_width,
                        cache_block_size=winner_block_size,
                        controls=controls,
                    )
                )
    batches.append(_cache_stage_batch("native_interaction", interaction_jobs))
    if not evidence.interactions_complete:
        return tuple(batches)

    qwen_jobs = [
        _cache_stage_job(
            canonical_config,
            lane="qwen_heal",
            stage="qwen_heal",
            backend="qwen",
            arm_id=arm_id,
            seed=seed,
            task="ruler",
            comparison_semantics=_cache_comparison_semantics(
                comparison_key="qwen_heal",
                selector_arm=winner_selector_arm,
                read_arm=winner_read_arm,
                cache_width=winner_width,
                cache_block_size=winner_block_size,
                controls=controls,
                ruler_episodes_per_cell=ruler_episodes_per_cell,
            ),
            selector_arm=(None if arm_id == "native" else arm_id),
            read_arm=(None if arm_id == "native" else winner_read_arm),
            cache_width=(None if arm_id == "native" else winner_width),
            cache_block_size=(None if arm_id == "native" else winner_block_size),
            controls=controls,
            ruler_episodes_per_cell=ruler_episodes_per_cell,
        )
        for seed in heal_seeds
        for arm_id in promotion_arms
    ]
    batches.append(_cache_stage_batch("qwen_heal", qwen_jobs))
    return tuple(batches)


def trapezoid_convolution_interaction_allowed(*, trapezoid_promoted: bool) -> bool:
    """Gate the replacement interaction on a completed individual screen."""

    if type(trapezoid_promoted) is not bool:
        raise TypeError("trapezoid_promoted must be a bool")
    return trapezoid_promoted


def momentum_decay_erase_interaction_allowed(*, momentum_promoted: bool) -> bool:
    """Gate momentum interaction cells on the individual momentum screen."""

    if type(momentum_promoted) is not bool:
        raise TypeError("momentum_promoted must be a bool")
    return momentum_promoted


def lookahead_convolution_interaction_allowed(*, lookahead_promoted: bool) -> bool:
    """Gate the lookahead/convolution cell on the lookahead screen."""

    if type(lookahead_promoted) is not bool:
        raise TypeError("lookahead_promoted must be a bool")
    return lookahead_promoted


def lookahead_trapezoid_interaction_allowed(
    *, lookahead_promoted: bool, trapezoid_promoted: bool
) -> bool:
    """Require both individual additions before their interaction cell."""

    if type(lookahead_promoted) is not bool:
        raise TypeError("lookahead_promoted must be a bool")
    if type(trapezoid_promoted) is not bool:
        raise TypeError("trapezoid_promoted must be a bool")
    return lookahead_promoted and trapezoid_promoted


def bc_bias_trapezoid_interaction_allowed(
    *, bc_bias_promoted: bool, trapezoid_promoted: bool
) -> bool:
    """Require both individual additions before the bias/trapezoid cell."""

    if type(bc_bias_promoted) is not bool:
        raise TypeError("bc_bias_promoted must be a bool")
    if type(trapezoid_promoted) is not bool:
        raise TypeError("trapezoid_promoted must be a bool")
    return bc_bias_promoted and trapezoid_promoted


def bc_bias_pair_convolution_interaction_allowed(
    *, bc_bias_promoted: bool, trapezoid_promoted: bool, pair_promoted: bool
) -> bool:
    """Gate the winning bias/trapezoid pair's convolution interaction."""

    for name, value in (
        ("bc_bias_promoted", bc_bias_promoted),
        ("trapezoid_promoted", trapezoid_promoted),
        ("pair_promoted", pair_promoted),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be a bool")
    return bc_bias_promoted and trapezoid_promoted and pair_promoted


__all__ = [
    "CacheControlProfile",
    "CacheStageBatch",
    "CacheStageEvidence",
    "CacheStageJob",
    "EqualStateByteControl",
    "ParameterMatchResult",
    "TinyArmAccounting",
    "VARIANT_REGISTRY",
    "VariantCompatibilityError",
    "VariantSpec",
    "all_variants",
    "bc_bias_pair_convolution_interaction_allowed",
    "bc_bias_trapezoid_interaction_allowed",
    "construct_equal_state_byte_control",
    "expand_exact_cache_stages",
    "get_variant",
    "lookup_variant",
    "lookahead_convolution_interaction_allowed",
    "lookahead_trapezoid_interaction_allowed",
    "match_tiny_parameter_count",
    "momentum_decay_erase_interaction_allowed",
    "trapezoid_convolution_interaction_allowed",
    "validate_cache_compatibility",
    "validate_matched_cache_controls",
    "validate_variant_compatibility",
]
