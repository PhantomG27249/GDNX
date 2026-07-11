"""Immutable configuration and stable identity for KMD-2 ablation jobs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Any


SCHEMA_VERSION = "1.0.0"
SUITE_VERSION = "1.0.0"
_MAX_TASK_PARAMS_DEPTH = 64

_BACKEND_ALIASES = {"torch_reference": "tiny", "qwen_native": "qwen"}
_BACKENDS = {"tiny", "qwen", *_BACKEND_ALIASES}
_QWEN_RUN_MODES = {"reliance", "heal", "initial_exact_cache"}
_BASELINES = {"gdn2_native", "kmd2_native", "native_continuation"}
_MECHANISMS = {
    "native",
    "rotation",
    "convolution",
    "trapezoid",
    "bc_bias",
    "corrected_momentum",
    "causal_lookahead",
    "state_size",
    "true_mimo",
    "gdn2_decoupled",
    "exact_cache",
    "current_block_only",
}
_VARIANTS = {
    "native",
    "current_rotation",
    "rotation_off",
    "constant_rate_rotation",
    "non_cumulative_rotation",
    "fixed_rope",
    "moving_frame_oracle",
    "convolution_on",
    "convolution_off",
    "trapezoid",
    "bc_bias",
    "diagonal_rescale",
    "constant_coordinate_oracle",
    "corrected_momentum",
    "causal_lookahead",
    "state_size_sweep",
    "true_mimo_sweep",
    "channelwise_erase_write",
    "cache_off",
    "chunk_only",
    "top_surprise",
    "coupled_surprise",
    "residual_only",
    "write_value_only",
    "recency",
    "reservoir",
    "future_query_oracle",
    "unbounded_oracle",
    "per_slot_read",
    "cache_rotation_factorial",
    "cache_r_out_factorial",
}
_TASKS = {
    "mqar",
    "ruler",
    "state_tracking",
    "parity",
    "modular_counter",
    "toggle_fsm",
    "irregular_integration",
    "drift_reversal",
    "trajectory",
    "local_binding",
    "structured_exceptions",
    "far_surprise",
    "freshness",
    "affine_associative_regression",
}
_OPTIMIZERS = {"adamw"}
_SCHEDULES = {"cosine"}
_DIRECTIONS = {"maximize", "minimize"}
_STAGES = {
    "local_correctness",
    "mechanism_screen",
    "tiny_promotion",
    "qwen_reliance",
    "qwen_heal",
    "selector_replay",
    "read_screen",
    "capacity_screen",
    "native_interaction",
    "streaming_promotion",
}
_DEVICES = {"cuda", "cpu"}
_DTYPES = {"bfloat16", "float32"}

_CACHE_SCORE_POLICIES = {
    "exact_outer",
    "coupled_paper",
    "residual_only",
    "write_value",
    "recency",
    "reservoir",
    "future_query_oracle",
}
_CACHE_READ_POLICIES = {"unit_l2", "fixed_temperature", "rmsnorm"}
_CACHE_READ_INITIALIZATIONS = {"gamma_one_sink_zero_amplitude_zero"}
_CACHE_STORAGE_DTYPES = {"fp32", "bf16"}
_CACHE_COMPUTE_DTYPES = {"fp32"}

# Input-only compatibility spellings. CacheConfig and canonical serialization
# always expose the approved scientific names on the right-hand side.
_CACHE_POLICY_ALIASES = {
    "score": {"surprise_l2": "exact_outer"},
    "read": {"softmax": "fixed_temperature"},
    "read_init": {"zero": "gamma_one_sink_zero_amplitude_zero"},
    "storage_dtype": {"float32": "fp32", "bfloat16": "bf16"},
    "compute_dtype": {"float32": "fp32"},
}

_QWEN_FIELDS = {
    "model_asset",
    "tokenizer_asset",
    "run_mode",
    "streaming",
    "decode",
    "packing",
    "padding",
    "attention_mask",
}
_TASK_FIELDS = {"name", "params"}
_BUDGET_FIELDS = {"tokens", "updates"}
_OPTIMIZER_FIELDS = {
    "name",
    "learning_rate",
    "betas",
    "eps",
    "weight_decay",
}
_SCHEDULE_FIELDS = {"name", "warmup_updates"}
_MODEL_FIELDS = {
    "hidden_size",
    "num_layers",
    "num_heads",
    "state_key_dim",
    "state_value_dim",
    "ffn_dim",
    "ffn_match_lower",
    "ffn_match_upper",
}
_LENGTH_FIELDS = {"curriculum", "extrapolation"}
_EVALUATION_FIELDS = {"primary_metric", "direction"}
_THRESHOLD_FIELDS = {
    "min_useful_addition",
    "min_reliance",
    "equivalence_tolerance",
    "harm_threshold",
    "synergy_threshold",
}
_PROTECTED_METRIC_FIELDS = {"name", "max_regression"}
_PROMOTION_FIELDS = {
    "min_gate_mean",
    "min_gate_max",
    "min_persistent_hit_rate",
    "min_conditional_read_accuracy",
    "min_shuffled_cache_dependence",
    "min_adjacent_capacity_lcb",
}
_CACHE_FIELDS = {
    "width",
    "block_size",
    "score",
    "read",
    "read_init",
    "eps_cache",
    "coordinate_frame",
    "pre_rotation_diagnostic",
    "storage_dtype",
    "compute_dtype",
    "inclusive",
    "tie_policy",
    "lr_cache",
    "weight_decay_cache",
}
_RUNTIME_FIELDS = {"output_path", "device_ordinal"}


@dataclass(frozen=True)
class QwenConfig:
    model_asset: str
    tokenizer_asset: str
    run_mode: str
    streaming: bool
    decode: bool
    packing: bool
    padding: str
    attention_mask: str


@dataclass(frozen=True)
class TaskConfig:
    name: str
    params: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.params, Mapping):
            raise TypeError("task.params must be a mapping")
        object.__setattr__(
            self,
            "params",
            _freeze_json(self.params, "task.params"),
        )


@dataclass(frozen=True)
class BudgetConfig:
    tokens: int
    updates: int

    def __post_init__(self) -> None:
        _require_int("budget.tokens", self.tokens, minimum=1)
        _require_int("budget.updates", self.updates, minimum=1)


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    betas: tuple[float, ...]
    eps: float
    weight_decay: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "learning_rate",
            _canonical_real(
                "optimizer.learning_rate",
                self.learning_rate,
                minimum=0.0,
                strict_minimum=True,
            ),
        )
        if type(self.betas) is not tuple or len(self.betas) != 2:
            raise TypeError("optimizer.betas must be a list or tuple of two numbers")
        object.__setattr__(
            self,
            "betas",
            tuple(
                _canonical_real(
                    "optimizer.betas",
                    beta,
                    minimum=0.0,
                    maximum=1.0,
                    strict_maximum=True,
                )
                for beta in self.betas
            ),
        )
        object.__setattr__(
            self,
            "eps",
            _canonical_real(
                "optimizer.eps",
                self.eps,
                minimum=0.0,
                strict_minimum=True,
            ),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _canonical_real(
                "optimizer.weight_decay",
                self.weight_decay,
                minimum=0.0,
            ),
        )


@dataclass(frozen=True)
class ScheduleConfig:
    name: str
    warmup_updates: int

    def __post_init__(self) -> None:
        _require_int("schedule.warmup_updates", self.warmup_updates, minimum=0)


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    state_key_dim: int
    state_value_dim: int
    ffn_dim: int
    ffn_match_lower: int
    ffn_match_upper: int

    def __post_init__(self) -> None:
        for field_name in (
            "hidden_size",
            "num_layers",
            "num_heads",
            "state_key_dim",
            "state_value_dim",
        ):
            _require_int(
                f"model.{field_name}", getattr(self, field_name), minimum=1
            )
        for field_name in ("ffn_dim", "ffn_match_lower", "ffn_match_upper"):
            value = getattr(self, field_name)
            _require_int(f"model.{field_name}", value, minimum=8)
            if value % 8:
                raise ValueError(f"model.{field_name} must be divisible by 8")
        if not self.ffn_match_lower <= self.ffn_dim <= self.ffn_match_upper:
            raise ValueError(
                "model.ffn_dim must satisfy "
                "ffn_match_lower <= ffn_dim <= ffn_match_upper"
            )


@dataclass(frozen=True)
class LengthConfig:
    curriculum: tuple[int, ...]
    extrapolation: tuple[int, ...]

    def __post_init__(self) -> None:
        for field_name in ("curriculum", "extrapolation"):
            values = getattr(self, field_name)
            if type(values) is not tuple:
                raise TypeError(f"lengths.{field_name} must be a list or tuple")
            if not values:
                raise ValueError(f"lengths.{field_name} must not be empty")
            for value in values:
                _require_int(f"lengths.{field_name}", value, minimum=1)


@dataclass(frozen=True)
class EvaluationConfig:
    primary_metric: str
    direction: str


@dataclass(frozen=True)
class ScientificThresholds:
    min_useful_addition: float
    min_reliance: float
    equivalence_tolerance: float
    harm_threshold: float
    synergy_threshold: float

    def __post_init__(self) -> None:
        for field_name in (
            "min_useful_addition",
            "min_reliance",
            "equivalence_tolerance",
            "harm_threshold",
            "synergy_threshold",
        ):
            object.__setattr__(
                self,
                field_name,
                _canonical_real(
                    f"thresholds.{field_name}", getattr(self, field_name)
                ),
            )

        if self.min_useful_addition <= 0:
            raise ValueError("thresholds.min_useful_addition must be greater than 0")
        if self.equivalence_tolerance < 0:
            raise ValueError("thresholds.equivalence_tolerance must be at least 0")
        if self.min_reliance <= self.equivalence_tolerance:
            raise ValueError(
                "thresholds.min_reliance must be greater than "
                "thresholds.equivalence_tolerance"
            )
        if self.harm_threshold <= self.equivalence_tolerance:
            raise ValueError(
                "thresholds.harm_threshold must be greater than "
                "thresholds.equivalence_tolerance"
            )
        if self.synergy_threshold < 0:
            raise ValueError("thresholds.synergy_threshold must be at least 0")


@dataclass(frozen=True)
class ProtectedMetric:
    """A protected metric and its metric-specific regression allowance."""

    name: str
    max_regression: float

    def __post_init__(self) -> None:
        _require_string("protected_metrics.name", self.name)
        object.__setattr__(
            self,
            "max_regression",
            _canonical_real(
                "protected_metrics.max_regression",
                self.max_regression,
                minimum=0.0,
            ),
        )


@dataclass(frozen=True)
class PromotionThresholds:
    min_gate_mean: float = 0.005
    min_gate_max: float = 0.02
    min_persistent_hit_rate: float = 0.25
    min_conditional_read_accuracy: float = 0.50
    min_shuffled_cache_dependence: float = 0.05
    min_adjacent_capacity_lcb: float = 0.05

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(
                self,
                field.name,
                _canonical_real(
                    f"promotion.{field.name}",
                    getattr(self, field.name),
                    minimum=0.0,
                    maximum=1.0,
                ),
            )

    @property
    def min_conditional_read_rate(self) -> float:
        """Compatibility name for the conditional-read promotion threshold."""
        return self.min_conditional_read_accuracy

    @property
    def min_shuffled_dependence(self) -> float:
        """Compatibility name for the shuffled-cache promotion threshold."""
        return self.min_shuffled_cache_dependence


@dataclass(frozen=True)
class CacheConfig:
    width: int = 32
    block_size: int = 64
    score: str = "exact_outer"
    read: str = "unit_l2"
    read_init: str = "gamma_one_sink_zero_amplitude_zero"
    eps_cache: float = 1.0e-6
    coordinate_frame: str = "rotated_recurrence"
    pre_rotation_diagnostic: bool = False
    storage_dtype: str = "bf16"
    compute_dtype: str = "fp32"
    inclusive: bool = True
    tie_policy: str = "score_desc_position_desc"
    lr_cache: float = 2.0e-3
    weight_decay_cache: float = 0.0

    def __post_init__(self) -> None:
        for field_name, aliases in _CACHE_POLICY_ALIASES.items():
            value = getattr(self, field_name)
            if type(value) is str:
                object.__setattr__(self, field_name, aliases.get(value, value))

        _require_int("cache.width", self.width, minimum=0)
        _require_int("cache.block_size", self.block_size, minimum=1)
        _require_choice("cache.score", self.score, _CACHE_SCORE_POLICIES)
        _require_choice("cache.read", self.read, _CACHE_READ_POLICIES)
        _require_choice(
            "cache.read_init", self.read_init, _CACHE_READ_INITIALIZATIONS
        )
        object.__setattr__(
            self,
            "eps_cache",
            _canonical_real(
                "cache.eps_cache",
                self.eps_cache,
                minimum=0.0,
                strict_minimum=True,
            ),
        )
        _require_choice(
            "cache.coordinate_frame",
            self.coordinate_frame,
            {"rotated_recurrence", "pre_rotation"},
        )
        _require_bool("cache.pre_rotation_diagnostic", self.pre_rotation_diagnostic)
        if self.coordinate_frame == "pre_rotation" and not self.pre_rotation_diagnostic:
            raise ValueError(
                "cache.coordinate_frame pre_rotation requires "
                "cache.pre_rotation_diagnostic=true"
            )
        _require_choice(
            "cache.storage_dtype", self.storage_dtype, _CACHE_STORAGE_DTYPES
        )
        _require_choice(
            "cache.compute_dtype", self.compute_dtype, _CACHE_COMPUTE_DTYPES
        )
        _require_bool("cache.inclusive", self.inclusive)
        if not self.inclusive:
            raise ValueError("cache.inclusive must be true")
        _require_choice(
            "cache.tie_policy", self.tie_policy, {"score_desc_position_desc"}
        )
        object.__setattr__(
            self,
            "lr_cache",
            _canonical_real(
                "cache.lr_cache",
                self.lr_cache,
                minimum=0.0,
                strict_minimum=True,
            ),
        )
        object.__setattr__(
            self,
            "weight_decay_cache",
            _canonical_real("cache.weight_decay_cache", self.weight_decay_cache),
        )
        if self.weight_decay_cache != 0:
            raise ValueError("cache.weight_decay_cache must be exactly 0")


@dataclass(frozen=True)
class RuntimeConfig:
    output_path: str
    device_ordinal: int

    def __post_init__(self) -> None:
        _require_int("runtime.device_ordinal", self.device_ordinal, minimum=0)


def _freeze(value: Any) -> Any:
    """Recursively detach and freeze JSON-compatible containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    return value


def _freeze_json(
    value: Any,
    path: str,
    *,
    _active_containers: set[int] | None = None,
    _depth: int = 0,
) -> Any:
    """Validate a JSON value recursively while returning frozen containers."""
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} float values must be finite")
        return value
    if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
        active_containers = (
            set() if _active_containers is None else _active_containers
        )
        container_id = id(value)
        if container_id in active_containers:
            raise ValueError(f"{path} contains a JSON cycle")
        if _depth > _MAX_TASK_PARAMS_DEPTH:
            raise ValueError(
                f"{path} exceeds the maximum task.params JSON depth of "
                f"{_MAX_TASK_PARAMS_DEPTH}"
            )

        active_containers.add(container_id)
        try:
            if isinstance(value, Mapping):
                frozen = {}
                for key, item in value.items():
                    if type(key) is not str:
                        raise TypeError(f"{path} mapping keys must be strings")
                    frozen[key] = _freeze_json(
                        item,
                        f"{path}[{key!r}]",
                        _active_containers=active_containers,
                        _depth=_depth + 1,
                    )
                return MappingProxyType(frozen)
            return tuple(
                _freeze_json(
                    item,
                    f"{path}[{index}]",
                    _active_containers=active_containers,
                    _depth=_depth + 1,
                )
                for index, item in enumerate(value)
            )
        finally:
            active_containers.remove(container_id)
    raise TypeError(
        f"{path} contains unsupported JSON value type {value_type.__name__}"
    )


def _plain(value: Any) -> Any:
    """Return a fresh JSON-compatible representation of frozen values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _mapping(
    raw: Mapping[str, Any],
    key: str,
    required: set[str] | frozenset[str] = frozenset(),
    optional: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    value = raw[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    result = dict(value)
    if required or optional:
        unknown = set(result) - required - optional
        missing = required - set(result)
        if unknown:
            names = ", ".join(sorted(map(str, unknown)))
            raise ValueError(f"unknown {key} configuration keys: {names}")
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"missing {key} configuration keys: {names}")
    return result


def _require_bool(name: str, value: Any) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")


def _require_int(name: str, value: Any, *, minimum: int | None = None) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_number(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
    maximum: float | None = None,
    strict_maximum: bool = False,
) -> None:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        invalid = value <= minimum if strict_minimum else value < minimum
        if invalid:
            relation = "greater than" if strict_minimum else "at least"
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None:
        invalid = value >= maximum if strict_maximum else value > maximum
        if invalid:
            relation = "less than" if strict_maximum else "at most"
            raise ValueError(f"{name} must be {relation} {maximum}")


def _canonical_real(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
    maximum: float | None = None,
    strict_maximum: bool = False,
) -> float:
    """Validate and return the one canonical float spelling of a real value."""
    _require_number(
        name,
        value,
        minimum=minimum,
        strict_minimum=strict_minimum,
        maximum=maximum,
        strict_maximum=strict_maximum,
    )
    canonical = float(value)
    return 0.0 if canonical == 0.0 else canonical


def _require_list(name: str, value: Any) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a list")
    return value


def _require_string(name: str, value: Any) -> None:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty string")


def _require_choice(
    name: str, value: Any, choices: set[str] | frozenset[str]
) -> None:
    _require_string(name, value)
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {allowed}")


def _require_string_list(name: str, value: Any, choices: set[str] | None = None) -> None:
    items = _require_list(name, value)
    if not items:
        raise ValueError(f"{name} must not be empty")
    for item in items:
        _require_string(name, item)
        if choices is not None and item not in choices:
            allowed = ", ".join(sorted(choices))
            raise ValueError(f"{name} entries must be one of: {allowed}")


def _validate_qwen_config(qwen: Mapping[str, Any]) -> None:
    _require_string("qwen.model_asset", qwen["model_asset"])
    _require_string("qwen.tokenizer_asset", qwen["tokenizer_asset"])
    _require_choice("qwen.run_mode", qwen["run_mode"], _QWEN_RUN_MODES)
    for field_name in ("streaming", "decode", "packing"):
        _require_bool(f"qwen.{field_name}", qwen[field_name])
    _require_string("qwen.padding", qwen["padding"])
    _require_string("qwen.attention_mask", qwen["attention_mask"])

    if qwen["run_mode"] != "initial_exact_cache":
        return

    if qwen["streaming"] or qwen["decode"] or qwen["packing"]:
        raise ValueError(
            "initial exact-cache qwen mode requires streaming, decode, "
            "and packing to be false"
        )
    if qwen["padding"] != "none":
        raise ValueError("initial exact-cache qwen mode requires padding=none")
    if qwen["attention_mask"] not in {
        "none",
        "all_ones",
    }:
        raise ValueError(
            "initial exact-cache qwen mode requires an unpadded full-sequence "
            "attention mask"
        )


def _build_cache_config(
    cache_values: Mapping[str, Any],
    mechanism: str,
    variant: str,
    lengths: Mapping[str, Any],
) -> CacheConfig:
    cache = CacheConfig(**cache_values)

    zero_width_control = mechanism == "current_block_only" and variant == "chunk_only"
    if (cache.width == 0) != zero_width_control:
        raise ValueError(
            "cache.width must be 0 if and only if mechanism=current_block_only "
            "and variant=chunk_only"
        )

    if variant == "top_surprise":
        if cache.width < 1:
            raise ValueError("variant=top_surprise requires cache.width >= 1")

        curriculum = lengths["curriculum"]
        if not curriculum or max(curriculum) < 2 * cache.block_size:
            raise ValueError(
                "variant=top_surprise requires a maximum curriculum length "
                "covering at least two processing blocks"
            )
        if max(curriculum) <= cache.width:
            raise ValueError(
                "variant=top_surprise requires eviction: maximum curriculum "
                "length must be greater than cache.width"
            )

    return cache


def _build_protected_metrics(value: Any) -> tuple[ProtectedMetric, ...]:
    items = _require_list("protected_metrics", value)
    if not items:
        raise ValueError("protected_metrics must not be empty")

    metrics: list[ProtectedMetric] = []
    names: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"protected_metrics[{index}] must be a mapping with name and "
                "max_regression"
            )
        values = dict(item)
        unknown = set(values) - _PROTECTED_METRIC_FIELDS
        missing = _PROTECTED_METRIC_FIELDS - set(values)
        if unknown:
            fields_list = ", ".join(sorted(map(str, unknown)))
            raise ValueError(
                f"unknown protected_metrics[{index}] keys: {fields_list}"
            )
        if missing:
            fields_list = ", ".join(sorted(missing))
            raise ValueError(
                f"missing protected_metrics[{index}] keys: {fields_list}"
            )

        metric = ProtectedMetric(**values)
        if metric.name in names:
            raise ValueError(
                f"protected_metrics contains duplicate metric name: {metric.name}"
            )
        names.add(metric.name)
        metrics.append(metric)
    return tuple(metrics)


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str
    suite_version: str
    backend: str
    qwen: QwenConfig
    baseline: str
    mechanism: str
    variant: str
    task: TaskConfig
    seeds: tuple[int, ...]
    budget: BudgetConfig
    optimizer: OptimizerConfig
    schedule: ScheduleConfig
    model: ModelConfig
    lengths: LengthConfig
    evaluation: EvaluationConfig
    thresholds: ScientificThresholds
    promotion: PromotionThresholds
    protected_metrics: tuple[ProtectedMetric, ...]
    device_preferences: tuple[str, ...]
    dtype_preferences: tuple[str, ...]
    required_stage: str
    cache: CacheConfig
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        record_fields = (
            ("qwen", QwenConfig),
            ("task", TaskConfig),
            ("budget", BudgetConfig),
            ("optimizer", OptimizerConfig),
            ("schedule", ScheduleConfig),
            ("model", ModelConfig),
            ("lengths", LengthConfig),
            ("evaluation", EvaluationConfig),
            ("thresholds", ScientificThresholds),
            ("promotion", PromotionThresholds),
            ("cache", CacheConfig),
            ("runtime", RuntimeConfig),
        )
        for field_name, record_type in record_fields:
            value = getattr(self, field_name)
            if not isinstance(value, record_type):
                raise TypeError(
                    f"{field_name} must be a {record_type.__name__} record"
                )

        for field_name in (
            "seeds",
            "protected_metrics",
            "device_preferences",
            "dtype_preferences",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"{field_name} must be a list or tuple")
            object.__setattr__(self, field_name, tuple(value))

        if not self.seeds:
            raise ValueError("seeds must not be empty")
        for seed in self.seeds:
            _require_int("seeds", seed)

        if not self.protected_metrics:
            raise ValueError("protected_metrics must not be empty")
        protected_metric_names: set[str] = set()
        for index, metric in enumerate(self.protected_metrics):
            if not isinstance(metric, ProtectedMetric):
                raise TypeError(
                    f"protected_metrics[{index}] must be a ProtectedMetric record"
                )
            if metric.name in protected_metric_names:
                raise ValueError(
                    "protected_metrics contains duplicate metric name: "
                    f"{metric.name}"
                )
            protected_metric_names.add(metric.name)

        for preference in self.device_preferences:
            _require_choice("device_preferences", preference, _DEVICES)
        if not self.device_preferences:
            raise ValueError("device_preferences must not be empty")
        for preference in self.dtype_preferences:
            _require_choice("dtype_preferences", preference, _DTYPES)
        if not self.dtype_preferences:
            raise ValueError("dtype_preferences must not be empty")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        """Build an immutable configuration from the complete schema mapping."""
        if not isinstance(raw, Mapping):
            raise TypeError("configuration must be a mapping")

        optional = {"promotion"}
        required = {
            "schema_version",
            "suite_version",
            "backend",
            "qwen",
            "baseline",
            "mechanism",
            "variant",
            "task",
            "seeds",
            "budget",
            "optimizer",
            "schedule",
            "model",
            "lengths",
            "evaluation",
            "thresholds",
            "protected_metrics",
            "device_preferences",
            "dtype_preferences",
            "required_stage",
            "cache",
            "runtime",
        }
        keys = set(raw)
        unknown = keys - required - optional
        missing = required - keys
        if unknown:
            names = ", ".join(sorted(map(str, unknown)))
            raise ValueError(f"unknown top-level configuration keys: {names}")
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"missing top-level configuration keys: {names}")

        qwen = _mapping(raw, "qwen", required=_QWEN_FIELDS)
        task = _mapping(raw, "task", required=_TASK_FIELDS)
        budget = _mapping(raw, "budget", required=_BUDGET_FIELDS)
        optimizer = _mapping(raw, "optimizer", required=_OPTIMIZER_FIELDS)
        schedule = _mapping(raw, "schedule", required=_SCHEDULE_FIELDS)
        model = _mapping(raw, "model", required=_MODEL_FIELDS)
        lengths = _mapping(raw, "lengths", required=_LENGTH_FIELDS)
        evaluation = _mapping(raw, "evaluation", required=_EVALUATION_FIELDS)
        thresholds = _mapping(raw, "thresholds", required=_THRESHOLD_FIELDS)
        cache = _mapping(raw, "cache", required=_CACHE_FIELDS)
        runtime = _mapping(raw, "runtime", required=_RUNTIME_FIELDS)
        promotion_values = (
            _mapping(raw, "promotion", required=_PROMOTION_FIELDS)
            if "promotion" in raw
            else None
        )

        _require_choice("schema_version", raw["schema_version"], {SCHEMA_VERSION})
        _require_choice("suite_version", raw["suite_version"], {SUITE_VERSION})
        _require_choice("backend", raw["backend"], _BACKENDS)
        backend = _BACKEND_ALIASES.get(raw["backend"], raw["backend"])
        _require_choice("baseline", raw["baseline"], _BASELINES)
        _require_choice("mechanism", raw["mechanism"], _MECHANISMS)
        _require_choice("variant", raw["variant"], _VARIANTS)
        _validate_qwen_config(qwen)
        _require_choice("task.name", task["name"], _TASKS)
        if not isinstance(task["params"], Mapping):
            raise TypeError("task.params must be a mapping")
        _require_choice("optimizer.name", optimizer["name"], _OPTIMIZERS)
        _require_choice("schedule.name", schedule["name"], _SCHEDULES)
        _require_string("evaluation.primary_metric", evaluation["primary_metric"])
        _require_choice(
            "evaluation.direction", evaluation["direction"], _DIRECTIONS
        )
        _require_choice("required_stage", raw["required_stage"], _STAGES)
        protected_metrics = _build_protected_metrics(raw["protected_metrics"])
        _require_string_list(
            "device_preferences", raw["device_preferences"], _DEVICES
        )
        _require_string_list(
            "dtype_preferences", raw["dtype_preferences"], _DTYPES
        )
        _require_string("runtime.output_path", runtime["output_path"])

        seeds = _require_list("seeds", raw["seeds"])
        if not seeds:
            raise ValueError("seeds must not be empty")
        for seed in seeds:
            _require_int("seeds", seed)

        promotion = (
            PromotionThresholds(**promotion_values)
            if promotion_values is not None
            else PromotionThresholds()
        )

        return cls(
            schema_version=raw["schema_version"],
            suite_version=raw["suite_version"],
            backend=backend,
            qwen=QwenConfig(**qwen),
            baseline=raw["baseline"],
            mechanism=raw["mechanism"],
            variant=raw["variant"],
            task=TaskConfig(
                name=task["name"],
                params=task["params"],
            ),
            seeds=_freeze(seeds),
            budget=BudgetConfig(**budget),
            optimizer=OptimizerConfig(
                name=optimizer["name"],
                learning_rate=optimizer["learning_rate"],
                betas=_freeze(optimizer["betas"]),
                eps=optimizer["eps"],
                weight_decay=optimizer["weight_decay"],
            ),
            schedule=ScheduleConfig(**schedule),
            model=ModelConfig(**model),
            lengths=LengthConfig(
                curriculum=_freeze(lengths["curriculum"]),
                extrapolation=_freeze(lengths["extrapolation"]),
            ),
            evaluation=EvaluationConfig(**evaluation),
            thresholds=ScientificThresholds(**thresholds),
            promotion=promotion,
            protected_metrics=protected_metrics,
            device_preferences=_freeze(raw["device_preferences"]),
            dtype_preferences=_freeze(raw["dtype_preferences"]),
            required_stage=raw["required_stage"],
            cache=_build_cache_config(
                cache, raw["mechanism"], raw["variant"], lengths
            ),
            runtime=RuntimeConfig(**runtime),
        )

    def semantic_dict(self) -> dict[str, Any]:
        """Return the scientific fields that define experiment identity."""
        return {
            field.name: _plain(getattr(self, field.name))
            for field in fields(self)
            if field.name != "runtime"
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.semantic_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @property
    def experiment_id(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
