"""Deterministic synthetic tasks for the portable KMD-2 ablation suite."""

from __future__ import annotations

import importlib
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor


TASK_SCHEMA_VERSION = "1.0.0"
SUPPORTED_SPLITS = frozenset({"train", "id", "ood_2x", "ood_4x"})
SPLIT_MULTIPLIERS = MappingProxyType(
    {"train": 1, "id": 1, "ood_2x": 2, "ood_4x": 4}
)

_TASK_ENTRY_POINTS = {
    "affine_associative_regression": ("affine", "generate_affine"),
    "drift_reversal": ("dynamics", "generate_drift_reversal"),
    "far_surprise": ("far_surprise", "generate_far_surprise"),
    "freshness": ("freshness", "generate_freshness"),
    "irregular_integration": ("integration", "generate_irregular_integration"),
    "local_binding": ("local_binding", "generate_local_binding"),
    "modular_counter": ("state_tracking", "generate_modular_counter"),
    "mqar": ("mqar", "generate_mqar"),
    "parity": ("state_tracking", "generate_parity"),
    "state_tracking": ("state_tracking", "generate_state_tracking"),
    "structured_exceptions": ("structured", "generate_structured_exceptions"),
    "toggle_fsm": ("state_tracking", "generate_toggle_fsm"),
    "trajectory": ("dynamics", "generate_trajectory"),
}
TASK_NAMES = frozenset(_TASK_ENTRY_POINTS)


class _FrozenJSONDict(dict[str, Any]):
    """A JSON-serializable dict whose mutation methods are disabled."""

    __slots__ = ("_locked",)

    def __init__(self, *args: object, **kwargs: object) -> None:
        if getattr(self, "_locked", False):
            raise TypeError("frozen JSON mappings are immutable")
        dict.__init__(self, *args, **kwargs)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_locked" and not hasattr(self, "_locked"):
            object.__setattr__(self, name, value)
            return
        raise TypeError("frozen JSON mappings are immutable")

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("frozen JSON mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _tensor_storage_key(tensor: Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _ensure_no_tensor_aliases(named_tensors: Sequence[tuple[str, Tensor]]) -> None:
    owners: dict[int, str] = {}
    for name, tensor in named_tensors:
        storage = _tensor_storage_key(tensor)
        previous = owners.get(storage)
        if previous is not None:
            raise ValueError(f"{name} must not alias tensor storage owned by {previous}")
        owners[storage] = name


def _clone_tensor(name: str, value: Any) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise TypeError(f"{name} must use strided tensor layout")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value.detach().clone(memory_format=torch.contiguous_format)


def _freeze_json(value: Any, path: str) -> Any:
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} JSON mapping keys must be strings")
            frozen[key] = _freeze_json(item, f"{path}.{key}")
        return _FrozenJSONDict(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _validate_mask(name: str, value: Tensor, shape: tuple[int, int]) -> Tensor:
    tensor = _clone_tensor(name, value)
    if tensor.dtype != torch.bool or tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must be bool with shape {shape}")
    return tensor


@dataclass(frozen=True)
class EpisodeBatch:
    """Frozen, validated batch/time-leading task data.

    Tensor payloads are detached copies, so callers cannot mutate an episode by
    retaining and editing constructor inputs. Container fields are immutable.
    """

    task: str
    split: str
    seed: int
    example_ids: tuple[str, ...]
    input_ids: Tensor | None
    continuous_inputs: Tensor | None
    direct_factors: Mapping[str, Tensor] | None
    targets: Tensor
    valid: Tensor
    positions: Tensor
    loss_mask: Tensor
    query_mask: Tensor
    boundaries: Tensor
    source_spans: Tensor
    strata: Mapping[str, Tensor]
    metadata: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if type(self.task) is not str or not self.task:
            raise TypeError("task must be a non-empty string")
        if self.task not in TASK_NAMES:
            raise ValueError(f"task must be a registered synthetic task; got {self.task!r}")
        if self.split not in SUPPORTED_SPLITS:
            allowed = ", ".join(sorted(SUPPORTED_SPLITS))
            raise ValueError(f"split must be one of: {allowed}")
        if type(self.seed) is not int:
            raise TypeError("seed must be an int")
        if not isinstance(self.example_ids, (list, tuple)) or not self.example_ids:
            raise ValueError("example_ids must be a non-empty list or tuple")
        example_ids = tuple(self.example_ids)
        if any(type(item) is not str or not item for item in example_ids):
            raise TypeError("example_ids entries must be non-empty strings")
        if len(set(example_ids)) != len(example_ids):
            raise ValueError("example_ids must be unique")

        modalities = (
            self.input_ids is not None,
            self.continuous_inputs is not None,
            self.direct_factors is not None,
        )
        if sum(modalities) != 1:
            raise ValueError("exactly one input modality must be provided")

        raw_tensors: list[tuple[str, Tensor]] = []
        if self.input_ids is not None:
            raw_tensors.append(("input_ids", self.input_ids))
        if self.continuous_inputs is not None:
            raw_tensors.append(("continuous_inputs", self.continuous_inputs))
        raw_tensors.extend(
            (
                ("targets", self.targets),
                ("valid", self.valid),
                ("positions", self.positions),
                ("loss_mask", self.loss_mask),
                ("query_mask", self.query_mask),
                ("boundaries", self.boundaries),
                ("source_spans", self.source_spans),
            )
        )
        if self.direct_factors is not None:
            if not isinstance(self.direct_factors, Mapping) or not self.direct_factors:
                raise ValueError("direct_factors must be a non-empty mapping")
            for key, tensor in self.direct_factors.items():
                if type(key) is not str or not key:
                    raise TypeError("direct_factors keys must be non-empty strings")
                raw_tensors.append((f"direct_factors.{key}", tensor))
        if not isinstance(self.strata, Mapping):
            raise TypeError("strata must be a mapping")
        for key, tensor in self.strata.items():
            if type(key) is not str or not key:
                raise TypeError("strata keys must be non-empty strings")
            raw_tensors.append((f"strata.{key}", tensor))
        _ensure_no_tensor_aliases(raw_tensors)

        input_ids: Tensor | None = None
        continuous_inputs: Tensor | None = None
        direct_factors: Mapping[str, Tensor] | None = None
        if self.input_ids is not None:
            input_ids = _clone_tensor("input_ids", self.input_ids)
            if input_ids.dtype != torch.int64 or input_ids.ndim != 2:
                raise ValueError("input_ids must be int64 with shape [B,T]")
            batch_size, time = input_ids.shape
        elif self.continuous_inputs is not None:
            continuous_inputs = _clone_tensor(
                "continuous_inputs", self.continuous_inputs
            )
            if not continuous_inputs.is_floating_point() or continuous_inputs.ndim != 3:
                raise ValueError(
                    "continuous_inputs must be floating point with shape [B,T,D]"
                )
            batch_size, time = continuous_inputs.shape[:2]
        else:
            assert self.direct_factors is not None
            copied_factors: dict[str, Tensor] = {}
            leading_shape: tuple[int, int] | None = None
            for key, value in self.direct_factors.items():
                copied = _clone_tensor(f"direct_factors.{key}", value)
                if copied.ndim < 2:
                    raise ValueError(
                        f"direct_factors.{key} must have batch/time leading dimensions"
                    )
                current = (copied.shape[0], copied.shape[1])
                if leading_shape is None:
                    leading_shape = current
                elif current != leading_shape:
                    raise ValueError("direct_factors must share [B,T] leading dimensions")
                copied_factors[key] = copied
            assert leading_shape is not None
            batch_size, time = leading_shape
            direct_factors = MappingProxyType(copied_factors)

        shape = (batch_size, time)
        if len(example_ids) != batch_size:
            raise ValueError("example_ids length must equal batch size")

        targets = _clone_tensor("targets", self.targets)
        if targets.ndim == 2:
            if targets.dtype != torch.int64 or tuple(targets.shape) != shape:
                raise ValueError("token/class targets must be int64 with shape [B,T]")
        elif targets.ndim == 3:
            if not targets.is_floating_point() or tuple(targets.shape[:2]) != shape:
                raise ValueError(
                    "regression targets must be floating point with shape [B,T,Dout]"
                )
        else:
            raise ValueError("targets must have shape [B,T] or [B,T,Dout]")

        valid = _validate_mask("valid", self.valid, shape)
        loss_mask = _validate_mask("loss_mask", self.loss_mask, shape)
        query_mask = _validate_mask("query_mask", self.query_mask, shape)
        boundaries = _validate_mask("boundaries", self.boundaries, shape)
        if torch.any(loss_mask & ~query_mask):
            raise ValueError("loss_mask must be a subset of query_mask")
        if torch.any(query_mask & ~valid):
            raise ValueError("query_mask must be a subset of valid")
        if torch.any(boundaries & ~valid):
            raise ValueError("boundaries must be a subset of valid")

        positions = _clone_tensor("positions", self.positions)
        if positions.dtype != torch.int64 or tuple(positions.shape) != shape:
            raise ValueError(f"positions must be int64 with shape {shape}")
        for batch_index in range(batch_size):
            expected = -1
            saw_valid = False
            for token_index in range(time):
                if not bool(valid[batch_index, token_index]):
                    if positions[batch_index, token_index].item() != -1:
                        raise ValueError("positions must be -1 at invalid tokens")
                    continue
                if not saw_valid and not bool(boundaries[batch_index, token_index]):
                    raise ValueError("the first valid token must declare a boundary")
                if bool(boundaries[batch_index, token_index]):
                    expected = 0
                else:
                    expected += 1
                if positions[batch_index, token_index].item() != expected:
                    raise ValueError(
                        "positions must start at zero and increase within each boundary"
                    )
                saw_valid = True

        source_spans = _clone_tensor("source_spans", self.source_spans)
        if source_spans.dtype != torch.int64 or tuple(source_spans.shape) != (
            batch_size,
            time,
            2,
        ):
            raise ValueError(f"source_spans must be int64 with shape {(batch_size, time, 2)}")
        for batch_index in range(batch_size):
            for token_index in range(time):
                start, end = source_spans[batch_index, token_index].tolist()
                if bool(query_mask[batch_index, token_index]):
                    if start < 0 or end <= start or end > token_index:
                        raise ValueError(
                            "source_spans at queries must be causal, non-empty half-open spans"
                        )
                    if not bool(valid[batch_index, start:end].all()):
                        raise ValueError("source_spans must reference valid source tokens")
                elif (start, end) != (-1, -1):
                    raise ValueError("source_spans must use [-1,-1] away from queries")

        if input_ids is not None and torch.any(input_ids[~valid] != 0):
            raise ValueError("input_ids must not leak data at invalid tokens")
        if continuous_inputs is not None and torch.any(continuous_inputs[~valid] != 0):
            raise ValueError("continuous_inputs must not leak data at invalid tokens")
        if targets.ndim == 2:
            if torch.any(targets[~loss_mask] != -100):
                raise ValueError("token targets must be -100 outside loss_mask")
        elif torch.any(targets[~loss_mask] != 0):
            raise ValueError("regression targets must be zero outside loss_mask")
        if direct_factors is not None:
            for key, tensor in direct_factors.items():
                if tensor.is_floating_point() and torch.any(tensor[~valid] != 0):
                    raise ValueError(
                        f"direct_factors.{key} must not leak data at invalid tokens"
                    )

        copied_strata: dict[str, Tensor] = {}
        for key, value in self.strata.items():
            copied = _clone_tensor(f"strata.{key}", value)
            if copied.ndim < 2 or tuple(copied.shape[:2]) != shape:
                raise ValueError(f"strata.{key} must have [B,T] leading dimensions")
            copied_strata[key] = copied

        if not isinstance(self.metadata, (list, tuple)) or len(self.metadata) != batch_size:
            raise ValueError("metadata must contain one mapping per example")
        frozen_metadata: list[Mapping[str, Any]] = []
        for index, item in enumerate(self.metadata):
            if not isinstance(item, Mapping):
                raise TypeError(f"metadata[{index}] must be a JSON mapping")
            frozen = _freeze_json(item, f"metadata[{index}]")
            assert isinstance(frozen, Mapping)
            frozen_metadata.append(MappingProxyType(dict(frozen)))

        object.__setattr__(self, "example_ids", example_ids)
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "continuous_inputs", continuous_inputs)
        object.__setattr__(self, "direct_factors", direct_factors)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "loss_mask", loss_mask)
        object.__setattr__(self, "query_mask", query_mask)
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "source_spans", source_spans)
        object.__setattr__(self, "strata", MappingProxyType(copied_strata))
        object.__setattr__(self, "metadata", tuple(frozen_metadata))


def generate_task(
    name: str,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    params: Mapping[str, Any],
) -> EpisodeBatch:
    """Generate a task batch without touching global random state."""
    if name == "ruler":
        raise ValueError(
            "RULER is a Qwen-only scored task; use the dedicated Task 12 Qwen "
            "RULER generator instead of the synthetic task registry"
        )
    if name not in TASK_NAMES:
        allowed = ", ".join(sorted(TASK_NAMES))
        raise ValueError(f"unknown task {name!r}; available synthetic tasks: {allowed}")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive int")
    if type(length) is not int or length < 1:
        raise ValueError("length must be a positive int")
    if type(seed) is not int:
        raise TypeError("seed must be an int")
    if split not in SUPPORTED_SPLITS:
        allowed = ", ".join(sorted(SUPPORTED_SPLITS))
        raise ValueError(f"split must be one of: {allowed}")
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")

    module_name, function_name = _TASK_ENTRY_POINTS[name]
    try:
        module = importlib.import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as error:
        if error.name == f"{__name__}.{module_name}":
            raise RuntimeError(f"task {name!r} is registered but not implemented") from error
        raise
    generator = getattr(module, function_name)
    return generator(
        batch_size=batch_size,
        length=length,
        seed=seed,
        split=split,
        params=params,
    )


__all__ = [
    "EpisodeBatch",
    "SPLIT_MULTIPLIERS",
    "SUPPORTED_SPLITS",
    "TASK_NAMES",
    "TASK_SCHEMA_VERSION",
    "generate_task",
]


def _canonical_identity_params(value: Any, path: str = "task_params") -> Any:
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} numbers must be finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if type(key) is not str:
                raise TypeError(f"{path} keys must be strings")
            result[key] = _canonical_identity_params(value[key], f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _canonical_identity_params(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _example_identity(
    task: str,
    task_schema_version: str,
    task_params: Mapping[str, Any],
    seed: int,
    split: str,
    logical_length: int,
    index: int,
) -> tuple[str, torch.Generator]:
    """Return a stable identity and isolated CPU generator for one example."""
    if task not in TASK_NAMES:
        raise ValueError(f"task must be registered before deriving identity: {task!r}")
    if type(task_schema_version) is not str or not task_schema_version:
        raise TypeError("task_schema_version must be a non-empty string")
    canonical_params = _canonical_identity_params(task_params)
    domain = json.dumps(
        {
            "index": index,
            "length": logical_length,
            "params": canonical_params,
            "schema": TASK_SCHEMA_VERSION,
            "seed": seed,
            "split": split,
            "task": task,
            "task_schema": task_schema_version,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256(domain).digest()
    identity = digest.hex()
    generator_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(generator_seed)
    return identity, generator


def _operation_count(length: int, split: str) -> int:
    return length * SPLIT_MULTIPLIERS[split]


def _segment_positions(valid: Tensor, boundaries: Tensor) -> Tensor:
    positions = torch.full(valid.shape, -1, dtype=torch.int64)
    for batch_index in range(valid.shape[0]):
        current = -1
        for token_index in range(valid.shape[1]):
            if not bool(valid[batch_index, token_index]):
                continue
            if bool(boundaries[batch_index, token_index]):
                current = 0
            else:
                current += 1
            positions[batch_index, token_index] = current
    return positions
