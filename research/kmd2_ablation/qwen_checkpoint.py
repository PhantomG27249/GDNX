"""Atomic, strict full-state checkpoints for paired Qwen heal arms."""

from __future__ import annotations

import copy
import math
import os
import random
import tempfile
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Callable

import torch


QWEN_CHECKPOINT_SCHEMA_VERSION = 3
_PAYLOAD_FIELDS = {
    "schema_version",
    "metadata",
    "target_module_names",
    "model_state",
    "tensor_manifest",
    "optimizer_parameter_names",
    "optimizer_state",
    "scheduler_state",
    "rng_state",
    "grad_scaler_state",
    "amplitude_range",
}
_ARMS = {"native", "recency", "surprise"}
_HYBRID_ARMS = {
    "gdn2-mimo-r4-braid-shared-hola-w64",
    "gdn2-mimo-r4-braid-four-state-hola-w64",
}
_HYBRID_PERSISTENT_BUFFER_NAMES = frozenset({
    "components.specialization_probe",
    "components.specialization_value_probe",
    "components.specialization_coefficients",
})
_FROZEN_QWEN_FIELDS = {
    "hidden_size", "linear_num_value_heads", "linear_num_key_heads",
    "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
    "rms_norm_eps", "rms_norm_type", "dtype", "use_cache", "num_hidden_layers",
    "tie_word_embeddings", "rope_theta", "rope_scaling", "max_position_embeddings",
    "partial_rotary_factor",
}


@dataclass(frozen=True)
class QwenHybridCheckpointIdentity:
    architecture_registry_sha256: str
    implementation_sha256: str
    model_tree_sha256: str
    ordered_examples_sha256: str
    pre_replacement_checkpoint_sha256: str
    teacher_sha256: str
    frozen_qwen_config: Mapping[str, object]
    cache_policy: Mapping[str, object]
    trainable_manifest: tuple[Mapping[str, object], ...]
    target_module_names: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("architecture_registry_sha256", "implementation_sha256",
                      "model_tree_sha256", "ordered_examples_sha256",
                      "pre_replacement_checkpoint_sha256", "teacher_sha256"):
            _sha256(field, getattr(self, field))
        for field in ("frozen_qwen_config", "cache_policy"):
            value = _freeze_json(getattr(self, field), field)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{field} must be a nonempty mapping")
            object.__setattr__(self, field, value)
        if set(self.frozen_qwen_config) != _FROZEN_QWEN_FIELDS:
            raise ValueError("frozen_qwen_config fields are incomplete or unknown")
        if self.frozen_qwen_config.get("use_cache") is not False:
            raise ValueError("frozen_qwen_config.use_cache must be false")
        if {key: self.cache_policy.get(key) for key in ("policy", "width", "block_size")} != {
            "policy": "exact_outer", "width": 64, "block_size": 256
        }:
            raise ValueError("cache_policy must be exact_outer W64 C256")
        names = self.target_module_names
        if (type(names) is not tuple or not names or tuple(sorted(names)) != names
                or len(set(names)) != len(names) or any(type(x) is not str or not x for x in names)):
            raise ValueError("target_module_names must be unique canonical nonempty names")
        if type(self.trainable_manifest) is not tuple or not self.trainable_manifest:
            raise TypeError("trainable_manifest must be a tuple")
        seen: set[str] = set()
        normalized = []
        for row in self.trainable_manifest:
            if not isinstance(row, Mapping) or set(row) != {"name", "shape", "dtype"}:
                raise ValueError("trainable_manifest rows require name/shape/dtype")
            name, shape, dtype = row["name"], row["shape"], row["dtype"]
            if type(name) is not str or not name or name in seen:
                raise ValueError("trainable_manifest contains missing or duplicate names")
            seen.add(name)
            if not isinstance(shape, (list, tuple)) or any(type(x) is not int or x < 0 for x in shape):
                raise ValueError("trainable_manifest shape is invalid")
            if type(dtype) is not str or not dtype.startswith("torch."):
                raise ValueError("trainable_manifest dtype is invalid")
            normalized.append(MappingProxyType({"name": name, "shape": tuple(shape), "dtype": dtype}))
        object.__setattr__(self, "trainable_manifest", tuple(normalized))
        if tuple(row["name"] for row in normalized) != tuple(sorted(row["name"] for row in normalized)):
            raise ValueError("trainable_manifest must be sorted by name")

    def as_dict(self) -> dict[str, object]:
        return {field.name: _thaw_json(getattr(self, field.name)) for field in dataclasses.fields(self)}


def _hybrid_tensor_contract(
    *, architecture_arm_id: str, heads: int, key_dim: int, value_dim: int,
    hidden_size: int, conv_kernel: int, dtype: str,
) -> dict[str, tuple[tuple[int, ...], str]]:
    if architecture_arm_id not in _HYBRID_ARMS:
        raise ValueError("unknown hybrid architecture")
    H, dk, dv, hidden, conv_k = heads, key_dim, value_dim, hidden_size, conv_kernel
    if (any(type(value) is not int for value in (H, dk, dv, hidden, conv_k))
            or min(H, dk, dv, hidden, conv_k) < 1 or dk % 2):
        raise ValueError("frozen_qwen_config dimensions are incompatible")
    if dtype not in {"torch.float32", "torch.bfloat16"}:
        raise ValueError("frozen Qwen dtype must be float32 or bfloat16")
    K, V, R = H * dk, H * dv, 4
    shared = architecture_arm_id.endswith("shared-hola-w64")
    compact = not shared and dk % (2 * R) == 0
    component_dk = dk if shared or not compact else dk // R
    component_K = H * component_dk
    shapes = {
        "components.q_weight": (R, component_K, hidden), "components.k_weight": (R, component_K, hidden),
        "components.v_weight": (R, V, hidden), "components.erase_weight": (R, component_K, hidden),
        "components.write_weight": (R, V, hidden), "components.z_weight": (R, V, hidden),
        "components.write_offset": (R, H),
        "components.native_decay_weight": (H, hidden),
        "components.native_A_log": (H,),
        "components.native_dt_bias": (H,),
        "components.phase_proj.weight": ((H * dk // 2 if shared else R * H * component_dk // 2), hidden),
        "components.phase_proj.bias": (H * dk // 2 if shared else R * H * component_dk // 2,),
        "components.output_mixer": ((H, R, dv, dv) if shared else (H, R, R, dv, dv)),
        "components.d_q": (H, R, component_dk), "components.d_k": (H, R, component_dk),
        "components.b_q": (H, R, component_dk), "components.b_k": (H, R, component_dk),
        "components.alpha_q": (H, R), "components.beta_q": (H, R),
        "components.alpha_k": (H, R), "components.beta_k": (H, R),
        "components.cache_gate_logit": (H,), "components.conv1d.weight": (2*K+V, 1, conv_k),
        "components.specialization_probe": (component_K, hidden),
        "components.specialization_value_probe": (V, hidden),
        "components.specialization_coefficients": (R,),
        "components.norm.weight": (dv,), "components.out_proj.weight": (hidden, V),
        "rot_proj.weight": (H * dk // 2, hidden), "rot_proj.bias": (H * dk // 2,),
        "hola.gamma_q": (H, R, component_dk), "hola.gamma_k": (H, R, component_dk), "hola.sink_logit": (H, R),
    }
    if shared:
        shapes.update({"components.native_decay_chan": (H, dk),
                       "components.braid_residual": (H, dk, R),
                       "components.trapezoid_gate": (H, R), "components.lookahead_gate": (H, R),
                       "components.c_logits": (H, R), "components.d_raw": (H, R),
                       "components.braid_router.weight": (H * dk * R, hidden),
                       "components.braid_router.bias": (H * dk * R,),
                       "hola_output_mixer": (H, R, R, dv, dv)})
    else:
        shapes.update({"components.native_decay_pair": (H, R, component_dk // 2),
                       "components.trapezoid_proj.weight": (H * R, hidden),
                       "components.trapezoid_proj.bias": (H * R,)})
    contract = {name: (tuple(shape), dtype) for name, shape in shapes.items()}
    contract["components.specialization_probe"] = ((component_K, hidden), "torch.float32")
    contract["components.specialization_value_probe"] = ((V, hidden), "torch.float32")
    contract["components.specialization_coefficients"] = ((R,), "torch.float32")
    return contract


def hybrid_tensor_element_counts(
    *, architecture_arm_id: str, heads: int, key_dim: int, value_dim: int,
    hidden_size: int, conv_kernel: int,
) -> dict[str, int]:
    """Count hybrid parameters/buffers from the checkpoint's canonical shapes."""
    contract = _hybrid_tensor_contract(
        architecture_arm_id=architecture_arm_id, heads=heads, key_dim=key_dim,
        value_dim=value_dim, hidden_size=hidden_size, conv_kernel=conv_kernel,
        dtype="torch.float32",
    )
    maximum = (1 << 63) - 1
    counts: dict[str, int] = {}
    total_elements = 0
    for name, (shape, _dtype) in contract.items():
        count = math.prod(shape)
        if count > maximum - total_elements:
            raise ValueError("hybrid tensor contract exceeds the exact element-count bound")
        counts[name] = count
        total_elements += count
    buffer_elements = sum(
        count for name, count in counts.items()
        if name in _HYBRID_PERSISTENT_BUFFER_NAMES
    )
    return {
        "parameter_elements": total_elements - buffer_elements,
        "persistent_buffer_elements": buffer_elements,
    }


def expected_hybrid_tensor_contract(identity: QwenHybridCheckpointIdentity,
                                    architecture_arm_id: str) -> dict[str, tuple[tuple[int, ...], str]]:
    """Derive target parameter contracts without constructing a model."""
    c = identity.frozen_qwen_config
    heads = int(c["linear_num_key_heads"])
    if int(c["linear_num_value_heads"]) != heads:
        raise ValueError("frozen_qwen_config dimensions are incompatible")
    return _hybrid_tensor_contract(
        architecture_arm_id=architecture_arm_id,
        heads=heads,
        key_dim=int(c["linear_key_head_dim"]),
        value_dim=int(c["linear_value_head_dim"]),
        hidden_size=int(c["hidden_size"]),
        conv_kernel=int(c["linear_conv_kernel_dim"]),
        dtype=str(c["dtype"]),
    )


class QwenCheckpointError(ValueError):
    """A checkpoint is incomplete, incompatible, corrupt, or unsafe to apply."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _prevalidate_hybrid_source(native: torch.nn.Module) -> None:
    """Validate the complete R1 source before a replacement is allocated."""
    required = ("H", "dk", "dv", "key_dim", "value_dim", "conv_k", "r_out",
                "in_proj_qkv", "in_proj_b", "in_proj_z", "in_proj_a", "conv1d",
                "dt_bias", "A_log", "norm", "out_proj", "rot_proj", "decay_chan", "bw_off")
    missing = tuple(name for name in required if not hasattr(native, name))
    if missing:
        raise QwenCheckpointError("source_tensor_missing", f"native source is missing {missing}")
    if native.r_out != 1:
        raise QwenCheckpointError("source_topology_mismatch", "hybrid conversion requires native R1")
    hidden = native.in_proj_qkv.in_features
    shapes = {
        "in_proj_qkv.weight": (2 * native.key_dim + native.value_dim, hidden),
        "in_proj_b.weight": (native.H, hidden), "in_proj_z.weight": (native.value_dim, hidden),
        "in_proj_a.weight": (native.H, hidden), "out_proj.weight": (hidden, native.value_dim),
        "dt_bias": (native.H,), "A_log": (native.H,), "bw_off": (native.H,),
        "decay_chan": (native.H, native.dk),
        "rot_proj.weight": (native.H * (native.dk // 2), hidden),
        "rot_proj.bias": (native.H * (native.dk // 2),), "norm.weight": (native.dv,),
    }
    state = native.state_dict()
    for name, shape in shapes.items():
        if name not in state:
            raise QwenCheckpointError("source_tensor_missing", f"missing source tensor {name}")
        if tuple(state[name].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    expected = set(shapes) | {"conv1d.weight"}
    if set(state) != expected:
        raise QwenCheckpointError("source_tensor_names", "source checkpoint has missing or unexpected tensors")
    tensors = tuple(state.values())
    if any(not value.is_floating_point() for value in tensors):
        raise QwenCheckpointError("source_tensor_dtype", "source tensors must be floating point")
    if len({value.dtype for value in tensors}) != 1:
        raise QwenCheckpointError("source_tensor_dtype", "source tensors must have one exact dtype")
    if any(not bool(value.detach().isfinite().all()) for value in tensors):
        raise QwenCheckpointError("nonfinite_tensor", "source tensors must be finite")


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_conversion_sha256(model: torch.nn.Module,
                             target_module_names: tuple[str, ...]) -> str:
    modules = dict(model.named_modules())
    rows = []
    for target in target_module_names:
        if target not in modules:
            raise QwenCheckpointError("source_module_missing", f"source module {target!r} is missing")
        rows.append((target, _state_sha256(_cpu_state(dict(modules[target].state_dict())))))
    encoded = json.dumps(rows, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_qwen_architecture_checkpoint(
    model: torch.nn.Module,
    *,
    target_module_names: tuple[str, ...],
    architecture_arm_id: str,
    factory: Callable[[torch.nn.Module], torch.nn.Module] | None = None,
    identity: QwenHybridCheckpointIdentity,
) -> dict[str, object]:
    """Build a portable, source-optimizer-free hybrid conversion checkpoint.

    The source is exhaustively validated before ``factory`` is called.  This is
    intentionally separate from run-resume checkpoints: conversion never
    imports optimizer slots from the source model.
    """
    if architecture_arm_id not in _HYBRID_ARMS:
        raise ValueError("architecture_arm_id is not a dual hybrid package")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(identity, QwenHybridCheckpointIdentity):
        raise TypeError("identity must be QwenHybridCheckpointIdentity")
    if target_module_names != identity.target_module_names:
        raise QwenCheckpointError("target_module_mismatch", "identity target modules differ")
    expected_contract = expected_hybrid_tensor_contract(identity, architecture_arm_id)
    expected_trainables = tuple(sorted(
        ({"name": f"{target}.{name}", "shape": tuple(shape), "dtype": dtype}
         for target in target_module_names
         for name, (shape, dtype) in expected_contract.items()),
        key=lambda row: row["name"],
    ))
    if tuple(dict(row) for row in identity.trainable_manifest) != expected_trainables:
        raise QwenCheckpointError("trainable_manifest_mismatch", "hybrid trainable contract differs")
    modules = dict(model.named_modules())
    if any(name not in modules for name in target_module_names):
        raise QwenCheckpointError("source_module_missing", "a target source module is missing")
    sources = [modules[name] for name in target_module_names]
    # Complete validation of every source precedes the first factory call.
    for source in sources:
        _prevalidate_hybrid_source(source)
    if identity.pre_replacement_checkpoint_sha256 != source_conversion_sha256(model, target_module_names):
        raise QwenCheckpointError("pre_replacement_checkpoint_mismatch",
                                  "identity does not match the aggregate source tensors")
    if factory is None:
        if architecture_arm_id.endswith("shared-hola-w64"):
            from .qwen_hybrid_shared import QwenSharedBraidHybrid
            factory = QwenSharedBraidHybrid.from_native
        else:
            from .qwen_hybrid_four_state import QwenFourStateHybrid
            factory = QwenFourStateHybrid.from_native
    state: dict[str, torch.Tensor] = {}
    layer_manifests: dict[str, object] = {}
    for target, source in zip(target_module_names, sources, strict=True):
        source_state = _cpu_state(dict(source.state_dict()))
        replacement = factory(source)
        if not isinstance(replacement, torch.nn.Module):
            raise TypeError("hybrid factory must return a torch module")
        replacement_state = _cpu_state(dict(replacement.state_dict()))
        actual_contract = {name: (tuple(tensor.shape), str(tensor.dtype))
                           for name, tensor in replacement_state.items()}
        if actual_contract != expected_contract:
            raise QwenCheckpointError("target_tensor_contract_mismatch", f"target contract differs for {target}")
        state.update({f"{target}.{name}": tensor for name, tensor in replacement_state.items()})
        layer_manifests[target] = {
            "source_sha256": _state_sha256(source_state),
            "source_tensors": _tensor_manifest(source_state),
            "target_tensors": _tensor_manifest(replacement_state),
            "transformation": replacement.transformation_manifest(),
        }
    manifest = {"layers": layer_manifests, "history_initialization": "all_lanes_equal",
                "source_optimizer_imported": False, "out_proj": "copied_once",
                "mixers": "uniform_identity"}
    return {
        "schema_version": 1,
        "kind": "qwen_hybrid_architecture_conversion",
        "architecture_arm_id": architecture_arm_id,
        "identity": identity.as_dict(),
        "model_state": state,
        "tensor_manifest": _tensor_manifest(state),
        "conversion_manifest": manifest,
        "optimizer_state": None,
    }


def _encode_hybrid_cache(value: object) -> object:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(value.isfinite().all()):
            raise QwenCheckpointError("nonfinite_cache", "hybrid cache contains a nonfinite tensor")
        return value.detach().to(device="cpu", copy=True).contiguous()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"__cache_type__": type(value).__name__, **{
            field.name: _encode_hybrid_cache(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }}
    raise QwenCheckpointError("cache_state_invalid", f"unsupported cache member {type(value).__name__}")


def _decode_hybrid_cache(value: object, *, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if value is None or type(value) in (bool, int, float, str):
        return value
    if not isinstance(value, Mapping) or "__cache_type__" not in value:
        raise QwenCheckpointError("cache_state_invalid", "hybrid cache record is malformed")
    from .qwen_hybrid_hola import HybridHOLAState
    from .qwen_hybrid_shared import SharedHybridCache
    from .qwen_hybrid_four_state import FourStateHybridCache
    classes = {cls.__name__: cls for cls in (HybridHOLAState, SharedHybridCache, FourStateHybridCache)}
    name = value["__cache_type__"]
    if name not in classes:
        raise QwenCheckpointError("cache_state_invalid", f"unknown hybrid cache type {name!r}")
    return classes[name](**{
        key: _decode_hybrid_cache(item, device=device)
        for key, item in value.items() if key != "__cache_type__"
    })


def _optimizer_state_manifest(state: Mapping[str, object]) -> list[dict[str, object]]:
    groups = []
    raw_state = state["state"]
    if not isinstance(raw_state, Mapping):
        raise QwenCheckpointError("optimizer_state_invalid", "optimizer state must be a mapping")
    state_manifest = []
    for parameter_id, slots in sorted(raw_state.items()):
        if not isinstance(slots, Mapping):
            raise QwenCheckpointError("optimizer_state_invalid", "optimizer slots must be mappings")
        slot_rows = []
        for name, value in sorted(slots.items()):
            if isinstance(value, torch.Tensor):
                slot_rows.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)})
            elif type(value) in (int, float, bool):
                slot_rows.append({"name": name, "scalar_type": type(value).__name__})
            else:
                raise QwenCheckpointError("optimizer_state_invalid", "unsupported optimizer slot type")
        state_manifest.append({"parameter_id": parameter_id, "slots": slot_rows})
    return state_manifest


def _hybrid_optimizer_identity(module: torch.nn.Module,
                               optimizer: torch.optim.Optimizer) -> dict[str, object]:
    names = _optimizer_parameter_names(module, optimizer)
    state = optimizer.state_dict()
    groups = []
    for saved, parameter_names in zip(state["param_groups"], names, strict=True):
        hyperparameters = {key: copy.deepcopy(value) for key, value in saved.items()
                           if key not in {"params", "parameter_names"}}
        groups.append({"parameter_names": parameter_names,
                       "hyperparameters": hyperparameters})
    return {"class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "param_groups": groups, "state_manifest": _optimizer_state_manifest(state)}


def _validate_decoded_hybrid_cache(module: torch.nn.Module, cache: object) -> None:
    from .qwen_hybrid_shared import QwenSharedBraidHybrid, SharedHybridCache
    from .qwen_hybrid_four_state import QwenFourStateHybrid, FourStateHybridCache
    parameter = next(module.parameters())
    if type(module) is QwenFourStateHybrid:
        if type(cache) is not FourStateHybridCache:
            raise QwenCheckpointError("cache_package_mismatch", "four-state package cache type differs")
        hidden = torch.empty(cache.states.shape[0], 0, module.components.hidden,
                             device=parameter.device, dtype=parameter.dtype)
        try:
            module._validate_cache(cache, hidden)
        except (TypeError, ValueError) as error:
            raise QwenCheckpointError("cache_state_invalid", str(error)) from error
        batch = cache.states.shape[0]
    elif type(module) is QwenSharedBraidHybrid:
        if type(cache) is not SharedHybridCache:
            raise QwenCheckpointError("cache_package_mismatch", "shared package cache type differs")
        batch = cache.state.shape[0] if isinstance(cache.state, torch.Tensor) and cache.state.ndim else -1
        hidden = torch.empty(batch, 0, module.components.hidden,
                             device=parameter.device, dtype=parameter.dtype)
        try:
            module._validate_cache(cache, hidden)
        except (TypeError, ValueError) as error:
            raise QwenCheckpointError("cache_state_invalid", str(error)) from error
    else:
        raise QwenCheckpointError("cache_package_mismatch", "module is not a canonical hybrid package")
    if cache.hola_state is None:
        raise QwenCheckpointError("cache_state_invalid", "HOLA state is required on resume")
    try:
        module.hola._validate_state(cache.hola_state, batch, parameter.device)
    except (TypeError, ValueError) as error:
        raise QwenCheckpointError("cache_state_invalid", str(error)) from error


def _validate_saved_hybrid_optimizer(module: torch.nn.Module,
                                     optimizer: torch.optim.Optimizer,
                                     state: object,
                                     identity: Mapping[str, object]) -> None:
    if type(optimizer) not in {torch.optim.Adam, torch.optim.AdamW}:
        raise QwenCheckpointError("optimizer_state_invalid", "hybrid resume supports Adam or AdamW exactly")
    if not isinstance(state, Mapping) or set(state) != {"state", "param_groups"}:
        raise QwenCheckpointError("optimizer_state_invalid", "optimizer state fields are invalid")
    groups, slots = state["param_groups"], state["state"]
    identity_groups = identity.get("param_groups")
    if not isinstance(groups, list) or not isinstance(slots, Mapping) or not isinstance(identity_groups, list):
        raise QwenCheckpointError("optimizer_state_invalid", "optimizer groups or slots are malformed")
    if len(groups) != len(identity_groups) or len(groups) != len(optimizer.param_groups):
        raise QwenCheckpointError("optimizer_parameter_mismatch", "optimizer group count differs")
    current_groups = optimizer.state_dict()["param_groups"]
    named = dict(module.named_parameters()); all_ids: list[int] = []; bindings: dict[int, torch.nn.Parameter] = {}
    group_amsgrad: dict[int, bool] = {}
    for saved_group, identity_group, current_group in zip(
            groups, identity_groups, current_groups, strict=True):
        if not isinstance(saved_group, Mapping) or not isinstance(identity_group, Mapping):
            raise QwenCheckpointError("optimizer_state_invalid", "optimizer group is malformed")
        ids, names = saved_group.get("params"), identity_group.get("parameter_names")
        if not isinstance(ids, list) or not isinstance(names, list) or len(ids) != len(names):
            raise QwenCheckpointError("optimizer_parameter_mismatch", "optimizer ID/name membership differs")
        saved_hyper = {key: value for key, value in saved_group.items()
                       if key not in {"params", "parameter_names"}}
        current_hyper = {key: value for key, value in current_group.items()
                         if key not in {"params", "parameter_names"}}
        identity_hyper = identity_group.get("hyperparameters")
        if (not isinstance(identity_hyper, Mapping)
                or set(saved_hyper) != set(identity_hyper)
                or set(saved_hyper) != set(current_hyper)
                or not _values_equal(saved_hyper, identity_hyper)
                or not _values_equal(saved_hyper, current_hyper)):
            raise QwenCheckpointError("optimizer_state_invalid",
                                      "saved optimizer group hyperparameters differ")
        for parameter_id, name in zip(ids, names, strict=True):
            if type(parameter_id) is not int or parameter_id in bindings or name not in named:
                raise QwenCheckpointError("optimizer_parameter_mismatch", "optimizer parameter binding is invalid")
            bindings[parameter_id] = named[name]; all_ids.append(parameter_id)
            group_amsgrad[parameter_id] = bool(saved_group.get("amsgrad", False))
    if slots and set(slots) != set(all_ids):
        raise QwenCheckpointError("optimizer_state_invalid", "optimizer slots must cover every bound parameter exactly")
    for parameter_id, slot in slots.items():
        parameter = bindings[parameter_id]
        expected = {"step", "exp_avg", "exp_avg_sq"}
        if group_amsgrad[parameter_id]: expected.add("max_exp_avg_sq")
        if not isinstance(slot, Mapping) or set(slot) != expected:
            raise QwenCheckpointError("optimizer_state_invalid", "Adam slot fields are missing or unknown")
        step = slot["step"]
        if (not isinstance(step, torch.Tensor) or step.shape != torch.Size([])
                or not step.is_floating_point() or not bool(step.isfinite().all())):
            raise QwenCheckpointError("optimizer_state_invalid", "Adam step must be a finite floating scalar")
        for field in expected - {"step"}:
            tensor = slot[field]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape
                    or tensor.dtype != parameter.dtype or not bool(tensor.isfinite().all())):
                raise QwenCheckpointError("optimizer_state_invalid",
                                          f"Adam {field} must match its parameter shape/dtype and be finite")


def save_hybrid_resume_checkpoint(path: str | os.PathLike[str], *, module: torch.nn.Module,
                                  cache: object, optimizer: torch.optim.Optimizer,
                                  identity: Mapping[str, object]) -> Path:
    """Atomically save package weights, exact optimizer identity, and all histories."""
    destination = Path(path)
    names = _optimizer_parameter_names(module, optimizer)
    state = _cpu_state(dict(module.state_dict()))
    payload = {"schema_version": 1, "kind": "qwen_hybrid_resume",
               "identity": _thaw_json(_freeze_json(identity, "identity")),
               "model_state": state, "tensor_manifest": _tensor_manifest(state),
               "optimizer_parameter_names": names,
               "optimizer_identity": _hybrid_optimizer_identity(module, optimizer),
               "optimizer_state": _portable_cpu_copy(optimizer.state_dict()),
               "cache_state": _encode_hybrid_cache(cache)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        decoded = torch.load(temporary, map_location="cpu", weights_only=True)
        if not _values_equal(decoded, payload):
            raise QwenCheckpointError("checkpoint_serialization_mismatch", "hybrid resume candidate differs")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_hybrid_resume_checkpoint(path: str | os.PathLike[str], *, module: torch.nn.Module,
                                  optimizer: torch.optim.Optimizer,
                                  identity: Mapping[str, object]) -> object:
    """Prevalidate a hybrid resume and transactionally restore weights/slots/cache."""
    payload = _decode_checkpoint_payload(Path(path))
    fields = {"schema_version", "kind", "identity", "model_state", "tensor_manifest",
              "optimizer_parameter_names", "optimizer_identity", "optimizer_state", "cache_state"}
    if not isinstance(payload, Mapping) or set(payload) != fields or payload["schema_version"] != 1:
        raise QwenCheckpointError("checkpoint_fields_invalid", "hybrid resume fields are invalid")
    expected_identity = _thaw_json(_freeze_json(identity, "identity"))
    if payload["identity"] != expected_identity:
        raise QwenCheckpointError("resume_identity_mismatch", "hybrid resume identity differs")
    current = dict(module.state_dict())
    loaded = payload["model_state"]
    if not isinstance(loaded, Mapping) or tuple(loaded) != tuple(current):
        raise QwenCheckpointError("tensor_name_mismatch", "hybrid tensor names/order differ")
    for name, target in current.items():
        tensor = loaded[name]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != target.shape:
            raise QwenCheckpointError("tensor_shape_mismatch", f"hybrid tensor {name!r} shape differs")
        if tensor.dtype != target.dtype:
            raise QwenCheckpointError("tensor_dtype_mismatch", f"hybrid tensor {name!r} dtype differs")
        if tensor.is_floating_point() and not bool(tensor.isfinite().all()):
            raise QwenCheckpointError("nonfinite_tensor", f"hybrid tensor {name!r} is nonfinite")
    if payload["tensor_manifest"] != _tensor_manifest(loaded):
        raise QwenCheckpointError("tensor_manifest_mismatch", "hybrid tensor manifest is stale")
    if payload["optimizer_parameter_names"] != _optimizer_parameter_names(module, optimizer):
        raise QwenCheckpointError("optimizer_parameter_mismatch", "hybrid optimizer identity differs")
    configured_identity = _hybrid_optimizer_identity(module, optimizer)
    saved_identity = payload["optimizer_identity"]
    if (not isinstance(saved_identity, Mapping)
            or saved_identity.get("class") != configured_identity["class"]
            or saved_identity.get("param_groups") != configured_identity["param_groups"]):
        raise QwenCheckpointError("optimizer_parameter_mismatch", "hybrid optimizer class/groups/hyperparameters differ")
    if saved_identity.get("state_manifest") != _optimizer_state_manifest(payload["optimizer_state"]):
        raise QwenCheckpointError("optimizer_state_invalid", "hybrid optimizer state manifest differs")
    _validate_saved_hybrid_optimizer(module, optimizer, payload["optimizer_state"], saved_identity)
    device = next(module.parameters()).device
    cache = _decode_hybrid_cache(payload["cache_state"], device=device)
    _validate_decoded_hybrid_cache(module, cache)
    model_snapshot = {name: value.detach().clone() for name, value in current.items()}
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
    try:
        module.load_state_dict(loaded, strict=True)
        optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state"]))
        module.last_recurrent_cache = cache
    except BaseException as error:
        module.load_state_dict(model_snapshot, strict=True)
        optimizer.load_state_dict(optimizer_snapshot)
        raise QwenCheckpointError("resume_apply_failed", "hybrid resume rolled back") from error
    return cache


def _sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _freeze_json(value: object, context: str) -> object:
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a nonfinite value")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key in sorted(value):
            if type(key) is not str or not key:
                raise ValueError(f"{context} keys must be nonempty strings")
            frozen[key] = _freeze_json(value[key], f"{context}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_json(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{context} must contain only JSON-compatible values")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _validate_identity_fields(
    *,
    job_id: object,
    pairing_id: object,
    arm: object,
    source_hashes: object,
    data_identity: object,
    example_ids: object,
    promotion_config: object,
) -> tuple[str, str, str, Mapping[str, object], Mapping[str, object], tuple[str, ...], Mapping[str, object]]:
    if type(job_id) is not str or not job_id:
        raise ValueError("job_id must be a nonempty string")
    pairing = _sha256("pairing_id", pairing_id)
    if arm not in _ARMS:
        raise ValueError("arm must be native, recency, or surprise")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("source_hashes must be a nonempty mapping")
    normalized_hashes: dict[str, str] = {}
    for name in sorted(source_hashes):
        if type(name) is not str or not name:
            raise ValueError("source_hashes names must be nonempty strings")
        normalized_hashes[name] = _sha256(f"source_hashes.{name}", source_hashes[name])
    if not isinstance(data_identity, Mapping) or not data_identity:
        raise ValueError("data_identity must be a nonempty mapping")
    frozen_data = _freeze_json(data_identity, "data_identity")
    assert isinstance(frozen_data, Mapping)
    if type(example_ids) is not tuple or not example_ids:
        raise ValueError("example_ids must be a nonempty tuple")
    if any(type(item) is not str or not item for item in example_ids):
        raise ValueError("example_ids must contain nonempty strings")
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("example_ids must not contain duplicates")
    if not isinstance(promotion_config, Mapping) or not promotion_config:
        raise ValueError("promotion_config must be a nonempty mapping")
    frozen_promotion = _freeze_json(promotion_config, "promotion_config")
    assert isinstance(frozen_promotion, Mapping)
    return (
        job_id,
        pairing,
        arm,
        MappingProxyType(normalized_hashes),
        frozen_data,
        example_ids,
        frozen_promotion,
    )


@dataclass(frozen=True)
class QwenCheckpointMetadata:
    """Run identity and progress stored in every complete checkpoint."""

    job_id: str
    pairing_id: str
    arm: str
    step: int
    tokens_seen: int
    source_hashes: Mapping[str, object]
    data_identity: Mapping[str, object]
    example_ids: tuple[str, ...]
    promotion_config: Mapping[str, object]
    architecture_arm_id: str
    architecture_registry_sha256: str
    example_cursor: int = 0
    auxiliary_identity: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identity = _validate_identity_fields(
            job_id=self.job_id,
            pairing_id=self.pairing_id,
            arm=self.arm,
            source_hashes=self.source_hashes,
            data_identity=self.data_identity,
            example_ids=self.example_ids,
            promotion_config=self.promotion_config,
        )
        for name, value in zip(
            (
                "job_id",
                "pairing_id",
                "arm",
                "source_hashes",
                "data_identity",
                "example_ids",
                "promotion_config",
            ),
            identity,
        ):
            object.__setattr__(self, name, value)
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a nonnegative integer")
        if type(self.tokens_seen) is not int or self.tokens_seen < 0:
            raise ValueError("tokens_seen must be a nonnegative integer")
        if (self.step == 0) != (self.tokens_seen == 0):
            raise ValueError("step and tokens_seen must both be zero or both positive")
        if type(self.example_cursor) is not int or self.example_cursor < 0:
            raise ValueError("example_cursor must be a nonnegative integer")
        frozen_auxiliary = _freeze_json(self.auxiliary_identity, "auxiliary_identity")
        assert isinstance(frozen_auxiliary, Mapping)
        object.__setattr__(self, "auxiliary_identity", frozen_auxiliary)
        if type(self.architecture_arm_id) is not str or not self.architecture_arm_id:
            raise ValueError("architecture_arm_id must be a nonempty string")
        if (
            type(self.architecture_registry_sha256) is not str
            or len(self.architecture_registry_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.architecture_registry_sha256)
        ):
            raise ValueError("architecture_registry_sha256 must be lowercase SHA-256")

    def as_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "pairing_id": self.pairing_id,
            "arm": self.arm,
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "source_hashes": _thaw_json(self.source_hashes),
            "data_identity": _thaw_json(self.data_identity),
            "example_ids": list(self.example_ids),
            "promotion_config": _thaw_json(self.promotion_config),
            "architecture_arm_id": self.architecture_arm_id,
            "architecture_registry_sha256": self.architecture_registry_sha256,
            "example_cursor": self.example_cursor,
            "auxiliary_identity": _thaw_json(self.auxiliary_identity),
        }

    @classmethod
    def from_dict(cls, value: object) -> "QwenCheckpointMetadata":
        if not isinstance(value, Mapping):
            raise QwenCheckpointError("metadata_invalid", "metadata must be a mapping")
        expected = {
            "job_id",
            "pairing_id",
            "arm",
            "step",
            "tokens_seen",
            "source_hashes",
            "data_identity",
            "example_ids",
            "promotion_config",
            "architecture_arm_id",
            "architecture_registry_sha256",
            "example_cursor",
            "auxiliary_identity",
        }
        architecture_fields = {
            "architecture_arm_id", "architecture_registry_sha256"
        }
        if not architecture_fields.issubset(value):
            raise QwenCheckpointError(
                "architecture_identity_mismatch",
                "checkpoint architecture identity is missing",
            )
        if set(value) != expected:
            raise QwenCheckpointError(
                "metadata_invalid", "metadata fields are incomplete or unknown"
            )
        try:
            return cls(
                job_id=value["job_id"],
                pairing_id=value["pairing_id"],
                arm=value["arm"],
                step=value["step"],
                tokens_seen=value["tokens_seen"],
                source_hashes=value["source_hashes"],
                data_identity=value["data_identity"],
                example_ids=tuple(value["example_ids"]),
                promotion_config=value["promotion_config"],
                architecture_arm_id=value["architecture_arm_id"],
                architecture_registry_sha256=value["architecture_registry_sha256"],
                example_cursor=value["example_cursor"],
                auxiliary_identity=value["auxiliary_identity"],
            )
        except (TypeError, ValueError) as error:
            raise QwenCheckpointError("metadata_invalid", str(error)) from error


@dataclass(frozen=True)
class QwenResumeExpectation:
    """Immutable run identity expected by a process before it accepts resume."""

    job_id: str
    pairing_id: str
    arm: str
    source_hashes: Mapping[str, object]
    data_identity: Mapping[str, object]
    example_ids: tuple[str, ...]
    promotion_config: Mapping[str, object]
    architecture_arm_id: str
    architecture_registry_sha256: str
    auxiliary_identity: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identity = _validate_identity_fields(
            job_id=self.job_id,
            pairing_id=self.pairing_id,
            arm=self.arm,
            source_hashes=self.source_hashes,
            data_identity=self.data_identity,
            example_ids=self.example_ids,
            promotion_config=self.promotion_config,
        )
        for name, value in zip(
            (
                "job_id",
                "pairing_id",
                "arm",
                "source_hashes",
                "data_identity",
                "example_ids",
                "promotion_config",
            ),
            identity,
        ):
            object.__setattr__(self, name, value)
        if type(self.architecture_arm_id) is not str or not self.architecture_arm_id:
            raise ValueError("architecture_arm_id must be a nonempty string")
        if (
            type(self.architecture_registry_sha256) is not str
            or len(self.architecture_registry_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.architecture_registry_sha256)
        ):
            raise ValueError("architecture_registry_sha256 must be lowercase SHA-256")
        frozen_auxiliary = _freeze_json(self.auxiliary_identity, "auxiliary_identity")
        assert isinstance(frozen_auxiliary, Mapping)
        object.__setattr__(self, "auxiliary_identity", frozen_auxiliary)

    @classmethod
    def from_metadata(cls, metadata: QwenCheckpointMetadata) -> "QwenResumeExpectation":
        if not isinstance(metadata, QwenCheckpointMetadata):
            raise TypeError("metadata must be QwenCheckpointMetadata")
        return cls(
            job_id=metadata.job_id,
            pairing_id=metadata.pairing_id,
            arm=metadata.arm,
            source_hashes=metadata.source_hashes,
            data_identity=metadata.data_identity,
            example_ids=metadata.example_ids,
            promotion_config=metadata.promotion_config,
            architecture_arm_id=metadata.architecture_arm_id,
            architecture_registry_sha256=metadata.architecture_registry_sha256,
            auxiliary_identity=metadata.auxiliary_identity,
        )


@dataclass(frozen=True)
class QwenResumeState:
    job_id: str
    pairing_id: str
    arm: str
    step: int
    tokens_seen: int
    example_cursor: int = 0


def _validate_target_names(
    model: torch.nn.Module, target_module_names: tuple[str, ...]
) -> tuple[str, ...]:
    if type(target_module_names) is not tuple or not target_module_names:
        raise ValueError("target_module_names must be a nonempty tuple")
    if any(type(name) is not str or not name for name in target_module_names):
        raise ValueError("target_module_names must contain nonempty strings")
    if len(set(target_module_names)) != len(target_module_names):
        raise ValueError("target_module_names must not contain duplicates")
    if tuple(sorted(target_module_names)) != target_module_names:
        raise ValueError("target_module_names must be in canonical sorted order")
    for left in target_module_names:
        for right in target_module_names:
            if left != right and right.startswith(left + "."):
                raise ValueError("target_module_names must not overlap")
    modules = dict(model.named_modules())
    missing = [name for name in target_module_names if name not in modules]
    if missing:
        raise KeyError("target modules are missing: " + ", ".join(missing))
    return target_module_names


def _selected_state(
    model: torch.nn.Module, target_module_names: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    targets = _validate_target_names(model, target_module_names)
    state = model.state_dict()
    selected = {
        name: tensor
        for name, tensor in state.items()
        if any(name == target or name.startswith(target + ".") for target in targets)
    }
    for target in targets:
        if not any(name == target or name.startswith(target + ".") for name in selected):
            raise ValueError(f"target module {target!r} has no checkpoint tensors")
    return {name: selected[name] for name in sorted(selected)}


def _cpu_state(selected: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, tensor in selected.items():
        detached = tensor.detach()
        if detached.is_floating_point() and not bool(torch.isfinite(detached).all()):
            raise QwenCheckpointError(
                "nonfinite_tensor", f"target tensor {name!r} is nonfinite"
            )
        result[name] = detached.to(device="cpu", copy=True).contiguous()
    return result


def _portable_cpu_copy(value: object) -> object:
    """Recursively snapshot tensor-bearing runtime state onto portable CPU storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True).contiguous()
    if isinstance(value, Mapping):
        return {key: _portable_cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_portable_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _tensor_manifest(state: Mapping[str, torch.Tensor]) -> list[dict[str, object]]:
    return [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in state.items()
    ]


def _amplitude_range(state: Mapping[str, torch.Tensor]) -> list[float] | None:
    amplitudes = [
        tensor.float().reshape(-1)
        for name, tensor in state.items()
        if name == "cache_amplitude" or name.endswith(".cache_amplitude")
    ]
    if not amplitudes:
        return None
    values = torch.cat(amplitudes)
    if not bool(torch.isfinite(values).all()):
        raise QwenCheckpointError(
            "amplitude_out_of_range", "cache amplitude contains a nonfinite value"
        )
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < 0.0 or maximum > 1.0:
        raise QwenCheckpointError(
            "amplitude_out_of_range",
            f"cache amplitude range [{minimum}, {maximum}] is outside [0,1]",
        )
    return [minimum, maximum]


def _optimizer_parameter_names(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> list[list[str]]:
    by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    seen: set[int] = set()
    groups: list[list[str]] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in by_identity:
                raise QwenCheckpointError(
                    "optimizer_parameter_mismatch",
                    "optimizer references a parameter outside the model",
                )
            if identity in seen:
                raise QwenCheckpointError(
                    "optimizer_parameter_mismatch",
                    "optimizer parameter appears in more than one group",
                )
            seen.add(identity)
            names.append(by_identity[identity])
        declared = group.get("parameter_names")
        if declared is not None and tuple(names) != tuple(declared):
            raise QwenCheckpointError(
                "optimizer_parameter_mismatch",
                "optimizer parameter_names do not match parameter identity/order",
            )
        groups.append(names)
    return groups


def _validate_optimizer_target_coverage(
    optimizer_names: list[list[str]], selected_state: Mapping[str, torch.Tensor]
) -> None:
    missing = sorted(
        name
        for group in optimizer_names
        for name in group
        if name not in selected_state
    )
    if missing:
        raise QwenCheckpointError(
            "optimizer_target_coverage",
            "optimizer-owned parameters are outside checkpoint targets: "
            + ", ".join(missing),
        )


def _rng_state() -> dict[str, object]:
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in cuda],
    }


def _restore_rng(value: Mapping[str, object]) -> None:
    random.setstate(value["python"])
    torch.set_rng_state(value["torch_cpu"])
    cuda = value["torch_cuda"]
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


def _validate_rng(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "python",
        "torch_cpu",
        "torch_cuda",
    }:
        raise QwenCheckpointError("rng_state_invalid", "RNG state fields are invalid")
    cpu = value["torch_cpu"]
    cuda = value["torch_cuda"]
    if not isinstance(cpu, torch.Tensor) or cpu.dtype != torch.uint8 or cpu.ndim != 1:
        raise QwenCheckpointError("rng_state_invalid", "torch CPU RNG state is invalid")
    if not isinstance(cuda, list) or any(
        not isinstance(item, torch.Tensor) or item.dtype != torch.uint8 or item.ndim != 1
        for item in cuda
    ):
        raise QwenCheckpointError("rng_state_invalid", "torch CUDA RNG state is invalid")
    expected_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda) != expected_cuda:
        raise QwenCheckpointError(
            "rng_state_invalid", "CUDA RNG device count does not match this process"
        )
    # Validate the opaque Python state without altering the process state.
    probe = random.Random()
    try:
        probe.setstate(value["python"])
    except (TypeError, ValueError) as error:
        raise QwenCheckpointError("rng_state_invalid", "Python RNG state is invalid") from error
    return {"python": value["python"], "torch_cpu": cpu, "torch_cuda": cuda}


def _payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    metadata: QwenCheckpointMetadata,
    target_module_names: tuple[str, ...],
    grad_scaler: object | None = None,
) -> dict[str, object]:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if getattr(scheduler, "optimizer", None) is not optimizer:
        raise ValueError("scheduler must be bound to the supplied optimizer")
    if not isinstance(metadata, QwenCheckpointMetadata):
        raise TypeError("metadata must be QwenCheckpointMetadata")
    names = _validate_target_names(model, target_module_names)
    state = _cpu_state(_selected_state(model, names))
    optimizer_names = _optimizer_parameter_names(model, optimizer)
    _validate_optimizer_target_coverage(optimizer_names, state)
    optimizer_state = _portable_cpu_copy(optimizer.state_dict())
    scheduler_state = _portable_cpu_copy(scheduler.state_dict())
    validated_optimizer_state = _validate_optimizer_resume_state(
        optimizer_state,
        model=model,
        optimizer=optimizer,
        expected_names=optimizer_names,
        step=metadata.step,
    )
    _validate_scheduler_resume_state(
        scheduler_state,
        scheduler=scheduler,
        optimizer_state=validated_optimizer_state,
        step=metadata.step,
    )
    return {
        "schema_version": QWEN_CHECKPOINT_SCHEMA_VERSION,
        "metadata": metadata.as_dict(),
        "target_module_names": list(names),
        "model_state": state,
        "tensor_manifest": _tensor_manifest(state),
        "optimizer_parameter_names": optimizer_names,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "rng_state": _rng_state(),
        "grad_scaler_state": (None if grad_scaler is None
                              else _portable_cpu_copy(grad_scaler.state_dict())),
        "amplitude_range": _amplitude_range(state),
    }


def save_qwen_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    metadata: QwenCheckpointMetadata,
    target_module_names: tuple[str, ...],
    save_function: Callable[[object, Path], None] | None = None,
    grad_scaler: object | None = None,
) -> Path:
    """Flush a complete checkpoint beside the destination, then atomically replace."""
    try:
        destination = Path(path)
    except TypeError as error:
        raise TypeError("checkpoint path must be path-like") from error
    if not destination.name:
        raise ValueError("checkpoint path must name a file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=target_module_names,
        grad_scaler=grad_scaler,
    )
    writer = torch.save if save_function is None else save_function
    if not callable(writer):
        raise TypeError("save_function must be callable or None")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer_payload = payload if save_function is None else copy.deepcopy(payload)
        writer(writer_payload, temporary)
        del writer_payload
        # Windows requires a writable descriptor for ``fsync``.
        try:
            with temporary.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise QwenCheckpointError(
                "checkpoint_candidate_io_failed",
                "could not flush the serialized checkpoint candidate",
            ) from error
        serialized = _decode_checkpoint_payload(temporary)
        _validate_loaded_payload(
            serialized,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=QwenResumeExpectation.from_metadata(metadata),
            target_module_names=target_module_names,
            grad_scaler=grad_scaler,
        )
        if not _values_equal(serialized, payload):
            raise QwenCheckpointError(
                "checkpoint_serialization_mismatch",
                "serialized checkpoint candidate differs from the validated payload",
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _values_equal(left: object, right: object) -> bool:
    """Exact nested equality that is safe for tensor-bearing state dictionaries."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and left.device == right.device
            and bool(torch.equal(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and tuple(left) == tuple(right)
            and all(_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)  # type: ignore[arg-type]
            and all(
                _values_equal(left_item, right_item)
                for left_item, right_item in zip(left, right)  # type: ignore[arg-type]
            )
        )
    return left == right


def _validate_optimizer_resume_state(
    value: object,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_names: list[list[str]],
    step: int,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"state", "param_groups"}:
        raise QwenCheckpointError(
            "optimizer_state_invalid", "checkpoint optimizer state is malformed"
        )
    if not isinstance(optimizer, torch.optim.AdamW):
        raise QwenCheckpointError(
            "optimizer_state_invalid", "Qwen heal resume requires AdamW"
        )
    saved_groups = value["param_groups"]
    saved_slots = value["state"]
    template = optimizer.state_dict()
    template_groups = template["param_groups"]
    if (
        type(saved_groups) is not list
        or len(saved_groups) != len(expected_names)
        or len(saved_groups) != len(template_groups)
    ):
        raise QwenCheckpointError(
            "optimizer_state_invalid", "optimizer group count is incompatible"
        )

    parameters_by_id: dict[int, torch.nn.Parameter] = {}
    expected_ids: list[int] = []
    for index, (saved, current, live, names) in enumerate(
        zip(
            saved_groups,
            template_groups,
            optimizer.param_groups,
            expected_names,
            strict=True,
        )
    ):
        if type(saved) is not dict or set(saved) != set(current):
            raise QwenCheckpointError(
                "optimizer_state_invalid",
                f"optimizer group {index} fields are incompatible",
            )
        saved_ids = saved.get("params")
        current_ids = current.get("params")
        if saved_ids != current_ids:
            raise QwenCheckpointError(
                "optimizer_parameter_mismatch",
                f"optimizer group {index} parameter IDs/order differ",
            )
        if saved.get("parameter_names") != tuple(names):
            raise QwenCheckpointError(
                "optimizer_parameter_mismatch",
                f"optimizer group {index} parameter-name mapping differs",
            )
        if not isinstance(saved_ids, list) or len(saved_ids) != len(live["params"]):
            raise QwenCheckpointError(
                "optimizer_parameter_mismatch",
                f"optimizer group {index} parameter count differs",
            )
        for parameter_id, parameter in zip(saved_ids, live["params"], strict=True):
            if type(parameter_id) is not int or parameter_id in parameters_by_id:
                raise QwenCheckpointError(
                    "optimizer_parameter_mismatch",
                    "optimizer parameter IDs must be unique integers",
                )
            if not isinstance(parameter, torch.nn.Parameter):
                raise QwenCheckpointError(
                    "optimizer_parameter_mismatch",
                    "optimizer group contains a non-Parameter value",
                )
            parameters_by_id[parameter_id] = parameter
            expected_ids.append(parameter_id)
        for field, current_value in current.items():
            if field in {"params", "lr"}:
                continue
            if not _values_equal(saved[field], current_value):
                raise QwenCheckpointError(
                    "optimizer_state_invalid",
                    f"optimizer group {index} field {field!r} is incompatible",
                )
        learning_rate = saved["lr"]
        if (
            type(learning_rate) not in (int, float)
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) < 0.0
        ):
            raise QwenCheckpointError(
                "optimizer_state_invalid",
                f"optimizer group {index} learning rate is invalid",
            )

    if type(saved_slots) is not dict:
        raise QwenCheckpointError(
            "optimizer_state_invalid", "optimizer slots must be a dictionary"
        )
    expected_slot_ids = set(expected_ids) if step > 0 else set()
    if set(saved_slots) != expected_slot_ids:
        raise QwenCheckpointError(
            "optimizer_state_invalid",
            "optimizer slot parameter IDs do not exactly match active parameters",
        )
    for parameter_id in expected_ids:
        slot = saved_slots[parameter_id]
        parameter = parameters_by_id[parameter_id]
        group = next(
            group for group in saved_groups if parameter_id in group["params"]
        )
        expected_fields = {"step", "exp_avg", "exp_avg_sq"}
        if group["amsgrad"]:
            expected_fields.add("max_exp_avg_sq")
        if type(slot) is not dict or set(slot) != expected_fields:
            raise QwenCheckpointError(
                "optimizer_state_invalid",
                f"AdamW slot {parameter_id} fields are incompatible",
            )
        saved_step = slot["step"]
        if (
            not isinstance(saved_step, torch.Tensor)
            or saved_step.device.type != "cpu"
            or saved_step.shape != torch.Size([])
            or not saved_step.is_floating_point()
            or not bool(torch.isfinite(saved_step).all())
            or float(saved_step) != float(step)
        ):
            raise QwenCheckpointError(
                "optimizer_state_invalid",
                f"AdamW slot {parameter_id} step does not match global progress",
            )
        for field in expected_fields - {"step"}:
            tensor = slot[field]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.shape != parameter.shape
                or tensor.dtype != parameter.dtype
                or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))
            ):
                raise QwenCheckpointError(
                    "optimizer_state_invalid",
                    f"AdamW slot {parameter_id} {field} shape/dtype/finiteness differs",
                )
    return value


def _validate_scheduler_resume_state(
    value: object,
    *,
    scheduler: object,
    optimizer_state: Mapping[str, object],
    step: int,
) -> Mapping[str, object]:
    template = scheduler.state_dict()
    if not isinstance(value, Mapping) or set(value) != set(template):
        raise QwenCheckpointError(
            "scheduler_state_invalid", "checkpoint scheduler fields are incompatible"
        )
    dynamic = {"last_epoch", "_step_count", "_last_lr"}
    for field, current_value in template.items():
        if field not in dynamic and not _values_equal(value[field], current_value):
            raise QwenCheckpointError(
                "scheduler_state_invalid",
                f"checkpoint scheduler field {field!r} is incompatible",
            )
    if value.get("last_epoch") != step or value.get("_step_count") != step + 1:
        raise QwenCheckpointError(
            "scheduler_state_invalid", "checkpoint scheduler progress is inconsistent"
        )
    rates = value.get("_last_lr")
    groups = optimizer_state["param_groups"]
    if (
        type(rates) is not list
        or not isinstance(groups, list)
        or len(rates) != len(groups)
    ):
        raise QwenCheckpointError(
            "scheduler_state_invalid", "checkpoint scheduler LR groups are incompatible"
        )
    base_lrs = getattr(scheduler, "base_lrs", None)
    if (
        type(base_lrs) is not list
        or len(base_lrs) != len(groups)
        or any(
            type(rate) not in (int, float)
            or not math.isfinite(float(rate))
            or float(rate) < 0.0
            for rate in base_lrs
        )
    ):
        raise QwenCheckpointError(
            "scheduler_state_invalid", "configured scheduler base LRs are invalid"
        )
    if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        lambdas = scheduler.lr_lambdas
        if len(lambdas) != len(base_lrs) or any(not callable(item) for item in lambdas):
            raise QwenCheckpointError(
                "scheduler_state_invalid", "configured scheduler lambdas are invalid"
            )
        configured_rates = [
            float(base_rate) * float(multiplier(step))
            for base_rate, multiplier in zip(base_lrs, lambdas, strict=True)
        ]
    else:
        probe = copy.deepcopy(scheduler)
        closed_form = getattr(probe, "_get_closed_form_lr", None)
        if not callable(closed_form):
            raise QwenCheckpointError(
                "scheduler_state_invalid",
                "configured scheduler cannot derive learning rates from progress",
            )
        probe.last_epoch = step
        configured_rates = list(closed_form())
    if (
        len(configured_rates) != len(groups)
        or any(not math.isfinite(rate) or rate < 0.0 for rate in configured_rates)
    ):
        raise QwenCheckpointError(
            "scheduler_state_invalid", "configured scheduler learning rates are invalid"
        )
    for index, (rate, group, base_rate, configured_rate) in enumerate(
        zip(rates, groups, base_lrs, configured_rates, strict=True)
    ):
        if (
            type(rate) not in (int, float)
            or not math.isfinite(float(rate))
            or float(rate) < 0.0
            or not isinstance(group, Mapping)
            or float(rate) != float(group["lr"])
            or float(group.get("initial_lr", -1.0)) != float(base_rate)
            or float(rate) != configured_rate
        ):
            raise QwenCheckpointError(
                "scheduler_state_invalid",
                f"checkpoint scheduler LR for group {index} is inconsistent",
            )
    return value


def _validate_loaded_payload(
    payload: object,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    expectation: QwenResumeExpectation,
    target_module_names: tuple[str, ...],
    grad_scaler: object | None = None,
) -> tuple[QwenCheckpointMetadata, dict[str, torch.Tensor], dict[str, object]]:
    if not isinstance(payload, Mapping) or "schema_version" not in payload:
        raise QwenCheckpointError(
            "checkpoint_fields_invalid", "checkpoint fields are incomplete or unknown"
        )
    if payload["schema_version"] != QWEN_CHECKPOINT_SCHEMA_VERSION:
        raise QwenCheckpointError(
            "checkpoint_schema_mismatch", "checkpoint schema version is incompatible"
        )
    if set(payload) != _PAYLOAD_FIELDS:
        raise QwenCheckpointError(
            "checkpoint_fields_invalid", "checkpoint fields are incomplete or unknown"
        )
    metadata = QwenCheckpointMetadata.from_dict(payload["metadata"])
    identity_fields = (
        "job_id",
        "pairing_id",
        "arm",
        "source_hashes",
        "data_identity",
        "example_ids",
        "promotion_config",
        "auxiliary_identity",
    )
    mismatched = [
        name
        for name in identity_fields
        if getattr(metadata, name) != getattr(expectation, name)
    ]
    if mismatched:
        raise QwenCheckpointError(
            "resume_identity_mismatch",
            "checkpoint identity differs for: " + ", ".join(mismatched),
        )
    architecture_mismatched = [
        name
        for name in ("architecture_arm_id", "architecture_registry_sha256")
        if getattr(metadata, name) != getattr(expectation, name)
    ]
    if architecture_mismatched:
        raise QwenCheckpointError(
            "architecture_identity_mismatch",
            "checkpoint architecture identity differs for: "
            + ", ".join(architecture_mismatched),
        )
    scaler_state = payload["grad_scaler_state"]
    if (scaler_state is None) != (grad_scaler is None):
        raise QwenCheckpointError("grad_scaler_state_mismatch", "checkpoint GradScaler presence differs")
    if scaler_state is not None and not isinstance(scaler_state, Mapping):
        raise QwenCheckpointError("grad_scaler_state_invalid", "checkpoint GradScaler state is malformed")
    names = _validate_target_names(model, target_module_names)
    if payload["target_module_names"] != list(names):
        raise QwenCheckpointError(
            "target_module_mismatch", "checkpoint target modules do not match"
        )
    current = _selected_state(model, names)
    loaded_state = payload["model_state"]
    if not isinstance(loaded_state, Mapping) or tuple(loaded_state) != tuple(current):
        raise QwenCheckpointError(
            "tensor_name_mismatch", "checkpoint tensor names/order do not match"
        )
    validated_state: dict[str, torch.Tensor] = {}
    for name, target in current.items():
        tensor = loaded_state[name]
        if not isinstance(tensor, torch.Tensor):
            raise QwenCheckpointError(
                "tensor_type_mismatch", f"checkpoint value {name!r} is not a tensor"
            )
        if tensor.device.type != "cpu":
            raise QwenCheckpointError(
                "tensor_device_mismatch", f"checkpoint tensor {name!r} is not CPU portable"
            )
        if tensor.shape != target.shape:
            raise QwenCheckpointError(
                "tensor_shape_mismatch", f"checkpoint tensor {name!r} shape differs"
            )
        if tensor.dtype != target.dtype:
            raise QwenCheckpointError(
                "tensor_dtype_mismatch", f"checkpoint tensor {name!r} dtype differs"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise QwenCheckpointError(
                "nonfinite_tensor", f"checkpoint tensor {name!r} is nonfinite"
            )
        validated_state[name] = tensor
    if payload["tensor_manifest"] != _tensor_manifest(validated_state):
        raise QwenCheckpointError(
            "tensor_manifest_mismatch", "checkpoint tensor manifest is stale or corrupt"
        )
    if payload["amplitude_range"] != _amplitude_range(validated_state):
        raise QwenCheckpointError(
            "amplitude_manifest_mismatch", "checkpoint amplitude range is stale"
        )
    current_optimizer_names = _optimizer_parameter_names(model, optimizer)
    _validate_optimizer_target_coverage(current_optimizer_names, current)
    if payload["optimizer_parameter_names"] != current_optimizer_names:
        raise QwenCheckpointError(
            "optimizer_parameter_mismatch", "checkpoint optimizer parameter order differs"
        )
    optimizer_state = _validate_optimizer_resume_state(
        payload["optimizer_state"],
        model=model,
        optimizer=optimizer,
        expected_names=current_optimizer_names,
        step=metadata.step,
    )
    _validate_scheduler_resume_state(
        payload["scheduler_state"],
        scheduler=scheduler,
        optimizer_state=optimizer_state,
        step=metadata.step,
    )
    rng = _validate_rng(payload["rng_state"])
    return metadata, validated_state, rng


def _decode_checkpoint_payload(checkpoint_path: Path) -> object:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise QwenCheckpointError(
            "checkpoint_decode_failed", f"could not decode {checkpoint_path}"
        ) from error


def load_qwen_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    expectation: QwenResumeExpectation,
    target_module_names: tuple[str, ...],
    grad_scaler: object | None = None,
) -> QwenResumeState:
    """Prevalidate every field, then transactionally restore all dynamic state."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if getattr(scheduler, "optimizer", None) is not optimizer:
        raise ValueError("scheduler must be bound to the supplied optimizer")
    if not isinstance(expectation, QwenResumeExpectation):
        raise TypeError("expectation must be QwenResumeExpectation")
    try:
        checkpoint_path = Path(path)
    except TypeError as error:
        raise TypeError("checkpoint path must be path-like") from error
    payload = _decode_checkpoint_payload(checkpoint_path)
    metadata, loaded_state, loaded_rng = _validate_loaded_payload(
        payload,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expectation=expectation,
        target_module_names=target_module_names,
        grad_scaler=grad_scaler,
    )
    current = _selected_state(model, target_module_names)
    model_snapshot = {name: tensor.detach().clone() for name, tensor in current.items()}
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
    scheduler_snapshot = copy.deepcopy(scheduler.state_dict())
    rng_snapshot = _rng_state()
    scaler_snapshot = None if grad_scaler is None else copy.deepcopy(grad_scaler.state_dict())
    try:
        with torch.no_grad():
            for name, target in current.items():
                target.copy_(loaded_state[name].to(device=target.device))
        optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state"]))
        scheduler.load_state_dict(copy.deepcopy(payload["scheduler_state"]))
        _restore_rng(loaded_rng)
        if grad_scaler is not None:
            grad_scaler.load_state_dict(copy.deepcopy(payload["grad_scaler_state"]))
    except BaseException as error:
        with torch.no_grad():
            for name, target in current.items():
                target.copy_(model_snapshot[name])
        optimizer.load_state_dict(optimizer_snapshot)
        scheduler.load_state_dict(scheduler_snapshot)
        _restore_rng(rng_snapshot)
        if grad_scaler is not None:
            grad_scaler.load_state_dict(scaler_snapshot)
        if not isinstance(error, Exception):
            raise
        raise QwenCheckpointError(
            "resume_apply_failed", "checkpoint application failed and was rolled back"
        ) from error
    return QwenResumeState(
        job_id=metadata.job_id,
        pairing_id=metadata.pairing_id,
        arm=metadata.arm,
        step=metadata.step,
        tokens_seen=metadata.tokens_seen,
        example_cursor=metadata.example_cursor,
    )


__all__ = [
    "QWEN_CHECKPOINT_SCHEMA_VERSION",
    "QwenCheckpointError",
    "QwenCheckpointMetadata",
    "QwenHybridCheckpointIdentity",
    "QwenResumeExpectation",
    "QwenResumeState",
    "build_qwen_architecture_checkpoint",
    "expected_hybrid_tensor_contract",
    "hybrid_tensor_element_counts",
    "source_conversion_sha256",
    "save_hybrid_resume_checkpoint",
    "load_hybrid_resume_checkpoint",
    "load_qwen_checkpoint",
    "save_qwen_checkpoint",
]
