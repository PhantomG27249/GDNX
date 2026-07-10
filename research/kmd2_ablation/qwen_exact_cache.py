"""Qwen-native KMD-2 exact-cache installation and full-recompute adapter.

This module is intentionally not imported by :mod:`research.kmd2_ablation`.
Importing it opts into the optional Transformers-backed Qwen implementation.
"""

from __future__ import annotations

import copy
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from gdn3.kmd2_native import KMD2NativeAttn

from .config import CacheConfig
from .exact_cache import (
    CacheReadDiagnostics,
    cache_read_blocks,
    initialize_cache_read_parameters,
    merge_persistent_cache,
    reference_scan_with_scores,
)


CACHE_PARAMETER_BASENAMES = (
    "cache_gamma_q",
    "cache_gamma_k",
    "cache_sink_logit",
    "cache_amplitude",
)
CACHE_RESUME_SCHEMA_VERSION = 2
_CACHE_RESUME_FIELDS = {
    "schema_version",
    "job_id",
    "cache_parameter_names",
    "cache_tensors",
    "optimizer_parameter_names",
    "optimizer_state",
    "scheduler_spec",
    "scheduler_state",
}


class FullRecomputeCallError(ValueError):
    """Actionable rejection of an unsupported exact-cache model call."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


_UNSUPPORTED_CALL_FIELDS = {
    "cache_params": (
        "cross_call_cache_unsupported",
        "cross-call cache parameters are not supported",
    ),
    "cache_state": (
        "cross_call_cache_unsupported",
        "cross-call cache state is not supported",
    ),
    "past_state": (
        "cross_call_cache_unsupported",
        "cross-call recurrent state is not supported",
    ),
    "past_key_values": (
        "incremental_decode_unsupported",
        "past_key_values/incremental decode is not supported",
    ),
    "past_key_value": (
        "incremental_decode_unsupported",
        "past_key_value/incremental decode is not supported",
    ),
    "cache_position": (
        "cache_position_unsupported",
        "cache_position/incremental decode is not supported",
    ),
    "cu_seqlens": (
        "packing_unsupported",
        "packed sequence metadata is not supported",
    ),
    "segment_ids": (
        "segments_unsupported",
        "segment metadata is not supported",
    ),
    "sequence_ids": (
        "segments_unsupported",
        "sequence/segment metadata is not supported",
    ),
    "reset_mask": (
        "reset_unsupported",
        "position or state resets are not supported",
    ),
}


def _is_empty_call_field(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        return value.numel() == 0
    if isinstance(value, (tuple, list, dict, set, frozenset)):
        return len(value) == 0
    return False


def _infer_call_shape(
    model_args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[int, int] | None:
    candidates: list[tuple[str, tuple[int, int]]] = []
    if model_args and isinstance(model_args[0], torch.Tensor):
        primary = model_args[0]
        if primary.ndim >= 2:
            candidates.append(("first positional input", (primary.shape[0], primary.shape[1])))
    for name in ("input_ids", "inputs_embeds", "hidden_states"):
        value = kwargs.get(name)
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            candidates.append((name, (value.shape[0], value.shape[1])))
    if not candidates:
        for name in ("attention_mask", "position_ids"):
            value = kwargs.get(name)
            if isinstance(value, torch.Tensor) and value.ndim == 2:
                candidates.append((name, (value.shape[0], value.shape[1])))
                break
    if not candidates:
        return None
    shape = candidates[0][1]
    conflicts = [(name, candidate) for name, candidate in candidates if candidate != shape]
    if conflicts:
        details = ", ".join(f"{name}={candidate}" for name, candidate in conflicts)
        raise FullRecomputeCallError(
            "call_shape_mismatch",
            f"batch/sequence shapes disagree with {candidates[0][0]}={shape}: {details}",
        )
    return shape


def validate_full_recompute_call(*model_args: object, **kwargs: object) -> None:
    """Validate the strict dense, stateless Qwen exact-cache call contract."""
    if len(model_args) > 1:
        raise FullRecomputeCallError(
            "positional_arguments_unsupported",
            "only the primary model input may be positional; pass masks and "
            "position metadata by keyword so they can be validated",
        )
    use_cache = kwargs.get("use_cache", False)
    if type(use_cache) is not bool:
        raise FullRecomputeCallError(
            "use_cache_malformed", "use_cache must be the exact boolean False"
        )
    if use_cache:
        raise FullRecomputeCallError(
            "use_cache_unsupported", "use_cache=True/incremental decode is unsupported"
        )

    for flag, code, description in (
        ("packing", "packing_unsupported", "packed execution"),
        ("decode", "incremental_decode_unsupported", "incremental decode"),
        ("streaming", "incremental_decode_unsupported", "streaming execution"),
    ):
        if flag not in kwargs:
            continue
        value = kwargs[flag]
        if type(value) is not bool:
            raise FullRecomputeCallError(code, f"{flag} must be false or absent")
        if value:
            raise FullRecomputeCallError(code, f"{description} is not supported")

    for name, (code, message) in _UNSUPPORTED_CALL_FIELDS.items():
        if name in kwargs and not _is_empty_call_field(kwargs[name]):
            raise FullRecomputeCallError(code, f"{name}: {message}")

    expected_shape = _infer_call_shape(model_args, kwargs)
    attention_mask = kwargs.get("attention_mask")
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            raise FullRecomputeCallError(
                "attention_mask_shape", "attention_mask must be a [B,T] tensor"
            )
        if attention_mask.ndim != 2 or (
            expected_shape is not None and tuple(attention_mask.shape) != expected_shape
        ):
            raise FullRecomputeCallError(
                "attention_mask_shape",
                f"attention_mask must have exact shape [B,T]={expected_shape}; "
                f"got {tuple(attention_mask.shape)}",
            )
        if attention_mask.dtype == torch.bool:
            all_one = bool(attention_mask.all())
        elif torch.is_floating_point(attention_mask) or attention_mask.dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            all_one = bool((attention_mask == 1).all())
        else:
            raise FullRecomputeCallError(
                "attention_mask_shape", "attention_mask dtype must be bool, integer, or floating"
            )
        if not all_one:
            raise FullRecomputeCallError(
                "padding_unsupported",
                "attention_mask must contain exact ones; padding is unsupported",
            )

    position_ids = kwargs.get("position_ids")
    if position_ids is not None:
        if not isinstance(position_ids, torch.Tensor):
            raise FullRecomputeCallError(
                "position_ids_shape", "position_ids must be a [B,T] tensor"
            )
        if position_ids.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise FullRecomputeCallError(
                "position_ids_dtype", "position_ids must use an integer dtype"
            )
        if position_ids.ndim != 2 or (
            expected_shape is not None and tuple(position_ids.shape) != expected_shape
        ):
            raise FullRecomputeCallError(
                "position_ids_shape",
                f"position_ids must have exact shape [B,T]={expected_shape}; "
                f"got {tuple(position_ids.shape)}",
            )
        positions = position_ids.to(dtype=torch.int64)
        if bool((positions[:, 0] != 0).any()):
            raise FullRecomputeCallError(
                "position_offset", "every position row must begin at zero"
            )
        deltas = positions[:, 1:] - positions[:, :-1]
        reset = (positions[:, 1:] == 0) & (positions[:, :-1] > 0)
        if bool(reset.any()):
            raise FullRecomputeCallError(
                "position_reset", "position resets inside a row are unsupported"
            )
        if bool((deltas == 0).any()):
            raise FullRecomputeCallError(
                "position_duplicate", "duplicate positions are unsupported"
            )
        if bool((deltas < 0).any()):
            raise FullRecomputeCallError(
                "position_decreasing", "decreasing positions are unsupported"
            )
        if bool((deltas > 1).any()):
            raise FullRecomputeCallError(
                "position_gap", "gapped positions are unsupported"
            )


def guarded_model_forward(model: object, *model_args: object, **kwargs: object):
    """Validate a call before invoking the model and force cache output off."""
    validate_full_recompute_call(*model_args, **kwargs)
    kwargs["use_cache"] = False
    return model(*model_args, **kwargs)


@dataclass(frozen=True)
class CompactCacheBlockDiagnostics:
    """Bounded detached metrics for one cache processing block."""

    block_start: int
    block_stop: int
    persistent_selected_positions: torch.Tensor
    top1_positions: torch.Tensor
    attention_entropy: torch.Tensor
    top1_mass: torch.Tensor
    sink_mass: torch.Tensor
    persistent_bytes: int
    block_bytes: int


@dataclass(frozen=True)
class CacheBlockObservation:
    """Ephemeral full local attention passed synchronously to an observer."""

    block_start: int
    block_stop: int
    candidate_positions: torch.Tensor
    candidate_valid: torch.Tensor
    attention_weights: torch.Tensor


@dataclass(frozen=True)
class QwenExactCacheDiagnostics:
    """Compact detached observations from the latest full-recompute scan."""

    update_scores: torch.Tensor
    state_output_norm: torch.Tensor
    cache_output_norm: torch.Tensor
    final_output_norm: torch.Tensor
    blocks: tuple[CompactCacheBlockDiagnostics, ...]
    final_selected_positions: torch.Tensor
    final_selected_scores: torch.Tensor
    final_selected_valid: torch.Tensor
    persistent_bytes: int


def _compact_read_diagnostics(
    diagnostics: CacheReadDiagnostics,
    *,
    block_start: int,
    block_stop: int,
) -> CompactCacheBlockDiagnostics:
    def detached(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().clone()

    return CompactCacheBlockDiagnostics(
        block_start=block_start,
        block_stop=block_stop,
        persistent_selected_positions=detached(
            diagnostics.persistent_selected_positions
        ),
        top1_positions=detached(diagnostics.top1_positions),
        attention_entropy=detached(diagnostics.attention_entropy),
        top1_mass=detached(diagnostics.top1_mass),
        sink_mass=detached(diagnostics.sink_mass),
        persistent_bytes=diagnostics.persistent_bytes,
        block_bytes=diagnostics.block_bytes,
    )


def _validate_model_config(native: KMD2NativeAttn, model_config: object) -> None:
    expected = {
        "hidden_size": native.in_proj_qkv.in_features,
        "linear_num_value_heads": native.H,
        "linear_num_key_heads": native.key_dim // native.dk,
        "linear_key_head_dim": native.dk,
        "linear_value_head_dim": native.dv,
        "linear_conv_kernel_dim": native.conv_k,
    }
    for name, value in expected.items():
        if not hasattr(model_config, name):
            raise TypeError(f"model_config is missing required attribute {name}")
        if getattr(model_config, name) != value:
            raise ValueError(
                f"model_config.{name} does not match the native layer: "
                f"expected {value}, got {getattr(model_config, name)!r}"
            )


class KMD2ExactCacheAttn(KMD2NativeAttn):
    """Identity-gated exact-cache branch around the native scan boundary."""

    @classmethod
    def from_native(
        cls,
        native: KMD2NativeAttn,
        model_config: object,
        cache_config: CacheConfig,
    ) -> "KMD2ExactCacheAttn":
        """Deep-clone an installed native layer and add only cache parameters."""
        if isinstance(native, cls):
            raise ValueError("native layer is already an exact-cache installation")
        if not isinstance(native, KMD2NativeAttn):
            raise TypeError("native must be a KMD2NativeAttn")
        if not isinstance(cache_config, CacheConfig):
            raise TypeError("cache_config must be a CacheConfig")
        if cache_config.score != "exact_outer":
            raise ValueError(
                "KMD2ExactCacheAttn supports only cache.score=exact_outer; "
                f"got {cache_config.score!r}"
            )
        if (
            cache_config.coordinate_frame != "rotated_recurrence"
            or cache_config.pre_rotation_diagnostic
        ):
            raise ValueError(
                "KMD2ExactCacheAttn supports only the rotated_recurrence "
                "coordinate frame with pre_rotation_diagnostic=false"
            )
        _validate_model_config(native, model_config)

        replacement = copy.deepcopy(native)
        replacement.__class__ = cls
        source_parameters = tuple(native.parameters())
        if not source_parameters:
            raise ValueError("native layer has no parameters")
        device = source_parameters[0].device
        read_parameters = initialize_cache_read_parameters(
            key_dim=native.dk,
            heads=native.H,
            device=device,
        )
        replacement.register_parameter("cache_gamma_q", read_parameters.gamma_q)
        replacement.register_parameter("cache_gamma_k", read_parameters.gamma_k)
        replacement.register_parameter(
            "cache_sink_logit", read_parameters.sink_logit
        )
        replacement.register_parameter(
            "cache_amplitude", read_parameters.amplitude
        )
        replacement.cache_config = cache_config
        replacement.last_cache_diagnostics = None
        replacement._cache_diagnostic_observer = None
        return replacement

    def set_cache_diagnostic_observer(self, observer) -> None:
        """Set a synchronous ephemeral full-block observer, or disable it."""
        if observer is not None and not callable(observer):
            raise TypeError("cache diagnostic observer must be callable or None")
        self._cache_diagnostic_observer = observer

    def _native_state_and_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta_e: torch.Tensor,
        beta_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from gdn3 import kmd2_native as native_module

        out_mix = self.out_mix if self.r_out > 1 else None
        if native_module._FAST_SCAN:
            from gdn3.kmd2_fast_scan import scan_with_update_norm

            return scan_with_update_norm(
                q,
                k,
                v,
                g,
                beta_e,
                beta_w,
                out_mix,
            )

        y_state = super()._scan(q, k, v, g, beta_e, beta_w)
        _, update_scores = reference_scan_with_scores(
            q,
            k,
            v,
            g,
            beta_e,
            beta_w,
            out_mix=out_mix,
        )
        return y_state, update_scores

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def _effective_query(self, q: torch.Tensor) -> torch.Tensor:
        if self.r_out == 1:
            return q[:, :, :, 0, :].float()
        return torch.einsum(
            "bthrd,hr->bthd", q.float(), self.out_mix.float()
        )

    def _scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta_e: torch.Tensor,
        beta_w: torch.Tensor,
    ) -> torch.Tensor:
        """Return the unchanged native state read plus a bounded cache read."""
        y_state, update_scores = self._native_state_and_scores(
            q, k, v, g, beta_e, beta_w
        )
        q_eff = self._effective_query(q)
        batch, steps = k.shape[:2]
        positions = torch.arange(
            steps, dtype=torch.int64, device=k.device
        ).view(1, steps).expand(batch, steps)
        valid = torch.ones(batch, steps, dtype=torch.bool, device=k.device)
        storage_dtype = (
            torch.float32
            if self.cache_config.storage_dtype == "fp32"
            else torch.bfloat16
        )

        state = None
        cache_outputs: list[torch.Tensor] = []
        block_diagnostics: list[CompactCacheBlockDiagnostics] = []
        for block_start in range(0, steps, self.cache_config.block_size):
            block_stop = min(steps, block_start + self.cache_config.block_size)
            block_slice = slice(block_start, block_stop)
            block_output, diagnostics = cache_read_blocks(
                q_eff=q_eff[:, block_slice],
                query_positions=positions[:, block_slice],
                state=state,
                block_k=k[:, block_slice],
                block_v=v[:, block_slice],
                block_scores=update_scores[:, block_slice],
                block_positions=positions[:, block_slice],
                block_valid=valid[:, block_slice],
                config=self.cache_config,
                gamma_q=self.cache_gamma_q,
                gamma_k=self.cache_gamma_k,
                sink_logit=self.cache_sink_logit,
            )
            cache_outputs.append(block_output)
            observer = self._cache_diagnostic_observer
            if observer is not None:
                observer(
                    CacheBlockObservation(
                        block_start=block_start,
                        block_stop=block_stop,
                        candidate_positions=diagnostics.hit_ready_positions.detach(),
                        candidate_valid=diagnostics.candidate_valid.detach(),
                        attention_weights=diagnostics.attention_weights.detach(),
                    )
                )
            block_diagnostics.append(
                _compact_read_diagnostics(
                    diagnostics,
                    block_start=block_start,
                    block_stop=block_stop,
                )
            )
            state = merge_persistent_cache(
                state=state,
                block_k=k[:, block_slice],
                block_v=v[:, block_slice],
                block_scores=update_scores[:, block_slice],
                block_positions=positions[:, block_slice],
                block_valid=valid[:, block_slice],
                width=self.cache_config.width,
                storage_dtype=storage_dtype,
            )

        y_cache = torch.cat(cache_outputs, dim=1)
        combined = y_state + (
            self.cache_amplitude.view(1, 1, self.H, 1) * y_cache
        )
        assert state is not None
        self.last_cache_diagnostics = QwenExactCacheDiagnostics(
            update_scores=update_scores.detach().clone(),
            state_output_norm=torch.linalg.vector_norm(
                y_state.float(), dim=-1
            ).detach(),
            cache_output_norm=torch.linalg.vector_norm(
                y_cache.float(), dim=-1
            ).detach(),
            final_output_norm=torch.linalg.vector_norm(
                combined.float(), dim=-1
            ).detach(),
            blocks=tuple(block_diagnostics),
            final_selected_positions=state.positions.detach().clone(),
            final_selected_scores=state.scores.detach().clone(),
            final_selected_valid=state.valid.detach().clone(),
            persistent_bytes=state.nbytes,
        )
        return combined


def _checkpoint_tensor_mapping(
    checkpoint: Mapping[str, object] | str | os.PathLike[str] | None,
) -> dict[str, torch.Tensor]:
    if checkpoint is None:
        return {}
    loaded: object
    if isinstance(checkpoint, (str, os.PathLike)):
        loaded = torch.load(
            Path(checkpoint),
            map_location="cpu",
            weights_only=True,
        )
    else:
        loaded = checkpoint
    if not isinstance(loaded, Mapping):
        raise TypeError("native checkpoint must be a tensor mapping or path")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
        if not isinstance(loaded, Mapping):
            raise TypeError("native checkpoint state_dict must be a mapping")
    result: dict[str, torch.Tensor] = {}
    for name, tensor in loaded.items():
        if type(name) is not str or not name:
            raise TypeError("native checkpoint names must be nonempty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"native checkpoint tensor {name!r} is not a tensor")
        result[name] = tensor
    return result


def _validate_upgraded_indices(model: object, manager: object, raw: object) -> tuple[int, ...]:
    if not isinstance(raw, (tuple, list)):
        raise TypeError("upgrade manager must return a list of layer indices")
    indices = tuple(raw)
    if not indices:
        raise ValueError("upgrade manager must install at least one native layer")
    if any(type(index) is not int for index in indices):
        raise TypeError("upgrade manager layer indices must be exact integers")
    if len(set(indices)) != len(indices):
        raise ValueError("upgrade manager returned duplicate layer indices")
    try:
        layers = model.model.layers
    except AttributeError as error:
        raise TypeError("model must expose model.layers") from error
    if any(index < 0 or index >= len(layers) for index in indices):
        raise ValueError("upgrade manager returned an out-of-range layer index")
    declared = getattr(manager, "upgraded_layers", None)
    if declared is not None and tuple(declared) != indices:
        raise ValueError(
            "upgrade manager returned indices that disagree with upgraded_layers"
        )
    return indices


def _module_name(model: torch.nn.Module, target: torch.nn.Module) -> str:
    matches = [name for name, module in model.named_modules() if module is target]
    if len(matches) != 1 or not matches[0]:
        raise ValueError("native checkpoint target module has no unique model name")
    return matches[0]


def _verify_replacement(
    source: KMD2NativeAttn,
    replacement: KMD2ExactCacheAttn,
) -> None:
    source_parameters = dict(source.named_parameters())
    replacement_parameters = dict(replacement.named_parameters())
    inherited_parameters = {
        name: parameter
        for name, parameter in replacement_parameters.items()
        if name.rsplit(".", 1)[-1] not in CACHE_PARAMETER_BASENAMES
    }
    if tuple(inherited_parameters) != tuple(source_parameters):
        raise RuntimeError("exact-cache replacement inherited parameter names differ")
    cache_names = set(replacement_parameters) - set(source_parameters)
    if cache_names != set(CACHE_PARAMETER_BASENAMES):
        raise RuntimeError("exact-cache replacement initialized undeclared parameters")
    for name, source_parameter in source_parameters.items():
        target_parameter = inherited_parameters[name]
        if (
            source_parameter.shape != target_parameter.shape
            or source_parameter.dtype != target_parameter.dtype
            or source_parameter.device != target_parameter.device
            or source_parameter.requires_grad != target_parameter.requires_grad
            or not torch.equal(source_parameter, target_parameter)
            or source_parameter.data_ptr() == target_parameter.data_ptr()
        ):
            raise RuntimeError(
                f"exact-cache replacement did not strictly clone parameter {name}"
            )
    source_buffers = dict(source.named_buffers())
    replacement_buffers = dict(replacement.named_buffers())
    if tuple(replacement_buffers) != tuple(source_buffers):
        raise RuntimeError("exact-cache replacement inherited buffer names differ")
    for name, source_buffer in source_buffers.items():
        target_buffer = replacement_buffers[name]
        if (
            source_buffer.shape != target_buffer.shape
            or source_buffer.dtype != target_buffer.dtype
            or source_buffer.device != target_buffer.device
            or not torch.equal(source_buffer, target_buffer)
            or source_buffer.data_ptr() == target_buffer.data_ptr()
        ):
            raise RuntimeError(
                f"exact-cache replacement did not strictly clone buffer {name}"
            )
    if (
        replacement.layer_idx != source.layer_idx
        or replacement.r_out != source.r_out
        or replacement.training != source.training
    ):
        raise RuntimeError("exact-cache replacement changed native layer attributes")


def named_cache_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    """Return exact-cache parameters in stable fully-qualified name order."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    items: list[tuple[str, torch.nn.Parameter]] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, KMD2ExactCacheAttn):
            continue
        for basename in CACHE_PARAMETER_BASENAMES:
            parameter = getattr(module, basename, None)
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError(
                    f"exact-cache module {module_name!r} is missing parameter {basename}"
                )
            name = f"{module_name}.{basename}" if module_name else basename
            items.append((name, parameter))
    items.sort(key=lambda item: item[0])
    if not items:
        raise ValueError("model has no exact-cache parameters")
    names = tuple(name for name, _ in items)
    if len(set(names)) != len(names):
        raise ValueError("model has duplicate exact-cache parameter names")
    for name, parameter in items:
        if parameter.dtype != torch.float32:
            raise TypeError(f"cache parameter {name} must remain fp32")
    return tuple(items)


def cache_parameter_group(
    model: torch.nn.Module,
    cache_config: CacheConfig,
    *,
    betas: tuple[float, float],
    eps: float,
) -> dict[str, object]:
    """Return the stable cache group for a shared memory/cache AdamW."""
    if not isinstance(cache_config, CacheConfig):
        raise TypeError("cache_config must be a CacheConfig")
    if (
        type(betas) is not tuple
        or len(betas) != 2
        or any(type(value) not in (float, int) for value in betas)
        or any(not 0.0 <= float(value) < 1.0 for value in betas)
    ):
        raise ValueError("betas must be two finite values in [0,1)")
    if type(eps) not in (float, int) or not 0.0 < float(eps) < float("inf"):
        raise ValueError("eps must be finite and positive")
    named = named_cache_parameters(model)
    return {
        "name": "cache",
        "params": [parameter for _, parameter in named],
        "lr": cache_config.lr_cache,
        "betas": (float(betas[0]), float(betas[1])),
        "eps": float(eps),
        "weight_decay": 0.0,
    }


def register_cache_amplitude_projection(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
):
    """Install the post-step cache-amplitude clamp on an existing optimizer."""
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if hasattr(optimizer, "_kmd2_amplitude_projection_hook_handle"):
        raise ValueError("cache amplitude projection is already registered")
    named = named_cache_parameters(model)
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    amplitudes = tuple(
        parameter
        for name, parameter in named
        if name.endswith(".cache_amplitude") or name == "cache_amplitude"
    )
    missing = [
        name
        for name, parameter in named
        if (name.endswith(".cache_amplitude") or name == "cache_amplitude")
        and id(parameter) not in optimizer_parameter_ids
    ]
    if missing:
        raise ValueError(
            "optimizer is missing cache amplitude parameters: " + ", ".join(missing)
        )

    def project_amplitudes(
        _optimizer: torch.optim.Optimizer,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
    ) -> None:
        with torch.no_grad():
            for amplitude in amplitudes:
                amplitude.clamp_(0.0, 1.0)

    handle = optimizer.register_step_post_hook(project_amplitudes)
    optimizer._kmd2_amplitude_projection_hook_handle = handle
    return handle


def build_cache_optimizer(
    model: torch.nn.Module,
    cache_config: CacheConfig,
    *,
    betas: tuple[float, float],
    eps: float,
    scheduler_factory=None,
    scheduler_spec: object | None = None,
) -> torch.optim.AdamW:
    """Build the dedicated zero-decay cache AdamW group and projection hook."""
    named = named_cache_parameters(model)
    optimizer = torch.optim.AdamW(
        [cache_parameter_group(model, cache_config, betas=betas, eps=eps)]
    )
    register_cache_amplitude_projection(optimizer, model)
    optimizer._kmd2_cache_parameter_names = tuple(name for name, _ in named)
    if scheduler_factory is None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _step: 1.0
        )
    else:
        if not callable(scheduler_factory):
            raise TypeError("scheduler_factory must be callable or None")
        scheduler = scheduler_factory(optimizer)
    if getattr(scheduler, "optimizer", None) is not optimizer:
        raise ValueError(
            "scheduler_factory must return a scheduler bound to the new cache optimizer"
        )
    optimizer._kmd2_shared_scheduler = scheduler
    optimizer._kmd2_scheduler_spec = copy.deepcopy(scheduler_spec)
    return optimizer


def _optimizer_parameter_names(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    context: str,
) -> list[list[str]]:
    named = named_cache_parameters(model)
    by_identity = {id(parameter): name for name, parameter in named}
    groups: list[list[str]] = []
    seen: list[str] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            name = by_identity.get(id(parameter))
            if name is None:
                raise ValueError(
                    f"{context} optimizer contains a non-cache parameter"
                )
            names.append(name)
            seen.append(name)
        groups.append(names)
    expected = [name for name, _ in named]
    if seen != expected:
        raise ValueError(
            f"{context} optimizer parameter order does not exactly match cache names"
        )
    return groups


def _clone_resume_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_resume_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_resume_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_resume_value(item) for item in value)
    return copy.deepcopy(value)


def _bound_cache_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    context: str,
):
    scheduler = getattr(optimizer, "_kmd2_shared_scheduler", None)
    if scheduler is None or getattr(scheduler, "optimizer", None) is not optimizer:
        raise ValueError(
            f"{context} optimizer must own an actual bound cache scheduler"
        )
    if not callable(getattr(scheduler, "state_dict", None)) or not callable(
        getattr(scheduler, "load_state_dict", None)
    ):
        raise TypeError(f"{context} cache scheduler must support state_dict/load_state_dict")
    return scheduler


def build_cache_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    job_id: str,
) -> dict[str, object]:
    """Build the versioned cache-only model/optimizer resume envelope."""
    if type(job_id) is not str or not job_id:
        raise ValueError("resume job_id must be a nonempty string")
    named = named_cache_parameters(model)
    if optimizer is None:
        optimizer_names: list[list[str]] = []
        optimizer_state = None
        scheduler_spec = None
        scheduler_state = None
    else:
        optimizer_names = _optimizer_parameter_names(
            model, optimizer, context="resume"
        )
        optimizer_state = _clone_resume_value(optimizer.state_dict())
        scheduler_spec = _clone_resume_value(
            getattr(optimizer, "_kmd2_scheduler_spec", None)
        )
        scheduler = _bound_cache_scheduler(optimizer, context="resume")
        scheduler_state = _clone_resume_value(scheduler.state_dict())
    return {
        "schema_version": CACHE_RESUME_SCHEMA_VERSION,
        "job_id": job_id,
        "cache_parameter_names": [name for name, _ in named],
        "cache_tensors": [
            parameter.detach().cpu().clone() for _, parameter in named
        ],
        "optimizer_parameter_names": optimizer_names,
        "optimizer_state": optimizer_state,
        "scheduler_spec": scheduler_spec,
        "scheduler_state": scheduler_state,
    }


def save_cache_resume(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    job_id: str,
) -> None:
    """Atomically write a cache resume on the destination filesystem."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    envelope = build_cache_resume(model, optimizer, job_id=job_id)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(envelope, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _load_cache_resume(checkpoint: object) -> Mapping[str, object]:
    loaded: object
    if isinstance(checkpoint, (str, os.PathLike)):
        loaded = torch.load(
            Path(checkpoint), map_location="cpu", weights_only=True
        )
    else:
        loaded = checkpoint
    if not isinstance(loaded, Mapping):
        raise TypeError("resume checkpoint must be a mapping or path")
    return loaded


def _validate_optimizer_resume_state(
    optimizer_state: object,
    optimizer_parameter_names: list[list[str]],
    model_parameters: dict[str, torch.nn.Parameter],
) -> None:
    if not isinstance(optimizer_state, Mapping):
        raise TypeError("resume optimizer_state must be a mapping")
    if set(optimizer_state) != {"state", "param_groups"}:
        raise KeyError("resume optimizer_state has missing or unexpected fields")
    state = optimizer_state["state"]
    groups = optimizer_state["param_groups"]
    if not isinstance(state, Mapping) or not isinstance(groups, list):
        raise TypeError("resume optimizer state/groups have invalid types")
    if len(groups) != len(optimizer_parameter_names):
        raise ValueError("resume optimizer group count mismatch")
    id_to_parameter: dict[object, torch.nn.Parameter] = {}
    id_to_expected_fields: dict[object, set[str]] = {}
    referenced_ids: list[object] = []
    for saved_group, names in zip(groups, optimizer_parameter_names):
        if not isinstance(saved_group, Mapping):
            raise TypeError("resume optimizer param_group must be a mapping")
        amsgrad = saved_group.get("amsgrad", False)
        if type(amsgrad) is not bool:
            raise TypeError("resume optimizer amsgrad must be a boolean")
        parameter_ids = saved_group.get("params")
        if not isinstance(parameter_ids, list) or len(parameter_ids) != len(names):
            raise ValueError("resume optimizer param_group size mismatch")
        for parameter_id, name in zip(parameter_ids, names):
            if parameter_id in id_to_parameter:
                raise ValueError("resume optimizer repeats a parameter id")
            id_to_parameter[parameter_id] = model_parameters[name]
            expected_fields = {"step", "exp_avg", "exp_avg_sq"}
            if amsgrad:
                expected_fields.add("max_exp_avg_sq")
            id_to_expected_fields[parameter_id] = expected_fields
            referenced_ids.append(parameter_id)
    if not set(state).issubset(set(referenced_ids)):
        raise ValueError("resume optimizer state contains an unexpected parameter id")
    for parameter_id, parameter_state in state.items():
        if not isinstance(parameter_state, Mapping):
            raise TypeError("resume optimizer per-parameter state must be a mapping")
        if set(parameter_state) != id_to_expected_fields[parameter_id]:
            raise KeyError(
                "resume optimizer per-parameter AdamW state has missing or "
                "unexpected fields"
            )
        parameter = id_to_parameter[parameter_id]
        for state_name, value in parameter_state.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"resume optimizer state {state_name} must be a tensor"
                )
            if not bool(torch.isfinite(value.detach()).all()):
                raise ValueError(
                    f"resume optimizer state {state_name} is nonfinite"
                )
            if state_name == "step":
                if value.ndim != 0 or value.dtype not in {
                    torch.float32,
                    torch.float64,
                }:
                    raise TypeError(
                        "resume optimizer state step must be a scalar float32/float64 tensor"
                    )
                continue
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"resume optimizer state {state_name} shape mismatch"
                )
            if value.dtype != parameter.dtype:
                raise TypeError(
                    f"resume optimizer state {state_name} dtype mismatch"
                )


def _validate_scheduler_structure(
    saved: object,
    current: object,
    *,
    path: str,
) -> None:
    if isinstance(current, Mapping):
        if not isinstance(saved, Mapping) or set(saved) != set(current):
            raise KeyError(f"resume scheduler structure mismatch at {path}")
        for key in current:
            _validate_scheduler_structure(
                saved[key], current[key], path=f"{path}.{key}"
            )
        return
    if isinstance(current, (list, tuple)):
        if type(saved) is not type(current) or len(saved) != len(current):
            raise ValueError(f"resume scheduler sequence mismatch at {path}")
        for index, (saved_item, current_item) in enumerate(zip(saved, current)):
            _validate_scheduler_structure(
                saved_item,
                current_item,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(current, torch.Tensor):
        if not isinstance(saved, torch.Tensor):
            raise TypeError(f"resume scheduler tensor missing at {path}")
        if saved.shape != current.shape or saved.dtype != current.dtype:
            raise TypeError(f"resume scheduler tensor shape/dtype mismatch at {path}")
        if not bool(torch.isfinite(saved.detach()).all()):
            raise ValueError(f"resume scheduler tensor is nonfinite at {path}")
        return
    if current is None:
        if saved is not None:
            raise TypeError(f"resume scheduler expected None at {path}")
        return
    if type(saved) is not type(current):
        raise TypeError(f"resume scheduler value type mismatch at {path}")
    if isinstance(saved, float) and not math.isfinite(saved):
        raise ValueError(f"resume scheduler value is nonfinite at {path}")


def _validate_scheduler_resume_state(
    scheduler_state: object,
    scheduler: object,
    optimizer_state: Mapping[str, object],
) -> None:
    if not isinstance(scheduler_state, Mapping):
        raise TypeError("resume scheduler_state must be a mapping")
    current_state = scheduler.state_dict()
    _validate_scheduler_structure(
        scheduler_state,
        current_state,
        path="scheduler_state",
    )
    last_epoch = scheduler_state.get("last_epoch")
    if type(last_epoch) is not int or last_epoch < -1:
        raise ValueError("resume scheduler last_epoch must be an integer >= -1")
    saved_groups = optimizer_state["param_groups"]
    last_lrs = scheduler_state.get("_last_lr")
    base_lrs = scheduler_state.get("base_lrs")
    if not isinstance(last_lrs, list) or len(last_lrs) != len(saved_groups):
        raise ValueError("resume scheduler _last_lr/group count mismatch")
    if not isinstance(base_lrs, list) or len(base_lrs) != len(saved_groups):
        raise ValueError("resume scheduler base_lrs/group count mismatch")
    for index, (last_lr, base_lr, group) in enumerate(
        zip(last_lrs, base_lrs, saved_groups)
    ):
        if last_lr != group.get("lr"):
            raise ValueError(
                f"resume scheduler _last_lr disagrees with optimizer group {index}"
            )
        initial_lr = group.get("initial_lr", base_lr)
        if base_lr != initial_lr:
            raise ValueError(
                f"resume scheduler base_lr disagrees with optimizer group {index}"
            )


def strict_load_cache_resume(
    model: torch.nn.Module,
    checkpoint: object,
    expected_job_id: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Strictly and atomically restore cache parameters and optimizer state."""
    if type(expected_job_id) is not str or not expected_job_id:
        raise ValueError("resume expected_job_id must be a nonempty string")
    envelope = _load_cache_resume(checkpoint)
    keys = set(envelope)
    if keys != _CACHE_RESUME_FIELDS:
        missing = sorted(_CACHE_RESUME_FIELDS - keys)
        unexpected = sorted(keys - _CACHE_RESUME_FIELDS)
        raise KeyError(
            f"resume envelope fields mismatch; missing={missing}, unexpected={unexpected}"
        )
    if envelope["schema_version"] != CACHE_RESUME_SCHEMA_VERSION:
        raise ValueError("resume schema_version mismatch")
    if envelope["job_id"] != expected_job_id:
        raise ValueError("resume job_id mismatch")

    named = named_cache_parameters(model)
    expected_names = [name for name, _ in named]
    saved_names = envelope["cache_parameter_names"]
    saved_tensors = envelope["cache_tensors"]
    if not isinstance(saved_names, list) or any(
        type(name) is not str for name in saved_names
    ):
        raise TypeError("resume cache_parameter_names must be a string list")
    if saved_names != expected_names:
        raise ValueError("resume cache parameter names/order mismatch")
    if not isinstance(saved_tensors, list) or len(saved_tensors) != len(named):
        raise ValueError("resume cache tensor count mismatch")
    for (name, parameter), tensor in zip(named, saved_tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"resume cache tensor {name} is not a tensor")
        if tuple(tensor.shape) != tuple(parameter.shape):
            raise ValueError(f"resume cache tensor {name} shape mismatch")
        if tensor.dtype != parameter.dtype:
            raise TypeError(f"resume cache tensor {name} dtype mismatch")
        if not bool(torch.isfinite(tensor.detach()).all()):
            raise ValueError(f"resume cache tensor {name} is nonfinite")
        if name.endswith(".cache_amplitude") and not bool(
            ((tensor >= 0.0) & (tensor <= 1.0)).all()
        ):
            raise ValueError(f"resume cache amplitude {name} is out of range")

    optimizer_names = envelope["optimizer_parameter_names"]
    if not isinstance(optimizer_names, list) or any(
        not isinstance(group, list)
        or any(type(name) is not str for name in group)
        for group in optimizer_names
    ):
        raise TypeError("resume optimizer_parameter_names must be nested lists")
    optimizer_state = envelope["optimizer_state"]
    scheduler_spec = envelope["scheduler_spec"]
    scheduler_state = envelope["scheduler_state"]
    scheduler = None
    if optimizer is None:
        if (
            optimizer_names
            or optimizer_state is not None
            or scheduler_spec is not None
            or scheduler_state is not None
        ):
            raise ValueError(
                "resume contains optimizer/scheduler state but no optimizer was supplied"
            )
    else:
        current_optimizer_names = _optimizer_parameter_names(
            model, optimizer, context="resume"
        )
        if optimizer_names != current_optimizer_names:
            raise ValueError("resume optimizer parameter names/order mismatch")
        if scheduler_spec != getattr(optimizer, "_kmd2_scheduler_spec", None):
            raise ValueError("resume scheduler_spec mismatch")
        scheduler = _bound_cache_scheduler(optimizer, context="resume")
        _validate_optimizer_resume_state(
            optimizer_state,
            optimizer_names,
            dict(named),
        )
        _validate_scheduler_resume_state(
            scheduler_state,
            scheduler,
            optimizer_state,
        )
        current_groups = optimizer.state_dict()["param_groups"]
        saved_groups = optimizer_state["param_groups"]
        hyperparameters = ("betas", "eps", "weight_decay", "initial_lr")
        for current_group, saved_group in zip(current_groups, saved_groups):
            for field in hyperparameters:
                if current_group.get(field) != saved_group.get(field):
                    raise ValueError(
                        f"resume optimizer hyperparameter {field} mismatch"
                    )

    parameter_snapshots = {
        name: parameter.detach().clone() for name, parameter in named
    }
    optimizer_snapshot = (
        None if optimizer is None else copy.deepcopy(optimizer.state_dict())
    )
    scheduler_snapshot = (
        None if scheduler is None else copy.deepcopy(scheduler.state_dict())
    )
    try:
        with torch.no_grad():
            for (_, parameter), tensor in zip(named, saved_tensors):
                parameter.copy_(tensor.to(device=parameter.device))
        if optimizer is not None:
            optimizer.load_state_dict(copy.deepcopy(optimizer_state))
            scheduler.load_state_dict(copy.deepcopy(scheduler_state))
    except Exception:
        with torch.no_grad():
            for name, parameter in named:
                parameter.copy_(parameter_snapshots[name])
        if optimizer is not None and optimizer_snapshot is not None:
            optimizer.load_state_dict(optimizer_snapshot)
        if scheduler is not None and scheduler_snapshot is not None:
            scheduler.load_state_dict(scheduler_snapshot)
        raise


def build_cache_optimizer_and_resume(
    model: torch.nn.Module,
    cache_config: CacheConfig,
    checkpoint: object,
    *,
    expected_job_id: str,
    betas: tuple[float, float],
    eps: float,
    scheduler_factory=None,
    scheduler_spec: object | None = None,
) -> torch.optim.AdamW:
    """Build against installed cache params, then strictly restore optimizer state."""
    optimizer = build_cache_optimizer(
        model,
        cache_config,
        betas=betas,
        eps=eps,
        scheduler_factory=scheduler_factory,
        scheduler_spec=scheduler_spec,
    )
    strict_load_cache_resume(
        model,
        checkpoint,
        expected_job_id,
        optimizer=optimizer,
    )
    return optimizer


def load_native_then_install(
    model: torch.nn.Module,
    manager: object,
    model_config: object,
    cache_config: CacheConfig,
    native_checkpoint: Mapping[str, object] | str | os.PathLike[str] | None,
    cache_resume: object | None = None,
    *,
    expected_job_id: str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, ...]:
    """Apply native mode, load its checkpoint, then atomically install cache layers."""
    if not isinstance(cache_config, CacheConfig):
        raise TypeError("cache_config must be a CacheConfig")
    if optimizer is not None:
        raise ValueError(
            "a pre-install optimizer cannot reference future cache parameters; "
            "use the two-phase build_cache_optimizer_and_resume helper after install"
        )
    prior_native_mode = os.environ.get("GDN3_KMD2_NATIVE")
    os.environ["GDN3_KMD2_NATIVE"] = "1"
    try:
        indices = _validate_upgraded_indices(
            model, manager, manager.apply_upgrade()
        )
        layers = model.model.layers
        originals: dict[int, KMD2NativeAttn] = {}
        prefixes: dict[int, str] = {}
        for index in indices:
            native = layers[index].linear_attn
            if isinstance(native, KMD2ExactCacheAttn):
                raise ValueError(f"layer {index} is already exact-cache installed")
            if not isinstance(native, KMD2NativeAttn):
                raise TypeError(
                    f"upgraded layer {index} is not an actual KMD2NativeAttn"
                )
            if native.layer_idx != index:
                raise ValueError(
                    f"upgraded layer {index} carries mismatched layer_idx={native.layer_idx}"
                )
            originals[index] = native
            prefixes[index] = _module_name(model, native) + "."

        checkpoint_tensors = _checkpoint_tensor_mapping(native_checkpoint)
        model_state = model.state_dict()
        targets: dict[str, torch.Tensor] = {}
        allowed_prefixes = tuple(prefixes.values())
        for name, tensor in checkpoint_tensors.items():
            if not name.startswith(allowed_prefixes) or name not in model_state:
                raise KeyError(
                    f"native checkpoint key {name!r} does not target an upgraded native tensor"
                )
            target = model_state[name]
            if tuple(tensor.shape) != tuple(target.shape):
                raise ValueError(
                    f"native checkpoint shape mismatch for {name}: "
                    f"expected {tuple(target.shape)}, got {tuple(tensor.shape)}"
                )
            if not bool(torch.isfinite(tensor.detach()).all()):
                raise ValueError(f"native checkpoint tensor {name} is nonfinite")
            targets[name] = target

        target_snapshots = {
            name: target.detach().clone() for name, target in targets.items()
        }
        replacements: dict[int, KMD2ExactCacheAttn] = {}
        try:
            with torch.no_grad():
                for name, target in targets.items():
                    target.copy_(checkpoint_tensors[name].to(device=target.device))
            for index in indices:
                native = originals[index]
                replacement = KMD2ExactCacheAttn.from_native(
                    native,
                    model_config=model_config,
                    cache_config=cache_config,
                )
                _verify_replacement(native, replacement)
                replacements[index] = replacement
            for index in indices:
                layers[index].linear_attn = replacements[index]
            if cache_resume is not None:
                if type(expected_job_id) is not str or not expected_job_id:
                    raise ValueError(
                        "expected_job_id is required when loading a cache resume"
                    )
                strict_load_cache_resume(
                    model,
                    cache_resume,
                    expected_job_id,
                    optimizer=optimizer,
                )
        except Exception:
            for index, native in originals.items():
                layers[index].linear_attn = native
            with torch.no_grad():
                for name, target in targets.items():
                    target.copy_(target_snapshots[name])
            raise
        return indices
    finally:
        if prior_native_mode is None:
            os.environ.pop("GDN3_KMD2_NATIVE", None)
        else:
            os.environ["GDN3_KMD2_NATIVE"] = prior_native_mode
