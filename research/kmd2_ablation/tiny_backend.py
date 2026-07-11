"""Small, dependency-light PyTorch backend for causal KMD-2 experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import CacheConfig
from .exact_cache import (
    admission_scores,
    cache_read_blocks,
    initialize_cache_read_parameters,
    merge_persistent_cache,
)


TINY_BACKEND_SCHEMA_VERSION = "1.1.0"
_MAX_UNBOUNDED_CACHE_TOKENS = 131_072
_MAX_CACHE_DIAGNOSTIC_BYTES = 512 * 1024 * 1024
# Retain the previous private tuning hook while applying the budget to every cache.
_MAX_UNBOUNDED_CACHE_DIAGNOSTIC_BYTES = _MAX_CACHE_DIAGNOSTIC_BYTES
_ROTATION_MODES = {
    "none",
    "current",
    "constant_rate",
    "non_cumulative",
    "fixed_rope",
    "moving_frame",
}
_FLOAT_DTYPES = {torch.float32, torch.float64, torch.bfloat16}
_BC_BIAS_MODES = {
    "none",
    "additive",
    "diagonal_rescale",
    "constant_coordinate_oracle",
}


class CacheDiagnosticBudgetError(ValueError):
    """A typed preallocation failure for cache diagnostics."""

    def __init__(
        self,
        *,
        estimated_bytes: int,
        budget_bytes: int,
        batch: int,
        steps: int,
        heads: int,
        query_slots: int,
        value_dim: int,
        cache_width: int,
        block_size: int,
        per_slot_read: bool,
        layers: int,
        unbounded_cache: bool,
    ) -> None:
        self.code = "diagnostic_budget_exceeded"
        self.context = {
            "estimated_bytes": estimated_bytes,
            "budget_bytes": budget_bytes,
            "batch": batch,
            "steps": steps,
            "heads": heads,
            "query_slots": query_slots,
            "value_dim": value_dim,
            "cache_width": cache_width,
            "block_size": block_size,
            "per_slot_read": per_slot_read,
            "layers": layers,
            "unbounded_cache": unbounded_cache,
        }
        super().__init__(
            f"{self.code}: cache diagnostics require "
            f"{estimated_bytes} bytes, exceeding the {budget_bytes}-byte budget; "
            f"context={self.context}"
        )


def _cache_diagnostic_allocation_bytes(
    *,
    batch: int,
    steps: int,
    heads: int,
    query_slots: int,
    value_dim: int,
    block_size: int,
    per_slot_read: bool,
    cache_width: int | None = None,
    layers: int = 1,
) -> int:
    """Conservatively estimate peak bytes for materialized cache diagnostics."""

    cache_width = steps if cache_width is None else cache_width
    candidates = cache_width + block_size
    diagnostic_slots = query_slots if per_slot_read else 0
    time_heads = steps * heads
    float32_bytes = 4
    int64_bytes = 8
    bool_bytes = 1
    row_bytes = (
        time_heads * value_dim * float32_bytes
        + time_heads * float32_bytes
        + time_heads * candidates * int64_bytes
        + time_heads * cache_width * int64_bytes
        + time_heads * candidates * bool_bytes
        + time_heads * (candidates + 1) * float32_bytes
        + time_heads * int64_bytes
        + 2 * time_heads * float32_bytes
        + time_heads * diagnostic_slots * value_dim * float32_bytes
        + time_heads
        * diagnostic_slots
        * (candidates + 1)
        * float32_bytes
        + time_heads * diagnostic_slots * int64_bytes
        + 3 * time_heads * diagnostic_slots * float32_bytes
        + heads * cache_width * int64_bytes
    )
    # Row tensors remain live while their batched copies are stacked, and every
    # layer's diagnostics remain reachable from TinyModelOutput.
    return 2 * batch * layers * row_bytes


def _positive_int(name: str, value: object) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive int")


def _unit_gate(name: str, value: object) -> None:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise TypeError(f"{name} must be a finite number")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0,1]")


def _validate_bc_operands(
    q: Tensor,
    k: Tensor,
    q_amplitude: Tensor,
    k_amplitude: Tensor,
    q_vector: Tensor,
    k_vector: Tensor,
) -> tuple[int, int]:
    named = {
        "q": q,
        "k": k,
        "q_amplitude": q_amplitude,
        "k_amplitude": k_amplitude,
        "q_vector": q_vector,
        "k_vector": k_vector,
    }
    if any(not isinstance(value, Tensor) for value in named.values()):
        raise TypeError("B/C operands must be torch tensors")
    if q.ndim != 5 or k.ndim != 5:
        raise ValueError("q and k must have shape [B,T,H,slots,dk]")
    if q.shape[:3] != k.shape[:3] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must share batch/time/head/key dimensions")
    heads, key_dim = q.shape[2], q.shape[-1]
    if q_amplitude.shape != (heads,) or k_amplitude.shape != (heads,):
        raise ValueError("B/C amplitudes must have shape [H]")
    if q_vector.shape != (heads, key_dim) or k_vector.shape != (heads, key_dim):
        raise ValueError("B/C vectors must have shape [H,dk]")
    if len({value.device for value in named.values()}) != 1:
        raise ValueError("B/C operands must share a device")
    if any(not value.is_floating_point() for value in named.values()):
        raise TypeError("B/C operands must be floating point")
    if any(not bool(torch.isfinite(value.detach()).all()) for value in named.values()):
        raise ValueError("B/C operands must be finite")
    return heads, key_dim


def apply_bc_additive(
    q: Tensor,
    k: Tensor,
    q_amplitude: Tensor,
    k_amplitude: Tensor,
    q_bias: Tensor,
    k_bias: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply separate post-normalization additive q/k channels."""

    _validate_bc_operands(q, k, q_amplitude, k_amplitude, q_bias, k_bias)
    q_offset = q_amplitude[:, None] * q_bias
    k_offset = k_amplitude[:, None] * k_bias
    return (
        q + q_offset[None, None, :, None].to(q.dtype),
        k + k_offset[None, None, :, None].to(k.dtype),
    )


def apply_bc_diagonal_rescale(
    q: Tensor,
    k: Tensor,
    q_amplitude: Tensor,
    k_amplitude: Tensor,
    q_scale: Tensor,
    k_scale: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply the equal-parameter multiplicative control without an offset."""

    _validate_bc_operands(q, k, q_amplitude, k_amplitude, q_scale, k_scale)
    q_factor = 1.0 + q_amplitude[:, None] * q_scale
    k_factor = 1.0 + k_amplitude[:, None] * k_scale
    return (
        q * q_factor[None, None, :, None].to(q.dtype),
        k * k_factor[None, None, :, None].to(k.dtype),
    )


def append_constant_coordinate(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    """Return the diagnostic q/k factors with one exact constant coordinate."""

    if not isinstance(q, Tensor) or not isinstance(k, Tensor):
        raise TypeError("q and k must be torch tensors")
    if q.ndim != 5 or k.ndim != 5 or q.shape[:3] != k.shape[:3]:
        raise ValueError("q and k must have compatible [B,T,H,slots,dk] shapes")
    if q.shape[-1] < 1 or k.shape[-1] < 1:
        raise ValueError("q and k must contain at least one learned coordinate")
    if q.device != k.device or q.dtype != k.dtype or not q.is_floating_point():
        raise ValueError("q and k must share a floating dtype and device")
    return (
        torch.cat((q, torch.ones_like(q[..., :1])), dim=-1),
        torch.cat((k, torch.ones_like(k[..., :1])), dim=-1),
    )


def true_mimo_update(
    state_bar: Tensor,
    key: Tensor,
    value: Tensor,
    beta_e: Tensor,
    beta_w: Tensor,
) -> Tensor:
    """Apply one simultaneous normalized rank-R update to a shared state."""

    operands = (state_bar, key, value, beta_e, beta_w)
    if not all(isinstance(item, Tensor) for item in operands):
        raise TypeError("true-MIMO operands must be torch tensors")
    if len({item.device for item in operands}) != 1:
        raise ValueError("true-MIMO operands must share one device")
    if any(not item.is_floating_point() for item in operands):
        raise TypeError("true-MIMO operands must be floating point")
    if len({item.dtype for item in operands}) != 1:
        raise ValueError("true-MIMO operands must share one dtype")
    if any(not bool(torch.isfinite(item.detach()).all()) for item in operands):
        raise ValueError("true-MIMO operands must contain only finite values")
    if bool((beta_e.detach() < 0).any()):
        raise ValueError("true-MIMO beta_e must be nonnegative")
    if state_bar.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("true-MIMO state/key/value ranks are invalid")
    batch, heads, key_dim, value_dim = state_bar.shape
    if key.shape[:2] != (batch, heads) or key.shape[-1] != key_dim:
        raise ValueError("true-MIMO key must have shape [B,H,R,dk]")
    rank = key.shape[2]
    if rank < 1 or value.shape != (batch, heads, rank, value_dim):
        raise ValueError("true-MIMO value must have shape [B,H,R,dv]")
    if beta_e.shape != (batch, heads, rank) or beta_w.shape != (batch, heads, rank):
        raise ValueError("true-MIMO gates must have shape [B,H,R]")
    if rank == 1:
        key_one = key[:, :, 0]
        value_one = value[:, :, 0]
        memory = torch.matmul(key_one.unsqueeze(-2), state_bar).squeeze(-2)
        update = (
            beta_w[:, :, 0].unsqueeze(-1) * value_one
            - beta_e[:, :, 0].unsqueeze(-1) * memory
        )
        return state_bar + key_one.unsqueeze(-1) * update.unsqueeze(-2)
    memory = torch.matmul(key, state_bar)
    erase = torch.einsum(
        "bhrd,bhrv->bhdv",
        key,
        (beta_e / rank).unsqueeze(-1) * memory,
    )
    write = torch.einsum(
        "bhrd,bhrv->bhdv", key, beta_w.unsqueeze(-1) * value
    )
    return state_bar - erase + write


def _rotate_state_rows(state: Tensor, phase: Tensor) -> Tensor:
    half = state.shape[-2] // 2
    transposed = state.transpose(-2, -1)
    first, second = transposed[..., :half], transposed[..., half:]
    cosine = phase.cos().unsqueeze(-2)
    sine = phase.sin().unsqueeze(-2)
    rotated = torch.cat(
        (first * cosine - second * sine, first * sine + second * cosine),
        dim=-1,
    )
    return rotated.transpose(-2, -1)


def moving_frame_transport_diagnostic(
    state: Tensor,
    decay: Tensor,
    previous_phase: Tensor,
    current_phase: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return exact fixed-frame transport and the moving-frame simplification."""

    if not all(
        isinstance(item, Tensor)
        for item in (state, decay, previous_phase, current_phase)
    ):
        raise TypeError("moving-frame operands must be torch tensors")
    if state.ndim < 3 or state.shape[-2] % 2:
        raise ValueError("state key rows must be positive and even")
    expected_decay = state.shape[:-1]
    expected_phase = state.shape[:-2] + (state.shape[-2] // 2,)
    if decay.shape != expected_decay:
        raise ValueError("decay must match the state key rows")
    if previous_phase.shape != expected_phase or current_phase.shape != expected_phase:
        raise ValueError("phases must have shape [...,dk/2]")
    fixed_previous = _rotate_state_rows(state, previous_phase)
    exact = _rotate_state_rows(decay.unsqueeze(-1) * fixed_previous, -current_phase)
    moving = decay.unsqueeze(-1) * _rotate_state_rows(
        state, previous_phase - current_phase
    )
    return exact, moving


def _validate_sequence_layout(
    valid: Tensor, positions: Tensor, boundaries: Tensor | None
) -> None:
    if valid.dtype != torch.bool or valid.ndim != 2:
        raise ValueError("valid must be bool with shape [B,T]")
    if positions.dtype != torch.int64 or positions.shape != valid.shape:
        raise ValueError("positions must be int64 with shape [B,T]")
    if bool((positions[valid] < 0).any()) or bool((positions[~valid] != -1).any()):
        raise ValueError("positions must be nonnegative when valid and -1 otherwise")
    if boundaries is not None:
        if boundaries.dtype != torch.bool or boundaries.shape != valid.shape:
            raise ValueError("boundaries must be bool with shape [B,T]")
        if bool((boundaries & ~valid).any()):
            raise ValueError("boundaries must be a subset of valid")
        if bool((boundaries & (positions != 0)).any()):
            raise ValueError("boundary tokens must have position zero")
        if bool(((positions == 0) & valid & ~boundaries).any()):
            raise ValueError("boundaries must mark every valid position zero")
    for batch_index in range(valid.shape[0]):
        expected: int | None = None
        for token_index in range(valid.shape[1]):
            if not bool(valid[batch_index, token_index]):
                continue
            declared_boundary = boundaries is not None and bool(
                boundaries[batch_index, token_index]
            )
            if declared_boundary:
                expected = 0
            elif expected is None:
                if boundaries is not None:
                    raise ValueError("the first valid token must be a boundary")
                expected = 0
            else:
                expected += 1
            if int(positions[batch_index, token_index]) != expected:
                raise ValueError(
                    "positions must start at zero and increase by one "
                    "over valid tokens within each boundary"
                )


def future_query_relevance(episode: Any) -> Tensor:
    """Count future supervised queries whose causal source span includes each token."""

    valid = getattr(episode, "valid", None)
    query_mask = getattr(episode, "query_mask", None)
    source_spans = getattr(episode, "source_spans", None)
    if (
        not isinstance(valid, Tensor)
        or valid.dtype != torch.bool
        or valid.ndim != 2
        or not isinstance(query_mask, Tensor)
        or query_mask.dtype != torch.bool
        or query_mask.shape != valid.shape
        or not isinstance(source_spans, Tensor)
        or source_spans.dtype != torch.int64
        or source_spans.shape != (*valid.shape, 2)
    ):
        raise ValueError(
            "future-query oracle requires valid/query_mask/source_spans annotations"
        )
    if len({valid.device, query_mask.device, source_spans.device}) != 1:
        raise ValueError("future-query oracle annotations must share one device")
    relevance = torch.zeros(valid.shape, dtype=torch.float32, device=valid.device)
    for batch_index in range(valid.shape[0]):
        queries = torch.nonzero(query_mask[batch_index], as_tuple=False).flatten()
        for query_index in queries.tolist():
            start, stop = source_spans[batch_index, query_index].tolist()
            if start < 0 or stop <= start or stop > query_index:
                raise ValueError(
                    "future-query oracle source spans must be causal and nonempty"
                )
            relevance[batch_index, start:stop] += 1.0
    return relevance.detach()


@dataclass(frozen=True)
class TinyKMD2Config:
    d_model: int
    heads: int
    dk: int
    dv: int
    layers: int
    vocab_size: int
    d_ff: int
    r_out: int = 1
    mimo_rank: int = 1
    continuous_input_dim: int | None = None
    output_dim: int | None = None
    conv_kernel: int = 3
    dtype: torch.dtype = torch.float32
    eps: float = 1.0e-6
    rotation_mode: str = "current"
    convolution_gate_init: float = 0.0
    rotation_gate_init: float = 0.0
    channel_decay_gate_init: float = 0.0
    write_offset_gate_init: float = 0.0
    trapezoid: bool = False
    trapezoid_gate_init: float = 0.0
    cache: CacheConfig | None = None
    corrected_momentum: bool = False
    momentum_gamma_init: float = 0.0
    causal_lookahead: bool = False
    lookahead_rho_init: float = 0.0
    bc_bias_mode: str = "none"
    selector_seed: int = 0
    unbounded_cache: bool = False
    per_slot_cache_read: bool = False

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "heads",
            "dk",
            "dv",
            "layers",
            "vocab_size",
            "d_ff",
            "r_out",
            "mimo_rank",
            "conv_kernel",
        ):
            _positive_int(name, getattr(self, name))
        for name in ("continuous_input_dim", "output_dim"):
            value = getattr(self, name)
            if value is not None:
                _positive_int(name, value)
        if self.r_out > 1 and self.mimo_rank > 1:
            raise ValueError("r_out and mimo_rank cannot both exceed one")
        if type(self.rotation_mode) is not str or self.rotation_mode not in _ROTATION_MODES:
            allowed = ", ".join(sorted(_ROTATION_MODES))
            raise ValueError(f"rotation_mode must be one of: {allowed}")
        if self.rotation_mode != "none" and self.dk % 2:
            raise ValueError("dk must be even when a paired rotation is enabled")
        if self.dtype not in _FLOAT_DTYPES:
            raise TypeError("dtype must be float32, float64, or bfloat16")
        if type(self.eps) not in (int, float) or not math.isfinite(float(self.eps)):
            raise TypeError("eps must be a finite number")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        for name in (
            "convolution_gate_init",
            "rotation_gate_init",
            "channel_decay_gate_init",
            "write_offset_gate_init",
            "trapezoid_gate_init",
            "momentum_gamma_init",
            "lookahead_rho_init",
        ):
            _unit_gate(name, getattr(self, name))
            object.__setattr__(self, name, float(getattr(self, name)))
        if type(self.trapezoid) is not bool:
            raise TypeError("trapezoid must be a bool")
        if type(self.corrected_momentum) is not bool:
            raise TypeError("corrected_momentum must be a bool")
        if type(self.causal_lookahead) is not bool:
            raise TypeError("causal_lookahead must be a bool")
        if type(self.selector_seed) is not int:
            raise TypeError("selector_seed must be an int")
        if type(self.unbounded_cache) is not bool:
            raise TypeError("unbounded_cache must be a bool")
        if type(self.per_slot_cache_read) is not bool:
            raise TypeError("per_slot_cache_read must be a bool")
        if self.mimo_rank > 1:
            siso_only = tuple(
                name
                for name, enabled in (
                    ("trapezoid", self.trapezoid),
                    ("corrected_momentum", self.corrected_momentum),
                    ("causal_lookahead", self.causal_lookahead),
                )
                if enabled
            )
            if siso_only:
                raise ValueError(
                    "mimo_rank>1 is incompatible with SISO-only features: "
                    + ", ".join(siso_only)
                )
        if type(self.bc_bias_mode) is not str or self.bc_bias_mode not in _BC_BIAS_MODES:
            allowed = ", ".join(sorted(_BC_BIAS_MODES))
            raise ValueError(f"bc_bias_mode must be one of: {allowed}")
        if self.bc_bias_mode == "constant_coordinate_oracle":
            if self.dk < 2:
                raise ValueError(
                    "constant-coordinate oracle requires dk>=2 including its fixed coordinate"
                )
            if self.rotation_mode != "none":
                raise ValueError(
                    "constant-coordinate oracle requires rotation_mode=none"
                )
            if self.cache is not None:
                raise ValueError("constant-coordinate oracle cannot be combined with cache")
        if self.corrected_momentum and self.trapezoid:
            raise ValueError("corrected_momentum and trapezoid cannot be combined")
        object.__setattr__(self, "eps", float(self.eps))
        if self.cache is not None and not isinstance(self.cache, CacheConfig):
            raise TypeError("cache must be a CacheConfig or None")
        if self.cache is not None and self.mimo_rank != 1:
            raise ValueError("exact cache currently requires mimo_rank=1")
        if self.unbounded_cache and self.cache is None:
            raise ValueError("unbounded_cache requires an exact cache configuration")
        if self.per_slot_cache_read and self.cache is None:
            raise ValueError("per_slot_cache_read requires an exact cache configuration")


def _validate_cache_diagnostic_preallocation(
    config: TinyKMD2Config, *, batch: int, steps: int
) -> None:
    cache = config.cache
    if cache is None:
        return
    cache_width = steps if config.unbounded_cache else cache.width
    query_slots = config.r_out if config.mimo_rank == 1 else config.mimo_rank
    estimated_bytes = _cache_diagnostic_allocation_bytes(
        batch=batch,
        steps=steps,
        heads=config.heads,
        query_slots=query_slots,
        value_dim=config.dv,
        cache_width=cache_width,
        block_size=cache.block_size,
        per_slot_read=config.per_slot_cache_read,
        layers=config.layers,
    )
    budget_bytes = min(
        _MAX_CACHE_DIAGNOSTIC_BYTES,
        _MAX_UNBOUNDED_CACHE_DIAGNOSTIC_BYTES,
    )
    if estimated_bytes > budget_bytes:
        raise CacheDiagnosticBudgetError(
            estimated_bytes=estimated_bytes,
            budget_bytes=budget_bytes,
            batch=batch,
            steps=steps,
            heads=config.heads,
            query_slots=query_slots,
            value_dim=config.dv,
            cache_width=cache_width,
            block_size=cache.block_size,
            per_slot_read=config.per_slot_cache_read,
            layers=config.layers,
            unbounded_cache=config.unbounded_cache,
        )
    if config.unbounded_cache and steps > _MAX_UNBOUNDED_CACHE_TOKENS:
        raise ValueError(
            "unbounded cache episode exceeds the finite safety bound of "
            f"{_MAX_UNBOUNDED_CACHE_TOKENS} tokens"
        )


@dataclass(frozen=True)
class TinyFactors:
    q: Tensor
    k: Tensor
    v: Tensor
    decay: Tensor
    beta_e: Tensor
    beta_w: Tensor
    out_mix: Tensor
    valid: Tensor
    positions: Tensor
    read_gate: Tensor | None = None
    trapezoid_rho: Tensor | None = None
    momentum_gamma: Tensor | None = None
    lookahead_rho: Tensor | None = None
    moving_frame_phase: Tensor | None = None
    cache_q: Tensor | None = None
    cache_k: Tensor | None = None

    def __post_init__(self) -> None:
        named = {
            "q": self.q,
            "k": self.k,
            "v": self.v,
            "decay": self.decay,
            "beta_e": self.beta_e,
            "beta_w": self.beta_w,
            "out_mix": self.out_mix,
            "valid": self.valid,
            "positions": self.positions,
        }
        if self.trapezoid_rho is not None:
            named["trapezoid_rho"] = self.trapezoid_rho
        if self.read_gate is not None:
            named["read_gate"] = self.read_gate
        if self.momentum_gamma is not None:
            named["momentum_gamma"] = self.momentum_gamma
        if self.lookahead_rho is not None:
            named["lookahead_rho"] = self.lookahead_rho
        if self.moving_frame_phase is not None:
            named["moving_frame_phase"] = self.moving_frame_phase
        if (self.cache_q is None) != (self.cache_k is None):
            raise ValueError("cache_q and cache_k must be provided together")
        if self.cache_q is not None:
            named["cache_q"] = self.cache_q
            assert self.cache_k is not None
            named["cache_k"] = self.cache_k
        for name, tensor in named.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if self.q.ndim != 5:
            raise ValueError("q must have shape [B,T,H,Q,dk]")
        batch, steps, heads, q_slots, key_dim = self.q.shape
        if min(batch, steps, heads, q_slots, key_dim) < 1:
            raise ValueError("q dimensions must be positive")
        if self.k.ndim != 5:
            raise ValueError("k must have shape [B,T,H,R,dk]")
        write_slots = self.k.shape[3]
        if self.k.shape != (batch, steps, heads, write_slots, key_dim):
            raise ValueError("k must match q batch/time/head/key dimensions")
        if write_slots < 1:
            raise ValueError("k write-slot dimension must be positive")
        if self.v.ndim != 5 or self.v.shape[:4] != (
            batch,
            steps,
            heads,
            write_slots,
        ):
            raise ValueError("v must have shape [B,T,H,R,dv] matching k")
        value_dim = self.v.shape[-1]
        if value_dim < 1:
            raise ValueError("v value dimension must be positive")
        expected_shapes = {
            "decay": (batch, steps, heads, key_dim),
            "beta_e": (batch, steps, heads, write_slots),
            "beta_w": (batch, steps, heads, write_slots),
            "valid": (batch, steps),
            "positions": (batch, steps),
        }
        for name, shape in expected_shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        valid_out_mix_shapes = {
            (batch, steps, heads, q_slots),
            (batch, steps, heads, q_slots, value_dim),
        }
        if tuple(self.out_mix.shape) not in valid_out_mix_shapes:
            raise ValueError(
                "out_mix must have shape [B,T,H,Q] or [B,T,H,Q,dv]"
            )
        if self.read_gate is not None and self.read_gate.shape != (
            batch,
            steps,
            heads,
            q_slots,
            value_dim,
        ):
            raise ValueError("read_gate must have shape [B,T,H,Q,dv]")
        if self.trapezoid_rho is not None:
            if not isinstance(self.trapezoid_rho, Tensor):
                raise TypeError("trapezoid_rho must be a torch.Tensor or None")
            if self.trapezoid_rho.shape != (batch, steps, heads):
                raise ValueError("trapezoid_rho must have shape [B,T,H]")
            if not self.trapezoid_rho.is_floating_point():
                raise TypeError("trapezoid_rho must be floating point")
            if not bool(torch.isfinite(self.trapezoid_rho.detach()).all()):
                raise ValueError("trapezoid_rho must contain only finite values")
            if bool(
                (
                    (self.trapezoid_rho.detach() < 0)
                    | (self.trapezoid_rho.detach() > 1)
                ).any()
            ):
                raise ValueError("trapezoid_rho must be in [0,1]")
        if self.momentum_gamma is not None:
            if not isinstance(self.momentum_gamma, Tensor):
                raise TypeError("momentum_gamma must be a torch.Tensor or None")
            if self.momentum_gamma.shape != (batch, steps, heads):
                raise ValueError("momentum_gamma must have shape [B,T,H]")
            if not self.momentum_gamma.is_floating_point():
                raise TypeError("momentum_gamma must be floating point")
            if not bool(torch.isfinite(self.momentum_gamma.detach()).all()):
                raise ValueError("momentum_gamma must contain only finite values")
            if bool(
                (
                    (self.momentum_gamma.detach() < 0)
                    | (self.momentum_gamma.detach() > 1)
                ).any()
            ):
                raise ValueError("momentum_gamma must be in [0,1]")
        if self.lookahead_rho is not None:
            if not isinstance(self.lookahead_rho, Tensor):
                raise TypeError("lookahead_rho must be a torch.Tensor or None")
            if self.lookahead_rho.shape != (batch, steps, heads):
                raise ValueError("lookahead_rho must have shape [B,T,H]")
            if not self.lookahead_rho.is_floating_point():
                raise TypeError("lookahead_rho must be floating point")
            if not bool(torch.isfinite(self.lookahead_rho.detach()).all()):
                raise ValueError("lookahead_rho must contain only finite values")
            if bool(
                (
                    (self.lookahead_rho.detach() < 0)
                    | (self.lookahead_rho.detach() > 1)
                ).any()
            ):
                raise ValueError("lookahead_rho must be in [0,1]")
        if self.moving_frame_phase is not None:
            if self.moving_frame_phase.shape != (
                batch,
                steps,
                heads,
                key_dim // 2,
            ):
                raise ValueError("moving_frame_phase must have shape [B,T,H,dk/2]")
            if key_dim % 2 or not self.moving_frame_phase.is_floating_point():
                raise TypeError("moving_frame_phase requires even dk and floating dtype")
            if not bool(torch.isfinite(self.moving_frame_phase.detach()).all()):
                raise ValueError("moving_frame_phase must be finite")
        if self.cache_q is not None:
            assert self.cache_k is not None
            if self.cache_q.shape != self.q.shape or self.cache_k.shape != self.k.shape:
                raise ValueError("cache_q/cache_k must match q/k shapes")
            if not self.cache_q.is_floating_point() or not self.cache_k.is_floating_point():
                raise TypeError("cache_q/cache_k must be floating point")
            if not bool(torch.isfinite(self.cache_q.detach()).all()) or not bool(
                torch.isfinite(self.cache_k.detach()).all()
            ):
                raise ValueError("cache_q/cache_k must be finite")
        for name in ("q", "k", "v", "decay", "beta_e", "beta_w", "out_mix"):
            tensor = getattr(self, name)
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must be floating point")
            if not bool(torch.isfinite(tensor.detach()).all()):
                raise ValueError(f"{name} must contain only finite values")
        if self.valid.dtype != torch.bool:
            raise TypeError("valid must have bool dtype")
        if self.positions.dtype != torch.int64:
            raise TypeError("positions must have int64 dtype")
        if bool((self.positions[self.valid] < 0).any()):
            raise ValueError("positions must be nonnegative at valid tokens")
        if bool((self.positions[~self.valid] != -1).any()):
            raise ValueError("positions must be -1 at invalid tokens")
        if len({tensor.device for tensor in named.values()}) != 1:
            raise ValueError("all TinyFactors tensors must share a device")

    @property
    def shape(self) -> tuple[int, int, int, int, int, int, int]:
        batch, steps, heads, q_slots, key_dim = self.q.shape
        write_slots = self.k.shape[3]
        return batch, steps, heads, q_slots, write_slots, key_dim, self.v.shape[-1]


@dataclass(frozen=True)
class TinyCellOutput:
    read: Tensor
    final_state: Tensor
    scores: Tensor
    state_read: Tensor
    cache_read: Tensor
    selected_positions: Tensor
    sink_mass: Tensor
    hit_ready_positions: Tensor
    persistent_selected_positions: Tensor
    candidate_valid: Tensor
    attention_weights: Tensor
    top1_positions: Tensor
    attention_entropy: Tensor
    top1_mass: Tensor
    slot_cache_read: Tensor
    slot_attention_weights: Tensor
    slot_top1_positions: Tensor
    slot_sink_mass: Tensor
    slot_attention_entropy: Tensor
    slot_top1_mass: Tensor
    retention_count: int
    eviction_count: int
    state_bytes: int
    cache_persistent_bytes: int
    cache_block_bytes: int


@dataclass(frozen=True)
class TinyModelOutput:
    logits: Tensor
    loss: Tensor | None
    final_states: tuple[Tensor, ...]
    cell_outputs: tuple[TinyCellOutput, ...]


class TinyKMD2Cell(nn.Module):
    """Exact fp32 post-update KMD-2 recurrence over explicit factors."""

    def __init__(self, config: TinyKMD2Config):
        super().__init__()
        if not isinstance(config, TinyKMD2Config):
            raise TypeError("config must be a TinyKMD2Config")
        self.config = config
        if config.cache is not None:
            parameters = initialize_cache_read_parameters(config.dk, config.heads)
            self.cache_gamma_q = parameters.gamma_q
            self.cache_gamma_k = parameters.gamma_k
            self.cache_sink_logit = parameters.sink_logit
            self.cache_amplitude = parameters.amplitude
        if config.causal_lookahead:
            self.lookahead_projection = nn.Linear(config.dv, config.dv, bias=False)
            nn.init.eye_(self.lookahead_projection.weight)
        if config.bc_bias_mode in {"additive", "diagonal_rescale"}:
            self.bc_q_amplitude = nn.Parameter(torch.zeros(config.heads))
            self.bc_k_amplitude = nn.Parameter(torch.zeros(config.heads))
            base = torch.linspace(-0.5, 0.5, config.dk).repeat(config.heads, 1)
            if config.bc_bias_mode == "additive":
                self.bc_q_bias = nn.Parameter(base.clone())
                self.bc_k_bias = nn.Parameter(base.flip(-1).clone())
            else:
                self.bc_q_scale = nn.Parameter(base.clone())
                self.bc_k_scale = nn.Parameter(base.flip(-1).clone())

    def _apply(self, fn):
        result = super()._apply(fn)
        if self.config.cache is not None:
            for name in (
                "cache_gamma_q",
                "cache_gamma_k",
                "cache_sink_logit",
                "cache_amplitude",
            ):
                parameter = getattr(self, name)
                if parameter.dtype != torch.float32:
                    parameter.data = parameter.data.float()
                    if parameter.grad is not None:
                        parameter.grad.data = parameter.grad.data.float()
        return result

    def _validate_factors(self, factors: TinyFactors) -> None:
        if not isinstance(factors, TinyFactors):
            raise TypeError("factors must be TinyFactors")
        _, _, heads, q_slots, write_slots, key_dim, value_dim = factors.shape
        valid_key_dims = {self.config.dk}
        if self.config.bc_bias_mode == "constant_coordinate_oracle":
            valid_key_dims.add(self.config.dk - 1)
        if (
            heads != self.config.heads
            or key_dim not in valid_key_dims
            or value_dim != self.config.dv
        ):
            raise ValueError("factor head/dk/dv dimensions must match config")
        if self.config.mimo_rank == 1:
            if write_slots != 1 or q_slots != self.config.r_out:
                raise ValueError("native factors require R=1 and Q=config.r_out")
            if factors.out_mix.ndim != 4:
                raise ValueError("native factors require scalar output-slot mixing")
        elif write_slots != self.config.mimo_rank or q_slots != self.config.mimo_rank:
            raise ValueError("true-MIMO factors require R=Q=config.mimo_rank")
        elif factors.out_mix.ndim != 5 or factors.read_gate is None:
            raise ValueError(
                "true-MIMO factors require channelwise output mixing and rankwise gates"
            )
        if self.config.trapezoid and factors.trapezoid_rho is None:
            raise ValueError("trapezoid factors require trapezoid_rho")
        if not self.config.trapezoid and factors.trapezoid_rho is not None:
            raise ValueError("trapezoid_rho requires config.trapezoid=true")
        if self.config.corrected_momentum and factors.momentum_gamma is None:
            raise ValueError("corrected momentum factors require momentum_gamma")
        if not self.config.corrected_momentum and factors.momentum_gamma is not None:
            raise ValueError(
                "momentum_gamma requires config.corrected_momentum=true"
            )
        if self.config.causal_lookahead and factors.lookahead_rho is None:
            raise ValueError("causal lookahead factors require lookahead_rho")
        if not self.config.causal_lookahead and factors.lookahead_rho is not None:
            raise ValueError("lookahead_rho requires config.causal_lookahead=true")
        if self.config.rotation_mode == "moving_frame":
            if factors.moving_frame_phase is None:
                raise ValueError("moving-frame factors require moving_frame_phase")
        elif factors.moving_frame_phase is not None:
            raise ValueError("moving_frame_phase requires rotation_mode=moving_frame")
        pre_rotation = (
            self.config.cache is not None
            and self.config.cache.coordinate_frame == "pre_rotation"
        )
        if pre_rotation and (factors.cache_q is None or factors.cache_k is None):
            raise ValueError("pre-rotation cache requires explicit cache_q/cache_k")
        if not pre_rotation and (factors.cache_q is not None or factors.cache_k is not None):
            raise ValueError("cache_q/cache_k require a pre-rotation cache configuration")

    def forward(
        self,
        factors: TinyFactors,
        state: Tensor | None = None,
        boundaries: Tensor | None = None,
        future_relevance: Tensor | None = None,
    ) -> TinyCellOutput:
        device_type = factors.q.device.type if isinstance(factors, TinyFactors) else "cpu"
        if device_type in {"cpu", "cuda"}:
            with torch.autocast(device_type=device_type, enabled=False):
                return self._forward_fp32(
                    factors, state, boundaries, future_relevance
                )
        return self._forward_fp32(factors, state, boundaries, future_relevance)

    def _forward_fp32(
        self,
        factors: TinyFactors,
        state: Tensor | None,
        boundaries: Tensor | None,
        future_relevance: Tensor | None,
    ) -> TinyCellOutput:
        self._validate_factors(factors)
        _validate_cache_diagnostic_preallocation(
            self.config,
            batch=factors.q.shape[0],
            steps=factors.q.shape[1],
        )
        (
            batch,
            steps,
            heads,
            q_slots,
            write_slots,
            source_key_dim,
            value_dim,
        ) = factors.shape
        key_dim = (
            self.config.dk
            if self.config.bc_bias_mode == "constant_coordinate_oracle"
            else source_key_dim
        )
        device = factors.q.device
        if state is None:
            current = torch.zeros(
                batch, heads, key_dim, value_dim, dtype=torch.float32, device=device
            )
        else:
            if not isinstance(state, Tensor):
                raise TypeError("state must be a tensor or None")
            if state.dtype != torch.float32:
                raise TypeError("state must have float32 dtype")
            if state.shape != (batch, heads, key_dim, value_dim):
                raise ValueError("state must have shape [B,H,dk,dv]")
            if state.device != device:
                raise ValueError("state and factors must share a device")
            if not bool(torch.isfinite(state.detach()).all()):
                raise ValueError("state must be finite")
            current = state
        if boundaries is None:
            reset = torch.zeros(batch, steps, dtype=torch.bool, device=device)
        else:
            if not isinstance(boundaries, Tensor):
                raise TypeError("boundaries must be a tensor or None")
            if boundaries.dtype != torch.bool or boundaries.shape != (batch, steps):
                raise ValueError("boundaries must be bool with shape [B,T]")
            if boundaries.device != device:
                raise ValueError("boundaries and factors must share a device")
            if bool((boundaries & ~factors.valid).any()):
                raise ValueError("boundaries must be a subset of valid")
            reset = boundaries

        _validate_sequence_layout(factors.valid, factors.positions, boundaries)

        q = factors.q.float()
        k = factors.k.float()
        v = factors.v.float()
        decay = factors.decay.float()
        beta_e = factors.beta_e.float()
        beta_w = factors.beta_w.float()
        out_mix = factors.out_mix.float()
        read_gate = None if factors.read_gate is None else factors.read_gate.float()
        if self.config.bc_bias_mode == "constant_coordinate_oracle":
            if source_key_dim == self.config.dk - 1:
                q, k = append_constant_coordinate(q, k)
                # Raw direct factors do not carry the projector's ``g_head``.
                # Their oracle scalar is therefore defined only when the data
                # channels share one decay, which is reused for the fixed row.
                scalar_decay = decay[..., :1]
                if not torch.equal(decay, scalar_decay.expand_as(decay)):
                    raise ValueError(
                        "raw constant-coordinate factors require channel-tied decay"
                    )
                decay = torch.cat((decay, scalar_decay), dim=-1)
            elif not (
                torch.equal(q[..., -1], torch.ones_like(q[..., -1]))
                and torch.equal(k[..., -1], torch.ones_like(k[..., -1]))
            ):
                raise ValueError(
                    "constant-coordinate factors must have an exact final q/k coordinate"
                )
        if self.config.bc_bias_mode == "additive":
            q, k = apply_bc_additive(
                q,
                k,
                self.bc_q_amplitude.float(),
                self.bc_k_amplitude.float(),
                self.bc_q_bias.float(),
                self.bc_k_bias.float(),
            )
        elif self.config.bc_bias_mode == "diagonal_rescale":
            q, k = apply_bc_diagonal_rescale(
                q,
                k,
                self.bc_q_amplitude.float(),
                self.bc_k_amplitude.float(),
                self.bc_q_scale.float(),
                self.bc_k_scale.float(),
            )
        trapezoid_rho = (
            None if factors.trapezoid_rho is None else factors.trapezoid_rho.float()
        )
        momentum_gamma = (
            None if factors.momentum_gamma is None else factors.momentum_gamma.float()
        )
        lookahead_rho = (
            None if factors.lookahead_rho is None else factors.lookahead_rho.float()
        )
        moving_frame_phase = (
            None
            if factors.moving_frame_phase is None
            else factors.moving_frame_phase.float()
        )
        velocity = None
        if momentum_gamma is not None:
            velocity = torch.zeros_like(current)
        previous_value = None
        if lookahead_rho is not None:
            previous_value = torch.zeros(
                batch, heads, value_dim, dtype=torch.float32, device=device
            )
        previous_key = None
        previous_write = None
        if trapezoid_rho is not None:
            previous_key = torch.zeros(
                batch, heads, key_dim, dtype=torch.float32, device=device
            )
            previous_write = torch.zeros(
                batch, heads, value_dim, dtype=torch.float32, device=device
            )
        previous_phase = None
        if moving_frame_phase is not None:
            previous_phase = torch.zeros(
                batch,
                heads,
                key_dim // 2,
                dtype=torch.float32,
                device=device,
            )
        outputs: list[Tensor] = []
        scores: list[Tensor] = []
        cache_memories: list[Tensor] = []
        for token in range(steps):
            current = torch.where(
                reset[:, token, None, None, None],
                torch.zeros((), dtype=torch.float32, device=device),
                current,
            )
            if velocity is not None:
                velocity = torch.where(
                    reset[:, token, None, None, None],
                    torch.zeros((), dtype=torch.float32, device=device),
                    velocity,
                )
            if previous_phase is None:
                state_bar = decay[:, token].unsqueeze(-1) * current
                current_phase = None
            else:
                assert moving_frame_phase is not None
                previous_phase = torch.where(
                    reset[:, token, None, None],
                    torch.zeros((), dtype=torch.float32, device=device),
                    previous_phase,
                )
                current_phase = moving_frame_phase[:, token]
                state_bar = decay[:, token].unsqueeze(-1) * _rotate_state_rows(
                    current, previous_phase - current_phase
                )
            key = k[:, token]
            value = v[:, token]
            if write_slots == 1:
                key_one = key[:, :, 0]
                value_one = value[:, :, 0]
                cache_memory = torch.matmul(
                    key_one.unsqueeze(-2), state_bar
                ).squeeze(-2)
                cache_memories.append(cache_memory)
                value_target = value_one
                if lookahead_rho is not None:
                    assert previous_value is not None
                    gate = lookahead_rho[:, token]
                    first_or_boundary = (
                        (factors.positions[:, token] == 0) | reset[:, token]
                    )
                    gate = torch.where(
                        first_or_boundary[:, None], torch.zeros_like(gate), gate
                    )
                    projected_difference = F.linear(
                        value_one - previous_value,
                        self.lookahead_projection.weight.float(),
                    )
                    value_target = (
                        value_one + gate.unsqueeze(-1) * projected_difference
                    )
                state_look = state_bar
                velocity_bar = None
                gamma = None
                if momentum_gamma is not None:
                    assert velocity is not None
                    gamma = momentum_gamma[:, token]
                    velocity_bar = decay[:, token].unsqueeze(-1) * velocity
                    state_look = state_bar + gamma[:, :, None, None] * velocity_bar
                memory = torch.matmul(key_one.unsqueeze(-2), state_look).squeeze(-2)
                update = (
                    beta_w[:, token, :, 0].unsqueeze(-1) * value_target
                    - beta_e[:, token, :, 0].unsqueeze(-1) * memory
                )
                native_outer = key_one.unsqueeze(-1) * update.unsqueeze(-2)
                velocity_candidate = None
                if gamma is None:
                    candidate = state_bar + native_outer
                else:
                    assert velocity_bar is not None
                    velocity_candidate = (
                        gamma[:, :, None, None] * velocity_bar + native_outer
                    )
                    candidate = state_bar + velocity_candidate
                native_score = torch.linalg.vector_norm(
                    key_one, dim=-1
                ) * torch.linalg.vector_norm(update, dim=-1)
                if trapezoid_rho is not None:
                    assert previous_key is not None and previous_write is not None
                    current_write_value = (
                        beta_w[:, token, :, 0].unsqueeze(-1) * value_target
                    )
                    current_write_outer = (
                        key_one.unsqueeze(-1) * current_write_value.unsqueeze(-2)
                    )
                    gate = trapezoid_rho[:, token]
                    first_or_boundary = (
                        (factors.positions[:, token] == 0) | reset[:, token]
                    )
                    gate = torch.where(
                        first_or_boundary[:, None], torch.zeros_like(gate), gate
                    )
                    previous_outer = (
                        previous_key.unsqueeze(-1) * previous_write.unsqueeze(-2)
                    )
                    transported_previous = decay[:, token].unsqueeze(-1) * previous_outer
                    correction = gate.unsqueeze(-1).unsqueeze(-1) * (
                        transported_previous - current_write_outer
                    )
                    candidate = candidate + correction
                    active_score = torch.linalg.matrix_norm(
                        native_outer + correction, dim=(-2, -1)
                    )
                    score = torch.where(gate == 0, native_score, active_score)
                else:
                    score = native_score
            else:
                candidate = true_mimo_update(
                    state_bar,
                    key,
                    value,
                    beta_e[:, token],
                    beta_w[:, token],
                )
                score = torch.linalg.matrix_norm(candidate - state_bar, dim=(-2, -1))
            active = factors.valid[:, token, None, None, None]
            current = torch.where(active, candidate, current)
            if previous_phase is not None:
                assert current_phase is not None
                previous_phase = torch.where(
                    factors.valid[:, token, None, None],
                    current_phase,
                    previous_phase,
                )
            if velocity is not None:
                assert velocity_candidate is not None
                velocity = torch.where(active, velocity_candidate, velocity)
            if previous_value is not None:
                previous_value = torch.where(
                    factors.valid[:, token, None, None], value_one, previous_value
                )
            if write_slots == 1 and trapezoid_rho is not None:
                assert previous_key is not None and previous_write is not None
                carry_active = factors.valid[:, token, None, None]
                previous_key = torch.where(carry_active, key_one, previous_key)
                previous_write = torch.where(
                    carry_active, current_write_value, previous_write
                )
            slot_read = torch.matmul(q[:, token], current)
            if read_gate is not None:
                slot_read = slot_read * F.silu(read_gate[:, token])
            token_mix = out_mix[:, token]
            if token_mix.ndim == 4:
                read = (slot_read * token_mix).sum(dim=-2)
            else:
                read = (slot_read * token_mix.unsqueeze(-1)).sum(dim=-2)
            read = torch.where(
                factors.valid[:, token, None, None],
                read,
                torch.zeros((), dtype=torch.float32, device=device),
            )
            score = torch.where(
                factors.valid[:, token, None],
                score,
                torch.zeros((), dtype=torch.float32, device=device),
            )
            outputs.append(read)
            scores.append(score)
        state_read = torch.stack(outputs, dim=1)
        score_tensor = torch.stack(scores, dim=1).detach()
        if self.config.cache is not None:
            if write_slots != 1 or len(cache_memories) != steps:
                raise RuntimeError("exact cache requires one memory/write slot per token")
            score_tensor = admission_scores(
                policy=self.config.cache.score,
                key=k[:, :, :, 0],
                value=v[:, :, :, 0],
                memory=torch.stack(cache_memories, dim=1),
                beta_e=beta_e[:, :, :, 0],
                beta_w=beta_w[:, :, :, 0],
                positions=factors.positions,
                valid=factors.valid,
                selector_seed=self.config.selector_seed,
                future_relevance=future_relevance,
            )
        if self.config.cache is None:
            cache_read = torch.zeros_like(state_read)
            selected_positions = torch.empty(
                batch, heads, 0, dtype=torch.int64, device=device
            )
            sink_mass = torch.zeros(
                batch, steps, heads, dtype=torch.float32, device=device
            )
            hit_ready_positions = torch.empty(
                batch, steps, heads, 0, dtype=torch.int64, device=device
            )
            persistent_selected_positions = torch.empty(
                batch, steps, heads, 0, dtype=torch.int64, device=device
            )
            candidate_valid = torch.empty(
                batch, steps, heads, 0, dtype=torch.bool, device=device
            )
            attention_weights = torch.zeros(
                batch, steps, heads, 1, dtype=torch.float32, device=device
            )
            top1_positions = torch.full(
                (batch, steps, heads), -1, dtype=torch.int64, device=device
            )
            attention_entropy = torch.zeros(
                batch, steps, heads, dtype=torch.float32, device=device
            )
            top1_mass = torch.zeros(
                batch, steps, heads, dtype=torch.float32, device=device
            )
            slot_cache_read = torch.empty(
                batch, steps, heads, 0, value_dim, dtype=torch.float32, device=device
            )
            slot_attention_weights = torch.empty(
                batch, steps, heads, 0, 1, dtype=torch.float32, device=device
            )
            slot_top1_positions = torch.empty(
                batch, steps, heads, 0, dtype=torch.int64, device=device
            )
            slot_sink_mass = torch.empty(
                batch, steps, heads, 0, dtype=torch.float32, device=device
            )
            slot_attention_entropy = torch.empty_like(slot_sink_mass)
            slot_top1_mass = torch.empty_like(slot_sink_mass)
            retention_count = 0
            eviction_count = 0
            cache_persistent_bytes = 0
            cache_block_bytes = 0
        else:
            (
                cache_read,
                selected_positions,
                sink_mass,
                hit_ready_positions,
                persistent_selected_positions,
                candidate_valid,
                attention_weights,
                top1_positions,
                attention_entropy,
                top1_mass,
                slot_cache_read,
                slot_attention_weights,
                slot_top1_positions,
                slot_sink_mass,
                slot_attention_entropy,
                slot_top1_mass,
                retention_count,
                eviction_count,
                cache_persistent_bytes,
                cache_block_bytes,
            ) = self._cache_forward(factors, score_tensor, reset)
        read = state_read
        if self.config.cache is not None:
            read = state_read + self.cache_amplitude.view(1, 1, heads, 1) * cache_read
        return TinyCellOutput(
            read=read,
            final_state=current,
            scores=score_tensor,
            state_read=state_read,
            cache_read=cache_read,
            selected_positions=selected_positions,
            sink_mass=sink_mass,
            hit_ready_positions=hit_ready_positions,
            persistent_selected_positions=persistent_selected_positions,
            candidate_valid=candidate_valid,
            attention_weights=attention_weights,
            top1_positions=top1_positions,
            attention_entropy=attention_entropy,
            top1_mass=top1_mass,
            slot_cache_read=slot_cache_read,
            slot_attention_weights=slot_attention_weights,
            slot_top1_positions=slot_top1_positions,
            slot_sink_mass=slot_sink_mass,
            slot_attention_entropy=slot_attention_entropy,
            slot_top1_mass=slot_top1_mass,
            retention_count=retention_count,
            eviction_count=eviction_count,
            state_bytes=(
                current.numel() * current.element_size()
                + (0 if velocity is None else velocity.numel() * velocity.element_size())
                + (
                    0
                    if previous_value is None
                    else previous_value.numel() * previous_value.element_size()
                )
                + (
                    0
                    if previous_phase is None
                    else previous_phase.numel() * previous_phase.element_size()
                )
                + (
                    0
                    if previous_key is None
                    else previous_key.numel() * previous_key.element_size()
                )
                + (
                    0
                    if previous_write is None
                    else previous_write.numel() * previous_write.element_size()
                )
            ),
            cache_persistent_bytes=cache_persistent_bytes,
            cache_block_bytes=cache_block_bytes,
        )

    def _cache_forward(
        self, factors: TinyFactors, scores: Tensor, boundaries: Tensor
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        int,
        int,
        int,
        int,
    ]:
        config = self.config.cache
        assert config is not None
        batch, steps, heads, q_slots, _, key_dim, value_dim = factors.shape
        cache_width = steps if self.config.unbounded_cache else config.width
        effective_config = (
            replace(config, width=cache_width)
            if cache_width != config.width
            else config
        )
        cache_q = factors.q if factors.cache_q is None else factors.cache_q
        cache_k = factors.k if factors.cache_k is None else factors.cache_k
        q_eff = torch.einsum(
            "bthqd,bthq->bthd", cache_q.float(), factors.out_mix.float()
        )
        keys = cache_k[:, :, :, 0].float()
        values = factors.v[:, :, :, 0].float()
        cache_rows: list[Tensor] = []
        sink_rows: list[Tensor] = []
        selected_rows: list[Tensor] = []
        hit_ready_rows: list[Tensor] = []
        persistent_selected_rows: list[Tensor] = []
        candidate_valid_rows: list[Tensor] = []
        attention_rows: list[Tensor] = []
        top1_position_rows: list[Tensor] = []
        entropy_rows: list[Tensor] = []
        top1_mass_rows: list[Tensor] = []
        slot_cache_rows: list[Tensor] = []
        slot_attention_rows: list[Tensor] = []
        slot_top1_position_rows: list[Tensor] = []
        slot_sink_rows: list[Tensor] = []
        slot_entropy_rows: list[Tensor] = []
        slot_top1_mass_rows: list[Tensor] = []
        retention_total = 0
        eviction_total = 0
        persistent_total = 0
        block_total = 0
        maximum_candidates = cache_width + config.block_size
        storage_dtype = (
            torch.float32 if config.storage_dtype == "fp32" else torch.bfloat16
        )
        for batch_index in range(batch):
            diagnostic_slots = q_slots if self.config.per_slot_cache_read else 0
            row_cache = torch.zeros(
                steps, heads, value_dim, dtype=torch.float32, device=factors.q.device
            )
            row_sink = torch.zeros(
                steps, heads, dtype=torch.float32, device=factors.q.device
            )
            row_hit_ready = torch.full(
                (steps, heads, maximum_candidates),
                -1,
                dtype=torch.int64,
                device=factors.q.device,
            )
            row_persistent_selected = torch.full(
                (steps, heads, cache_width),
                -1,
                dtype=torch.int64,
                device=factors.q.device,
            )
            row_candidate_valid = torch.zeros(
                steps,
                heads,
                maximum_candidates,
                dtype=torch.bool,
                device=factors.q.device,
            )
            row_attention = torch.zeros(
                steps,
                heads,
                maximum_candidates + 1,
                dtype=torch.float32,
                device=factors.q.device,
            )
            row_top1_positions = torch.full(
                (steps, heads), -1, dtype=torch.int64, device=factors.q.device
            )
            row_attention_entropy = torch.zeros(
                steps, heads, dtype=torch.float32, device=factors.q.device
            )
            row_top1_mass = torch.zeros(
                steps, heads, dtype=torch.float32, device=factors.q.device
            )
            row_slot_cache = torch.zeros(
                steps,
                heads,
                diagnostic_slots,
                value_dim,
                dtype=torch.float32,
                device=factors.q.device,
            )
            row_slot_attention = torch.zeros(
                steps,
                heads,
                diagnostic_slots,
                maximum_candidates + 1,
                dtype=torch.float32,
                device=factors.q.device,
            )
            row_slot_top1_positions = torch.full(
                (steps, heads, diagnostic_slots),
                -1,
                dtype=torch.int64,
                device=factors.q.device,
            )
            row_slot_sink = torch.zeros(
                steps,
                heads,
                diagnostic_slots,
                dtype=torch.float32,
                device=factors.q.device,
            )
            row_slot_entropy = torch.zeros_like(row_slot_sink)
            row_slot_top1_mass = torch.zeros_like(row_slot_sink)
            final_positions = torch.full(
                (heads, cache_width),
                -1,
                dtype=torch.int64,
                device=factors.q.device,
            )
            row_persistent_peak = 0
            row_block_peak = 0
            valid_indices = torch.nonzero(
                factors.valid[batch_index], as_tuple=False
            ).flatten()
            if valid_indices.numel() == 0:
                cache_rows.append(row_cache)
                sink_rows.append(row_sink)
                selected_rows.append(final_positions)
                hit_ready_rows.append(row_hit_ready)
                persistent_selected_rows.append(row_persistent_selected)
                candidate_valid_rows.append(row_candidate_valid)
                attention_rows.append(row_attention)
                top1_position_rows.append(row_top1_positions)
                entropy_rows.append(row_attention_entropy)
                top1_mass_rows.append(row_top1_mass)
                slot_cache_rows.append(row_slot_cache)
                slot_attention_rows.append(row_slot_attention)
                slot_top1_position_rows.append(row_slot_top1_positions)
                slot_sink_rows.append(row_slot_sink)
                slot_entropy_rows.append(row_slot_entropy)
                slot_top1_mass_rows.append(row_slot_top1_mass)
                continue
            segments: list[list[int]] = []
            current_segment: list[int] = []
            for raw_index in valid_indices.tolist():
                if current_segment and bool(boundaries[batch_index, raw_index]):
                    segments.append(current_segment)
                    current_segment = []
                current_segment.append(raw_index)
            segments.append(current_segment)
            for segment in segments:
                persistent = None
                block_start = 0
                while block_start < len(segment):
                    block_stop = min(len(segment), block_start + config.block_size)
                    raw_indices = torch.tensor(
                        segment[block_start:block_stop],
                        dtype=torch.int64,
                        device=factors.q.device,
                    )
                    block_valid = torch.ones(
                        1,
                        raw_indices.numel(),
                        dtype=torch.bool,
                        device=factors.q.device,
                    )
                    def read_queries(query: Tensor):
                        return cache_read_blocks(
                            q_eff=query,
                            query_positions=factors.positions[
                                batch_index : batch_index + 1
                            ].index_select(1, raw_indices),
                            state=persistent,
                            block_k=keys[
                                batch_index : batch_index + 1
                            ].index_select(1, raw_indices),
                            block_v=values[
                                batch_index : batch_index + 1
                            ].index_select(1, raw_indices),
                            block_scores=scores[
                                batch_index : batch_index + 1
                            ].index_select(1, raw_indices),
                            block_positions=factors.positions[
                                batch_index : batch_index + 1
                            ].index_select(1, raw_indices),
                            block_valid=block_valid,
                            config=effective_config,
                            gamma_q=self.cache_gamma_q,
                            gamma_k=self.cache_gamma_k,
                            sink_logit=self.cache_sink_logit,
                        )

                    if self.config.per_slot_cache_read:
                        slot_results = tuple(
                            read_queries(
                                cache_q[
                                    batch_index : batch_index + 1
                                ].index_select(1, raw_indices)[:, :, :, slot]
                            )
                            for slot in range(q_slots)
                        )
                        slot_block = torch.stack(
                            [result[0] for result in slot_results], dim=3
                        )
                        slot_diagnostics = tuple(
                            result[1] for result in slot_results
                        )
                        block_mix = factors.out_mix[
                            batch_index : batch_index + 1
                        ].index_select(1, raw_indices).float()
                        block_output = (
                            slot_block * block_mix.unsqueeze(-1)
                        ).sum(dim=3)
                        diagnostics = slot_diagnostics[0]
                        legacy_attention = torch.stack(
                            [item.attention_weights for item in slot_diagnostics],
                            dim=3,
                        ).mean(dim=3)
                        legacy_sink = torch.stack(
                            [item.sink_mass for item in slot_diagnostics], dim=3
                        ).mean(dim=3)
                        legacy_entropy = torch.stack(
                            [item.attention_entropy for item in slot_diagnostics],
                            dim=3,
                        ).mean(dim=3)
                        legacy_top1_mass = torch.stack(
                            [item.top1_mass for item in slot_diagnostics], dim=3
                        ).mean(dim=3)
                    else:
                        block_output, diagnostics = read_queries(
                            q_eff[
                                batch_index : batch_index + 1
                            ].index_select(1, raw_indices)
                        )
                        slot_block = None
                        slot_diagnostics = ()
                        legacy_attention = diagnostics.attention_weights
                        legacy_sink = diagnostics.sink_mass
                        legacy_entropy = diagnostics.attention_entropy
                        legacy_top1_mass = diagnostics.top1_mass
                    row_cache[raw_indices] = block_output[0]
                    row_sink[raw_indices] = legacy_sink[0]
                    candidate_count = diagnostics.hit_ready_positions.shape[-1]
                    row_hit_ready[raw_indices, :, :candidate_count] = (
                        diagnostics.hit_ready_positions[0]
                    )
                    persistent_width = diagnostics.persistent_selected_positions.shape[-1]
                    row_persistent_selected[
                        raw_indices, :, :persistent_width
                    ] = diagnostics.persistent_selected_positions[0]
                    row_candidate_valid[raw_indices, :, :candidate_count] = (
                        diagnostics.candidate_valid[0]
                    )
                    row_attention[raw_indices, :, :candidate_count] = (
                        legacy_attention[0, ..., :candidate_count]
                    )
                    row_attention[raw_indices, :, -1] = (
                        legacy_attention[0, ..., candidate_count]
                    )
                    row_top1_positions[raw_indices] = diagnostics.top1_positions[0]
                    row_attention_entropy[raw_indices] = legacy_entropy[0]
                    row_top1_mass[raw_indices] = legacy_top1_mass[0]
                    if slot_block is not None:
                        row_slot_cache[raw_indices] = slot_block[0]
                        slot_attention = torch.stack(
                            [item.attention_weights for item in slot_diagnostics],
                            dim=3,
                        )[0]
                        row_slot_attention[
                            raw_indices, :, :, :candidate_count
                        ] = slot_attention[..., :candidate_count]
                        row_slot_attention[raw_indices, :, :, -1] = slot_attention[
                            ..., candidate_count
                        ]
                        row_slot_top1_positions[raw_indices] = torch.stack(
                            [item.top1_positions for item in slot_diagnostics],
                            dim=3,
                        )[0]
                        row_slot_sink[raw_indices] = torch.stack(
                            [item.sink_mass for item in slot_diagnostics], dim=3
                        )[0]
                        row_slot_entropy[raw_indices] = torch.stack(
                            [item.attention_entropy for item in slot_diagnostics],
                            dim=3,
                        )[0]
                        row_slot_top1_mass[raw_indices] = torch.stack(
                            [item.top1_mass for item in slot_diagnostics], dim=3
                        )[0]
                    row_block_peak = max(row_block_peak, diagnostics.block_bytes)
                    prior_valid = (
                        0 if persistent is None else int(persistent.valid.sum().item())
                    )
                    incoming_valid = int(block_valid.sum().item()) * heads
                    persistent = merge_persistent_cache(
                        state=persistent,
                        block_k=keys[batch_index : batch_index + 1].index_select(
                            1, raw_indices
                        ),
                        block_v=values[batch_index : batch_index + 1].index_select(
                            1, raw_indices
                        ),
                        block_scores=scores[
                            batch_index : batch_index + 1
                        ].index_select(1, raw_indices),
                        block_positions=factors.positions[
                            batch_index : batch_index + 1
                        ].index_select(1, raw_indices),
                        block_valid=block_valid,
                        width=cache_width,
                        storage_dtype=storage_dtype,
                    )
                    retained = int(persistent.valid.sum().item())
                    retention_total += retained
                    eviction_total += prior_valid + incoming_valid - retained
                    row_persistent_peak = max(row_persistent_peak, persistent.nbytes)
                    block_start = block_stop
                assert persistent is not None
                final_positions = persistent.positions[0]
            cache_rows.append(row_cache)
            sink_rows.append(row_sink)
            selected_rows.append(final_positions)
            hit_ready_rows.append(row_hit_ready)
            persistent_selected_rows.append(row_persistent_selected)
            candidate_valid_rows.append(row_candidate_valid)
            attention_rows.append(row_attention)
            top1_position_rows.append(row_top1_positions)
            entropy_rows.append(row_attention_entropy)
            top1_mass_rows.append(row_top1_mass)
            slot_cache_rows.append(row_slot_cache)
            slot_attention_rows.append(row_slot_attention)
            slot_top1_position_rows.append(row_slot_top1_positions)
            slot_sink_rows.append(row_slot_sink)
            slot_entropy_rows.append(row_slot_entropy)
            slot_top1_mass_rows.append(row_slot_top1_mass)
            persistent_total += row_persistent_peak
            block_total += row_block_peak
        return (
            torch.stack(cache_rows),
            torch.stack(selected_rows),
            torch.stack(sink_rows),
            torch.stack(hit_ready_rows),
            torch.stack(persistent_selected_rows),
            torch.stack(candidate_valid_rows),
            torch.stack(attention_rows),
            torch.stack(top1_position_rows),
            torch.stack(entropy_rows),
            torch.stack(top1_mass_rows),
            torch.stack(slot_cache_rows),
            torch.stack(slot_attention_rows),
            torch.stack(slot_top1_position_rows),
            torch.stack(slot_sink_rows),
            torch.stack(slot_entropy_rows),
            torch.stack(slot_top1_mass_rows),
            retention_total,
            eviction_total,
            persistent_total,
            block_total,
        )


class TinyFactorProjector(nn.Module):
    def __init__(self, config: TinyKMD2Config):
        super().__init__()
        self.config = config
        h, dk, dv = config.heads, config.dk, config.dv
        rank = config.mimo_rank
        factor_dk = dk - 1 if config.bc_bias_mode == "constant_coordinate_oracle" else dk
        self.factor_dk = factor_dk
        self.q_proj = nn.Linear(config.d_model, h * rank * factor_dk, bias=False)
        self.k_proj = nn.Linear(config.d_model, h * rank * factor_dk, bias=False)
        # Mamba-3 keeps one base V/Z projection per head and expands it across
        # the MIMO rank with cheap, data-independent channelwise scalings.
        self.v_proj = nn.Linear(config.d_model, h * dv, bias=False)
        self.z_proj = nn.Linear(config.d_model, h * dv, bias=False)
        self.a_proj = nn.Linear(config.d_model, h, bias=False)
        self.b_proj = nn.Linear(config.d_model, h * rank, bias=False)
        mixed_dim = h * (2 * rank * factor_dk + dv)
        self.conv = nn.Conv1d(
            mixed_dim,
            mixed_dim,
            config.conv_kernel,
            groups=mixed_dim,
            bias=False,
            padding=config.conv_kernel - 1,
        )
        self.convolution_gate = nn.Parameter(torch.tensor(config.convolution_gate_init))
        self.rotation_gate = nn.Parameter(torch.tensor(config.rotation_gate_init))
        self.channel_decay_gate = nn.Parameter(
            torch.tensor(config.channel_decay_gate_init)
        )
        self.write_offset_gate = nn.Parameter(
            torch.tensor(config.write_offset_gate_init)
        )
        if config.trapezoid:
            self.rho_head = nn.Parameter(
                torch.full((h,), config.trapezoid_gate_init, dtype=torch.float32)
            )
            self.rho_proj = nn.Linear(config.d_model, h, bias=False)
            nn.init.zeros_(self.rho_proj.weight)
        if config.corrected_momentum:
            self.momentum_gamma = nn.Parameter(
                torch.full((h,), config.momentum_gamma_init, dtype=torch.float32)
            )
        if config.causal_lookahead:
            self.lookahead_rho = nn.Parameter(
                torch.full((h,), config.lookahead_rho_init, dtype=torch.float32)
            )
        self.A_log = nn.Parameter(torch.zeros(h))
        self.dt_bias = nn.Parameter(torch.ones(h))
        self.decay_chan = nn.Parameter(torch.zeros(h, factor_dk))
        self.bw_off = nn.Parameter(torch.zeros(h, rank))
        if rank == 1:
            q_slots = config.r_out
            self.q_slot_scale = nn.Parameter(torch.zeros(h, q_slots, factor_dk))
            mix = torch.zeros(h, q_slots)
            mix[:, 0] = 1.0
            self.out_mix = nn.Parameter(mix)
        else:
            # Match the released Mamba-3 parameterization: V is expanded from
            # one base input, Z gates each rank before contraction, and O is a
            # channelwise rank-down projection. 1/R initialization keeps the
            # initial aggregate scale controlled without an artificial norm.
            self.mimo_v = nn.Parameter(torch.full((h, rank, dv), 1.0 / rank))
            self.mimo_z = nn.Parameter(torch.ones(h, rank, dv))
            self.mimo_out = nn.Parameter(torch.full((h, rank, dv), 1.0 / rank))
        if config.rotation_mode != "none":
            self.rot_proj = nn.Linear(config.d_model, h * (dk // 2), bias=True)
            nn.init.zeros_(self.rot_proj.weight)
            nn.init.constant_(self.rot_proj.bias, -2.0)
            self.rotation_rate = nn.Parameter(torch.full((h, dk // 2), 0.01))

    @staticmethod
    def _rope(value: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
        half = value.shape[-1] // 2
        first, second = value[..., :half], value[..., half:]
        return torch.cat(
            (first * cosine - second * sine, first * sine + second * cosine),
            dim=-1,
        )

    def _segmented_causal_conv(
        self, values: Tensor, valid: Tensor, positions: Tensor
    ) -> Tensor:
        kernel = self.conv.weight[:, 0]
        width = kernel.shape[-1]
        batches: list[Tensor] = []
        for batch_index in range(values.shape[0]):
            tokens: list[Tensor] = []
            logical_sources: dict[int, int] = {}
            for token_index in range(values.shape[1]):
                if not bool(valid[batch_index, token_index]):
                    tokens.append(torch.zeros_like(values[batch_index, token_index]))
                    continue
                current_position = int(positions[batch_index, token_index])
                if current_position == 0:
                    logical_sources.clear()
                logical_sources[current_position] = token_index
                accumulated = torch.zeros_like(values[batch_index, token_index])
                for lag in range(width):
                    source = logical_sources.get(current_position - lag)
                    if source is None:
                        break
                    accumulated = accumulated + (
                        values[batch_index, source] * kernel[:, width - 1 - lag]
                    )
                tokens.append(accumulated)
            batches.append(torch.stack(tokens))
        return torch.stack(batches)

    @staticmethod
    def _segmented_cumsum(values: Tensor, valid: Tensor, positions: Tensor) -> Tensor:
        batches: list[Tensor] = []
        for batch_index in range(values.shape[0]):
            running = torch.zeros_like(values[batch_index, 0])
            tokens: list[Tensor] = []
            for token_index in range(values.shape[1]):
                if not bool(valid[batch_index, token_index]):
                    tokens.append(torch.zeros_like(running))
                    continue
                if int(positions[batch_index, token_index]) == 0:
                    running = torch.zeros_like(running)
                running = running + values[batch_index, token_index]
                tokens.append(running)
            batches.append(torch.stack(tokens))
        return torch.stack(batches)

    def forward(self, hidden: Tensor, valid: Tensor, positions: Tensor) -> TinyFactors:
        batch, steps, _ = hidden.shape
        c = self.config
        h, rank, dk, dv = c.heads, c.mimo_rank, c.dk, c.dv
        factor_dk = self.factor_dk
        hidden = torch.where(valid.unsqueeze(-1), hidden, torch.zeros_like(hidden))
        q_raw = self.q_proj(hidden).view(batch, steps, h, rank, factor_dk)
        k_raw = self.k_proj(hidden).view(batch, steps, h, rank, factor_dk)
        v_raw = self.v_proj(hidden).view(batch, steps, h, dv)
        z_base = self.z_proj(hidden).float().view(batch, steps, h, dv)
        flattened = torch.cat(
            (q_raw.flatten(2), k_raw.flatten(2), v_raw.flatten(2)), dim=-1
        )
        base = F.silu(flattened)
        convolved = F.silu(self._segmented_causal_conv(flattened, valid, positions))
        mixed = base + self.convolution_gate * (convolved - base)
        q_count = h * rank * factor_dk
        k_count = h * rank * factor_dk
        q_raw, k_raw, v_raw = torch.split(
            mixed, (q_count, k_count, h * dv), dim=-1
        )
        q_base = F.normalize(
            q_raw.view(batch, steps, h, rank, factor_dk).float(), dim=-1, eps=c.eps
        ) * (factor_dk**-0.5)
        k = F.normalize(
            k_raw.view(batch, steps, h, rank, factor_dk).float(), dim=-1, eps=c.eps
        )
        v_base = v_raw.view(batch, steps, h, dv).float()
        q_slots = c.r_out if rank == 1 else rank
        if rank == 1:
            q = q_base * (1.0 + self.q_slot_scale[None, None])
            v = v_base.unsqueeze(3)
            read_gate = z_base.unsqueeze(3).expand(batch, steps, h, q_slots, dv)
            out_mix = self.out_mix.float()[None, None].expand(
                batch, steps, h, q_slots
            )
        else:
            q = q_base
            v = v_base.unsqueeze(3) * self.mimo_v.float()[None, None]
            read_gate = z_base.unsqueeze(3) * self.mimo_z.float()[None, None]
            out_mix = self.mimo_out.float()[None, None].expand(
                batch, steps, h, rank, dv
            )

        if c.bc_bias_mode == "constant_coordinate_oracle":
            q, k = append_constant_coordinate(q, k)

        cache_q = None
        cache_k = None
        if c.cache is not None and c.cache.coordinate_frame == "pre_rotation":
            cache_q = q
            cache_k = k

        moving_frame_phase = None
        if c.rotation_mode != "none":
            theta_data = F.softplus(self.rot_proj(hidden)).view(
                batch, steps, h, dk // 2
            )
            if c.rotation_mode == "constant_rate":
                theta = positions.clamp_min(0).to(theta_data.dtype)[..., None, None]
                theta = theta * self.rotation_rate[None, None]
            elif c.rotation_mode == "non_cumulative":
                theta = theta_data
            elif c.rotation_mode == "fixed_rope":
                frequencies = torch.exp(
                    -math.log(10000.0)
                    * torch.arange(dk // 2, device=hidden.device, dtype=torch.float32)
                    / max(1, dk // 2)
                )
                theta = positions.clamp_min(0).float()[..., None, None] * frequencies
                theta = theta.expand(batch, steps, h, dk // 2)
            else:
                theta = self._segmented_cumsum(theta_data, valid, positions)
            theta = self.rotation_gate * theta
            if c.rotation_mode == "moving_frame":
                moving_frame_phase = theta
            else:
                q = self._rope(q, theta.cos().unsqueeze(-2), theta.sin().unsqueeze(-2))
                k = self._rope(k, theta.cos().unsqueeze(-2), theta.sin().unsqueeze(-2))

        a = self.a_proj(hidden).float().view(batch, steps, h)
        b = self.b_proj(hidden).float().view(batch, steps, h, rank)
        g_head = -self.A_log.float().exp() * F.softplus(a + self.dt_bias.float())
        decay = (
            g_head.unsqueeze(-1)
            + self.channel_decay_gate * self.decay_chan.float()[None, None]
        ).exp().clamp(max=1.0)
        if c.bc_bias_mode == "constant_coordinate_oracle":
            constant_decay = g_head.exp().clamp(max=1.0).unsqueeze(-1)
            decay = torch.cat((decay, constant_decay), dim=-1)
        beta_e = torch.sigmoid(b)
        beta_w = torch.sigmoid(
            b + self.write_offset_gate * self.bw_off.float()[None, None]
        )
        trapezoid_rho = None
        if c.trapezoid:
            trapezoid_rho = self.rho_head.float()[None, None] * torch.sigmoid(
                self.rho_proj(hidden).float()
            )
            trapezoid_rho = torch.where(
                valid.unsqueeze(-1), trapezoid_rho, torch.zeros_like(trapezoid_rho)
            )
        momentum_gamma = None
        if c.corrected_momentum:
            momentum_gamma = self.momentum_gamma.float()[None, None].expand(
                batch, steps, h
            )
            momentum_gamma = torch.where(
                valid.unsqueeze(-1), momentum_gamma, torch.zeros_like(momentum_gamma)
            )
        lookahead_rho = None
        if c.causal_lookahead:
            lookahead_rho = self.lookahead_rho.float()[None, None].expand(
                batch, steps, h
            )
            lookahead_rho = torch.where(
                valid.unsqueeze(-1), lookahead_rho, torch.zeros_like(lookahead_rho)
            )
        return TinyFactors(
            q=q,
            k=k,
            v=v,
            decay=decay,
            beta_e=beta_e,
            beta_w=beta_w,
            out_mix=out_mix,
            valid=valid,
            positions=positions,
            read_gate=read_gate,
            trapezoid_rho=trapezoid_rho,
            momentum_gamma=momentum_gamma,
            lookahead_rho=lookahead_rho,
            moving_frame_phase=moving_frame_phase,
            cache_q=cache_q,
            cache_k=cache_k,
        )


def project_trapezoid_gates_(module: nn.Module) -> None:
    """Project every trapezoid head gate in ``module`` onto its valid interval."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.rsplit(".", 1)[-1] == "rho_head":
                parameter.clamp_(0.0, 1.0)


def project_momentum_gates_(module: nn.Module) -> None:
    """Project every corrected-momentum coefficient onto its valid interval."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.rsplit(".", 1)[-1] == "momentum_gamma":
                parameter.clamp_(0.0, 1.0)


def project_lookahead_gates_(module: nn.Module) -> None:
    """Project every causal-lookahead coefficient onto its valid interval."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.rsplit(".", 1)[-1] == "lookahead_rho":
                parameter.clamp_(0.0, 1.0)


class TinyKMD2Block(nn.Module):
    def __init__(self, config: TinyKMD2Config):
        super().__init__()
        self.norm = nn.RMSNorm(config.d_model, eps=config.eps)
        self.projector = TinyFactorProjector(config)
        self.cell = TinyKMD2Cell(config)
        self.out_proj = nn.Linear(config.heads * config.dv, config.d_model, bias=False)
        self.ffn_norm = nn.RMSNorm(config.d_model, eps=config.eps)
        self.ffn_up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.ffn_down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(
        self,
        hidden: Tensor,
        valid: Tensor,
        positions: Tensor,
        boundaries: Tensor | None,
        future_relevance: Tensor | None = None,
    ) -> tuple[Tensor, TinyCellOutput]:
        _validate_cache_diagnostic_preallocation(
            self.cell.config,
            batch=hidden.shape[0],
            steps=hidden.shape[1],
        )
        factors = self.projector(self.norm(hidden), valid, positions)
        cell_output = self.cell(
            factors,
            boundaries=boundaries,
            future_relevance=future_relevance,
        )
        merged = cell_output.read.reshape(hidden.shape[0], hidden.shape[1], -1)
        hidden = hidden + self.out_proj(merged.to(hidden.dtype))
        hidden = hidden + self.ffn_down(F.silu(self.ffn_up(self.ffn_norm(hidden))))
        hidden = torch.where(valid.unsqueeze(-1), hidden, torch.zeros_like(hidden))
        return hidden, cell_output


class TinyKMD2Model(nn.Module):
    def __init__(self, config: TinyKMD2Config, *, init_seed: int = 0):
        super().__init__()
        if not isinstance(config, TinyKMD2Config):
            raise TypeError("config must be TinyKMD2Config")
        if type(init_seed) is not int:
            raise TypeError("init_seed must be an int")
        self.config = config
        self.init_seed = init_seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.continuous_projection = (
                None
                if config.continuous_input_dim is None
                else nn.Linear(config.continuous_input_dim, config.d_model, bias=False)
            )
            self.blocks = nn.ModuleList(
                TinyKMD2Block(config) for _ in range(config.layers)
            )
            self.final_norm = nn.RMSNorm(config.d_model, eps=config.eps)
            self.token_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.regression_head = (
                None
                if config.output_dim is None
                else nn.Linear(config.d_model, config.output_dim, bias=False)
            )
            self.direct_head = (
                None
                if config.output_dim is None
                else nn.Linear(config.heads * config.dv, config.output_dim, bias=False)
            )
        self.to(dtype=config.dtype)

    @staticmethod
    def _default_sequence_metadata(
        shape: tuple[int, int], device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, steps = shape
        valid = torch.ones(batch, steps, dtype=torch.bool, device=device)
        positions = torch.arange(steps, dtype=torch.int64, device=device).repeat(batch, 1)
        boundaries = torch.zeros(batch, steps, dtype=torch.bool, device=device)
        boundaries[:, 0] = True
        return valid, positions, boundaries

    @staticmethod
    def _compute_loss(logits: Tensor, targets: Tensor, loss_mask: Tensor) -> Tensor:
        if loss_mask.dtype != torch.bool or loss_mask.shape != logits.shape[:2]:
            raise ValueError("loss_mask must be bool with shape [B,T]")
        if not bool(loss_mask.any()):
            raise ValueError("loss_mask must select at least one target")
        if targets.ndim == 2:
            if targets.dtype != torch.int64 or targets.shape != logits.shape[:2]:
                raise ValueError("class targets must be int64 with shape [B,T]")
            return F.cross_entropy(logits[loss_mask], targets[loss_mask])
        if targets.ndim == 3:
            if not targets.is_floating_point() or targets.shape != logits.shape:
                raise ValueError("regression targets must match logits [B,T,Dout]")
            if not bool(torch.isfinite(logits[loss_mask]).all()):
                raise ValueError("selected regression logits must be finite")
            if not bool(torch.isfinite(targets[loss_mask]).all()):
                raise ValueError("selected regression targets must be finite")
            return F.mse_loss(logits[loss_mask], targets[loss_mask].to(logits.dtype))
        raise ValueError("targets must have shape [B,T] or [B,T,Dout]")

    @staticmethod
    def _validate_sequence_layout(
        valid: Tensor, positions: Tensor, boundaries: Tensor | None
    ) -> None:
        _validate_sequence_layout(valid, positions, boundaries)

    def forward(
        self,
        input_ids: Tensor | None = None,
        continuous_inputs: Tensor | None = None,
        factors: TinyFactors | None = None,
        targets: Tensor | None = None,
        loss_mask: Tensor | None = None,
        boundaries: Tensor | None = None,
        valid: Tensor | None = None,
        positions: Tensor | None = None,
        future_relevance: Tensor | None = None,
    ) -> TinyModelOutput:
        if sum(item is not None for item in (input_ids, continuous_inputs, factors)) != 1:
            raise ValueError("exactly one input modality must be provided")
        if factors is not None:
            if valid is not None or positions is not None:
                raise ValueError("factors already contain valid and positions")
            if self.config.layers != 1:
                raise ValueError("direct-factor execution requires layers=1")
            _validate_cache_diagnostic_preallocation(
                self.config,
                batch=factors.q.shape[0],
                steps=factors.q.shape[1],
            )
            self._validate_sequence_layout(
                factors.valid, factors.positions, boundaries
            )
            cell_output = self.blocks[0].cell(
                factors,
                boundaries=boundaries,
                future_relevance=future_relevance,
            )
            if self.direct_head is None:
                raise ValueError("direct-factor execution requires output_dim")
            merged = cell_output.read.reshape(factors.q.shape[0], factors.q.shape[1], -1)
            logits = self.direct_head(merged.to(self.direct_head.weight.dtype))
            outputs = (cell_output,)
            effective_valid = factors.valid
        else:
            source = input_ids if input_ids is not None else continuous_inputs
            assert source is not None
            if source.ndim < 2:
                raise ValueError("input modality must have [B,T] leading dimensions")
            batch, steps = source.shape[:2]
            _validate_cache_diagnostic_preallocation(
                self.config, batch=batch, steps=steps
            )
            if valid is None or positions is None:
                default_valid, default_positions, default_boundaries = (
                    self._default_sequence_metadata((batch, steps), source.device)
                )
                valid = default_valid if valid is None else valid
                positions = default_positions if positions is None else positions
                boundaries = default_boundaries if boundaries is None else boundaries
            self._validate_sequence_layout(valid, positions, boundaries)
            if input_ids is not None:
                if input_ids.dtype != torch.int64 or input_ids.ndim != 2:
                    raise ValueError("input_ids must be int64 with shape [B,T]")
                hidden = self.token_embedding(input_ids)
                use_regression_head = False
            else:
                assert continuous_inputs is not None
                if self.continuous_projection is None:
                    raise ValueError("continuous_input_dim is not configured")
                if (
                    not continuous_inputs.is_floating_point()
                    or continuous_inputs.ndim != 3
                    or continuous_inputs.shape[-1] != self.config.continuous_input_dim
                ):
                    raise ValueError("continuous_inputs must match configured [B,T,D]")
                hidden = self.continuous_projection(
                    continuous_inputs.to(self.continuous_projection.weight.dtype)
                )
                use_regression_head = True
            hidden = torch.where(valid.unsqueeze(-1), hidden, torch.zeros_like(hidden))
            cell_outputs: list[TinyCellOutput] = []
            for block in self.blocks:
                hidden, cell_output = block(
                    hidden,
                    valid,
                    positions,
                    boundaries,
                    future_relevance,
                )
                cell_outputs.append(cell_output)
            hidden = self.final_norm(hidden)
            if use_regression_head:
                if self.regression_head is None:
                    raise ValueError("continuous inputs require output_dim")
                logits = self.regression_head(hidden)
            else:
                logits = self.token_head(hidden)
            outputs = tuple(cell_outputs)
            effective_valid = valid
        loss = None
        if targets is not None:
            if loss_mask is None:
                raise ValueError("targets require loss_mask")
            if bool((loss_mask & ~effective_valid).any()):
                raise ValueError("loss_mask must be a subset of valid")
            loss = self._compute_loss(logits, targets, loss_mask)
        elif loss_mask is not None:
            raise ValueError("loss_mask requires targets")
        return TinyModelOutput(
            logits=logits,
            loss=loss,
            final_states=tuple(output.final_state for output in outputs),
            cell_outputs=outputs,
        )

    def forward_episode(self, episode: Any) -> TinyModelOutput:
        factors = (
            tiny_factors_from_episode(episode)
            if getattr(episode, "direct_factors", None) is not None
            else None
        )
        cache = self.config.cache
        relevance = (
            future_query_relevance(episode)
            if cache is not None and cache.score == "future_query_oracle"
            else None
        )
        return self(
            input_ids=getattr(episode, "input_ids", None),
            continuous_inputs=getattr(episode, "continuous_inputs", None),
            factors=factors,
            targets=episode.targets,
            loss_mask=episode.loss_mask,
            boundaries=episode.boundaries,
            valid=None if factors is not None else episode.valid,
            positions=None if factors is not None else episode.positions,
            future_relevance=relevance,
        )


def tiny_factors_from_episode(episode: Any) -> TinyFactors:
    mapping = getattr(episode, "direct_factors", None)
    if not isinstance(mapping, Mapping):
        raise TypeError("episode.direct_factors must be a mapping")
    required = {"q", "k", "v", "decay", "beta_e", "beta_w", "out_mix"}
    allowed = required | {
        "write_mask",
        "query_role",
        "cache_q",
        "cache_k",
        "read_gate",
    }
    missing = required - set(mapping)
    unknown = set(mapping) - allowed
    if missing or unknown:
        raise ValueError(
            f"direct factor keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    query_role = mapping.get("query_role")
    if query_role is not None and not torch.equal(query_role, episode.query_mask):
        raise ValueError("direct factor query_role must equal episode.query_mask")
    write_mask = mapping.get("write_mask")
    if write_mask is not None and bool((write_mask & episode.query_mask).any()):
        raise ValueError("direct factor write_mask and query_mask must be disjoint")
    return TinyFactors(
        q=mapping["q"],
        k=mapping["k"],
        v=mapping["v"],
        decay=mapping["decay"],
        beta_e=mapping["beta_e"],
        beta_w=mapping["beta_w"],
        out_mix=mapping["out_mix"],
        valid=episode.valid,
        positions=episode.positions,
        read_gate=mapping.get("read_gate"),
        cache_q=mapping.get("cache_q"),
        cache_k=mapping.get("cache_k"),
    )


__all__ = [
    "CacheDiagnosticBudgetError",
    "TINY_BACKEND_SCHEMA_VERSION",
    "TinyCellOutput",
    "TinyFactors",
    "TinyKMD2Cell",
    "TinyKMD2Config",
    "TinyKMD2Model",
    "TinyModelOutput",
    "append_constant_coordinate",
    "apply_bc_additive",
    "apply_bc_diagonal_rescale",
    "future_query_relevance",
    "moving_frame_transport_diagnostic",
    "tiny_factors_from_episode",
    "true_mimo_update",
]
