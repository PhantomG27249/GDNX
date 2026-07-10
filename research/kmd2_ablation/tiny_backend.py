"""Small, dependency-light PyTorch backend for causal KMD-2 experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import CacheConfig
from .exact_cache import (
    cache_read_blocks,
    initialize_cache_read_parameters,
    merge_persistent_cache,
)


TINY_BACKEND_SCHEMA_VERSION = "1.0.0"
_ROTATION_MODES = {
    "none",
    "current",
    "constant_rate",
    "non_cumulative",
    "fixed_rope",
    "moving_frame",
}
_FLOAT_DTYPES = {torch.float32, torch.float64, torch.bfloat16}


def _positive_int(name: str, value: object) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive int")


def _unit_gate(name: str, value: object) -> None:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise TypeError(f"{name} must be a finite number")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0,1]")


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
        ):
            _unit_gate(name, getattr(self, name))
            object.__setattr__(self, name, float(getattr(self, name)))
        if type(self.trapezoid) is not bool:
            raise TypeError("trapezoid must be a bool")
        object.__setattr__(self, "eps", float(self.eps))
        if self.cache is not None and not isinstance(self.cache, CacheConfig):
            raise TypeError("cache must be a CacheConfig or None")
        if self.cache is not None and self.mimo_rank != 1:
            raise ValueError("exact cache currently requires mimo_rank=1")
        if self.cache is not None and self.cache.score != "exact_outer":
            raise NotImplementedError(
                "tiny exact cache supports score=exact_outer only; "
                "other score policies require Task 9"
            )
        if self.cache is not None and (
            self.cache.coordinate_frame != "rotated_recurrence"
            or self.cache.pre_rotation_diagnostic
        ):
            raise NotImplementedError(
                "tiny exact cache supports rotated_recurrence coordinates only; "
                "pre-rotation cache modes require Task 9"
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
    trapezoid_rho: Tensor | None = None

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
            "out_mix": (batch, steps, heads, q_slots),
            "valid": (batch, steps),
            "positions": (batch, steps),
        }
        for name, shape in expected_shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
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
        if (heads, key_dim, value_dim) != (
            self.config.heads,
            self.config.dk,
            self.config.dv,
        ):
            raise ValueError("factor head/dk/dv dimensions must match config")
        if self.config.mimo_rank == 1:
            if write_slots != 1 or q_slots != self.config.r_out:
                raise ValueError("native factors require R=1 and Q=config.r_out")
        elif write_slots != self.config.mimo_rank or q_slots != self.config.mimo_rank:
            raise ValueError("true-MIMO factors require R=Q=config.mimo_rank")
        if self.config.trapezoid and factors.trapezoid_rho is None:
            raise ValueError("trapezoid factors require trapezoid_rho")
        if not self.config.trapezoid and factors.trapezoid_rho is not None:
            raise ValueError("trapezoid_rho requires config.trapezoid=true")

    def forward(
        self,
        factors: TinyFactors,
        state: Tensor | None = None,
        boundaries: Tensor | None = None,
    ) -> TinyCellOutput:
        device_type = factors.q.device.type if isinstance(factors, TinyFactors) else "cpu"
        if device_type in {"cpu", "cuda"}:
            with torch.autocast(device_type=device_type, enabled=False):
                return self._forward_fp32(factors, state, boundaries)
        return self._forward_fp32(factors, state, boundaries)

    def _forward_fp32(
        self,
        factors: TinyFactors,
        state: Tensor | None,
        boundaries: Tensor | None,
    ) -> TinyCellOutput:
        self._validate_factors(factors)
        if self.config.mimo_rank > 1:
            raise NotImplementedError(
                "true MIMO requires Task 9 variant implementation"
            )
        batch, steps, heads, q_slots, write_slots, key_dim, value_dim = factors.shape
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
        trapezoid_rho = (
            None if factors.trapezoid_rho is None else factors.trapezoid_rho.float()
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
        outputs: list[Tensor] = []
        scores: list[Tensor] = []
        for token in range(steps):
            current = torch.where(
                reset[:, token, None, None, None],
                torch.zeros((), dtype=torch.float32, device=device),
                current,
            )
            state_bar = decay[:, token].unsqueeze(-1) * current
            key = k[:, token]
            value = v[:, token]
            if write_slots == 1:
                key_one = key[:, :, 0]
                value_one = value[:, :, 0]
                memory = torch.matmul(key_one.unsqueeze(-2), state_bar).squeeze(-2)
                update = (
                    beta_w[:, token, :, 0].unsqueeze(-1) * value_one
                    - beta_e[:, token, :, 0].unsqueeze(-1) * memory
                )
                native_outer = key_one.unsqueeze(-1) * update.unsqueeze(-2)
                candidate = state_bar + native_outer
                native_score = torch.linalg.vector_norm(
                    key_one, dim=-1
                ) * torch.linalg.vector_norm(update, dim=-1)
                if trapezoid_rho is not None:
                    assert previous_key is not None and previous_write is not None
                    current_write_value = (
                        beta_w[:, token, :, 0].unsqueeze(-1) * value_one
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
                erase_scale = torch.sqrt(beta_e[:, token] / write_slots).unsqueeze(-1)
                erase_key = erase_scale * key
                erase_memory = torch.matmul(erase_key, state_bar)
                erase = torch.einsum("bhrd,bhrv->bhdv", erase_key, erase_memory)
                write = torch.einsum(
                    "bhrd,bhrv->bhdv",
                    key,
                    beta_w[:, token].unsqueeze(-1) * value,
                ) / math.sqrt(write_slots)
                candidate = state_bar - erase + write
                score = torch.linalg.matrix_norm(candidate - state_bar, dim=(-2, -1))
            active = factors.valid[:, token, None, None, None]
            current = torch.where(active, candidate, current)
            if write_slots == 1 and trapezoid_rho is not None:
                assert previous_key is not None and previous_write is not None
                carry_active = factors.valid[:, token, None, None]
                previous_key = torch.where(carry_active, key_one, previous_key)
                previous_write = torch.where(
                    carry_active, current_write_value, previous_write
                )
            slot_read = torch.matmul(q[:, token], current)
            read = (slot_read * out_mix[:, token].unsqueeze(-1)).sum(dim=-2)
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
        if self.config.cache is None:
            cache_read = torch.zeros_like(state_read)
            selected_positions = torch.empty(
                batch, heads, 0, dtype=torch.int64, device=device
            )
            sink_mass = torch.zeros(
                batch, steps, heads, dtype=torch.float32, device=device
            )
            cache_persistent_bytes = 0
            cache_block_bytes = 0
        else:
            (
                cache_read,
                selected_positions,
                sink_mass,
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
            state_bytes=current.numel() * current.element_size(),
            cache_persistent_bytes=cache_persistent_bytes,
            cache_block_bytes=cache_block_bytes,
        )

    def _cache_forward(
        self, factors: TinyFactors, scores: Tensor, boundaries: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, int, int]:
        config = self.config.cache
        assert config is not None
        batch, steps, heads, _, _, key_dim, value_dim = factors.shape
        q_eff = torch.einsum(
            "bthqd,bthq->bthd", factors.q.float(), factors.out_mix.float()
        )
        keys = factors.k[:, :, :, 0].float()
        values = factors.v[:, :, :, 0].float()
        cache_rows: list[Tensor] = []
        sink_rows: list[Tensor] = []
        selected_rows: list[Tensor] = []
        persistent_total = 0
        block_total = 0
        storage_dtype = (
            torch.float32 if config.storage_dtype == "fp32" else torch.bfloat16
        )
        for batch_index in range(batch):
            row_cache = torch.zeros(
                steps, heads, value_dim, dtype=torch.float32, device=factors.q.device
            )
            row_sink = torch.zeros(
                steps, heads, dtype=torch.float32, device=factors.q.device
            )
            final_positions = torch.full(
                (heads, config.width),
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
                    block_output, diagnostics = cache_read_blocks(
                        q_eff=q_eff[batch_index : batch_index + 1].index_select(
                            1, raw_indices
                        ),
                        query_positions=factors.positions[
                            batch_index : batch_index + 1
                        ].index_select(1, raw_indices),
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
                        config=config,
                        gamma_q=self.cache_gamma_q,
                        gamma_k=self.cache_gamma_k,
                        sink_logit=self.cache_sink_logit,
                    )
                    row_cache[raw_indices] = block_output[0]
                    row_sink[raw_indices] = diagnostics.sink_mass[0]
                    row_block_peak = max(row_block_peak, diagnostics.block_bytes)
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
                        width=config.width,
                        storage_dtype=storage_dtype,
                    )
                    row_persistent_peak = max(row_persistent_peak, persistent.nbytes)
                    block_start = block_stop
                assert persistent is not None
                final_positions = persistent.positions[0]
            cache_rows.append(row_cache)
            sink_rows.append(row_sink)
            selected_rows.append(final_positions)
            persistent_total += row_persistent_peak
            block_total += row_block_peak
        return (
            torch.stack(cache_rows),
            torch.stack(selected_rows),
            torch.stack(sink_rows),
            persistent_total,
            block_total,
        )


class TinyFactorProjector(nn.Module):
    def __init__(self, config: TinyKMD2Config):
        super().__init__()
        self.config = config
        h, dk, dv = config.heads, config.dk, config.dv
        rank = config.mimo_rank
        self.q_proj = nn.Linear(config.d_model, h * rank * dk, bias=False)
        self.k_proj = nn.Linear(config.d_model, h * rank * dk, bias=False)
        self.v_proj = nn.Linear(config.d_model, h * rank * dv, bias=False)
        self.a_proj = nn.Linear(config.d_model, h, bias=False)
        self.b_proj = nn.Linear(config.d_model, h * rank, bias=False)
        mixed_dim = h * rank * (2 * dk + dv)
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
        self.A_log = nn.Parameter(torch.zeros(h))
        self.dt_bias = nn.Parameter(torch.ones(h))
        self.decay_chan = nn.Parameter(torch.zeros(h, dk))
        self.bw_off = nn.Parameter(torch.zeros(h, rank))
        q_slots = config.r_out if rank == 1 else rank
        self.q_slot_scale = nn.Parameter(torch.zeros(h, q_slots, dk))
        mix = torch.zeros(h, q_slots)
        if rank == 1:
            mix[:, 0] = 1.0
        else:
            mix.fill_(1.0 / math.sqrt(rank))
        self.out_mix = nn.Parameter(mix)
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
        hidden = torch.where(valid.unsqueeze(-1), hidden, torch.zeros_like(hidden))
        q_raw = self.q_proj(hidden).view(batch, steps, h, rank, dk)
        k_raw = self.k_proj(hidden).view(batch, steps, h, rank, dk)
        v_raw = self.v_proj(hidden).view(batch, steps, h, rank, dv)
        flattened = torch.cat(
            (q_raw.flatten(2), k_raw.flatten(2), v_raw.flatten(2)), dim=-1
        )
        base = F.silu(flattened)
        convolved = F.silu(self._segmented_causal_conv(flattened, valid, positions))
        mixed = base + self.convolution_gate * (convolved - base)
        q_count = h * rank * dk
        k_count = h * rank * dk
        q_raw, k_raw, v_raw = torch.split(
            mixed, (q_count, k_count, h * rank * dv), dim=-1
        )
        q_base = F.normalize(
            q_raw.view(batch, steps, h, rank, dk).float(), dim=-1, eps=c.eps
        ) * (dk**-0.5)
        k = F.normalize(
            k_raw.view(batch, steps, h, rank, dk).float(), dim=-1, eps=c.eps
        )
        v = v_raw.view(batch, steps, h, rank, dv).float()
        q_slots = c.r_out if rank == 1 else rank
        if rank == 1:
            q = q_base * (1.0 + self.q_slot_scale[None, None])
        else:
            q = q_base

        if c.rotation_mode == "moving_frame":
            if bool(self.rotation_gate.detach() != 0):
                raise NotImplementedError(
                    "active moving-frame rotation requires Task 9 variant implementation"
                )
        elif c.rotation_mode != "none":
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
            q = self._rope(q, theta.cos().unsqueeze(-2), theta.sin().unsqueeze(-2))
            k = self._rope(k, theta.cos().unsqueeze(-2), theta.sin().unsqueeze(-2))

        a = self.a_proj(hidden).float().view(batch, steps, h)
        b = self.b_proj(hidden).float().view(batch, steps, h, rank)
        g_head = -self.A_log.float().exp() * F.softplus(a + self.dt_bias.float())
        decay = (
            g_head.unsqueeze(-1)
            + self.channel_decay_gate * self.decay_chan.float()[None, None]
        ).exp().clamp(max=1.0)
        beta_e = torch.sigmoid(b)
        beta_w = torch.sigmoid(
            b + self.write_offset_gate * self.bw_off.float()[None, None]
        )
        out_mix = self.out_mix.float()[None, None].expand(
            batch, steps, h, q_slots
        )
        trapezoid_rho = None
        if c.trapezoid:
            trapezoid_rho = self.rho_head.float()[None, None] * torch.sigmoid(
                self.rho_proj(hidden).float()
            )
            trapezoid_rho = torch.where(
                valid.unsqueeze(-1), trapezoid_rho, torch.zeros_like(trapezoid_rho)
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
            trapezoid_rho=trapezoid_rho,
        )


def project_trapezoid_gates_(module: nn.Module) -> None:
    """Project every trapezoid head gate in ``module`` onto its valid interval."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.rsplit(".", 1)[-1] == "rho_head":
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
    ) -> tuple[Tensor, TinyCellOutput]:
        factors = self.projector(self.norm(hidden), valid, positions)
        cell_output = self.cell(factors, boundaries=boundaries)
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
    ) -> TinyModelOutput:
        if sum(item is not None for item in (input_ids, continuous_inputs, factors)) != 1:
            raise ValueError("exactly one input modality must be provided")
        if factors is not None:
            if valid is not None or positions is not None:
                raise ValueError("factors already contain valid and positions")
            if self.config.layers != 1:
                raise ValueError("direct-factor execution requires layers=1")
            self._validate_sequence_layout(
                factors.valid, factors.positions, boundaries
            )
            cell_output = self.blocks[0].cell(factors, boundaries=boundaries)
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
                hidden, cell_output = block(hidden, valid, positions, boundaries)
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
        return self(
            input_ids=getattr(episode, "input_ids", None),
            continuous_inputs=getattr(episode, "continuous_inputs", None),
            factors=factors,
            targets=episode.targets,
            loss_mask=episode.loss_mask,
            boundaries=episode.boundaries,
            valid=None if factors is not None else episode.valid,
            positions=None if factors is not None else episode.positions,
        )


def tiny_factors_from_episode(episode: Any) -> TinyFactors:
    mapping = getattr(episode, "direct_factors", None)
    if not isinstance(mapping, Mapping):
        raise TypeError("episode.direct_factors must be a mapping")
    required = {"q", "k", "v", "decay", "beta_e", "beta_w", "out_mix"}
    allowed = required | {"write_mask", "query_role"}
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
    )


__all__ = [
    "TINY_BACKEND_SCHEMA_VERSION",
    "TinyCellOutput",
    "TinyFactors",
    "TinyKMD2Cell",
    "TinyKMD2Config",
    "TinyKMD2Model",
    "TinyModelOutput",
    "tiny_factors_from_episode",
]
