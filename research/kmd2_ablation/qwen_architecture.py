"""Canonical, transactional Qwen architecture replacement boundary."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gdn3.kmd2_native import KMD2NativeAttn
from .architecture import (
    ArchitectureRecord,
    _channelwise_gdn2_update_unchecked,
    _true_mimo_update_unchecked,
    architecture_record,
    registry_sha256,
)
from .qwen_variants import (
    KMD2BCBiasAttn,
    KMD2DiagonalQKAttn,
    KMD2LookaheadAttn,
    KMD2TrapezoidAttn,
)

_INCREMENTAL_TYPES = {
    "trapezoid": KMD2TrapezoidAttn,
    "lookahead": KMD2LookaheadAttn,
    "qk-bc-additive": KMD2BCBiasAttn,
    "qk-diagonal": KMD2DiagonalQKAttn,
}


def _rotation_accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _paired_rotate_unchecked(x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    cosine, sine = phase.cos(), phase.sin()
    return torch.cat((first * cosine - second * sine, first * sine + second * cosine), dim=-1)


def paired_rotate(x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Apply ``R(phase)``; low-precision operands produce an FP32 result."""
    if not isinstance(x, torch.Tensor) or not isinstance(phase, torch.Tensor):
        raise TypeError("paired rotation operands must be torch tensors")
    if not x.is_floating_point() or not phase.is_floating_point():
        raise TypeError("paired rotation operands must be floating point")
    if x.device != phase.device or x.dtype != phase.dtype:
        raise ValueError("paired rotation operands must share dtype and device")
    if x.ndim < 1 or x.shape[-1] % 2:
        raise ValueError("paired rotation width must be positive and even")
    half = x.shape[-1] // 2
    if half == 0:
        raise ValueError("paired rotation width must be positive and even")
    if tuple(phase.shape) != (*x.shape[:-1], half):
        raise ValueError(f"phase_shape_invalid: expected {(*x.shape[:-1], half)}, got {tuple(phase.shape)}")
    if not bool(torch.isfinite(x.detach()).all() & torch.isfinite(phase.detach()).all()):
        raise ValueError("paired rotation operands must be finite")
    accumulation_dtype = _rotation_accumulation_dtype(x.dtype)
    return _paired_rotate_unchecked(x.to(accumulation_dtype), phase.to(accumulation_dtype))


def _cumulative_phase_unchecked(delta: torch.Tensor, reset_mask: torch.Tensor | None) -> torch.Tensor:
    if reset_mask is None:
        return delta.cumsum(dim=1)
    running = torch.zeros_like(delta[:, 0])
    phases = []
    for token in range(delta.shape[1]):
        running = torch.where(reset_mask[:, token, None, None], torch.zeros_like(running), running)
        running = running + delta[:, token]
        phases.append(running)
    return torch.stack(phases, dim=1)


def _rotation_phase_unchecked(projected, mode, rotation_rate, reset_mask):
    B, T, H, half = projected.shape
    if mode in {"current", "moving-frame-oracle"}:
        return _cumulative_phase_unchecked(F.softplus(projected), reset_mask)
    if mode == "noncumulative":
        return F.softplus(projected)
    if mode == "off":
        return torch.zeros_like(projected)
    positions = torch.arange(T, device=projected.device, dtype=projected.dtype)
    if mode == "constant":
        phase = (positions + 1)[None, :, None, None] * rotation_rate[None, None]
        return phase.expand(B, -1, -1, -1)
    exponent = -2 * torch.arange(half, device=projected.device, dtype=projected.dtype) / (2 * half)
    inv_freq = torch.pow(projected.new_tensor(10000.0), exponent)
    phase = positions[None, :, None, None] * inv_freq[None, None, None]
    return phase.expand(B, -1, H, -1)


def rotation_phase(
    projected: torch.Tensor,
    mode: str,
    *,
    rotation_rate: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the exact phase ``[B,T,H,dk/2]`` for a rotation arm."""
    modes = {"current", "moving-frame-oracle", "off", "constant", "noncumulative", "fixed-rope"}
    if mode not in modes:
        raise ValueError(f"rotation mode must be one of {sorted(modes)}")
    if not isinstance(projected, torch.Tensor):
        raise TypeError("projected rotation logits must be a torch tensor")
    if not projected.is_floating_point():
        raise TypeError("projected rotation logits must be floating point")
    if projected.ndim != 4 or projected.shape[-1] < 1:
        raise ValueError("projected_shape_invalid: expected [B,T,H,dk/2]")
    B, T, H, half = projected.shape
    if reset_mask is not None:
        if not isinstance(reset_mask, torch.Tensor) or reset_mask.dtype != torch.bool:
            raise TypeError("reset_mask must be a boolean torch tensor")
        if reset_mask.device != projected.device or tuple(reset_mask.shape) != (B, T):
            raise ValueError(f"reset_mask_shape_invalid: expected {(B, T)} on projected device")
    if mode == "constant":
        if not isinstance(rotation_rate, torch.Tensor):
            raise TypeError("constant rotation requires rotation_rate tensor")
        if rotation_rate.device != projected.device or rotation_rate.dtype != projected.dtype:
            raise ValueError("rotation_rate must share dtype and device")
        if tuple(rotation_rate.shape) != (H, half):
            raise ValueError(f"rotation_rate_shape_invalid: expected {(H, half)}")
    valid = torch.isfinite(projected.detach()).all()
    if rotation_rate is not None and mode == "constant":
        valid = valid & torch.isfinite(rotation_rate.detach()).all()
    if not bool(valid):
        raise ValueError("rotation phase operands must be finite")
    accumulation_dtype = _rotation_accumulation_dtype(projected.dtype)
    projected_acc = projected.to(accumulation_dtype)
    rate_acc = rotation_rate.to(accumulation_dtype) if mode == "constant" else None
    return _rotation_phase_unchecked(projected_acc, mode, rate_acc, reset_mask)


def _moving_frame_scan_unchecked(q, k, v, g, beta_e, beta_w, phase, reset_mask):
    B, T, H, dk = q.shape
    state = torch.zeros(B, H, dk, v.shape[-1], dtype=q.dtype, device=q.device)
    previous = torch.zeros(B, H, dk // 2, dtype=q.dtype, device=q.device)
    outputs = []
    for token in range(T):
        if reset_mask is not None:
            reset = reset_mask[:, token, None, None]
            state = torch.where(reset[..., None], torch.zeros_like(state), state)
            previous = torch.where(reset, torch.zeros_like(previous), previous)
        row_phase = previous - phase[:, token]
        first, second = state[:, :, :dk // 2], state[:, :, dk // 2:]
        cosine, sine = row_phase.cos()[..., None], row_phase.sin()[..., None]
        state = torch.cat((first * cosine - second * sine, first * sine + second * cosine), dim=2)
        state = g[:, token].unsqueeze(-1) * state
        memory = torch.einsum("bhd,bhdv->bhv", k[:, token], state)
        state = state - torch.einsum(
            "bhd,bhv->bhdv", k[:, token], beta_e[:, token, :, None] * memory,
        )
        state = state + torch.einsum(
            "bhd,bhv->bhdv", k[:, token], beta_w[:, token, :, None] * v[:, token],
        )
        outputs.append(torch.einsum("bhd,bhdv->bhv", q[:, token], state))
        previous = phase[:, token]
    return torch.stack(outputs, dim=1)


def moving_frame_scan(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    beta_e: torch.Tensor, beta_w: torch.Tensor, phase: torch.Tensor,
    *, reset_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run native scalar GDN-2 in the transported, unrotated local frame."""
    named = (("q", q), ("k", k), ("v", v), ("g", g), ("beta_e", beta_e),
             ("beta_w", beta_w), ("phase", phase))
    if any(not isinstance(tensor, torch.Tensor) for _, tensor in named):
        raise TypeError("moving-frame operands must be torch tensors")
    tensors = [tensor for _, tensor in named]
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("moving-frame operands must be floating point")
    if len({tensor.dtype for tensor in tensors}) != 1 or len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("moving-frame operands must share dtype and device")
    if q.ndim != 4 or q.shape[-1] < 2 or q.shape[-1] % 2:
        raise ValueError("q_shape_invalid: expected [B,T,H,positive even dk]")
    B, T, H, dk = q.shape
    if T == 0:
        raise ValueError("moving-frame sequence length must be positive")
    if v.ndim != 4:
        raise ValueError("v_shape_invalid: expected [B,T,H,dv]")
    dv = v.shape[-1]
    expected = {"q": (B, T, H, dk), "k": (B, T, H, dk), "v": (B, T, H, dv),
                "g": (B, T, H, dk), "beta_e": (B, T, H), "beta_w": (B, T, H),
                "phase": (B, T, H, dk // 2)}
    for name, tensor in named:
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(f"{name}_shape_invalid: expected {expected[name]}, got {tuple(tensor.shape)}")
    if reset_mask is not None:
        if not isinstance(reset_mask, torch.Tensor) or reset_mask.dtype != torch.bool:
            raise TypeError("reset_mask must be a boolean torch tensor")
        if reset_mask.device != q.device or tuple(reset_mask.shape) != (B, T):
            raise ValueError(f"reset_mask_shape_invalid: expected {(B, T)} on operand device")
    valid = torch.ones((), dtype=torch.bool, device=q.device)
    for tensor in tensors:
        valid = valid & torch.isfinite(tensor.detach()).all()
    valid = valid & (g.detach() >= 0).all() & (beta_e.detach() >= 0).all() & (beta_w.detach() >= 0).all()
    if not bool(valid):
        raise ValueError("moving-frame operands must be finite and decay/scalar gates nonnegative")
    accumulation_dtype = _rotation_accumulation_dtype(q.dtype)
    operands = [tensor.to(accumulation_dtype) for tensor in tensors]
    return _moving_frame_scan_unchecked(*operands, reset_mask)


def _true_mimo_sequence_scan_unchecked(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    beta_e: torch.Tensor, beta_w: torch.Tensor, z: torch.Tensor,
    out_mix: torch.Tensor, state: torch.Tensor,
) -> torch.Tensor:
    """Run recurrence arithmetic after caller validation and accumulation casting."""
    outputs = []
    for token in range(q.shape[1]):
        state_bar = g[:, token].unsqueeze(-1) * state
        state = _true_mimo_update_unchecked(
            state_bar, k[:, token], v[:, token], beta_e[:, token], beta_w[:, token],
        )
        read = torch.matmul(q[:, token], state)
        gated = read * F.silu(z[:, token])
        outputs.append((out_mix[:, token] * gated).sum(dim=2))
    return torch.stack(outputs, dim=1)


def true_mimo_sequence_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta_e: torch.Tensor,
    beta_w: torch.Tensor,
    z: torch.Tensor,
    out_mix: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the validated genuine rank-2/rank-4 MIMO recurrence."""

    named = (("q", q), ("k", k), ("v", v), ("g", g), ("beta_e", beta_e),
             ("beta_w", beta_w), ("z", z), ("out_mix", out_mix))
    operands = [tensor for _, tensor in named]
    if initial_state is not None:
        operands.append(initial_state)
    if any(not isinstance(tensor, torch.Tensor) for tensor in operands):
        raise TypeError("true-MIMO scan operands must be torch tensors")
    if any(not tensor.is_floating_point() for tensor in operands):
        raise TypeError("true-MIMO scan operands must be floating point")
    if len({tensor.device for tensor in operands}) != 1:
        raise ValueError("true-MIMO scan operands must share one device")
    if len({tensor.dtype for tensor in operands}) != 1:
        raise ValueError("true-MIMO scan operands must share one dtype")
    valid_values = torch.ones((), dtype=torch.bool, device=q.device)
    for tensor in operands:
        valid_values = valid_values & torch.isfinite(tensor.detach()).all()
    valid_values = valid_values & (beta_e.detach() >= 0).all() & (beta_w.detach() >= 0).all()
    if not bool(valid_values):
        raise ValueError("true-MIMO scan operands must be finite and scalar gates nonnegative")
    if q.ndim != 5:
        raise ValueError("q_shape_invalid: expected [B,T,H,R,dk]")
    B, T, H, rank, dk = q.shape
    if rank not in (2, 4):
        raise ValueError("true-MIMO scan R must be 2 or 4")
    if v.ndim != 5:
        raise ValueError("v_shape_invalid: expected [B,T,H,R,dv]")
    dv = v.shape[-1]
    expected = {
        "q": (B, T, H, rank, dk), "k": (B, T, H, rank, dk),
        "v": (B, T, H, rank, dv), "g": (B, T, H, dk),
        "beta_e": (B, T, H, rank), "beta_w": (B, T, H, rank),
        "z": (B, T, H, rank, dv), "out_mix": (B, T, H, rank, dv),
    }
    for name, tensor in named:
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(f"{name}_shape_invalid: expected {expected[name]}, got {tuple(tensor.shape)}")
    state_shape = (B, H, dk, dv)
    if initial_state is not None and tuple(initial_state.shape) != state_shape:
        raise ValueError(f"initial_state_shape_invalid: expected {state_shape}, got {tuple(initial_state.shape)}")

    # FP32 is the production accumulation floor; FP64 remains FP64 for the
    # high-precision reference contract and gradient checks.
    accumulation_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q_acc, k_acc, v_acc, g_acc, beta_e_acc, beta_w_acc, z_acc, out_mix_acc = (
        tensor.to(accumulation_dtype) for tensor in (q, k, v, g, beta_e, beta_w, z, out_mix)
    )
    state = (torch.zeros(state_shape, device=q.device, dtype=accumulation_dtype)
             if initial_state is None else initial_state.to(accumulation_dtype))
    return _true_mimo_sequence_scan_unchecked(
        q_acc, k_acc, v_acc, g_acc, beta_e_acc, beta_w_acc, z_acc, out_mix_acc, state,
    )


class QwenArchitectureInstallError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _fingerprint_state_tensor(tensor: torch.Tensor) -> tuple[tuple[int, ...], torch.dtype, str]:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return tuple(tensor.shape), tensor.dtype, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class QwenArchitectureConfig:
    arm_id: str
    registry_sha256: str
    record: ArchitectureRecord
    diagnostic_training: bool = False

    def __post_init__(self) -> None:
        if type(self.diagnostic_training) is not bool:
            raise TypeError("diagnostic_training must be boolean")
        if self.diagnostic_training and self.arm_id != "rot-moving-frame-oracle":
            raise ValueError("diagnostic_training is only valid for moving-frame oracle")
        rotation_ids = {
            "rot-off", "rot-constant", "rot-noncumulative", "rot-fixed-rope",
            "rot-moving-frame-oracle",
        }
        if self.arm_id not in {"gdn2-channel-r1", "rout-4", "mimo-r2", "mimo-r4", "gdn2-mimo-r4-braid-shared-hola-w64", "gdn2-mimo-r4-braid-four-state-hola-w64", *rotation_ids, *_INCREMENTAL_TYPES}:
            raise ValueError("architecture_not_implemented")
        canonical = architecture_record(self.arm_id)
        if self.record != canonical:
            raise ValueError("architecture_record_mismatch")
        if self.registry_sha256 != registry_sha256():
            raise ValueError("architecture_registry_hash_mismatch")
        record = self.record
        common = (
            record.output_width == (4 if self.arm_id == "rout-4" else 1)
            and record.convolution_on and not record.cache.enabled
            and (record.rotation_mode == "current" or self.arm_id in rotation_ids)
            and (record.state_input_mode == ("trapezoid" if self.arm_id == "trapezoid" else "native"))
            and ((record.qk_mode != "none") if self.arm_id.startswith("qk-") else record.qk_mode == "none")
            and record.lookahead == (self.arm_id == "lookahead")
        )
        arm_specific = (
            record.mimo_rank == 1 and record.gate_mode == "scalar"
            if self.arm_id == "rout-4"
            else record.mimo_rank == 1 and record.gate_mode == "channelwise"
            if self.arm_id == "gdn2-channel-r1"
            else record.mimo_rank == int(self.arm_id[-1]) and record.gate_mode == "scalar"
            if self.arm_id.startswith("mimo-")
            else (self.arm_id in rotation_ids or self.arm_id in _INCREMENTAL_TYPES)
            and record.mimo_rank == 1 and record.gate_mode == "scalar"
        )
        if self.arm_id in {"gdn2-mimo-r4-braid-shared-hola-w64", "gdn2-mimo-r4-braid-four-state-hola-w64"}:
            common = arm_specific = True
        if not (common and arm_specific):
            raise ValueError("architecture_record_mismatch")


class KMD2RotationControlAttn(KMD2NativeAttn):
    """Canonical R1 with exactly one preregistered rotation intervention."""

    implementation_reference = "qwen_architecture.KMD2RotationControlAttn.reference_fp32"

    @classmethod
    def _convert_cloned_native(cls, native: KMD2NativeAttn, mode: str, *, diagnostic_training: bool = False) -> "KMD2RotationControlAttn":
        modes = {"off", "constant", "noncumulative", "fixed-rope", "moving-frame-oracle"}
        if mode not in modes:
            raise ValueError("rotation_mode_unsupported")
        if type(native) is not KMD2NativeAttn or native.r_out != 1:
            raise TypeError("rotation control requires exact canonical R1 KMD2NativeAttn")
        collisions = tuple(name for name in ("rotation_rate", "inv_freq") if hasattr(native, name))
        if collisions:
            raise ValueError(f"rotation control tensor-name collision: {collisions}")
        native.__class__ = cls
        native.rotation_mode = mode
        native.rot_proj.weight.requires_grad_(mode == "noncumulative" or (mode == "moving-frame-oracle" and diagnostic_training))
        native.rot_proj.bias.requires_grad_(mode == "noncumulative" or (mode == "moving-frame-oracle" and diagnostic_training))
        device, dtype = native.rot_proj.weight.device, native.rot_proj.weight.dtype
        if mode == "constant":
            native.rotation_rate = torch.nn.Parameter(torch.full((native.H, native.dk // 2), .01, device=device, dtype=dtype))
        elif mode == "fixed-rope":
            j = torch.arange(native.dk // 2, device=device, dtype=dtype)
            native.register_buffer("inv_freq", 10000.0 ** (-2 * j / native.dk), persistent=True)
        return native

    @classmethod
    def from_native(cls, native: KMD2NativeAttn, mode: str) -> "KMD2RotationControlAttn":
        return cls._convert_cloned_native(copy.deepcopy(native), mode)

    def transformation_manifest(self) -> dict[str, object]:
        new = ({"constant": ("rotation_rate",), "fixed-rope": ("inv_freq",)}
               .get(self.rotation_mode, ()))
        copied = tuple(name for name in self.state_dict() if name not in set(new))
        return {"copied": copied, "transformed": (), "new": new}

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, **kwargs) -> torch.Tensor:
        if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
            raise ValueError("rotation controls do not support cache")
        B, T, _ = hidden_states.shape
        H, dk, dv = self.H, self.dk, self.dv
        mixed = F.silu(self.conv1d(self.in_proj_qkv(hidden_states).transpose(1, 2))[:, :, :T]).transpose(1, 2)
        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = F.normalize(query.reshape(B, T, H, dk).float(), dim=-1, eps=1e-6) * (dk ** -.5)
        k = F.normalize(key.reshape(B, T, H, dk).float(), dim=-1, eps=1e-6)
        v = value.reshape(B, T, H, dv).float()
        z = self.in_proj_z(hidden_states)
        b, a = self.in_proj_b(hidden_states).float(), self.in_proj_a(hidden_states).float()
        beta_e, beta_w = torch.sigmoid(b), torch.sigmoid(b + self.bw_off.float())
        g_head = -self.A_log.float().exp() * F.softplus(a + self.dt_bias.float())
        g = (g_head.unsqueeze(-1) + self.decay_chan.float()).exp().clamp(max=1.0)
        if self.rotation_mode in {"noncumulative", "moving-frame-oracle"}:
            projected = self.rot_proj(hidden_states).view(B, T, H, dk // 2).float()
            phase = _rotation_phase_unchecked(
                projected, self.rotation_mode, None, kwargs.get("reset_mask")
            )
        elif self.rotation_mode == "constant":
            positions = torch.arange(1, T + 1, device=q.device, dtype=q.dtype)
            phase = (positions[None, :, None, None] * self.rotation_rate.float()[None, None]).expand(B, -1, -1, -1)
        elif self.rotation_mode == "fixed-rope":
            positions = torch.arange(T, device=q.device, dtype=q.dtype)
            phase = (positions[None, :, None, None] * self.inv_freq.float()[None, None, None]).expand(B, -1, H, -1)
        else:
            phase = torch.zeros(B, T, H, dk // 2, device=q.device, dtype=q.dtype)
        if self.rotation_mode == "moving-frame-oracle":
            y = _moving_frame_scan_unchecked(
                q, k, v, g, beta_e, beta_w, phase, kwargs.get("reset_mask")
            )
        else:
            q, k = _paired_rotate_unchecked(q, phase), _paired_rotate_unchecked(k, phase)
            y = self._scan(q.unsqueeze(3), k, v, g, beta_e, beta_w)
        y = self.norm(y.reshape(-1, dv).to(z.dtype), z.reshape(-1, dv)).reshape(B, T, self.value_dim)
        out = self.out_proj(y)
        if attention_mask is not None:
            if attention_mask.ndim == 2: attention_mask = attention_mask.unsqueeze(-1)
            out = out * attention_mask
        return out


class KMD2ChannelwiseGDN2Attn(KMD2NativeAttn):
    """Canonical R1 Qwen KMD-2 with independent channelwise erase/write gates."""

    @classmethod
    def _convert_cloned_native(cls, native: KMD2NativeAttn) -> "KMD2ChannelwiseGDN2Attn":
        if type(native) is not KMD2NativeAttn or native.r_out != 1:
            raise TypeError("channelwise GDN-2 conversion requires exact canonical R1 KMD2NativeAttn")
        collisions = tuple(name for name in ("erase_proj", "write_proj", "write_offset") if hasattr(native, name))
        if collisions:
            raise ValueError(f"channelwise GDN-2 tensor-name collision: {collisions}")
        native.__class__ = cls
        device, dtype = native.in_proj_b.weight.device, native.in_proj_b.weight.dtype
        hidden = native.in_proj_b.in_features
        native.erase_proj = torch.nn.Linear(hidden, native.H * native.dk, bias=False, device=device, dtype=dtype)
        native.write_proj = torch.nn.Linear(hidden, native.H * native.dv, bias=False, device=device, dtype=dtype)
        native.write_offset = torch.nn.Parameter(native.bw_off.detach().clone())
        with torch.no_grad():
            native.erase_proj.weight.copy_(native.in_proj_b.weight[:, None, :].expand(-1, native.dk, -1).reshape(-1, hidden))
            native.write_proj.weight.copy_(native.in_proj_b.weight[:, None, :].expand(-1, native.dv, -1).reshape(-1, hidden))
        # Bound the live recurrent trace during long-sequence training.  This
        # is a runtime control rather than model state, so conversion manifests
        # and checkpoint tensor names remain unchanged.  ``None`` disables
        # within-layer segmentation, matching the four-state implementation.
        native.checkpoint_segment_tokens = 64
        return native

    @classmethod
    def from_native(cls, native: KMD2NativeAttn) -> "KMD2ChannelwiseGDN2Attn":
        return cls._convert_cloned_native(copy.deepcopy(native))

    def transformation_manifest(self) -> dict[str, object]:
        inherited = tuple(
            name for name in self.state_dict()
            if name not in {"erase_proj.weight", "write_proj.weight", "write_offset"}
        )
        return {
            "copied": inherited,
            "transformed": (
                ("in_proj_b.weight", "erase_proj.weight", "row_copy_dk"),
                ("in_proj_b.weight", "write_proj.weight", "row_copy_dv"),
                ("bw_off", "write_offset", "copy"),
            ),
            "new": (),
        }

    def _factor_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        erase = self.erase_proj(x).reshape(B, T, self.H, self.dk)
        write = self.write_proj(x).reshape(B, T, self.H, self.dv)
        return erase, write + self.write_offset[None, None, :, None]

    def _validate_channelwise_scan(self, q, k, v, g, erase, write):
        if q.ndim != 5 or q.shape[2:] != (self.H, 1, self.dk):
            raise ValueError(f"q_shape_invalid: expected [B,T,{self.H},1,{self.dk}]")
        B, T = q.shape[:2]
        expected = {
            "k": (B, T, self.H, self.dk),
            "v": (B, T, self.H, self.dv),
            "g": (B, T, self.H, self.dk),
            "erase": (B, T, self.H, self.dk),
            "write": (B, T, self.H, self.dv),
        }
        for name, tensor in (("k", k), ("v", v), ("g", g), ("erase", erase), ("write", write)):
            if tuple(tensor.shape) != expected[name]:
                raise ValueError(f"{name}_shape_invalid: expected {expected[name]}, got {tuple(tensor.shape)}")

    def _scan_channelwise_reference(self, q, k, v, g, erase, write):
        """Exact token-loop oracle retained for parity and unsupported shapes."""
        self._validate_channelwise_scan(q, k, v, g, erase, write)
        B, T = q.shape[:2]
        H, dk = self.H, self.dk
        state = torch.zeros(B, H, dk, self.dv, dtype=torch.float32, device=k.device)
        outputs = []
        for t in range(T):
            state = state * g[:, t].float().unsqueeze(-1)
            state = _channelwise_gdn2_update_unchecked(
                state, k[:, t].float().unsqueeze(2), v[:, t].float().unsqueeze(2),
                erase[:, t].float().unsqueeze(2), write[:, t].float().unsqueeze(2),
            )
            outputs.append(torch.matmul(q[:, t, :, 0].float().unsqueeze(-2), state).squeeze(-2))
        return torch.stack(outputs, dim=1)

    def _scan_channelwise(self, q, k, v, g, erase, write):
        """Dispatch the campaign shape to a checkpointed fused FP32 scan."""
        self._validate_channelwise_scan(q, k, v, g, erase, write)
        if getattr(self, "force_reference_path", False):
            return self._scan_channelwise_reference(q, k, v, g, erase, write)

        B, T = q.shape[:2]
        segment = getattr(self, "checkpoint_segment_tokens", 64)
        segmented = type(segment) is int and segment > 0
        step = segment if segmented else T
        if step < 1:
            return self._scan_channelwise_reference(q, k, v, g, erase, write)

        # The raw kernel consumes the singleton query-rank dimension squeezed
        # away.  Splitting each complete sequence once gives autograd one join
        # per operand instead of one full-T SliceBackward allocation per
        # segment, which is material at T=4096.
        # Rotation and the depthwise convolution deliberately produce
        # token-minor views.  The persistent kernel advances tokens inside one
        # program, so materialize token-major Q/K/V once per full scan; gates
        # already have this layout and ``contiguous`` is a no-op for them.
        scan_inputs = tuple(
            value.contiguous()
            for value in (q[:, :, :, 0], k, v, erase, write, g)
        )
        segment_inputs = tuple(
            torch.split(value, step, dim=1) for value in scan_inputs
        )
        first = tuple(chunks[0] for chunks in segment_inputs)
        state = torch.zeros(
            B, self.H, self.dk, self.dv,
            dtype=torch.float32, device=k.device,
        )
        from .qwen_gdn2_triton import (
            can_use_triton_gdn2_segment,
            triton_gdn2_segment,
        )
        if not can_use_triton_gdn2_segment(*first, state):
            return self._scan_channelwise_reference(q, k, v, g, erase, write)

        use_checkpoint = (
            segmented and torch.is_grad_enabled()
            and any(tensor.requires_grad for tensor in scan_inputs)
        )
        pieces = []
        for chunks in zip(*segment_inputs):
            args = (*chunks, state)
            if use_checkpoint:
                from torch.utils.checkpoint import checkpoint
                output, state = checkpoint(
                    triton_gdn2_segment, *args, use_reentrant=False
                )
            else:
                output, state = triton_gdn2_segment(*args)
            pieces.append(output)
        return torch.cat(pieces, dim=1)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, **kwargs) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, dk, dv = self.H, self.dk, self.dv
        x = hidden_states
        mixed = F.silu(self.conv1d(self.in_proj_qkv(x).transpose(1, 2))[:, :, :T]).transpose(1, 2)
        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = F.normalize(query.reshape(B, T, H, dk).float(), dim=-1, eps=1e-6) * (dk ** -0.5)
        k = F.normalize(key.reshape(B, T, H, dk).float(), dim=-1, eps=1e-6)
        v = value.reshape(B, T, H, dv).float()
        z = self.in_proj_z(x)
        a = self.in_proj_a(x).float()
        erase_logits, write_logits = self._factor_logits(x)
        erase, write = torch.sigmoid(erase_logits.float()), torch.sigmoid(write_logits.float())
        g_head = -self.A_log.float().exp() * F.softplus(a + self.dt_bias.float())
        g = (g_head.unsqueeze(-1) + self.decay_chan).exp().clamp(max=1.0)
        theta = F.softplus(self.rot_proj(x)).view(B, T, H, dk // 2).float().cumsum(dim=1)
        cos, sin = theta.cos(), theta.sin()
        def rope(tensor):
            first, second = tensor[..., :dk // 2], tensor[..., dk // 2:]
            return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
        k, q = rope(k), rope(q)
        y = self._scan_channelwise(q.unsqueeze(3), k, v, g, erase, write)
        y = self.norm(y.reshape(-1, dv).to(z.dtype), z.reshape(-1, dv)).reshape(B, T, self.value_dim)
        out = self.out_proj(y)
        if attention_mask is not None:
            if attention_mask.ndim == 2: attention_mask = attention_mask.unsqueeze(-1)
            out = out * attention_mask
        return out


class KMD2SharedQueryWideningAttn(KMD2NativeAttn):
    """Four query reads over one canonical R1 K/V/write/recurrent state."""

    architecture_classification = "control"
    promotable = False
    identity_at_initialization = True
    implementation_reference = "qwen_architecture.KMD2SharedQueryWideningAttn.reference_fp32"
    implementation_path = "reference_fp32_fast_scan_fail_closed"

    @property
    def output_width(self) -> int:
        return 4

    @classmethod
    def _convert_cloned_native(
        cls, native: KMD2NativeAttn, width: int,
    ) -> "KMD2SharedQueryWideningAttn":
        if type(width) is not int or width != 4:
            raise ValueError("shared-query widening width must be exactly 4")
        if type(native) is not KMD2NativeAttn or native.r_out != 1:
            raise TypeError("shared-query widening requires exact canonical R1 KMD2NativeAttn")
        collisions = tuple(name for name in ("q_slot_scale", "out_mix") if hasattr(native, name))
        if collisions:
            raise ValueError(f"shared-query widening tensor-name collision: {collisions}")
        native.__class__ = cls
        native.r_out = native.width = width
        device, dtype = native.in_proj_qkv.weight.device, native.in_proj_qkv.weight.dtype
        native.q_slot_scale = torch.nn.Parameter(torch.zeros(native.H, width, native.dk, device=device, dtype=dtype))
        mix = torch.zeros(native.H, width, device=device, dtype=dtype)
        mix[:, 0] = 1
        native.out_mix = torch.nn.Parameter(mix)
        return native

    @classmethod
    def from_native(cls, native: KMD2NativeAttn, width: int = 4) -> "KMD2SharedQueryWideningAttn":
        return cls._convert_cloned_native(copy.deepcopy(native), width)

    def transformation_manifest(self) -> dict[str, object]:
        new = ("q_slot_scale", "out_mix")
        return {
            "copied": tuple(name for name in self.state_dict() if name not in new),
            "transformed": (),
            "new": new,
        }

    def recurrent_state_bytes(self, batch_size: int) -> int:
        if type(batch_size) is not int or batch_size < 0:
            raise ValueError("batch_size must be a nonnegative integer")
        return batch_size * self.H * self.dk * self.dv * 4

    @staticmethod
    def _cache_value_populated(value: object) -> bool:
        if value is None or value is False:
            return False
        if isinstance(value, torch.Tensor):
            return value.numel() > 0
        if isinstance(value, (tuple, list, dict, set, str, bytes)):
            return len(value) > 0
        return True

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, **kwargs) -> torch.Tensor:
        cache_keys = (
            "past_key_values", "past_key_value", "cache_position",
            "cache_params", "cache_state",
        )
        if kwargs.get("use_cache") is True or any(
            self._cache_value_populated(kwargs.get(name)) for name in cache_keys
        ):
            raise ValueError("shared_query_widening_cache_unsupported")
        return super().forward(hidden_states, attention_mask=attention_mask, **kwargs)

    def _scan(self, q, k, v, g, beta_e, beta_w):
        """Reference FP32 scan; native fast scan is deliberately fail-closed."""
        B, T, H, width, dk = q.shape
        if (H, width, dk) != (self.H, 4, self.dk):
            raise ValueError(f"q_shape_invalid: expected [B,T,{self.H},4,{self.dk}]")
        N, dv = B * H, self.dv
        def flat(tensor, *tail):
            return tensor.permute(1, 0, 2, *range(3, tensor.dim())).reshape(T, N, *tail).float()
        q_, k_, v_ = flat(q, width, dk), flat(k, dk), flat(v, dv)
        g_, be_, bw_ = flat(g, dk), flat(beta_e), flat(beta_w)
        mix = self.out_mix[None].expand(B, -1, -1).reshape(N, 1, width).float()
        state = torch.zeros(N, dk, dv, dtype=torch.float32, device=k.device)
        outputs = []
        for token in range(T):
            state = state * g_[token].unsqueeze(-1)
            kt = k_[token]
            memory = torch.bmm(kt.unsqueeze(1), state).squeeze(1)
            state = state - torch.bmm(kt.unsqueeze(2), (be_[token].unsqueeze(-1) * memory).unsqueeze(1))
            state = state + torch.bmm(kt.unsqueeze(2), (bw_[token].unsqueeze(-1) * v_[token]).unsqueeze(1))
            reads = torch.bmm(q_[token], state)
            outputs.append((reads * mix.transpose(1, 2)).sum(1))
        return torch.stack(outputs).reshape(T, B, H, dv).permute(1, 0, 2, 3)


class KMD2TrueMIMOAttn(KMD2NativeAttn):
    """Genuine rank-2/rank-4 MIMO over one canonical Qwen recurrent state."""

    architecture_classification = "cold_redesign"
    identity_at_initialization = False
    implementation_reference = "qwen_architecture.KMD2TrueMIMOAttn.reference_fp32"

    @classmethod
    def from_native(
        cls, native: KMD2NativeAttn, rank: int, *, rotation_mode: str = "current",
    ) -> "KMD2TrueMIMOAttn":
        if rotation_mode != "current":
            raise ValueError("true_mimo_rotation_mode_unsupported: expected current")
        from gdn3 import kmd2_native
        if kmd2_native._FAST_SCAN:
            raise ValueError("true MIMO rejects the native fast scan")
        if type(rank) is not int or rank not in (2, 4):
            raise ValueError("true MIMO rank must be 2 or 4")
        if type(native) is not KMD2NativeAttn or native.r_out != 1:
            raise TypeError("true MIMO requires exact canonical R1 KMD2NativeAttn")
        forbidden = ("mimo_q_transform", "mimo_k_transform", "mimo_v", "mimo_z", "mimo_out")
        collisions = tuple(name for name in forbidden if hasattr(native, name))
        if collisions:
            raise ValueError(f"true MIMO tensor-name collision: {collisions}")
        converted = copy.deepcopy(native)
        converted.__class__ = cls
        converted.rank = rank
        device, dtype = converted.in_proj_qkv.weight.device, converted.in_proj_qkv.weight.dtype
        identity = torch.eye(converted.dk, device=device, dtype=dtype)
        converted.mimo_q_transform = torch.nn.Parameter(identity.expand(converted.H, rank, -1, -1).clone())
        converted.mimo_k_transform = torch.nn.Parameter(identity.expand(converted.H, rank, -1, -1).clone())
        shape = (converted.H, rank, converted.dv)
        converted.mimo_v = torch.nn.Parameter(torch.full(shape, 1.0 / rank, device=device, dtype=dtype))
        converted.mimo_z = torch.nn.Parameter(torch.ones(shape, device=device, dtype=dtype))
        converted.mimo_out = torch.nn.Parameter(torch.full(shape, 1.0 / rank, device=device, dtype=dtype))
        return converted

    def transformation_manifest(self) -> dict[str, object]:
        new = ("mimo_q_transform", "mimo_k_transform", "mimo_v", "mimo_z", "mimo_out")
        return {
            "copied": tuple(name for name in self.state_dict() if name not in new),
            "transformed": (),
            "new": new,
        }

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, **kwargs) -> torch.Tensor:
        if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
            raise ValueError("true MIMO does not support cache")
        B, T, _ = hidden_states.shape
        H, dk, dv, rank = self.H, self.dk, self.dv, self.rank
        x = hidden_states
        mixed = F.silu(self.conv1d(self.in_proj_qkv(x).transpose(1, 2))[:, :, :T]).transpose(1, 2)
        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q0 = query.reshape(B, T, H, dk)
        k0 = key.reshape(B, T, H, dk)
        q = torch.einsum("bthd,hrde->bthre", q0.float(), self.mimo_q_transform.float())
        k = torch.einsum("bthd,hrde->bthre", k0.float(), self.mimo_k_transform.float())
        q = F.normalize(q, dim=-1, eps=1e-6) * (dk ** -0.5)
        k = F.normalize(k, dim=-1, eps=1e-6)

        theta = F.softplus(self.rot_proj(x)).view(B, T, H, dk // 2).float().cumsum(dim=1)
        cos, sin = theta.cos().unsqueeze(3), theta.sin().unsqueeze(3)
        def rotate(tensor: torch.Tensor) -> torch.Tensor:
            first, second = tensor[..., :dk // 2], tensor[..., dk // 2:]
            return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
        q, k = rotate(q), rotate(k)

        v0 = value.reshape(B, T, H, dv).float()
        v = v0.unsqueeze(3) * self.mimo_v[None, None].float()
        z0 = self.in_proj_z(x).reshape(B, T, H, dv).float()
        z = z0.unsqueeze(3) * self.mimo_z[None, None].float()
        b = self.in_proj_b(x).float()
        beta_e = torch.sigmoid(b).unsqueeze(-1).expand(-1, -1, -1, rank)
        beta_w = torch.sigmoid(b + self.bw_off).unsqueeze(-1).expand_as(beta_e)
        a = self.in_proj_a(x).float()
        g_head = -self.A_log.float().exp() * F.softplus(a + self.dt_bias.float())
        g = (g_head.unsqueeze(-1) + self.decay_chan).exp().clamp(max=1.0)
        out_mix = self.mimo_out[None, None].float().expand(B, T, -1, -1, -1)
        y = true_mimo_sequence_scan(q, k, v, g, beta_e, beta_w, z, out_mix)

        # Inherit only the RMS scale: genuine MIMO already applies rankwise Z.
        y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + self.norm.variance_epsilon)
        y = y * self.norm.weight.float()
        out = self.out_proj(y.reshape(B, T, self.value_dim).to(hidden_states.dtype))
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(-1)
            out = out * attention_mask
        return out


def _placeholder_factory(native: torch.nn.Module, _config: QwenArchitectureConfig):
    if _config.arm_id == "gdn2-mimo-r4-braid-shared-hola-w64":
        from .qwen_hybrid_shared import QwenSharedBraidHybrid
        return QwenSharedBraidHybrid.from_native(native)
    if _config.arm_id == "gdn2-mimo-r4-braid-four-state-hola-w64":
        from .qwen_hybrid_four_state import QwenFourStateHybrid
        return QwenFourStateHybrid.from_native(native)
    if _config.arm_id in _INCREMENTAL_TYPES:
        return _INCREMENTAL_TYPES[_config.arm_id].from_native(native)
    if _config.arm_id.startswith("rot-"):
        return KMD2RotationControlAttn._convert_cloned_native(
            native, _config.record.rotation_mode,
            diagnostic_training=_config.diagnostic_training,
        )
    if _config.arm_id == "rout-4":
        return KMD2SharedQueryWideningAttn._convert_cloned_native(
            native, _config.record.output_width,
        )
    if _config.record.mimo_rank > 1:
        return KMD2TrueMIMOAttn.from_native(
            native, _config.record.mimo_rank,
            rotation_mode=_config.record.rotation_mode,
        )
    return KMD2ChannelwiseGDN2Attn._convert_cloned_native(native)


def build_qwen_architecture(
    native: torch.nn.Module,
    config: QwenArchitectureConfig,
    *,
    factory: Callable[[torch.nn.Module, QwenArchitectureConfig], torch.nn.Module] | None = None,
    native_type: type[torch.nn.Module] | None = None,
    expected_type: type[torch.nn.Module] | None = None,
) -> torch.nn.Module:
    if (config.record.mimo_rank > 1 and config.record.rotation_mode != "current"
            and getattr(config, "arm_id", None) not in {"gdn2-mimo-r4-braid-shared-hola-w64", "gdn2-mimo-r4-braid-four-state-hola-w64"}):
        raise ValueError("true_mimo_rotation_mode_unsupported: expected current")
    if native_type is None:
        from gdn3.kmd2_native import KMD2NativeAttn
        native_type = KMD2NativeAttn
    if type(native) is not native_type:
        raise TypeError("architecture source must be exact KMD2NativeAttn")
    if getattr(native, "r_out", None) != 1:
        raise ValueError("architecture source requires r_out == 1")
    for tensor in (*native.parameters(), *native.buffers()):
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor.detach()).all()):
            raise ValueError("architecture source state must be finite")
    source_tensors = tuple(native.parameters()) + tuple(native.buffers())
    source_device = source_tensors[0].device if source_tensors else torch.device("cpu")
    if any(tensor.device != source_device for tensor in source_tensors):
        raise ValueError("architecture source has mixed dtype/device")
    floating = tuple(tensor for tensor in source_tensors if tensor.is_floating_point())
    source_dtype = floating[0].dtype if floating else None
    if source_dtype is not None and any(tensor.dtype != source_dtype for tensor in floating):
        raise ValueError("architecture source has mixed dtype/device")
    cloned = copy.deepcopy(native)
    replacement = (factory or _placeholder_factory)(cloned, config)
    if not isinstance(replacement, torch.nn.Module):
        raise TypeError("architecture factory must return a torch.nn.Module")
    if expected_type is None:
        if config.arm_id == "gdn2-mimo-r4-braid-shared-hola-w64":
            from .qwen_hybrid_shared import QwenSharedBraidHybrid
            expected_type = QwenSharedBraidHybrid
        elif config.arm_id == "gdn2-mimo-r4-braid-four-state-hola-w64":
            from .qwen_hybrid_four_state import QwenFourStateHybrid
            expected_type = QwenFourStateHybrid
        else:
            expected_type = (_INCREMENTAL_TYPES[config.arm_id] if config.arm_id in _INCREMENTAL_TYPES else
                         KMD2RotationControlAttn if config.arm_id.startswith("rot-") else
                         KMD2SharedQueryWideningAttn if config.arm_id == "rout-4" else
                         KMD2TrueMIMOAttn if config.record.mimo_rank > 1 else KMD2ChannelwiseGDN2Attn)
    if type(replacement) is not expected_type:
        raise TypeError("architecture factory must return exact expected architecture class")
    if config.record.mimo_rank > 1 and getattr(replacement, "rank", None) != config.record.mimo_rank:
        raise ValueError("architecture MIMO rank mismatch")
    replacement.train(native.training)
    replacement_tensors = tuple(replacement.parameters()) + tuple(replacement.buffers())
    # The specialization probes are identity-pinned fp32 by the campaign
    # contract (deterministic_unit_rms_fp32_...); like the fp32 recurrent
    # state math they deliberately do NOT follow the module dtype, and the
    # auxiliary loss casts them at use.  They must still share the device.
    dtype_exempt = {
        id(tensor)
        for name, tensor in replacement.named_buffers()
        if name.rsplit(".", 1)[-1] in {
            "specialization_probe",
            "specialization_value_probe",
            "specialization_coefficients",
        }
    }
    if any(tensor.device != source_device for tensor in replacement_tensors) or any(
        source_dtype is not None and tensor.is_floating_point()
        and tensor.dtype != source_dtype and id(tensor) not in dtype_exempt
        for tensor in replacement_tensors
    ):
        raise ValueError("architecture replacement did not inherit dtype/device")
    return replacement


def build_maximum_control_architecture(
    native: torch.nn.Module, control_id: str
) -> torch.nn.Module:
    """Build the exact frozen maximum-control implementation from native Qwen."""
    from .qwen_variants import maximum_control_contract
    contract = maximum_control_contract(control_id)
    if not contract.replacement:
        setattr(native, "maximum_control_contract", contract)
        return native
    if contract.module_kind == "gdn2":
        replacement = KMD2ChannelwiseGDN2Attn._convert_cloned_native(copy.deepcopy(native))
    elif contract.module_kind == "mimo":
        replacement = KMD2TrueMIMOAttn.from_native(
            copy.deepcopy(native), contract.input_rank, rotation_mode="current"
        )
    elif contract.module_kind == "widening":
        replacement = KMD2SharedQueryWideningAttn._convert_cloned_native(
            copy.deepcopy(native), contract.output_rank
        )
    elif contract.module_kind == "package_a":
        from .qwen_hybrid_shared import QwenSharedBraidHybrid
        replacement = QwenSharedBraidHybrid.from_native(copy.deepcopy(native))
    elif contract.module_kind == "package_b":
        from .qwen_hybrid_four_state import QwenFourStateHybrid
        replacement = QwenFourStateHybrid.from_native(copy.deepcopy(native))
    else:
        raise ValueError(f"unsupported executable maximum control: {control_id}")
    setattr(replacement, "maximum_control_contract", contract)
    setattr(replacement, "active_feature_flags", {
        "braid": contract.braid, "trapezoid": contract.trapezoid,
        "lookahead": contract.lookahead, "affine_qk": contract.affine_qk,
        "cache_policy": contract.cache_policy,
    })
    if hasattr(replacement, "hola"):
        replacement.hola.policy = (
            "recency" if contract.cache_policy == "recency_w64" else "exact_outer"
        )
    disabled_groups = (
        (("braid",), not contract.braid),
        (("trapezoid",), not contract.trapezoid),
        (("lookahead",), not contract.lookahead),
        (("d_q", "d_k", "b_q", "b_k", "alpha_q", "alpha_k", "beta_q", "beta_k"), not contract.affine_qk),
        (("cache_gate_logit", "hola", "hola_output"), contract.cache_policy == "none"),
    )
    for name, parameter in replacement.named_parameters():
        if any(disabled and any(token in name for token in tokens) for tokens, disabled in disabled_groups):
            parameter.requires_grad_(False)
    setattr(replacement, "declared_trainable_components", contract.trainable_components)
    return replacement


def install_qwen_architecture(
    model: torch.nn.Module,
    indices: Sequence[int],
    config: QwenArchitectureConfig,
    *,
    factory: Callable[[torch.nn.Module, QwenArchitectureConfig], torch.nn.Module] | None = None,
    configure_trainables: Callable[[torch.nn.Module, tuple[str, ...]], object] | None = None,
    declared_trainables: tuple[str, ...] = (),
    verify_conversion: Callable[[torch.nn.Module, tuple[int, ...]], object] | None = None,
    native_type: type[torch.nn.Module] | None = None,
    expected_type: type[torch.nn.Module] | None = None,
    expected_indices: Sequence[int] | None = None,
    event: Callable[[str], object] | None = None,
    swap_verifier: Callable[[torch.nn.Module, int, torch.nn.Module], object] | None = None,
) -> tuple[int, ...]:
    if expected_type is None:
        if config.arm_id == "gdn2-mimo-r4-braid-shared-hola-w64":
            from .qwen_hybrid_shared import QwenSharedBraidHybrid
            expected_type = QwenSharedBraidHybrid
        elif config.arm_id == "gdn2-mimo-r4-braid-four-state-hola-w64":
            from .qwen_hybrid_four_state import QwenFourStateHybrid
            expected_type = QwenFourStateHybrid
        else:
            expected_type = (_INCREMENTAL_TYPES[config.arm_id] if config.arm_id in _INCREMENTAL_TYPES else
                         KMD2RotationControlAttn if config.arm_id.startswith("rot-") else
                         KMD2SharedQueryWideningAttn if config.arm_id == "rout-4" else
                         KMD2TrueMIMOAttn if config.record.mimo_rank > 1 else KMD2ChannelwiseGDN2Attn)
    ordered = tuple(indices)
    canonical_indices = tuple(config.record.target_layers if expected_indices is None else expected_indices)
    if ordered != canonical_indices:
        raise QwenArchitectureInstallError(
            "architecture_target_indices_mismatch", "installed target indices are not exact"
        )
    originals = {index: model.model.layers[index].linear_attn for index in ordered}
    flags = {parameter: parameter.requires_grad for parameter in model.parameters()}
    try:
        if event: event("prepare_replacements")
        replacements = {
            index: build_qwen_architecture(
                originals[index], config, factory=factory, native_type=native_type
                , expected_type=expected_type
            ) for index in ordered
        }
        prepared_state = {
            index: {name: _fingerprint_state_tensor(value) for name, value in replacement.state_dict().items()}
            for index, replacement in replacements.items()
        }
        if event: event("swap_replacements")
        for index in ordered:
            model.model.layers[index].linear_attn = replacements[index]
            if model.model.layers[index].linear_attn is not replacements[index]:
                raise RuntimeError(f"replacement swap verification failed for layer {index}")
            if swap_verifier is not None:
                swap_verifier(model, index, replacements[index])
        if event: event("configure_trainables")
        if configure_trainables is not None:
            configure_trainables(model, declared_trainables)
        if event: event("verify_conversion")
        if verify_conversion is not None:
            verify_conversion(model, ordered)
        for index in ordered:
            installed = model.model.layers[index].linear_attn
            if type(installed) is not expected_type or installed is not replacements[index]:
                raise RuntimeError(f"conversion class/identity verification failed for layer {index}")
            if config.record.mimo_rank > 1 and getattr(installed, "rank", None) != config.record.mimo_rank:
                raise RuntimeError(f"conversion MIMO rank verification failed for layer {index}")
            actual_state = installed.state_dict()
            expected_state = prepared_state[index]
            if tuple(actual_state) != tuple(expected_state):
                raise RuntimeError(f"conversion state names changed for layer {index}")
            if any(_fingerprint_state_tensor(actual_state[name]) != expected_state[name] for name in expected_state):
                raise RuntimeError(f"conversion state values changed for layer {index}")
        return ordered
    except BaseException as cause:
        for index, original in originals.items():
            model.model.layers[index].linear_attn = original
        for parameter, requires_grad in flags.items():
            parameter.requires_grad_(requires_grad)
        if not isinstance(cause, Exception):
            raise
        raise QwenArchitectureInstallError("architecture_install_failed", str(cause)) from cause


__all__ = ["KMD2ChannelwiseGDN2Attn", "KMD2RotationControlAttn", "KMD2SharedQueryWideningAttn", "KMD2TrueMIMOAttn", "QwenArchitectureConfig", "QwenArchitectureInstallError", "build_qwen_architecture", "install_qwen_architecture", "true_mimo_sequence_scan"]
