"""Package A: one-state, four-rank maximum-potential Qwen hybrid."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .qwen_hybrid_components import HybridComponents, SHARED_TIMESCALES
from .qwen_hybrid_math import (RANK, REFERENCE_IMPLEMENTATION, apply_complex_rotation,
                               braided_decay, shared_state_step)


@dataclass(frozen=True)
class SharedHybridCache:
    state: Tensor
    phase: Tensor
    previous_value: Tensor
    previous_write: Tensor
    conv_tail: Tensor
    has_history: Tensor
    hola_state: object | None = None


class QwenSharedBraidHybrid(nn.Module):
    """Reference-first shared-state GDN-2/R4 hybrid with exact chunk carry."""

    rank = RANK
    scan_implementation = REFERENCE_IMPLEMENTATION

    @staticmethod
    def actual_implementation_identity() -> str:
        return REFERENCE_IMPLEMENTATION

    def __init__(self, native: nn.Module) -> None:
        super().__init__()
        self.H, self.dk, self.dv = native.H, native.dk, native.dv
        self.key_dim, self.value_dim, self.conv_k = native.key_dim, native.value_dim, native.conv_k
        self.r_out = RANK
        self.components = HybridComponents.from_native(native, package="shared")
        self.rot_proj = copy.deepcopy(native.rot_proj)
        from .qwen_hybrid_hola import HybridHOLACache
        self.hola = HybridHOLACache(width=64, block_size=256, heads=self.H, rank_in=RANK,
                                    key_dim=self.dk, value_dim=self.dv)
        self.hola.to(device=self.components.q_weight.device, dtype=self.components.q_weight.dtype)
        identity = torch.eye(self.dv, device=self.components.q_weight.device,
                             dtype=self.components.q_weight.dtype) / (RANK * RANK)
        self.hola_output_mixer = nn.Parameter(
            identity.view(1, 1, 1, self.dv, self.dv).expand(
                self.H, RANK, RANK, self.dv, self.dv).clone()
        )

    def _active(self, name: str, default=True):
        return getattr(self, "active_feature_flags", {}).get(name, default)

    @classmethod
    def from_native(cls, native: nn.Module) -> "QwenSharedBraidHybrid":
        return cls(native)

    def recurrent_state_bytes(self, *, batch_size: int = 1) -> int:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        state = batch_size * self.H * self.dk * self.dv * 4
        rank_history = batch_size * self.H * RANK * self.dk * self.dv * 4
        return state + rank_history + batch_size

    def resource_report(self, *, batch_size: int = 1) -> dict[str, object]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        conv_element = self.components.q_weight.element_size()
        persistent = {
            "states": batch_size * self.H * self.dk * self.dv * 4,
            "phase_history": batch_size * self.H * (self.dk // 2) * 4,
            "previous_value": batch_size * self.H * RANK * self.dv * 4,
            "previous_write": batch_size * self.H * RANK * self.dk * self.dv * 4,
            "convolution_history": (
                batch_size * (self.conv_k - 1) * self.components.hidden * conv_element
            ),
            "history_flags": batch_size,
        }
        return {
            "persistent": persistent,
            "persistent_bytes": sum(persistent.values()),
            "state_topology": "one_shared_state_with_four_rank_write_history",
        }

    def transformation_manifest(self) -> dict[str, object]:
        result = self.components.transformation_manifest()
        result["parameters"] = {
            name: {"shape": tuple(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in self.state_dict().items()
        }
        result.update({"state_shape": ("B", self.H, self.dk, self.dv),
                       "state_bytes_per_batch": self.recurrent_state_bytes(),
                       "hola_resources": (self.hola.resource_report()
                                          if self._active("cache_policy", "hola_exact_outer_w64") != "none" else None),
                       "hola_output_mixer_bytes": (self.hola_output_mixer.numel()
                       * self.hola_output_mixer.element_size()
                       if self._active("cache_policy", "hola_exact_outer_w64") != "none" else 0),
                       "hola_implementation": (self.hola.implementation_reference
                                               if self._active("cache_policy", "hola_exact_outer_w64") != "none" else None),
                       "implementation": self.actual_implementation_identity(),
                       "scan_implementation": self.scan_implementation})
        return result

    def _mix_hola_reads(self, reads: Tensor) -> Tensor:
        if reads.ndim != 5 or tuple(reads.shape[1:]) != (self.H, RANK, RANK, self.dv):
            raise ValueError("HOLA reads must be [B,H,Rout,Rin,V]")
        return torch.einsum("hijvw,bhijw->bhv", self.hola_output_mixer.float(), reads.float())

    def architecture_tensor_manifest(self) -> dict[str, tuple]:
        return {"copied": (), "transformed": (),
                "new": tuple(name for name, _ in self.named_parameters())}

    def _project_convolved(self, hidden: Tensor, tail: Tensor, boundary: Tensor,
                           valid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        B, T = hidden.shape[:2]
        rank_pieces: list[list[tuple[Tensor, Tensor, Tensor]]] = [[] for _ in range(RANK)]
        signal_tokens = []
        for token in range(T):
            reset = (boundary[:, token] & valid[:, token])[:, None, None]
            tail = torch.where(reset, torch.zeros_like(tail), tail)
            window = torch.cat((tail, hidden[:, token:token + 1]), 1)
            q, k, v, _, _, _ = self.components.project_inputs(window)
            token_mixed = []
            for rank in range(RANK):
                qkv = torch.cat((q[:, :, :, rank].flatten(2), k[:, :, :, rank].flatten(2),
                                 v[:, :, :, rank].flatten(2)), -1)
                mixed = F.silu(F.conv1d(qkv.transpose(1, 2), self.components.conv1d.weight,
                                        groups=qkv.shape[-1])).transpose(1, 2)[:, 0]
                split = torch.split(mixed, (self.key_dim, self.key_dim, self.value_dim), -1)
                rank_pieces[rank].append(split)
                token_mixed.append(mixed)
            signal_tokens.append(torch.stack(token_mixed).mean(0))
            shifted = torch.cat((tail[:, 1:], hidden[:, token:token + 1]), 1)
            tail = torch.where(valid[:, token, None, None], shifted, tail)
        q = torch.stack([torch.stack([item[0] for item in pieces], 1).reshape(B, T, self.H, self.dk)
                         for pieces in rank_pieces], 3)
        k = torch.stack([torch.stack([item[1] for item in pieces], 1).reshape(B, T, self.H, self.dk)
                         for pieces in rank_pieces], 3)
        v = torch.stack([torch.stack([item[2] for item in pieces], 1).reshape(B, T, self.H, self.dv)
                         for pieces in rank_pieces], 3)
        mixed_signal = torch.stack(signal_tokens, 1)
        routing_hidden = torch.einsum(
            "btc,cd->btd", mixed_signal,
            torch.cat((self.components.q_weight[0], self.components.k_weight[0], self.components.v_weight[0]), 0),
        )
        return q, k, v, routing_hidden, tail

    @staticmethod
    def history_active(boundary: Tensor, valid: Tensor, has_history: Tensor) -> Tensor:
        """Return where causal history features may act for the current token."""
        return valid & ~boundary & has_history

    def _initial_cache(self, hidden: Tensor) -> SharedHybridCache:
        batch = hidden.shape[0]
        state = torch.zeros(batch, self.H, self.dk, self.dv, device=hidden.device)
        phase = torch.zeros(batch, self.H, self.dk // 2, device=hidden.device)
        previous_value = torch.zeros(batch, self.H, RANK, self.dv, device=hidden.device)
        return SharedHybridCache(
            state=state,
            phase=phase,
            previous_value=previous_value,
            previous_write=torch.zeros(batch, self.H, RANK, self.dk, self.dv,
                                       dtype=torch.float32, device=hidden.device),
            conv_tail=hidden.new_zeros(batch, self.conv_k - 1, hidden.shape[-1]),
            has_history=torch.zeros(batch, dtype=torch.bool, device=hidden.device),
            hola_state=None,
        )

    def _validate_cache(self, cache: SharedHybridCache, hidden: Tensor) -> None:
        if type(cache) is not SharedHybridCache:
            raise TypeError("initial_cache must be SharedHybridCache")
        batch = hidden.shape[0]
        expected = {
            "state": ((batch, self.H, self.dk, self.dv), torch.float32),
            "phase": ((batch, self.H, self.dk // 2), torch.float32),
            "previous_value": ((batch, self.H, RANK, self.dv), torch.float32),
            "previous_write": ((batch, self.H, RANK, self.dk, self.dv), torch.float32),
            "conv_tail": ((batch, self.conv_k - 1, self.components.hidden), hidden.dtype),
            "has_history": ((batch,), torch.bool),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(cache, name)
            if (not isinstance(value, Tensor) or tuple(value.shape) != shape
                    or value.dtype != dtype or value.device != hidden.device):
                raise ValueError(f"initial_cache {name} shape/dtype/device mismatch")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"initial_cache {name} must be finite")

    @staticmethod
    def _adjacent_complex_layout(value: Tensor) -> Tensor:
        """Map native half-split real/imag channels to the oracle's adjacent layout."""
        half = value.shape[-1] // 2
        return torch.stack((value[..., :half], value[..., half:]), -1).flatten(-2)

    def _scan(self, hidden: Tensor, boundary: Tensor, valid: Tensor,
              initial_cache: SharedHybridCache | None) -> tuple[Tensor, SharedHybridCache]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.components.hidden:
            raise ValueError("hidden states have the wrong shape")
        B, T = hidden.shape[:2]
        if boundary.shape != (B, T) or valid.shape != (B, T) or boundary.dtype != torch.bool or valid.dtype != torch.bool:
            raise ValueError("boundary and valid must be boolean [B,T]")
        device, dtype = hidden.device, hidden.dtype
        if initial_cache is None:
            initialized = self._initial_cache(hidden)
            state, phase = initialized.state, initialized.phase
            previous_value, previous_write = initialized.previous_value, initialized.previous_write
            tail, has_history = initialized.conv_tail, initialized.has_history
            hola_state = initialized.hola_state
        else:
            self._validate_cache(initial_cache, hidden)
            state, phase = initial_cache.state, initial_cache.phase
            previous_value, previous_write, tail = initial_cache.previous_value, initial_cache.previous_write, initial_cache.conv_tail
            has_history = initial_cache.has_history
            hola_state = initial_cache.hola_state
        q, k, v, routing_hidden, new_tail = self._project_convolved(hidden, tail, boundary, valid)
        _, _, _, erase_logits, write_logits, z = self.components.project_inputs(hidden)
        if self._active("affine_qk"):
            q, k = self.components.affine_qk(q, k)
        q, k = self._adjacent_complex_layout(q), self._adjacent_complex_layout(k)
        q = F.normalize(q.float(), dim=-1, eps=1e-6) * self.dk ** -0.5
        k = F.normalize(k.float(), dim=-1, eps=1e-6)
        v, erase, write = v.float(), erase_logits.float().sigmoid(), write_logits.float().sigmoid()
        native_gamma = self.components.native_decay(hidden).float()
        if self._active("braid"):
            probabilities = self.components.braid_probabilities(routing_hidden).float()
            self.components.record_runtime_braid_probabilities(probabilities, valid)
            residual = self.components.braid_residual.float()[None, None].expand_as(probabilities)
            tau = residual.new_tensor(SHARED_TIMESCALES)
            gamma = braided_decay(native_gamma, probabilities, residual, tau)
        else:
            gamma = native_gamma
        gamma = self._adjacent_complex_layout(gamma)
        erase = self._adjacent_complex_layout(erase)
        if self._active("rotation"):
            theta = F.softplus(self.rot_proj(hidden)).reshape(B, T, self.H, self.dk // 2).float()
            theta = theta + self.components.phase_logits(routing_hidden).float()
        else:
            # Inference-time rotation reliance control mirroring Package B.
            theta = hidden.new_zeros(B, T, self.H, self.dk // 2, dtype=torch.float32)
        cache_active = self._active("cache_policy", "hola_exact_outer_w64") != "none"
        if cache_active:
            hola_state = self.hola._empty(B, hidden.device) if hola_state is None else hola_state
            self.hola._validate_inputs(q, k, v, torch.zeros(B, T, self.H, device=hidden.device),
                                       None, valid, boundary)
            self.hola._validate_state(hola_state, B, hidden.device)
        outputs = []
        for token in range(T):
            reset = (boundary[:, token] & valid[:, token])[:, None, None]
            state = torch.where(reset[..., None], torch.zeros_like(state), state)
            phase = torch.where(reset, torch.zeros_like(phase), phase)
            previous_value = torch.where(reset[..., None], torch.zeros_like(previous_value), previous_value)
            previous_write = torch.where(reset[..., None, None], torch.zeros_like(previous_write), previous_write)
            active = valid[:, token, None, None]
            history_active = self.history_active(boundary[:, token], valid[:, token], has_history)
            next_phase = phase + theta[:, token]
            phase = torch.where(active, next_phase, phase)
            qr = apply_complex_rotation(q[:, token], phase[:, :, None].expand(-1, -1, RANK, -1))
            kr = apply_complex_rotation(k[:, token], phase[:, :, None].expand(-1, -1, RANK, -1))
            rho = (self.components.lookahead_gate.float()[None, :, :, None]
                   if self._active("lookahead") else 0.0) * history_active[:, None, None, None]
            vv = v[:, token] + rho * (v[:, token] - previous_value)
            next_state, _, write_delta = shared_state_step(
                state, kr, vv, erase[:, token], write[:, token], gamma[:, token],
                self.components.c.float()[None].expand(B, -1, -1),
                self.components.d.float()[None].expand(B, -1, -1), qr,
                self.components.output_mixer.float(), previous_write=previous_write,
                trap_rho=(self.components.trapezoid_gate.float()
                          if self._active("trapezoid") else None),
                history_active=(history_active if self._active("trapezoid") else None),
            )
            post_trap_update = next_state - gamma[:, token, ..., None] * state
            state = torch.where(active[..., None], next_state, state)
            previous_write = torch.where(active[..., None, None], write_delta, previous_write)
            previous_value = torch.where(active[..., None], v[:, token], previous_value)
            has_history = has_history | valid[:, token]
            reads = torch.einsum("bhrk,bhkv->bhrv", qr, state)
            zr = z[:, token].float()
            normalized = []
            for rank in range(RANK):
                normalized.append(self.components.norm(reads[:, :, rank].reshape(-1, self.dv).to(dtype),
                                                         zr[:, :, rank].reshape(-1, self.dv).to(dtype)).reshape(B, self.H, self.dv))
            mixed = torch.einsum("hrvw,bhrw->bhv", self.components.output_mixer, torch.stack(normalized, 2))
            if cache_active:
                from .qwen_hybrid_hola import shared_exact_update_score
                score = shared_exact_update_score(post_trap_update)
                hola_read, hola_state = self.hola.step_unchecked(
                    hola_state, qr, kr, vv, score, hola_state.next_position,
                    valid[:, token], boundary[:, token])
                cache_mixed = self._mix_hola_reads(hola_read)
                mixed = mixed + self.components.cache_gate_amplitude.float()[None, :, None] * cache_mixed
            out = self.components.out_proj(mixed.flatten(1).to(dtype))
            outputs.append(torch.where(valid[:, token, None], out, torch.zeros_like(out)))
        out = torch.stack(outputs, 1) if outputs else hidden.new_empty(B, 0, self.components.hidden)
        return out, SharedHybridCache(state, phase, previous_value, previous_write, new_tail, has_history, hola_state)

    def scan(self, hidden_states: Tensor, *, boundary: Tensor | None = None, valid: Tensor | None = None,
             initial_cache: SharedHybridCache | None = None) -> tuple[Tensor, SharedHybridCache]:
        B, T = hidden_states.shape[:2]
        if T == 0:
            cache = initial_cache if initial_cache is not None else self._initial_cache(hidden_states)
            return hidden_states.new_empty(B, 0, self.components.hidden), cache
        boundary = torch.zeros(B, T, dtype=torch.bool, device=hidden_states.device) if boundary is None else boundary
        valid = torch.ones(B, T, dtype=torch.bool, device=hidden_states.device) if valid is None else valid
        return self._scan(hidden_states, boundary, valid, initial_cache)

    def reference(self, hidden_states: Tensor, *, boundary: Tensor, valid: Tensor) -> Tensor:
        return self._scan(hidden_states, boundary, valid, None)[0]

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None, *,
                boundary: Tensor | None = None, valid: Tensor | None = None, **kwargs) -> Tensor:
        use_cache = kwargs.pop("use_cache", False)
        initial_cache = kwargs.pop("past_key_value", kwargs.pop("past_key_values", None))
        kwargs.pop("cache_position", None)
        cache_params = kwargs.pop("cache_params", None)
        kwargs.pop("output_hidden_states", None)
        if cache_params is not None:
            raise ValueError("unsupported incremental arguments: cache_params")
        if kwargs:
            raise ValueError("unsupported incremental arguments: " + ", ".join(sorted(kwargs)))
        if attention_mask is not None:
            mask = attention_mask if attention_mask.ndim == 2 else attention_mask.squeeze(-1)
            valid = mask.bool() if valid is None else valid & mask.bool()
        out, cache = self.scan(hidden_states, boundary=boundary, valid=valid, initial_cache=initial_cache)
        if use_cache:
            self.last_recurrent_cache = cache
        return out


__all__ = ["QwenSharedBraidHybrid", "SharedHybridCache"]
