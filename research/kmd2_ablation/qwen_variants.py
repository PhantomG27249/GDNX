"""Qwen-native warm-start wrappers for experimental KMD-2 recurrences."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from gdn3.kmd2_native import KMD2NativeAttn


def _require_floating_tensor(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def trapezoid_reference_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta_e: torch.Tensor,
    beta_w: torch.Tensor,
    rho: torch.Tensor,
    *,
    out_mix: torch.Tensor | None = None,
    boundaries: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference exponential-trapezoid factor-carry recurrence.

    The implementation deliberately retains the native subtract-then-add update
    and adds ``rho * (transported_previous - current_write)``.  It is therefore
    exactly the native arithmetic path at ``rho=0`` while exposing the boundary
    derivative needed to open the warm-start gate.
    """

    q = _require_floating_tensor("q", q)
    k = _require_floating_tensor("k", k)
    v = _require_floating_tensor("v", v)
    g = _require_floating_tensor("g", g)
    beta_e = _require_floating_tensor("beta_e", beta_e)
    beta_w = _require_floating_tensor("beta_w", beta_w)
    rho = _require_floating_tensor("rho", rho)
    if q.ndim != 5:
        raise ValueError("q must have shape [B,T,H,r_out,dk]")
    batch, steps, heads, r_out, key_dim = q.shape
    if k.shape != (batch, steps, heads, key_dim):
        raise ValueError("k must have shape [B,T,H,dk]")
    if v.ndim != 4 or v.shape[:3] != (batch, steps, heads):
        raise ValueError("v must have shape [B,T,H,dv]")
    value_dim = v.shape[-1]
    if value_dim < 1:
        raise ValueError("dv must be positive")
    if g.shape != (batch, steps, heads, key_dim):
        raise ValueError("g must have shape [B,T,H,dk]")
    for name, tensor in (("beta_e", beta_e), ("beta_w", beta_w), ("rho", rho)):
        if tensor.shape != (batch, steps, heads):
            raise ValueError(f"{name} must have shape [B,T,H]")
    if bool(((rho.detach() < 0) | (rho.detach() > 1)).any()):
        raise ValueError("rho must be in [0,1]")
    devices = {tensor.device for tensor in (q, k, v, g, beta_e, beta_w, rho)}
    if len(devices) != 1:
        raise ValueError("all recurrence tensors must share a device")

    if boundaries is None:
        reset = torch.zeros(batch, steps, dtype=torch.bool, device=k.device)
    else:
        if not isinstance(boundaries, torch.Tensor):
            raise TypeError("boundaries must be a torch.Tensor or None")
        if boundaries.dtype != torch.bool or boundaries.shape != (batch, steps):
            raise ValueError("boundaries must be bool with shape [B,T]")
        if boundaries.device != k.device:
            raise ValueError("boundaries must share the recurrence device")
        reset = boundaries

    if r_out > 1:
        if not isinstance(out_mix, torch.Tensor):
            raise ValueError("out_mix is required when r_out > 1")
        _require_floating_tensor("out_mix", out_mix)
        if out_mix.shape != (heads, r_out):
            raise ValueError("out_mix must have shape [H,r_out]")
        if out_mix.device != k.device:
            raise ValueError("out_mix must share the recurrence device")
    elif out_mix is not None:
        _require_floating_tensor("out_mix", out_mix)
        if out_mix.shape != (heads, 1):
            raise ValueError("out_mix must have shape [H,1]")

    count = batch * heads

    def flat(tensor: torch.Tensor, *tail: int) -> torch.Tensor:
        return tensor.permute(1, 0, 2, *range(3, tensor.dim())).reshape(
            steps, count, *tail
        ).float()

    q_flat = flat(q, r_out, key_dim)
    k_flat = flat(k, key_dim)
    v_flat = flat(v, value_dim)
    g_flat = flat(g, key_dim)
    erase_flat = flat(beta_e)
    write_flat = flat(beta_w)
    rho_flat = flat(rho)
    reset_flat = reset.permute(1, 0).unsqueeze(-1).expand(steps, batch, heads)
    reset_flat = reset_flat.reshape(steps, count)
    if r_out > 1:
        assert out_mix is not None
        mix = out_mix[None].expand(batch, -1, -1).reshape(count, 1, r_out).float()

    state = torch.zeros(
        count, key_dim, value_dim, dtype=torch.float32, device=k.device
    )
    previous_key = torch.zeros(
        count, key_dim, dtype=torch.float32, device=k.device
    )
    previous_write = torch.zeros(
        count, value_dim, dtype=torch.float32, device=k.device
    )
    outputs: list[torch.Tensor] = []
    for token in range(steps):
        token_reset = reset_flat[token]
        state = torch.where(token_reset[:, None, None], torch.zeros_like(state), state)
        previous_key = torch.where(
            token_reset[:, None], torch.zeros_like(previous_key), previous_key
        )
        previous_write = torch.where(
            token_reset[:, None], torch.zeros_like(previous_write), previous_write
        )
        state = state * g_flat[token].unsqueeze(-1)
        key = k_flat[token]
        memory = torch.bmm(key.unsqueeze(1), state).squeeze(1)
        erase_outer = torch.bmm(
            key.unsqueeze(2),
            (erase_flat[token].unsqueeze(-1) * memory).unsqueeze(1),
        )
        write_value = write_flat[token].unsqueeze(-1) * v_flat[token]
        current_write = torch.bmm(key.unsqueeze(2), write_value.unsqueeze(1))
        transported_previous = g_flat[token].unsqueeze(-1) * torch.bmm(
            previous_key.unsqueeze(2), previous_write.unsqueeze(1)
        )
        gate = rho_flat[token]
        if token == 0:
            gate = torch.zeros_like(gate)
        gate = torch.where(token_reset, torch.zeros_like(gate), gate)
        state = state - erase_outer
        state = state + current_write
        state = state + gate[:, None, None] * (
            transported_previous - current_write
        )
        read_slots = torch.bmm(q_flat[token], state)
        if r_out > 1:
            read = (read_slots * mix.transpose(1, 2)).sum(1)
        else:
            read = read_slots.squeeze(1)
        outputs.append(read)
        previous_key = key
        previous_write = write_value
    return (
        torch.stack(outputs, 0)
        .reshape(steps, batch, heads, value_dim)
        .permute(1, 0, 2, 3)
    )


class KMD2TrapezoidAttn(KMD2NativeAttn):
    """Native Qwen layer plus an identity-gated trapezoid factor carry."""

    @classmethod
    def from_native(cls, native: KMD2NativeAttn) -> "KMD2TrapezoidAttn":
        if isinstance(native, cls):
            raise ValueError("native layer is already a trapezoid installation")
        if type(native) is not KMD2NativeAttn:
            raise TypeError("native must be an unwrapped KMD2NativeAttn")
        source_state = {
            name: value.detach().clone() for name, value in native.state_dict().items()
        }
        source_parameters = tuple(native.parameters())
        if not source_parameters:
            raise ValueError("native layer has no parameters")
        device = source_parameters[0].device
        model_dtype = source_parameters[0].dtype

        replacement = copy.deepcopy(native)
        replacement.__class__ = cls
        replacement.register_parameter(
            "rho_head",
            nn.Parameter(torch.zeros(native.H, dtype=torch.float32, device=device)),
        )
        rho_proj = nn.Linear(
            native.in_proj_qkv.in_features,
            native.H,
            bias=False,
            device=device,
            dtype=model_dtype,
        )
        nn.init.zeros_(rho_proj.weight)
        replacement.add_module("rho_proj", rho_proj)

        replacement_state = replacement.state_dict()
        inherited_names = set(replacement_state) - {"rho_head", "rho_proj.weight"}
        if inherited_names != set(source_state):
            raise RuntimeError("trapezoid installation changed inherited state names")
        for name, expected in source_state.items():
            if not torch.equal(replacement_state[name], expected):
                raise RuntimeError(
                    f"trapezoid installation changed inherited tensor {name}"
                )
        return replacement

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        boundaries: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if "_trapezoid_signal" in self.__dict__:
            raise RuntimeError("reentrant trapezoid forward is unsupported")
        for field in ("cu_seqlens", "segment_ids", "reset_mask"):
            value = kwargs.get(field)
            populated = value is not None
            if isinstance(value, torch.Tensor):
                populated = value.numel() != 0
            if populated:
                raise ValueError(
                    f"packed metadata {field} is unsupported; pass explicit "
                    "bool boundaries [B,T] for recurrence boundary clearing"
                )
        if boundaries is not None:
            if not isinstance(boundaries, torch.Tensor):
                raise TypeError("boundaries must be a torch.Tensor or None")
            expected_shape = hidden_states.shape[:2]
            if boundaries.dtype != torch.bool or boundaries.shape != expected_shape:
                raise ValueError("boundaries must be bool with shape [B,T]")
            if boundaries.device != hidden_states.device:
                raise ValueError("boundaries must share the hidden-state device")
        signal = torch.sigmoid(
            self.rho_proj(hidden_states.to(self.rho_proj.weight.dtype)).float()
        )
        self._trapezoid_signal = signal
        self._trapezoid_boundaries = boundaries
        try:
            return super().forward(
                hidden_states, attention_mask=attention_mask, **kwargs
            )
        finally:
            del self._trapezoid_signal
            del self._trapezoid_boundaries

    def _scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta_e: torch.Tensor,
        beta_w: torch.Tensor,
    ) -> torch.Tensor:
        signal = self.__dict__.get("_trapezoid_signal")
        if not isinstance(signal, torch.Tensor):
            raise RuntimeError("trapezoid _scan must be called through forward")
        expected_shape = (q.shape[0], q.shape[1], self.H)
        if signal.shape != expected_shape:
            raise RuntimeError(
                f"trapezoid projection shape mismatch: expected {expected_shape}, "
                f"got {tuple(signal.shape)}"
            )
        boundaries = self.__dict__.get("_trapezoid_boundaries")
        if boundaries is not None and not isinstance(boundaries, torch.Tensor):
            raise RuntimeError("invalid internal trapezoid boundary state")
        rho = self.rho_head.view(1, 1, self.H) * signal
        return trapezoid_reference_scan(
            q,
            k,
            v,
            g,
            beta_e,
            beta_w,
            rho,
            out_mix=self.out_mix if self.r_out > 1 else None,
            boundaries=boundaries,
        )

    def project_trapezoid_gate_(self) -> None:
        """Apply the required post-optimizer projection in place."""

        with torch.no_grad():
            self.rho_head.clamp_(0.0, 1.0)


__all__ = [
    "KMD2TrapezoidAttn",
    "trapezoid_reference_scan",
]
