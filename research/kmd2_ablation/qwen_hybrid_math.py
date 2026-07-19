"""Dependency-light tensor equations for the Qwen GDN-2 R4 hybrid designs."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

RANK = 4
REFERENCE_IMPLEMENTATION = "pytorch_cuda_fp32_reference_token_loop_hola_host_sync"
# Machine-readable runtime backend, distinct from the semantic-oracle
# identity above: REFERENCE_IMPLEMENTATION names the numerical contract
# every path must match; RUNTIME_BACKEND names what production actually
# dispatches on eligible segments.
RUNTIME_BACKEND = (
    "triton_fp32_fused_or_pytorch_generalized_wy_rematerialized_segment_"
    "recurrence_with_reference_fallback"
)
PRODUCTION_ACCELERATION_STATUS = (
    "strict optional Triton FP32 recurrence forward/backward enabled for "
    "B1 H16/H8 R4 K128 V128 segments; rematerialized generalized-WY training "
    "enabled for canonical B1/B2 16-64 token CUDA segments; vectorized HOLA/"
    "global read mixing and joined checkpoint gradients enabled; fail-closed "
    "reference fallback retained"
)
# Backward-compatible evidence field name used by existing manifests.
DEFERRED_FUSION_WARNING = PRODUCTION_ACCELERATION_STATUS


def _check_tensors(*values: Tensor) -> None:
    if not values:
        raise ValueError("at least one tensor is required")
    for value in values:
        if not isinstance(value, Tensor):
            raise TypeError("all operands must be tensors")
    first = values[0]
    if not first.is_floating_point():
        raise TypeError("oracle tensors must have a floating dtype")
    for value in values:
        if value.dtype != first.dtype or value.device != first.device:
            raise ValueError("all operands must share dtype and device")
        if not value.is_floating_point():
            raise TypeError("oracle tensors must have a floating dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("all operands must be finite")


def _shape(value: Tensor, expected: tuple[int, ...], name: str) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")


def apply_complex_rotation(value: Tensor, phase: Tensor) -> Tensor:
    """Rotate adjacent real channels as complex numbers by ``phase`` radians."""
    _check_tensors(value, phase)
    if value.ndim < 1 or value.shape[-1] <= 0 or value.shape[-1] % 2:
        raise ValueError("value's final dimension must be positive and even")
    _shape(phase, value.shape[:-1] + (value.shape[-1] // 2,), "phase")
    real, imag = value[..., 0::2], value[..., 1::2]
    cosine, sine = torch.cos(phase), torch.sin(phase)
    return torch.stack((real * cosine - imag * sine, real * sine + imag * cosine), dim=-1).flatten(-2)


def cumulative_phase(
    theta: Tensor,
    boundary: Tensor,
    valid: Tensor,
    initial_phase: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Accumulate phase over time, resetting before valid boundary tokens.

    Invalid/padding tokens expose the carried phase and change no state.
    """
    _check_tensors(theta)
    if theta.ndim < 2:
        raise ValueError("theta must have batch and time dimensions")
    if not isinstance(boundary, Tensor) or not isinstance(valid, Tensor):
        raise TypeError("boundary and valid must be tensors")
    if boundary.dtype != torch.bool or valid.dtype != torch.bool:
        raise TypeError("boundary and valid must be boolean")
    if boundary.device != theta.device or valid.device != theta.device:
        raise ValueError("masks and theta must share a device")
    _shape(boundary, theta.shape[:2], "boundary")
    _shape(valid, theta.shape[:2], "valid")
    carry_shape = theta.shape[:1] + theta.shape[2:]
    if initial_phase is None:
        carry = torch.zeros(carry_shape, dtype=theta.dtype, device=theta.device)
    else:
        _check_tensors(theta, initial_phase)
        _shape(initial_phase, carry_shape, "initial_phase")
        carry = initial_phase
    phases = []
    expand = (slice(None),) + (None,) * (theta.ndim - 2)
    for index in range(theta.shape[1]):
        is_valid = valid[:, index][expand]
        reset = (valid[:, index] & boundary[:, index])[expand]
        base = torch.where(reset, torch.zeros_like(carry), carry)
        carry = torch.where(is_valid, base + theta[:, index], carry)
        phases.append(carry)
    output = torch.stack(phases, dim=1) if phases else theta.clone()
    return output, carry


def causal_lookahead(value: Tensor, previous: Tensor, rho: Tensor, active: Tensor) -> Tensor:
    """Apply causal first-difference lookahead where ``active`` is true."""
    _check_tensors(value, previous, rho)
    _shape(previous, value.shape, "previous")
    if not isinstance(active, Tensor):
        raise TypeError("active must be a tensor")
    if active.dtype != torch.bool or active.device != value.device:
        raise TypeError("active must be a boolean tensor on value's device")
    expected = value.shape[:-1]
    _shape(rho, expected, "rho")
    _shape(active, expected, "active")
    if bool(((rho < 0) | (rho > 1)).any()):
        raise ValueError("rho must lie in [0, 1]")
    while rho.ndim < value.ndim:
        rho = rho.unsqueeze(-1)
    gate = active
    while gate.ndim < value.ndim:
        gate = gate.unsqueeze(-1)
    return value + torch.where(gate, rho * (value - previous), torch.zeros_like(value))


def braided_decay(native: Tensor, probabilities: Tensor, residual_logits: Tensor, tau: Tensor) -> Tensor:
    """Package-A residual timescale mixture, neutral when logits are zero."""
    _check_tensors(native, probabilities, residual_logits, tau)
    if native.shape + (RANK,) != probabilities.shape:
        raise ValueError("probabilities must append exactly four timescales to native")
    _shape(residual_logits, probabilities.shape, "residual_logits")
    _shape(tau, (RANK,), "tau")
    floor = 2.0**-24
    if bool(((native < floor) | (native > 1)).any()):
        raise ValueError("native decay must lie in [2^-24, 1]")
    if bool((probabilities < 0).any()):
        raise ValueError("routing probabilities must be nonnegative")
    sums = probabilities.sum(-1)
    if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-6, atol=1e-7):
        raise ValueError("routing probabilities must sum to one")
    if bool((tau <= 0).any()):
        raise ValueError("tau must be positive")
    neutral = torch.nn.functional.softplus(torch.zeros_like(residual_logits))
    delta = -(torch.nn.functional.softplus(residual_logits) - neutral) / tau
    result = (native * torch.exp((probabilities * delta).sum(-1))).clamp(min=floor, max=1.0)
    if not bool(torch.isfinite(result).all()) or bool(((result < floor) | (result > 1)).any()):
        raise ValueError("computed decay must be finite and lie in [2^-24, 1]")
    return result


def identity_output_mixer(
    package: str,
    heads: int,
    value_width: int,
    *,
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Construct the exact warm-start output mixer for Package A or B.

    Package A averages four identical reads of the one shared state, so eye/4
    is output-preserving.  Package B's four states diverge immediately (braided
    horizons, CMS clocks), so averaging sixteen cross reads is NOT
    output-preserving; the warm start reads only the native lane-0 state with
    the lane-0 query ("Option A", 2026-07-15).  Gradients still reach the
    other fifteen mixer blocks because their reads are nonzero.
    """
    if not isinstance(heads, int) or not isinstance(value_width, int) or heads <= 0 or value_width <= 0:
        raise ValueError("heads and value_width must be positive integers")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")
    eye = torch.eye(value_width, dtype=dtype, device=device)
    if package == "shared":
        return eye.expand(heads, RANK, value_width, value_width).clone() / RANK
    if package == "four_state":
        mixer = torch.zeros(heads, RANK, RANK, value_width, value_width,
                            dtype=dtype, device=device)
        mixer[:, 0, 0] = eye
        return mixer
    raise ValueError("package must be 'shared' or 'four_state'")


def shared_state_step(
    state: Tensor,
    key: Tensor,
    value: Tensor,
    erase: Tensor,
    write: Tensor,
    gamma: Tensor,
    erase_weight: Tensor,
    write_weight: Tensor,
    query: Tensor,
    output_mixer: Tensor,
    *,
    previous_write: Optional[Tensor] = None,
    trap_rho: Optional[Tensor] = None,
    history_active: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """One simultaneous Package-A update and its four-query mixed read."""
    _check_tensors(state, key, value, erase, write, gamma, erase_weight, write_weight, query, output_mixer)
    if state.ndim != 4:
        raise ValueError("state must be [B,H,K,V]")
    b, h, kdim, vdim = state.shape
    _shape(key, (b, h, RANK, kdim), "key")
    _shape(query, key.shape, "query")
    _shape(value, (b, h, RANK, vdim), "value")
    _shape(erase, key.shape, "erase")
    _shape(write, value.shape, "write")
    _shape(gamma, (b, h, kdim), "gamma")
    _shape(erase_weight, (b, h, RANK), "erase_weight")
    _shape(write_weight, (b, h, RANK), "write_weight")
    _shape(output_mixer, (h, RANK, vdim, vdim), "output_mixer")
    decayed = gamma[..., None] * state
    erased_value = torch.einsum("bhrk,bhkv->bhrv", erase * key, decayed)
    erase_delta = torch.einsum("bhr,bhrk,bhrv->bhkv", erase_weight, key, erased_value)
    rank_write = key[..., None] * (write * value)[..., None, :]
    write_delta = torch.einsum("bhr,bhrkv->bhkv", write_weight, rank_write)
    active = -erase_delta + write_delta
    supplied = (previous_write is not None, trap_rho is not None, history_active is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("previous_write, trap_rho, and history_active must be supplied together")
    if previous_write is not None and trap_rho is not None:
        _check_tensors(state, previous_write, trap_rho)
        _shape(previous_write, (b, h, RANK, kdim, vdim), "previous_write")
        if trap_rho.shape not in {(h, RANK), (b, h, RANK)}:
            raise ValueError("trap_rho must be [H,4] or [B,H,4]")
        if not isinstance(history_active, Tensor) or history_active.dtype != torch.bool:
            raise TypeError("history_active must be a boolean tensor")
        if history_active.device != state.device:
            raise ValueError("history_active must share the state device")
        _shape(history_active, (b,), "history_active")
        if bool(((trap_rho < 0) | (trap_rho > 1)).any()):
            raise ValueError("trap_rho must lie in [0, 1]")
        rho = trap_rho.reshape((1, h, RANK) if trap_rho.ndim == 2 else (b, h, RANK))
        trap_delta = torch.einsum(
            "bhr,bhr,bhrkv->bhkv", write_weight, rho.expand(b, -1, -1),
            gamma[:, :, None, :, None] * previous_write - rank_write,
        )
        active = active + history_active[:, None, None, None] * trap_delta
    next_state = decayed + active
    reads = torch.einsum("bhrk,bhkv->bhrv", query, next_state)
    output = torch.einsum("hrvw,bhrw->bhv", output_mixer, reads)
    return next_state, output, rank_write


def four_state_step(
    states: Tensor,
    key: Tensor,
    value: Tensor,
    erase: Tensor,
    write: Tensor,
    gamma: Tensor,
    query: Tensor,
    output_mixer: Tensor,
    *,
    transition_probabilities: Optional[Tensor] = None,
    transition_gate: Optional[Tensor] = None,
    previous_write: Optional[Tensor] = None,
    trap_rho: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """One Package-B step: four owned updates followed by sixteen reads."""
    _check_tensors(states, key, value, erase, write, gamma, query, output_mixer)
    if states.ndim != 5:
        raise ValueError("states must be [B,H,4,K,V]")
    b, h, ranks, kdim, vdim = states.shape
    if ranks != RANK:
        raise ValueError("Package B requires exactly four states")
    _shape(key, (b, h, RANK, kdim), "key")
    _shape(query, key.shape, "query")
    _shape(value, (b, h, RANK, vdim), "value")
    _shape(erase, key.shape, "erase")
    _shape(write, value.shape, "write")
    _shape(gamma, (b, h, RANK, kdim), "gamma")
    _shape(output_mixer, (h, RANK, RANK, vdim, vdim), "output_mixer")
    decayed = gamma[..., None] * states
    if (transition_probabilities is None) != (transition_gate is None):
        raise ValueError("transition probabilities and gate must be supplied together")
    base = decayed if transition_probabilities is None else four_state_transition(
        decayed, transition_probabilities, transition_gate)
    erased = torch.einsum("bhrk,bhrkv->bhrv", erase * key, base)
    erase_delta = key[..., :, None] * erased[..., None, :]
    write_delta = key[..., :, None] * (write * value)[..., None, :]
    active = -erase_delta + write_delta
    if (previous_write is None) != (trap_rho is None):
        raise ValueError("previous_write and trap_rho must be supplied together")
    if previous_write is not None:
        _check_tensors(states, previous_write, trap_rho)
        _shape(previous_write, states.shape, "previous_write")
        if trap_rho.shape not in {(h, RANK), (b, h, RANK)}:
            raise ValueError("trap_rho must be [H,4] or [B,H,4]")
        if bool(((trap_rho < 0) | (trap_rho > 1)).any()):
            raise ValueError("trap_rho must lie in [0, 1]")
        rho = trap_rho.reshape((1, h, RANK, 1, 1) if trap_rho.ndim == 2 else (b, h, RANK, 1, 1))
        active = active + rho * (gamma[..., None] * previous_write - write_delta)
    next_states = base + active
    reads = torch.einsum("bhik,bhjkv->bhijv", query, next_states)
    output = torch.einsum("hijvw,bhijw->bhv", output_mixer, reads)
    return next_states, output, active


def four_state_transition(decayed: Tensor, probabilities: Tensor, gate: Tensor) -> Tensor:
    """Convexly route four independently decayed states over the source axis."""
    _check_tensors(decayed, probabilities, gate)
    if decayed.ndim != 5 or decayed.shape[2] != RANK:
        raise ValueError("decayed states must be [B,H,4,K,V]")
    b, h = decayed.shape[:2]
    _shape(probabilities, (b, h, RANK, RANK), "probabilities")
    if gate.shape not in {(h, RANK), (b, h, RANK)}:
        raise ValueError("gate must be [H,4] or [B,H,4]")
    # Router softmax may have been evaluated in a low-precision module dtype
    # before the reference recurrence promotes it to FP32.
    tolerance = max(1e-7, 1e-2 if probabilities.dtype == torch.float32 else
                    8 * torch.finfo(probabilities.dtype).eps)
    if bool((probabilities < 0).any()) or not torch.allclose(
            probabilities.sum(-1), torch.ones_like(probabilities[..., 0]), rtol=tolerance, atol=tolerance):
        raise ValueError("transition probabilities must be source-row stochastic")
    if bool(((gate < 0) | (gate > .25)).any()):
        raise ValueError("transition gate must lie in [0, 0.25]")
    return four_state_transition_unchecked(decayed, probabilities, gate)


def four_state_transition_unchecked(decayed: Tensor, probabilities: Tensor, gate: Tensor) -> Tensor:
    """Pure tensor transition for callers that validated the full scan inputs once."""
    b, h = decayed.shape[:2]
    coefficient = gate.reshape((1, h, RANK, 1, 1) if gate.ndim == 2 else (b, h, RANK, 1, 1))
    routed = torch.einsum("bhdl,bhlkv->bhdkv", probabilities, decayed)
    # This algebraically identical form is bitwise identity at g=0 while
    # retaining the exact derivative (routed - decayed) for direct gate learning.
    return decayed + coefficient * (routed - decayed)


def exact_outer_score(left: Tensor, right: Tensor) -> Tensor:
    """Frobenius norm of a sum of outer products without materializing it."""
    _check_tensors(left, right)
    if left.ndim < 2 or right.ndim != left.ndim or left.shape[:-1] != right.shape[:-1]:
        raise ValueError("left/right must be [..., N, K] and [..., N, V]")
    left_gram = torch.einsum("...ik,...jk->...ij", left, left)
    right_gram = torch.einsum("...iv,...jv->...ij", right, right)
    squared = (left_gram * right_gram).sum(dim=(-2, -1))
    return torch.sqrt(torch.clamp_min(squared, 0))


__all__ = [
    "apply_complex_rotation",
    "braided_decay",
    "causal_lookahead",
    "cumulative_phase",
    "exact_outer_score",
    "four_state_step",
    "four_state_transition",
    "four_state_transition_unchecked",
    "identity_output_mixer",
    "REFERENCE_IMPLEMENTATION",
    "RUNTIME_BACKEND",
    "PRODUCTION_ACCELERATION_STATUS",
    "DEFERRED_FUSION_WARNING",
    "shared_state_step",
]
