"""Optional fused CUDA core for compact Package-B four-state segments.

The campaign fast path owns four K/4-by-V states (K=32, V=128) and retains
the existing four input lanes, four output queries, and sixteen logical reads.
Off-tick CMS lanes accumulate a K-vector decay scale instead of multiplying a
state matrix.  Trapezoid history is stored exactly as the outer-product factors
``previous_key`` and ``previous_value``.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

try:  # Keep CPU-only installs and source inspection importable.
    import triton
    import triton.language as tl
except (ImportError, OSError):  # pragma: no cover - CPU-only CI.
    triton = None
    tl = None


_B: Final = 1
_HEADS: Final = (8, 16)
_R: Final = 4
_K: Final = 32
_V: Final = 128
_MAX_T: Final = 64


if triton is not None:

    @triton.jit
    def _four_state_segment_fwd(
        q_ptr, k_ptr, v_ptr, erase_ptr, write_ptr, gamma_ptr, lam_ptr,
        state_ptr, previous_key_ptr, previous_value_ptr,
        history_ptr, update_count_ptr,
        reads_ptr, state_out_ptr, previous_key_out_ptr,
        previous_value_out_ptr, innovation_ptr,
        state_trace_ptr, previous_key_trace_ptr, previous_value_trace_ptr,
        T: tl.constexpr, H: tl.constexpr, R: tl.constexpr,
        K: tl.constexpr, V: tl.constexpr, SAVE_TRACES: tl.constexpr,
    ):
        pid = tl.program_id(0)
        lane = pid % R
        head = pid // R
        offs_k = tl.arange(0, K)
        offs_v = tl.arange(0, V)
        matrix_offsets = (((head * R + lane) * K + offs_k[:, None]) * V
                          + offs_v[None, :])
        key_offsets = (head * R + lane) * K + offs_k
        value_offsets = (head * R + lane) * V + offs_v

        state = tl.load(state_ptr + matrix_offsets)
        previous_key = tl.load(previous_key_ptr + key_offsets)
        previous_value = tl.load(previous_value_ptr + value_offsets)
        scale = tl.full((K,), 1.0, tl.float32)
        has_history = tl.load(history_ptr + lane)
        update_count = tl.load(update_count_ptr).to(tl.int64)
        period = tl.where(
            lane == 0, 1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)

        for token in tl.range(0, T):
            key_base = ((token * H + head) * R + lane) * K
            value_base = ((token * H + head) * R + lane) * V
            gamma = tl.load(gamma_ptr + key_base + offs_k)
            key = tl.load(k_ptr + key_base + offs_k)
            erased_key = tl.load(erase_ptr + key_base + offs_k) * key
            value = tl.load(v_ptr + value_base + offs_v)
            write = tl.load(write_ptr + value_base + offs_v)
            scale *= gamma
            tick = ((update_count + token) % period) == 0
            innovation = tl.zeros((K, V), dtype=tl.float32)

            if tick:
                decayed = scale[:, None] * state
                memory = tl.sum(erased_key[:, None] * decayed, axis=0)
                homogeneous = decayed - key[:, None] * memory[None, :]
                decayed_previous_key = scale * previous_key
                previous_memory = tl.sum(erased_key * decayed_previous_key)
                transported_key = decayed_previous_key - key * previous_memory
                write_value = write * value
                lam = tl.load(lam_ptr + (token * H + head) * R + lane)
                lam = tl.where(has_history, lam, 1.0)
                state = (
                    homogeneous
                    + (1.0 - lam) * transported_key[:, None]
                      * previous_value[None, :]
                    + lam * key[:, None] * write_value[None, :]
                )
                previous_key = key
                previous_value = write_value
                scale = tl.full((K,), 1.0, tl.float32)
                innovation = state - decayed
                has_history = True

            physical_previous_key = scale * previous_key
            for query_lane in tl.static_range(0, R):
                query_base = ((token * H + head) * R + query_lane) * K
                query = tl.load(q_ptr + query_base + offs_k)
                # Move lazy scale onto the query.  This avoids a separate KxV
                # decay multiply on off-tick inference while preserving the
                # exact physical read (q * scale)^T state.
                read = tl.sum((query * scale)[:, None] * state, axis=0)
                output_base = ((((token * H + head) * R + query_lane) * R
                                + lane) * V)
                tl.store(reads_ptr + output_base + offs_v, read)

            tl.store(innovation_ptr + (token * H + head) * R + lane,
                     tl.sum(tl.sum(innovation * innovation, axis=0), axis=0))
            if SAVE_TRACES:
                physical_state = scale[:, None] * state
                trace_matrix = ((((token * H + head) * R + lane) * K
                                 + offs_k[:, None]) * V + offs_v[None, :])
                trace_key = (((token * H + head) * R + lane) * K + offs_k)
                trace_value = (((token * H + head) * R + lane) * V + offs_v)
                tl.store(state_trace_ptr + trace_matrix, physical_state)
                tl.store(previous_key_trace_ptr + trace_key, physical_previous_key)
                tl.store(previous_value_trace_ptr + trace_value, previous_value)

        tl.store(state_out_ptr + matrix_offsets, scale[:, None] * state)
        tl.store(previous_key_out_ptr + key_offsets, scale * previous_key)
        tl.store(previous_value_out_ptr + value_offsets, previous_value)


    @triton.jit
    def _four_state_segment_bwd(
        q_ptr, k_ptr, v_ptr, erase_ptr, write_ptr, gamma_ptr, lam_ptr,
        state_ptr, previous_key_ptr, previous_value_ptr,
        history_ptr, update_count_ptr,
        state_trace_ptr, previous_key_trace_ptr, previous_value_trace_ptr,
        dreads_ptr, dstate_out_ptr, dprevious_key_out_ptr,
        dprevious_value_out_ptr,
        dq_partial_ptr, dk_ptr, dv_ptr, derase_ptr, dwrite_ptr,
        dgamma_ptr, dlam_ptr, dstate_ptr, dprevious_key_ptr,
        dprevious_value_ptr,
        T: tl.constexpr, H: tl.constexpr, R: tl.constexpr,
        K: tl.constexpr, V: tl.constexpr,
    ):
        pid = tl.program_id(0)
        lane = pid % R
        head = pid // R
        offs_k = tl.arange(0, K)
        offs_v = tl.arange(0, V)
        matrix_offsets = (((head * R + lane) * K + offs_k[:, None]) * V
                          + offs_v[None, :])
        key_offsets = (head * R + lane) * K + offs_k
        value_offsets = (head * R + lane) * V + offs_v
        dstate = tl.load(dstate_out_ptr + matrix_offsets)
        dprevious_key = tl.load(dprevious_key_out_ptr + key_offsets)
        dprevious_value = tl.load(dprevious_value_out_ptr + value_offsets)
        initial_history = tl.load(history_ptr + lane)
        update_count = tl.load(update_count_ptr).to(tl.int64)
        period = tl.where(
            lane == 0, 1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        remainder = update_count % period
        first_tick = tl.where(remainder == 0, 0, period - remainder)

        for reverse_token in tl.range(0, T):
            token = T - 1 - reverse_token
            key_base = ((token * H + head) * R + lane) * K
            value_base = ((token * H + head) * R + lane) * V
            gamma = tl.load(gamma_ptr + key_base + offs_k)
            key = tl.load(k_ptr + key_base + offs_k)
            erase = tl.load(erase_ptr + key_base + offs_k)
            erased_key = erase * key
            value = tl.load(v_ptr + value_base + offs_v)
            write = tl.load(write_ptr + value_base + offs_v)

            if token == 0:
                state_before = tl.load(state_ptr + matrix_offsets)
                previous_key_before = tl.load(previous_key_ptr + key_offsets)
                previous_value_before = tl.load(previous_value_ptr + value_offsets)
            else:
                trace_matrix = (((((token - 1) * H + head) * R + lane) * K
                                 + offs_k[:, None]) * V + offs_v[None, :])
                trace_key = ((((token - 1) * H + head) * R + lane) * K + offs_k)
                trace_value = ((((token - 1) * H + head) * R + lane) * V + offs_v)
                state_before = tl.load(state_trace_ptr + trace_matrix)
                previous_key_before = tl.load(previous_key_trace_ptr + trace_key)
                previous_value_before = tl.load(previous_value_trace_ptr + trace_value)

            decayed = gamma[:, None] * state_before
            decayed_previous_key = gamma * previous_key_before
            tick = ((update_count + token) % period) == 0
            has_history = initial_history | (token > first_tick)
            lam = tl.load(lam_ptr + (token * H + head) * R + lane)
            lam_effective = tl.where(has_history, lam, 1.0)
            state_current = decayed
            transported_key = decayed_previous_key
            write_value = write * value
            if tick:
                memory = tl.sum(erased_key[:, None] * decayed, axis=0)
                homogeneous = decayed - key[:, None] * memory[None, :]
                previous_memory = tl.sum(erased_key * decayed_previous_key)
                transported_key = decayed_previous_key - key * previous_memory
                state_current = (
                    homogeneous
                    + (1.0 - lam_effective) * transported_key[:, None]
                      * previous_value_before[None, :]
                    + lam_effective * key[:, None] * write_value[None, :]
                )

            for query_lane in tl.static_range(0, R):
                query_base = ((token * H + head) * R + query_lane) * K
                query = tl.load(q_ptr + query_base + offs_k)
                dread_base = ((((token * H + head) * R + query_lane) * R
                               + lane) * V)
                dread = tl.load(dreads_ptr + dread_base + offs_v)
                dstate += query[:, None] * dread[None, :]
                dq_part = tl.sum(state_current * dread[None, :], axis=1)
                dq_base = (((((token * H + head) * R + query_lane) * R
                             + lane) * K))
                tl.store(dq_partial_ptr + dq_base + offs_k, dq_part)

            zero_k = tl.zeros((K,), dtype=tl.float32)
            zero_v = tl.zeros((V,), dtype=tl.float32)
            dk = zero_k
            derase = zero_k
            dgamma = zero_k
            dlam = 0.0
            dvalue = zero_v
            dwrite = zero_v

            if tick:
                # Endpoint factors become (key, write*value) on a tick.
                dk += dprevious_key
                dwrite_value = dprevious_value

                dtransported_key = (
                    (1.0 - lam_effective)
                    * tl.sum(dstate * previous_value_before[None, :], axis=1)
                )
                dprevious_value_before = (
                    (1.0 - lam_effective)
                    * tl.sum(dstate * transported_key[:, None], axis=0)
                )
                dk += lam_effective * tl.sum(
                    dstate * write_value[None, :], axis=1
                )
                dwrite_value += lam_effective * tl.sum(
                    dstate * key[:, None], axis=0
                )
                if has_history:
                    dlam = tl.sum(tl.sum(
                        dstate * (
                            key[:, None] * write_value[None, :]
                            - transported_key[:, None]
                              * previous_value_before[None, :]
                        ), axis=0
                    ), axis=0)

                # transported_key = ad - key * dot(erased_key, ad)
                previous_memory = tl.sum(erased_key * decayed_previous_key)
                ddecayed_previous_key = dtransported_key
                dk -= dtransported_key * previous_memory
                dprevious_memory = -tl.sum(dtransported_key * key)
                derased_key = dprevious_memory * decayed_previous_key
                ddecayed_previous_key += dprevious_memory * erased_key

                # homogeneous = D - key * (erased_key^T D)
                memory = tl.sum(erased_key[:, None] * decayed, axis=0)
                ddecayed = dstate
                dk -= tl.sum(dstate * memory[None, :], axis=1)
                dmemory = -tl.sum(dstate * key[:, None], axis=0)
                derased_key += tl.sum(decayed * dmemory[None, :], axis=1)
                ddecayed += erased_key[:, None] * dmemory[None, :]

                derase = derased_key * key
                dk += derased_key * erase
                dgamma = (
                    tl.sum(ddecayed * state_before, axis=1)
                    + ddecayed_previous_key * previous_key_before
                )
                dstate = gamma[:, None] * ddecayed
                dprevious_key = gamma * ddecayed_previous_key
                dprevious_value = dprevious_value_before
                dwrite = dwrite_value * value
                dvalue = dwrite_value * write
            else:
                dgamma = (
                    tl.sum(dstate * state_before, axis=1)
                    + dprevious_key * previous_key_before
                )
                dstate = gamma[:, None] * dstate
                dprevious_key = gamma * dprevious_key
                # previous_value is an identity carry off tick.

            tl.store(dk_ptr + key_base + offs_k, dk)
            tl.store(derase_ptr + key_base + offs_k, derase)
            tl.store(dgamma_ptr + key_base + offs_k, dgamma)
            tl.store(dlam_ptr + (token * H + head) * R + lane, dlam)
            tl.store(dv_ptr + value_base + offs_v, dvalue)
            tl.store(dwrite_ptr + value_base + offs_v, dwrite)

        tl.store(dstate_ptr + matrix_offsets, dstate)
        tl.store(dprevious_key_ptr + key_offsets, dprevious_key)
        tl.store(dprevious_value_ptr + value_offsets, dprevious_value)


    @triton.jit
    def _reduce_q_partials(
        partial_ptr, output_ptr,
        T: tl.constexpr, H: tl.constexpr, R: tl.constexpr, K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        query_lane = pid % R
        head = (pid // R) % H
        token = pid // (R * H)
        offs_k = tl.arange(0, K)
        total = tl.zeros((K,), dtype=tl.float32)
        for state_lane in tl.static_range(0, R):
            base = (((((token * H + head) * R + query_lane) * R
                      + state_lane) * K))
            total += tl.load(partial_ptr + base + offs_k)
        output_base = ((token * H + head) * R + query_lane) * K
        tl.store(output_ptr + output_base + offs_k, total)


def triton_four_state_segment_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _dispatch_failure(
    q: object, k: object, v: object, erase: object, write: object,
    gamma: object, lam: object, state: object, previous_key: object,
    previous_value: object, history: object, update_count: object,
) -> str | None:
    if not triton_four_state_segment_available():
        return "Triton CUDA runtime is unavailable"
    tensors = (q, k, v, erase, write, gamma, lam, state, previous_key,
               previous_value, history, update_count)
    if any(not isinstance(tensor, Tensor) for tensor in tensors):
        return "all arguments must be tensors"
    assert isinstance(q, Tensor)
    if q.ndim != 5 or q.shape[0] != _B or q.shape[2] not in _HEADS:
        return "q must be [1,T,H,4,32] with H in {8,16}"
    T, H = q.shape[1], q.shape[2]
    if not 1 <= T <= _MAX_T:
        return "segment length must be in [1,64]"
    key_shape = (_B, T, H, _R, _K)
    value_shape = (_B, T, H, _R, _V)
    if any(tensor.shape != key_shape for tensor in (q, k, erase, gamma)):
        return "q/k/erase/gamma must be [1,T,H,4,32]"
    if any(tensor.shape != value_shape for tensor in (v, write)):
        return "v/write must be [1,T,H,4,128]"
    if lam.shape != (_B, T, H, _R):
        return "lambda must be [1,T,H,4]"
    if state.shape != (_B, H, _R, _K, _V):
        return "state must be [1,H,4,32,128]"
    if previous_key.shape != (_B, H, _R, _K):
        return "previous_key must be [1,H,4,32]"
    if previous_value.shape != (_B, H, _R, _V):
        return "previous_value must be [1,H,4,128]"
    if history.shape != (_B, _R) or history.dtype != torch.bool:
        return "history must be bool [1,4]"
    if update_count.shape != (_B,) or update_count.dtype != torch.int64:
        return "update_count must be int64 [1]"
    floating = (q, k, v, erase, write, gamma, lam, state,
                previous_key, previous_value)
    if any(tensor.dtype != torch.float32 for tensor in floating):
        return "all floating inputs must be FP32"
    if q.device.type != "cuda" or any(tensor.device != q.device for tensor in tensors):
        return "all inputs must share one CUDA device"
    if any(not tensor.is_contiguous() for tensor in tensors):
        return "all inputs must be contiguous"
    return None


def can_use_triton_four_state_segment(
    q: object, k: object, v: object, erase: object, write: object,
    gamma: object, lam: object, state: object, previous_key: object,
    previous_value: object, history: object, update_count: object,
) -> bool:
    """Check the raw compact-op dispatch contract without synchronizing."""
    return _dispatch_failure(
        q, k, v, erase, write, gamma, lam, state, previous_key,
        previous_value, history, update_count,
    ) is None


class _FourStateSegmentFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q: Tensor, k: Tensor, v: Tensor, erase: Tensor, write: Tensor,
        gamma: Tensor, lam: Tensor, state: Tensor, previous_key: Tensor,
        previous_value: Tensor, history: Tensor, update_count: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        T, H = q.shape[1], q.shape[2]
        reads = torch.empty((_B, T, H, _R, _R, _V), device=q.device)
        state_out = torch.empty_like(state)
        previous_key_out = torch.empty_like(previous_key)
        previous_value_out = torch.empty_like(previous_value)
        innovation_sq = torch.empty((_B, T, H, _R), device=q.device)
        needs_backward = any(ctx.needs_input_grad[:10])
        if needs_backward:
            state_trace = torch.empty((T, H, _R, _K, _V), device=q.device)
            previous_key_trace = torch.empty((T, H, _R, _K), device=q.device)
            previous_value_trace = torch.empty((T, H, _R, _V), device=q.device)
        else:
            state_trace = torch.empty((0,), device=q.device)
            previous_key_trace = torch.empty((0,), device=q.device)
            previous_value_trace = torch.empty((0,), device=q.device)
        with torch.cuda.device(q.device):
            _four_state_segment_fwd[(H * _R,)](
                q, k, v, erase, write, gamma, lam, state,
                previous_key, previous_value, history, update_count,
                reads, state_out, previous_key_out, previous_value_out,
                innovation_sq, state_trace, previous_key_trace,
                previous_value_trace,
                T=T, H=H, R=_R, K=_K, V=_V, SAVE_TRACES=needs_backward,
                num_warps=8,
            )
        if needs_backward:
            ctx.save_for_backward(
                q, k, v, erase, write, gamma, lam, state,
                previous_key, previous_value, history, update_count,
                state_trace, previous_key_trace, previous_value_trace,
            )
        ctx.segment_tokens, ctx.heads = T, H
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(innovation_sq)
        return (reads, state_out, previous_key_out, previous_value_out,
                innovation_sq)

    @staticmethod
    def backward(
        ctx, dreads: Tensor | None, dstate_out: Tensor | None,
        dprevious_key_out: Tensor | None,
        dprevious_value_out: Tensor | None,
        _dinnovation: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        (q, k, v, erase, write, gamma, lam, state, previous_key,
         previous_value, history, update_count, state_trace,
         previous_key_trace, previous_value_trace) = ctx.saved_tensors
        T, H = ctx.segment_tokens, ctx.heads
        dreads = (torch.zeros((_B, T, H, _R, _R, _V), device=q.device)
                  if dreads is None else dreads.contiguous())
        dstate_out = (torch.zeros_like(state) if dstate_out is None
                      else dstate_out.contiguous())
        dprevious_key_out = (
            torch.zeros_like(previous_key) if dprevious_key_out is None
            else dprevious_key_out.contiguous()
        )
        dprevious_value_out = (
            torch.zeros_like(previous_value) if dprevious_value_out is None
            else dprevious_value_out.contiguous()
        )
        dq_partial = torch.empty((_B, T, H, _R, _R, _K), device=q.device)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        derase, dwrite = torch.empty_like(erase), torch.empty_like(write)
        dgamma, dlam = torch.empty_like(gamma), torch.empty_like(lam)
        dstate = torch.empty_like(state)
        dprevious_key = torch.empty_like(previous_key)
        dprevious_value = torch.empty_like(previous_value)
        with torch.cuda.device(q.device):
            _four_state_segment_bwd[(H * _R,)](
                q, k, v, erase, write, gamma, lam, state,
                previous_key, previous_value, history, update_count,
                state_trace, previous_key_trace, previous_value_trace,
                dreads, dstate_out, dprevious_key_out, dprevious_value_out,
                dq_partial, dk, dv, derase, dwrite, dgamma, dlam,
                dstate, dprevious_key, dprevious_value,
                T=T, H=H, R=_R, K=_K, V=_V, num_warps=8,
            )
            _reduce_q_partials[(T * H * _R,)](
                dq_partial, dq, T=T, H=H, R=_R, K=_K, num_warps=4
            )
        return (dq, dk, dv, derase, dwrite, dgamma, dlam, dstate,
                dprevious_key, dprevious_value, None, None)


def triton_four_state_segment(
    q: Tensor, k: Tensor, v: Tensor, erase: Tensor, write: Tensor,
    gamma: Tensor, lam: Tensor, state: Tensor, previous_key: Tensor,
    previous_value: Tensor, history: Tensor, update_count: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    failure = _dispatch_failure(
        q, k, v, erase, write, gamma, lam, state, previous_key,
        previous_value, history, update_count,
    )
    if failure is not None:
        raise RuntimeError(f"Triton four-state segment is not dispatchable: {failure}")
    return _FourStateSegmentFunction.apply(
        q, k, v, erase, write, gamma, lam, state, previous_key,
        previous_value, history, update_count,
    )


__all__ = [
    "can_use_triton_four_state_segment",
    "triton_four_state_segment",
    "triton_four_state_segment_available",
]
