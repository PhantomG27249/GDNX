"""Optional fused CUDA core for the channelwise GDN-2 R1 scan.

This module intentionally lives outside :mod:`gdn3`: the native fast-scan
sources are provenance pinned, while ``KMD2ChannelwiseGDN2Attn`` is an
ablation-local retrofit.  The raw operation implements one boundary-free
segment with the exact reference recurrence and FP32 state math::

    decayed = gamma * state
    memory = (erase * key) @ decayed
    state = decayed + outer(key, write * value - memory)
    output = query @ state

Only the campaign fast-path shapes are accepted (B=1, H in {8, 16},
K=V=128, T<=64).  Unsupported inputs fail closed so the caller can retain the
PyTorch token loop as its numerical oracle.
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
_K: Final = 128
_V: Final = 128
_MAX_T: Final = 64
_BLOCK_V: Final = 32
_TILES_V: Final = _V // _BLOCK_V


if triton is not None:

    @triton.jit
    def _gdn2_segment_fwd(
        q_ptr, k_ptr, v_ptr, erase_ptr, write_ptr, gamma_ptr, state_ptr,
        output_ptr, state_out_ptr, state_trace_ptr,
        T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
        V: tl.constexpr, BLOCK_V: tl.constexpr,
        SAVE_TRACE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tiles_v: tl.constexpr = V // BLOCK_V
        tile_v = pid % tiles_v
        head = pid // tiles_v

        offs_k = tl.arange(0, K)
        offs_v = tile_v * BLOCK_V + tl.arange(0, BLOCK_V)
        matrix_offsets = ((head * K + offs_k[:, None]) * V
                          + offs_v[None, :])
        state = tl.load(state_ptr + matrix_offsets)

        for token in tl.range(0, T):
            key_base = (token * H + head) * K
            value_base = (token * H + head) * V
            query = tl.load(q_ptr + key_base + offs_k)
            key = tl.load(k_ptr + key_base + offs_k)
            erase = tl.load(erase_ptr + key_base + offs_k)
            gamma = tl.load(gamma_ptr + key_base + offs_k)
            value = tl.load(v_ptr + value_base + offs_v)
            write = tl.load(write_ptr + value_base + offs_v)

            decayed = gamma[:, None] * state
            memory = tl.sum((erase * key)[:, None] * decayed, axis=0)
            update = write * value - memory
            state = decayed + key[:, None] * update[None, :]
            output = tl.sum(query[:, None] * state, axis=0)
            tl.store(output_ptr + value_base + offs_v, output)

            if SAVE_TRACE:
                trace_offsets = (((token * H + head) * K
                                  + offs_k[:, None]) * V + offs_v[None, :])
                tl.store(state_trace_ptr + trace_offsets, state)

        tl.store(state_out_ptr + matrix_offsets, state)


    @triton.jit
    def _gdn2_segment_bwd(
        q_ptr, k_ptr, v_ptr, erase_ptr, write_ptr, gamma_ptr, state_ptr,
        state_trace_ptr, doutput_ptr, dstate_out_ptr,
        dq_partial_ptr, dk_partial_ptr, derase_partial_ptr,
        dgamma_partial_ptr, dv_ptr, dwrite_ptr, dstate_ptr,
        T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
        V: tl.constexpr, BLOCK_V: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tiles_v: tl.constexpr = V // BLOCK_V
        tile_v = pid % tiles_v
        head = pid // tiles_v

        offs_k = tl.arange(0, K)
        offs_v = tile_v * BLOCK_V + tl.arange(0, BLOCK_V)
        matrix_offsets = ((head * K + offs_k[:, None]) * V
                          + offs_v[None, :])
        dstate = tl.load(dstate_out_ptr + matrix_offsets)

        for reverse_token in tl.range(0, T):
            token = T - 1 - reverse_token
            key_base = (token * H + head) * K
            value_base = (token * H + head) * V
            query = tl.load(q_ptr + key_base + offs_k)
            key = tl.load(k_ptr + key_base + offs_k)
            erase = tl.load(erase_ptr + key_base + offs_k)
            gamma = tl.load(gamma_ptr + key_base + offs_k)
            value = tl.load(v_ptr + value_base + offs_v)
            write = tl.load(write_ptr + value_base + offs_v)

            if token == 0:
                state_before = tl.load(state_ptr + matrix_offsets)
            else:
                before_offsets = (((((token - 1) * H + head) * K)
                                   + offs_k[:, None]) * V + offs_v[None, :])
                state_before = tl.load(state_trace_ptr + before_offsets)

            decayed = gamma[:, None] * state_before
            erased_key = erase * key
            memory = tl.sum(erased_key[:, None] * decayed, axis=0)
            update = write * value - memory
            state_current = decayed + key[:, None] * update[None, :]

            doutput = tl.load(doutput_ptr + value_base + offs_v)
            dstate += query[:, None] * doutput[None, :]
            dq = tl.sum(state_current * doutput[None, :], axis=1)

            dupdate = tl.sum(key[:, None] * dstate, axis=0)
            dk = tl.sum(dstate * update[None, :], axis=1)
            d_erased_key = -tl.sum(decayed * dupdate[None, :], axis=1)
            dk += d_erased_key * erase
            derase = d_erased_key * key
            dwrite = dupdate * value
            dvalue = dupdate * write
            ddecayed = dstate - erased_key[:, None] * dupdate[None, :]
            dgamma = tl.sum(ddecayed * state_before, axis=1)
            dstate = gamma[:, None] * ddecayed

            partial_base = ((((token * H + head) * tiles_v + tile_v) * K))
            tl.store(dq_partial_ptr + partial_base + offs_k, dq)
            tl.store(dk_partial_ptr + partial_base + offs_k, dk)
            tl.store(derase_partial_ptr + partial_base + offs_k, derase)
            tl.store(dgamma_partial_ptr + partial_base + offs_k, dgamma)
            tl.store(dv_ptr + value_base + offs_v, dvalue)
            tl.store(dwrite_ptr + value_base + offs_v, dwrite)

        tl.store(dstate_ptr + matrix_offsets, dstate)


    @triton.jit
    def _reduce_key_partials(
        dq_partial_ptr, dk_partial_ptr, derase_partial_ptr,
        dgamma_partial_ptr, dq_ptr, dk_ptr, derase_ptr, dgamma_ptr,
        T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
        TILES_V: tl.constexpr,
    ):
        pid = tl.program_id(0)
        head = pid % H
        token = pid // H
        offs_k = tl.arange(0, K)
        total_q = tl.zeros((K,), dtype=tl.float32)
        total_k = tl.zeros((K,), dtype=tl.float32)
        total_erase = tl.zeros((K,), dtype=tl.float32)
        total_gamma = tl.zeros((K,), dtype=tl.float32)
        for tile_v in tl.static_range(0, TILES_V):
            base = ((((token * H + head) * TILES_V + tile_v) * K))
            total_q += tl.load(dq_partial_ptr + base + offs_k)
            total_k += tl.load(dk_partial_ptr + base + offs_k)
            total_erase += tl.load(derase_partial_ptr + base + offs_k)
            total_gamma += tl.load(dgamma_partial_ptr + base + offs_k)
        output_base = (token * H + head) * K
        tl.store(dq_ptr + output_base + offs_k, total_q)
        tl.store(dk_ptr + output_base + offs_k, total_k)
        tl.store(derase_ptr + output_base + offs_k, total_erase)
        tl.store(dgamma_ptr + output_base + offs_k, total_gamma)


def triton_gdn2_segment_available() -> bool:
    """Return whether this process can launch the optional CUDA/Triton op."""
    return triton is not None and torch.cuda.is_available()


def _dispatch_failure(
    q: object, k: object, v: object, erase: object, write: object,
    gamma: object, state: object,
) -> str | None:
    if not triton_gdn2_segment_available():
        return "Triton CUDA runtime is unavailable"
    tensors = (q, k, v, erase, write, gamma, state)
    if any(not isinstance(tensor, Tensor) for tensor in tensors):
        return "all arguments must be tensors"
    assert isinstance(q, Tensor)
    if q.ndim != 4 or q.shape[0] != _B or q.shape[2] not in _HEADS:
        return "q must be [1,T,H,128] with H in {8,16}"
    T, H = q.shape[1:3]
    if not 1 <= T <= _MAX_T:
        return "segment length must be in [1,64]"
    key_shape = (_B, T, H, _K)
    value_shape = (_B, T, H, _V)
    if any(tensor.shape != key_shape for tensor in (q, k, erase, gamma)):
        return "q/k/erase/gamma must be [1,T,H,128]"
    if any(tensor.shape != value_shape for tensor in (v, write)):
        return "v/write must be [1,T,H,128]"
    if state.shape != (_B, H, _K, _V):
        return "state must be [1,H,128,128]"
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        return "all inputs must be FP32"
    if q.device.type != "cuda" or any(tensor.device != q.device for tensor in tensors):
        return "all inputs must share one CUDA device"
    if any(not tensor.is_contiguous() for tensor in tensors):
        return "all inputs must be contiguous"
    return None


def can_use_triton_gdn2_segment(
    q: object, k: object, v: object, erase: object, write: object,
    gamma: object, state: object,
) -> bool:
    """Check the exact raw-op dispatch contract without synchronizing CUDA."""
    return _dispatch_failure(q, k, v, erase, write, gamma, state) is None


class _GDN2SegmentFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q: Tensor, k: Tensor, v: Tensor, erase: Tensor, write: Tensor,
        gamma: Tensor, state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        T, H = q.shape[1:3]
        output = torch.empty(
            (_B, T, H, _V), device=q.device, dtype=q.dtype
        )
        state_out = torch.empty_like(state)
        needs_backward = any(ctx.needs_input_grad)
        if needs_backward:
            state_trace = torch.empty(
                (T, H, _K, _V), device=q.device, dtype=q.dtype
            )
        else:
            state_trace = torch.empty((0,), device=q.device, dtype=q.dtype)
        grid = (H * _TILES_V,)
        with torch.cuda.device(q.device):
            _gdn2_segment_fwd[grid](
                q, k, v, erase, write, gamma, state, output, state_out,
                state_trace, T=T, H=H, K=_K, V=_V, BLOCK_V=_BLOCK_V,
                SAVE_TRACE=needs_backward, num_warps=8,
            )
        if needs_backward:
            ctx.save_for_backward(
                q, k, v, erase, write, gamma, state, state_trace
            )
        ctx.segment_tokens = T
        ctx.heads = H
        ctx.set_materialize_grads(False)
        return output, state_out

    @staticmethod
    def backward(
        ctx, doutput: Tensor | None, dstate_out: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        (q, k, v, erase, write, gamma, state,
         state_trace) = ctx.saved_tensors
        T = ctx.segment_tokens
        H = ctx.heads
        doutput = (
            torch.zeros((_B, T, H, _V), device=q.device, dtype=q.dtype)
            if doutput is None else doutput.contiguous()
        )
        dstate_out = (
            torch.zeros_like(state)
            if dstate_out is None else dstate_out.contiguous()
        )

        partial_shape = (_B, T, H, _TILES_V, _K)
        dq_partial = torch.empty(
            partial_shape, device=q.device, dtype=q.dtype
        )
        dk_partial = torch.empty_like(dq_partial)
        derase_partial = torch.empty_like(dq_partial)
        dgamma_partial = torch.empty_like(dq_partial)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        derase = torch.empty_like(erase)
        dwrite = torch.empty_like(write)
        dgamma = torch.empty_like(gamma)
        dstate = torch.empty_like(state)

        grid = (H * _TILES_V,)
        reduction_grid = (T * H,)
        with torch.cuda.device(q.device):
            _gdn2_segment_bwd[grid](
                q, k, v, erase, write, gamma, state, state_trace,
                doutput, dstate_out, dq_partial, dk_partial,
                derase_partial, dgamma_partial, dv, dwrite, dstate,
                T=T, H=H, K=_K, V=_V, BLOCK_V=_BLOCK_V, num_warps=8,
            )
            _reduce_key_partials[reduction_grid](
                dq_partial, dk_partial, derase_partial, dgamma_partial,
                dq, dk, derase, dgamma, T=T, H=H, K=_K,
                TILES_V=_TILES_V, num_warps=4,
            )
        return dq, dk, dv, derase, dwrite, dgamma, dstate


def triton_gdn2_segment(
    q: Tensor, k: Tensor, v: Tensor, erase: Tensor, write: Tensor,
    gamma: Tensor, state: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run one fused channelwise GDN-2 segment or fail before launch."""
    failure = _dispatch_failure(q, k, v, erase, write, gamma, state)
    if failure is not None:
        raise RuntimeError(f"Triton GDN-2 segment is not dispatchable: {failure}")
    return _GDN2SegmentFunction.apply(q, k, v, erase, write, gamma, state)


__all__ = [
    "can_use_triton_gdn2_segment",
    "triton_gdn2_segment",
    "triton_gdn2_segment_available",
]
