"""All-Triton generalized-WY chunked training backend for Package B.

PyTorch owns allocation, dispatch, and the custom-autograd seam.  Gamma
products, CMS/trapezoid factors, coupling matrices, triangular solves,
innovations, state reconstruction, all 16 MIMO reads, and their analytical
VJP execute in Triton with FP32 recurrence math.  Backward rematerializes the
chunk factors without rebuilding a PyTorch graph, numerically inverting a
transition, or calling the persistent token-recurrent kernel.

The separate PyTorch recurrence remains the ground-truth fallback and parity
authority.  The backend is training-only and strictly fail-closed.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .qwen_hybrid_chunkwise import _validate_inputs

try:  # Importing the research package must remain safe without Triton/CUDA.
    import triton
    import triton.language as tl
except (ImportError, RuntimeError):  # pragma: no cover - CPU-only environment
    triton = None
    tl = None


_CANONICAL_HEADS = (8, 16)
_MIN_TOKENS = 16
_MAX_TOKENS = 64
_RANK = 4
_KEY_DIM = 32
_VALUE_DIM = 128
_BLOCK_VALUE = 16
_NUM_WARPS = 4


if triton is not None:

    @triton.jit
    def _wy_reconstruct_reads_fwd(
        q_ptr,
        prefix_ptr,
        transport_ptr,
        innovation_ptr,
        state_ptr,
        update_count_ptr,
        reads_ptr,
        state_out_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Reconstruct generalized-WY states and fuse all MIMO reads.

        Programs are independent over ``(batch, token, head, memory lane,
        value tile)``.  The only sequence reduction is over already-solved WY
        innovations, so this kernel does not perform a tokenwise recurrence.
        """
        row = tl.program_id(0)
        value_block = tl.program_id(1)

        lane = row % R
        head = (row // R) % H
        token = (row // (R * H)) % T
        batch = row // (R * H * T)

        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)

        offs_k = tl.arange(0, K)
        offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        value_mask = offs_v < V

        prefix_offset = (
            ((((batch * T + token) * H + head) * R + lane) * K)
            + offs_k
        )
        state_offset = (
            ((((batch * H + head) * R + lane) * K + offs_k[:, None]) * V)
            + offs_v[None, :]
        )
        state_tile = (
            tl.load(prefix_ptr + prefix_offset).to(tl.float32)[:, None]
            * tl.load(
                state_ptr + state_offset,
                mask=value_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        )

        # transport[token, source] is G_{token:source+1}.  Reducing solved
        # innovations is the generalized-WY reconstruction, not a recurrent
        # dependency between programs or between source iterations.
        for source in tl.range(0, T, num_stages=1):
            # CMS only creates an innovation on a lane's update ticks.  The
            # dense factor ABI keeps explicit zero rows for parity/debugging,
            # but reconstruction must not spend KxV work loading them.
            source_tick = ((update_count + source) % period) == 0
            if source_tick:
                transport_offset = (
                    (((((batch * T + token) * T + source) * H + head) * R
                      + lane) * K)
                    + offs_k
                )
                innovation_offset = (
                    (((((batch * T + source) * H + head) * R + lane) * K
                      + offs_k[:, None]) * V)
                    + offs_v[None, :]
                )
                state_tile += (
                    tl.load(transport_ptr + transport_offset).to(tl.float32)[:, None]
                    * tl.load(
                        innovation_ptr + innovation_offset,
                        mask=value_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                )

        tl.store(
            state_out_ptr + state_offset,
            state_tile,
            mask=(token == T - 1) & value_mask[None, :],
        )

        for query_lane in tl.static_range(0, R):
            q_offset = (
                ((((batch * T + token) * H + head) * R + query_lane) * K)
                + offs_k
            )
            query = tl.load(q_ptr + q_offset).to(tl.float32)
            read = tl.sum(query[:, None] * state_tile, axis=0)
            read_offset = (
                (((((batch * T + token) * H + head) * R + query_lane) * R
                  + lane) * V)
                + offs_v
            )
            tl.store(reads_ptr + read_offset, read, mask=value_mask)


    @triton.jit
    def _wy_reconstruct_reads_bwd(
        q_ptr,
        prefix_ptr,
        transport_ptr,
        innovation_ptr,
        state_ptr,
        update_count_ptr,
        grad_reads_ptr,
        grad_state_out_ptr,
        grad_q_ptr,
        grad_trace_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
        HAS_GRAD_READS: tl.constexpr,
        HAS_GRAD_STATE_OUT: tl.constexpr,
        WRITE_GRAD_Q: tl.constexpr,
        WRITE_GRAD_TRACE: tl.constexpr,
    ):
        """VJP of generalized-WY reconstruction and the 16 read contractions.

        The kernel recomputes a value tile of ``S_t`` only when ``dQ`` is
        requested.  It forms ``dS_t`` directly from read and endpoint
        cotangents, then stores each ``dS_t`` tile exactly once.  Batched FP32
        contractions outside the kernel reduce that trace into compact-factor
        adjoints without atomic contention.
        """
        row = tl.program_id(0)
        value_block = tl.program_id(1)

        lane = row % R
        head = (row // R) % H
        token = (row // (R * H)) % T
        batch = row // (R * H * T)

        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)

        offs_k = tl.arange(0, K)
        offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        value_mask = offs_v < V
        matrix_mask = value_mask[None, :]

        prefix_offset = (
            ((((batch * T + token) * H + head) * R + lane) * K)
            + offs_k
        )
        state_offset = (
            ((((batch * H + head) * R + lane) * K + offs_k[:, None]) * V)
            + offs_v[None, :]
        )
        prefix = tl.load(prefix_ptr + prefix_offset).to(tl.float32)
        initial_state = tl.load(
            state_ptr + state_offset, mask=matrix_mask, other=0.0
        ).to(tl.float32)

        # dQ needs the same reconstructed state tile as forward.  Other
        # adjoints need only dS and the compact factors, so compilation removes
        # this work when q did not request a gradient.
        if WRITE_GRAD_Q:
            reconstructed = prefix[:, None] * initial_state
            for source in tl.range(0, T, num_stages=1):
                source_tick = ((update_count + source) % period) == 0
                if source_tick:
                    transport_offset = (
                        (((((batch * T + token) * T + source) * H + head) * R
                          + lane) * K)
                        + offs_k
                    )
                    innovation_offset = (
                        (((((batch * T + source) * H + head) * R + lane) * K
                          + offs_k[:, None]) * V)
                        + offs_v[None, :]
                    )
                    reconstructed += (
                        tl.load(transport_ptr + transport_offset).to(tl.float32)[:, None]
                        * tl.load(
                            innovation_ptr + innovation_offset,
                            mask=matrix_mask,
                            other=0.0,
                        ).to(tl.float32)
                    )

        grad_reconstructed = tl.zeros((K, BLOCK_V), dtype=tl.float32)
        if HAS_GRAD_READS:
            for query_lane in tl.static_range(0, R):
                q_offset = (
                    ((((batch * T + token) * H + head) * R + query_lane) * K)
                    + offs_k
                )
                read_offset = (
                    (((((batch * T + token) * H + head) * R + query_lane) * R
                      + lane) * V)
                    + offs_v
                )
                query = tl.load(q_ptr + q_offset).to(tl.float32)
                grad_read = tl.load(
                    grad_reads_ptr + read_offset,
                    mask=value_mask,
                    other=0.0,
                ).to(tl.float32)
                grad_reconstructed += query[:, None] * grad_read[None, :]
                if WRITE_GRAD_Q:
                    grad_query = tl.sum(
                        reconstructed * grad_read[None, :], axis=1
                    )
                    tl.atomic_add(grad_q_ptr + q_offset, grad_query)

        if HAS_GRAD_STATE_OUT:
            endpoint_grad = tl.load(
                grad_state_out_ptr + state_offset,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            grad_reconstructed += tl.where(
                token == T - 1, endpoint_grad, 0.0
            )

        if WRITE_GRAD_TRACE:
            grad_trace_offset = (
                (((((batch * T + token) * H + head) * R + lane) * K
                  + offs_k[:, None]) * V)
                + offs_v[None, :]
            )
            tl.store(
                grad_trace_ptr + grad_trace_offset,
                grad_reconstructed,
                mask=matrix_mask,
            )


def liger_chunked_four_state_available() -> bool:
    """Return whether the local Triton runtime can launch the backend."""
    return triton is not None and torch.cuda.is_available()


def _dispatch_failure(
    q: object,
    k: object,
    v: object,
    erase: object,
    write: object,
    gamma: object,
    lam: object,
    state: object,
    previous_key: object,
    previous_value: object,
    history: object,
    update_count: object,
) -> str | None:
    tensors = (
        q, k, v, erase, write, gamma, lam, state,
        previous_key, previous_value, history, update_count,
    )
    if not liger_chunked_four_state_available():
        return "Triton/CUDA is unavailable"
    if any(not isinstance(tensor, Tensor) for tensor in tensors):
        return "all arguments must be tensors"
    assert isinstance(q, Tensor)
    if not torch.is_grad_enabled():
        return "the chunked Liger backend is training-only"
    if torch.are_deterministic_algorithms_enabled():
        return "deterministic algorithms require the PyTorch fallback"
    floating = tensors[:10]
    if not any(tensor.requires_grad for tensor in floating):
        return "at least one recurrent floating input must require gradients"
    try:
        B, T, H, K, V = _validate_inputs(*tensors)
    except (TypeError, ValueError) as error:
        return str(error)
    if B < 1:
        return "batch size must be positive"
    if not (_MIN_TOKENS <= T <= _MAX_TOKENS):
        return f"token count must be in [{_MIN_TOKENS},{_MAX_TOKENS}]"
    if H not in _CANONICAL_HEADS:
        return f"head count must be one of {_CANONICAL_HEADS}"
    if K != _KEY_DIM or V != _VALUE_DIM:
        return "the compact Package-B dimensions must be K=32 and V=128"
    if q.device.type != "cuda":
        return "all inputs must be CUDA tensors"
    return None


def can_use_liger_chunked_four_state(
    q: object,
    k: object,
    v: object,
    erase: object,
    write: object,
    gamma: object,
    lam: object,
    state: object,
    previous_key: object,
    previous_value: object,
    history: object,
    update_count: object,
) -> bool:
    """Check the batch-generic production envelope without device sync."""
    return _dispatch_failure(
        q, k, v, erase, write, gamma, lam, state, previous_key,
        previous_value, history, update_count,
    ) is None


def _launch_reconstruct_reads(
    q: Tensor,
    prefix: Tensor,
    transport: Tensor,
    innovation: Tensor,
    state: Tensor,
    update_count: Tensor,
) -> tuple[Tensor, Tensor]:
    if triton is None:  # pragma: no cover - guarded by public validation
        raise RuntimeError("Triton is unavailable")
    B, T, H, R, K = q.shape
    V = state.shape[-1]
    q_c = q.contiguous()
    prefix_c = prefix.contiguous()
    transport_c = transport.contiguous()
    innovation_c = innovation.contiguous()
    state_c = state.contiguous()
    update_count_c = update_count.contiguous()
    reads = torch.empty((B, T, H, R, R, V), device=q.device, dtype=torch.float32)
    state_out = torch.empty_like(state_c)
    grid = (B * T * H * R, triton.cdiv(V, _BLOCK_VALUE))
    _wy_reconstruct_reads_fwd[grid](
        q_c,
        prefix_c,
        transport_c,
        innovation_c,
        state_c,
        update_count_c,
        reads,
        state_out,
        T=T,
        H=H,
        R=R,
        K=K,
        V=V,
        BLOCK_V=_BLOCK_VALUE,
        num_warps=_NUM_WARPS,
    )
    return reads, state_out


def _launch_reconstruct_reads_backward(
    q: Tensor,
    prefix: Tensor,
    transport: Tensor,
    innovation: Tensor,
    state: Tensor,
    update_count: Tensor,
    grad_reads: Tensor | None,
    grad_state_out: Tensor | None,
    *,
    need_grad_q: bool,
    need_factor_grads: bool,
    need_grad_state: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    """Launch the state-trace-free Triton VJP for reconstruction and reads."""
    if triton is None:  # pragma: no cover - guarded by public validation
        raise RuntimeError("Triton is unavailable")
    if grad_reads is None and grad_state_out is None:
        return (None, None, None, None, None)

    B, T, H, R, K = q.shape
    V = state.shape[-1]
    # Custom backward is first-order by contract.  Cotangent construction must
    # not accidentally build a second autograd graph through rematerialized
    # factors merely because this launcher runs inside ``torch.enable_grad``.
    q_c = q.detach().contiguous()
    prefix_c = prefix.detach().contiguous()
    transport_c = transport.detach().contiguous()
    innovation_c = innovation.detach().contiguous()
    state_c = state.detach().contiguous()
    update_count_c = update_count.detach().contiguous()
    grad_reads_c = (
        grad_reads.detach().contiguous()
        if grad_reads is not None
        else q_c
    )
    grad_state_out_c = (
        grad_state_out.detach().contiguous()
        if grad_state_out is not None
        else state_c
    )

    grad_q = torch.zeros_like(q_c) if need_grad_q else None
    need_grad_trace = need_factor_grads or need_grad_state
    grad_trace = (
        torch.empty_like(innovation_c) if need_grad_trace else None
    )

    # Disabled constexpr branches never dereference their dummy pointers.
    dummy = q_c
    grid = (B * T * H * R, triton.cdiv(V, _BLOCK_VALUE))
    _wy_reconstruct_reads_bwd[grid](
        q_c,
        prefix_c,
        transport_c,
        innovation_c,
        state_c,
        update_count_c,
        grad_reads_c,
        grad_state_out_c,
        grad_q if grad_q is not None else dummy,
        grad_trace if grad_trace is not None else dummy,
        T=T,
        H=H,
        R=R,
        K=K,
        V=V,
        BLOCK_V=_BLOCK_VALUE,
        HAS_GRAD_READS=grad_reads is not None,
        HAS_GRAD_STATE_OUT=grad_state_out is not None,
        WRITE_GRAD_Q=need_grad_q and grad_reads is not None,
        WRITE_GRAD_TRACE=need_grad_trace,
        num_warps=_NUM_WARPS,
    )
    grad_prefix = None
    grad_transport = None
    grad_innovation = None
    grad_state = None
    if grad_trace is not None:
        from .qwen_hybrid_liger_wy import reconstruction_factor_grads

        (
            reconstructed_grad_prefix,
            reconstructed_grad_transport,
            reconstructed_grad_innovation,
            reconstructed_grad_state,
        ) = reconstruction_factor_grads(
            prefix_c,
            transport_c,
            innovation_c,
            state_c,
            grad_trace,
            update_count_c,
        )
        if need_factor_grads:
            grad_prefix = reconstructed_grad_prefix
            grad_transport = reconstructed_grad_transport
            grad_innovation = reconstructed_grad_innovation
        if need_grad_state:
            grad_state = reconstructed_grad_state
    return grad_q, grad_prefix, grad_transport, grad_innovation, grad_state


class _LigerChunkedFourState(torch.autograd.Function):
    """Generalized-WY forward and analytical VJP implemented in Triton."""

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        erase: Tensor,
        write: Tensor,
        gamma: Tensor,
        lam: Tensor,
        state: Tensor,
        previous_key: Tensor,
        previous_value: Tensor,
        history: Tensor,
        update_count: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        from .qwen_hybrid_liger_wy import build_wy_factors

        factors = build_wy_factors(
            k,
            v,
            erase,
            write,
            gamma,
            lam,
            state,
            previous_key,
            previous_value,
            history,
            update_count,
        )
        (
            prefix,
            transport,
            innovation,
            previous_key_out,
            previous_value_out,
            innovation_sq,
            history_out,
            update_count_out,
        ) = factors[:8]
        reads, state_out = _launch_reconstruct_reads(
            q, prefix, transport, innovation, state, update_count
        )
        ctx.save_for_backward(
            q, k, v, erase, write, gamma, lam, state, previous_key,
            previous_value, history, update_count,
        )
        outputs = (
            reads,
            state_out,
            previous_key_out,
            previous_value_out,
            innovation_sq,
            history_out,
            update_count_out,
        )
        ctx.mark_non_differentiable(outputs[4], outputs[5], outputs[6])
        ctx.set_materialize_grads(False)
        return outputs

    @staticmethod
    def backward(
        ctx,
        grad_reads: Tensor | None,
        grad_state: Tensor | None,
        grad_previous_key: Tensor | None,
        grad_previous_value: Tensor | None,
        _grad_innovation_sq: Tensor | None,
        _grad_history: Tensor | None,
        _grad_update_count: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        saved = ctx.saved_tensors
        floating = saved[:10]
        history, update_count = saved[10:]
        requested_indices = [
            index for index, needed in enumerate(ctx.needs_input_grad[:10])
            if needed
        ]
        if not requested_indices:
            return (None,) * 12

        # q participates only in the fused read contraction.  Every remaining
        # VJP is evaluated analytically by the all-Triton WY pipeline; backward
        # rematerializes factors but never rebuilds a PyTorch autograd graph.
        factor_requested_indices = [
            index for index in requested_indices if index != 0
        ]
        from .qwen_hybrid_liger_wy import (
            backward_wy_factors,
            build_wy_factors,
        )

        factors = build_wy_factors(
            floating[1],
            floating[2],
            floating[3],
            floating[4],
            floating[5],
            floating[6],
            floating[7],
            floating[8],
            floating[9],
            history,
            update_count,
        )
        prefix, transport, innovation = factors[:3]
        (
            grad_q,
            grad_prefix,
            grad_transport,
            grad_innovation,
            direct_grad_state,
        ) = _launch_reconstruct_reads_backward(
            floating[0],
            prefix,
            transport,
            innovation,
            floating[7],
            update_count,
            grad_reads,
            grad_state,
            need_grad_q=0 in requested_indices,
            need_factor_grads=bool(factor_requested_indices),
            need_grad_state=7 in requested_indices,
        )

        factor_gradients: tuple[Tensor, ...] = ()
        if factor_requested_indices:
            grad_prefix = (
                torch.zeros_like(prefix)
                if grad_prefix is None else grad_prefix
            )
            grad_transport = (
                torch.zeros_like(transport)
                if grad_transport is None else grad_transport
            )
            grad_innovation = (
                torch.zeros_like(innovation)
                if grad_innovation is None else grad_innovation
            )
            direct_grad_state = (
                torch.zeros_like(floating[7])
                if direct_grad_state is None else direct_grad_state
            )
            factor_gradients = backward_wy_factors(
                floating[1],
                floating[2],
                floating[3],
                floating[4],
                floating[5],
                floating[6],
                floating[7],
                floating[8],
                floating[9],
                history,
                update_count,
                factors,
                grad_prefix,
                grad_transport,
                grad_innovation,
                direct_grad_state,
                grad_previous_key,
                grad_previous_value,
            )

        input_gradients: list[Tensor | None] = [None] * 10
        if 0 in requested_indices:
            input_gradients[0] = (
                grad_q if grad_q is not None else torch.zeros_like(floating[0])
            )
        for index in factor_requested_indices:
            # factor_gradients is ordered exactly as floating[1:10].
            input_gradients[index] = factor_gradients[index - 1]
        return (*input_gradients, None, None)


def liger_chunked_four_state_segment(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    erase: Tensor,
    write: Tensor,
    gamma: Tensor,
    lam: Tensor,
    state: Tensor,
    previous_key: Tensor,
    previous_value: Tensor,
    history: Tensor,
    update_count: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run the opt-in chunked Triton training operator.

    The returned empty diagnostic trace intentionally matches the production
    rematerialized PyTorch ABI.  Invalid calls raise here; module dispatch uses
    :func:`can_use_liger_chunked_four_state` and falls back before calling it.
    """
    failure = _dispatch_failure(
        q, k, v, erase, write, gamma, lam, state, previous_key,
        previous_value, history, update_count,
    )
    if failure is not None:
        raise ValueError(f"chunked Liger dispatch rejected the inputs: {failure}")
    result = _LigerChunkedFourState.apply(
        q, k, v, erase, write, gamma, lam, state, previous_key,
        previous_value, history, update_count,
    )
    diagnostic_trace = result[0].new_empty(0)
    return (result[0], diagnostic_trace, *result[1:])


def set_liger_chunked_training(module: torch.nn.Module, enabled: bool = True) -> int:
    """Enable or disable the default backend on every Package-B child layer.

    Returns the number of layers changed.  This mutates runtime attributes only
    and therefore does not alter checkpoints, parameter names, or state dicts.
    """
    if type(enabled) is not bool:
        raise TypeError("enabled must be a bool")
    changed = 0
    for child in module.modules():
        if hasattr(child, "use_liger_chunked_kernel"):
            child.use_liger_chunked_kernel = enabled
            changed += 1
    return changed


__all__ = [
    "can_use_liger_chunked_four_state",
    "liger_chunked_four_state_available",
    "liger_chunked_four_state_segment",
    "set_liger_chunked_training",
]
