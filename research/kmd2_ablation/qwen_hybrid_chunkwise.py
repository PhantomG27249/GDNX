"""Internal exact Package-B PyTorch recurrence implementations.

Canonical CUDA training segments use the generalized-WY recurrence with an
exact-rematerialization backward.  A narrow B2 cached-decode specialization is
also dispatched internally.  Neither optimization adds a public model option;
``QwenFourStateHybrid``'s complete token recurrence remains the authoritative
fallback and can be forced for ground-truth comparisons.
"""

from __future__ import annotations

import torch
from torch import Tensor


_PERIODS = (1, 16, 64, 256)
_CANONICAL_HEADS = (8, 16)
_MIN_DISPATCH_T = 16
_MAX_DISPATCH_T = 64
_MAX_DISPATCH_BATCH = 2


def _validate_inputs(
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
) -> tuple[int, int, int, int, int]:
    if q.ndim != 5:
        raise ValueError("q must be [B,T,H,4,K]")
    B, T, H, R, K = q.shape
    if T < 1:
        raise ValueError("the chunk must contain at least one token")
    if R != 4:
        raise ValueError("the chunk recurrence requires exactly four lanes")
    if k.shape != q.shape or erase.shape != q.shape or gamma.shape != q.shape:
        raise ValueError("k, erase, and gamma must match q [B,T,H,4,K]")
    if v.ndim != 5 or v.shape[:4] != (B, T, H, R):
        raise ValueError("v must be [B,T,H,4,V]")
    V = v.shape[-1]
    if write.shape != v.shape:
        raise ValueError("write must match v [B,T,H,4,V]")
    if lam.shape != (B, T, H, R):
        raise ValueError("lambda must be [B,T,H,4]")
    if state.shape != (B, H, R, K, V):
        raise ValueError("state must be [B,H,4,K,V]")
    if previous_key.shape != (B, H, R, K):
        raise ValueError("previous_key must be [B,H,4,K]")
    if previous_value.shape != (B, H, R, V):
        raise ValueError("previous_value must be [B,H,4,V]")
    if history.shape != (B, R) or history.dtype != torch.bool:
        raise ValueError("history must be bool [B,4]")
    if update_count.shape != (B,) or update_count.dtype != torch.int64:
        raise ValueError("update_count must be int64 [B]")
    floating = (
        q, k, v, erase, write, gamma, lam, state,
        previous_key, previous_value,
    )
    if any(tensor.dtype != torch.float32 for tensor in floating):
        raise TypeError("all recurrent floating inputs must be FP32")
    if any(tensor.device != q.device for tensor in (*floating, history, update_count)):
        raise ValueError("all inputs must share one device")
    return B, T, H, K, V


def can_use_torch_chunk_four_state_segment(
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
    """Check the narrow inference dispatch envelope without synchronizing."""
    tensors = (
        q, k, v, erase, write, gamma, lam, state,
        previous_key, previous_value, history, update_count,
    )
    if torch.is_grad_enabled() or any(
        not isinstance(tensor, Tensor) for tensor in tensors
    ):
        return False
    assert isinstance(q, Tensor)
    if q.ndim != 5:
        return False
    B, T, H, R, K = q.shape
    if not (
        1 <= B <= _MAX_DISPATCH_BATCH
        and _MIN_DISPATCH_T <= T <= _MAX_DISPATCH_T
        and H in _CANONICAL_HEADS
        and R == 4
        and K == 32
    ):
        return False
    key_shape = (B, T, H, R, K)
    value_shape = (B, T, H, R, 128)
    if any(tensor.shape != key_shape for tensor in (k, erase, gamma)):
        return False
    if any(tensor.shape != value_shape for tensor in (v, write)):
        return False
    if lam.shape != (B, T, H, R):
        return False
    if state.shape != (B, H, R, K, 128):
        return False
    if previous_key.shape != (B, H, R, K):
        return False
    if previous_value.shape != (B, H, R, 128):
        return False
    if history.shape != (B, R) or history.dtype != torch.bool:
        return False
    if update_count.shape != (B,) or update_count.dtype != torch.int64:
        return False
    floating = (
        q, k, v, erase, write, gamma, lam, state,
        previous_key, previous_value,
    )
    if any(tensor.dtype != torch.float32 for tensor in floating):
        return False
    return q.device.type == "cuda" and all(
        tensor.device == q.device for tensor in tensors
    )


def can_use_rematerialized_torch_chunk_four_state_segment(
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
    """Check the production generalized-WY training envelope.

    Shape, dtype, device, and segment limits intentionally stay identical to
    the already validated inference prototype.  Training additionally requires
    a live autograd graph; grad-disabled execution retains the bit-exact token
    recurrence used as the authority for HOLA survivor selection.
    """
    floating = (
        q, k, v, erase, write, gamma, lam, state,
        previous_key, previous_value,
    )
    if (
        not torch.is_grad_enabled()
        or any(not isinstance(tensor, Tensor) for tensor in floating)
        or not any(tensor.requires_grad for tensor in floating)
    ):
        return False
    # Reuse one fail-closed envelope instead of maintaining a second copy.  Its
    # only inference-specific condition is grad mode, which is suppressed for
    # this validation call without changing any tensor or synchronization.
    with torch.no_grad():
        return can_use_torch_chunk_four_state_segment(
            q, k, v, erase, write, gamma, lam, state, previous_key,
            previous_value, history, update_count,
        )


def _can_use_torch_four_state_decode_step(
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
    """Check the canonical Package-B T=1 direct-recurrence envelope."""
    tensors = (
        q, k, v, erase, write, gamma, lam, state,
        previous_key, previous_value, history, update_count,
    )
    if torch.is_grad_enabled() or any(
        not isinstance(tensor, Tensor) for tensor in tensors
    ):
        return False
    assert isinstance(q, Tensor)
    if torch.is_autocast_enabled(q.device.type):
        return False
    if q.ndim != 5:
        return False
    B, T, H, R, K = q.shape
    if not (B in (1, 2) and T == 1 and H == 16 and R == 4 and K == 32):
        return False
    key_shape = (B, 1, H, R, K)
    value_shape = (B, 1, H, R, 128)
    if any(tensor.shape != key_shape for tensor in (k, erase, gamma)):
        return False
    if any(tensor.shape != value_shape for tensor in (v, write)):
        return False
    if lam.shape != (B, 1, H, R):
        return False
    if state.shape != (B, H, R, K, 128):
        return False
    if previous_key.shape != (B, H, R, K):
        return False
    if previous_value.shape != (B, H, R, 128):
        return False
    if history.shape != (B, R) or history.dtype != torch.bool:
        return False
    if update_count.shape != (B,) or update_count.dtype != torch.int64:
        return False
    floating = (
        q, k, v, erase, write, gamma, lam, state,
        previous_key, previous_value,
    )
    if any(tensor.dtype != torch.float32 for tensor in floating):
        return False
    return q.device.type == "cuda" and all(
        tensor.device == q.device for tensor in tensors
    )


def _can_use_package_b_decode_step(
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
    """Restrict the production specialization to the winning B2 envelope."""
    return (
        isinstance(q, Tensor)
        and q.ndim == 5
        and q.shape[0] == 2
        and _can_use_torch_four_state_decode_step(
            q, k, v, erase, write, gamma, lam, state,
            previous_key, previous_value, history, update_count,
        )
    )


def _torch_four_state_decode_step(
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
    """Evaluate one exact compact recurrence token without a T-by-T WY solve."""
    B, T, _H, _K, _V = _validate_inputs(
        q, k, v, erase, write, gamma, lam, state, previous_key,
        previous_value, history, update_count,
    )
    if T != 1:
        raise ValueError("the direct decode recurrence requires exactly one token")

    zero = state.new_zeros(())
    one = state.new_ones(())
    periods = update_count.new_tensor(_PERIODS)
    ticks = update_count[:, None].remainder(periods[None]).eq(0)
    tick_state = ticks[:, None, :, None, None]
    tick_vector = ticks[:, None, :, None]
    q_t, k_t = q[:, 0], k[:, 0]
    v_t, erase_t, write_t = v[:, 0], erase[:, 0], write[:, 0]
    gamma_t = gamma[:, 0, ..., None]

    decayed = gamma_t * state
    erased_key = erase_t * k_t
    with torch.autocast(q.device.type, enabled=False):
        memory = torch.einsum("bhrk,bhrkv->bhrv", erased_key, decayed)
    full_homogeneous = decayed - k_t[..., None] * memory[..., None, :]
    homogeneous = torch.where(tick_state, full_homogeneous, decayed)

    previous_decayed_key = gamma[:, 0] * previous_key
    with torch.autocast(q.device.type, enabled=False):
        previous_memory = torch.einsum(
            "bhrk,bhrk->bhr", erased_key, previous_decayed_key,
        )
    previous_transported_key = (
        previous_decayed_key - k_t * previous_memory[..., None]
    )
    previous_transported = (
        previous_transported_key[..., None] * previous_value[..., None, :]
    )
    current_value = write_t * v_t
    current_write = k_t[..., None] * current_value[..., None, :]
    effective_lambda = lam[:, 0, ..., None, None]
    effective_lambda = torch.where(
        history[:, None, :, None, None], effective_lambda, one,
    )
    tick_update = (
        (1.0 - effective_lambda) * previous_transported
        + effective_lambda * current_write
    )
    input_update = torch.where(tick_state, tick_update, zero)
    next_state = homogeneous + input_update
    innovation = next_state - decayed
    innovation_sq = innovation.square().sum((-2, -1)).detach()
    with torch.autocast(q.device.type, enabled=False):
        reads = torch.einsum(
            "bhik,bhjkv->bhijv", q_t, next_state
        )[:, None]
    previous_key_out = torch.where(tick_vector, k_t, previous_decayed_key)
    previous_value_out = torch.where(
        tick_vector, current_value, previous_value
    )
    return (
        reads,
        next_state,
        previous_key_out,
        previous_value_out,
        innovation_sq[:, None],
        history | ticks,
        update_count + 1,
    )


def _interval_products(gamma: Tensor) -> tuple[Tensor, Tensor]:
    """Return G[t:0] and G[t:j+1], using products without division.

    ``G[t:j] = D_t D_{t-1} ... D_j`` and ``G[t:t+1] = I``.  Because every
    D is diagonal, the tensors below store only their coordinatewise diagonals.
    """
    B, T, H, R, K = gamma.shape
    prefix = gamma.cumprod(1)
    identity = torch.ones_like(gamma[:, :1])
    columns = []
    for source in range(T):
        before = gamma.new_zeros(B, source, H, R, K)
        after = (
            gamma[:, source + 1:].cumprod(1)
            if source + 1 < T
            else gamma.new_empty(B, 0, H, R, K)
        )
        columns.append(torch.cat((before, identity, after), 1))
    return prefix, torch.stack(columns, 2)


def _gather_endpoint_source(
    initial: Tensor, tokens: Tensor, previous_tick: Tensor
) -> Tensor:
    """Gather initial or prior-tick endpoint values without a token scan."""
    B, T, H, R = tokens.shape[:4]
    width = tokens.shape[-1]
    padded = torch.cat(
        (initial.unsqueeze(3), tokens.permute(0, 2, 3, 1, 4)), 3
    )
    indices = (previous_tick + 1).permute(0, 2, 1)
    indices = indices[:, None, :, :, None].expand(B, H, R, T, width)
    return padded.gather(3, indices).permute(0, 3, 1, 2, 4)


def _torch_chunk_four_state_segment_fp32(
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
    *,
    _return_factors: bool = False,
) -> tuple[Tensor, ...]:
    """Evaluate the exact boundary-free, all-valid compact recurrence.

    Let ``D_t=diag(gamma_t)``, ``u_t=erase_t*k_t`` and
    ``w_t=write_t*v_t``.  At ticking indices,

    ``a_t=(I-k_t u_t^T)D_t p_{t-1}``,
    ``b_t=(1-lambda_t)c_{t-1}``, and ``d_t=lambda_t w_t``.

    With ``rho_t=u_t^T D_t S_{t-1}``, the transition is

    ``S_t=D_t S_{t-1}+k_t(d_t-rho_t)^T+a_t b_t^T``.

    The implementation embeds the tick-compressed system in a full T-by-T
    unit-lower matrix.  Every off-tick row and column is masked.  It solves
    ``(I+K)rho=H+K d+P b`` chronologically with ``solve_triangular`` and
    reconstructs every state from interval products; it never divides by a
    decay, forms a numerical inverse, or scans the matrix state tokenwise.
    """
    B, T, H, K, V = _validate_inputs(
        q, k, v, erase, write, gamma, lam, state, previous_key,
        previous_value, history, update_count,
    )
    R = 4
    periods = update_count.new_tensor(_PERIODS)
    positions = update_count[:, None] + torch.arange(
        T, device=update_count.device, dtype=update_count.dtype
    )[None]
    ticks = positions[:, :, None].remainder(periods[None, None]).eq(0)
    tick_bthrk = ticks[:, :, None, :, None]
    tick_bthr = ticks[:, :, None, :]

    prefix, transport = _interval_products(gamma)
    # transport[t,j] is exactly G_{t:j+1}; its diagonal is the empty product I.
    source_transport = torch.cat((prefix.unsqueeze(2), transport), 2)

    token_indices = torch.arange(T, device=q.device, dtype=torch.int64)
    tick_indices = torch.where(
        ticks, token_indices[None, :, None], torch.full_like(ticks, -1, dtype=torch.int64)
    )
    last_tick = tick_indices.cummax(1).values
    previous_tick = torch.cat(
        (torch.full_like(last_tick[:, :1], -1), last_tick[:, :-1]), 1
    )

    current_value = write * v
    endpoint_key = _gather_endpoint_source(previous_key, k, previous_tick)
    endpoint_value = _gather_endpoint_source(
        previous_value, current_value, previous_tick
    )
    source_indices = (previous_tick + 1)[:, :, None, None, :, None]
    source_indices = source_indices.expand(B, T, 1, H, R, K)
    endpoint_decay = source_transport.gather(2, source_indices).squeeze(2)
    previous_key_decayed = endpoint_decay * endpoint_key

    erased_key = erase * k
    previous_projection = (erased_key * previous_key_decayed).sum(-1)
    a = previous_key_decayed - k * previous_projection[..., None]
    prior_tick = torch.cat(
        (torch.zeros_like(ticks[:, :1]), ticks[:, :-1].cummax(1).values), 1
    )
    history_before = history[:, None, :] | prior_tick
    effective_lam = torch.where(
        history_before[:, :, None, :], lam, torch.ones_like(lam)
    )
    b = (1.0 - effective_lam)[..., None] * endpoint_value
    d = effective_lam[..., None] * current_value

    zeros_k = torch.zeros_like(k)
    zeros_v = torch.zeros_like(d)
    k_tick = torch.where(tick_bthrk, k, zeros_k)
    u_tick = torch.where(tick_bthrk, erased_key, zeros_k)
    a = torch.where(tick_bthrk, a, zeros_k)
    b = torch.where(tick_bthr[..., None], b, zeros_v)
    d = torch.where(tick_bthr[..., None], d, zeros_v)

    # K[t,j]=u_t^T G_{t:j+1} k_j and P[t,j]=u_t^T G_{t:j+1} a_j.
    coupling_k = torch.einsum(
        "bthrk,btjhrk,bjhrk->bhrtj", u_tick, transport, k_tick
    )
    coupling_p = torch.einsum(
        "bthrk,btjhrk,bjhrk->bhrtj", u_tick, transport, a
    )
    strict_lower = torch.tril(
        torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=-1
    )
    coupling_k = torch.where(strict_lower, coupling_k, 0.0)
    coupling_p = torch.where(strict_lower, coupling_p, 0.0)

    if _return_factors:
        # The fused Triton path consumes compact WY factors and never needs the
        # dense transported base-state trace.  Contract the two diagonal
        # coefficients first so forward and backward avoid materializing
        # O(B*T*H*R*K*V) storage merely to reduce it over K.
        h_term = torch.einsum(
            "bthrk,bhrkv->bhrtv", u_tick * prefix, state
        )
    else:
        base_state = prefix[..., None] * state[:, None]
        h_term = torch.einsum("bthrk,bthrkv->bhrtv", u_tick, base_state)
    d_bhrtv = d.permute(0, 2, 3, 1, 4)
    rhs = (
        h_term
        + torch.einsum("bhrtj,bjhrv->bhrtv", coupling_k, d)
        + torch.einsum("bhrtj,bjhrv->bhrtv", coupling_p, b)
    )
    system = coupling_k + torch.eye(T, dtype=q.dtype, device=q.device)
    rho = torch.linalg.solve_triangular(
        system, rhs, upper=False, unitriangular=True
    )
    y = (d_bhrtv - rho).permute(0, 3, 1, 2, 4)
    innovation = (
        k_tick[..., None] * y[..., None, :]
        + a[..., None] * b[..., None, :]
    )

    final_tick = last_tick[:, -1]
    final_source = final_tick + 1
    padded_key = torch.cat(
        (previous_key.unsqueeze(3), k.permute(0, 2, 3, 1, 4)), 3
    )
    padded_value = torch.cat(
        (
            previous_value.unsqueeze(3),
            current_value.permute(0, 2, 3, 1, 4),
        ),
        3,
    )
    key_index = final_source[:, None, :, None, None].expand(B, H, R, 1, K)
    value_index = final_source[:, None, :, None, None].expand(B, H, R, 1, V)
    previous_key_out = padded_key.gather(3, key_index).squeeze(3)
    previous_value_out = padded_value.gather(3, value_index).squeeze(3)
    final_transport_index = final_source[:, None, None, :, None].expand(
        B, 1, H, R, K
    )
    final_decay = source_transport[:, -1].gather(
        1, final_transport_index
    ).squeeze(1)
    previous_key_out = final_decay * previous_key_out

    innovation_sq = innovation.square().sum((-2, -1)).detach()
    if _return_factors:
        # Private seam for the chunked Triton reconstruction/read kernel.  The
        # generalized-WY solve above remains the mathematical authority; these
        # compact factors avoid retaining the dense per-token state trace.
        return (
            prefix,
            transport,
            innovation,
            previous_key_out,
            previous_value_out,
            innovation_sq,
            history | ticks.any(1),
            update_count + T,
        )

    state_trace = base_state + torch.einsum(
        "btjhrk,bjhrkv->bthrkv", transport, innovation
    )
    reads = torch.einsum("bthik,bthjkv->bthijv", q, state_trace)
    return (
        reads,
        state_trace,
        state_trace[:, -1],
        previous_key_out,
        previous_value_out,
        innovation_sq,
        history | ticks.any(1),
        update_count + T,
    )


def torch_chunk_four_state_segment(
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
    *,
    _return_factors: bool = False,
) -> tuple[Tensor, ...]:
    """Evaluate the generalized-WY recurrence under an FP32 math contract.

    Recurrent floating inputs are already required to be FP32.  Disabling
    ambient autocast here prevents its einsum and triangular-solve policies
    from silently lowering the recurrence while leaving the surrounding model
    free to use BF16/FP16 autocast.
    """
    device_type = q.device.type if isinstance(q, Tensor) else "cpu"
    with torch.autocast(device_type, enabled=False):
        return _torch_chunk_four_state_segment_fp32(
            q, k, v, erase, write, gamma, lam, state, previous_key,
            previous_value, history, update_count,
            _return_factors=_return_factors,
        )


# Rematerialized backward must replay the eager recurrence rather than its
# production wrapper, otherwise recomputation would recurse through autograd.
_eager_torch_chunk_four_state_segment = torch_chunk_four_state_segment


class _RematerializedTorchChunkFourStateSegment(torch.autograd.Function):
    """Recompute the exact eager recurrence instead of retaining its graph."""

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
        result = _eager_torch_chunk_four_state_segment(
            q, k, v, erase, write, gamma, lam, state, previous_key,
            previous_value, history, update_count,
        )
        ctx.save_for_backward(
            q, k, v, erase, write, gamma, lam, state, previous_key,
            previous_value, history, update_count,
        )
        outputs = (
            result[0], result[2].clone(), result[3], result[4], result[5],
            result[6], result[7],
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

        recreated = []
        requested = []
        for index, tensor in enumerate(floating):
            view = tensor.detach()
            if index in requested_indices:
                view.requires_grad_(True)
                requested.append(view)
            recreated.append(view)

        with torch.enable_grad():
            with torch.autocast(floating[0].device.type, enabled=False):
                result = _eager_torch_chunk_four_state_segment(
                    *recreated, history, update_count
                )
            # Differentiate after leaving the local recurrence context; the
            # caller's surrounding AMP state remains untouched.
            outputs = (result[0], result[2], result[3], result[4])
            cotangents = (
                grad_reads, grad_state, grad_previous_key,
                grad_previous_value,
            )
            active = [
                (output, cotangent)
                for output, cotangent in zip(outputs, cotangents, strict=True)
                if cotangent is not None and output.requires_grad
            ]
            if active:
                gradients = torch.autograd.grad(
                    tuple(output for output, _ in active),
                    requested,
                    grad_outputs=tuple(cotangent for _, cotangent in active),
                    allow_unused=True,
                )
            else:
                gradients = (None,) * len(requested)

        input_gradients: list[Tensor | None] = [None] * 10
        for index, gradient in zip(
            requested_indices, gradients, strict=True
        ):
            input_gradients[index] = (
                gradient
                if gradient is not None
                else torch.zeros_like(floating[index])
            )
        return (*input_gradients, None, None)


def _rematerialized_torch_chunk_four_state_segment(
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
    """Custom-autograd view of the generalized-WY chunk recurrence."""
    return _RematerializedTorchChunkFourStateSegment.apply(
        q, k, v, erase, write, gamma, lam, state, previous_key,
        previous_value, history, update_count,
    )


def rematerialized_torch_chunk_four_state_segment(
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
    """Production ABI for memory-efficient generalized-WY training.

    The eager diagnostic state trace is deliberately absent: retaining it is
    the principal activation-memory cost that rematerialization removes.  An
    empty tensor preserves the internal eight-output recurrence ABI without
    allowing callers to mistake a final-state endpoint for a token trace.
    """
    result = _rematerialized_torch_chunk_four_state_segment(
        q, k, v, erase, write, gamma, lam, state, previous_key,
        previous_value, history, update_count,
    )
    diagnostic_trace = result[0].new_empty(0)
    return (result[0], diagnostic_trace, *result[1:])
