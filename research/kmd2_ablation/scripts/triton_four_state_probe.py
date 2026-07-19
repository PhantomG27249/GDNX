"""Isolated Triton forward/backward feasibility probe for Package-B recurrence.

This is not a production implementation.  It intentionally accepts already
rotated FP32 q/k vectors and measures only the canonical four-state transition
plus the sixteen direct reads over one boundary-free/all-valid checkpoint
segment.  Backward recomputes an FP32 state/previous-write trajectory, reverses
each value tile independently, and reduces fixed-order gradient partials without
atomics.  The PyTorch oracle below is the corresponding subset of
``QwenFourStateHybrid._hybrid_segment``.  This file is deliberately outside the
canonical dispatch path.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import triton
import triton.language as tl


INTEGRATION_CONTRACT = {
    "inputs": {
        "q_k_erase_gamma": "FP32 contiguous [T,H,4,128]",
        "v_write": "FP32 contiguous [T,H,4,128]",
        "lambda": "FP32 contiguous [T,H,4]",
        "state_previous": "FP32 contiguous [H,4,128,128]",
        "history": "bool contiguous [4]",
        "update_count": "nonnegative scalar count of prior valid tokens",
    },
    "outputs": {
        "reads": "FP32 [T,H,4,4,128]",
        "state_previous": "FP32 [H,4,128,128] endpoints",
        "innovation_sq": "detached FP32 [T,H,4] lane squared norms",
    },
    "dispatch": (
        "B=1, CUDA, boundary-free, all-valid fast path only; 1<=T<=64; "
        "reference fallback for every other shape/mask"
    ),
    "backward": (
        "deterministic reverse-time V tiles and fixed-order partial reductions; "
        "no atomics; FP32 S/P trajectories"
    ),
    "scratch_fp32_elements": (
        "2*T*H*4*128*128 + T*H*4*4*(128/BV)*128 + "
        "3*T*H*4*(128/BV)*128 + T*H*4*(128/BV)"
    ),
}


@triton.jit
def _four_state_segment_fwd(
    q_ptr, k_ptr, v_ptr, erase_ptr, write_ptr, gamma_ptr, lam_ptr,
    state_ptr, previous_ptr, history_ptr, periods_ptr,
    reads_ptr, state_out_ptr, previous_out_ptr, innovation_ptr,
    state_trace_ptr, previous_trace_ptr,
    update_count: tl.constexpr,
    T: tl.constexpr, H: tl.constexpr, R: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BLOCK_V: tl.constexpr,
    WRITE_OUTPUTS: tl.constexpr, STORE_TRACE: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_v: tl.constexpr = V // BLOCK_V
    tile_v = pid % tiles_v
    lane = (pid // tiles_v) % R
    head = pid // (tiles_v * R)

    offs_k = tl.arange(0, K)
    offs_v = tile_v * BLOCK_V + tl.arange(0, BLOCK_V)
    matrix_offsets = (((head * R + lane) * K + offs_k[:, None]) * V
                      + offs_v[None, :])
    state = tl.load(state_ptr + matrix_offsets)
    previous = tl.load(previous_ptr + matrix_offsets)
    has_history = tl.load(history_ptr + lane)
    period = tl.load(periods_ptr + lane)

    for token in tl.range(0, T):
        key_base = ((token * H + head) * R + lane) * K
        value_base = ((token * H + head) * R + lane) * V
        gamma = tl.load(gamma_ptr + key_base + offs_k)
        key = tl.load(k_ptr + key_base + offs_k)
        erased_key = tl.load(erase_ptr + key_base + offs_k) * key
        value = tl.load(v_ptr + value_base + offs_v)
        write = tl.load(write_ptr + value_base + offs_v)

        decayed = gamma[:, None] * state
        previous_decayed = gamma[:, None] * previous
        tick = ((update_count + token) % period) == 0
        innovation = tl.zeros((K, BLOCK_V), dtype=tl.float32)

        if tick:
            memory = tl.sum(erased_key[:, None] * decayed, axis=0)
            homogeneous = decayed - key[:, None] * memory[None, :]
            previous_memory = tl.sum(
                erased_key[:, None] * previous_decayed, axis=0
            )
            previous_transported = (
                previous_decayed - key[:, None] * previous_memory[None, :]
            )
            current_write = key[:, None] * (write * value)[None, :]
            lam = tl.load(lam_ptr + (token * H + head) * R + lane)
            lam = tl.where(has_history, lam, 1.0)
            tick_update = ((1.0 - lam) * previous_transported
                           + lam * current_write)
            state = homogeneous + tick_update
            previous = current_write
            innovation = state - decayed
            has_history = True
        else:
            state = decayed
            previous = previous_decayed

        if WRITE_OUTPUTS:
            # Each state lane is read by every query lane.  This is the exact
            # ``bhik,bhjkv->bhijv`` contraction, tiled only along V.
            for query_lane in tl.static_range(0, R):
                query_base = ((token * H + head) * R + query_lane) * K
                query = tl.load(q_ptr + query_base + offs_k)
                read = tl.sum(query[:, None] * state, axis=0)
                output_offsets = (((((token * H + head) * R + query_lane) * R
                                    + lane) * V) + offs_v)
                tl.store(reads_ptr + output_offsets, read)

            partial = tl.sum(tl.sum(innovation * innovation, axis=0), axis=0)
            partial_offset = (((token * H + head) * R + lane) * tiles_v + tile_v)
            tl.store(innovation_ptr + partial_offset, partial)

        if STORE_TRACE:
            trace_offsets = ((((token * H + head) * R + lane) * K
                              + offs_k[:, None]) * V + offs_v[None, :])
            tl.store(state_trace_ptr + trace_offsets, state)
            tl.store(previous_trace_ptr + trace_offsets, previous)

    tl.store(state_out_ptr + matrix_offsets, state)
    tl.store(previous_out_ptr + matrix_offsets, previous)


@triton.jit
def _four_state_segment_bwd(
    q_ptr, k_ptr, v_ptr, erase_ptr, write_ptr, gamma_ptr, lam_ptr,
    state_ptr, previous_ptr, history_ptr, periods_ptr,
    state_trace_ptr, previous_trace_ptr,
    dreads_ptr, dstate_out_ptr, dprevious_out_ptr,
    dq_partial_ptr, dk_partial_ptr, derase_partial_ptr,
    dgamma_partial_ptr, dlam_partial_ptr, dv_ptr, dwrite_ptr,
    dstate_ptr, dprevious_ptr,
    update_count: tl.constexpr,
    T: tl.constexpr, H: tl.constexpr, R: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BLOCK_V: tl.constexpr,
):
    """Reverse one segment, emitting deterministic V-tile partial gradients."""
    pid = tl.program_id(0)
    tiles_v: tl.constexpr = V // BLOCK_V
    tile_v = pid % tiles_v
    lane = (pid // tiles_v) % R
    head = pid // (tiles_v * R)

    offs_k = tl.arange(0, K)
    offs_v = tile_v * BLOCK_V + tl.arange(0, BLOCK_V)
    matrix_offsets = (((head * R + lane) * K + offs_k[:, None]) * V
                      + offs_v[None, :])
    dstate = tl.load(dstate_out_ptr + matrix_offsets)
    dprevious = tl.load(dprevious_out_ptr + matrix_offsets)
    initial_history = tl.load(history_ptr + lane)
    period = tl.load(periods_ptr + lane)
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
            previous_before = tl.load(previous_ptr + matrix_offsets)
        else:
            trace_offsets = (((((token - 1) * H + head) * R + lane) * K
                              + offs_k[:, None]) * V + offs_v[None, :])
            state_before = tl.load(state_trace_ptr + trace_offsets)
            previous_before = tl.load(previous_trace_ptr + trace_offsets)

        decayed = gamma[:, None] * state_before
        previous_decayed = gamma[:, None] * previous_before
        tick = ((update_count + token) % period) == 0
        has_history = initial_history | (token > first_tick)
        lam = tl.load(lam_ptr + (token * H + head) * R + lane)
        lam_effective = tl.where(has_history, lam, 1.0)

        # Recompute the current state from the saved pre-token trajectory.  It
        # is needed for dQ and avoids a second 512-MiB trace read.
        memory = tl.zeros((BLOCK_V,), dtype=tl.float32)
        previous_memory = tl.zeros((BLOCK_V,), dtype=tl.float32)
        previous_transported = previous_decayed
        current_write = tl.zeros((K, BLOCK_V), dtype=tl.float32)
        state_current = decayed
        if tick:
            memory = tl.sum(erased_key[:, None] * decayed, axis=0)
            homogeneous = decayed - key[:, None] * memory[None, :]
            previous_memory = tl.sum(
                erased_key[:, None] * previous_decayed, axis=0
            )
            previous_transported = (
                previous_decayed - key[:, None] * previous_memory[None, :]
            )
            current_write = key[:, None] * (write * value)[None, :]
            state_current = (
                homogeneous
                + (1.0 - lam_effective) * previous_transported
                + lam_effective * current_write
            )
        else:
            state_current = decayed

        # y[i,v] = sum_k q[i,k] * S[k,v].  Each state-lane/V-tile
        # program contributes a fixed partial to dQ and adds its four read
        # adjoints to the recurrent dS carry.
        for query_lane in tl.static_range(0, R):
            query_base = ((token * H + head) * R + query_lane) * K
            query = tl.load(q_ptr + query_base + offs_k)
            dreads_base = (((((token * H + head) * R + query_lane) * R
                              + lane) * V) + offs_v)
            dread = tl.load(dreads_ptr + dreads_base)
            dstate += query[:, None] * dread[None, :]
            dq_partial = tl.sum(state_current * dread[None, :], axis=1)
            dq_partial_base = (((((token * H + head) * R + query_lane) * R
                                  + lane) * tiles_v + tile_v) * K)
            tl.store(dq_partial_ptr + dq_partial_base + offs_k, dq_partial)

        zero_k = tl.zeros((K,), dtype=tl.float32)
        zero_v = tl.zeros((BLOCK_V,), dtype=tl.float32)
        dk = zero_k
        derase = zero_k
        dgamma = zero_k
        dlam = 0.0
        dvalue = zero_v
        dwrite = zero_v

        if tick:
            # S = A(S_before) + (1-lambda)A(P_before) + lambda*C,
            # P = C.  ``dstate`` and ``dprevious`` are the reverse-time
            # carries on these two outputs.
            d_homogeneous = dstate
            d_previous_transported = (1.0 - lam_effective) * dstate
            d_current_write = lam_effective * dstate + dprevious
            if has_history:
                dlam = tl.sum(tl.sum(
                    dstate * (current_write - previous_transported), axis=0
                ), axis=0)

            # Backward of A(X)=D-k(u^T D), u=erase*k.  For adjoint G,
            # dD=G-u(k^T G); the two key paths are the explicit outer-product
            # key and u=erase*k.
            projected_state_grad = tl.sum(
                key[:, None] * d_homogeneous, axis=0
            )
            d_decayed = (
                d_homogeneous
                - erased_key[:, None] * projected_state_grad[None, :]
            )
            d_erased_from_state = -tl.sum(
                decayed * projected_state_grad[None, :], axis=1
            )
            dk = (
                -tl.sum(d_homogeneous * memory[None, :], axis=1)
                + d_erased_from_state * erase
            )
            derase = d_erased_from_state * key

            projected_previous_grad = tl.sum(
                key[:, None] * d_previous_transported, axis=0
            )
            d_previous_decayed = (
                d_previous_transported
                - erased_key[:, None] * projected_previous_grad[None, :]
            )
            d_erased_from_previous = -tl.sum(
                previous_decayed * projected_previous_grad[None, :], axis=1
            )
            dk += (
                -tl.sum(
                    d_previous_transported * previous_memory[None, :], axis=1
                )
                + d_erased_from_previous * erase
            )
            derase += d_erased_from_previous * key

            write_value = write * value
            dk += tl.sum(d_current_write * write_value[None, :], axis=1)
            d_write_value = tl.sum(d_current_write * key[:, None], axis=0)
            dwrite = d_write_value * value
            dvalue = d_write_value * write

            dgamma = tl.sum(
                d_decayed * state_before
                + d_previous_decayed * previous_before,
                axis=1,
            )
            dstate = gamma[:, None] * d_decayed
            dprevious = gamma[:, None] * d_previous_decayed
        else:
            dgamma = tl.sum(
                dstate * state_before + dprevious * previous_before, axis=1
            )
            dstate = gamma[:, None] * dstate
            dprevious = gamma[:, None] * dprevious

        lane_partial_base = ((((token * H + head) * R + lane) * tiles_v
                              + tile_v) * K)
        tl.store(dk_partial_ptr + lane_partial_base + offs_k, dk)
        tl.store(derase_partial_ptr + lane_partial_base + offs_k, derase)
        tl.store(dgamma_partial_ptr + lane_partial_base + offs_k, dgamma)
        scalar_partial = (((token * H + head) * R + lane) * tiles_v + tile_v)
        tl.store(dlam_partial_ptr + scalar_partial, dlam)
        tl.store(dv_ptr + value_base + offs_v, dvalue)
        tl.store(dwrite_ptr + value_base + offs_v, dwrite)

    tl.store(dstate_ptr + matrix_offsets, dstate)
    tl.store(dprevious_ptr + matrix_offsets, dprevious)


@triton.jit
def _reduce_q_partials(
    partial_ptr, output_ptr,
    T: tl.constexpr, H: tl.constexpr, R: tl.constexpr,
    K: tl.constexpr, TILES_V: tl.constexpr,
):
    pid = tl.program_id(0)
    query_lane = pid % R
    head = (pid // R) % H
    token = pid // (R * H)
    offs_k = tl.arange(0, K)
    total = tl.zeros((K,), dtype=tl.float32)
    for state_lane in tl.static_range(0, R):
        for tile_v in tl.static_range(0, TILES_V):
            base = (((((token * H + head) * R + query_lane) * R
                       + state_lane) * TILES_V + tile_v) * K)
            total += tl.load(partial_ptr + base + offs_k)
    output_base = ((token * H + head) * R + query_lane) * K
    tl.store(output_ptr + output_base + offs_k, total)


@triton.jit
def _reduce_lane_partials(
    dk_partial_ptr, derase_partial_ptr, dgamma_partial_ptr, dlam_partial_ptr,
    dk_ptr, derase_ptr, dgamma_ptr, dlam_ptr,
    T: tl.constexpr, H: tl.constexpr, R: tl.constexpr,
    K: tl.constexpr, TILES_V: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = pid % R
    head = (pid // R) % H
    token = pid // (R * H)
    offs_k = tl.arange(0, K)
    total_k = tl.zeros((K,), dtype=tl.float32)
    total_erase = tl.zeros((K,), dtype=tl.float32)
    total_gamma = tl.zeros((K,), dtype=tl.float32)
    total_lam = 0.0
    for tile_v in tl.static_range(0, TILES_V):
        base = ((((token * H + head) * R + lane) * TILES_V + tile_v) * K)
        total_k += tl.load(dk_partial_ptr + base + offs_k)
        total_erase += tl.load(derase_partial_ptr + base + offs_k)
        total_gamma += tl.load(dgamma_partial_ptr + base + offs_k)
        scalar = (((token * H + head) * R + lane) * TILES_V + tile_v)
        total_lam += tl.load(dlam_partial_ptr + scalar)
    output_base = ((token * H + head) * R + lane) * K
    tl.store(dk_ptr + output_base + offs_k, total_k)
    tl.store(derase_ptr + output_base + offs_k, total_erase)
    tl.store(dgamma_ptr + output_base + offs_k, total_gamma)
    tl.store(dlam_ptr + (token * H + head) * R + lane, total_lam)


def torch_oracle(q, k, v, erase, write, gamma, lam, state, previous,
                 history, periods, update_count):
    outputs = []
    innovations = []
    history = history.clone()
    for token in range(q.shape[0]):
        tick_lanes = (update_count + token).remainder(periods).eq(0)
        tick = tick_lanes[None, :, None, None]
        decayed = gamma[token][..., None] * state
        erased_key = erase[token] * k[token]
        memory = torch.einsum("hrk,hrkv->hrv", erased_key, decayed)
        full_homogeneous = decayed - k[token][..., None] * memory[..., None, :]
        homogeneous = torch.where(tick, full_homogeneous, decayed)
        current_write = k[token][..., None] * (write[token] * v[token])[..., None, :]
        previous_decayed = gamma[token][..., None] * previous
        previous_memory = torch.einsum(
            "hrk,hrkv->hrv", erased_key, previous_decayed
        )
        previous_transported = (
            previous_decayed - k[token][..., None] * previous_memory[..., None, :]
        )
        lam_t = torch.where(history[None], lam[token], torch.ones_like(lam[token]))
        tick_update = ((1.0 - lam_t[..., None, None]) * previous_transported
                       + lam_t[..., None, None] * current_write)
        state = homogeneous + torch.where(tick, tick_update, 0.0)
        previous = torch.where(tick, current_write, previous_decayed)
        history = history | tick_lanes
        outputs.append(torch.einsum("hik,hjkv->hijv", q[token], state))
        innovations.append((state - decayed).square().sum((-2, -1)))
    return torch.stack(outputs), state, previous, torch.stack(innovations)


def _events_ms(fn, repetitions: int) -> list[float]:
    result = []
    for _ in range(repetitions):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        result.append(start.elapsed_time(end))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--block-v", type=int, default=16, choices=(8, 16, 32, 64, 128))
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--update-count", type=int, default=0)
    parser.add_argument("--history-empty", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    T, H, R, K, V = args.tokens, 16, 4, 128, 128
    torch.manual_seed(20260715)
    q = torch.nn.functional.normalize(
        torch.randn(T, H, R, K, device=device), dim=-1
    ) * K ** -0.5
    k = torch.nn.functional.normalize(
        torch.randn(T, H, R, K, device=device), dim=-1
    )
    v = torch.randn(T, H, R, V, device=device) * 0.1
    erase = torch.rand(T, H, R, K, device=device)
    write = torch.rand(T, H, R, V, device=device)
    gamma = 0.99 + 0.009 * torch.rand(T, H, R, K, device=device)
    lam = 0.9 + 0.09 * torch.rand(T, H, R, device=device)
    state = torch.randn(H, R, K, V, device=device) * 0.01
    previous = torch.randn_like(state) * 0.01
    history = torch.full(
        (R,), not args.history_empty, dtype=torch.bool, device=device
    )
    periods = torch.tensor((1, 16, 64, 256), dtype=torch.int32, device=device)
    reads = torch.empty(T, H, R, R, V, device=device)
    state_out = torch.empty_like(state)
    previous_out = torch.empty_like(previous)
    partial = torch.empty(T, H, R, V // args.block_v, device=device)
    grid = (H * R * (V // args.block_v),)

    def triton_run():
        _four_state_segment_fwd[grid](
            q, k, v, erase, write, gamma, lam, state, previous,
            history, periods, reads, state_out, previous_out, partial,
            state_out, previous_out,
            update_count=args.update_count, T=T, H=H, R=R, K=K, V=V,
            BLOCK_V=args.block_v, WRITE_OUTPUTS=True, STORE_TRACE=False,
            num_warps=8,
        )
        return reads, state_out, previous_out, partial.sum(-1)

    print("compiling Triton kernel...")
    before = time.perf_counter()
    compiled = _four_state_segment_fwd.warmup(
        q, k, v, erase, write, gamma, lam, state, previous,
        history, periods, reads, state_out, previous_out, partial,
        state_out, previous_out,
        update_count=args.update_count, T=T, H=H, R=R, K=K, V=V,
        BLOCK_V=args.block_v, WRITE_OUTPUTS=True, STORE_TRACE=False,
        num_warps=8, grid=grid,
    )
    print(
        "kernel_resources="
        + repr({
            "instance_keys": sorted(vars(compiled)),
            "metadata": repr(getattr(compiled, "metadata", None)),
        })
    )
    triton_result = triton_run()
    torch.cuda.synchronize(device)
    print(f"compile_and_first_seconds={time.perf_counter() - before:.3f}")
    with torch.inference_mode():
        oracle = torch_oracle(
            q, k, v, erase, write, gamma, lam, state, previous,
            history, periods, torch.tensor(args.update_count, device=device),
        )
        torch.cuda.synchronize(device)
    for name, actual, expected in zip(
        ("reads", "state", "previous", "innovation_sq"), triton_result, oracle
    ):
        delta = (actual.float() - expected.float()).abs()
        print(
            f"{name}: max_abs={delta.max().item():.9g} "
            f"mean_abs={delta.mean().item():.9g} "
            f"exact_fraction={(actual == expected).float().mean().item():.6f}"
        )

    for _ in range(3):
        triton_run()
        with torch.inference_mode():
            torch_oracle(q, k, v, erase, write, gamma, lam, state, previous,
                         history, periods, torch.tensor(args.update_count, device=device))
    torch.cuda.synchronize(device)
    triton_times = _events_ms(triton_run, args.repetitions)
    with torch.inference_mode():
        torch_times = _events_ms(
            lambda: torch_oracle(
                q, k, v, erase, write, gamma, lam, state, previous,
                history, periods, torch.tensor(args.update_count, device=device),
            ),
            args.repetitions,
        )
    print(f"triton_ms={triton_times}")
    print(f"torch_ms={torch_times}")
    triton_median = statistics.median(triton_times)
    torch_median = statistics.median(torch_times)
    print(f"median_triton_ms={triton_median:.6f}")
    print(f"median_torch_ms={torch_median:.6f}")
    print(f"forward_speedup={torch_median / triton_median:.3f}")

    if args.forward_only:
        return
    if T > 64:
        raise ValueError("backward probe is intentionally limited to T<=64 scratch")

    tiles_v = V // args.block_v
    state_trace = torch.empty(T, H, R, K, V, device=device)
    previous_trace = torch.empty_like(state_trace)
    dreads = torch.randn_like(reads) * 0.01
    dstate_out = torch.randn_like(state) * 0.01
    dprevious_out = torch.randn_like(previous) * 0.01

    dq_partial = torch.empty(T, H, R, R, tiles_v, K, device=device)
    dk_partial = torch.empty(T, H, R, tiles_v, K, device=device)
    derase_partial = torch.empty_like(dk_partial)
    dgamma_partial = torch.empty_like(dk_partial)
    dlam_partial = torch.empty(T, H, R, tiles_v, device=device)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    derase = torch.empty_like(erase)
    dwrite = torch.empty_like(write)
    dgamma = torch.empty_like(gamma)
    dlam = torch.empty_like(lam)
    dstate = torch.empty_like(state)
    dprevious = torch.empty_like(previous)

    def trace_run():
        _four_state_segment_fwd[grid](
            q, k, v, erase, write, gamma, lam, state, previous,
            history, periods, reads, state_out, previous_out, partial,
            state_trace, previous_trace,
            update_count=args.update_count, T=T, H=H, R=R, K=K, V=V,
            BLOCK_V=args.block_v, WRITE_OUTPUTS=False, STORE_TRACE=True,
            num_warps=8,
        )

    def backward_run():
        trace_run()
        _four_state_segment_bwd[grid](
            q, k, v, erase, write, gamma, lam, state, previous,
            history, periods, state_trace, previous_trace,
            dreads, dstate_out, dprevious_out,
            dq_partial, dk_partial, derase_partial, dgamma_partial,
            dlam_partial, dv, dwrite, dstate, dprevious,
            update_count=args.update_count, T=T, H=H, R=R, K=K, V=V,
            BLOCK_V=args.block_v, num_warps=8,
        )
        reduction_grid = (T * H * R,)
        _reduce_q_partials[reduction_grid](
            dq_partial, dq, T=T, H=H, R=R, K=K, TILES_V=tiles_v,
            num_warps=4,
        )
        _reduce_lane_partials[reduction_grid](
            dk_partial, derase_partial, dgamma_partial, dlam_partial,
            dk, derase, dgamma, dlam,
            T=T, H=H, R=R, K=K, TILES_V=tiles_v, num_warps=4,
        )

    def custom_forward_backward():
        triton_run()
        backward_run()

    count = torch.tensor(args.update_count, device=device)

    def oracle_vjp():
        inputs = tuple(
            tensor.detach().requires_grad_(True)
            for tensor in (q, k, v, erase, write, gamma, lam, state, previous)
        )
        result = torch_oracle(
            *inputs[:7], inputs[7], inputs[8], history, periods, count,
        )
        loss = (
            (result[0] * dreads).sum()
            + (result[1] * dstate_out).sum()
            + (result[2] * dprevious_out).sum()
        )
        gradients = torch.autograd.grad(loss, inputs)
        return result, gradients

    print("compiling backward kernels...")
    before = time.perf_counter()
    backward_run()
    torch.cuda.synchronize(device)
    print(f"backward_compile_and_first_seconds={time.perf_counter() - before:.3f}")
    _, reference_gradients = oracle_vjp()
    torch.cuda.synchronize(device)
    custom_gradients = (
        dq, dk, dv, derase, dwrite, dgamma, dlam, dstate, dprevious,
    )
    gradient_names = (
        "dq", "dk", "dv", "derase", "dwrite", "dgamma", "dlambda",
        "dstate0", "dprevious0",
    )
    for name, actual, expected in zip(
        gradient_names, custom_gradients, reference_gradients, strict=True
    ):
        delta = (actual.float() - expected.float()).abs()
        scale = expected.float().abs().clamp_min(1e-12)
        print(
            f"{name}: max_abs={delta.max().item():.9g} "
            f"mean_abs={delta.mean().item():.9g} "
            f"max_rel={(delta / scale).max().item():.9g} "
            f"exact_fraction={(actual == expected).float().mean().item():.6f}"
        )

    scratch = (
        state_trace.numel() * state_trace.element_size()
        + previous_trace.numel() * previous_trace.element_size()
        + dq_partial.numel() * dq_partial.element_size()
        + 3 * dk_partial.numel() * dk_partial.element_size()
        + dlam_partial.numel() * dlam_partial.element_size()
    )
    print(f"backward_scratch_mib={scratch / 2**20:.3f}")
    for _ in range(2):
        custom_forward_backward()
        oracle_vjp()
    torch.cuda.synchronize(device)
    custom_times = _events_ms(custom_forward_backward, args.repetitions)
    torch_fb_times = _events_ms(oracle_vjp, args.repetitions)
    print(f"custom_forward_backward_ms={custom_times}")
    print(f"torch_forward_backward_ms={torch_fb_times}")
    custom_median = statistics.median(custom_times)
    torch_fb_median = statistics.median(torch_fb_times)
    print(f"median_custom_forward_backward_ms={custom_median:.6f}")
    print(f"median_torch_forward_backward_ms={torch_fb_median:.6f}")
    print(f"forward_backward_speedup={torch_fb_median / custom_median:.3f}")


if __name__ == "__main__":
    main()
