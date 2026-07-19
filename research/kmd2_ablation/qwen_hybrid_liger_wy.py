"""All-Triton generalized-WY factors for compact Package-B chunks.

This module contains only the chunk-parallel factor pipeline.  PyTorch owns
allocation and dispatch, while every recurrent mathematical operation is
performed by Triton kernels in FP32.  The token-loop PyTorch implementation
remains the external authority and is not called from this hot path.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

try:  # Keep CPU-only source inspection importable.
    import triton
    import triton.language as tl
except (ImportError, OSError, RuntimeError):  # pragma: no cover - CPU-only CI.
    triton = None
    tl = None


_R: Final = 4
_K: Final = 32
_V: Final = 128
_BLOCK_V: Final = 16
_SOLVE_BLOCK_V: Final = 4


if triton is not None:

    @triton.jit
    def _prefix_transport_fwd(
        gamma_ptr,
        prefix_ptr,
        transport_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        token = (row // (R * H)) % T
        batch = row // (R * H * T)
        offs_k = tl.arange(0, K)

        running = tl.full((K,), 1.0, tl.float32)
        for reverse in tl.range(0, T, num_stages=1):
            source = token - reverse
            valid = source >= 0
            safe_source = tl.maximum(source, 0)
            transport_offset = (
                (((((batch * T + token) * T + safe_source) * H + head)
                  * R + lane) * K)
                + offs_k
            )
            tl.store(transport_ptr + transport_offset, running, mask=valid)
            gamma_offset = (
                ((((batch * T + safe_source) * H + head) * R + lane) * K)
                + offs_k
            )
            gamma = tl.load(
                gamma_ptr + gamma_offset, mask=valid, other=1.0
            ).to(tl.float32)
            running *= gamma

        prefix_offset = (
            ((((batch * T + token) * H + head) * R + lane) * K)
            + offs_k
        )
        tl.store(prefix_ptr + prefix_offset, running)

        # The reconstruction kernel intentionally performs a fixed-length
        # source loop, so future-source coefficients must be explicit zeros.
        for source in tl.range(0, T, num_stages=1):
            future = source > token
            transport_offset = (
                (((((batch * T + token) * T + source) * H + head)
                  * R + lane) * K)
                + offs_k
            )
            tl.store(transport_ptr + transport_offset, 0.0, mask=future)


    @triton.jit
    def _preprocess_key_fwd(
        k_ptr,
        erase_ptr,
        lam_ptr,
        prefix_ptr,
        transport_ptr,
        previous_key_ptr,
        history_ptr,
        update_count_ptr,
        a_ptr,
        effective_lam_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        token = (row // (R * H)) % T
        batch = row // (R * H * T)
        offs_k = tl.arange(0, K)
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        remainder = update_count % period
        first_tick = tl.where(remainder == 0, 0, period - remainder)
        tick = ((update_count + token) % period) == 0
        internal_previous = tick & (token >= period)
        previous_token = tl.maximum(token - period, 0)

        key_offset = (
            ((((batch * T + token) * H + head) * R + lane) * K)
            + offs_k
        )
        previous_token_offset = (
            ((((batch * T + previous_token) * H + head) * R + lane) * K)
            + offs_k
        )
        previous_key_offset = (
            (((batch * H + head) * R + lane) * K) + offs_k
        )
        prefix_offset = key_offset
        transport_offset = (
            (((((batch * T + token) * T + previous_token) * H + head)
              * R + lane) * K)
            + offs_k
        )

        key = tl.load(k_ptr + key_offset).to(tl.float32)
        erased_key = (
            tl.load(erase_ptr + key_offset).to(tl.float32) * key
        )
        internal_key = tl.load(k_ptr + previous_token_offset).to(tl.float32)
        initial_key = tl.load(previous_key_ptr + previous_key_offset).to(
            tl.float32
        )
        internal_decay = tl.load(transport_ptr + transport_offset).to(
            tl.float32
        )
        initial_decay = tl.load(prefix_ptr + prefix_offset).to(tl.float32)
        previous_decayed = tl.where(
            internal_previous,
            internal_decay * internal_key,
            initial_decay * initial_key,
        )
        projection = tl.sum(erased_key * previous_decayed, axis=0)
        a = previous_decayed - key * projection
        tl.store(a_ptr + key_offset, tl.where(tick, a, 0.0))

        lam_offset = ((batch * T + token) * H + head) * R + lane
        lam = tl.load(lam_ptr + lam_offset).to(tl.float32)
        initial_history = tl.load(history_ptr + batch * R + lane)
        has_history = initial_history | (token > first_tick)
        tl.store(
            effective_lam_ptr + lam_offset,
            tl.where(has_history, lam, 1.0),
        )


    @triton.jit
    def _value_factors_fwd(
        v_ptr,
        write_ptr,
        effective_lam_ptr,
        previous_value_ptr,
        update_count_ptr,
        b_ptr,
        d_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        value_block = tl.program_id(1)
        lane = row % R
        head = (row // R) % H
        token = (row // (R * H)) % T
        batch = row // (R * H * T)
        offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        value_mask = offs_v < V
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        tick = ((update_count + token) % period) == 0
        internal_previous = tick & (token >= period)
        previous_token = tl.maximum(token - period, 0)

        value_offset = (
            ((((batch * T + token) * H + head) * R + lane) * V)
            + offs_v
        )
        previous_token_offset = (
            ((((batch * T + previous_token) * H + head) * R + lane) * V)
            + offs_v
        )
        previous_value_offset = (
            (((batch * H + head) * R + lane) * V) + offs_v
        )
        current = (
            tl.load(v_ptr + value_offset, mask=value_mask, other=0.0).to(
                tl.float32
            )
            * tl.load(
                write_ptr + value_offset, mask=value_mask, other=0.0
            ).to(tl.float32)
        )
        internal_value = (
            tl.load(
                v_ptr + previous_token_offset,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            * tl.load(
                write_ptr + previous_token_offset,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
        )
        initial_value = tl.load(
            previous_value_ptr + previous_value_offset,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        endpoint = tl.where(
            internal_previous, internal_value, initial_value
        )
        lam_offset = ((batch * T + token) * H + head) * R + lane
        lam = tl.load(effective_lam_ptr + lam_offset).to(tl.float32)
        b = tl.where(tick, (1.0 - lam) * endpoint, 0.0)
        d = tl.where(tick, lam * current, 0.0)
        tl.store(b_ptr + value_offset, b, mask=value_mask)
        tl.store(d_ptr + value_offset, d, mask=value_mask)


    @triton.jit
    def _coupling_fwd(
        k_ptr,
        erase_ptr,
        a_ptr,
        transport_ptr,
        update_count_ptr,
        system_ptr,
        coupling_p_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        source = (row // (R * H)) % T
        token = (row // (R * H * T)) % T
        batch = row // (R * H * T * T)
        offs_k = tl.arange(0, K)
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        tick_token = ((update_count + token) % period) == 0
        tick_source = ((update_count + source) % period) == 0
        active = (source < token) & tick_token & tick_source

        token_offset = (
            ((((batch * T + token) * H + head) * R + lane) * K)
            + offs_k
        )
        source_offset = (
            ((((batch * T + source) * H + head) * R + lane) * K)
            + offs_k
        )
        transport_offset = (
            (((((batch * T + token) * T + source) * H + head) * R
              + lane) * K)
            + offs_k
        )
        key_token = tl.load(k_ptr + token_offset).to(tl.float32)
        u_token = (
            tl.load(erase_ptr + token_offset).to(tl.float32) * key_token
        )
        transported = tl.load(transport_ptr + transport_offset).to(
            tl.float32
        )
        key_source = tl.load(k_ptr + source_offset).to(tl.float32)
        a_source = tl.load(a_ptr + source_offset).to(tl.float32)
        coupling_k = tl.sum(
            u_token * transported * key_source, axis=0
        )
        coupling_p = tl.sum(
            u_token * transported * a_source, axis=0
        )
        matrix_offset = (
            ((((batch * H + head) * R + lane) * T + token) * T)
            + source
        )
        system = tl.where(token == source, 1.0, 0.0)
        system = tl.where(active, coupling_k, system)
        tl.store(system_ptr + matrix_offset, system)
        tl.store(
            coupling_p_ptr + matrix_offset,
            tl.where(active, coupling_p, 0.0),
        )


    @triton.jit
    def _h_term_fwd(
        k_ptr,
        erase_ptr,
        prefix_ptr,
        state_ptr,
        update_count_ptr,
        h_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        value_block = tl.program_id(1)
        lane = row % R
        head = (row // R) % H
        token = (row // (R * H)) % T
        batch = row // (R * H * T)
        offs_k = tl.arange(0, K)
        offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        value_mask = offs_v < V
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        tick = ((update_count + token) % period) == 0
        key_offset = (
            ((((batch * T + token) * H + head) * R + lane) * K)
            + offs_k
        )
        state_offset = (
            ((((batch * H + head) * R + lane) * K + offs_k[:, None])
              * V)
            + offs_v[None, :]
        )
        key = tl.load(k_ptr + key_offset).to(tl.float32)
        u = tl.load(erase_ptr + key_offset).to(tl.float32) * key
        prefix = tl.load(prefix_ptr + key_offset).to(tl.float32)
        state = tl.load(
            state_ptr + state_offset,
            mask=value_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        result = tl.sum((u * prefix)[:, None] * state, axis=0)
        value_offset = (
            ((((batch * T + token) * H + head) * R + lane) * V)
            + offs_v
        )
        tl.store(h_ptr + value_offset, tl.where(tick, result, 0.0),
                 mask=value_mask)


    @triton.jit
    def _solve_fwd(
        system_ptr,
        coupling_p_ptr,
        h_ptr,
        b_ptr,
        d_ptr,
        rho_ptr,
        y_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        V: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        value_block = tl.program_id(1)
        lane = row % R
        head = (row // R) % H
        batch = row // (R * H)
        offs_t = tl.arange(0, BLOCK_T)
        offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        token_mask = offs_t < T
        value_mask = offs_v < V
        matrix_mask = token_mask[:, None] & value_mask[None, :]
        value_offset = (
            ((((batch * T + offs_t[:, None]) * H + head) * R + lane) * V)
            + offs_v[None, :]
        )
        rhs = tl.load(
            h_ptr + value_offset, mask=matrix_mask, other=0.0
        ).to(tl.float32)

        for source in tl.static_range(0, T):
            source_value_offset = (
                ((((batch * T + source) * H + head) * R + lane) * V)
                + offs_v
            )
            d_source = tl.load(
                d_ptr + source_value_offset,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            b_source = tl.load(
                b_ptr + source_value_offset,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            factor_offset = (
                ((((batch * H + head) * R + lane) * T + offs_t) * T)
                + source
            )
            system_column = tl.load(
                system_ptr + factor_offset, mask=token_mask, other=0.0
            ).to(tl.float32)
            p_column = tl.load(
                coupling_p_ptr + factor_offset,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            lower = offs_t > source
            rhs += tl.where(
                lower[:, None],
                system_column[:, None] * d_source[None, :]
                + p_column[:, None] * b_source[None, :],
                0.0,
            )

        rho = rhs
        for source in tl.static_range(0, T):
            pivot = tl.sum(
                tl.where((offs_t == source)[:, None], rho, 0.0), axis=0
            )
            factor_offset = (
                ((((batch * H + head) * R + lane) * T + offs_t) * T)
                + source
            )
            system_column = tl.load(
                system_ptr + factor_offset, mask=token_mask, other=0.0
            ).to(tl.float32)
            rho = tl.where(
                (offs_t > source)[:, None],
                rho - system_column[:, None] * pivot[None, :],
                rho,
            )

        d_values = tl.load(
            d_ptr + value_offset, mask=matrix_mask, other=0.0
        ).to(tl.float32)
        tl.store(rho_ptr + value_offset, rho, mask=matrix_mask)
        tl.store(y_ptr + value_offset, d_values - rho, mask=matrix_mask)


    @triton.jit
    def _innovation_fwd(
        k_ptr,
        a_ptr,
        b_ptr,
        y_ptr,
        update_count_ptr,
        innovation_ptr,
        innovation_sq_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        FAST_ONLY: tl.constexpr,
    ):
        row = tl.program_id(0)
        key_coordinate = row % K
        if FAST_ONLY:
            lane = 0
            head = (row // K) % H
            token = (row // (K * H)) % T
            batch = row // (K * H * T)
        else:
            lane = (row // K) % R
            head = (row // (K * R)) % H
            token = (row // (K * R * H)) % T
            batch = row // (K * R * H * T)
        offs_v = tl.arange(0, V)
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        tick = ((update_count + token) % period) == 0
        key_offset = (
            ((((batch * T + token) * H + head) * R + lane) * K)
            + key_coordinate
        )
        value_offset = (
            ((((batch * T + token) * H + head) * R + lane) * V)
            + offs_v
        )
        innovation_offset = key_offset * V + offs_v
        key = tl.load(k_ptr + key_offset).to(tl.float32)
        a = tl.load(a_ptr + key_offset).to(tl.float32)
        b = tl.load(b_ptr + value_offset).to(tl.float32)
        y = tl.load(y_ptr + value_offset).to(tl.float32)
        innovation = tl.where(tick, key * y + a * b, 0.0)
        tl.store(innovation_ptr + innovation_offset, innovation)
        square = tl.sum(innovation * innovation, axis=0)
        score_offset = ((batch * T + token) * H + head) * R + lane
        tl.atomic_add(innovation_sq_ptr + score_offset, square)


    @triton.jit
    def _innovation_fwd_sparse_cms(
        k_ptr,
        a_ptr,
        b_ptr,
        y_ptr,
        update_count_ptr,
        innovation_ptr,
        innovation_sq_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
    ):
        row = tl.program_id(0)
        key_coordinate = row % K
        slot = (row // K) % 6
        head = (row // (K * 6)) % H
        batch = row // (K * 6 * H)
        lane = tl.where(slot < 4, 1, slot - 2)
        ordinal = tl.where(slot < 4, slot, 0)
        period = tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)).to(
            tl.int64
        )
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        remainder = update_count % period
        first_tick = tl.where(remainder == 0, 0, period - remainder)
        token = first_tick + ordinal * period
        active = token < T
        safe_token = tl.maximum(tl.minimum(token, T - 1), 0)
        offs_v = tl.arange(0, V)
        key_offset = (
            ((((batch * T + safe_token) * H + head) * R + lane) * K)
            + key_coordinate
        )
        value_offset = (
            ((((batch * T + safe_token) * H + head) * R + lane) * V)
            + offs_v
        )
        innovation_offset = key_offset * V + offs_v
        key = tl.load(k_ptr + key_offset).to(tl.float32)
        a = tl.load(a_ptr + key_offset).to(tl.float32)
        b = tl.load(b_ptr + value_offset).to(tl.float32)
        y = tl.load(y_ptr + value_offset).to(tl.float32)
        innovation = key * y + a * b
        tl.store(
            innovation_ptr + innovation_offset,
            innovation,
            mask=active,
        )
        square = tl.sum(innovation * innovation, axis=0)
        score_offset = ((batch * T + safe_token) * H + head) * R + lane
        tl.atomic_add(
            innovation_sq_ptr + score_offset,
            tl.where(active, square, 0.0),
        )


    @triton.jit
    def _endpoint_key_fwd(
        k_ptr,
        prefix_ptr,
        transport_ptr,
        previous_key_ptr,
        update_count_ptr,
        previous_key_out_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        batch = row // (R * H)
        offs_k = tl.arange(0, K)
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        remainder = update_count % period
        first_tick = tl.where(remainder == 0, 0, period - remainder)
        has_tick = first_tick < T
        last_tick = first_tick + ((T - 1 - first_tick) // period) * period
        safe_last = tl.maximum(tl.minimum(last_tick, T - 1), 0)
        key_offset = (
            ((((batch * T + safe_last) * H + head) * R + lane) * K)
            + offs_k
        )
        previous_offset = (
            (((batch * H + head) * R + lane) * K) + offs_k
        )
        prefix_offset = (
            ((((batch * T + T - 1) * H + head) * R + lane) * K)
            + offs_k
        )
        transport_offset = (
            (((((batch * T + T - 1) * T + safe_last) * H + head)
              * R + lane) * K)
            + offs_k
        )
        source_key = tl.where(
            has_tick,
            tl.load(k_ptr + key_offset).to(tl.float32),
            tl.load(previous_key_ptr + previous_offset).to(tl.float32),
        )
        decay = tl.where(
            has_tick,
            tl.load(transport_ptr + transport_offset).to(tl.float32),
            tl.load(prefix_ptr + prefix_offset).to(tl.float32),
        )
        tl.store(previous_key_out_ptr + previous_offset, decay * source_key)


    @triton.jit
    def _endpoint_value_fwd(
        v_ptr,
        write_ptr,
        previous_value_ptr,
        update_count_ptr,
        previous_value_out_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        V: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        batch = row // (R * H)
        offs_v = tl.arange(0, V)
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        remainder = update_count % period
        first_tick = tl.where(remainder == 0, 0, period - remainder)
        has_tick = first_tick < T
        last_tick = first_tick + ((T - 1 - first_tick) // period) * period
        safe_last = tl.maximum(tl.minimum(last_tick, T - 1), 0)
        value_offset = (
            ((((batch * T + safe_last) * H + head) * R + lane) * V)
            + offs_v
        )
        previous_offset = (
            (((batch * H + head) * R + lane) * V) + offs_v
        )
        current = (
            tl.load(v_ptr + value_offset).to(tl.float32)
            * tl.load(write_ptr + value_offset).to(tl.float32)
        )
        initial = tl.load(previous_value_ptr + previous_offset).to(tl.float32)
        tl.store(
            previous_value_out_ptr + previous_offset,
            tl.where(has_tick, current, initial),
        )


    @triton.jit
    def _metadata_fwd(
        history_ptr,
        update_count_ptr,
        history_out_ptr,
        update_count_out_ptr,
        T: tl.constexpr,
        R: tl.constexpr,
    ):
        batch = tl.program_id(0)
        lanes = tl.arange(0, R)
        periods = tl.where(
            lanes == 0,
            1,
            tl.where(lanes == 1, 16, tl.where(lanes == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        remainder = update_count % periods
        first_tick = tl.where(remainder == 0, 0, periods - remainder)
        history = tl.load(history_ptr + batch * R + lanes)
        tl.store(
            history_out_ptr + batch * R + lanes,
            history | (first_tick < T),
        )
        tl.store(update_count_out_ptr + batch, update_count + T)


    @triton.jit
    def _reconstruction_factor_bwd(
        prefix_ptr,
        transport_ptr,
        innovation_ptr,
        state_ptr,
        grad_trace_ptr,
        grad_prefix_ptr,
        grad_transport_ptr,
        grad_innovation_ptr,
        grad_state_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        value_block = tl.program_id(1)
        lane = row % R
        head = (row // R) % H
        token = (row // (R * H)) % T
        batch = row // (R * H * T)
        offs_k = tl.arange(0, K)
        offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        value_mask = offs_v < V
        matrix_mask = value_mask[None, :]
        trace_offset = (
            (((((batch * T + token) * H + head) * R + lane) * K
              + offs_k[:, None]) * V)
            + offs_v[None, :]
        )
        state_offset = (
            ((((batch * H + head) * R + lane) * K + offs_k[:, None])
              * V)
            + offs_v[None, :]
        )
        factor_offset = (
            ((((batch * T + token) * H + head) * R + lane) * K)
            + offs_k
        )
        grad_state_token = tl.load(
            grad_trace_ptr + trace_offset,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        initial_state = tl.load(
            state_ptr + state_offset,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        prefix = tl.load(prefix_ptr + factor_offset).to(tl.float32)
        tl.atomic_add(
            grad_prefix_ptr + factor_offset,
            tl.sum(grad_state_token * initial_state, axis=1),
        )
        tl.atomic_add(
            grad_state_ptr + state_offset,
            prefix[:, None] * grad_state_token,
            mask=matrix_mask,
        )

        for source in tl.range(0, T, num_stages=1):
            transport_offset = (
                (((((batch * T + token) * T + source) * H + head)
                  * R + lane) * K)
                + offs_k
            )
            innovation_offset = (
                (((((batch * T + source) * H + head) * R + lane) * K
                  + offs_k[:, None]) * V)
                + offs_v[None, :]
            )
            transport = tl.load(transport_ptr + transport_offset).to(
                tl.float32
            )
            innovation = tl.load(
                innovation_ptr + innovation_offset,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            tl.atomic_add(
                grad_transport_ptr + transport_offset,
                tl.sum(grad_state_token * innovation, axis=1),
            )
            tl.atomic_add(
                grad_innovation_ptr + innovation_offset,
                transport[:, None] * grad_state_token,
                mask=matrix_mask,
            )


    @triton.jit
    def _reconstruction_prefix_bwd(
        state_ptr,
        grad_trace_ptr,
        grad_prefix_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        token = (row // (R * H)) % T
        batch = row // (R * H * T)
        offs_k = tl.arange(0, K)
        result = tl.zeros((K,), dtype=tl.float32)
        for value_block in tl.static_range(0, V // BLOCK_V):
            offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
            trace_offset = (
                (((((batch * T + token) * H + head) * R + lane) * K
                  + offs_k[:, None]) * V)
                + offs_v[None, :]
            )
            state_offset = (
                ((((batch * H + head) * R + lane) * K
                  + offs_k[:, None]) * V)
                + offs_v[None, :]
            )
            grad_state_token = tl.load(
                grad_trace_ptr + trace_offset
            ).to(tl.float32)
            initial_state = tl.load(state_ptr + state_offset).to(tl.float32)
            result += tl.sum(grad_state_token * initial_state, axis=1)
        factor_offset = (
            ((((batch * T + token) * H + head) * R + lane) * K)
            + offs_k
        )
        tl.store(grad_prefix_ptr + factor_offset, result)


    @triton.jit
    def _reconstruction_state_bwd(
        prefix_ptr,
        grad_trace_ptr,
        grad_state_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        value_block = tl.program_id(1)
        lane = row % R
        head = (row // R) % H
        batch = row // (R * H)
        offs_k = tl.arange(0, K)
        offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        result = tl.zeros((K, BLOCK_V), dtype=tl.float32)
        for token in tl.static_range(0, T):
            factor_offset = (
                ((((batch * T + token) * H + head) * R + lane) * K)
                + offs_k
            )
            trace_offset = (
                (((((batch * T + token) * H + head) * R + lane) * K
                  + offs_k[:, None]) * V)
                + offs_v[None, :]
            )
            prefix = tl.load(prefix_ptr + factor_offset).to(tl.float32)
            grad_state_token = tl.load(
                grad_trace_ptr + trace_offset
            ).to(tl.float32)
            result += prefix[:, None] * grad_state_token
        state_offset = (
            ((((batch * H + head) * R + lane) * K + offs_k[:, None]) * V)
            + offs_v[None, :]
        )
        tl.store(grad_state_ptr + state_offset, result)


    @triton.jit
    def _reconstruction_transport_bwd(
        innovation_ptr,
        grad_trace_ptr,
        grad_transport_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        source = (row // (R * H)) % T
        token = (row // (R * H * T)) % T
        batch = row // (R * H * T * T)
        offs_k = tl.arange(0, K)
        result = tl.zeros((K,), dtype=tl.float32)
        for value_block in tl.static_range(0, V // BLOCK_V):
            offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
            trace_offset = (
                (((((batch * T + token) * H + head) * R + lane) * K
                  + offs_k[:, None]) * V)
                + offs_v[None, :]
            )
            innovation_offset = (
                (((((batch * T + source) * H + head) * R + lane) * K
                  + offs_k[:, None]) * V)
                + offs_v[None, :]
            )
            grad_state_token = tl.load(
                grad_trace_ptr + trace_offset
            ).to(tl.float32)
            innovation = tl.load(
                innovation_ptr + innovation_offset
            ).to(tl.float32)
            result += tl.sum(grad_state_token * innovation, axis=1)
        transport_offset = (
            (((((batch * T + token) * T + source) * H + head)
              * R + lane) * K)
            + offs_k
        )
        tl.store(
            grad_transport_ptr + transport_offset,
            tl.where(source <= token, result, 0.0),
        )


    @triton.jit
    def _reconstruction_transport_bwd_tiled(
        innovation_ptr,
        grad_trace_ptr,
        grad_transport_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_SOURCE: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        lane = row % R
        head = (row // R) % H
        token = (row // (R * H)) % T
        batch = row // (R * H * T)
        source_blocks = tl.cdiv(T, BLOCK_SOURCE)
        key_block = tile // source_blocks
        source_block = tile % source_blocks
        offs_s = source_block * BLOCK_SOURCE + tl.arange(0, BLOCK_SOURCE)
        offs_k = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        source_mask = offs_s < T
        key_mask = offs_k < K
        result = tl.zeros((BLOCK_SOURCE, BLOCK_K), dtype=tl.float32)
        for value_block in tl.static_range(0, V // BLOCK_V):
            offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
            trace_offset = (
                (((((batch * T + token) * H + head) * R + lane) * K
                  + offs_k[:, None]) * V)
                + offs_v[None, :]
            )
            innovation_offset = (
                (((((batch * T + offs_s[:, None, None]) * H + head) * R
                   + lane) * K + offs_k[None, :, None]) * V)
                + offs_v[None, None, :]
            )
            grad_state_token = tl.load(
                grad_trace_ptr + trace_offset,
                mask=key_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            innovation = tl.load(
                innovation_ptr + innovation_offset,
                mask=source_mask[:, None, None] & key_mask[None, :, None],
                other=0.0,
            ).to(tl.float32)
            result += tl.sum(
                innovation * grad_state_token[None, :, :], axis=2
            )
        transport_offset = (
            (((((batch * T + token) * T + offs_s[:, None]) * H + head)
              * R + lane) * K)
            + offs_k[None, :]
        )
        active = (
            source_mask[:, None]
            & key_mask[None, :]
            & (offs_s[:, None] <= token)
        )
        tl.store(
            grad_transport_ptr + transport_offset,
            result,
            mask=active,
        )


    @triton.jit
    def _reconstruction_transport_bwd_gemm(
        innovation_ptr,
        grad_trace_ptr,
        grad_transport_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_SOURCE: tl.constexpr,
        BLOCK_V: tl.constexpr,
        FAST_ONLY: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        key_coordinate = row % K
        if FAST_ONLY:
            lane = 0
            head = (row // K) % H
            batch = row // (K * H)
        else:
            lane = (row // K) % R
            head = (row // (K * R)) % H
            batch = row // (K * R * H)
        source_blocks = tl.cdiv(T, BLOCK_SOURCE)
        token_block = tile // source_blocks
        source_block = tile % source_blocks
        offs_t = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_s = source_block * BLOCK_SOURCE + tl.arange(0, BLOCK_SOURCE)
        accumulator = tl.zeros(
            (BLOCK_T, BLOCK_SOURCE), dtype=tl.float32
        )
        for value_block in tl.static_range(0, V // BLOCK_V):
            offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
            trace_offset = (
                (((((batch * T + offs_t[:, None]) * H + head) * R + lane)
                  * K + key_coordinate) * V)
                + offs_v[None, :]
            )
            innovation_offset = (
                (((((batch * T + offs_s[:, None]) * H + head) * R + lane)
                  * K + key_coordinate) * V)
                + offs_v[None, :]
            )
            grad_state_token = tl.load(
                grad_trace_ptr + trace_offset,
                mask=(offs_t < T)[:, None],
                other=0.0,
            ).to(tl.float32)
            innovation = tl.load(
                innovation_ptr + innovation_offset,
                mask=(offs_s < T)[:, None],
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.dot(
                grad_state_token,
                tl.trans(innovation),
                input_precision="ieee",
            )
        transport_offset = (
            (((((batch * T + offs_t[:, None]) * T + offs_s[None, :])
               * H + head) * R + lane) * K)
            + key_coordinate
        )
        mask = (
            (offs_t < T)[:, None]
            & (offs_s < T)[None, :]
            & (offs_s[None, :] <= offs_t[:, None])
        )
        tl.store(grad_transport_ptr + transport_offset, accumulator, mask=mask)


    @triton.jit
    def _reconstruction_transport_bwd_sparse_cms(
        innovation_ptr,
        grad_trace_ptr,
        update_count_ptr,
        grad_transport_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Transport VJP only for the six possible slow-lane ticks.

        In a 64-token WY tile, lane 1 has at most four ticks and lanes 2/3
        have at most one each.  Mapping programs to those tick ordinals avoids
        treating their explicit zero innovation rows as dense sources.
        """
        row = tl.program_id(0)
        slot = row % 6
        token = (row // 6) % T
        head = (row // (6 * T)) % H
        batch = row // (6 * T * H)
        lane = tl.where(slot < 4, 1, slot - 2)
        ordinal = tl.where(slot < 4, slot, 0)
        period = tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)).to(
            tl.int64
        )
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        remainder = update_count % period
        first_tick = tl.where(remainder == 0, 0, period - remainder)
        source = first_tick + ordinal * period
        active = (source < T) & (source <= token)
        safe_source = tl.maximum(tl.minimum(source, T - 1), 0)
        offs_k = tl.arange(0, K)
        result = tl.zeros((K,), dtype=tl.float32)
        if active:
            for value_block in tl.static_range(0, V // BLOCK_V):
                offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
                trace_offset = (
                    (((((batch * T + token) * H + head) * R + lane) * K
                      + offs_k[:, None]) * V)
                    + offs_v[None, :]
                )
                innovation_offset = (
                    (((((batch * T + safe_source) * H + head) * R + lane) * K
                      + offs_k[:, None]) * V)
                    + offs_v[None, :]
                )
                grad_state_token = tl.load(
                    grad_trace_ptr + trace_offset
                ).to(tl.float32)
                innovation = tl.load(
                    innovation_ptr + innovation_offset
                ).to(tl.float32)
                result += tl.sum(grad_state_token * innovation, axis=1)
            transport_offset = (
                (((((batch * T + token) * T + safe_source) * H + head)
                  * R + lane) * K)
                + offs_k
            )
            tl.store(grad_transport_ptr + transport_offset, result)


    @triton.jit
    def _reconstruction_innovation_bwd(
        transport_ptr,
        grad_trace_ptr,
        grad_innovation_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        value_block = tl.program_id(1)
        lane = row % R
        head = (row // R) % H
        source = (row // (R * H)) % T
        batch = row // (R * H * T)
        offs_k = tl.arange(0, K)
        offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        result = tl.zeros((K, BLOCK_V), dtype=tl.float32)
        for token in tl.static_range(0, T):
            active = token >= source
            transport_offset = (
                (((((batch * T + token) * T + source) * H + head)
                  * R + lane) * K)
                + offs_k
            )
            trace_offset = (
                (((((batch * T + token) * H + head) * R + lane) * K
                  + offs_k[:, None]) * V)
                + offs_v[None, :]
            )
            transport = tl.load(transport_ptr + transport_offset).to(
                tl.float32
            )
            grad_state_token = tl.load(
                grad_trace_ptr + trace_offset
            ).to(tl.float32)
            result += tl.where(
                active, transport[:, None] * grad_state_token, 0.0
            )
        innovation_offset = (
            (((((batch * T + source) * H + head) * R + lane) * K
              + offs_k[:, None]) * V)
            + offs_v[None, :]
        )
        tl.store(grad_innovation_ptr + innovation_offset, result)


    @triton.jit
    def _reconstruction_innovation_bwd_gemm(
        transport_ptr,
        grad_trace_ptr,
        grad_innovation_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_SOURCE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_V: tl.constexpr,
        FAST_ONLY: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        key_coordinate = row % K
        if FAST_ONLY:
            lane = 0
            head = (row // K) % H
            batch = row // (K * H)
        else:
            lane = (row // K) % R
            head = (row // (K * R)) % H
            batch = row // (K * R * H)
        value_blocks = tl.cdiv(V, BLOCK_V)
        source_block = tile // value_blocks
        value_block = tile % value_blocks
        offs_s = source_block * BLOCK_SOURCE + tl.arange(0, BLOCK_SOURCE)
        offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        accumulator = tl.zeros((BLOCK_SOURCE, BLOCK_V), dtype=tl.float32)
        for token_block in tl.static_range(0, tl.cdiv(T, BLOCK_T)):
            offs_t = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
            transport_offset = (
                (((((batch * T + offs_t[:, None]) * T + offs_s[None, :])
                   * H + head) * R + lane) * K)
                + key_coordinate
            )
            trace_offset = (
                (((((batch * T + offs_t[:, None]) * H + head) * R + lane)
                  * K + key_coordinate) * V)
                + offs_v[None, :]
            )
            transport = tl.load(
                transport_ptr + transport_offset,
                mask=(offs_t < T)[:, None] & (offs_s < T)[None, :],
                other=0.0,
            ).to(tl.float32)
            grad_state_token = tl.load(
                grad_trace_ptr + trace_offset,
                mask=(offs_t < T)[:, None] & (offs_v < V)[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.dot(
                tl.trans(transport),
                grad_state_token,
                input_precision="ieee",
            )
        innovation_offset = (
            (((((batch * T + offs_s[:, None]) * H + head) * R + lane)
              * K + key_coordinate) * V)
            + offs_v[None, :]
        )
        tl.store(
            grad_innovation_ptr + innovation_offset,
            accumulator,
            mask=(offs_s < T)[:, None] & (offs_v < V)[None, :],
        )


    @triton.jit
    def _reconstruction_innovation_bwd_sparse_cms(
        transport_ptr,
        grad_trace_ptr,
        update_count_ptr,
        grad_innovation_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        value_block = tl.program_id(1)
        slot = row % 6
        head = (row // 6) % H
        batch = row // (6 * H)
        lane = tl.where(slot < 4, 1, slot - 2)
        ordinal = tl.where(slot < 4, slot, 0)
        period = tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)).to(
            tl.int64
        )
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        remainder = update_count % period
        first_tick = tl.where(remainder == 0, 0, period - remainder)
        source = first_tick + ordinal * period
        active_source = source < T
        safe_source = tl.maximum(tl.minimum(source, T - 1), 0)
        offs_k = tl.arange(0, K)
        offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        value_mask = offs_v < V
        result = tl.zeros((K, BLOCK_V), dtype=tl.float32)
        if active_source:
            for token in tl.static_range(0, T):
                active = token >= safe_source
                transport_offset = (
                    (((((batch * T + token) * T + safe_source) * H + head)
                      * R + lane) * K)
                    + offs_k
                )
                trace_offset = (
                    (((((batch * T + token) * H + head) * R + lane) * K
                      + offs_k[:, None]) * V)
                    + offs_v[None, :]
                )
                transport = tl.load(transport_ptr + transport_offset).to(
                    tl.float32
                )
                grad_state_token = tl.load(
                    grad_trace_ptr + trace_offset,
                    mask=value_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                result += tl.where(
                    active,
                    transport[:, None] * grad_state_token,
                    0.0,
                )
            innovation_offset = (
                (((((batch * T + safe_source) * H + head) * R + lane) * K
                  + offs_k[:, None]) * V)
                + offs_v[None, :]
            )
            tl.store(
                grad_innovation_ptr + innovation_offset,
                result,
                mask=value_mask[None, :],
            )


    @triton.jit
    def _innovation_bwd_k(
        grad_innovation_ptr,
        y_ptr,
        b_ptr,
        update_count_ptr,
        grad_k_ptr,
        grad_a_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        FAST_ONLY: tl.constexpr,
    ):
        row = tl.program_id(0)
        key_coordinate = row % K
        if FAST_ONLY:
            lane = 0
            head = (row // K) % H
            token = (row // (K * H)) % T
            batch = row // (K * H * T)
        else:
            lane = (row // K) % R
            head = (row // (K * R)) % H
            token = (row // (K * R * H)) % T
            batch = row // (K * R * H * T)
        offs_v = tl.arange(0, V)
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        tick = ((update_count + token) % period) == 0
        key_offset = (
            ((((batch * T + token) * H + head) * R + lane) * K)
            + key_coordinate
        )
        value_offset = (
            ((((batch * T + token) * H + head) * R + lane) * V)
            + offs_v
        )
        grad_innovation = tl.where(tick, tl.load(
            grad_innovation_ptr + key_offset * V + offs_v
        ).to(tl.float32), 0.0)
        y = tl.load(y_ptr + value_offset).to(tl.float32)
        b = tl.load(b_ptr + value_offset).to(tl.float32)
        tl.atomic_add(
            grad_k_ptr + key_offset,
            tl.sum(grad_innovation * y, axis=0),
        )
        tl.store(
            grad_a_ptr + key_offset,
            tl.sum(grad_innovation * b, axis=0),
        )


    @triton.jit
    def _innovation_bwd_v(
        grad_innovation_ptr,
        k_ptr,
        a_ptr,
        update_count_ptr,
        grad_rho_ptr,
        grad_d_ptr,
        grad_b_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        FAST_ONLY: tl.constexpr,
    ):
        row = tl.program_id(0)
        value_coordinate = row % V
        if FAST_ONLY:
            lane = 0
            head = (row // V) % H
            token = (row // (V * H)) % T
            batch = row // (V * H * T)
        else:
            lane = (row // V) % R
            head = (row // (V * R)) % H
            token = (row // (V * R * H)) % T
            batch = row // (V * R * H * T)
        offs_k = tl.arange(0, K)
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        tick = ((update_count + token) % period) == 0
        key_offset = (
            ((((batch * T + token) * H + head) * R + lane) * K)
            + offs_k
        )
        value_offset = (
            ((((batch * T + token) * H + head) * R + lane) * V)
            + value_coordinate
        )
        innovation_offset = key_offset * V + value_coordinate
        grad_innovation = tl.where(tick, tl.load(
            grad_innovation_ptr + innovation_offset
        ).to(tl.float32), 0.0)
        key = tl.load(k_ptr + key_offset).to(tl.float32)
        a = tl.load(a_ptr + key_offset).to(tl.float32)
        grad_y = tl.sum(grad_innovation * key, axis=0)
        grad_b = tl.sum(grad_innovation * a, axis=0)
        tl.store(grad_rho_ptr + value_offset, -grad_y)
        tl.store(grad_d_ptr + value_offset, grad_y)
        tl.store(grad_b_ptr + value_offset, grad_b)


    @triton.jit
    def _innovation_bwd_k_sparse_cms(
        grad_innovation_ptr,
        y_ptr,
        b_ptr,
        update_count_ptr,
        grad_k_ptr,
        grad_a_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
    ):
        row = tl.program_id(0)
        key_coordinate = row % K
        slot = (row // K) % 6
        head = (row // (K * 6)) % H
        batch = row // (K * 6 * H)
        lane = tl.where(slot < 4, 1, slot - 2)
        ordinal = tl.where(slot < 4, slot, 0)
        period = tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)).to(
            tl.int64
        )
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        remainder = update_count % period
        first_tick = tl.where(remainder == 0, 0, period - remainder)
        token = first_tick + ordinal * period
        active = token < T
        safe_token = tl.maximum(tl.minimum(token, T - 1), 0)
        offs_v = tl.arange(0, V)
        key_offset = (
            ((((batch * T + safe_token) * H + head) * R + lane) * K)
            + key_coordinate
        )
        value_offset = (
            ((((batch * T + safe_token) * H + head) * R + lane) * V)
            + offs_v
        )
        grad_innovation = tl.load(
            grad_innovation_ptr + key_offset * V + offs_v
        ).to(tl.float32)
        y = tl.load(y_ptr + value_offset).to(tl.float32)
        b = tl.load(b_ptr + value_offset).to(tl.float32)
        tl.atomic_add(
            grad_k_ptr + key_offset,
            tl.where(active, tl.sum(grad_innovation * y, axis=0), 0.0),
        )
        tl.store(
            grad_a_ptr + key_offset,
            tl.sum(grad_innovation * b, axis=0),
            mask=active,
        )


    @triton.jit
    def _innovation_bwd_v_sparse_cms(
        grad_innovation_ptr,
        k_ptr,
        a_ptr,
        update_count_ptr,
        grad_rho_ptr,
        grad_d_ptr,
        grad_b_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
    ):
        row = tl.program_id(0)
        value_coordinate = row % V
        slot = (row // V) % 6
        head = (row // (V * 6)) % H
        batch = row // (V * 6 * H)
        lane = tl.where(slot < 4, 1, slot - 2)
        ordinal = tl.where(slot < 4, slot, 0)
        period = tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)).to(
            tl.int64
        )
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        remainder = update_count % period
        first_tick = tl.where(remainder == 0, 0, period - remainder)
        token = first_tick + ordinal * period
        active = token < T
        safe_token = tl.maximum(tl.minimum(token, T - 1), 0)
        offs_k = tl.arange(0, K)
        key_offset = (
            ((((batch * T + safe_token) * H + head) * R + lane) * K)
            + offs_k
        )
        value_offset = (
            ((((batch * T + safe_token) * H + head) * R + lane) * V)
            + value_coordinate
        )
        innovation_offset = key_offset * V + value_coordinate
        grad_innovation = tl.load(
            grad_innovation_ptr + innovation_offset
        ).to(tl.float32)
        key = tl.load(k_ptr + key_offset).to(tl.float32)
        a = tl.load(a_ptr + key_offset).to(tl.float32)
        grad_y = tl.sum(grad_innovation * key, axis=0)
        grad_b = tl.sum(grad_innovation * a, axis=0)
        tl.store(grad_rho_ptr + value_offset, -grad_y, mask=active)
        tl.store(grad_d_ptr + value_offset, grad_y, mask=active)
        tl.store(grad_b_ptr + value_offset, grad_b, mask=active)


    @triton.jit
    def _solve_bwd(
        system_ptr,
        grad_rho_ptr,
        grad_rhs_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        V: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        value_block = tl.program_id(1)
        lane = row % R
        head = (row // R) % H
        batch = row // (R * H)
        offs_t = tl.arange(0, BLOCK_T)
        offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        token_mask = offs_t < T
        value_mask = offs_v < V
        matrix_mask = token_mask[:, None] & value_mask[None, :]
        value_offset = (
            ((((batch * T + offs_t[:, None]) * H + head) * R + lane) * V)
            + offs_v[None, :]
        )
        grad_rhs = tl.load(
            grad_rho_ptr + value_offset, mask=matrix_mask, other=0.0
        ).to(tl.float32)
        for reverse in tl.static_range(0, T):
            source = T - 1 - reverse
            pivot = tl.sum(
                tl.where((offs_t == source)[:, None], grad_rhs, 0.0),
                axis=0,
            )
            factor_offset = (
                ((((batch * H + head) * R + lane) * T + source) * T)
                + offs_t
            )
            system_row = tl.load(
                system_ptr + factor_offset, mask=token_mask, other=0.0
            ).to(tl.float32)
            grad_rhs = tl.where(
                (offs_t < source)[:, None],
                grad_rhs - system_row[:, None] * pivot[None, :],
                grad_rhs,
            )
        tl.store(grad_rhs_ptr + value_offset, grad_rhs, mask=matrix_mask)


    @triton.jit
    def _rhs_matrix_bwd(
        grad_rhs_ptr,
        rho_ptr,
        b_ptr,
        d_ptr,
        grad_system_ptr,
        grad_coupling_p_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        V: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        source = (row // (R * H)) % T
        token = (row // (R * H * T)) % T
        batch = row // (R * H * T * T)
        offs_v = tl.arange(0, V)
        token_offset = (
            ((((batch * T + token) * H + head) * R + lane) * V)
            + offs_v
        )
        source_offset = (
            ((((batch * T + source) * H + head) * R + lane) * V)
            + offs_v
        )
        grad_rhs = tl.load(grad_rhs_ptr + token_offset).to(tl.float32)
        rho_source = tl.load(rho_ptr + source_offset).to(tl.float32)
        d_source = tl.load(d_ptr + source_offset).to(tl.float32)
        b_source = tl.load(b_ptr + source_offset).to(tl.float32)
        active = source < token
        grad_system = tl.sum(
            grad_rhs * (d_source - rho_source), axis=0
        )
        grad_p = tl.sum(grad_rhs * b_source, axis=0)
        matrix_offset = (
            ((((batch * H + head) * R + lane) * T + token) * T)
            + source
        )
        tl.store(
            grad_system_ptr + matrix_offset,
            tl.where(active, grad_system, 0.0),
        )
        tl.store(
            grad_coupling_p_ptr + matrix_offset,
            tl.where(active, grad_p, 0.0),
        )


    @triton.jit
    def _rhs_value_bwd(
        system_ptr,
        coupling_p_ptr,
        grad_rhs_ptr,
        grad_b_ptr,
        grad_d_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        value_block = tl.program_id(1)
        lane = row % R
        head = (row // R) % H
        source = (row // (R * H)) % T
        batch = row // (R * H * T)
        offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        value_mask = offs_v < V
        source_offset = (
            ((((batch * T + source) * H + head) * R + lane) * V)
            + offs_v
        )
        grad_b = tl.load(
            grad_b_ptr + source_offset, mask=value_mask, other=0.0
        ).to(tl.float32)
        # y=d-rho: grad_d_ptr is initialized from d(y), while -d(y) was
        # separately passed through the transposed solve as d(rho).
        grad_d = tl.load(
            grad_d_ptr + source_offset, mask=value_mask, other=0.0
        ).to(tl.float32)
        for token in tl.static_range(0, T):
            active = token > source
            token_offset = (
                ((((batch * T + token) * H + head) * R + lane) * V)
                + offs_v
            )
            grad_rhs = tl.load(
                grad_rhs_ptr + token_offset,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            matrix_offset = (
                ((((batch * H + head) * R + lane) * T + token) * T)
                + source
            )
            system = tl.load(system_ptr + matrix_offset).to(tl.float32)
            coupling_p = tl.load(
                coupling_p_ptr + matrix_offset
            ).to(tl.float32)
            grad_d += tl.where(active, system * grad_rhs, 0.0)
            grad_b += tl.where(active, coupling_p * grad_rhs, 0.0)
        tl.store(grad_d_ptr + source_offset, grad_d, mask=value_mask)
        tl.store(grad_b_ptr + source_offset, grad_b, mask=value_mask)


    @triton.jit
    def _h_bwd_key(
        k_ptr,
        erase_ptr,
        prefix_ptr,
        state_ptr,
        grad_h_ptr,
        grad_u_ptr,
        grad_prefix_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
    ):
        row = tl.program_id(0)
        key_coordinate = row % K
        lane = (row // K) % R
        head = (row // (K * R)) % H
        token = (row // (K * R * H)) % T
        batch = row // (K * R * H * T)
        offs_v = tl.arange(0, V)
        key_offset = (
            ((((batch * T + token) * H + head) * R + lane) * K)
            + key_coordinate
        )
        state_offset = (
            ((((batch * H + head) * R + lane) * K + key_coordinate) * V)
            + offs_v
        )
        value_offset = (
            ((((batch * T + token) * H + head) * R + lane) * V)
            + offs_v
        )
        key = tl.load(k_ptr + key_offset).to(tl.float32)
        u = tl.load(erase_ptr + key_offset).to(tl.float32) * key
        prefix = tl.load(prefix_ptr + key_offset).to(tl.float32)
        state = tl.load(state_ptr + state_offset).to(tl.float32)
        grad_h = tl.load(grad_h_ptr + value_offset).to(tl.float32)
        contraction = tl.sum(grad_h * state, axis=0)
        tl.atomic_add(grad_u_ptr + key_offset, prefix * contraction)
        tl.atomic_add(grad_prefix_ptr + key_offset, u * contraction)


    @triton.jit
    def _h_bwd_state(
        k_ptr,
        erase_ptr,
        prefix_ptr,
        grad_h_ptr,
        grad_state_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        value_block = tl.program_id(1)
        key_coordinate = row % K
        lane = (row // K) % R
        head = (row // (K * R)) % H
        batch = row // (K * R * H)
        offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        value_mask = offs_v < V
        result = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for token in tl.static_range(0, T):
            key_offset = (
                ((((batch * T + token) * H + head) * R + lane) * K)
                + key_coordinate
            )
            value_offset = (
                ((((batch * T + token) * H + head) * R + lane) * V)
                + offs_v
            )
            key = tl.load(k_ptr + key_offset).to(tl.float32)
            u = tl.load(erase_ptr + key_offset).to(tl.float32) * key
            prefix = tl.load(prefix_ptr + key_offset).to(tl.float32)
            grad_h = tl.load(
                grad_h_ptr + value_offset, mask=value_mask, other=0.0
            ).to(tl.float32)
            result += u * prefix * grad_h
        state_offset = (
            ((((batch * H + head) * R + lane) * K + key_coordinate) * V)
            + offs_v
        )
        tl.atomic_add(
            grad_state_ptr + state_offset, result, mask=value_mask
        )


    @triton.jit
    def _coupling_bwd(
        k_ptr,
        erase_ptr,
        a_ptr,
        transport_ptr,
        update_count_ptr,
        grad_system_ptr,
        grad_coupling_p_ptr,
        grad_u_ptr,
        grad_k_ptr,
        grad_a_ptr,
        grad_transport_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        source = (row // (R * H)) % T
        token = (row // (R * H * T)) % T
        batch = row // (R * H * T * T)
        offs_k = tl.arange(0, K)
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        active = (
            (source < token)
            & (((update_count + token) % period) == 0)
            & (((update_count + source) % period) == 0)
        )
        token_offset = (
            ((((batch * T + token) * H + head) * R + lane) * K)
            + offs_k
        )
        source_offset = (
            ((((batch * T + source) * H + head) * R + lane) * K)
            + offs_k
        )
        transport_offset = (
            (((((batch * T + token) * T + source) * H + head)
              * R + lane) * K)
            + offs_k
        )
        matrix_offset = (
            ((((batch * H + head) * R + lane) * T + token) * T)
            + source
        )
        key_token = tl.load(k_ptr + token_offset).to(tl.float32)
        u_token = (
            tl.load(erase_ptr + token_offset).to(tl.float32) * key_token
        )
        key_source = tl.load(k_ptr + source_offset).to(tl.float32)
        a_source = tl.load(a_ptr + source_offset).to(tl.float32)
        transport = tl.load(transport_ptr + transport_offset).to(tl.float32)
        grad_system = tl.load(grad_system_ptr + matrix_offset).to(tl.float32)
        grad_p = tl.load(grad_coupling_p_ptr + matrix_offset).to(tl.float32)
        grad_system = tl.where(active, grad_system, 0.0)
        grad_p = tl.where(active, grad_p, 0.0)
        tl.atomic_add(
            grad_u_ptr + token_offset,
            transport
            * (grad_system * key_source + grad_p * a_source),
        )
        tl.atomic_add(
            grad_k_ptr + source_offset,
            grad_system * u_token * transport,
        )
        tl.atomic_add(
            grad_a_ptr + source_offset,
            grad_p * u_token * transport,
        )
        tl.atomic_add(
            grad_transport_ptr + transport_offset,
            u_token
            * (grad_system * key_source + grad_p * a_source),
        )


    @triton.jit
    def _value_factors_bwd(
        v_ptr,
        write_ptr,
        effective_lam_ptr,
        previous_value_ptr,
        history_ptr,
        update_count_ptr,
        grad_b_ptr,
        grad_d_ptr,
        grad_current_ptr,
        grad_lam_ptr,
        grad_previous_value_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        V: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        token = (row // (R * H)) % T
        batch = row // (R * H * T)
        offs_v = tl.arange(0, V)
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        remainder = update_count % period
        first_tick = tl.where(remainder == 0, 0, period - remainder)
        tick = ((update_count + token) % period) == 0
        internal_previous = tick & (token >= period)
        previous_token = tl.maximum(token - period, 0)
        value_offset = (
            ((((batch * T + token) * H + head) * R + lane) * V)
            + offs_v
        )
        previous_token_offset = (
            ((((batch * T + previous_token) * H + head) * R + lane) * V)
            + offs_v
        )
        previous_value_offset = (
            (((batch * H + head) * R + lane) * V) + offs_v
        )
        current = (
            tl.load(v_ptr + value_offset).to(tl.float32)
            * tl.load(write_ptr + value_offset).to(tl.float32)
        )
        internal_value = (
            tl.load(v_ptr + previous_token_offset).to(tl.float32)
            * tl.load(write_ptr + previous_token_offset).to(tl.float32)
        )
        initial_value = tl.load(
            previous_value_ptr + previous_value_offset
        ).to(tl.float32)
        endpoint = tl.where(
            internal_previous, internal_value, initial_value
        )
        grad_b = tl.load(grad_b_ptr + value_offset).to(tl.float32)
        grad_d = tl.load(grad_d_ptr + value_offset).to(tl.float32)
        lam_offset = ((batch * T + token) * H + head) * R + lane
        lam = tl.load(effective_lam_ptr + lam_offset).to(tl.float32)
        grad_current = tl.where(tick, lam * grad_d, 0.0)
        grad_endpoint = tl.where(tick, (1.0 - lam) * grad_b, 0.0)
        tl.atomic_add(grad_current_ptr + value_offset, grad_current)
        tl.atomic_add(
            grad_current_ptr + previous_token_offset,
            tl.where(internal_previous, grad_endpoint, 0.0),
        )
        tl.atomic_add(
            grad_previous_value_ptr + previous_value_offset,
            tl.where(tick & ~internal_previous, grad_endpoint, 0.0),
        )
        has_history = (
            tl.load(history_ptr + batch * R + lane) | (token > first_tick)
        )
        grad_lam = tl.sum(
            grad_d * current - grad_b * endpoint, axis=0
        )
        tl.store(
            grad_lam_ptr + lam_offset,
            tl.where(tick & has_history, grad_lam, 0.0),
        )


    @triton.jit
    def _current_value_bwd(
        v_ptr,
        write_ptr,
        grad_current_ptr,
        grad_v_ptr,
        grad_write_ptr,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        value = tl.load(v_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        write = tl.load(write_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_current = tl.load(
            grad_current_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        tl.store(grad_v_ptr + offsets, grad_current * write, mask=mask)
        tl.store(grad_write_ptr + offsets, grad_current * value, mask=mask)


    @triton.jit
    def _a_bwd(
        k_ptr,
        erase_ptr,
        prefix_ptr,
        transport_ptr,
        previous_key_ptr,
        update_count_ptr,
        grad_a_ptr,
        grad_k_ptr,
        grad_u_ptr,
        grad_prefix_ptr,
        grad_transport_ptr,
        grad_previous_key_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        token = (row // (R * H)) % T
        batch = row // (R * H * T)
        offs_k = tl.arange(0, K)
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        tick = ((update_count + token) % period) == 0
        internal_previous = tick & (token >= period)
        previous_token = tl.maximum(token - period, 0)
        key_offset = (
            ((((batch * T + token) * H + head) * R + lane) * K)
            + offs_k
        )
        previous_token_offset = (
            ((((batch * T + previous_token) * H + head) * R + lane) * K)
            + offs_k
        )
        previous_key_offset = (
            (((batch * H + head) * R + lane) * K) + offs_k
        )
        transport_offset = (
            (((((batch * T + token) * T + previous_token) * H + head)
              * R + lane) * K)
            + offs_k
        )
        key = tl.load(k_ptr + key_offset).to(tl.float32)
        u = tl.load(erase_ptr + key_offset).to(tl.float32) * key
        previous_token_key = tl.load(k_ptr + previous_token_offset).to(
            tl.float32
        )
        initial_key = tl.load(previous_key_ptr + previous_key_offset).to(
            tl.float32
        )
        internal_decay = tl.load(transport_ptr + transport_offset).to(
            tl.float32
        )
        initial_decay = tl.load(prefix_ptr + key_offset).to(tl.float32)
        source_key = tl.where(
            internal_previous, previous_token_key, initial_key
        )
        decay = tl.where(
            internal_previous, internal_decay, initial_decay
        )
        previous_decayed = decay * source_key
        projection = tl.sum(u * previous_decayed, axis=0)
        grad_a = tl.where(
            tick, tl.load(grad_a_ptr + key_offset).to(tl.float32), 0.0
        )
        key_dot = tl.sum(key * grad_a, axis=0)
        grad_previous_decayed = grad_a - key_dot * u
        tl.atomic_add(grad_k_ptr + key_offset, -projection * grad_a)
        tl.atomic_add(
            grad_u_ptr + key_offset, -key_dot * previous_decayed
        )
        tl.atomic_add(
            grad_k_ptr + previous_token_offset,
            tl.where(
                internal_previous,
                grad_previous_decayed * internal_decay,
                0.0,
            ),
        )
        tl.atomic_add(
            grad_transport_ptr + transport_offset,
            tl.where(
                internal_previous,
                grad_previous_decayed * previous_token_key,
                0.0,
            ),
        )
        tl.atomic_add(
            grad_previous_key_ptr + previous_key_offset,
            tl.where(
                tick & ~internal_previous,
                grad_previous_decayed * initial_decay,
                0.0,
            ),
        )
        tl.atomic_add(
            grad_prefix_ptr + key_offset,
            tl.where(
                tick & ~internal_previous,
                grad_previous_decayed * initial_key,
                0.0,
            ),
        )


    @triton.jit
    def _endpoint_key_bwd(
        k_ptr,
        prefix_ptr,
        transport_ptr,
        previous_key_ptr,
        update_count_ptr,
        grad_previous_key_out_ptr,
        grad_k_ptr,
        grad_prefix_ptr,
        grad_transport_ptr,
        grad_previous_key_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        batch = row // (R * H)
        offs_k = tl.arange(0, K)
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        remainder = update_count % period
        first_tick = tl.where(remainder == 0, 0, period - remainder)
        has_tick = first_tick < T
        last_tick = first_tick + ((T - 1 - first_tick) // period) * period
        safe_last = tl.maximum(tl.minimum(last_tick, T - 1), 0)
        key_offset = (
            ((((batch * T + safe_last) * H + head) * R + lane) * K)
            + offs_k
        )
        previous_offset = (
            (((batch * H + head) * R + lane) * K) + offs_k
        )
        prefix_offset = (
            ((((batch * T + T - 1) * H + head) * R + lane) * K)
            + offs_k
        )
        transport_offset = (
            (((((batch * T + T - 1) * T + safe_last) * H + head)
              * R + lane) * K)
            + offs_k
        )
        grad_output = tl.load(
            grad_previous_key_out_ptr + previous_offset
        ).to(tl.float32)
        source_key = tl.where(
            has_tick,
            tl.load(k_ptr + key_offset).to(tl.float32),
            tl.load(previous_key_ptr + previous_offset).to(tl.float32),
        )
        decay = tl.where(
            has_tick,
            tl.load(transport_ptr + transport_offset).to(tl.float32),
            tl.load(prefix_ptr + prefix_offset).to(tl.float32),
        )
        tl.atomic_add(
            grad_k_ptr + key_offset,
            tl.where(has_tick, grad_output * decay, 0.0),
        )
        tl.atomic_add(
            grad_transport_ptr + transport_offset,
            tl.where(has_tick, grad_output * source_key, 0.0),
        )
        tl.atomic_add(
            grad_previous_key_ptr + previous_offset,
            tl.where(~has_tick, grad_output * decay, 0.0),
        )
        tl.atomic_add(
            grad_prefix_ptr + prefix_offset,
            tl.where(~has_tick, grad_output * source_key, 0.0),
        )


    @triton.jit
    def _endpoint_value_bwd(
        update_count_ptr,
        grad_previous_value_out_ptr,
        grad_current_ptr,
        grad_previous_value_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        V: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        batch = row // (R * H)
        offs_v = tl.arange(0, V)
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        update_count = tl.load(update_count_ptr + batch).to(tl.int64)
        remainder = update_count % period
        first_tick = tl.where(remainder == 0, 0, period - remainder)
        has_tick = first_tick < T
        last_tick = first_tick + ((T - 1 - first_tick) // period) * period
        safe_last = tl.maximum(tl.minimum(last_tick, T - 1), 0)
        value_offset = (
            ((((batch * T + safe_last) * H + head) * R + lane) * V)
            + offs_v
        )
        previous_offset = (
            (((batch * H + head) * R + lane) * V) + offs_v
        )
        grad_output = tl.load(
            grad_previous_value_out_ptr + previous_offset
        ).to(tl.float32)
        tl.atomic_add(
            grad_current_ptr + value_offset,
            tl.where(has_tick, grad_output, 0.0),
        )
        tl.atomic_add(
            grad_previous_value_ptr + previous_offset,
            tl.where(~has_tick, grad_output, 0.0),
        )


    @triton.jit
    def _u_bwd(
        k_ptr,
        erase_ptr,
        grad_u_ptr,
        grad_k_ptr,
        grad_erase_ptr,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        key = tl.load(k_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        erase = tl.load(
            erase_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        grad_u = tl.load(
            grad_u_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        tl.atomic_add(grad_k_ptr + offsets, grad_u * erase, mask=mask)
        tl.store(grad_erase_ptr + offsets, grad_u * key, mask=mask)


    @triton.jit
    def _gamma_bwd(
        gamma_ptr,
        transport_ptr,
        grad_prefix_ptr,
        grad_transport_ptr,
        grad_gamma_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        token = (row // (R * H)) % T
        batch = row // (R * H * T)
        offs_k = tl.arange(0, K)
        prefix_offset = (
            ((((batch * T + token) * H + head) * R + lane) * K)
            + offs_k
        )
        grad_running = tl.load(grad_prefix_ptr + prefix_offset).to(
            tl.float32
        )
        for source in tl.static_range(0, T):
            active = source <= token
            gamma_offset = (
                ((((batch * T + source) * H + head) * R + lane) * K)
                + offs_k
            )
            transport_offset = (
                (((((batch * T + token) * T + source) * H + head)
                  * R + lane) * K)
                + offs_k
            )
            gamma = tl.load(gamma_ptr + gamma_offset).to(tl.float32)
            old_running = tl.load(transport_ptr + transport_offset).to(
                tl.float32
            )
            grad_transport = tl.load(
                grad_transport_ptr + transport_offset
            ).to(tl.float32)
            tl.atomic_add(
                grad_gamma_ptr + gamma_offset,
                tl.where(active, grad_running * old_running, 0.0),
            )
            grad_running = tl.where(
                active,
                grad_transport + grad_running * gamma,
                grad_running,
            )


def all_triton_wy_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def build_wy_factors(
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
) -> tuple[Tensor, ...]:
    """Build exact generalized-WY factors entirely with Triton kernels."""
    if not all_triton_wy_available():
        raise RuntimeError("Triton/CUDA is unavailable")
    B, T, H, R, K = k.shape
    V = v.shape[-1]
    if R != _R or K != _K or V != _V:
        raise ValueError("all-Triton WY requires R=4, K=32, and V=128")

    k_c = k.contiguous()
    v_c = v.contiguous()
    erase_c = erase.contiguous()
    write_c = write.contiguous()
    gamma_c = gamma.contiguous()
    lam_c = lam.contiguous()
    state_c = state.contiguous()
    previous_key_c = previous_key.contiguous()
    previous_value_c = previous_value.contiguous()
    history_c = history.contiguous()
    update_count_c = update_count.contiguous()

    prefix = torch.empty_like(gamma_c)
    transport = torch.empty(
        (B, T, T, H, R, K), device=k.device, dtype=torch.float32
    )
    a = torch.empty_like(k_c)
    effective_lam = torch.empty_like(lam_c)
    b = torch.empty_like(v_c)
    d = torch.empty_like(v_c)
    system = torch.empty(
        (B, H, R, T, T), device=k.device, dtype=torch.float32
    )
    coupling_p = torch.empty_like(system)
    h_term = torch.empty_like(v_c)
    rho = torch.empty_like(v_c)
    y = torch.empty_like(v_c)
    innovation = torch.zeros(
        (B, T, H, R, K, V), device=k.device, dtype=torch.float32
    )
    innovation_sq = torch.zeros(
        (B, T, H, R), device=k.device, dtype=torch.float32
    )
    previous_key_out = torch.empty_like(previous_key_c)
    previous_value_out = torch.empty_like(previous_value_c)
    history_out = torch.empty_like(history_c)
    update_count_out = torch.empty_like(update_count_c)

    with torch.cuda.device(k.device):
        _prefix_transport_fwd[(B * T * H * R,)](
            gamma_c,
            prefix,
            transport,
            T=T,
            H=H,
            R=R,
            K=K,
            num_warps=4,
        )
        _preprocess_key_fwd[(B * T * H * R,)](
            k_c,
            erase_c,
            lam_c,
            prefix,
            transport,
            previous_key_c,
            history_c,
            update_count_c,
            a,
            effective_lam,
            T=T,
            H=H,
            R=R,
            K=K,
            num_warps=4,
        )
        _value_factors_fwd[
            (B * T * H * R, triton.cdiv(V, _BLOCK_V))
        ](
            v_c,
            write_c,
            effective_lam,
            previous_value_c,
            update_count_c,
            b,
            d,
            T=T,
            H=H,
            R=R,
            V=V,
            BLOCK_V=_BLOCK_V,
            num_warps=4,
        )
        _coupling_fwd[(B * T * T * H * R,)](
            k_c,
            erase_c,
            a,
            transport,
            update_count_c,
            system,
            coupling_p,
            T=T,
            H=H,
            R=R,
            K=K,
            num_warps=4,
        )
        _h_term_fwd[
            (B * T * H * R, triton.cdiv(V, _BLOCK_V))
        ](
            k_c,
            erase_c,
            prefix,
            state_c,
            update_count_c,
            h_term,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            BLOCK_V=_BLOCK_V,
            num_warps=4,
        )
        block_t = triton.next_power_of_2(T)
        _solve_fwd[(B * H * R, triton.cdiv(V, _SOLVE_BLOCK_V))](
            system,
            coupling_p,
            h_term,
            b,
            d,
            rho,
            y,
            T=T,
            H=H,
            R=R,
            V=V,
            BLOCK_T=block_t,
            BLOCK_V=_SOLVE_BLOCK_V,
            num_warps=4,
        )
        _innovation_fwd[(B * T * H * K,)](
            k_c,
            a,
            b,
            y,
            update_count_c,
            innovation,
            innovation_sq,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            FAST_ONLY=True,
            num_warps=4,
        )
        _innovation_fwd_sparse_cms[(B * H * 6 * K,)](
            k_c,
            a,
            b,
            y,
            update_count_c,
            innovation,
            innovation_sq,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            num_warps=4,
        )
        _endpoint_key_fwd[(B * H * R,)](
            k_c,
            prefix,
            transport,
            previous_key_c,
            update_count_c,
            previous_key_out,
            T=T,
            H=H,
            R=R,
            K=K,
            num_warps=4,
        )
        _endpoint_value_fwd[(B * H * R,)](
            v_c,
            write_c,
            previous_value_c,
            update_count_c,
            previous_value_out,
            T=T,
            H=H,
            R=R,
            V=V,
            num_warps=4,
        )
        _metadata_fwd[(B,)](
            history_c,
            update_count_c,
            history_out,
            update_count_out,
            T=T,
            R=R,
            num_warps=1,
        )

    return (
        prefix,
        transport,
        innovation,
        previous_key_out,
        previous_value_out,
        innovation_sq,
        history_out,
        update_count_out,
        a,
        effective_lam,
        b,
        d,
        system,
        coupling_p,
        h_term,
        rho,
        y,
    )


def reconstruction_factor_grads(
    prefix: Tensor,
    transport: Tensor,
    innovation: Tensor,
    state: Tensor,
    grad_trace: Tensor,
    update_count: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Differentiate WY reconstruction without eager PyTorch contractions."""
    B, T, H, R, K = prefix.shape
    V = state.shape[-1]
    grad_prefix = torch.zeros_like(prefix)
    grad_transport = torch.zeros_like(transport)
    grad_innovation = torch.zeros_like(innovation)
    grad_state = torch.zeros_like(state)
    with torch.cuda.device(prefix.device):
        prefix_c = prefix.contiguous()
        transport_c = transport.contiguous()
        innovation_c = innovation.contiguous()
        state_c = state.contiguous()
        grad_trace_c = grad_trace.contiguous()
        update_count_c = update_count.contiguous()
        _reconstruction_prefix_bwd[(B * T * H * R,)](
            state_c,
            grad_trace_c,
            grad_prefix,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            BLOCK_V=_BLOCK_V,
            num_warps=4,
        )
        _reconstruction_state_bwd[
            (B * H * R, triton.cdiv(V, _BLOCK_V))
        ](
            prefix_c,
            grad_trace_c,
            grad_state,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            BLOCK_V=_BLOCK_V,
            num_warps=4,
        )
        block_token = 16
        block_source = 16
        block_value = 16
        _reconstruction_transport_bwd_gemm[
            (
                B * H * K,
                triton.cdiv(T, block_token)
                * triton.cdiv(T, block_source),
            )
        ](
            innovation_c,
            grad_trace_c,
            grad_transport,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            BLOCK_T=block_token,
            BLOCK_SOURCE=block_source,
            BLOCK_V=block_value,
            FAST_ONLY=True,
            num_warps=4,
        )
        _reconstruction_transport_bwd_sparse_cms[(B * T * H * 6,)](
            innovation_c,
            grad_trace_c,
            update_count_c,
            grad_transport,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            BLOCK_V=_BLOCK_V,
            num_warps=4,
        )
        _reconstruction_innovation_bwd_gemm[
            (
                B * H * K,
                triton.cdiv(T, block_source)
                * triton.cdiv(V, block_value),
            )
        ](
            transport_c,
            grad_trace_c,
            grad_innovation,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            BLOCK_SOURCE=block_source,
            BLOCK_T=block_token,
            BLOCK_V=block_value,
            FAST_ONLY=True,
            num_warps=4,
        )
        _reconstruction_innovation_bwd_sparse_cms[
            (B * H * 6, triton.cdiv(V, _BLOCK_V))
        ](
            transport_c,
            grad_trace_c,
            update_count_c,
            grad_innovation,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            BLOCK_V=_BLOCK_V,
            num_warps=4,
        )
    return grad_prefix, grad_transport, grad_innovation, grad_state


def backward_wy_factors(
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
    factors: tuple[Tensor, ...],
    grad_prefix: Tensor,
    grad_transport: Tensor,
    grad_innovation: Tensor,
    direct_grad_state: Tensor,
    grad_previous_key_out: Tensor | None,
    grad_previous_value_out: Tensor | None,
) -> tuple[Tensor, ...]:
    """Analytical Triton VJP of the complete generalized-WY factor graph."""
    (
        prefix,
        transport,
        _innovation,
        _previous_key_out,
        _previous_value_out,
        _innovation_sq,
        _history_out,
        _update_count_out,
        a,
        effective_lam,
        b,
        d,
        system,
        coupling_p,
        _h_term,
        rho,
        y,
    ) = factors
    B, T, H, R, K = k.shape
    V = v.shape[-1]
    k_c = k.contiguous()
    v_c = v.contiguous()
    erase_c = erase.contiguous()
    write_c = write.contiguous()
    gamma_c = gamma.contiguous()
    state_c = state.contiguous()
    previous_key_c = previous_key.contiguous()
    previous_value_c = previous_value.contiguous()
    history_c = history.contiguous()
    update_count_c = update_count.contiguous()

    grad_k = torch.zeros_like(k_c)
    grad_v = torch.empty_like(v_c)
    grad_erase = torch.empty_like(erase_c)
    grad_write = torch.empty_like(write_c)
    grad_gamma = torch.zeros_like(gamma_c)
    grad_lam = torch.empty_like(lam.contiguous())
    grad_state = direct_grad_state
    grad_previous_key = torch.zeros_like(previous_key_c)
    grad_previous_value = torch.zeros_like(previous_value_c)
    grad_u = torch.zeros_like(k_c)
    grad_a = torch.zeros_like(a)
    grad_rho = torch.zeros_like(v_c)
    grad_d = torch.zeros_like(v_c)
    grad_b = torch.zeros_like(v_c)
    grad_rhs = torch.empty_like(v_c)
    grad_system = torch.empty_like(system)
    grad_coupling_p = torch.empty_like(coupling_p)
    grad_current = torch.zeros_like(v_c)
    grad_previous_key_out_c = (
        torch.zeros_like(previous_key_c)
        if grad_previous_key_out is None
        else grad_previous_key_out.contiguous()
    )
    grad_previous_value_out_c = (
        torch.zeros_like(previous_value_c)
        if grad_previous_value_out is None
        else grad_previous_value_out.contiguous()
    )

    with torch.cuda.device(k.device):
        _innovation_bwd_k[(B * T * H * K,)](
            grad_innovation,
            y,
            b,
            update_count_c,
            grad_k,
            grad_a,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            FAST_ONLY=True,
            num_warps=4,
        )
        _innovation_bwd_k_sparse_cms[(B * H * 6 * K,)](
            grad_innovation,
            y,
            b,
            update_count_c,
            grad_k,
            grad_a,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            num_warps=4,
        )
        _innovation_bwd_v[(B * T * H * V,)](
            grad_innovation,
            k_c,
            a,
            update_count_c,
            grad_rho,
            grad_d,
            grad_b,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            FAST_ONLY=True,
            num_warps=4,
        )
        _innovation_bwd_v_sparse_cms[(B * H * 6 * V,)](
            grad_innovation,
            k_c,
            a,
            update_count_c,
            grad_rho,
            grad_d,
            grad_b,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            num_warps=4,
        )
        block_t = triton.next_power_of_2(T)
        _solve_bwd[(B * H * R, triton.cdiv(V, _SOLVE_BLOCK_V))](
            system,
            grad_rho,
            grad_rhs,
            T=T,
            H=H,
            R=R,
            V=V,
            BLOCK_T=block_t,
            BLOCK_V=_SOLVE_BLOCK_V,
            num_warps=4,
        )
        _rhs_matrix_bwd[(B * T * T * H * R,)](
            grad_rhs,
            rho,
            b,
            d,
            grad_system,
            grad_coupling_p,
            T=T,
            H=H,
            R=R,
            V=V,
            num_warps=4,
        )
        _rhs_value_bwd[
            (B * T * H * R, triton.cdiv(V, _BLOCK_V))
        ](
            system,
            coupling_p,
            grad_rhs,
            grad_b,
            grad_d,
            T=T,
            H=H,
            R=R,
            V=V,
            BLOCK_V=_BLOCK_V,
            num_warps=4,
        )
        _h_bwd_key[(B * T * H * R * K,)](
            k_c,
            erase_c,
            prefix,
            state_c,
            grad_rhs,
            grad_u,
            grad_prefix,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            num_warps=4,
        )
        _h_bwd_state[
            (B * H * R * K, triton.cdiv(V, _BLOCK_V))
        ](
            k_c,
            erase_c,
            prefix,
            grad_rhs,
            grad_state,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            BLOCK_V=_BLOCK_V,
            num_warps=4,
        )
        _coupling_bwd[(B * T * T * H * R,)](
            k_c,
            erase_c,
            a,
            transport,
            update_count_c,
            grad_system,
            grad_coupling_p,
            grad_u,
            grad_k,
            grad_a,
            grad_transport,
            T=T,
            H=H,
            R=R,
            K=K,
            num_warps=4,
        )
        _value_factors_bwd[(B * T * H * R,)](
            v_c,
            write_c,
            effective_lam,
            previous_value_c,
            history_c,
            update_count_c,
            grad_b,
            grad_d,
            grad_current,
            grad_lam,
            grad_previous_value,
            T=T,
            H=H,
            R=R,
            V=V,
            num_warps=4,
        )
        _a_bwd[(B * T * H * R,)](
            k_c,
            erase_c,
            prefix,
            transport,
            previous_key_c,
            update_count_c,
            grad_a,
            grad_k,
            grad_u,
            grad_prefix,
            grad_transport,
            grad_previous_key,
            T=T,
            H=H,
            R=R,
            K=K,
            num_warps=4,
        )
        _endpoint_key_bwd[(B * H * R,)](
            k_c,
            prefix,
            transport,
            previous_key_c,
            update_count_c,
            grad_previous_key_out_c,
            grad_k,
            grad_prefix,
            grad_transport,
            grad_previous_key,
            T=T,
            H=H,
            R=R,
            K=K,
            num_warps=4,
        )
        _endpoint_value_bwd[(B * H * R,)](
            update_count_c,
            grad_previous_value_out_c,
            grad_current,
            grad_previous_value,
            T=T,
            H=H,
            R=R,
            V=V,
            num_warps=4,
        )
        n_values = v_c.numel()
        value_block = 256
        _current_value_bwd[(triton.cdiv(n_values, value_block),)](
            v_c,
            write_c,
            grad_current,
            grad_v,
            grad_write,
            N=n_values,
            BLOCK=value_block,
            num_warps=4,
        )
        n_keys = k_c.numel()
        key_block = 256
        _u_bwd[(triton.cdiv(n_keys, key_block),)](
            k_c,
            erase_c,
            grad_u,
            grad_k,
            grad_erase,
            N=n_keys,
            BLOCK=key_block,
            num_warps=4,
        )
        _gamma_bwd[(B * T * H * R,)](
            gamma_c,
            transport,
            grad_prefix,
            grad_transport,
            grad_gamma,
            T=T,
            H=H,
            R=R,
            K=K,
            num_warps=4,
        )

    return (
        grad_k,
        grad_v,
        grad_erase,
        grad_write,
        grad_gamma,
        grad_lam,
        grad_state,
        grad_previous_key,
        grad_previous_value,
    )


__all__ = [
    "all_triton_wy_available",
    "backward_wy_factors",
    "build_wy_factors",
    "reconstruction_factor_grads",
]
