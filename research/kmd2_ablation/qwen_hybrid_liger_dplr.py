"""True chunked DPLR training backend for the Package-B recurrence.

One Package-B update is represented by two generalized DPLR substeps:

``Gamma S - k (erase*k)^T Gamma S + k d^T`` followed by ``a b^T``.

This is an algebraic re-expression of the PyTorch authority, not a model
approximation.  FLA's chunked DPLR kernels provide the two-phase WY scan and
analytical backward.  A small local Triton custom op prepares the trapezoidal
endpoint factors and differentiates them without a tokenwise PyTorch graph.
All FP32 dot products are forced to IEEE precision before any FLA kernel can
compile; TF32 is outside Package B's recurrence contract.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

# Triton 3.3 reads this contract from the environment, while newer releases
# additionally expose a mutable knob.  Establish it before importing Triton so
# neither eager nor lazy FLA compilation can silently select TF32.
os.environ["TRITON_F32_DEFAULT"] = "ieee"

try:
    import triton
    import triton.language as tl
except (ImportError, OSError, RuntimeError):  # pragma: no cover - CPU import
    triton = None
    tl = None

try:
    from fla.ops.generalized_delta_rule.dplr.chunk_A_bwd import (
        chunk_dplr_bwd_dqk_intra,
    )
    from fla.ops.generalized_delta_rule.dplr.chunk_A_fwd import (
        chunk_dplr_fwd_intra,
    )
    from fla.ops.generalized_delta_rule.dplr.chunk_h_bwd import (
        chunk_dplr_bwd_dhu,
    )
    from fla.ops.generalized_delta_rule.dplr.chunk_h_fwd import chunk_dplr_fwd_h
    from fla.ops.generalized_delta_rule.dplr.chunk_o_bwd import (
        chunk_dplr_bwd_dAu,
        chunk_dplr_bwd_dv,
        chunk_dplr_bwd_o,
    )
    from fla.ops.generalized_delta_rule.dplr.chunk_o_fwd import chunk_dplr_fwd_o
    from fla.ops.generalized_delta_rule.dplr.wy_fast_bwd import chunk_dplr_bwd_wy
    from fla.ops.generalized_delta_rule.dplr.wy_fast_fwd import (
        prepare_wy_repr_fwd,
    )
    from fla.ops.rwkv6.chunk import chunk_rwkv6_fwd_cumsum
except (ImportError, OSError, RuntimeError):  # pragma: no cover - old FLA/CPU
    chunk_dplr_bwd_dqk_intra = None
    chunk_dplr_fwd_intra = None
    chunk_dplr_bwd_dhu = None
    chunk_dplr_fwd_h = None
    chunk_dplr_bwd_dAu = None
    chunk_dplr_bwd_dv = None
    chunk_dplr_bwd_o = None
    chunk_dplr_fwd_o = None
    chunk_dplr_bwd_wy = None
    prepare_wy_repr_fwd = None
    chunk_rwkv6_fwd_cumsum = None


_R = 4
_K = 32
_V = 128
_DPLR_CHUNK_SIZE = 16


if triton is not None:
    # FLA kernels leave tl.dot precision unspecified.  This knob is part of
    # Triton's compilation cache key and must remain set through backward's
    # first lazy compilation, not merely around the forward call.
    if hasattr(triton, "knobs"):
        triton.knobs.language.fp32_default = "ieee"

    @triton.jit
    def _query_factors_fwd(
        q_ptr,
        k_ptr,
        b_ptr,
        gi_ptr,
        aqk_ptr,
        aqb_ptr,
        qg_ptr,
        T,
        H: tl.constexpr,
        K: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
    ):
        chunk_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        head_index = tl.program_id(2)
        token_start = chunk_index * BT
        if token_start >= T:
            return

        token_offsets = tl.arange(0, BT)
        key_offsets = tl.arange(0, BK)
        token_mask = token_start + token_offsets < T
        key_mask = key_offsets < K
        base = (batch_index * T * H + head_index) * K
        token_stride = H * K
        offsets = (
            base
            + (token_start + token_offsets[:, None]) * token_stride
            + key_offsets[None, :]
        )
        mask = token_mask[:, None] & key_mask[None, :]
        query = tl.load(q_ptr + offsets, mask=mask, other=0.0)
        key = tl.load(k_ptr + offsets, mask=mask, other=0.0)
        projection = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        gate = tl.load(gi_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        query_gated = query * tl.exp(gate)
        tl.store(qg_ptr + offsets, query_gated, mask=mask)
        output_base = (
            (batch_index * T + token_start + token_offsets) * H + head_index
        ) * BT
        for column in range(0, BT):
            row_mask = token_offsets == column
            column_valid = token_start + column < T
            key_column = tl.sum(
                tl.where(row_mask[:, None], key, 0.0), axis=0
            )[None, :]
            projection_column = tl.sum(
                tl.where(row_mask[:, None], projection, 0.0), axis=0
            )[None, :]
            gate_column = tl.sum(
                tl.where(row_mask[:, None], gate, 0.0), axis=0
            )[None, :]
            relative_gate = tl.exp(gate - gate_column)
            causal = token_offsets >= column
            aqk = tl.sum(query * key_column * relative_gate, axis=1) * causal
            aqb = (
                tl.sum(query * projection_column * relative_gate, axis=1)
                * causal
            )
            output_mask = token_mask & column_valid
            tl.store(
                aqk_ptr + output_base + column,
                aqk,
                mask=output_mask,
            )
            tl.store(
                aqb_ptr + output_base + column,
                aqb,
                mask=output_mask,
            )

    @triton.jit
    def _query_factors_bwd(
        q_ptr,
        k_ptr,
        b_ptr,
        gi_ptr,
        daqk_ptr,
        daqb_ptr,
        dqg_ptr,
        dkg_ptr,
        dbg_ptr,
        dgk_last_ptr,
        dq_ptr,
        dk_ptr,
        db_ptr,
        dgk_ptr,
        T,
        H: tl.constexpr,
        K: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
    ):
        chunk_index = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch_index = batch_head // H
        head_index = batch_head % H
        token_start = chunk_index * BT
        if token_start >= T:
            return

        token_offsets = tl.arange(0, BT)
        key_offsets = tl.arange(0, BK)
        token_mask = token_start + token_offsets < T
        key_mask = key_offsets < K
        base = (batch_index * T * H + head_index) * K
        token_stride = H * K
        offsets = (
            base
            + (token_start + token_offsets[:, None]) * token_stride
            + key_offsets[None, :]
        )
        mask = token_mask[:, None] & key_mask[None, :]
        query = tl.load(q_ptr + offsets, mask=mask, other=0.0)
        key = tl.load(k_ptr + offsets, mask=mask, other=0.0)
        projection = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        gate = tl.load(gi_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        output_base = (
            (batch_index * T + token_start + token_offsets) * H + head_index
        ) * BT
        matrix_offsets = output_base[:, None] + tl.arange(0, BT)[None, :]
        matrix_mask = token_mask[:, None] & (
            token_start + tl.arange(0, BT)[None, :] < T
        )
        grad_aqk = tl.load(daqk_ptr + matrix_offsets, mask=matrix_mask, other=0.0)
        grad_aqb = tl.load(daqb_ptr + matrix_offsets, mask=matrix_mask, other=0.0)

        grad_query = tl.zeros([BT, BK], dtype=tl.float32)
        grad_key = tl.zeros([BT, BK], dtype=tl.float32)
        grad_projection = tl.zeros([BT, BK], dtype=tl.float32)
        for column in range(0, BT):
            row_mask = token_offsets == column
            key_column = tl.sum(
                tl.where(row_mask[:, None], key, 0.0), axis=0
            )[None, :]
            projection_column = tl.sum(
                tl.where(row_mask[:, None], projection, 0.0), axis=0
            )[None, :]
            query_column = tl.sum(
                tl.where(row_mask[:, None], query, 0.0), axis=0
            )[None, :]
            gate_column = tl.sum(
                tl.where(row_mask[:, None], gate, 0.0), axis=0
            )[None, :]
            grad_aqk_column = tl.sum(
                tl.where(
                    row_mask[None, :],
                    grad_aqk,
                    0.0,
                ),
                axis=1,
            )[:, None]
            grad_aqb_column = tl.sum(
                tl.where(
                    row_mask[None, :],
                    grad_aqb,
                    0.0,
                ),
                axis=1,
            )[:, None]
            grad_aqk_row = tl.sum(
                tl.where(row_mask[:, None], grad_aqk, 0.0), axis=0
            )[:, None]
            grad_aqb_row = tl.sum(
                tl.where(row_mask[:, None], grad_aqb, 0.0), axis=0
            )[:, None]
            after = token_offsets[:, None] >= column
            before = token_offsets[:, None] <= column
            forward_gate = tl.exp(gate - gate_column)
            reverse_gate = tl.exp(gate_column - gate)
            grad_query += tl.where(
                after,
                grad_aqk_column * key_column * forward_gate,
                0.0,
            )
            grad_query += tl.where(
                after,
                grad_aqb_column * projection_column * forward_gate,
                0.0,
            )
            grad_key += tl.where(
                before,
                grad_aqk_row * query_column * reverse_gate,
                0.0,
            )
            grad_projection += tl.where(
                before,
                grad_aqb_row * query_column * reverse_gate,
                0.0,
            )

        last_token = min(token_start + BT, T) - 1
        last_offsets = (
            (batch_index * T * H + last_token * H + head_index) * K
            + key_offsets
        )
        last_gate = tl.load(
            gi_ptr + last_offsets, mask=key_mask, other=0.0
        )[None, :]
        grad_query += (
            tl.load(dqg_ptr + offsets, mask=mask, other=0.0)
            * tl.exp(gate)
        )
        reverse_gate = tl.exp(last_gate - gate)
        grad_key += (
            tl.load(dkg_ptr + offsets, mask=mask, other=0.0)
            * reverse_gate
        )
        grad_projection += (
            tl.load(dbg_ptr + offsets, mask=mask, other=0.0)
            * reverse_gate
        )
        gate_local = (
            grad_query * query
            - grad_key * key
            - grad_projection * projection
        ).to(tl.float32)
        tail_base = (
            (batch_index * tl.cdiv(T, BT) + chunk_index) * H + head_index
        ) * K
        tail = tl.load(
            dgk_last_ptr + tail_base + key_offsets,
            mask=key_mask,
            other=0.0,
        )[None, :]
        grad_gate = tl.cumsum(gate_local, axis=0, reverse=True) + tail
        tl.store(dq_ptr + offsets, grad_query, mask=mask)
        tl.store(dk_ptr + offsets, grad_key, mask=mask)
        tl.store(db_ptr + offsets, grad_projection, mask=mask)
        tl.store(dgk_ptr + offsets, grad_gate, mask=mask)

    @triton.jit
    def _endpoint_fwd(
        k_ptr,
        v_ptr,
        erase_ptr,
        write_ptr,
        gamma_ptr,
        lam_ptr,
        previous_key_ptr,
        previous_value_ptr,
        history_ptr,
        update_count_ptr,
        a_ptr,
        b_ptr,
        d_ptr,
        previous_key_before_ptr,
        previous_key_out_ptr,
        previous_value_out_ptr,
        history_out_ptr,
        update_count_out_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        batch = row // (R * H)
        offs_k = tl.arange(0, K)
        offs_v = tl.arange(0, V)
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        count = tl.load(update_count_ptr + batch).to(tl.int64)
        key_carry_offset = (((batch * H + head) * R + lane) * K) + offs_k
        value_carry_offset = (((batch * H + head) * R + lane) * V) + offs_v
        previous_key = tl.load(previous_key_ptr + key_carry_offset).to(tl.float32)
        previous_value = tl.load(previous_value_ptr + value_carry_offset).to(
            tl.float32
        )
        has_history = tl.load(history_ptr + batch * R + lane)

        for token in tl.range(0, T, num_stages=1):
            key_offset = (
                ((((batch * T + token) * H + head) * R + lane) * K)
                + offs_k
            )
            value_offset = (
                ((((batch * T + token) * H + head) * R + lane) * V)
                + offs_v
            )
            lam_offset = ((batch * T + token) * H + head) * R + lane
            tick = ((count + token) % period) == 0
            key = tl.load(k_ptr + key_offset).to(tl.float32)
            gamma = tl.load(gamma_ptr + key_offset).to(tl.float32)
            tl.store(previous_key_before_ptr + key_offset, previous_key)
            previous_decayed = gamma * previous_key
            erased_key = (
                tl.load(erase_ptr + key_offset).to(tl.float32) * key
            )
            projection = tl.sum(erased_key * previous_decayed, axis=0)
            a = previous_decayed - key * projection

            value = tl.load(v_ptr + value_offset).to(tl.float32)
            write = tl.load(write_ptr + value_offset).to(tl.float32)
            current = write * value
            lam = tl.load(lam_ptr + lam_offset).to(tl.float32)
            effective_lam = tl.where(has_history, lam, 1.0)
            b = (1.0 - effective_lam) * previous_value
            d = effective_lam * current
            tl.store(a_ptr + key_offset, a, mask=tick)
            tl.store(b_ptr + value_offset, b, mask=tick)
            tl.store(d_ptr + value_offset, d, mask=tick)

            previous_key = tl.where(tick, key, previous_decayed)
            previous_value = tl.where(tick, current, previous_value)
            has_history = has_history | tick

        tl.store(previous_key_out_ptr + key_carry_offset, previous_key)
        tl.store(previous_value_out_ptr + value_carry_offset, previous_value)
        # All heads in a lane produce identical metadata; one owns the store.
        tl.store(
            history_out_ptr + batch * R + lane,
            has_history,
            mask=head == 0,
        )
        tl.store(
            update_count_out_ptr + batch,
            count + T,
            mask=(head == 0) & (lane == 0),
        )


    @triton.jit
    def _endpoint_bwd(
        k_ptr,
        v_ptr,
        erase_ptr,
        write_ptr,
        gamma_ptr,
        lam_ptr,
        previous_value_ptr,
        history_ptr,
        update_count_ptr,
        previous_key_before_ptr,
        grad_a_ptr,
        grad_b_ptr,
        grad_d_ptr,
        grad_previous_key_out_ptr,
        grad_previous_value_out_ptr,
        grad_k_ptr,
        grad_v_ptr,
        grad_erase_ptr,
        grad_write_ptr,
        grad_gamma_ptr,
        grad_lam_ptr,
        grad_previous_key_ptr,
        grad_previous_value_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = row % R
        head = (row // R) % H
        batch = row // (R * H)
        offs_k = tl.arange(0, K)
        offs_v = tl.arange(0, V)
        period = tl.where(
            lane == 0,
            1,
            tl.where(lane == 1, 16, tl.where(lane == 2, 64, 256)),
        ).to(tl.int64)
        count = tl.load(update_count_ptr + batch).to(tl.int64)
        remainder = count % period
        first_tick = tl.where(remainder == 0, 0, period - remainder)
        key_carry_offset = (((batch * H + head) * R + lane) * K) + offs_k
        value_carry_offset = (((batch * H + head) * R + lane) * V) + offs_v
        grad_key_carry = tl.load(
            grad_previous_key_out_ptr + key_carry_offset
        ).to(tl.float32)
        grad_value_carry = tl.load(
            grad_previous_value_out_ptr + value_carry_offset
        ).to(tl.float32)
        initial_value = tl.load(
            previous_value_ptr + value_carry_offset
        ).to(tl.float32)
        initial_history = tl.load(history_ptr + batch * R + lane)

        for reverse in tl.range(0, T, num_stages=1):
            token = T - 1 - reverse
            key_offset = (
                ((((batch * T + token) * H + head) * R + lane) * K)
                + offs_k
            )
            value_offset = (
                ((((batch * T + token) * H + head) * R + lane) * V)
                + offs_v
            )
            lam_offset = ((batch * T + token) * H + head) * R + lane
            tick = ((count + token) % period) == 0
            key = tl.load(k_ptr + key_offset).to(tl.float32)
            gamma = tl.load(gamma_ptr + key_offset).to(tl.float32)
            previous_key = tl.load(
                previous_key_before_ptr + key_offset
            ).to(tl.float32)
            previous_decayed = gamma * previous_key
            erase = tl.load(erase_ptr + key_offset).to(tl.float32)
            erased_key = erase * key
            projection = tl.sum(erased_key * previous_decayed, axis=0)

            previous_tick = first_tick + ((token - 1 - first_tick) // period) * period
            has_previous_tick = (token > first_tick) & (previous_tick >= 0)
            safe_previous_tick = tl.maximum(
                tl.minimum(previous_tick, T - 1), 0
            )
            previous_value_offset = (
                ((((batch * T + safe_previous_tick) * H + head) * R + lane) * V)
                + offs_v
            )
            previous_endpoint = tl.where(
                has_previous_tick,
                tl.load(v_ptr + previous_value_offset).to(tl.float32)
                * tl.load(write_ptr + previous_value_offset).to(tl.float32),
                initial_value,
            )
            value = tl.load(v_ptr + value_offset).to(tl.float32)
            write = tl.load(write_ptr + value_offset).to(tl.float32)
            current = write * value
            lam = tl.load(lam_ptr + lam_offset).to(tl.float32)
            has_history = initial_history | (token > first_tick)
            effective_lam = tl.where(has_history, lam, 1.0)

            grad_a = tl.load(grad_a_ptr + key_offset).to(tl.float32)
            grad_b = tl.load(grad_b_ptr + value_offset).to(tl.float32)
            grad_d = tl.load(grad_d_ptr + value_offset).to(tl.float32)
            grad_a = tl.where(tick, grad_a, 0.0)
            grad_b = tl.where(tick, grad_b, 0.0)
            grad_d = tl.where(tick, grad_d, 0.0)

            key_dot = tl.sum(key * grad_a, axis=0)
            grad_previous_decayed = grad_a - erased_key * key_dot
            grad_key = -projection * grad_a
            grad_erased_key = -previous_decayed * key_dot
            grad_erase = grad_erased_key * key
            grad_key += grad_erased_key * erase

            grad_current = effective_lam * grad_d
            grad_previous_endpoint = (1.0 - effective_lam) * grad_b
            grad_lam = tl.sum(
                grad_d * current - grad_b * previous_endpoint,
                axis=0,
            )
            grad_lam = tl.where(tick & has_history, grad_lam, 0.0)

            # Endpoint carries reset on ticks and pass through otherwise.
            grad_key += tl.where(tick, grad_key_carry, 0.0)
            grad_previous_decayed += tl.where(
                tick, 0.0, grad_key_carry
            )
            grad_current += tl.where(tick, grad_value_carry, 0.0)
            grad_previous_endpoint += tl.where(
                tick, 0.0, grad_value_carry
            )

            grad_v = grad_current * write
            grad_write = grad_current * value

            tl.store(grad_k_ptr + key_offset, grad_key)
            tl.store(grad_v_ptr + value_offset, grad_v)
            tl.store(grad_erase_ptr + key_offset, grad_erase)
            tl.store(grad_write_ptr + value_offset, grad_write)
            tl.store(
                grad_gamma_ptr + key_offset,
                grad_previous_decayed * previous_key,
            )
            tl.store(grad_lam_ptr + lam_offset, grad_lam)
            grad_key_carry = grad_previous_decayed * gamma
            # Carry d(previous_value) back through off-tick identity steps;
            # at the preceding tick it becomes a direct d(write*v) term.
            grad_value_carry = grad_previous_endpoint

        tl.store(grad_previous_key_ptr + key_carry_offset, grad_key_carry)
        tl.store(
            grad_previous_value_ptr + value_carry_offset,
            grad_value_carry,
        )


class _EndpointFactors(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        k: Tensor,
        v: Tensor,
        erase: Tensor,
        write: Tensor,
        gamma: Tensor,
        lam: Tensor,
        previous_key: Tensor,
        previous_value: Tensor,
        history: Tensor,
        update_count: Tensor,
    ) -> tuple[Tensor, ...]:
        B, T, H, R, K = k.shape
        V = v.shape[-1]
        tensors = tuple(
            tensor.contiguous()
            for tensor in (
                k,
                v,
                erase,
                write,
                gamma,
                lam,
                previous_key,
                previous_value,
                history,
                update_count,
            )
        )
        (k_c, v_c, erase_c, write_c, gamma_c, lam_c, previous_key_c,
         previous_value_c, history_c, update_count_c) = tensors
        a = torch.zeros_like(k_c)
        b = torch.zeros_like(v_c)
        d = torch.zeros_like(v_c)
        previous_key_before = torch.empty_like(k_c)
        previous_key_out = torch.empty_like(previous_key_c)
        previous_value_out = torch.empty_like(previous_value_c)
        history_out = torch.empty_like(history_c)
        update_count_out = torch.empty_like(update_count_c)
        _endpoint_fwd[(B * H * R,)](
            k_c,
            v_c,
            erase_c,
            write_c,
            gamma_c,
            lam_c,
            previous_key_c,
            previous_value_c,
            history_c,
            update_count_c,
            a,
            b,
            d,
            previous_key_before,
            previous_key_out,
            previous_value_out,
            history_out,
            update_count_out,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            num_warps=4,
        )
        ctx.save_for_backward(*tensors, previous_key_before)
        outputs = (
            a,
            b,
            d,
            previous_key_out,
            previous_value_out,
            history_out,
            update_count_out,
        )
        ctx.mark_non_differentiable(history_out, update_count_out)
        ctx.set_materialize_grads(False)
        return outputs

    @staticmethod
    def backward(
        ctx,
        grad_a: Tensor | None,
        grad_b: Tensor | None,
        grad_d: Tensor | None,
        grad_previous_key_out: Tensor | None,
        grad_previous_value_out: Tensor | None,
        _grad_history: Tensor | None,
        _grad_update_count: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        (*tensors, previous_key_before) = ctx.saved_tensors
        (k, v, erase, write, gamma, lam, previous_key, previous_value,
         history, update_count) = tensors
        B, T, H, R, K = k.shape
        V = v.shape[-1]
        grad_a = torch.zeros_like(k) if grad_a is None else grad_a.contiguous()
        grad_b = torch.zeros_like(v) if grad_b is None else grad_b.contiguous()
        grad_d = torch.zeros_like(v) if grad_d is None else grad_d.contiguous()
        grad_previous_key_out = (
            torch.zeros_like(previous_key)
            if grad_previous_key_out is None
            else grad_previous_key_out.contiguous()
        )
        grad_previous_value_out = (
            torch.zeros_like(previous_value)
            if grad_previous_value_out is None
            else grad_previous_value_out.contiguous()
        )
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        grad_erase = torch.zeros_like(erase)
        grad_write = torch.zeros_like(write)
        grad_gamma = torch.zeros_like(gamma)
        grad_lam = torch.zeros_like(lam)
        grad_previous_key = torch.empty_like(previous_key)
        grad_previous_value = torch.empty_like(previous_value)
        _endpoint_bwd[(B * H * R,)](
            k,
            v,
            erase,
            write,
            gamma,
            lam,
            previous_value,
            history,
            update_count,
            previous_key_before,
            grad_a,
            grad_b,
            grad_d,
            grad_previous_key_out,
            grad_previous_value_out,
            grad_k,
            grad_v,
            grad_erase,
            grad_write,
            grad_gamma,
            grad_lam,
            grad_previous_key,
            grad_previous_value,
            T=T,
            H=H,
            R=R,
            K=K,
            V=V,
            num_warps=4,
        )
        gradients = (
            grad_k,
            grad_v,
            grad_erase,
            grad_write,
            grad_gamma,
            grad_lam,
            grad_previous_key,
            grad_previous_value,
            None,
            None,
        )
        return tuple(
            gradient if needed else None
            for gradient, needed in zip(
                gradients, ctx.needs_input_grad, strict=True
            )
        )


def _query_factors_only(
    q: Tensor,
    k: Tensor,
    b: Tensor,
    gi: Tensor,
    chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build only the query-dependent DPLR factors for another MIMO read."""
    assert triton is not None
    B, T, H, K = q.shape
    if chunk_size != _DPLR_CHUNK_SIZE or K != _K:
        raise ValueError("unsupported query-factor specialization")
    if not all(tensor.is_contiguous() for tensor in (q, k, b, gi)):
        raise ValueError("query-factor inputs must be contiguous")
    A_qk = q.new_empty(B, T, H, chunk_size)
    A_qb = q.new_empty(B, T, H, chunk_size)
    qg = torch.empty_like(q)
    _query_factors_fwd[(triton.cdiv(T, chunk_size), B, H)](
        q,
        k,
        b,
        gi,
        A_qk,
        A_qb,
        qg,
        T,
        H=H,
        K=K,
        BT=chunk_size,
        BK=triton.next_power_of_2(K),
        num_warps=4,
    )
    return A_qk, A_qb, qg


def _query_factors_only_backward(
    q: Tensor,
    k: Tensor,
    b: Tensor,
    gi: Tensor,
    dA_qk: Tensor,
    dA_qb: Tensor,
    dqg: Tensor,
    dkg: Tensor,
    dbg: Tensor,
    dgk_last: Tensor,
    chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Differentiate only q-dependent factors plus their shared k/b paths."""
    assert triton is not None
    B, T, H, K = q.shape
    tensors = (q, k, b, gi, dA_qk, dA_qb, dqg, dkg, dbg, dgk_last)
    if chunk_size != _DPLR_CHUNK_SIZE or K != _K:
        raise ValueError("unsupported query-factor backward specialization")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("query-factor backward inputs must be contiguous")
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    db = torch.empty_like(b)
    dgk = torch.empty_like(gi, dtype=torch.float32)
    _query_factors_bwd[(triton.cdiv(T, chunk_size), B * H)](
        q,
        k,
        b,
        gi,
        dA_qk,
        dA_qb,
        dqg,
        dkg,
        dbg,
        dgk_last,
        dq,
        dk,
        db,
        dgk,
        T,
        H=H,
        K=K,
        BT=chunk_size,
        BK=triton.next_power_of_2(K),
        num_warps=4,
    )
    return dq, dk, db, dgk


class _MultiQueryDPLR(torch.autograd.Function):
    """Share one WY/state pipeline across Package B's four query lanes."""

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        a: Tensor,
        b: Tensor,
        gk: Tensor,
        initial_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert chunk_rwkv6_fwd_cumsum is not None
        assert chunk_dplr_fwd_intra is not None
        assert prepare_wy_repr_fwd is not None
        assert chunk_dplr_fwd_h is not None
        assert chunk_dplr_fwd_o is not None
        chunk_size = _DPLR_CHUNK_SIZE
        gi, ge = chunk_rwkv6_fwd_cumsum(gk, chunk_size)
        query = q[:, :, :, 0].contiguous()
        A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(
            q=query,
            k=k,
            a=a,
            b=b,
            gi=gi,
            ge=ge,
            scale=1.0,
            chunk_size=chunk_size,
            safe_gate=False,
        )
        w, u, _A_ab_inv = prepare_wy_repr_fwd(
            ag=ag,
            A_ab=A_ab,
            A_ak=A_ak,
            v=v,
            cu_seqlens=None,
            chunk_size=chunk_size,
        )
        h, value_trace, final_state = chunk_dplr_fwd_h(
            kg=kg,
            bg=bg,
            v=v,
            w=w,
            u=u,
            gk=gi,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=chunk_size,
        )
        outputs = [
            chunk_dplr_fwd_o(
                qg=qg,
                v=v,
                v_new=value_trace,
                A_qk=A_qk,
                A_qb=A_qb,
                h=h,
                chunk_size=chunk_size,
            )
        ]
        # A_qk/A_qb/qg are the only query-dependent factors.  Stream the
        # remaining lanes so forward never retains four full intra-chunk
        # matrices at once.
        for query_lane in range(1, _R):
            query = q[:, :, :, query_lane].contiguous()
            A_qk, A_qb, qg = _query_factors_only(
                query,
                k,
                b,
                gi,
                chunk_size,
            )
            outputs.append(
                chunk_dplr_fwd_o(
                    qg=qg,
                    v=v,
                    v_new=value_trace,
                    A_qk=A_qk,
                    A_qb=A_qb,
                    h=h,
                    chunk_size=chunk_size,
                )
            )
        ctx.save_for_backward(q, k, v, a, b, gk, initial_state)
        ctx.mark_non_differentiable(value_trace)
        return torch.stack(outputs, dim=3), final_state, value_trace

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor,
        grad_final_state: Tensor,
        _grad_value_trace: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        assert chunk_rwkv6_fwd_cumsum is not None
        assert chunk_dplr_fwd_intra is not None
        assert prepare_wy_repr_fwd is not None
        assert chunk_dplr_fwd_h is not None
        assert chunk_dplr_bwd_dAu is not None
        assert chunk_dplr_bwd_dhu is not None
        assert chunk_dplr_bwd_dv is not None
        assert chunk_dplr_bwd_o is not None
        assert chunk_dplr_bwd_wy is not None
        assert chunk_dplr_bwd_dqk_intra is not None
        q, k, v, a, b, gk, initial_state = ctx.saved_tensors
        chunk_size = _DPLR_CHUNK_SIZE
        gi, ge = chunk_rwkv6_fwd_cumsum(gk, chunk_size)
        query = q[:, :, :, 0].contiguous()
        A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(
            q=query,
            k=k,
            a=a,
            b=b,
            gi=gi,
            ge=ge,
            scale=1.0,
            chunk_size=chunk_size,
            safe_gate=False,
        )
        first_query_factors = (A_qk, A_qb, qg)
        w, u, A_ab_inv = prepare_wy_repr_fwd(
            ag=ag,
            A_ab=A_ab,
            A_ak=A_ak,
            v=v,
            cu_seqlens=None,
            chunk_size=chunk_size,
        )
        h, value_trace, _ = chunk_dplr_fwd_h(
            kg=kg,
            bg=bg,
            v=v,
            w=w,
            u=u,
            gk=gi,
            initial_state=initial_state,
            output_final_state=False,
            chunk_size=chunk_size,
        )
        del u

        grad_value_total = torch.zeros_like(v)
        grad_value_trace_total = torch.zeros_like(v)
        grad_w_total = torch.zeros_like(w)
        grad_initial_state = torch.zeros_like(initial_state)
        zero_final = torch.zeros_like(grad_final_state)
        grad_queries = []
        grad_k_total = torch.zeros_like(k)
        grad_a_total = torch.zeros_like(a)
        grad_b_total = torch.zeros_like(b)
        grad_gate_total = torch.zeros_like(gk)
        for query_lane in range(_R):
            query = q[:, :, :, query_lane].contiguous()
            if query_lane == 0:
                A_qk, A_qb, qg = first_query_factors
                first_query_factors = None
            else:
                A_qk, A_qb, qg = _query_factors_only(
                    query,
                    k,
                    b,
                    gi,
                    chunk_size,
                )
            do = grad_output[:, :, :, query_lane].contiguous()
            dv_intra, dA_qk, dA_qb = chunk_dplr_bwd_dAu(
                v=v,
                v_new=value_trace,
                do=do,
                A_qb=A_qb,
                scale=1.0,
                chunk_size=chunk_size,
            )
            dh, dh0, dv_new = chunk_dplr_bwd_dhu(
                qg=qg,
                bg=bg,
                w=w,
                gk=gi,
                h0=initial_state,
                dht=grad_final_state if query_lane == 0 else zero_final,
                do=do,
                dv=dv_intra,
                chunk_size=chunk_size,
            )
            dv0 = chunk_dplr_bwd_dv(
                A_qk=A_qk,
                kg=kg,
                do=do,
                dh=dh,
                chunk_size=chunk_size,
            )
            dqg, dkg, dw, dbg, dgk_last = chunk_dplr_bwd_o(
                k=kg,
                b=bg,
                v=v,
                v_new=value_trace,
                do=do,
                h=h,
                dh=dh,
                dv=dv_new,
                w=w,
                gk=gi,
                chunk_size=chunk_size,
                scale=1.0,
            )
            grad_value_total.add_(dv0)
            grad_value_trace_total.add_(dv_new)
            grad_w_total.add_(dw)
            grad_initial_state.add_(dh0)
            # Consume each lane's query-dependent VJP immediately with the
            # local Triton specialization.  It omits all zero shared-A work and
            # avoids allocating a useless da tensor for every read lane.
            dq, dk, db, dgk = _query_factors_only_backward(
                query,
                k,
                b,
                gi,
                dA_qk,
                dA_qb,
                dqg,
                dkg,
                dbg,
                dgk_last,
                chunk_size,
            )
            grad_queries.append(dq)
            grad_k_total.add_(dk)
            grad_b_total.add_(db)
            grad_gate_total.add_(dgk)
            del (
                query,
                A_qk,
                A_qb,
                qg,
                do,
                dv_intra,
                dA_qk,
                dA_qb,
                dh,
                dh0,
                dv_new,
                dv0,
                dqg,
                dkg,
                dw,
                dbg,
                dgk_last,
                dq,
                dk,
                db,
                dgk,
            )
        dA_ab, dA_ak, grad_value_total, dag = chunk_dplr_bwd_wy(
            A_ab_inv=A_ab_inv,
            A_ak=A_ak,
            v=v,
            ag=ag,
            dw=grad_w_total,
            du=grad_value_trace_total,
            dv0=grad_value_total,
            cu_seqlens=None,
            chunk_size=chunk_size,
        )

        # Apply the one shared WY-factor VJP exactly once.  All query-specific
        # cotangents are zero in this pass, so its query gradient is identically
        # zero while k/a/b/g receive the missing shared contribution.
        # All four intra matrices share the same packed chunk shape.  The
        # original shared forward matrices are dead after the WY VJP, so clear
        # and reuse their storage for the zero query cotangents.
        A_ak.zero_()
        A_ab.zero_()
        zero_A_qk = A_ak
        zero_A_qb = A_ab
        zero_kg = torch.zeros_like(kg)
        zero_bg = torch.zeros_like(bg)
        zero_qg = torch.zeros_like(kg)
        nt = (gk.shape[1] + chunk_size - 1) // chunk_size
        zero_gk_last = torch.zeros(
            gk.shape[0],
            nt,
            gk.shape[2],
            gk.shape[3],
            device=gk.device,
            dtype=torch.float32,
        )
        _, dk, da, db, dgk = chunk_dplr_bwd_dqk_intra(
            q=q[:, :, :, 0].contiguous(),
            k=k,
            a=a,
            b=b,
            gi=gi,
            ge=ge,
            dAqk=zero_A_qk,
            dAqb=zero_A_qb,
            dAak=dA_ak,
            dAab=dA_ab,
            dgk_last=zero_gk_last,
            dqg=zero_qg,
            dkg=zero_kg,
            dag=dag,
            dbg=zero_bg,
            chunk_size=chunk_size,
            scale=1.0,
            safe_gate=False,
        )
        grad_k_total.add_(dk)
        grad_a_total.add_(da)
        grad_b_total.add_(db)
        grad_gate_total.add_(dgk)
        grad_q = torch.stack(grad_queries, dim=3)
        return (
            grad_q.to(q),
            grad_k_total.to(k),
            grad_value_total.to(v),
            grad_a_total.to(a),
            grad_b_total.to(b),
            grad_gate_total.to(gk),
            grad_initial_state,
        )


def true_chunked_dplr_available() -> bool:
    return (
        triton is not None
        and chunk_rwkv6_fwd_cumsum is not None
        and chunk_dplr_fwd_intra is not None
        and prepare_wy_repr_fwd is not None
        and chunk_dplr_fwd_h is not None
        and chunk_dplr_fwd_o is not None
        and chunk_dplr_bwd_dAu is not None
        and chunk_dplr_bwd_dhu is not None
        and chunk_dplr_bwd_dv is not None
        and chunk_dplr_bwd_o is not None
        and chunk_dplr_bwd_wy is not None
        and chunk_dplr_bwd_dqk_intra is not None
        and torch.cuda.is_available()
    )


def _interleave(first: Tensor, second: Tensor) -> Tensor:
    B, T = first.shape[:2]
    return torch.stack((first, second), dim=2).reshape(B, 2 * T, *first.shape[2:])


def true_chunked_dplr_four_state_sequence(
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
    """Run the exact Package-B recurrence as a sequence-level chunk scan."""
    if not true_chunked_dplr_available():
        raise RuntimeError("the true chunked DPLR Triton backend is unavailable")
    B, T, H, R, K = q.shape
    V = v.shape[-1]
    if (R, K, V) != (_R, _K, _V):
        raise ValueError("true chunked DPLR requires R=4, K=32, V=128")
    if T < 16:
        raise ValueError("true chunked DPLR requires at least 16 tokens")
    if any(tensor.device != q.device for tensor in (
        k, v, erase, write, gamma, lam, state, previous_key,
        previous_value, history, update_count,
    )):
        raise ValueError("all true chunked DPLR inputs must share a device")

    a, b, d, key_out, value_out, history_out, count_out = (
        _EndpointFactors.apply(
            k,
            v,
            erase,
            write,
            gamma,
            lam,
            previous_key,
            previous_value,
            history,
            update_count,
        )
    )
    periods = update_count.new_tensor((1, 16, 64, 256))
    positions = update_count[:, None] + torch.arange(
        T, device=q.device, dtype=torch.int64
    )[None]
    ticks = positions[:, :, None].remainder(periods[None, None]).eq(0)
    tick = ticks[:, :, None, :, None]
    zero_k = torch.zeros_like(k)
    zero_v = torch.zeros_like(v)
    destructive_key = torch.where(tick, k, zero_k)
    projection_key = torch.where(tick, erase * k * gamma, zero_k)
    projection_output = -destructive_key
    additive_key = a
    additive_value = b
    d = torch.where(tick, d, zero_v)
    log_gamma = gamma.log()

    packed_key = _interleave(destructive_key, additive_key)
    packed_value = _interleave(d, additive_value)
    packed_a = _interleave(projection_key, zero_k)
    packed_b = _interleave(projection_output, zero_k)
    packed_gate = _interleave(log_gamma, zero_k)
    length = 2 * T
    heads = H * R
    packed_key = packed_key.reshape(B, length, heads, K)
    packed_value = packed_value.reshape(B, length, heads, V)
    packed_a = packed_a.reshape(B, length, heads, K)
    packed_b = packed_b.reshape(B, length, heads, K)
    packed_gate = packed_gate.reshape(B, length, heads, K)
    flat_state = state.reshape(B, heads, K, V)

    queries = q[:, :, :, None].expand(B, T, H, R, R, K)
    packed_queries = _interleave(
        torch.zeros_like(queries), queries
    ).reshape(B, length, heads, R, K)
    output, state_out, value_trace = _MultiQueryDPLR.apply(
        packed_queries,
        packed_key,
        packed_value,
        packed_a,
        packed_b,
        packed_gate,
        flat_state,
    )
    reads = output[:, 1::2].reshape(B, T, H, R, R, V).transpose(3, 4)

    # HOLA admission scores are detached by design. ``v_new`` at the first
    # DPLR substep is exactly rho=u^T Gamma S_before, so the score requires no
    # fifth state-carry pass and no materialized KxV innovation tensor.
    with torch.no_grad():
        rho = value_trace[:, 0::2].reshape(B, T, H, R, V)
        y = d.detach() - rho
        k_detached = destructive_key.detach()
        a_detached = a.detach()
        b_detached = b.detach()
        innovation_sq = (
            k_detached.square().sum(-1) * y.square().sum(-1)
            + a_detached.square().sum(-1) * b_detached.square().sum(-1)
            + 2.0
            * (k_detached * a_detached).sum(-1)
            * (y * b_detached).sum(-1)
        ).clamp_min_(0.0)

    return (
        reads,
        state_out.reshape(B, H, R, K, V),
        key_out,
        value_out,
        innovation_sq,
        history_out,
        count_out,
    )


__all__ = [
    "true_chunked_dplr_available",
    "true_chunked_dplr_four_state_sequence",
]
