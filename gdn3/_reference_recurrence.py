"""FROZEN reference for the GDN3 Kronecker-Residual MIMO recurrence.

This is a byte-for-byte copy of `GDN3LinearAttn._gdn3_recurrent_state` as it
existed BEFORE the chunked-scan rewrite. It exists so the parity test
(`tests/test_chunk_parity.py`) can validate any new/optimized implementation of
the recurrence against the known-correct sequential token loop.

DO NOT "optimize" this file. Its only job is to be the ground truth. The
chunked kernel must reproduce its output within fp32 tolerance.

It calls the layer's own helpers (`_kron_read_vec`, `_stable_alpha_vec`,
`_compact_vec`) — those are NOT part of the rewrite, so they stay shared.

Exact semantics to preserve (see docs/HANDOFF_chunked_scan.md §3):
  * per token, scalar decay gamma_t in [0,1] per chain:
        A  *= gamma_t   (=> the outer product A(x)B decays by gamma_t**2)
        Bk *= gamma_t
        Vb *= gamma_t   (residual keys decay by gamma_t**1)
  * gated erase key   h = b_gate * k ;  gated write value  u = w_gate * v
  * read s_h = S*h ; delta r = u - s_h ; exact alpha = (1-exp(-k.h))/(k.h)
  * read s_q = S*q ; output  y = s_q + alpha*(k.q)*r   (post-update shortcut)
  * write slot p:  U[:,:,p] = alpha*r ,  Vb[:,:,p] = k   (p advances 0..P-1)
  * residual buffer is CIRCULAR and stays FULL: compaction blends UV^T into the
    Kronecker factors but RETURNS U, Vb UNCHANGED and only resets p=0, so the
    next window overwrites slots 0,1,... and every read uses all P slots (a
    sliding window of the last P writes that spans window boundaries).
"""
from __future__ import annotations

import torch


def reference_recurrent_state(layer, q_features, k_features, v_features,
                              b_gates, w_gates, decay_factors):
    """Ground-truth sequential recurrence. Signature matches
    GDN3LinearAttn._gdn3_recurrent_state (self -> layer)."""
    B, T, H, M, K = q_features.shape
    V = v_features.shape[-1]
    R, P = layer.R, layer.P
    a_k, b_k = layer.a_k, layer.b_k
    a_v, b_v = layer.a_v, layer.b_v
    N = B * H * M
    device = q_features.device
    dtype = torch.float32

    def _flat(x, d):
        return x.permute(1, 0, 2, 3, 4).reshape(T, N, d).to(dtype)
    q = _flat(q_features, K); k = _flat(k_features, K); v = _flat(v_features, V)
    bg = _flat(b_gates, K); wg = _flat(w_gates, V)
    dec = decay_factors.permute(1, 0, 2, 3).reshape(T, N).to(dtype)

    A = torch.zeros(N, R, a_v, a_k, dtype=dtype, device=device)
    Bk = torch.zeros(N, R, b_v, b_k, dtype=dtype, device=device)
    U = torch.zeros(N, V, P, dtype=dtype, device=device)
    Vb = torch.zeros(N, K, P, dtype=dtype, device=device)
    p = 0

    outs = []
    comp_err = 0.0
    for t in range(T):
        gamma = dec[t].clamp(0.0, 1.0)
        A = A * gamma.view(N, 1, 1, 1)
        Bk = Bk * gamma.view(N, 1, 1, 1)
        Vb = Vb * gamma.view(N, 1, 1)

        k_t, q_t = k[t], q[t]
        h = bg[t] * k_t
        u = wg[t] * v[t]

        s_h = layer._kron_read_vec(A, Bk, U, Vb, h)
        r = u - s_h

        c = (k_t * h).sum(-1)
        alpha = layer._stable_alpha_vec(c)

        s_q = layer._kron_read_vec(A, Bk, U, Vb, q_t)
        kq = (k_t * q_t).sum(-1)
        y = s_q + (alpha * kq).unsqueeze(-1) * r
        outs.append(y)

        new_u = (alpha.unsqueeze(-1) * r)
        U = torch.cat([U[:, :, :p], new_u.unsqueeze(-1), U[:, :, p + 1:]], dim=2)
        Vb = torch.cat([Vb[:, :, :p], k_t.unsqueeze(-1), Vb[:, :, p + 1:]], dim=2)
        p += 1

        if p >= P:
            with torch.no_grad():
                A, Bk, U, Vb, err = layer._compact_vec(A, Bk, U, Vb, layer.slow_decay)
            comp_err = comp_err + err
            p = 0

    Y = torch.stack(outs, dim=0)
    Y = Y.reshape(T, B, H, M, V).permute(1, 0, 2, 3, 4).contiguous()
    return Y
