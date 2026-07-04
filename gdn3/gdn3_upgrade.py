"""
GDN3 Linear Attention — Component-Compatible Drop-In for Qwen3.5

Replaces ONLY the SSM recurrence (A_log, dt_bias, in_proj_a/b state evolution)
with GDN3 Kronecker-Residual MIMO state. Preserves:
  - Qwen3.5's in_proj_qkv, in_proj_z, conv1d (learned projections)
  - Qwen3.5's norm, out_proj, output gating
  - Full attention layers (untouched)

Architecture mapping:
  Qwen3.5:  x -> proj_qkv -> conv -> SSM_state -> linear_attn -> norm -> gate -> out_proj
  GDN3:     x -> proj_qkv -> conv -> GDN3_KrMIMO_state -> read -> norm -> gate -> out_proj

Usage:
    from integration.gdn3_upgrade import GDN3UpgradeManager
    manager = GDN3UpgradeManager(model)
    manager.apply_upgrade()  # Only linear attention layers
    manager.save(output_dir)
"""

from __future__ import annotations

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

try:
    from .triton_kernels import kron_read_chunk_autograd
except Exception:  # pragma: no cover - optional acceleration path
    kron_read_chunk_autograd = None

# NOTE: this module is self-contained — the Kronecker read, exact-alpha write
# coefficient and two-timescale compaction are implemented as methods below
# (_kron_read_vec / _stable_alpha_vec / _compact_vec). The old top-level import
# from `gdn3_production.kernels` pulled in symbols that were never used here
# (they only appeared in a docstring), so it has been removed.


# =============================================================================
# GDN3 LINEAR ATTENTION — Qwen3.5 Component-Compatible
# =============================================================================

class GDN3LinearAttn(nn.Module):
    """
    GDN3 Kronecker-Residual MIMO replacement for Qwen3.5 linear attention.

    Preserves Qwen3.5's projection structure:
      - in_proj_qkv: learned Q,K,V projections [H_lin*3*K, D]
      - in_proj_z:   learned gate projections   [2*D, D]
      - conv1d:      causal convolution         [H_lin*3*K, 1, conv_k]
      - norm:        per-head RMSNorm           [K]
      - out_proj:    output projection          [D, 2*D]

    Replaces ONLY the SSM recurrence:
      - A_log, dt_bias, in_proj_a, in_proj_b -> GDN3 Kr-MIMO state
      - exp(A*dt) state evolution -> Kronecker-residual recurrence
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()

        D = config.hidden_size                        # 1024
        H = config.linear_num_key_heads              # 16
        K = config.linear_key_head_dim               # 128
        V = config.linear_value_head_dim             # 128
        conv_k = config.linear_conv_kernel_dim       # 4

        # GDN3 config derived from Qwen3.5 dimensions
        self.num_lanes = 4                            # MIMO capacity
        self.kron_rank = 4                           # Kronecker rank R
        self.residual_rank = int(os.environ.get("GDN3_P", 16))  # Exact write buffer P
        # ^ compaction fires every P tokens; raising P cuts compaction frequency
        #   (compaction is ~half the step time) and enlarges the exact recency cache.
        self.a_k, self.b_k = 16, 8                     # 16*8 = 128 = K
        self.a_v, self.b_v = 16, 8                     # 16*8 = 128 = V
        self.slow_decay = float(os.environ.get("GDN3_SLOW_DECAY", 0.97))   # two-timescale blend (sweepable)
        self.decay_clamp = float(os.environ.get("GDN3_DECAY_CLAMP", 0.999)) # forgetting floor (sweepable)
        self.compact_mode = os.environ.get("GDN3_COMPACT_MODE", "svd").lower()
        self.use_triton_kron = os.environ.get("GDN3_TRITON_KRON", "1") != "0"

        self.H, self.K, self.V, self.D = H, K, V, D
        self.M, self.R, self.P = self.num_lanes, self.kron_rank, self.residual_rank
        self.a_k, self.b_k = self.a_k, self.b_k
        self.a_v, self.b_v = self.a_v, self.b_v

        # ==================================================================
        # PRESERVED QWEN3.5 COMPONENTS (loaded from checkpoint)
        # ==================================================================

        # Q,K,V projections: [H*3*K, D] -> projects x to q,k,v for all heads
        self.in_proj_qkv = nn.Parameter(torch.empty(H * 3 * K, D))

        # Gate projections: [2*D, D] -> output gate + value gate
        self.in_proj_z = nn.Parameter(torch.empty(2 * D, D))

        # State factor projections (warm-start from Qwen3.5's in_proj_a/b)
        # in_proj_a: [H, D] -> projects x to SSM state a (now Kronecker factor init)
        self.in_proj_a = nn.Parameter(torch.empty(H, D))
        # in_proj_b: [H, D] -> projects x to SSM state b (now Kronecker factor init)
        self.in_proj_b = nn.Parameter(torch.empty(H, D))

        # Causal convolution: [H*3*K, 1, conv_k]
        # No padding — we handle causal padding manually in forward()
        self.conv1d = nn.Conv1d(
            in_channels=H * 3 * K,
            out_channels=H * 3 * K,
            kernel_size=conv_k,
            groups=H * 3 * K,
            bias=False,
        )
        self.conv_kernel_size = conv_k

        # Per-head RMSNorm: [K]
        self.norm = nn.Parameter(torch.ones(K))

        # Output projection: [D, 2*D] (gate + attention output concatenated)
        self.out_proj = nn.Linear(2 * D, D, bias=False)

        # ==================================================================
        # GDN3 KRONECKER-RESIDUAL MIMO STATE (replaces SSM recurrence)
        # ==================================================================

        # Braided decay projections: [H, M, T_braid, D]
        self.W_decay = nn.Parameter(torch.randn(H, self.M, 4, D) * 0.01)
        self.register_buffer('base_decay_rates',
                             torch.tensor([0.05, 0.02, 0.005, 0.001]))

        # Lane router: Linear(D -> H*M)
        self.router_proj = nn.Linear(D, H * self.M, bias=True)

        # Aggregation projection: [H*V] -> D
        # Created in __init__ so optimizers see it before first forward
        self._agg_proj = nn.Linear(H * V, D, bias=False)
        nn.init.kaiming_normal_(self._agg_proj.weight, a=0, mode='fan_in')

        # Write gate projection (GDN3): [H, M, V, D]
        # Initialized from in_proj_b pattern (warm-start)
        self.W_w = nn.Parameter(torch.zeros(H, self.M, V, D))

        # Erase gate projection (GDN3): [H, M, K, D]
        # Initialized from in_proj_a pattern (warm-start)
        self.W_b = nn.Parameter(torch.zeros(H, self.M, K, D))

        # ==================================================================
        # COPRODUCT CHANNELS (Hopf-inspired bilinear binding)
        # Blends Kronecker-factored outer products with dense projections
        # ==================================================================
        C = 4  # Coproduct rank
        self.coproduct_rank = C
        # Factor projections: [H, C, factor_dim, D] per feature type
        self.W_q_a = nn.Parameter(torch.randn(H, C, self.a_k, D) * 0.005)
        self.W_q_b = nn.Parameter(torch.randn(H, C, self.b_k, D) * 0.005)
        self.W_k_a = nn.Parameter(torch.randn(H, C, self.a_k, D) * 0.005)
        self.W_k_b = nn.Parameter(torch.randn(H, C, self.b_k, D) * 0.005)
        self.W_v_a = nn.Parameter(torch.randn(H, C, self.a_v, D) * 0.005)
        self.W_v_b = nn.Parameter(torch.randn(H, C, self.b_v, D) * 0.005)
        # Blend weights: learned mix between dense and coproduct per head
        self.coprod_mix_qk = nn.Parameter(torch.zeros(H))  # starts at 0 (dense-only)
        self.coprod_mix_v = nn.Parameter(torch.zeros(H))
        # Strength gates per head
        self.coprod_strength_qk = nn.Parameter(torch.ones(H))
        self.coprod_strength_v = nn.Parameter(torch.ones(H))

        # RoPE embeddings (for partial lane-specific rotation)
        max_seq_len = getattr(config, 'max_position_embeddings', 4096)
        partial_rotary_factor = getattr(config, 'partial_rotary_factor', 0.25)
        rope_theta = 10000.0

        dim_range = torch.arange(0, int(K * partial_rotary_factor) // 2, dtype=torch.float32)
        inv_freq = 1.0 / (rope_theta ** (dim_range / max(1, len(dim_range) - 1) if len(dim_range) > 1 else 1))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer('cos_pe', torch.cos(angles))
        self.register_buffer('sin_pe', torch.sin(angles))

        # Lane-specific RoPE config
        self.lane_rope_fractions = [0.50, 0.25, 0.50, 0.50][:self.M]
        self.lane_rope_scales = [1.0, 0.3, 0.5, 0.2][:self.M]

        # Diagnostics
        self.compaction_errors: List[float] = []
        self._gdn3_chunk_const_cache = {}
        self._gdn3_sketch_cache = {}
        self._gdn3_rope_cache = {}

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for GDN3 components."""
        nn.init.xavier_uniform_(self.W_decay, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.router_proj.weight, gain=2 ** -2.5)
        nn.init.zeros_(self.router_proj.bias)
        nn.init.xavier_uniform_(self.W_w, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.W_b, gain=2 ** -2.5)

    def _generate_coproduct_channels(
        self,
        q_dense: torch.Tensor,  # [B, T, H, K]
        k_dense: torch.Tensor,  # [B, T, H, K]
        v_dense: torch.Tensor,  # [B, T, H, K] (before V expansion)
        x: torch.Tensor,        # [B, T, D]
    ):
        """Blend coproduct bilinear features with dense projections.

        Coproduct: z = sum_c vec(W_a[c]@x x W_b[c]@x) / sqrt(C)
        Blend:     out = (1-alpha)*dense + alpha*strength*coprod
        """
        B, T, H = x.shape[0], x.shape[1], self.H
        C = self.coproduct_rank
        sqrt_C = math.sqrt(C)
        x_f = x.float()

        # Coproduct q/k: flatten per-head coproduct weights so cuBLAS handles
        # the projection as GEMM instead of a generic einsum.
        q_a = F.linear(x_f, self.W_q_a.float().reshape(-1, self.D)).view(B, T, H, C, self.a_k)
        q_b = F.linear(x_f, self.W_q_b.float().reshape(-1, self.D)).view(B, T, H, C, self.b_k)
        q_coprod = torch.einsum('bthci,bthcj->bthcij', q_a, q_b)  # [B,T,H,C,a_k,b_k]
        q_coprod = q_coprod.sum(dim=3) / sqrt_C                  # [B,T,H,a_k,b_k]
        q_coprod = q_coprod.reshape(B, T, H, -1)                          # [B,T,H,K]

        k_a = F.linear(x_f, self.W_k_a.float().reshape(-1, self.D)).view(B, T, H, C, self.a_k)
        k_b = F.linear(x_f, self.W_k_b.float().reshape(-1, self.D)).view(B, T, H, C, self.b_k)
        k_coprod = torch.einsum('bthci,bthcj->bthcij', k_a, k_b)
        k_coprod = k_coprod.sum(dim=3) / sqrt_C
        k_coprod = k_coprod.reshape(B, T, H, -1)

        # Coproduct v: produces [B,T,H,V] where V=a_v*b_v
        v_a = F.linear(x_f, self.W_v_a.float().reshape(-1, self.D)).view(B, T, H, C, self.a_v)
        v_b = F.linear(x_f, self.W_v_b.float().reshape(-1, self.D)).view(B, T, H, C, self.b_v)
        v_coprod = torch.einsum('bthci,bthcj->bthcij', v_a, v_b)
        v_coprod = v_coprod.sum(dim=3) / sqrt_C                          # [B,T,H,a_v,b_v]
        v_coprod = v_coprod.reshape(B, T, H, -1)                                # [B,T,H,V]

        # Blend with learned mix weights (sigmoid ensures 0-1 range)
        alpha_qk = torch.sigmoid(self.coprod_mix_qk).view(1, 1, H, 1)  # [1,1,H,1]
        alpha_v = torch.sigmoid(self.coprod_mix_v).view(1, 1, H, 1)

        str_qk = self.coprod_strength_qk.view(1, 1, H, 1)
        str_v = self.coprod_strength_v.view(1, 1, H, 1)

        q_out = (1 - alpha_qk) * q_dense + alpha_qk * str_qk * q_coprod
        k_out = (1 - alpha_qk) * k_dense + alpha_qk * str_qk * k_coprod
        v_out = (1 - alpha_v) * v_dense + alpha_v * str_v * v_coprod

        return q_out, k_out, v_out

    def load_qwen_weights(self, state_dict: Dict[str, torch.Tensor], layer_idx: int):
        """Load Qwen3.5 linear attention weights into this module.

        Maps Qwen3.5 parameter names to our component names.
        Warm-starts GDN3 components from Qwen3.5's SSM projections.
        """
        prefix = f"linear_attn."
        # Handle both flat keys and nested keys
        key_map = {
            'in_proj_qkv.weight': 'in_proj_qkv',
            'in_proj_z.weight': 'in_proj_z',
            'in_proj_a.weight': 'in_proj_a',
            'in_proj_b.weight': 'in_proj_b',
            'conv1d.weight': 'conv1d.weight',
            'norm.weight': 'norm',
            'out_proj.weight': 'out_proj.weight',
        }

        for qwen_key, our_attr in key_map.items():
            full_key = f"{prefix}{qwen_key}"
            if full_key in state_dict:
                val = state_dict[full_key]
            elif qwen_key in state_dict:
                val = state_dict[qwen_key]
            else:
                # Try to find it by shape matching
                found = False
                for k in state_dict:
                    if k.endswith(qwen_key) or qwen_key in k:
                        val = state_dict[k]
                        found = True
                        break
                if not found:
                    continue

            if hasattr(self, our_attr) and not '.' in our_attr:
                param = getattr(self, our_attr)
                if val.shape == param.shape:
                    param.data.copy_(val.to(param.device, param.dtype))
                else:
                    # Reshape if needed
                    param.data.copy_(val.reshape(param.shape).to(param.device, param.dtype))
            elif hasattr(self, our_attr.split('.')[0]):
                module = getattr(self, our_attr.split('.')[0])
                param_name = our_attr.split('.')[1]
                if hasattr(module, param_name):
                    param = getattr(module, param_name)
                    if val.shape == param.shape:
                        param.data.copy_(val.to(param.device, param.dtype))

        # Warm-start GDN3 components from Qwen3.5's SSM projections
        self._warm_start_gdn3(state_dict, prefix)

    def _warm_start_gdn3(self, state_dict: Dict[str, torch.Tensor], prefix: str):
        """Initialize GDN3 components from Qwen3.5's SSM parameters.

        Strategy:
        - W_decay: initialize from A_log + dt_bias decay pattern
        - W_b, W_w: zero-init (gates learn during fine-tuning)
        - Router bias: lane 0 preferred (warm-start toward single-lane behavior)
        """
        H, M = self.H, self.M

        # Get Qwen3.5's A_log and dt_bias for decay initialization
        a_log = None
        dt_bias = None
        for key, val in state_dict.items():
            if 'A_log' in key:
                a_log = val
            if 'dt_bias' in key:
                dt_bias = val

        # Initialize W_decay from A_log + dt_bias pattern
        if a_log is not None and dt_bias is not None:
            with torch.no_grad():
                for h in range(min(H, len(a_log))):
                    base_decay = torch.exp(a_log[h] + dt_bias[h]).clamp(min=0.001, max=1.0).item()
                    for m in range(M):
                        timescale_factor = 1.0 / (m + 1)
                        scale = base_decay ** timescale_factor
                        self.W_decay[h, m, :, :].data *= scale

        # Router bias: lane 0 gets positive bias (warm-start toward single lane)
        with torch.no_grad():
            bias = self.router_proj.bias
            for h in range(H):
                bias[h * M + 0] = 2.0   # Lane 0 preferred
                for m in range(1, M):
                    bias[h * M + m] = -1.0

    def _lane_rope_factors(self, m_idx: int, T: int, K_dim: int, device: torch.device):
        if not hasattr(self, "_gdn3_rope_cache"):
            self._gdn3_rope_cache = {}
        frac = self.lane_rope_fractions[m_idx]
        scale = self.lane_rope_scales[m_idx]
        d_pairs = int(K_dim * frac // 2)
        if d_pairs < 1:
            return None, None
        key = (device.type, device.index, m_idx, T, K_dim, d_pairs, float(scale))
        cached = self._gdn3_rope_cache.get(key)
        if cached is not None:
            return cached

        if scale != 1.0:
            inv_freq = 1.0 / (10000.0 ** (
                torch.arange(0, d_pairs, device=device, dtype=torch.float32) /
                max(1, int(self.K * 0.25) // 2 - 1)
            ))
            angles = (torch.arange(T, device=device, dtype=torch.float32) * scale)[:, None] * inv_freq[None, :]
        else:
            if d_pairs > self.cos_pe.shape[1]:
                cached = (
                    torch.ones(T, d_pairs, device=device, dtype=torch.float32),
                    torch.zeros(T, d_pairs, device=device, dtype=torch.float32),
                )
                self._gdn3_rope_cache[key] = cached
                return cached
            inv_freq = 1.0 / (10000.0 ** (
                torch.arange(0, d_pairs, device=device, dtype=torch.float32) /
                max(1, d_pairs - 1)
            ))
            angles = (torch.arange(T, device=device, dtype=torch.float32) * scale)[:, None] * inv_freq[None, :]
        cached = (torch.cos(angles), torch.sin(angles))
        self._gdn3_rope_cache[key] = cached
        return cached

    def _apply_partial_rope(self, x: torch.Tensor, t_max: int) -> torch.Tensor:
        """Apply lane-specific partial RoPE to key/query features."""
        B, T, H, M, K_dim = x.shape
        result = x.clone()

        for m_idx in range(self.M):
            cos_t, sin_t = self._lane_rope_factors(m_idx, min(T, t_max), K_dim, x.device)
            if cos_t is None:
                continue
            d_pairs = cos_t.shape[1]

            # Apply rotation
            lane_x = result[:, :, :, m_idx, :]  # [B, T, H, K]
            x1 = lane_x[..., :d_pairs]
            x2 = lane_x[..., d_pairs:2*d_pairs]

            cos_bc = cos_t.unsqueeze(0).unsqueeze(2)  # [1, T, 1, d_pairs]
            sin_bc = sin_t.unsqueeze(0).unsqueeze(2)

            lane_x[..., :d_pairs] = x1 * cos_bc - x2 * sin_bc
            lane_x[..., d_pairs:2*d_pairs] = x1 * sin_bc + x2 * cos_bc
            result[:, :, :, m_idx, :] = lane_x

        return result

    def _gdn3_recurrent_state(
        self,
        q_features: torch.Tensor,   # [B, T, H, M, K]
        k_features: torch.Tensor,   # [B, T, H, M, K]
        v_features: torch.Tensor,   # [B, T, H, M, V]
        b_gates: torch.Tensor,      # [B, T, H, M, K] - erase gates
        w_gates: torch.Tensor,      # [B, T, H, M, V] - write gates
        decay_factors: torch.Tensor, # [B, T, H, M] - decay
    ) -> torch.Tensor:
        """GDN3 Kronecker-Residual MIMO recurrent state evolution (vectorized).

        Mathematically identical to the per-(head,lane,batch) reference
        recurrence, but batched over N = B*H*M "chains" so only the time
        axis is a Python loop. Each step:
          1. scalar decay of Kronecker factors (A,B) and residual keys (Vb)
          2. gated erase key h = b_gate*k, gated write value u = w_gate*v
          3. Kronecker+residual read s_h = S*h ; delta r = u - s_h
          4. exact write coeff alpha = stable_alpha(k.h)
          5. read s_q = S*q ; output y = s_q + alpha*(k.q)*r  (post-update shortcut)
          6. append (alpha*r, k) into the P-slot residual buffer
          7. every P tokens, compact residual UV^T into top-R Kronecker
             factors via rearrangement SVD (under no_grad -> truncated BPTT).

        Returns [B, T, H, M, V] with autograd connectivity within each
        P-token compaction window.
        """
        B, T, H, M, K = q_features.shape
        V = v_features.shape[-1]
        R, P = self.R, self.P
        a_k, b_k = self.a_k, self.b_k
        a_v, b_v = self.a_v, self.b_v
        N = B * H * M
        device = q_features.device
        dtype = torch.float32

        # Flatten (B,H,M) -> N chains, time-major: [T, N, dim]
        def _flat(x, d):
            return x.permute(1, 0, 2, 3, 4).reshape(T, N, d).to(dtype)
        q = _flat(q_features, K); k = _flat(k_features, K); v = _flat(v_features, V)
        bg = _flat(b_gates, K); wg = _flat(w_gates, V)
        dec = decay_factors.permute(1, 0, 2, 3).reshape(T, N).to(dtype)  # [T, N]

        # Kronecker-residual state per chain
        A = torch.zeros(N, R, a_v, a_k, dtype=dtype, device=device)
        Bk = torch.zeros(N, R, b_v, b_k, dtype=dtype, device=device)
        U = torch.zeros(N, V, P, dtype=dtype, device=device)   # residual values
        Vb = torch.zeros(N, K, P, dtype=dtype, device=device)  # residual keys

        outs = []
        comp_err = 0.0
        eye_cache, lower_cache, old_mask_cache = self._gdn3_chunk_constants(P, dtype, device)

        def _kron_read_chunk(x_chunk: torch.Tensor) -> torch.Tensor:
            """Read the carried Kronecker state for [N,C,K] chunk inputs."""
            if self.use_triton_kron and kron_read_chunk_autograd is not None and A.is_cuda:
                return kron_read_chunk_autograd(
                    A, Bk, x_chunk, a_v=a_v, a_k=a_k, b_v=b_v, b_k=b_k
                )
            C = x_chunk.shape[1]
            X = x_chunk.reshape(N, C, a_k, b_k)
            AX = torch.einsum('nrau,ncub->ncrab', A, X)       # [N,C,R,a_v,b_k]
            AXB = torch.einsum('ncrab,nrdb->ncrad', AX, Bk)   # [N,C,R,a_v,b_v]
            return AXB.sum(2).reshape(N, C, a_v * b_v)

        for t0 in range(0, T, P):
            C = min(P, T - t0)
            q_c = q[t0:t0 + C].transpose(0, 1)                # [N,C,K]
            k_c = k[t0:t0 + C].transpose(0, 1)                # [N,C,K]
            v_c = v[t0:t0 + C].transpose(0, 1)                # [N,C,V]
            h_c = (bg[t0:t0 + C] * k[t0:t0 + C]).transpose(0, 1)
            u_c = (wg[t0:t0 + C] * v[t0:t0 + C]).transpose(0, 1)
            gamma = dec[t0:t0 + C].transpose(0, 1).clamp(0.0, 1.0)  # [N,C]

            prefix = gamma.cumprod(dim=1)                    # decay from chunk start through i
            prefix2 = prefix.square()
            denom = prefix.clamp_min(torch.finfo(dtype).tiny)
            rel_decay = prefix.unsqueeze(2) / denom.unsqueeze(1)    # [N,i,j] = prod gamma[j+1:i]

            old_mask = old_mask_cache[:C]

            def _old_residual_read(x_chunk: torch.Tensor) -> torch.Tensor:
                coeff = torch.einsum('nkp,nck->ncp', Vb, x_chunk)
                coeff = coeff * prefix.unsqueeze(-1) * old_mask.unsqueeze(0)
                return torch.einsum('nvp,ncp->ncv', U, coeff)

            kron_h = _kron_read_chunk(h_c) * prefix2.unsqueeze(-1)
            kron_q = _kron_read_chunk(q_c) * prefix2.unsqueeze(-1)
            old_h = _old_residual_read(h_c)
            old_q = _old_residual_read(q_c)

            base_h = u_c - kron_h - old_h
            alpha = self._stable_alpha_vec((k_c * h_c).sum(-1))      # [N,C]

            lower = lower_cache[:C, :C]
            kh = torch.einsum('njk,nik->nij', k_c, h_c) * rel_decay[:, :C, :C] * lower
            system = eye_cache[:, :C, :C] + alpha.unsqueeze(-1) * kh
            rhs = alpha.unsqueeze(-1) * base_h
            new_u = torch.linalg.solve_triangular(
                system, rhs, upper=False, unitriangular=True
            )                                                       # [N,C,V] = alpha*r

            kq_prev = torch.einsum('njk,nik->nij', k_c, q_c) * rel_decay[:, :C, :C] * lower
            prev_q = torch.einsum('njv,nij->niv', new_u, kq_prev)
            kq = (k_c * q_c).sum(-1)
            y = kron_q + old_q + prev_q + new_u * kq.unsqueeze(-1)
            outs.append(y.transpose(0, 1))                          # [C,N,V]

            end_decay = prefix[:, C - 1:C] / denom[:, :C]            # prod gamma[j+1:C-1]
            U_new = new_u.transpose(1, 2)                            # [N,V,C]
            Vb_new = (k_c * end_decay.unsqueeze(-1)).transpose(1, 2) # [N,K,C]
            if C == P:
                U, Vb = U_new, Vb_new
                with torch.no_grad():
                    A, Bk, U, Vb, err = self._compact_fast(A, Bk, U, Vb)
                comp_err = comp_err + err   # tensor accumulate (no per-window sync)
            else:
                tail_decay = prefix[:, C - 1].view(N, 1, 1)
                U = torch.cat([U_new, U[:, :, C:]], dim=2)
                Vb = torch.cat([Vb_new, Vb[:, :, C:] * tail_decay], dim=2)

        # Keep diagnostics on-device by default; converting to float here forces
        # one GPU sync per GDN3 layer and is visible in end-to-end training.
        if os.environ.get("GDN3_SYNC_COMP_ERR", "0") == "1":
            self._last_comp_err = float(comp_err) if torch.is_tensor(comp_err) else comp_err
        else:
            self._last_comp_err = comp_err.detach() if torch.is_tensor(comp_err) else comp_err
        Y = torch.cat(outs, dim=0)                         # [T, N, V]
        Y = Y.reshape(T, B, H, M, V).permute(1, 0, 2, 3, 4).contiguous()
        return Y

    def _cache_key(self, device: torch.device, dtype: torch.dtype, *shape):
        return (device.type, device.index, str(dtype), *shape)

    def _gdn3_chunk_constants(self, P: int, dtype: torch.dtype, device: torch.device):
        """Fixed per-window masks used by the chunked recurrence."""
        if not hasattr(self, "_gdn3_chunk_const_cache"):
            self._gdn3_chunk_const_cache = {}
        key = self._cache_key(device, dtype, P)
        cached = self._gdn3_chunk_const_cache.get(key)
        if cached is None:
            eye = torch.eye(P, dtype=dtype, device=device).unsqueeze(0)
            lower = torch.tril(torch.ones(P, P, dtype=dtype, device=device), diagonal=-1)
            slot_ids = torch.arange(P, device=device)
            row_ids = torch.arange(P, device=device)
            old_mask = (slot_ids.view(1, P) >= row_ids.view(P, 1)).to(dtype)
            cached = (eye, lower, old_mask)
            self._gdn3_chunk_const_cache[key] = cached
        return cached

    def _gdn3_sketch_matrix(self, N: int, n: int, q_rank: int,
                            dtype: torch.dtype, device: torch.device):
        """Deterministic Gaussian sketch used by no-grad compaction."""
        if not hasattr(self, "_gdn3_sketch_cache"):
            self._gdn3_sketch_cache = {}
        key = self._cache_key(device, dtype, N, n, q_rank)
        cached = self._gdn3_sketch_cache.get(key)
        if cached is None:
            gen = torch.Generator(device=device).manual_seed(1234)
            cached = torch.randn(N, n, q_rank, dtype=dtype, device=device, generator=gen)
            self._gdn3_sketch_cache[key] = cached
        return cached

    def _compact_fast(self, A, Bk, U, Vb):
        """No-grad compaction equivalent to _compact_vec, using fast CUDA SVD when available."""
        N, R = A.shape[0], A.shape[1]
        a_v, a_k, b_v, b_k = self.a_v, self.a_k, self.b_v, self.b_k

        vecA = A.reshape(N, R, a_v * a_k)
        vecB = Bk.reshape(N, R, b_v * b_k)
        R_mat = torch.einsum('nra,nrb->nab', vecA, vecB)
        S_res = torch.einsum('nvp,nkp->nvk', U, Vb)
        S_res = (S_res.reshape(N, a_v, b_v, a_k, b_k)
                       .permute(0, 1, 3, 2, 4)
                       .reshape(N, a_v * a_k, b_v * b_k))
        R_mat = R_mat + S_res

        if self.compact_mode in {"sketch_noqr", "sketch"}:
            G = self._gdn3_sketch_matrix(N, b_v * b_k, R, R_mat.dtype, R_mat.device)
            Y_sketch = R_mat @ G
            q_scale = Y_sketch.square().sum(1).sqrt().clamp_min(1e-12)
            Q = Y_sketch / q_scale[:, None, :]
            Bm = Q.transpose(-2, -1) @ R_mat
            b_scale = Bm.square().sum(-1).sqrt().clamp_min(1e-12)
            A_svd = ((Q * b_scale[:, None, :])
                     .permute(0, 2, 1)
                     .reshape(N, R, a_v, a_k))
            B_svd = (Bm / b_scale[:, :, None]).reshape(N, R, b_v, b_k)
            A_new = self.slow_decay * A + (1 - self.slow_decay) * A_svd
            B_new = self.slow_decay * Bk + (1 - self.slow_decay) * B_svd
            return A_new, B_new, U, Vb, torch.zeros((), dtype=R_mat.dtype, device=R_mat.device)

        total_energy = (R_mat ** 2).sum()
        m, n = R_mat.shape[-2], R_mat.shape[-1]
        q_rank = min(R + 4, m, n)
        G = self._gdn3_sketch_matrix(N, n, q_rank, R_mat.dtype, R_mat.device)
        Y_sketch = R_mat @ G
        for _ in range(2):
            Y_sketch = R_mat @ (R_mat.transpose(-2, -1) @ Y_sketch)
        Q, _ = torch.linalg.qr(Y_sketch)
        Bm = Q.transpose(-2, -1) @ R_mat
        try:
            if Bm.is_cuda:
                Ub, Ss, Vh = torch.linalg.svd(Bm, full_matrices=False, driver='gesvda')
            else:
                Ub, Ss, Vh = torch.linalg.svd(Bm, full_matrices=False)
        except Exception:
            Ub, Ss, Vh = torch.linalg.svd(Bm.cpu(), full_matrices=False)
            Ub, Ss, Vh = Ub.to(Bm.device), Ss.to(Bm.device), Vh.to(Bm.device)
        Us = Q @ Ub
        Vs = Vh.transpose(-2, -1)
        kept = (Ss[:, :R] ** 2).sum()
        err = (total_energy - kept).clamp(min=0.0)

        A_svd = ((Us[:, :, :R] * Ss[:, None, :R])
                 .permute(0, 2, 1)
                 .reshape(N, R, a_v, a_k))
        B_svd = (Vs[:, :, :R]
                 .permute(0, 2, 1)
                 .reshape(N, R, b_v, b_k))
        A_new = self.slow_decay * A + (1 - self.slow_decay) * A_svd
        B_new = self.slow_decay * Bk + (1 - self.slow_decay) * B_svd
        return A_new, B_new, U, Vb, err

    def _kron_read_vec(self, A, Bk, U, Vb, x):
        """Batched Kronecker+residual read S*x for N chains.

        A [N,R,a_v,a_k], Bk [N,R,b_v,b_k], U [N,V,P], Vb [N,K,P], x [N,K].
        (A_r (x) B_r) x = vec(A_r @ X @ B_r^T) with X = x.reshape(a_k,b_k).
        """
        N = x.shape[0]
        X = x.reshape(N, self.a_k, self.b_k)
        AX = torch.einsum('nrvk,nkb->nrvb', A, X)          # [N,R,a_v,b_k]
        AXB = torch.einsum('nrvb,nrwb->nrvw', AX, Bk)      # [N,R,a_v,b_v]
        y = AXB.sum(1).reshape(N, self.a_v * self.b_v)     # [N,V]
        coeff = torch.einsum('nkp,nk->np', Vb, x)          # [N,P]  = Vb^T x
        y = y + torch.einsum('nvp,np->nv', U, coeff)       # [N,V]  += U (Vb^T x)
        return y

    def _stable_alpha_vec(self, c, eps: float = 1e-6):
        """Exact write coefficient alpha = (1-exp(-c))/c, Taylor-safe near 0."""
        z = c  # delta = 1.0
        small = z.abs() < eps
        safe_z = torch.where(small, torch.ones_like(z), z)
        exact = -torch.expm1(-safe_z) / safe_z
        series = 1.0 - z / 2.0 + z * z / 6.0
        return torch.where(small, series, exact)

    def _compact_vec(self, A, Bk, U, Vb, slow_decay=0.97):
        """Batched rearrangement-SVD compaction with two-timescale blending.
        
        Preserves 97% of old Kronecker state, blends 3% from SVD.
        Keeps residual buffer exact (not zeroed) for perfect recency cache.
        """
        N, R = A.shape[0], A.shape[1]
        a_v, a_k, b_v, b_k = self.a_v, self.a_k, self.b_v, self.b_k
        
        # Save old state for two-timescale blend
        A_old = A.clone()
        Bk_old = Bk.clone()
        
        vecA = A.reshape(N, R, a_v * a_k)
        vecB = Bk.reshape(N, R, b_v * b_k)
        R_mat = torch.einsum('nra,nrb->nab', vecA, vecB)          # [N, av*ak, bv*bk]
        S_res = torch.einsum('nvp,nkp->nvk', U, Vb)              # [N, V, K]
        S_res = (S_res.reshape(N, a_v, b_v, a_k, b_k)
                       .permute(0, 1, 3, 2, 4)
                       .reshape(N, a_v * a_k, b_v * b_k))
        R_mat = R_mat + S_res
        # Truncated top-R factorization via randomized SVD.  We only need the
        # leading R components (compaction is intentionally lossy and runs under
        # no_grad).  cuSOLVER's default SVD driver crashes on the near
        # rank-deficient R_mat of early windows, so we run a random-projection
        # range finder and a robust `gesvd` on the tiny projected matrix.
        total_energy = (R_mat ** 2).sum()
        m, n = R_mat.shape[-2], R_mat.shape[-1]
        q = min(R + 4, m, n)
        # Deterministic Gaussian sketch (fixed seed) so gradient-checkpoint
        # recomputation reproduces the forward pass exactly.
        gen = torch.Generator(device=R_mat.device).manual_seed(1234)
        G = torch.randn(N, n, q, dtype=R_mat.dtype, device=R_mat.device, generator=gen)
        Y = R_mat @ G                                 # [N, m, q]
        for _ in range(2):                            # power iterations for accuracy
            Y = R_mat @ (R_mat.transpose(-2, -1) @ Y)
        Q, _ = torch.linalg.qr(Y)                    # [N, m, q]  orthonormal range basis
        Bm = Q.transpose(-2, -1) @ R_mat             # [N, q, n]  projected matrix
        # top-q SVD of the tiny projected matrix, taken DIRECTLY on Bm.
        # The old code formed C = Bm @ Bm^T and eigh'd it, which squares the
        # condition number and made the near rank-deficient early-window chains
        # crash cuSOLVER (and even CPU LAPACK: "repeated eigenvalues"). It also
        # forced a per-window GPU->CPU sync. Direct batched SVD of the small
        # [N, q, n] matrix is robust to rank deficiency and stays on-GPU; a CPU
        # fallback remains purely as a safety net so a long run can never die.
        try:
            Ub, Ss, Vh = torch.linalg.svd(Bm, full_matrices=False)   # [N,q,q],[N,q],[N,q,n]
        except Exception:                            # rare degenerate batch
            Ub, Ss, Vh = torch.linalg.svd(Bm.cpu(), full_matrices=False)
            Ub, Ss, Vh = Ub.to(Bm.device), Ss.to(Bm.device), Vh.to(Bm.device)
        Us = Q @ Ub                                  # [N, m, q] left singular vectors of R_mat
        Vs = Vh.transpose(-2, -1)                    # [N, n, q] right singular vectors
        kept = (Ss[:, :R] ** 2).sum()
        # keep the discarded-energy diagnostic ON-GPU; the caller converts to a
        # python float once per forward instead of syncing every P tokens.
        err = (total_energy - kept).clamp(min=0.0)
        
        # Extract SVD-derived factors
        A_svd = torch.zeros_like(A)
        B_svd = torch.zeros_like(Bk)
        for r in range(R):
            A_svd[:, r] = (Us[:, :, r] * Ss[:, r:r + 1]).reshape(N, a_v, a_k)
            B_svd[:, r] = Vs[:, :, r].reshape(N, b_v, b_k)
        
        # TWO-TIMESCALE BLEND: 97% old + 3% SVD
        A_new = slow_decay * A_old + (1 - slow_decay) * A_svd
        B_new = slow_decay * Bk_old + (1 - slow_decay) * B_svd
        
        # Residual: keep exact (not zeroed) — perfect recency cache
        return A_new, B_new, U, Vb, err

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, T, D]
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,  # Accept extra kwargs from Qwen3.5 (cache_params, etc.)
    ) -> torch.Tensor:                   # [B, T, D]
        """
        Forward pass: Qwen3.5-compatible linear attention with GDN3 state.

        Flow:
          1. Qwen projections: in_proj_qkv -> q,k,v  |  in_proj_z -> gates
          2. Causal convolution: conv1d on qkv features
          3. Reshape to [B, T, H, M, K/V] for GDN3 MIMO processing
          4. Apply partial RoPE per lane
          5. GDN3 Kronecker-Residual MIMO recurrence (replaces SSM)
          6. Lane routing + aggregation
          7. Qwen norm + output gate + out_proj
        """
        B, T, D = hidden_states.shape
        H, M = self.H, self.M
        K, V = self.K, self.V

        # ==================================================================
        # 1. QWEN3.5 PROJECTIONS (preserved)
        # ==================================================================
        # Q,K,V projections: [B, T, D] -> [B, T, H*3*K]
        qkv = F.linear(hidden_states, self.in_proj_qkv)  # [B, T, H*3*K]

        # Gate projections: [B, T, D] -> [B, T, 2*D]
        z = F.linear(hidden_states, self.in_proj_z)  # [B, T, 2*D]

        # State factor projections (used for GDN3 initialization hints)
        proj_a = F.linear(hidden_states, self.in_proj_a)   # [B, T, H]
        proj_b = F.linear(hidden_states, self.in_proj_b)   # [B, T, H]

        # ==================================================================
        # 2. CAUSAL CONVOLUTION (preserved)
        # ==================================================================
        # conv1d expects [B, channels, T]. Left-pad for causality.
        qkv_padded = F.pad(qkv.transpose(1, 2), (self.conv_kernel_size - 1, 0))  # [B, H*3*K, T+conv_k-1]
        qkv_conv = self.conv1d(qkv_padded)  # [B, H*3*K, T]
        qkv_conv = qkv_conv.transpose(1, 2)  # [B, T, H*3*K]

        # Split into q, k, v
        qkv_conv = qkv_conv.view(B, T, H, 3, K)  # [B, T, H, 3, K]
        q = qkv_conv[:, :, :, 0, :]  # [B, T, H, K]
        k = qkv_conv[:, :, :, 1, :]  # [B, T, H, K]
        v_raw = qkv_conv[:, :, :, 2, :]  # [B, T, H, K]

        # Blend in coproduct bilinear features (Hopf-inspired channel generation)
        q, k, v_raw = self._generate_coproduct_channels(q, k, v_raw, hidden_states)

        # Value projection: use v from qkv + z for gating
        z_split = z.view(B, T, 2, D)  # [B, T, 2, D]
        output_gate_raw = z_split[:, :, 0, :]  # [B, T, D]

        # Project value to V dimensions (may differ from K)
        v = v_raw  # Already [B, T, H, V], K == V in Qwen3.5

        # ==================================================================
        # 3. RESHAPE FOR GDN3 MIMO PROCESSING
        # ==================================================================
        # Expand from [B, T, H, K] to [B, T, H, M, K] (replicate across lanes)
        # Lane diversity comes from different GDN3 state per lane
        q_expanded = q.unsqueeze(3).expand(-1, -1, -1, M, -1)  # [B, T, H, M, K]
        k_expanded = k.unsqueeze(3).expand(-1, -1, -1, M, -1)  # [B, T, H, M, K]
        v_expanded = v.unsqueeze(3).expand(-1, -1, -1, M, -1).view(B, T, H, M, K)

        # For v, we need [V] not [K] dimensions
        v_expanded = v_expanded.view(B, T, H, M, K)
        if V != K:
            # Adjust value dimension
            v_expanded = v_expanded[..., :V] if V < K else F.pad(v_expanded, (0, V - K))

        # ==================================================================
        # 4. GDN3 GATES AND DECAY
        # ==================================================================
        # Erase gates from W_b projection + SSM proj_a influence
        b_proj = F.linear(hidden_states, self.W_b.reshape(-1, D)).view(B, T, H, M, K)
        # Blend with SSM projection pattern
        proj_a_expanded = proj_a.unsqueeze(3).expand(-1, -1, -1, M).unsqueeze(-1).expand(-1, -1, -1, -1, K)
        b_gates = torch.sigmoid(b_proj + proj_a_expanded * 0.1)  # [B, T, H, M, K]

        # Write gates from W_w projection + SSM proj_b influence
        w_proj = F.linear(hidden_states, self.W_w.reshape(-1, D)).view(B, T, H, M, V)
        proj_b_expanded = proj_b.unsqueeze(3).expand(-1, -1, -1, M).unsqueeze(-1).expand(-1, -1, -1, -1, V)
        w_gates = torch.sigmoid(w_proj + proj_b_expanded * 0.1)  # [B, T, H, M, V]

        # Braided decay (multi-timescale)
        # W_decay: [H, M, tau=4, D], hidden_states: [B, T_seq, D]
        # Output: [B, T_seq, H, M, tau]
        decay_proj = F.linear(hidden_states, self.W_decay.reshape(-1, D)).view(B, T, H, M, 4)
        g_raw = F.softplus(decay_proj).clamp(max=5.0)
        rates = self.base_decay_rates.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        decay_per_channel = torch.exp(-g_raw * rates)
        # Forgetting floor: when decay_proj is very negative, softplus->0 and
        # decay->exp(0)=1.0 (zero forgetting), so the Kronecker+residual state
        # accumulates unbounded across compaction windows and the forward blows
        # up (the observed progressive-divergence / skip-storm failure). Clamp
        # strictly below 1 so every channel forgets slightly -> bounded state.
        # Applied here (upstream of both recurrence paths) so parity is preserved.
        decay_factors = decay_per_channel.mean(dim=-1).clamp(max=self.decay_clamp)  # [B, T, H, M]

        # ==================================================================
        # 5. PARTIAL RoPE PER LANE
        # ==================================================================
        q_rope = self._apply_partial_rope(q_expanded, T)
        k_rope = self._apply_partial_rope(k_expanded, T)

        # ==================================================================
        # 6. GDN3 KRONECKER-RESIDUAL MIMO RECURRENCE
        # ==================================================================
        # This is the core replacement: SSM state -> GDN3 Kr-MIMO state
        gdn3_output = self._gdn3_recurrent_state(
            q_rope, k_rope, v_expanded,
            b_gates, w_gates, decay_factors
        )  # [B, T, H, M, V]

        # ==================================================================
        # 7. LANE ROUTING AND AGGREGATION
        # ==================================================================
        router_logits = self.router_proj(hidden_states)  # [B, T, H*M]
        router_logits = router_logits.view(B, T, H, M)  # [B, T, H, M]
        router_weights = F.softmax(router_logits, dim=-1)  # [B, T, H, M]

        # Route: weighted sum across lanes
        routed = (router_weights.unsqueeze(-1) * gdn3_output).sum(dim=3)  # [B, T, H, V]

        # Aggregate heads: reshape [B, T, H, V] -> [B, T, H*V] -> project to [B, T, D]
        aggregated = routed.view(B, T, H * V)  # [B, T, 2048]

        # Per-head norm on the head dimension before aggregation
        # Apply norm to each head's V-dim features
        routed_normed = F.normalize(routed, p=2, dim=-1, eps=1e-6) * torch.sqrt(torch.tensor(float(V)))
        routed_normed = routed_normed * self.norm  # Scale by learned norm param
        aggregated = routed_normed.view(B, T, H * V)

        # ==================================================================
        # 8. QWEN3.5 OUTPUT GATING AND PROJECTION (preserved)
        # ==================================================================
        # Output gate (silu activation as in Qwen3.5)
        output_gate = F.silu(output_gate_raw)  # [B, T, D]

        # Concatenate attention output + gate input for out_proj
        # Project aggregated features back to D
        if aggregated.shape[-1] != D:
            attn_output = self._agg_proj(aggregated)  # [B, T, D]
        else:
            attn_output = aggregated

        # Apply output gate
        gated_output = attn_output * output_gate  # [B, T, D]

        # Final out_proj (Qwen3.5 style: takes [attn_out, raw_input] concatenated)
        out_input = torch.cat([gated_output, hidden_states], dim=-1)  # [B, T, 2*D]
        final_output = self.out_proj(out_input)  # [B, T, D]

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(-1)
            final_output = final_output * attention_mask

        return final_output


# =============================================================================
# UPGRADE MANAGER — Applies GDN3 to Qwen3.5 Model
# =============================================================================

class GDN3UpgradeManager:
    """Manages GDN3 upgrade of Qwen3.5 model.

    Replaces ONLY linear attention layers with GDN3LinearAttn.
    Full attention layers are preserved completely.
    """

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or model.config
        self.upgraded_layers = []

    def apply_upgrade(self):
        """Replace linear attention layers with GDN3LinearAttn."""
        layer_types = self.config.to_dict().get('layer_types', [])

        print(f"\nGDN3 Upgrade: Scanning {len(layer_types)} layers...")

        for idx, layer_type in enumerate(layer_types):
            if layer_type == 'linear_attention':
                layer = self.model.model.layers[idx]

                # Create GDN3 replacement
                gdn3_attn = GDN3LinearAttn(self.config, layer_idx=idx)

                # Load Qwen3.5 weights (warm-start)
                layer_state = {}
                for name, param in layer.linear_attn.named_parameters():
                    layer_state[f"linear_attn.{name}"] = param.data.clone()

                # Also handle buffers
                for name, buf in layer.linear_attn.named_buffers():
                    layer_state[f"linear_attn.{name}"] = buf.clone()

                gdn3_attn.load_qwen_weights(layer_state, idx)

                # Replace
                layer.linear_attn = gdn3_attn.to(self.model.device)
                self.upgraded_layers.append(idx)

                print(f"  Layer {idx:2d}: linear_attention -> GDN3 Kr-MIMO [OK]")
            elif layer_type == 'full_attention':
                print(f"  Layer {idx:2d}: full_attention [PRESERVED]")

        print(f"\nUpgrade complete: {len(self.upgraded_layers)} layers upgraded")
        return self.upgraded_layers

    def save(self, output_dir: str):
        """Save upgraded model."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        self.model.config.gdn3_upgrade = {
            'version': '2.0.0-component-level-two-timescale',
            'upgraded_layers': self.upgraded_layers,
            'strategy': 'ssm-recurrence-replacement',
            'preserved_components': ['in_proj_qkv', 'in_proj_z', 'conv1d', 'norm', 'out_proj'],
            'replaced_components': ['A_log', 'dt_bias', 'ssm_recurrence'],
            'gdn3_config': {
                'num_lanes': 4,
                'kron_rank': 4,
                'residual_rank': 16,
                'a_k': 16, 'b_k': 8,
                'a_v': 16, 'b_v': 8,
                'compaction': 'two_timescale',
                'slow_decay': 0.97,  # 97% old + 3% SVD blend
            },
        }

        self.model.save_pretrained(output_dir)
        print(f"Saved upgraded model to: {output_dir}")

    def get_param_count(self):
        """Compare parameter counts."""
        return sum(p.numel() for p in self.model.parameters())


def upgrade_qwen35_gdn3(model_path: str, output_dir: str, device: str = 'cuda'):
    """Convenience function to upgrade Qwen3.5 with GDN3.

    Args:
        model_path: Path to Qwen3.5 model directory
        output_dir: Where to save the upgraded model
        device: Device to use for upgrade
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print("=" * 70)
    print("GDN3 Upgrade: Qwen3.5 Linear Attention -> Kronecker-Residual MIMO")
    print("=" * 70)

    # Load model
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    original_params = sum(p.numel() for p in model.parameters())
    print(f"\nOriginal params: {original_params:,}")

    # Apply upgrade
    manager = GDN3UpgradeManager(model, config)
    manager.apply_upgrade()

    new_params = manager.get_param_count()
    print(f"Upgraded params:   {new_params:,}")
    print(f"Delta:             {new_params - original_params:+,}")

    # Save
    manager.save(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nUpgrade complete!")
    print(f"  Output: {output_dir}")
    print(f"  Upgraded layers: {manager.upgraded_layers}")

    return model, tokenizer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GDN3 Qwen3.5 Upgrade')
    parser.add_argument('--input', type=str, default='./qwen35_gdn3_adapted',
                        help='Path to Qwen3.5 model directory')
    parser.add_argument('--output', type=str, default='./qwen35_gdn3_upgraded_v2',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()
    upgrade_qwen35_gdn3(args.input, args.output, args.device)
