"""KMD-2 native mode: warm-start at the GDN-2 point (fable_idea.txt §7).

Drop-in for Qwen3.5's GatedDeltaNet that IS the native layer at init — every
native parameter is loaded warm and the recurrence is mathematically identical
to `torch_recurrent_gated_delta_rule` — plus KMD-2's new degrees of freedom
initialized at IDENTITY so training can open them without breaking the teacher:

  * rotation      : cumulative data-dependent 2x2 rotations on q/k
                    (init ~1.2e-4 rad/token => identity)
  * output MIMO   : r_out query slots as per-slot scale vectors on the shared q
                    (Mamba-3 param-cheap widening; init scales=0, out_mix=one-hot)
  * channel decay : per-channel offsets on the per-head decay logit (init 0)
  * decoupled write: offset on the write-side beta logit (init 0 = GDN2-coupled)

Verification gates (run research/runs_fable/verify_native_init.py):
  init per-layer MSE ~ 0, init KL ~ 0, init RULER ~ teacher.
"""
from __future__ import annotations

import math
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


_FAST_SCAN = os.environ.get("GDN3_FAST_SCAN", "0") == "1"


class KMD2NativeAttn(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        D = config.hidden_size
        self.H = config.linear_num_value_heads          # 16 (v-heads = k-heads here)
        self.dk = config.linear_key_head_dim            # 128
        self.dv = config.linear_value_head_dim          # 128
        self.key_dim = self.dk * config.linear_num_key_heads
        self.value_dim = self.dv * self.H
        self.conv_k = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.r_out = _env_int("GDN3_KMD2_ROUT", 4)
        H, dk = self.H, self.dk

        # ---- native parameters (warm-loaded) ----
        conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = nn.Linear(D, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(D, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(D, self.H, bias=False)
        self.in_proj_a = nn.Linear(D, self.H, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, self.conv_k,
                                groups=conv_dim, bias=False,
                                padding=self.conv_k - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.H))
        self.A_log = nn.Parameter(torch.zeros(self.H))
        self.norm = Qwen3_5RMSNormGated(self.dv, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, D, bias=False)

        # ---- KMD-2 new degrees of freedom (identity at init) ----
        self.rot_proj = nn.Linear(D, H * (dk // 2), bias=True)
        nn.init.zeros_(self.rot_proj.weight)
        nn.init.constant_(self.rot_proj.bias, -9.0)      # softplus ~ 1.2e-4 rad/tok
        if self.r_out > 1:
            self.q_slot_scale = nn.Parameter(torch.zeros(H, self.r_out, dk))
            mix = torch.zeros(H, self.r_out); mix[:, 0] = 1.0
            self.out_mix = nn.Parameter(mix)             # one-hot slot 0 = identity
        self.decay_chan = nn.Parameter(torch.zeros(H, dk))   # per-channel decay offset
        self.bw_off = nn.Parameter(torch.zeros(H))           # write-beta decouple offset

    # GDN3UpgradeManager hands us the native layer's state dict.
    def load_qwen_weights(self, state_dict: Dict[str, torch.Tensor], layer_idx: int):
        own = dict(self.named_parameters())
        loaded = 0
        for key, val in state_dict.items():
            name = key.split("linear_attn.", 1)[-1]
            if name in own and tuple(own[name].shape) == tuple(val.shape):
                own[name].data.copy_(val.to(own[name].dtype))
                loaded += 1
        expected = ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight",
                    "in_proj_a.weight", "conv1d.weight", "dt_bias", "A_log",
                    "norm.weight", "out_proj.weight")
        missing = [n for n in expected if n not in {k.split('linear_attn.',1)[-1]
                                                    for k in state_dict}]
        if missing:
            print(f"  [KMD-2 native] layer {layer_idx}: MISSING warm weights {missing}")

    def _scan(self, q, k, v, g, beta_e, beta_w):
        """GDN2-exact gated delta scan + identity-init KMD-2 DOF.
        q [B,T,H,r_out,dk] (scaled, rotated); k [B,T,H,dk]; v [B,T,H,dv];
        g [B,T,H,dk] per-channel decay (native scalar + offsets, exp'd);
        beta_e/beta_w [B,T,H]. Returns y [B,T,H,dv]."""
        if _FAST_SCAN and q.is_cuda:
            # chunk-parallel Triton+compile kernel (A/B kernel search winner);
            # numerically equivalent to the reference loop below (bench-gated
            # fwd relMSE<2e-3, grad<1e-2). ~80x faster fwd+bwd.  CUDA-only:
            # CPU callers (preflight gate probes) take the reference loop.
            from gdn3.kmd2_fast_scan import scan as _fast_scan
            out_mix = self.out_mix if self.r_out > 1 else None
            return _fast_scan(q, k, v, g, beta_e, beta_w, out_mix)
        B, T, H = k.shape[0], k.shape[1], k.shape[2]
        dk, dv, r_out = self.dk, self.dv, self.r_out
        N = B * H

        def flat(x, *tail):
            return x.permute(1, 0, 2, *range(3, x.dim())).reshape(T, N, *tail).float()
        q_ = flat(q, r_out, dk); k_ = flat(k, dk); v_ = flat(v, dv)
        g_ = flat(g, dk); be_ = flat(beta_e); bw_ = flat(beta_w)
        if r_out > 1:
            mixw = self.out_mix[None].expand(B, -1, -1).reshape(N, 1, r_out).float()

        S = torch.zeros(N, dk, dv, dtype=torch.float32, device=k.device)
        outs = []
        for t in range(T):
            S = S * g_[t].unsqueeze(-1)                        # decay (key rows)
            kt = k_[t]                                          # [N,dk]
            kv_mem = torch.bmm(kt.unsqueeze(1), S).squeeze(1)   # S^T k -> [N,dv]
            # erase uses beta_e, write uses beta_w (equal at init = native coupled)
            S = S - torch.bmm(kt.unsqueeze(2), (be_[t].unsqueeze(-1) * kv_mem).unsqueeze(1))
            S = S + torch.bmm(kt.unsqueeze(2), (bw_[t].unsqueeze(-1) * v_[t]).unsqueeze(1))
            yt = torch.bmm(q_[t], S)                            # [N,r_out,dv]
            yt = (yt * mixw.transpose(1, 2)).sum(1) if r_out > 1 else yt.squeeze(1)
            outs.append(yt)
        return torch.stack(outs, 0).reshape(T, B, H, dv).permute(1, 0, 2, 3)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask=None, **kwargs) -> torch.Tensor:
        B, T, D = hidden_states.shape
        H, dk, dv = self.H, self.dk, self.dv
        x = hidden_states

        mixed = self.in_proj_qkv(x).transpose(1, 2)
        mixed = F.silu(self.conv1d(mixed)[:, :, :T]).transpose(1, 2)   # native conv+SiLU
        query, key, value = torch.split(
            mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = F.normalize(query.reshape(B, T, H, dk).float(), dim=-1, eps=1e-6)
        k = F.normalize(key.reshape(B, T, H, dk).float(), dim=-1, eps=1e-6)
        v = value.reshape(B, T, H, dv).float()
        q = q * (dk ** -0.5)                                    # native query scale

        z = self.in_proj_z(x)
        b = self.in_proj_b(x).float()
        a = self.in_proj_a(x).float()

        beta_e = torch.sigmoid(b)                               # native coupled beta
        beta_w = torch.sigmoid(b + self.bw_off)                 # = beta_e at init
        g_head = -self.A_log.float().exp() * F.softplus(a + self.dt_bias.float())
        g = (g_head.unsqueeze(-1) + self.decay_chan).exp()      # [B,T,H,dk], native at init
        g = g.clamp(max=1.0)

        # rotation (identity at init) applied to q and k consistently
        theta = F.softplus(self.rot_proj(x)).view(B, T, H, dk // 2).float()
        Theta = theta.cumsum(dim=1)
        cos, sin = Theta.cos(), Theta.sin()
        def rope(zz, c, s):
            z1, z2 = zz[..., :dk // 2], zz[..., dk // 2:]
            return torch.cat([z1 * c - z2 * s, z1 * s + z2 * c], dim=-1)
        k = rope(k, cos, sin)
        # output-MIMO query slots: shared q scaled per slot (identity at slot 0)
        if self.r_out > 1:
            qs = q.unsqueeze(3) * (1.0 + self.q_slot_scale)[None, None]
        else:
            qs = q.unsqueeze(3)
        qs = rope(qs, cos.unsqueeze(-2), sin.unsqueeze(-2))

        y = self._scan(qs, k, v, g, beta_e, beta_w)             # [B,T,H,dv]

        y = self.norm(y.reshape(-1, dv).to(z.dtype), z.reshape(-1, dv))
        y = y.reshape(B, T, self.value_dim)
        out = self.out_proj(y)
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(-1)
            out = out * attention_mask
        return out
