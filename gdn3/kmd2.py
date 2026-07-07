"""KMD-2 (Kronecker-MIMO Delta-2) — Fable's idea, implemented as a Qwen3.5
linear-attention drop-in for the robust MQAR proxy.

Source: ~/gdn3_fable/fable_idea.txt (Component A — the per-token rank-r MIMO
delta rule with a compact-WY / RLS T-factor). This is a DIFFERENT architecture
from the original GDN3 (Kronecker-residual state + SVD compaction + coproduct)
that the 41-experiment auto-research tested; KMD-2 was proposed later by Claude
Fable and reportedly hit ~17% MQAR recall in a lost web session. This file lets
us test whether that result survives the full frozen-backbone proxy.

Recurrence (per head, state S in R^{dv x dk}), per token t:
    S <- S * Diag(a_t)                          # channel-wise decay (key axis)
    S <- S - (S @ Ktil^T) @ T_t @ Ktil          # block erase, Ktil = B_t ⊙ K_t
    S <- S + (W_t ⊙ V_t) @ K_t^T                 # rank-r gated write
    y_t = S @ q_t                                # post-update read
where each token supplies r slots (K_t,V_t,B_t,W_t are r-column blocks) and
    T_t = (eps*I + Ktil Ktil^T)^{-1}   in R^{r x r}
is the RLS "exact multi-association overwrite" T-factor (Fable §2). r=1 reduces
to a single gated-delta write per token.

KEY DIFFERENCE vs original GDN3: q/k/v/gates come from KMD-2's OWN trainable
projections of the hidden state — so the query CAN learn to align to stored keys
(an induction head), the exact thing the frozen-Qwen-q/k original GDN3 could not
do. The output path (in_proj_z gate + out_proj) reuses the frozen Qwen
components, so this stays a faithful, robust drop-in — not a toy single layer.
"""
from __future__ import annotations

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


class KMD2LinearAttn(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        D = config.hidden_size                       # 1024

        # KMD-2 internal dims (kept modest so trainable param count ~ matches the
        # original GDN3 trainable set and fits one 16GB GPU with checkpointing).
        self.H = _env_int("GDN3_KMD2_H", 16)         # heads
        self.dk = _env_int("GDN3_KMD2_DK", 64)       # key/query head dim
        self.dv = _env_int("GDN3_KMD2_DV", 64)       # value head dim
        self.r = _env_int("GDN3_KMD2_R", 4)          # MIMO slots per token
        self.D = D
        H, dk, dv, r = self.H, self.dk, self.dv, self.r

        # RLS T-factor regularizer. Larger -> closer to a plain gated write
        # (more stable); smaller -> sharper exact overwrite.
        self.rls_eps = float(os.environ.get("GDN3_KMD2_EPS", "0.5"))

        # ---- Trainable KMD-2 projections (NOT frozen by the proxy) ----
        # query (one per token per head), and r slots each of K, V, erase(B),
        # write(W) gates, plus channel-wise decay a.
        self.q_proj = nn.Linear(D, H * dk, bias=False)
        self.k_slots = nn.Linear(D, H * r * dk, bias=False)
        self.v_slots = nn.Linear(D, H * r * dv, bias=False)
        self.bgate = nn.Linear(D, H * r * dk, bias=True)   # erase gate logits
        self.wgate = nn.Linear(D, H * r * dv, bias=True)   # write gate logits
        self.decay = nn.Linear(D, H * dk, bias=True)       # channel-wise decay logits
        self.agg = nn.Linear(H * dv, D, bias=False)        # heads -> model dim
        # trainable output scale on the per-head read (avoid substring "norm"
        # so the proxy does not freeze it)
        self.read_scale = nn.Parameter(torch.ones(H, dv))

        # ---- Reused FROZEN Qwen output path (loaded via load_qwen_weights) ----
        self.in_proj_z = nn.Parameter(torch.empty(2 * D, D))   # output gate source
        self.out_proj = nn.Linear(2 * D, D, bias=False)

        self._init_weights()
        self._maybe_probe_init(layer_idx)

    _PROBE_CACHE = {}

    def _maybe_probe_init(self, layer_idx: int):
        """Optionally seed q_proj / k_slots slot 0 from offline-trained InfoNCE
        alignment probes (GDN3_KMD2_QK_INIT=path.pt with {layer: {Wq,Wk}}).
        Gives the read partial q->k alignment AT INIT so CE only has to exploit
        it, not discover it (CE alone provably never does — see probe logs)."""
        path = os.environ.get("GDN3_KMD2_QK_INIT", "")
        if not path:
            return
        if path not in KMD2LinearAttn._PROBE_CACHE:
            KMD2LinearAttn._PROBE_CACHE[path] = torch.load(path, map_location="cpu")
        probes = KMD2LinearAttn._PROBE_CACHE[path]
        if layer_idx not in probes:
            print(f"  [KMD-2] no probe for layer {layer_idx}; keeping random init")
            return
        Wq = probes[layer_idx]["Wq"].T.contiguous()   # [dk, D]
        Wk = probes[layer_idx]["Wk"].T.contiguous()
        H, dk, r = self.H, self.dk, self.r
        if Wq.shape != (dk, self.D):
            print(f"  [KMD-2] probe dim mismatch {tuple(Wq.shape)} != ({dk},{self.D}); skip")
            return
        with torch.no_grad():
            for h in range(H):
                noise_q = torch.randn_like(Wq) * 0.002   # break head symmetry
                noise_k = torch.randn_like(Wk) * 0.002
                self.q_proj.weight[h * dk:(h + 1) * dk].copy_(Wq + noise_q)
                # slot 0 of each head gets the aligned key; slots 1..r-1 stay random
                s0 = (h * r + 0) * dk
                self.k_slots.weight[s0:s0 + dk].copy_(Wk + noise_k)

    def _init_weights(self):
        for lin in (self.q_proj, self.k_slots, self.v_slots, self.agg):
            nn.init.normal_(lin.weight, std=0.02)
        for lin in (self.bgate, self.wgate):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)                # gates start ~0.5 (sigmoid(0))
        nn.init.normal_(self.decay.weight, std=0.02)
        # decay bias -> a ~ sigmoid(6.0) ~ 0.9975 at init. v1 used 2.5 (a~0.924),
        # giving a ~13-token memory horizon — bindings written >50 tokens before
        # the query were erased before it arrived, so the retrieval gradient never
        # existed. 0.9975 retains ~61% over 200 tokens (the MQAR context span).
        self.decay_bias_init = float(os.environ.get("GDN3_KMD2_DECAY_BIAS", "6.0"))
        nn.init.constant_(self.decay.bias, self.decay_bias_init)
        nn.init.zeros_(self.in_proj_z)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    # The proxy's GDN3UpgradeManager calls this to warm-start from Qwen weights.
    def load_qwen_weights(self, state_dict: Dict[str, torch.Tensor], layer_idx: int):
        # keys look like "linear_attn.in_proj_z.weight" / "linear_attn.out_proj.weight".
        # Load the frozen Qwen output path; ignore SSM/qkv/conv (KMD-2 uses its own).
        want = {"in_proj_z.weight": self.in_proj_z, "out_proj.weight": self.out_proj.weight}
        loaded = []
        for key, val in state_dict.items():
            for suffix, param in want.items():
                if key.endswith(suffix) and tuple(val.shape) == tuple(param.shape):
                    param.data.copy_(val.to(param.device, param.dtype))
                    loaded.append(suffix)
        missing = set(want) - set(loaded)
        if missing:
            print(f"  [KMD-2] WARNING layer {layer_idx}: did not load {missing}")

    def _scan(self, q, K, V, Bg, Wg, a):
        """Vectorized per-token rank-r block-Householder delta scan.
        q [B,T,H,dk]; K,Bg [B,T,H,r,dk]; V,Wg [B,T,H,r,dv]; a [B,T,H,dk].
        Returns y [B,T,H,dv]. Chains N=B*H run in parallel; time is the loop.
        """
        B, T, H, dk = q.shape
        r, dv = K.shape[3], V.shape[-1]
        N = B * H
        dtype = torch.float32
        device = q.device

        def flat(x, *tail):
            # [B,T,H,...] -> [T, N, ...]
            return x.permute(1, 0, 2, *range(3, x.dim())).reshape(T, N, *tail).to(dtype)
        q_ = flat(q, dk)
        K_ = flat(K, r, dk); V_ = flat(V, r, dv)
        Bg_ = flat(Bg, r, dk); Wg_ = flat(Wg, r, dv)
        a_ = flat(a, dk)

        S = torch.zeros(N, dv, dk, dtype=dtype, device=device)
        eyeR = torch.eye(r, dtype=dtype, device=device).unsqueeze(0)
        outs = []
        for t in range(T):
            S = S * a_[t].unsqueeze(1)                     # decay key columns
            Kt, Vt = K_[t], V_[t]                          # [N,r,dk],[N,r,dv]
            Ktil = Bg_[t] * Kt                             # gated erase keys [N,r,dk]
            SK = torch.bmm(S, Ktil.transpose(1, 2))        # [N,dv,r]
            Gram = torch.bmm(Ktil, Ktil.transpose(1, 2))   # [N,r,r]
            Tt = torch.linalg.solve(eyeR * self.rls_eps + Gram, eyeR.expand(N, r, r))
            S = S - torch.bmm(torch.bmm(SK, Tt), Ktil)     # block erase
            WV = (Wg_[t] * Vt).transpose(1, 2)             # [N,dv,r]
            S = S + torch.bmm(WV, Kt)                       # rank-r write
            yt = torch.bmm(S, q_[t].unsqueeze(2)).squeeze(2)   # [N,dv] post-update read
            outs.append(yt)
        Y = torch.stack(outs, 0).reshape(T, B, H, dv).permute(1, 0, 2, 3).contiguous()
        return Y

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        B, T, D = hidden_states.shape
        H, dk, dv, r = self.H, self.dk, self.dv, self.r
        x = hidden_states

        q = self.q_proj(x).view(B, T, H, dk)
        # L2-normalize query and keys (DeltaNet-style stability)
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        K = self.k_slots(x).view(B, T, H, r, dk)
        K = F.normalize(K, p=2, dim=-1, eps=1e-6)
        V = self.v_slots(x).view(B, T, H, r, dv)
        Bg = torch.sigmoid(self.bgate(x).view(B, T, H, r, dk))
        Wg = torch.sigmoid(self.wgate(x).view(B, T, H, r, dv))
        a = torch.sigmoid(self.decay(x).view(B, T, H, dk)).clamp(max=0.999)

        Y = self._scan(q, K, V, Bg, Wg, a)                 # [B,T,H,dv]
        Y = Y * self.read_scale                            # per-head learned scale
        agg = self.agg(Y.reshape(B, T, H * dv))            # [B,T,D]

        # Frozen Qwen output path: gate then out_proj([gated, x])
        z = F.linear(x, self.in_proj_z).view(B, T, 2, D)
        output_gate = F.silu(z[:, :, 0, :])
        gated = agg * output_gate
        out = self.out_proj(torch.cat([gated, x], dim=-1))
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(-1)
            out = out * attention_mask
        return out
