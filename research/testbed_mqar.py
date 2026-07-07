#!/usr/bin/env python3
"""From-scratch MQAR testbed for KMD-2 / GDN-MIMO architecture validation.

Why this exists (see KMD2_STATUS.md): the frozen-Qwen CE-only proxy provably
cannot teach retrieval to ANY drop-in recurrence (45 runs, causal probe chain).
The architecture questions — does rank-r MIMO help? does the state survive
compaction? — need a testbed where CE actually works: everything trainable,
atomic-token MQAR (Zoology-style), enough episodes to cross the induction
phase transition. Minutes per run, so ablations are cheap.

Task (atomic tokens, no NL): sequences of  k1 v1 k2 v2 ... kN vN  Q k_i -> v_i
with n_query queries at the end. CE on the value-answer positions only.
Keys and values are disjoint token ranges; every episode resamples the k->v map,
so the model MUST do in-context retrieval (no memorization possible).

Model: emb -> L x [ KMD2Block(mixer + SwiGLU MLP, pre-LN) ] -> unemb (tied off).
Mixer = per-token rank-r block-Householder delta (RLS T-factor), the same math
as gdn3/kmd2.py's _scan, plus optional:
  --kron          Kronecker key expansion (B1: k = k1 (x) k2, dk -> dk^2 state cols)
  --compact P R   GDN3-style state compression: every P tokens, SVD-truncate the
                  per-head state S to rank R (two-timescale blend, slow_decay).
                  Tests "MIMO + compaction" directly: recall vs compression.

Usage:
  testbed_mqar.py --r 4 --steps 3000 --out runs_fable/tb_r4.json --device cuda:1
"""
from __future__ import annotations
import argparse, json, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- data -----------------------------

def make_batch(rng, B, n_pairs, n_query, n_keys_vocab, n_vals_vocab, device):
    """Atomic MQAR. Token map: 0=pad, 1=Q, keys: 2..2+K, vals: 2+K..2+K+V.
    Returns ids [B,T], loss_mask [B,T] (1 on answer positions), T."""
    K0 = 2
    V0 = 2 + n_keys_vocab
    T = 2 * n_pairs + 2 * n_query
    ids = torch.zeros(B, T, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        perm = torch.randperm(n_keys_vocab, generator=rng)[:n_pairs]
        vals = torch.randint(0, n_vals_vocab, (n_pairs,), generator=rng)
        pos = 0
        for i in range(n_pairs):
            ids[b, pos] = K0 + perm[i]; ids[b, pos + 1] = V0 + vals[i]; pos += 2
        qi = torch.randint(0, n_pairs, (n_query,), generator=rng)
        for i in range(n_query):
            ids[b, pos] = 1                        # Q marker... actually use key directly
            ids[b, pos] = K0 + perm[qi[i]]         # query = repeat the key token
            ids[b, pos + 1] = V0 + vals[qi[i]]     # answer = its value
            mask[b, pos + 1] = True                # predict value FROM the key position
            pos += 2
    return ids.to(device), mask.to(device), T

# ----------------------------- model -----------------------------

class KMD2Mixer(nn.Module):
    """Per-token rank-r block-Householder delta with RLS T-factor.
    Same recurrence as gdn3/kmd2.py, self-contained for the testbed.
    Optional Kronecker keys (dk = m*n via two small factors) and periodic
    SVD compaction of the per-head state."""

    CONV_K = 4   # short causal conv, the standard linear-attn ingredient:
                 # without it, the k written at the VALUE position cannot contain
                 # the KEY's identity (adjacent-token binding is impossible) and
                 # MQAR collapses to the "any context value" shortcut (CE=ln 16).

    def __init__(self, d, H=4, dk=32, dv=32, r=4, eps=0.5,
                 kron=False, compact_P=0, compact_R=0, slow_decay=0.97,
                 use_conv=True, compact_ste=False, trap=False,
                 rot=False, r_out=1, rope_mod=False):
        super().__init__()
        self.H, self.dk, self.dv, self.r, self.eps = H, dk, dv, r, eps
        self.kron = kron
        self.compact_P, self.compact_R, self.slow_decay = compact_P, compact_R, slow_decay
        self.compact_ste = compact_ste
        self.use_conv = use_conv
        # Mamba-3-style exponential-trapezoidal write (arXiv:2603.15569 Eq.5/6):
        # write_t = lam_t * O_t + (1-lam_t) * a_t (.) O_{t-1}, lam_t data-dependent
        # per head. Equivalent to a width-2 data-dependent conv on the state-input
        # INSIDE the recurrence — the principled replacement for the external
        # short conv (their 440M ablation: bias+trap obviates conv entirely).
        # Complex/rotational state transition (Mamba-3 §complex-SSM): S evolves
        # as S·(a_t ⊙ R_t) with data-dependent 2x2 rotation blocks. Implemented
        # via their RoPE-trick equivalence: apply CUMULATIVE rotation Θ_t=Σθ_i
        # to k_t and q_t, so k_j·q_t sees the relative angle Θ_t-Θ_j.
        self.rot = rot
        if rot:
            self.rot_proj = nn.Linear(d, H * (dk // 2), bias=True)
            nn.init.zeros_(self.rot_proj.weight)
            # softplus(-4.6) ~ 0.01 rad/token at init: gentle, learnable phase
            nn.init.constant_(self.rot_proj.bias, -4.6)
        # Data-dependent RoPE (author's variant): FIXED RoPE frequency ladder,
        # learned per-token scalar rate m_t -> data-dependent position
        # p_t = cumsum(m), angle = p_t * inv_freq. Vanilla RoPE is m == 1.
        self.rope_mod = rope_mod
        if rope_mod:
            assert not rot, "rot and rope_mod are exclusive rotation schemes"
            self.mod_proj = nn.Linear(d, H, bias=True)
            nn.init.zeros_(self.mod_proj.weight)
            nn.init.constant_(self.mod_proj.bias, 0.5413)   # softplus ~ 1.0
            half = dk // 2
            self.register_buffer(
                "rope_inv_freq",
                10000.0 ** (-torch.arange(half, dtype=torch.float32) / half))
        # Mamba-3-style output widening: r_out query slots read the state, then
        # a learned per-head mix recombines (their C_t ∈ R^{N x R} output MIMO).
        self.r_out = r_out
        if r_out > 1:
            self.out_mix = nn.Parameter(torch.full((H, r_out), 1.0 / r_out))
        self.trap = trap
        if trap:
            self.lam = nn.Linear(d, H, bias=True)
            nn.init.zeros_(self.lam.weight)
            nn.init.zeros_(self.lam.bias)         # lam = 0.5 (classical trapezoid)
            # Mamba-3's learnable data-independent channel biases on B/C -> our K/q
            self.q_bias = nn.Parameter(torch.zeros(H, dk))
            self.k_bias = nn.Parameter(torch.zeros(H, r, dk))
        if kron:
            self.m = int(math.isqrt(dk))
            assert self.m * self.m == dk, "kron needs square dk"
            self.q1 = nn.Linear(d, H * self.m, bias=False)
            self.q2 = nn.Linear(d, H * self.m, bias=False)
            self.k1 = nn.Linear(d, H * r * self.m, bias=False)
            self.k2 = nn.Linear(d, H * r * self.m, bias=False)
            conv_ch = 2 * H * self.m + 2 * H * r * self.m + H * r * dv
        else:
            assert not (kron and r_out > 1)
            self.q_proj = nn.Linear(d, H * r_out * dk, bias=False)
            self.k_slots = nn.Linear(d, H * r * dk, bias=False)
            conv_ch = H * r_out * dk + H * r * dk + H * r * dv
        self.v_slots = nn.Linear(d, H * r * dv, bias=False)
        self.bgate = nn.Linear(d, H * r * dk, bias=True)
        self.wgate = nn.Linear(d, H * r * dv, bias=True)
        self.decay = nn.Linear(d, H * dk, bias=True)
        self.o_proj = nn.Linear(H * dv, d, bias=False)
        if use_conv:
            self.conv = nn.Conv1d(conv_ch, conv_ch, self.CONV_K,
                                  groups=conv_ch, bias=False)
        nn.init.constant_(self.decay.bias, 6.0)     # ~0.9975: long horizon at init
        nn.init.zeros_(self.bgate.bias)
        nn.init.zeros_(self.wgate.bias)

    def _conv_mix(self, streams):
        """Depthwise causal conv + SiLU over concatenated projection streams."""
        if not self.use_conv:
            return streams
        sizes = [s.shape[-1] for s in streams]
        z = torch.cat(streams, dim=-1).transpose(1, 2)          # [B,C,T]
        z = self.conv(F.pad(z, (self.CONV_K - 1, 0)))
        z = F.silu(z).transpose(1, 2)                            # [B,T,C]
        return list(z.split(sizes, dim=-1))

    def _qkv(self, x, B, T):
        H, dk, dv, r = self.H, self.dk, self.dv, self.r
        if self.kron:
            m = self.m
            q1, q2, k1, k2, v = self._conv_mix(
                [self.q1(x), self.q2(x), self.k1(x), self.k2(x), self.v_slots(x)])
            q = torch.einsum('bthi,bthj->bthij',
                             q1.view(B, T, H, m),
                             q2.view(B, T, H, m)).reshape(B, T, H, dk)
            K = torch.einsum('bthri,bthrj->bthrij',
                             k1.view(B, T, H, r, m),
                             k2.view(B, T, H, r, m)).reshape(B, T, H, r, dk)
        else:
            qf, kf, v = self._conv_mix([self.q_proj(x), self.k_slots(x), self.v_slots(x)])
            q = qf.view(B, T, H, self.r_out, dk)
            K = kf.view(B, T, H, r, dk)
        if q.dim() == 4:                       # kron path: single query slot
            q = q.unsqueeze(3)
        if self.trap:
            q = q + self.q_bias[:, None, :]
            K = K + self.k_bias
        return (F.normalize(q, dim=-1, eps=1e-6),
                F.normalize(K, dim=-1, eps=1e-6),
                v.view(B, T, H, r, dv))

    def forward(self, x):
        B, T, d = x.shape
        H, dk, dv, r = self.H, self.dk, self.dv, self.r
        q, K, V = self._qkv(x, B, T)           # q [B,T,H,r_out,dk], K [B,T,H,r,dk]
        if self.rot or self.rope_mod:
            if self.rot:
                # free per-channel data-dependent angles, cumulative
                theta = F.softplus(self.rot_proj(x)).view(B, T, H, dk // 2)
                Theta = theta.cumsum(dim=1)
            else:
                # fixed RoPE ladder x learned per-token rate (adaptive position)
                m = F.softplus(self.mod_proj(x))                 # [B,T,H]
                pos = m.cumsum(dim=1)
                Theta = pos.unsqueeze(-1) * self.rope_inv_freq   # [B,T,H,dk/2]
            cos = Theta.cos().unsqueeze(-2)     # [B,T,H,1,dk/2]
            sin = Theta.sin().unsqueeze(-2)
            def rope(z):
                z1, z2 = z[..., :dk // 2], z[..., dk // 2:]
                return torch.cat([z1 * cos - z2 * sin, z1 * sin + z2 * cos], dim=-1)
            q, K = rope(q), rope(K)
        # Slot-redundancy diagnostic/penalty (fable_idea §6): off-diagonal Gram
        # of the r slot keys. Stashed for the train loop to add as aux loss.
        if r > 1:
            G = torch.einsum('bthrd,bthsd->bthrs', K, K)
            off = G - torch.eye(r, device=x.device, dtype=G.dtype)
            self.last_slot_ortho = off.square().mean()
        else:
            self.last_slot_ortho = x.new_zeros(())
        Bg = torch.sigmoid(self.bgate(x).view(B, T, H, r, dk))
        Wg = torch.sigmoid(self.wgate(x).view(B, T, H, r, dv))
        a = torch.sigmoid(self.decay(x).view(B, T, H, dk)).clamp(max=0.9995)

        N = B * H
        def flat(z, *tail):
            return z.permute(1, 0, 2, *range(3, z.dim())).reshape(T, N, *tail).float()
        q_, K_, V_ = flat(q, self.r_out, dk), flat(K, r, dk), flat(V, r, dv)
        Bg_, Wg_, a_ = flat(Bg, r, dk), flat(Wg, r, dv), flat(a, dk)
        if self.r_out > 1:
            mixw = self.out_mix[None].expand(B, -1, -1).reshape(N, 1, self.r_out).float()
        if self.trap:
            lam_ = torch.sigmoid(self.lam(x)).permute(1, 0, 2).reshape(T, N).float()

        S = torch.zeros(N, dv, dk, dtype=torch.float32, device=x.device)
        prevW = torch.zeros_like(S)
        eyeR = torch.eye(r, dtype=torch.float32, device=x.device).unsqueeze(0)
        outs = []
        for t in range(T):
            S = S * a_[t].unsqueeze(1)
            Kt, Vt = K_[t], V_[t]
            Ktil = Bg_[t] * Kt
            SK = torch.bmm(S, Ktil.transpose(1, 2))
            Gram = torch.bmm(Ktil, Ktil.transpose(1, 2))
            Tt = torch.linalg.solve(eyeR * self.eps + Gram, eyeR.expand(N, r, r))
            S = S - torch.bmm(torch.bmm(SK, Tt), Ktil)
            Wt = torch.bmm((Wg_[t] * Vt).transpose(1, 2), Kt)
            if self.trap:
                # exponential-trapezoidal write: blend current write with the
                # decayed previous write (carryover rides the same decay as S)
                lam_t = lam_[t].view(N, 1, 1)
                S = S + lam_t * Wt + (1 - lam_t) * (prevW * a_[t].unsqueeze(1))
                prevW = Wt
            else:
                S = S + Wt
            if self.compact_P and (t + 1) % self.compact_P == 0:
                # GDN3-style lossy compaction: truncate S to rank R, two-timescale
                # blend with the old state (SVD under no_grad, like the real kernel).
                with torch.no_grad():
                    U, Sg, Vh = torch.linalg.svd(S, full_matrices=False)
                    R_ = self.compact_R
                    S_lr = (U[:, :, :R_] * Sg[:, None, :R_]) @ Vh[:, :R_]
                if self.compact_ste:
                    # Straight-through: forward = compacted blend, backward =
                    # identity. Fixes the gradient wall (no_grad compaction
                    # detaches the state every P tokens -> cross-window write
                    # gradients vanish -> writes never learn; see frontier-2:
                    # even no-op R=32 truncation collapsed recall to 0.04).
                    S_fwd = self.slow_decay * S_lr + (1 - self.slow_decay) * S.detach()
                    S = S + (S_fwd - S).detach()
                else:
                    S = self.slow_decay * S_lr + (1 - self.slow_decay) * S
            yt = torch.bmm(S, q_[t].transpose(1, 2))            # [N, dv, r_out]
            yt = (yt * mixw).sum(-1) if self.r_out > 1 else yt.squeeze(-1)
            outs.append(yt)
        Y = torch.stack(outs, 0).reshape(T, B, H, dv).permute(1, 0, 2, 3)
        return self.o_proj(Y.reshape(B, T, H * dv).to(x.dtype))


class AttnMixer(nn.Module):
    """Vanilla causal MHA control: if THIS can't solve the task, the testbed is
    broken; if it can and the recurrence can't, the recurrence is at fault."""

    def __init__(self, d, H=4, **_ignored):
        super().__init__()
        self.H, self.hd = H, d // H
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def heads(z):
            return z.view(B, T, self.H, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    def __init__(self, d, arch="kmd2", **mix_kw):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.mix = AttnMixer(d, **mix_kw) if arch == "attn" else KMD2Mixer(d, **mix_kw)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.mix(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, d=128, L=2, max_T=1024, **mix_kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_T, d)   # NoPE cripples the attn control's
                                            # previous-token head; harmless for recurrences
        self.blocks = nn.ModuleList([Block(d, **mix_kw) for _ in range(L)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.emb(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))

# ----------------------------- train/eval -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["kmd2", "attn"], default="kmd2")
    ap.add_argument("--no_conv", action="store_true", help="ablate the short conv")
    ap.add_argument("--slot_ortho", type=float, default=0.0,
                    help="aux penalty weight on slot-key redundancy (r>1)")
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dk", type=int, default=32)
    ap.add_argument("--dv", type=int, default=32)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--kron", action="store_true")
    ap.add_argument("--compact_P", type=int, default=0, help="compact every P tokens (0=off)")
    ap.add_argument("--compact_R", type=int, default=8, help="rank kept by compaction")
    ap.add_argument("--compact_ste", action="store_true",
                    help="straight-through gradient across compaction boundaries")
    ap.add_argument("--trap", action="store_true",
                    help="Mamba-3 exponential-trapezoidal write (+q/k biases)")
    ap.add_argument("--rot", action="store_true",
                    help="data-dependent 2x2 rotating state transition (Mamba-3 complex SSM)")
    ap.add_argument("--rope_mod", action="store_true",
                    help="fixed RoPE freq ladder x learned per-token rate (adaptive position)")
    ap.add_argument("--r_out", type=int, default=1,
                    help="output MIMO rank: query slots recombined per head")
    ap.add_argument("--slow_decay", type=float, default=0.97)
    ap.add_argument("--n_pairs", type=int, default=16)
    ap.add_argument("--n_query", type=int, default=4)
    ap.add_argument("--keys_vocab", type=int, default=64)
    ap.add_argument("--vals_vocab", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    evalrng_seed = 99999

    vocab = 2 + args.keys_vocab + args.vals_vocab
    mix_kw = dict(H=args.heads)
    if args.arch == "kmd2":
        mix_kw.update(dk=args.dk, dv=args.dv, r=args.r, eps=args.eps,
                      kron=args.kron, compact_P=args.compact_P,
                      compact_R=args.compact_R, slow_decay=args.slow_decay,
                      use_conv=not args.no_conv, compact_ste=args.compact_ste,
                      trap=args.trap, rot=args.rot, r_out=args.r_out,
                      rope_mod=args.rope_mod)
    model = TinyLM(vocab, d=args.d, L=args.layers, arch=args.arch, **mix_kw).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))

    @torch.no_grad()
    def evaluate(n_batches=8):
        model.eval()
        erng = torch.Generator().manual_seed(evalrng_seed)
        tok_c = tok_n = ep_c = ep_n = 0
        for _ in range(n_batches):
            ids, mask, T = make_batch(erng, args.batch, args.n_pairs, args.n_query,
                                      args.keys_vocab, args.vals_vocab, dev)
            logits = model(ids)
            pred = logits[:, :-1].argmax(-1)
            tgt, m = ids[:, 1:], mask[:, 1:]
            correct = ((pred == tgt) & m)
            tok_c += int(correct.sum()); tok_n += int(m.sum())
            ep_ok = (correct.sum(1) == m.sum(1))
            ep_c += int(ep_ok.sum()); ep_n += ids.shape[0]
        model.train()
        return tok_c / max(1, tok_n), ep_c / max(1, ep_n)

    curve = {}
    t0 = time.time()
    for step in range(args.steps):
        ids, mask, T = make_batch(rng, args.batch, args.n_pairs, args.n_query,
                                  args.keys_vocab, args.vals_vocab, dev)
        logits = model(ids)
        lg = logits[:, :-1][mask[:, 1:]]
        loss = F.cross_entropy(lg.float(), ids[:, 1:][mask[:, 1:]])
        if args.slot_ortho > 0 and args.arch == "kmd2":
            loss = loss + args.slot_ortho * sum(b.mix.last_slot_ortho for b in model.blocks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            ta, er = evaluate()
            curve[step] = {"tok_acc": round(ta, 4), "recall": round(er, 4),
                           "ce": round(float(loss), 4)}
            print(f"step {step:5d}  ce {float(loss):.4f}  tok_acc {ta:.4f}  "
                  f"episode_recall {er:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    final = curve[max(curve)]
    result = {"config": vars(args), "n_params": nparams, "curve": curve,
              "final_tokacc": final["tok_acc"], "final_recall": final["recall"],
              "wall_s": round(time.time() - t0, 1)}
    print(json.dumps({k: result[k] for k in ("final_tokacc", "final_recall", "wall_s")}))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
