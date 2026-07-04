# Handoff: GDN3 chunked-scan recurrence kernel

**For:** an AI assistant implementing the optimization.
**Goal:** replace the token-by-token Python recurrence with a chunkwise-parallel
one, **without changing its numerics**, so overnight training on a single
RTX 5060 Ti becomes productive.

You are ONE function away from a 10–50× speedup. Read this fully before coding.

---

## 0. Success criteria (definition of done)

1. **Parity gate passes:** `python -m tests.test_chunk_parity` prints
   `PARITY OK ✅` (forward + backward match the frozen reference within
   `atol=1e-4, rtol=1e-3`, outputs finite).
2. **Real speedup:** the harness's `[time]` line shows a meaningful speedup
   (target ≥5×; the current implementation is ~44s/step across 18 layers at
   B=1, T=512). Measure fwd+bwd, not just fwd.
3. **Full-model smoke still runs:**
   `python -m train.train_gdn3_distill --smoke --seq-len 512 --freeze-preserved --no-discord`
   completes 3 steps and saves a checkpoint, at a lower s/step than ~44.
4. You changed **only** `GDN3LinearAttn._gdn3_recurrent_state` (and optionally
   added helper methods / a Triton file). Do NOT touch `_kron_read_vec`,
   `_stable_alpha_vec`, `_compact_vec`, the frozen reference, or the forward
   pre/post-processing.

Keep grad connectivity **identical**: gradients flow only *within* a P-token
compaction window (compaction runs under `torch.no_grad()` — truncated BPTT).
The rewrite must preserve the same truncation, so the backward-parity test is
meaningful.

---

## 1. Environment

- **Python/venv (has torch 2.12+cu128, transformers 5.12, 2 GPUs):**
  `/home/dev/gdn3_qwen35_package/.venv/bin/python`
- **GPUs:** 2× RTX 5060 Ti 16 GB. Always prefix runs with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (memory is tight; ~14.9 G
  used at B=1 unfrozen, ~12.7 G with `--freeze-preserved`).
- **Model snapshot (Qwen3.5-0.8B, config nests under `.get_text_config()`):**
  `/home/dev/.cache/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17`
- **Run the gate:** `cd /home/dev/gdn3_two_timescale_release &&
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  /home/dev/gdn3_qwen35_package/.venv/bin/python -m tests.test_chunk_parity`
- ⚠️ Do not `nvidia-smi kill` random PIDs — the human occasionally runs other
  GPU apps. If you need memory, use `--freeze-preserved` or smaller T.

---

## 2. What to change

**File:** `gdn3/gdn3_upgrade.py`
**Function:** `GDN3LinearAttn._gdn3_recurrent_state(self, q_features,
k_features, v_features, b_gates, w_gates, decay_factors) -> [B,T,H,M,V]`
(currently ~line 383).

Everything feeding it (projections, coproduct blend, gates, braided decay,
partial RoPE, lane expansion) and everything after it (routing, norm, output
gate, `out_proj`) is fine and fast — leave it alone. The Python `for t in
range(T)` loop inside this one method is the entire bottleneck.

The pre-rewrite body has been copied verbatim to
`gdn3/_reference_recurrence.py::reference_recurrent_state` — that is your
**ground truth**. The parity test compares your new live method against it.

---

## 3. Exact reference semantics (MUST reproduce)

State per chain `n` (there are `N = B*H*M` chains; default N=64 at B=1):
`S_n = Σ_r A_r ⊗ B_r  +  U Vbᵀ`, with `A:[R,a_v,a_k]`, `B:[R,b_v,b_k]`,
`U:[V,P]` (residual values), `Vb:[K,P]` (residual keys). Defaults:
`R=4, P=16, a_k=a_v=16, b_k=b_v=8, K=V=128, M=4`.

Per token `t` (scalar decay `γ_t∈[0,1]` per chain):
```
A  *= γ_t ;  Bk *= γ_t ;  Vb *= γ_t          # NOTE: A⊗B decays by γ_t², residual by γ_t
h   = b_gate ⊙ k ;  u = w_gate ⊙ v
s_h = S·h                                    # _kron_read_vec(A,Bk,U,Vb,h)
r   = u − s_h
α   = (1 − exp(−(k·h))) / (k·h)              # _stable_alpha_vec, Taylor-safe near 0
s_q = S·q
y_t = s_q + α·(k·q)·r                         # output for token t (post-update shortcut)
U[:, p] = α·r ;  Vb[:, p] = k ;  p += 1       # write into slot p
if p == P:  A,Bk,U,Vb = compact(A,Bk,U,Vb) ; p = 0   # under no_grad
```

**Two subtleties that are easy to get wrong:**

- **γ² on the Kronecker part.** Both `A` and `Bk` are multiplied by `γ_t`, so
  the rank-1 term `A_r⊗B_r` decays by `γ_t²` each step, while residual keys
  decay by `γ_t¹`. Replicate this exactly (even if it looks like a quirk —
  flag it in your writeup but match it; parity is the contract).
- **Circular, always-full residual buffer.** `compact` (two-timescale) blends
  `UVbᵀ` into the Kronecker factors but **returns `U, Vb` unchanged** and only
  resets `p=0`. So after the first window the P-slot buffer is always full;
  the next window overwrites slots 0,1,2,… and **every read uses all P slots**
  — a sliding window of the last P writes that spans window boundaries. This
  is the trickiest part to reproduce in a chunked form.

`_kron_read_vec(A,Bk,U,Vb,x)` = `vec(Σ_r A_r (x.reshape(a_k,b_k)) B_rᵀ) +
U (Vbᵀ x)`. See its body (~line 467) — do not change it; call it, or inline
its math in your chunk kernel.

---

## 4. Recommended approach

**Natural chunk = the compaction window (P tokens).** Compaction is the only
non-associative op and it already lands exactly on P-token boundaries, so make
the chunk size a multiple of P (start with C = P = 16). Within a chunk there is
NO compaction — just decay + delta-rule writes — which is the standard
chunkwise-parallel DeltaNet setting.

**Do it in two stages. Stop after Stage 1 if it clears the speedup bar.**

### Stage 1 — vectorized-within-chunk, pure PyTorch (do this first)
Replace the `T`-step Python loop with a `T/C`-step loop over chunks; inside each
chunk, compute all C outputs and the C residual writes with **batched matmuls**
instead of a per-token loop. This alone removes ~C× of the kernel-launch
overhead that dominates today (the GPU currently sits ~60% idle, launch-bound).

Sketch per chunk (all ops batched over N chains):
- Precompute intra-chunk relative decays via cumulative products of `γ` so the
  read at position i correctly sees earlier writes j<i decayed by `Π_{j<s≤i}γ_s`
  (and the carried Kronecker state decayed by the chunk-prefix product, squared).
- The causal intra-chunk delta interaction (each write depends on the state
  *including* earlier writes in the same chunk) is the DeltaNet "UT transform":
  build the C×C strictly-lower-triangular matrix of `α_i (k_i · h_j)` style
  couplings and invert `(I − L)` once per chunk (C=16 ⇒ trivial 16×16 solve,
  batched over N). References: "Parallelizing Linear Transformers with the Delta
  Rule over Sequence Length" (Yang et al.) and the `fla` library's
  `chunk_gated_delta_rule` — but note GDN3 differs (Kronecker prefix state,
  scalar γ, the `+α(k·q)r` output shortcut, circular residual buffer), so you
  must re-derive, not copy.
- Handle the Kronecker prefix state `A⊗B` as the chunk's initial state `S_0`
  (read it via `_kron_read_vec` with empty U/Vb, or fold it into the batched
  read). Keep residual reads exact via the sliding-window buffer.
- At the chunk end, do the write-back of the chunk's residuals into U/Vb slots
  and call `_compact_vec` once (unchanged).

Keep it in fp32 and autograd-native (no custom backward) — PyTorch differentiates
the matmuls for you, and the parity-backward test guards correctness.

### Stage 2 — Triton (only if Stage 1 is not enough)
Port the per-chunk math to a Triton kernel with a hand-written backward.
The prior repo's `kernels_triton_backward.py` is **not reusable** (it's gated
off for GDN3's dims `b_v=b_k=8<16`, only covers the plain Kronecker read, and
its `__all__` references undefined names). Treat it as scrap. If you go here,
write a fused chunk fwd+bwd from scratch and keep the Stage-1 PyTorch path as the
reference/fallback.

---

## 5. Gotchas / open questions

- **Numerical divergence.** Feeding raw `N(0,1)` to the recurrence makes the
  state blow up to non-finite over ~hundreds of steps (the state has no norm).
  This is why `test_chunk_parity` uses small, decay-heavy inputs. Real
  warm-started data was stable for a 3-step smoke, but **watch KL/gnorm early**
  in any real run — grad clipping (`--clip 1.0`) is on. Not your bug to fix, but
  don't let a "parity passes on tiny inputs, NaNs on real data" gap slip
  through: also run the full-model smoke (success criterion #3).
- **Backward = truncated BPTT.** Compaction is under `no_grad`, so grads don't
  cross window boundaries. Preserve this exactly; don't accidentally make the
  chunked version differentiate through compaction (it would change training
  dynamics and fail backward parity).
- **`P` is configurable** via env `GDN3_P` (default 16). Don't hardcode 16; use
  `self.P`. Chunk size should be `P` or a small multiple.
- **Memory.** Larger chunks store more intra-chunk activations; at T=512, N=64
  we're near the 16 GB limit. If you widen chunks, test memory with
  `--freeze-preserved`.
- **Determinism.** `_compact_vec` uses a fixed-seed Gaussian sketch so grad
  checkpoint recompute matches forward — keep that property.

---

## 6. Files at a glance

| File | Role | Touch? |
|---|---|---|
| `gdn3/gdn3_upgrade.py` → `_gdn3_recurrent_state` | the loop to rewrite | ✅ yes |
| `gdn3/_reference_recurrence.py` | frozen ground truth | ❌ never |
| `tests/test_chunk_parity.py` | parity + timing gate | ❌ (run it) |
| `_kron_read_vec` / `_stable_alpha_vec` / `_compact_vec` | shared helpers | ❌ no |
| everything else in `forward()` | fast already | ❌ no |

When done: leave the parity output and the `[time]` speedup line in your final
message so the next session can validate and move straight to launching training.
