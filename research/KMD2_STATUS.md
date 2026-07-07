# KMD-2 (Fable) validation — working state

## *** MILESTONE (2026-07-06): WORKING GDN WITH MIMO + COMPACTION ***
From-scratch testbed (`research/testbed_mqar.py`, atomic-token MQAR, 16 pairs,
6000 steps × batch 32, ~5 min/run on cuda:1), suite v2 results:
  attn control        recall 0.996  (task valid)
  kmd2 r=4 NO CONV    recall 0.000  <- conv ablation = the whole failure mode
  kmd2 r=1            recall 0.996
  kmd2 r=4            recall 1.000
  kmd2 r=4 +kron keys recall 0.996
  kmd2 r=4 +compact P32 R8 / R4   recall 0.988 / 0.953  (graceful degradation)
  kmd2 r=1 +compact P32 R8        recall 1.000
TWO required ingredients found: (1) short depthwise causal conv before q/K/V —
without it the k written at the value position can't contain the key identity
(adjacent-token binding impossible; CE settles at ln16 = "any context value");
(2) a trainable regime (the frozen-Qwen CE proxy denies alignment gradients to
ANY arch — see probe chain above). gdn3/kmd2.py (Qwen drop-in) still LACKS conv;
retrofit before any heal attempt (Qwen's frozen conv1d is available to reuse).
Caveat: 16 pairs doesn't stress capacity (r=1 ≈ r=4 expected); MIMO ADVANTAGE
still unproven -> frontier sweeps below.

## FRONTIER FINDINGS (2026-07-06/07)
Frontier 1 (load/state ladder): **MIMO r=4 HURTS under memory pressure**
(np64: r1 0.977 vs r4 0.762; np32 dk16: r1 0.965 vs r4 0.473) — redundant slots
multiply cross-token interference (fable_idea §6 risk, unmitigated in v1).
Compaction P32 R4 at np64: BOTH r collapse to 0.0.
Frontier 2: (a) **slot-ortho penalty 0.1 rescues MIMO** (np64 r4: 0.762->0.922;
dk16: 0.473->0.813) — mechanism confirmed; r=1 still best on pure capacity, so
MIMO's case = throughput/multi-fact tokens, not capacity. (b) **compaction has
TWO walls**: R=32 truncation on a 32x32 state is a forward NO-OP yet recall
collapsed to 0.043 -> the `no_grad` compaction DETACHES the state every P tokens
(truncated BPTT) so cross-window WRITE gradients vanish — writes never learn.
P64 (half the detach events) > P32 at same R (0.246 vs 0.012) confirms.
**The real GDN3 kernel has the same no_grad compaction at P=16/seq 512 = 32
detach boundaries -> yet another structural reason all 41 auto-research runs
couldn't learn long-range retrieval writes.**
Frontier 3 (STE across compaction boundaries, np64, P32): **the gradient wall is
FIXED and the two walls are now cleanly separated.**
  R=32 no-op control:  no_grad 0.043 -> STE **0.949**  (proves detachment was the killer)
  R=16:                no_grad 0.012 -> STE **0.926**
  R=8:                                  STE  0.785
  R=4:                 no_grad 0.000 -> STE  0.113   (info wall: R must scale with load)
  **FULL STACK r=4 + slot_ortho 0.1 + STE + R=16: recall 0.906** <- a working GDN
  with MIMO + compaction under 4x-capacity load. GOAL ACHIEVED in the testbed.
Info-wall law at load 64: R=16 ~93%, R=8 ~79%, R=4 ~11% -> keep R >= load/4-ish.
Deployment note: STE changes ONLY training gradients; inference math is identical
to the no_grad kernel, so the trained weights drop into the production kernel.

## NEXT (Qwen heal, in order)
1. Retrofit into gdn3/kmd2.py: short conv (reuse frozen Qwen conv1d), STE across
   its compaction (if compaction is kept), slot-ortho aux term.
2. Replace the training signal: layerwise distillation from Qwen's native
   GatedDeltaNet layers (dense signal) or aux InfoNCE alignment; CE-only frozen
   proxy is proven unable to train ANY drop-in.
3. Then A/B r=1 vs r=4+ortho at matched state in the healed model.

Goal: test whether Fable's KMD-2 idea (fable_idea.txt) gets nonzero MQAR recall on
the robust proxy (`research/proxy_mqar.py`), vs original GDN3's 0-recall ceiling.

## Setup done
- `gdn3_fable` is the self-contained sandbox. `proxy_mqar.py` ROOT repointed to it
  (was importing gdn3 from `gdn3_two_timescale_release`). Runs on `--device cuda:1`
  (GPU0 = user's other experiment). venv: `/home/dev/gdn3_qwen35_package/.venv/bin/python`.
- Branch `kmd2-fable` (off `main`).
- `gdn3/kmd2.py` = `KMD2LinearAttn`: per-token rank-r block-Householder delta
  (RLS T-factor), trainable q/k/v/gate projections, frozen Qwen output path
  (in_proj_z gate + out_proj). Selected by env `GDN3_KMD2=1` (set from config
  `"kmd2":true`). Knobs: kmd2_r/h/dk/dv/eps. Defaults H=16,dk=dv=64,r=4,eps=0.5.
  Wired in `gdn3/gdn3_upgrade.py::apply_upgrade` + proxy env mapping.
- Bug fixed: load_qwen_weights missed `in_proj_z.weight` (endswith) → gate was 0 →
  0 grad. Now 198/198 params train. Smoke: finite, ~20min/400steps, 6GB peak.

## Results
- BASELINE (original GDN3, exp001) reproduced in sandbox: tok_acc 0.2958, recall
  0.0, skip 0.0 — matches auto-research. Sandbox validated.
- `runs_fable/kmd2_001.json` (r=4, seq512, nkeys4, 400 steps): tok_acc 0.30,
  recall 0.0, skip 0.0, ce 1.71 — SAME ~0.30 plateau as GDN3. No recall.

## REVISED DIAGNOSIS (Fable, after inheriting session)
Two v1 problems found by author review; they overturn the "CE-only loss is the
structural ceiling" handoff:
1. **Memory horizon bug**: v1 decay init sigmoid(2.5)=0.924/token → ~13-token
   horizon; the queried binding (~150-200 tokens of context, shuffled placement)
   was ERASED before the query arrived. No surviving binding = no retrieval
   gradient, regardless of loss. Fixed in kmd2_002: bias 6.0 → 0.9975 (~61%
   retained over 200 tokens). Different failure cause than original GDN3 (frozen
   q/k can't align) — same 0.30 plateau from the outside.
2. **Data budget**: CE on answer tokens DOES reward exact retrieval (it's the only
   way below the format floor); the real issue is induction circuits emerge via a
   phase transition needing ~10k+ episodes (Zoology/MQAR literature). The proxy
   trains 400 episodes (batch 1, accum 1). ALL 41 auto-research runs + kmd2_001
   stopped pre-emergence. `grad_accum`+`steps` are legal config knobs to scale this.
   Fable's web 17% = a from-scratch run past the phase transition; nothing mystical.

Prediction: horizon fix + episode scale → recall emerges for KMD-2 (trainable q/k)
but NOT for original GDN3 (structurally blocked). Then r=1 vs r=4 isolates MIMO.
- kmd2_002 (horizon fix only): tok_acc 0.3208, recall 0.0, skip 0.0, ce 1.87.
  Slightly above v1 (0.30) and baseline (0.296) but no recall. Horizon fix was
  NECESSARY (restores retrieval gradient) but NOT SUFFICIENT — consistent with the
  data-phase-transition claim: gradient exists now but needs episodes to drive it.
- kmd2_003 (horizon + grad_accum 4 × steps 500 = 2000 episodes): tok_acc 0.2667,
  recall 0.0, skip 0.0. DATA-SCALE HYPOTHESIS FALSIFIED — tok_acc even dropped;
  no emergence trend across 2000 episodes.

## PROBE RESULTS (the decisive measurements, 2026-07-06)
1. `probe_kmd2_alignment.py` — trained KMD-2 (proxy recipe, 200 steps), measured
   q->k gold-vs-distractor cosine gap at answer position: **mean -0.044 across all
   18 layers, best-head ~+0.01 = NOISE. CE training produces ZERO alignment.**
2. `probe_contrastive_alignment.py` — InfoNCE-trained (Wq,Wk) probes on frozen
   hidden states, span-pooled keys: **eval 92-95% at every layer except L0
   (chance 8.3%). The representation is FINE; a solution exists in KMD-2's own
   parameter space (linear q/k projections). CE just never finds it.**
3. `probe_train_qk_init.py` — per-token keys (the recurrence's real constraint):
   eval drops to **29% mean (best layers 4/5/6/18/21/22: 40-64%)**. Key identity is
   DISTRIBUTED across the binding span, not in single tokens; overfits at 300 eps.
   Saved per-layer probes -> `runs_fable/qk_probe_init.pt`.

**MEASURED CEILING: CE-through-frozen-LM-head never CREATES q->k alignment even
though it's linearly there (95% pooled). Not arch, not horizon, not data, not
state-transition. Author's suggestions (Mamba3-style 2x2 state rotations, learned
lambda_t, trapezoidal B_{t-1}x_{t-1} carryover) address state-transition fidelity —
orthogonal to this bottleneck; revisit AFTER alignment exists.**

- kmd2_004 (probe-init: q_proj + k_slots slot-0 seeded from qk_probe_init.pt,
  otherwise identical to kmd2_002): RUNNING. Tests whether CE AMPLIFIES a
  partially-aligned init into recall or the format attractor DESTROYS it.
  Either way decisive: recall>0 => Fable's result recreated in trusted harness
  (init+config only, proxy untouched); still-0 => strongest evidence the LOSS
  must change (aux alignment loss — needs Billy's approval as a proxy edit).

## Key interpretation of v1 (Opus, superseded above)
KMD-2 gives the model exactly what the auto-research said original GDN3 lacked:
**trainable q/k that CAN form an induction head** (the auto-research's forbidden
ask 'e' — "unfreeze in_proj_qkv"). Yet KMD-2 STILL gets 0 recall / 0.30 tok_acc.
This corroborates the auto-research's exp031/exp035 (a fully-trainable query still
plateaus at format) and shifts the suspect from the FROZEN PROJECTIONS to the
**CE-only / format-attractor LOSS** (asks 'b' aux retrieval loss, 'c' non-format
values) as the real ceiling. i.e. the 0.30 plateau looks loss-side & arch-
independent, so Fable's ~17% almost certainly came from a from-scratch/looser web
regime, not this frozen-backbone CE-only proxy.

## Decisive next experiment (needs a proxy/task edit — user call)
Break the format attractor and see if KMD-2 (already has trainable q/k) crosses to
recall. Two options, both flagged by auto-research for a human:
 (b) aux retrieval/alignment loss (task+loss unchanged otherwise), OR
 (c) non-format-satisfiable answers (e.g. value = single rare word token) so
     "emit any 4-digit number" no longer partially satisfies CE.
If KMD-2 gets recall there -> arch works when loss cooperates; if still 0 -> deeper.
Alternative before editing proxy: retrain+probe whether KMD-2's q aligns to k
(localize failure); or 1-2 sharper KMD-2 arch configs (eps 0.05) — but alignment,
not overwrite-sharpness, is the likely crux, so arch tuning is expected to stay 0.
