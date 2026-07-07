# GDN3 / KMD-2 Experiment Ledger (through 2026-07-07)

One row per experiment (or cluster). Builder legend: **GLM 5.2** = autonomous
auto-research loop (exp001–041, frozen-Qwen CE proxy); **Opus 4.8** = session
work before Fable was summoned; **Fable** = everything after. Confidence =
educated-guess reliability of the *conclusion* (noise / implementation risk /
superseded), plus whether it's ever worth revisiting. Gaps in this table are
from imperfect memory of the 41-run history — see `research_log.md` +
`leaderboard.jsonl` for the full GLM record.

| # | Experiment(s) | Architecture / change | Builder | Result | Confidence & revisit |
|---|---|---|---|---|---|
| 1 | exp001–010, 015, 024, 027, 041 (~16) | Original GDN3, frozen proxy; single-knob config sweeps (lr, decays, rank, n_keys, seed, steps) | GLM 5.2 | tok_acc 0.27–0.31, recall 0.0 all; ~half diverged | High (observations); interpretations superseded — regime had 3 structural defects. No revisit. |
| 2 | exp011–016 | Faithful GDN3 mechanism edits, one each (timescale mix, coprod init, norm bypass, router sharpen, blend rule) | GLM 5.2 | all ~0.30 / 0.0 | High; mechanisms correctly exonerated. No revisit. |
| 3 | exp017–030 | Departures: sharp/softmax reads, output gain 3×/10×, zero-GDN3 ablation | GLM 5.2 | all 0.0 recall; exp025 identity ablation (0.067) informative | High; exp025 still a valid datum. Others superseded. |
| 4 | exp031–040 + diag scripts | No-train measurements: binding forms (cos .35), query≠key (.10), 5/16 heads induce, 0.30 = 3.5× format, LM head fine | GLM 5.2 | measurements valid | Med-high; causal frame partly right. Cite, don't re-run. |
| 5 | baseline_repro | exp001 rerun in sandbox after proxy ROOT repoint | Opus 4.8 | 0.2958 / 0.0 (exact match) | High. Sandbox validated. |
| 6 | kmd2_001 | KMD-2 v1 drop-in (rank-4 Householder delta, RLS T-factor, trainable q/k/v/gates), frozen proxy | Opus 4.8 | 0.30 / 0.0 | LOW — confounded (no conv + 13-token decay-horizon bug). Superseded. |
| 7 | kmd2_002 | + decay bias 2.5→6.0 (single) | Fable | 0.32 / 0.0 | Med-high; necessary-not-sufficient. Conv-less confound. |
| 8 | kmd2_003 | + grad_accum 4 × 500 = 2000 episodes | Fable | 0.267 / 0.0 | Medium; falsified data-scale *in that regime*; conv-less confound — don't over-generalize. |
| 9 | probe: trained alignment | q·k gold-vs-distractor gap after 200 CE steps | Fable | −0.044 all layers | High. |
| 10 | probe: contrastive (span-pooled) | InfoNCE (Wq,Wk) on frozen hidden states | Fable | 92–95% every layer (chance 8%) | High — cornerstone: info present, regime is ceiling. |
| 11 | probe: per-token | Same, per-token keys | Fable | 29% mean (best 40–64%) | High; key identity span-distributed → conv needed. |
| 12 | kmd2_004 | Probe-distilled q/k init, frozen proxy | Fable | 0.296 / 0.0 (peak 0.34 → erosion) | ⚠ MEDIUM — "CE erodes alignment" CONFOUNDED by missing conv. **REVISIT once drop-in has conv** (cheapest high-value loose end). |
| 13 | testbed suite v1 | From-scratch testbed, 6 configs | Fable | all ~0.10 / 0.0 incl. attn | INVALID (no conv, no pos-emb, double-launch). Superseded. |
| 14 | testbed suite v2 | +conv +pos-emb: attn ctrl / no-conv / r1 / r4 / kron / compact R8,R4 / r1+R8 | Fable | attn .996; **no-conv 0.0**; r1 .996; r4 1.0; kron .996; R8/R4 .988/.953 | High (control validates). Conv ablation = most load-bearing result. Kron only light-load → one np64 run worthwhile if Kron matters for heal. |
| 15 | frontier 1 | r1 vs r4 at np32/np64/dk16; np64+R4 compaction | Fable | r4 ≪ r1 under pressure (.47 vs .97 @dk16); R4 compaction both 0.0 | High; deltas ≫ noise, single seed. |
| 16 | frontier 2 | (a) slot-ortho 0.1 on r4; (b) compaction R16/R32/P64 (no_grad) | Fable | ortho rescues (.76→.92, .47→.81); **R32 no-op still .04 → gradient wall** | High — no-op anomaly = key mechanistic discovery. |
| 17 | frontier 3 | + STE across compaction; R ladder + full stack | Fable | R32 .949 / R16 .926 / R8 .785 / R4 .113; r4+ortho+STE+R16 = **.906** | High (no-op control airtight). Info-law R ≳ load/4 single-seed — re-verify at heal scale. |
| 18 | trap suite | Mamba-3 exp-trapezoidal write + q/k biases, ±conv | Fable | no-conv+trap 0.0 (×2); conv+trap .945 | High for "doesn't replace conv". Caveat: cross-carryover variant (v_t under k_{t−1}) UNTESTED — could still pan out. |
| 19 | Mamba-3 finalization | rot / rot+trap / r_out4 / r4+r_out4+ortho / rot+STE (1 seed) | Fable | .957 / .973 / .984 / .949 / .953 | Medium alone — resolved by #20. |
| 20 | seed replication (n=3) | rot+STE vs STE; r_out4 vs r1 | Fable | rot: +2.7/+2.3/0.0 (never neg); r_out4 dead-even | Med-high. rot = include; widening = efficiency-only. |
| 21 | rope_mod (2 seeds) | Fixed RoPE ladder × learned per-token rate, ±STE | Fable | .934 / .928 — below baseline & rot | Med-high skip. Caveat: full-dim ladder (not partial). |

## Never tested (known gaps)
- RLS ε (T-factor regularizer) sweep; r>4.
- Kronecker keys under load (np64) or with compaction.
- Cross-carryover trapezoid (v_t written under k_{t−1}) — principled conv alternative.
- Partial-RoPE version of rope_mod.
- **Throughput/wall-clock benchmarks** — MIMO's actual motivation; nothing measured speed.
- Multi-fact-per-token tasks (where rank-r *should* win on quality).
- The Qwen heal itself: conv+STE+rot retrofit trained by distillation / aux-InfoNCE.
- Row 12 revisit: frozen proxy + conv-retrofitted drop-in (decides if frozen-CE regime is truly unusable).

## Final architecture (locked, testbed evidence)
r=1 gated delta + short conv + learned decay | SVD compaction R ≳ load/4 +
two-timescale + **STE** + **rotating transition** | optional r_out=4 / rank-r+ortho
(efficiency only) | skip trap & data-dep RoPE | train w/ distillation or
aux-InfoNCE, never CE-only through a frozen head.
