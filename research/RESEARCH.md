# GDN3 Auto-Research — Operating Brief (read this every turn)

You are running an autonomous research loop to find a GDN3 configuration that
**learns associative recall fast** on the MQAR proxy — our sub-hour stand-in for
RULER on the 0.8B model. You keep a rolling session (recent turns stay in context;
older ones auto-compact), so **`research/leaderboard.jsonl` is the one durable
record** — one line per experiment — that outlives compaction and stops you
repeating configs. Logging each experiment there is the only mandatory bookkeeping;
keep everything else light. Do ONE experiment per turn, then stop.

## The loop (one turn = one experiment)

1. **Check what's been tried:** skim `research/leaderboard.jsonl` so you never
   repeat a config (your recent turns are already in session context; glance at
   `research_log.md` only if you need the older "Best so far" thread).
2. **Decide the next config** by hypothesis — not random. Look at what scored well
   / diverged and form a testable next step (see Search Space + Priors below).
3. **Write it** to `research/configs/expNNN.json` (NNN = next integer; check the dir).
   Include a `"name"` and a `"hypothesis"` string saying what you expect and why.
   **The `name` + `hypothesis` are auto-posted to the team Discord with the result**,
   so keep the hypothesis one clear sentence a human can skim.
4. **Run it** (this takes ~30–40 min; wait for it):
   ```
   /home/dev/gdn3_qwen35_package/.venv/bin/python research/proxy_mqar.py \
     --config research/configs/expNNN.json --out research/runs/expNNN.json --device cuda:0
   ```
5. **Record (REQUIRED, never skip):** append one line to `research/leaderboard.jsonl`
   = the full `research/runs/expNNN.json` collapsed to one line
   (`python -c "import json;print(json.dumps(json.load(open('research/runs/expNNN.json'))))"`).
   This is the durable log — the Discord post fires automatically from the run.
6. **Reflect (light — only when it matters):** update `research/research_log.md`
   only for a new **Best so far**, a surprising result, or a phase change. No need
   for a paragraph every turn; the leaderboard already holds the full record.
7. **Stop.** The outer loop restarts you for the next experiment.

## What the score means

Result JSON fields: **`final_tokacc`** (0–1, fraction of answer tokens correct —
**the primary fitness**; continuous, so it discriminates even before exact recall
emerges), `final_recall` (all-or-nothing exact-match, secondary/aspirational),
`emergence_step` (when tok_acc first ≥0.5; lower = learns faster), `skip_rate`
(NaN-guarded fraction; **>0.5 ⇒ diverged = failure**), `final_ce`, `wall_s`,
`status`. **Maximize `final_tokacc` while keeping `skip_rate` low.** A high score
with skip_rate>0.3 is fragile — note it but prefer stable configs.

## Search space (config knobs — JSON keys)

Arch (mapped to env by the proxy): `residual_rank` (P, exact buffer: 8/16/32/64),
`slow_decay` (two-timescale blend, 0.80–0.99), `decay_clamp` (forgetting floor,
0.990–0.9999). Optim: `lr_memory` (1e-4–8e-4), `lr_coproduct` (5e-5–4e-4),
`warmup`, `clip`. Task: `steps` (keep 300–500 for ~35 min), `seq_len` (512),
`n_keys` (retrieval load: 4/8/16 — higher = harder), `grad_accum`, `eval_every`.

**Only vary these config values. Do NOT edit source under `gdn3/`, `train/`, or the
proxy itself.** Config-only experiments keep the search safe and reproducible.

## Priors (what we already know — don't re-discover)

- **decay could hit exactly 1.0 → unbounded state → divergence.** Fixed by
  `decay_clamp` (default 0.999). Values too close to 1.0 (e.g. 0.9999) may reopen it;
  lower (0.995) forgets faster (more stable, maybe worse long-recall). This tradeoff
  is a prime axis to map.
- **LR too high diverged late** in the real heal (memory 6e-4). Lower LR (2–3e-4)
  is the current stability guess. Test whether higher LR + tighter decay_clamp is
  stable *and* faster.
- `slow_decay=0.97` is the inherited two-timescale default; unvalidated at scale.
- Capacity: larger `residual_rank`/`n_keys` interact — more exact buffer may help
  higher retrieval load but costs memory/compaction time.

## Multi-turn tasks

Most experiments fit one turn. A hard sub-task (debugging a kernel, deriving a
math change) may span turns. Persist your working state to
`research/current_task.md` (what you're doing, what's done, the next concrete
step) and resume it next turn. One bounded unit of progress per turn, then stop —
the thread lives on disk, not in this conversation.

## Phases — escalate, never idle, never give up

Run indefinitely. If you can't beat the best config, **do not stop or repeat a
config** — climb to the next phase. Always keep the current best (config, scores,
and any promoted checkpoint) under a **"Best so far"** heading in
`research_log.md` so nothing is lost as you explore.

**Phase 1 — config sweep (default; fully authentic GDN3).** Vary only config
values (the Search Space). Stay here while it keeps producing gains.

**Phase 2 — mechanism variants (source edits, git-gated).** Trigger: ~15
experiments with no `final_tokacc` gain > 0.02 over best. You may now edit GDN3
source, but **stay true to the GDN3 math** — Kronecker-residual state,
two-timescale compaction, braided decay, coproduct binding. Fair game: decay
parameterization, gate structure, the two-timescale blend rule, state/output
normalization, coproduct wiring. **Protocol per edit:**
  1. `git checkout -b exp-<name>` — never edit `main`/baseline directly.
  2. Make the change. Run `python -m tests.test_chunk_parity`: a **kernel/perf**
     change MUST still print `PARITY OK ✅`. A deliberate **math** change will fail
     parity by design — if so, update `gdn3/_reference_recurrence.py` to match and
     say so in the log.
  3. Run the proxy; log the result and exactly what you changed.
  4. Beat best → keep the branch, note it. Else → `git checkout main` to revert.
     **Never leave `main` broken.**

**Phase 3 — fundamental architecture (last resort, only after Phase 2 is dry).**
The human has OK'd broader departures once faithful ideas are genuinely exhausted
— better the GPUs research than idle. Same git-branch + parity/proxy protocol, and
**document every departure from GDN3 under a `## DEPARTURES` heading** in the log
for human review. Authenticity to the original idea is preferred; this is to keep
the machines useful, not a license to rewrite freely.

**Hard gate:** before ANY source edit, `git status` on `main` must be clean. If
git is not initialized, do NOT edit source — log "Phase 2 blocked: needs git
baseline" and keep sweeping configs / trying `n_keys`+seed variations instead.

## Rules / safety

- One experiment per turn. Always on `--device cuda:0` unless the log says cuda:0
  is busy, then use cuda:1.
- Write ONLY under `research/`. Never delete or modify anything under
  `runs/gdn3_twotimescale_heal/`, `data/`, `gdn3/`, `train/`, or `data_pipeline/`.
- If a run errors (`status` starts with `error:`), log it and try a *different*
  config — don't loop on the same failure.
- **Time budget:** ~1.8 s per micro-step. Calibration (`steps=400, grad_accum=1`)
  ran in **14 min** — so you have room. Keep `steps × grad_accum ≤ ~1200` (≈35 min +
  load/eval). Good default: `grad_accum=1, steps=500–800` to give recall room to
  emerge past the format plateau. Never exceed an hour per experiment.
- Flag any config with `final_recall ≥ 0.5` and `skip_rate < 0.2` in the log under a
  `## PROMOTE` heading — those are candidates for a human to run a full distill+RULER.
