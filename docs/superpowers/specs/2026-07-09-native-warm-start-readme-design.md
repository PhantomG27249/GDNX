# Native Warm-Start README Design

**Date:** 2026-07-09

## Objective

Replace the historical phase-0 root README with a current, evidence-grounded
guide to `train/train_gdn3_distill.py`, using native warm-start KMD-2 as the
canonical workflow. Preserve the historical README byte-for-byte under
`docs/history/` and update `HANDOFF.md` so it no longer describes the new root
README as historical.

This is a documentation-only change. It must not alter model, trainer, kernel,
dataset, checkpoint, or evaluation behavior.

## Source of Truth

The README must derive behavioral claims from these current files:

- `train/train_gdn3_distill.py` for trainer arguments, losses, optimization,
  data loading, checkpointing, resume, and stopping behavior
- `gdn3/gdn3_upgrade.py` for architecture selection
- `gdn3/kmd2_native.py` for native warm-start KMD-2 behavior
- `gdn3/kmd2_fast_scan.py` and `research/kernel_ab/POSTMORTEM.md` for the
  opt-in fast scan and the scope of its evidence
- `research/KMD2_STATUS.md`, `research/EXPERIMENT_LEDGER.md`, and committed
  RULER result files for milestone evidence and limitations
- `runs/kmd2_native_heal/final/gdn3_layers.pt` for the surviving checkpoint

The ignored local `runs/native_heal.log` may be used to reconstruct recorded
milestone settings, but the README must label those settings as locally
recorded evidence because the log is not part of the Git repository.

## Chosen Structure

The root README becomes the current native warm-start KMD-2 entry point. The
phase-0 document moves to:

`docs/history/phase-0-two-timescale-release.md`

The current README to be archived has:

- Git blob: `b9d1222f130bb0787db6245acbad939a371c42c7`
- SHA-256: `1e0552330a82007338ecd24fd4ac066ab582905f21d1e1b392f5c48745754b0c`
- byte size: `7081`

The archived copy must match all three values before the root README is
replaced.

## Root README Content

### 1. Project identity and status

Title the project GDN-X and describe it as a research implementation of native
warm-start KMD-2 for Qwen3.5 linear-attention layers. Do not call it
production-ready.

Lead with a short status summary:

- native warm-start checkpoint exists
- 18 Qwen linear-attention layers were upgraded
- short-context teacher-forced retrieval evidence exists
- fast scan is experimental and opt-in
- portability, loading, streaming decode, and long-context training remain
  incomplete

Link to `HANDOFF.md`, `research/KMD2_STATUS.md`, and
`research/EXPERIMENT_LEDGER.md` for deeper provenance.

### 2. What native warm-start KMD-2 implements

Describe only the trained native path:

- warm-loaded Qwen projections, convolution, normalization, and output path
- native depthwise convolution and SiLU
- small, nonzero cumulative data-dependent q/k rotation
- one write key/value per head
- `r_out=4` scaled copies of a shared query with learned output mixing
- static per-key-channel decay offsets
- per-head write-beta offset that decouples write from erase
- dense `dk x dv` recurrent state initialized to zero on every forward call

State explicitly that this path does **not** implement:

- rank-r writes
- Kronecker-residual state
- SVD/two-timescale compaction
- persistent recurrent/decode cache
- packed-sequence-aware state resets

The README may link the original GDN3 mode for comparison, but must not apply
its compaction, four-lane, coproduct, partial-RoPE, or exact-alpha claims to the
native KMD-2 checkpoint.

### 3. Mode selection

Document the exact current selection behavior:

| Environment | Selected replacement |
|---|---|
| `GDN3_KMD2_NATIVE=1` | `KMD2NativeAttn` (canonical) |
| `GDN3_KMD2=1` | cold `KMD2LinearAttn` research path |
| neither | original `GDN3LinearAttn` |

Native mode takes precedence when both variables are nonzero. Warn that the
trainer has no CLI mode flag and therefore trains original GDN3 if the native
environment variable is omitted, even though its logging currently says KMD-2.

### 4. Prerequisites

Document the current, actual prerequisites rather than presenting a package
installation flow that does not exist:

- Python environment with compatible PyTorch, Transformers, and related
  training dependencies
- CUDA GPU; the milestone used separate student and teacher GPUs
- materialized `data/mix_v1/blocks.pt` plus `manifest.json`
- local Qwen3.5-0.8B snapshot
- enough storage for checkpoints

State these portability limitations:

- there is no `pyproject.toml` or lockfile yet
- `MODEL_SNAP` is hard-coded in the trainer and must currently be changed in
  the source for a different machine
- `data/mix_v1/` is intentionally ignored and must be prepared locally
- Discord credentials are optional; missing files disable logging

Do not imply that `data_pipeline/` is imported by the trainer. The trainer uses
`data.data_mix.MaterializedMix` directly.

### 5. Canonical launch commands

Provide equivalent PowerShell and Bash examples. Both must set these before
Python starts:

```text
GDN3_KMD2_NATIVE=1
GDN3_KMD2_ROUT=4
GDN3_FAST_SCAN=0
```

Use an explicit native warm-start command rather than relying on trainer
defaults. The command should expose the milestone-shaped choices:

- `--steps 20000`
- `--seq-len 512`
- `--batch-size 2`
- `--grad-accum 2`
- `--lr-memory 1e-4`
- `--lr-preserved 1e-5`
- `--tau 2`
- `--w-kl 1`
- `--w-ce 0.02`
- `--w-layer 1`
- `--log-every 250`
- `--ckpt-every 500`
- `--max-hours 6.5`
- `--out runs/kmd2_native_heal`

The README must distinguish this explicit recommended command from parser
defaults. It must not claim the complete original launch command was committed;
the local log records the visible values but omits some flags.

Also include a three-step smoke command using the same architecture variables,
`--smoke`, `--w-layer 1`, `--no-discord`, and a disposable output directory.
Warn that smoke mode still loads both full models and the dataset.

### 6. Training behavior

Document:

- frozen bf16 teacher and fp32 student
- only upgraded linear-attention layers unfrozen by default
- optional `--freeze-preserved`
- KL plus next-token CE
- optional layerwise normalized residual-stream MSE controlled by `--w-layer`
- separate learning-rate groups; native KMD-2 has no coproduct parameter group
- AdamW, warmup/cosine schedule, gradient clipping, and non-finite-step guard
- fixed-width and optional doubling-window plateau logic
- wall-clock stop and periodic/final checkpointing

State that `--w-layer` defaults to zero, so canonical layerwise training must set
it explicitly.

### 7. Data and sequence-length behavior

Explain that `WindowedMix` samples random windows from fixed 2048-token
materialized blocks. The current dataset has 19,533 blocks / about 40M tokens.

State clearly:

- 512 was the milestone training length
- values below 2048 select a random crop
- values above the stored block length silently return a shorter block
- genuine training beyond 2048 requires rebuilding or repacking the data

Do not claim that longer-context training is implemented merely because
`--seq-len` accepts an integer.

### 8. Checkpoints and resume

Explain that checkpoints contain only upgraded linear-attention tensors plus a
small local metadata file. `--resume` is a weights-only warm continuation and
does not restore optimizer, scheduler, RNG, data position, plateau state, or
step numbering.

Document the LFS-managed milestone checkpoint path and the requirement to set
native mode plus `r_out=4` before loading it. Do not claim that
`AutoModelForCausalLM.from_pretrained` automatically reconstructs KMD-2.

### 9. Fast scan

Describe `GDN3_FAST_SCAN=1` as experimental and native-only. State:

- default is off
- C=16 repaired implementation exists
- saved evidence reports 33.2x scan-level forward/backward throughput versus
  the Python reference
- the milestone checkpoint was trained on the Python reference scan
- there is no committed end-to-end fast-training run
- saved repaired evidence does not include a backward-gradient parity field
- Triton is required and there is no fallback
- compilation can be long and recompiles by sequence length

Do not repeat the stale `~80x` source comment as a current result.

### 10. Evaluation evidence

Include a compact table for the committed n=32 short-context artifacts. It may
report the verified values at 512, 1024, and 2048 for 4 and 8 queries.

Label the evaluation correctly:

- teacher-forced answer-token scoring, not free generation
- trained student versus native teacher, not a controlled `r_out=4` versus
  `r_out=1` ablation
- short-context improvements are directly artifact-supported
- 8k crossover and 16k/32k degradation were observed
- attributing the cliff solely to 512-token training is a hypothesis until a
  longer-context retrain is performed

### 11. Advanced modes

Briefly document original GDN3 and cold KMD-2 as research alternatives. Warn
that they have different parameterizations and are not compatible with the
native milestone checkpoint.

### 12. Repository map, known gaps, history, and license

Provide a current path table for `gdn3/`, `train/`, `data/`, `data_pipeline/`,
`research/`, `runs/kmd2_native_heal/final/`, and the archived phase-0 README.

List the major known gaps:

- hard-coded model path
- missing packaging/lockfile
- environment-only mode selection
- weights-only resume
- no integrated validation
- no streaming/decode cache
- no demonstrated long-context retrain
- incomplete portable tests and hard-coded Linux paths in research tools

Retain the existing MIT license link and copyright identity.

## HANDOFF Update

Change only the stale opening reference in `HANDOFF.md`. It should say the
phase-0 release document is archived at
`docs/history/phase-0-two-timescale-release.md` and the root README is now the
current user-facing guide. Preserve the remaining handoff content, even where
the README gives more precise caveats.

## Verification

Implementation is complete only when all checks pass:

1. The archived phase-0 README matches blob
   `b9d1222f130bb0787db6245acbad939a371c42c7`, SHA-256
   `1e0552330a82007338ecd24fd4ac066ab582905f21d1e1b392f5c48745754b0c`,
   and 7081 bytes.
2. The new root README contains no `code.training`, `code/`, absent benchmark
   scripts, "Production-ready," KMD-2 compaction claim, or stale `~80x` claim.
3. Every repository-relative Markdown link in `README.md` and the modified
   `HANDOFF.md` resolves.
4. The documented environment variables match `GDN3UpgradeManager` and
   `KMD2NativeAttn` selection behavior.
5. The documented trainer flags and defaults match `python
   train/train_gdn3_distill.py --help` and source inspection.
6. RULER numbers in the README match committed JSON evidence exactly after
   rounding.
7. Fast-scan wording distinguishes recorded microbenchmark evidence from fresh
   end-to-end training evidence.
8. `git diff --check` passes for the documentation changes.
9. No source code, data, checkpoint, or result file changes.
10. Final `git status --short --branch --untracked-files=all` is clean after the
    documentation commit.
