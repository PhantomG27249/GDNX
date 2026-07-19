# Portable KMD-2 ablation suite

This package runs preregistered paired tests for native KMD-2, trapezoidal
write carry, corrected/Nesterov-style momentum, causal lookahead, true
MIMO/state-size controls, Gated DeltaNet-2 channelwise erase/write gates,
B/C bias, and the HOLA-inspired bounded exact cache.
It records evidence; it does not claim an arm is better until the configured
paired promotion gates pass.

The canonical entry point is:

```bash
python -m research.kmd2_ablation.run_ablation --help
```

## Committed testing matrix

| Config | Paired arms | Seeds | Primary task / purpose |
|---|---:|---:|---|
| `configs/smoke.json` | native + exact surprise cache | 1 | CPU MQAR plumbing |
| `configs/trapezoid_screening.json` | native + trapezoidal carry | 3 | irregular-time integration |
| `configs/corrected_momentum_screening.json` | native + corrected momentum | 3 | drift/reversal adaptation |
| `configs/causal_lookahead_screening.json` | native + causal lookahead | 3 | trajectory extrapolation |
| `configs/gdn2_decoupled_screening.json` | scalar-offset native + GDN-2 channelwise gates | 3 | MQAR erase/write control |
| `configs/screening.json` | native + exact surprise cache | 3 | MQAR screen |
| `configs/promotion.json` | native + exact surprise cache | 5 | structured exceptions promotion |
| `configs/qwen_exact_cache.json` | native + matched recency + surprise | 3 | mandatory long-cell RULER heal |

The typed registry additionally expands the serial exact-cache selector, read,
capacity, intervention, oracle, equal-state, and four-cell rotation/`r_out`
matrix. Those jobs share deterministic seeds/pairing IDs and are gated so a
later stage cannot be treated as evidence before its prerequisite passes.

## Architecture registry gate

Stage 1 freezes 28 arms covering the stock and converted KMD-2 controls;
rotation, convolution, trapezoid, QK, and lookahead mechanisms; state-size and
`R_out` controls; true MIMO R2/R4; channelwise GDN-2 R1; and bounded, block-only,
selector, read, and oracle cache variants. The registry labels their scientific
classes explicitly as source, converted control, reliance, alternative,
diagnostic, addition, cold redesign, control, or replacement. In particular,
shared-query widening is not true MIMO.

The committed cardinality is 28 Stage 1 records, 62 preregistered Stage 2
pairs, and 248 Stage 2 four-cell records. Cache score/read formulas, width,
block size, coordinate frame, dtype, inclusivity, tie policy, initialization,
and selector seed are frozen semantics. QK mode and its diagonal parameter
bounds are likewise frozen; compatibility fails closed for undefined MIMO,
GDN-2, `R_out`, cache, and QK crosses.

This registry gate records architecture identities and compatibility only; it
is not scientific evidence by itself. The approved Qwen conversion and
execution path is now runnable, but promotion still requires completed matched
jobs and the immutable evaluation ledger.

## Verify and extract an archive

Distribute each archive together with its adjacent verifier, `verify_bundle.py`,
and run that sidecar before extraction:

```bash
python verify_bundle.py kmd2-tiny.zip
mkdir -p kmd2-tiny && python -m zipfile -e kmd2-tiny.zip kmd2-tiny
cd kmd2-tiny
```

`verify_bundle.py` rejects unexpected members, unsafe paths, duplicates,
noncanonical order, and hash/manifest mismatches. Do not run a bundle that
fails verification.

## Environment

Tiny CPU lane:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r research/kmd2_ablation/requirements-tiny.txt
```

Qwen/CUDA lane:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r research/kmd2_ablation/requirements-qwen.txt
```

The Qwen requirement pins Transformers 5.13.1, the verified release in this
suite with
`transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5RMSNormGated`.
Preflight checks the installed distribution for that import capability without
importing Transformers or constructing a model.

Use the cluster-approved PyTorch/Triton build when the generic constraints do
not select a build compatible with the server driver. Never change dtype,
model dimensions, cache dimensions, or budget inside a paired comparison.

## External Qwen assets

Large assets are not bundled. Set them on the remote host:

```bash
export MODEL_PATH=/srv/models/qwen35-base
export TOKENIZER_PATH=/srv/models/qwen35-base
export NATIVE_CHECKPOINT=/srv/checkpoints/kmd2-native.pt
export DATA_PATH=/srv/data/kmd2-heal.jsonl
export TEACHER_MODEL_PATH=/srv/models/qwen35-teacher
export OUTPUT_PATH=/srv/results/kmd2-exact-cache
export GDN3_FAST_SCAN=1
export GDN3_KMD2_ROUT=4
```

`MODEL_PATH` must expose safetensors headers so dry-run preflight can count
parameters without loading model tensors. The names in
`configs/qwen_exact_cache.json` target the declared 18-layer linear-attention
layout. The official Qwen3.5-0.8B multimodal repository is accepted directly:
the loader extracts its text backbone and tied LM head into a causal-LM shell
and releases the unused vision tower. For another layout, create a new
versioned config with exact parameter names.

Build the canonical immutable data bundle with:

```bash
python -m research.kmd2_ablation.scripts.build_data_bundle \
  --band "$HEAL_BAND" --tokenizer "$TOKENIZER_PATH" \
  --out "$DATA_PATH" --seeds 11 29 47
```

It contains exactly 64 ordered 4096-token training microbatches, plus one
evaluation partition per seed. Each partition has the complete 18-cell RULER
matrix: 64 teacher-forced episodes and 8 deterministic free-generation
episodes per cell (1296 rows per seed). Token IDs are stored as compact int32
tensors; labels and dense query/source annotations are reconstructed for only
the selected seed. Overriding `--eval-grid` makes the bundle feasibility-only.
Metadata-only dry-run rejects a config whose declared windows cannot realize
its update, accumulation, example-ID, and token budgets. An optional
`--assets-manifest` can pin sizes, SHA-256 values, and tree identities.

## Preflight

Tiny:

```bash
python -m research.kmd2_ablation.run_ablation preflight \
  --backend tiny --config research/kmd2_ablation/configs/smoke.json \
  --out "$OUTPUT_PATH" --device cpu --dtype float32 --dry-run
```

Qwen:

```bash
python -m research.kmd2_ablation.run_ablation preflight \
  --backend qwen --mode heal \
  --config research/kmd2_ablation/configs/qwen_exact_cache.json \
  --model "$MODEL_PATH" --tokenizer "$TOKENIZER_PATH" \
  --native-checkpoint "$NATIVE_CHECKPOINT" --data "$DATA_PATH" \
  --teacher-model "$TEACHER_MODEL_PATH" \
  --student-device cuda:0 --teacher-device cuda:1 --dtype bfloat16 \
  --out "$OUTPUT_PATH" --dry-run
```

Preflight hashes sources/assets, runs measured identity and active-effect
gates, performs phase-separated resource accounting, expands jobs, and writes immutable
`manifest.json` and `jobs.json`. Qwen dry-run reads identity and tensor metadata
only; it does not construct a model or load tensor payloads. It counts the
official nested text model, replaced native layers, hybrid parameters, FP32
state/history, streamed-loss/evaluation workspaces, CPU-offloaded optimizer
moments, and teacher as distinct phases, with a 20% margin for each. It records
the selected implementation under
`manifest.json.environment.qwen_execution`. The Qwen launcher exports the two
pinned `GDN3_*` values before preflight; direct CLI users must do the same.

## Measured flagship throughput

On two RTX 5060 Ti 16-GiB cards, the real production update at B=1/T=4096
measured 30.98 seconds steady-state. This includes the official 24-layer Qwen
model with 18 Package-B replacements, the frozen stock teacher, CE + full-vocab
KL + layerwise + auxiliary losses, backward, finite checks, transactional CPU
snapshots, and fused AdamW with phase-local CPU moment offload. That is 116.2
job steps/hour, or 58.1 steps/GPU-hour when both occupied cards are charged.
Peak allocated memory was 12.93 GiB on the student and 3.55 GiB on the teacher.

The strict optional Triton recurrence is used only for the canonical CUDA
segment shape and falls back to the FP32 reference path otherwise. The largest
remaining wall-clock risk is the complete long-context evaluation—especially
free generation without cross-call recurrence caching—not the 64-step heal.

## Run, shard, resume, and summarize

The launchers preflight before execution. A single-shard launch also summarizes
automatically:

```bash
bash research/kmd2_ablation/scripts/run_remote_tiny.sh \
  --out "$OUTPUT_PATH" --device cpu --job-index 0 --num-jobs 1

bash research/kmd2_ablation/scripts/run_remote_qwen.sh \
  --model "$MODEL_PATH" --tokenizer "$TOKENIZER_PATH" \
  --native-checkpoint "$NATIVE_CHECKPOINT" --data "$DATA_PATH" \
  --teacher-model "$TEACHER_MODEL_PATH" --out "$OUTPUT_PATH" \
  --student-device cuda:0 --teacher-device cuda:1 \
  --job-index 0 --num-jobs 1
```

For a metadata-gated one-update feasibility run, `--smoke` deterministically
writes a canonical derived config under `OUTPUT_PATH/.generated/`, switches to
the explicit synthetic-only objective, and does not require a teacher:

```bash
bash research/kmd2_ablation/scripts/run_remote_qwen.sh --smoke \
  --model "$MODEL_PATH" --native-checkpoint "$NATIVE_CHECKPOINT" \
  --data "$DATA_PATH" --out "$OUTPUT_PATH" --student-device cuda:0
```

For manual sharding, launch every `--job-index` in `[0, --num-jobs)` with the
same inputs and output root. Assignment is deterministic and exhaustive. A
minimal Slurm array is:

```bash
#SBATCH --array=0-7
bash research/kmd2_ablation/scripts/run_remote_qwen.sh \
  --model "$MODEL_PATH" --native-checkpoint "$NATIVE_CHECKPOINT" \
  --data "$DATA_PATH" --teacher-model "$TEACHER_MODEL_PATH" \
  --out "$OUTPUT_PATH" --student-device cuda:0 --teacher-device cuda:0 \
  --job-index "$SLURM_ARRAY_TASK_ID" --num-jobs 8
```

Launchers pass `--resume`. Only a provenance-, schema-, assignment-, and
identity-valid completed record is skipped. Failed, missing, interrupted, and
stale records rerun; conflicts go under `quarantine/`. Multi-shard workers do
not summarize, preventing a late partial shard from overwriting a complete
snapshot. After every array worker exits, run exactly one post-array coordinator
to publish the ledger (or rerun one launcher once with `--summarize`):

```bash
python -m research.kmd2_ablation.run_ablation summarize \
  --backend qwen --mode heal \
  --config research/kmd2_ablation/configs/qwen_exact_cache.json \
  --out "$OUTPUT_PATH"
```

## Output and failures

```text
OUTPUT_PATH/
  manifest.json
  jobs.json
  runs/<experiment>/<stage>/<seed>-<job-id>.json
  checkpoints/<job-id>/
  events/worker-<index>-of-<count>.jsonl
  quarantine/<job-id>/
  summary/ledger.jsonl
  summary/results.json
  summary/results.csv
```

Execution status is `completed` or `failed`; missing is derived from absence.
OOM, malformed input, non-finite loss/gradient, unsupported mask/streaming/
decode, stale provenance, and asset mismatch use distinct codes. The runner
never silently shrinks a batch, context, cache, dtype, state, or budget.

## Promotion interpretation

Three-seed screening is feasibility evidence. Five-seed Tiny promotion and the
complete three-arm Qwen RULER matrix can enter promotion only with matched
examples/cells. An addition requires the primary paired 95% interval lower
bound to clear `min_useful_addition` and all protected metrics to remain safe.
Factorial interactions require all four cells. Rotation/convolution use
separate reliance labels. Qwen Option 3 is ordered: surprise, otherwise
recency, otherwise `no_promote`.

No result is implied by these configs. Inspect the actual immutable ledger and
recorded intervals.
