# Package B Remaining Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development
> (if subagents are available) or superpowers:executing-plans.  Execute one task
> at a time in RED/GREEN/benchmark/commit order.

**Goal:** Finish only benchmark-justified optimization of the authoritative
complete `package-b-hola-w64` PyTorch execution while preserving every semantic
and compatibility contract.

**Architecture:** First create the repeatable complete-module performance gate.
Then add exact custom-autograd rematerialization around the proven generalized-
WY recurrence.  Decode, memory/layout, dense-mixer, and HOLA candidates remain
private until raw parity, complete-module parity, and a complete-module win are
all demonstrated.  Rejected candidates are removed; they never gain dispatch.

**Performance gate:** Each candidate uses three synchronized, interleaved arms
of the full Package-B/HOLA module: (1) `force_torch_recurrence=True` as the
semantic authority, (2) the current auto-dispatch with the new candidate absent
as the incremental performance baseline, and (3) current auto-dispatch plus the
private candidate adapter.  Correctness is checked against arm 1; performance
acceptance compares arm 3 only with arm 2 so earlier optimization gains cannot
be misattributed.  Report median milliseconds/tokens per second and peak CUDA
allocation at 32/64/128/256.  A candidate is accepted only with at least a 5%
median throughput improvement or a 10% peak-allocation reduction, no greater
than 2% regression in the other metric, and complete correctness.  Raw-op
profiling may select candidates but can never accept a production change.

---

## Chunk 0: Reproducible complete-module performance gate

### Task 1: Benchmark CLI before any new dispatch

**Files:**
- Create: `research/kmd2_ablation/scripts/benchmark_qwen_chunkwise.py`
- Modify: `tests/ablation/test_qwen_hybrid_chunkwise.py`

- [ ] RED: add
  `test_package_b_benchmark_cli_smoke_and_json_schema`; run
  `pytest -q tests/ablation/test_qwen_hybrid_chunkwise.py::test_package_b_benchmark_cli_smoke_and_json_schema`.
  It must fail because the CLI is absent.
- [ ] GREEN: implement deterministic seeds, warmup/iteration controls, CUDA
  synchronization around each sample, alternating baseline/candidate order,
  `max_memory_allocated` reset/collection, and JSON output.  It must instantiate
  the campaign-resolved `package-b-hola-w64` module with HOLA enabled and compare
  the three arms defined above; it must not benchmark a recurrence-only
  surrogate.
- [ ] Add a script-private `--candidate` selector with choices `auto`,
  `rematerialized`, `decode`, `direct-reads`, `mixer`, and `hola`.  Non-auto
  choices install a temporary benchmark-only internal adapter around the private
  prototype while executing the complete module.  This selector is not imported
  by the model, stored in config/checkpoints, or exposed as a public toggle.
  Add `--integrated` for post-integration measurement: arm 2 temporarily
  suppresses only the named production optimization, while arm 3 uses normal
  production dispatch and a call counter proves it executed.  `auto` is reserved
  for aggregate correctness/smoke and is never used to claim an incremental win.
- [ ] Verify the same node passes and run the smoke command
  `python research/kmd2_ablation/scripts/benchmark_qwen_chunkwise.py --mode inference --candidate auto --lengths 32 64 128 256 --warmup 1 --iterations 2 --json`.
  Expected: four complete-module records containing forced-authority,
  current-auto, and candidate medians/peak-byte counts, plus the incremental
  current-auto-to-candidate 5%/10% gate result.
- [ ] Add modes `training` and `decode`; decode starts from populated recurrent
  and HOLA cache and accepts `--batch-sizes 1 2`.  Add parser/adapter tests for
  all modes, candidates, and decode batch sizes.
- [ ] Run `pytest -q tests/ablation/test_qwen_hybrid_chunkwise.py -k benchmark_cli`;
  expected: all benchmark CLI smoke/parser tests pass.
- [ ] Commit: `bench: add reproducible Package B performance gate`.

## Chunk 1: Exact rematerialized training recurrence

### Task 2: Custom-autograd recurrence rematerialization

**Files:**
- Modify: `research/kmd2_ablation/qwen_hybrid_chunkwise.py`
- Modify: `tests/ablation/test_qwen_hybrid_chunkwise.py`

- [ ] RED: add
  `test_rematerialized_chunk_recurrence_forward_vjp_and_saved_tensors`; run its
  exact pytest node and expect import/attribute failure for the missing private
  wrapper.
- [ ] The test covers T=17/64 and history false/true.  It compares reads, final
  four-lane state, previous key/value endpoint carries, detached innovation
  scores, and unchanged history/count metadata with the independent dense oracle.
  It requests endpoint cotangents and VJPs for q/k/v/erase/write/gamma/lambda/
  state/previous-key/previous-value.
- [ ] Under `torch.autograd.graph.saved_tensors_hooks`, assert the wrapper saves
  exactly those ten original FP32 tensors plus only the bool history and integer
  counter carry—no interval products, triangular factors, or per-token state/key/
  value trace.  This structural assertion is separate from CUDA peak memory.
- [ ] GREEN: implement a private `torch.autograd.Function`.  Forward reuses the
  proven exact recurrence, marks innovation scores non-differentiable, and saves
  only the listed inputs/metadata.  Backward detaches/re-enables the requested
  inputs, reruns the exact recurrence under `torch.enable_grad()`, and calls
  `torch.autograd.grad`; it never performs numerical inversion.
- [ ] Run
  `pytest -q tests/ablation/test_qwen_hybrid_chunkwise.py::test_rematerialized_chunk_recurrence_forward_vjp_and_saved_tensors`;
  expected GREEN.  Then run
  `pytest -q tests/ablation/test_qwen_hybrid_chunkwise.py -k 'chunk and (vjp or oracle or interval or rematerial)'`;
  expected all selected recurrence tests pass.
- [ ] Add
  `test_rematerialized_chunk_recurrence_cuda_peak_memory`, comparing identical
  synchronized raw-op forward/backward runs.  Expected: parity and lower peak
  allocation than retaining the diagnostic trace; this result does not by itself
  authorize production dispatch.
- [ ] Commit: `perf: rematerialize exact chunk recurrence backward`.

### Task 3: Benchmark-gated fail-closed training dispatch

**Files:**
- Modify: `research/kmd2_ablation/qwen_hybrid_four_state.py`
- Modify: `research/kmd2_ablation/qwen_hybrid_chunkwise.py`
- Modify: `tests/ablation/test_qwen_hybrid_chunkwise.py`

- [ ] RED: add
  `test_private_package_b_rematerialized_training_full_vjp_fp32_bf16` and run
  its exact node.  Expected: failure because the benchmark-only complete-module
  adapter for the rematerialized wrapper is absent.
- [ ] GREEN: implement only the private adapter, without production dispatch.
  The test resolves the actual Package-B campaign config, keeps HOLA enabled,
  uses canonical batch 2 at T=17/64, and compares FP32 and bf16 modules against a
  cloned `force_torch_recurrence` baseline.  Compare the full output and every
  cache field including `conv_tail`, recurrent state/phase/factors/history/flags/
  counters, and all HOLA tensors/counters/epochs/staging/visible sets/scores.
  Compare VJPs for input and every trainable named Package-B/HOLA parameter.
- [ ] Run the exact private parity node; expected GREEN.  Add
  `test_package_b_training_two_optimizer_steps_exact_state`, comparing cloned
  module parameters, optimizer tensor/scalar state, outputs, and complete caches
  after two identical steps.
- [ ] Run
  `pytest -q tests/ablation/test_qwen_hybrid_chunkwise.py -k 'package_b_training'`;
  expected all training parity/optimizer tests pass.
- [ ] Run the complete training gate:
  `python research/kmd2_ablation/scripts/benchmark_qwen_chunkwise.py --mode training --candidate rematerialized --lengths 32 64 128 256 --warmup 5 --iterations 20 --json`.
  Continue to integration only if the private candidate meets the incremental
  5%/10% gate; otherwise keep the exact wrapper un-dispatched and record the
  rejected measurement.
- [ ] Winner-only RED: add
  `test_package_b_training_dispatch_full_vjp_fp32_bf16`; run it and expect the
  call counter to report zero production optimized grad-enabled calls.
- [ ] Winner-only GREEN: add one private canonical eligibility predicate and
  production dispatch.  Triton remains first, `force_torch_recurrence` is
  absolute, and unsupported device/dtype/optimization shape, mask, reset, packed
  boundary, nonuniform HOLA occupancy, or noncanonical carry falls back to the
  complete authoritative PyTorch loop.  Separately assert Package-B construction
  fails if native `K` cannot form four compact `K/4` complex-pair bands;
  full-width lanes are never a Package-B fallback.
- [ ] Rerun the production parity node and the training CLI with
  `--candidate rematerialized --integrated`; expected parity, a nonzero
  production call counter in arm 3 only, plus the same incremental win.  Then run
  the Package-B/HOLA/Triton focused suites and commit only a winner as
  `perf: dispatch rematerialized chunk recurrence for training`:
  `pytest -q tests/ablation/test_qwen_hybrid_chunkwise.py tests/ablation/test_qwen_hybrid_four_state.py tests/ablation/test_qwen_hybrid_hola.py tests/ablation/test_qwen_hybrid_triton.py`.
  Expected: zero failures.

## Chunk 2: Decode, layout, and bounded memory

### Task 4: Separate decode candidate

**Files:**
- Modify only during investigation: `research/kmd2_ablation/qwen_hybrid_chunkwise.py`
- Modify only for a winner: `research/kmd2_ablation/qwen_hybrid_four_state.py`
- Modify: `tests/ablation/test_qwen_hybrid_chunkwise.py`

- [ ] Establish baseline with
  `python research/kmd2_ablation/scripts/benchmark_qwen_chunkwise.py --mode decode --candidate auto --batch-sizes 1 2 --lengths 32 64 128 256 --warmup 5 --iterations 20 --json`
  for B=1/2 populated caches; retain the JSON result.
- [ ] RED: add `test_private_decode_step_raw_and_full_package_b_parity`; run its
  exact node and expect failure because the private vectorized T=1 step is absent.
  Cover all CMS phase offsets, tick/off-tick readable state, HOLA visibility,
  valid-token counting, and reset/packed-boundary fallback.
- [ ] GREEN prototype: implement the private exact T=1 step without dispatch.
  Run the raw/full node, token-by-token cached output/cache parity, and exhaustive
  input/parameter VJPs; expected all pass.
- [ ] Run the synchronized complete decode gate above against the existing B=1
  Triton and B=2 PyTorch paths, substituting `--candidate decode`.  Dispatch only
  if it meets 5%/10%; otherwise use
  `apply_patch` to remove the rejected prototype/test changes and retain baseline.
- [ ] For a winner, add narrow fail-closed dispatch, rerun the gate, and commit
  only after a winner-only RED/GREEN
  `test_package_b_decode_production_dispatch_call_selection` proves baseline/
  unsupported cases do not select it and the canonical case does.  Rerun the
  named gate with `--candidate decode --integrated`; arm 2 suppresses the
  specialization, arm 3 uses actual production dispatch, and the counter must be
  nonzero only in arm 3.  Commit
  `perf: specialize exact Package B cached decode`.  Before commit run
  `pytest -q tests/ablation/test_qwen_hybrid_chunkwise.py::test_private_decode_step_raw_and_full_package_b_parity tests/ablation/test_qwen_hybrid_chunkwise.py::test_package_b_decode_production_dispatch_call_selection tests/ablation/test_qwen_hybrid_hola.py::test_chunk_decode_reset_and_padding_continuation_parity tests/ablation/test_qwen_hybrid_triton.py::test_canonical_module_triton_dispatch_matches_forced_torch_vjp`;
  expected: four passing nodes (including parametrizations).  For rejection,
  verify `git diff -- research/kmd2_ablation/qwen_hybrid_chunkwise.py research/kmd2_ablation/qwen_hybrid_four_state.py tests/ablation/test_qwen_hybrid_chunkwise.py`
  contains no prototype/dispatch changes.  Make no code commit.

### Task 5: Direct reads without a per-token dense state trace

**Files:**
- Modify only during investigation: `research/kmd2_ablation/qwen_hybrid_chunkwise.py`
- Modify: `tests/ablation/test_qwen_hybrid_chunkwise.py`

- [ ] Profile raw allocations for interval products, triangular factors, state
  trace, read contraction, and solve at T=32/64/128/256.  This selects the next
  candidate but cannot accept it.
- [ ] RED: add `test_chunk_direct_reads_do_not_form_dense_state_trace`; run its
  exact node and expect failure because the distinct private direct-read path is
  absent.  Instrument its internal allocations/shapes and require that no
  `[B,T,H,4,K/4,V]` state trace is formed while matching oracle reads/final carry.
- [ ] GREEN prototype: contract interval products/rank updates directly into all
  sixteen logical reads and separately form the final carry.  The diagnostic
  oracle continues to expose per-token states.  Module-global mutable workspaces
  are forbidden; local preallocation/`out=` is no-grad only and must remain safe
  for checkpoint recomputation, reentrancy, and data parallelism.
- [ ] Test exact T=1/2/3 indexing, T=17/64, CMS/off-tick behavior, carries, and
  VJPs; then compare shorter exact subchunks and time-first layouts privately.
- [ ] Run inference and training complete-module gates at 32/64/128/256:
  `python research/kmd2_ablation/scripts/benchmark_qwen_chunkwise.py --mode inference --candidate direct-reads --lengths 32 64 128 256 --warmup 5 --iterations 20 --json`
  and the same command with `--mode training`.  Keep
  only a 5%/10% winner with no allocation regression.  Remove rejected paths.
- [ ] Before committing a winner run
  `pytest -q tests/ablation/test_qwen_hybrid_chunkwise.py::test_chunk_direct_reads_do_not_form_dense_state_trace`
  and separately
  `pytest -q tests/ablation/test_qwen_hybrid_chunkwise.py -k 't1 or t2 or t3 or vjp or package_b'`;
  expected: zero failures.  Add winner-only fail-closed production integration,
  repeat full parity, and rerun both gates with
  `--candidate direct-reads --integrated`; expected arm 3 exercises production
  dispatch and the private incremental win is retained.  Commit as
  `perf: form Package B chunk reads without state trace`.  For rejection, use
  `git diff` on the two task files to verify the prototype/test was removed.

## Chunk 3: Individually gated fusion candidates

### Task 6: Dense read/normalization/mixer candidate

**Files:**
- Modify during investigation; retain only for a winner: `research/kmd2_ablation/qwen_hybrid_four_state.py`
- Modify: `tests/ablation/test_qwen_hybrid_chunkwise.py`

- [ ] Profile recurrence, RMS normalization, sixteen-read dense mixer, and output
  projection on the complete path.  Proceed only if mixer work is material.
- [ ] RED: add `test_package_b_fused_reads_mixer_full_parity`; expect missing
  private fused function.  It compares full output/cache and exhaustive VJPs.
- [ ] GREEN private prototype, then run
  `pytest -q tests/ablation/test_qwen_hybrid_chunkwise.py::test_package_b_fused_reads_mixer_full_parity`;
  expected GREEN.  Run complete-module gates with
  `python research/kmd2_ablation/scripts/benchmark_qwen_chunkwise.py --mode inference --candidate mixer --lengths 32 64 128 256 --warmup 5 --iterations 20 --json`
  and the same command with `--mode training`.
  Preserve mixer parameter identity, shape, state-dict key, accumulation order,
  and gradients.  Remove a rejected prototype.
- [ ] For a private winner only, wire fail-closed production integration, repeat
  full parity, and rerun both complete-module gates with
  `--candidate mixer --integrated`; expected arm 3 exercises production dispatch
  and the incremental win remains.  Before commit run
  `pytest -q tests/ablation/test_qwen_hybrid_chunkwise.py tests/ablation/test_qwen_hybrid_four_state.py`;
  expected: zero failures.  Commit only:
  `perf: fuse exact Package B read normalization mixer`.  For rejection, verify
  `git diff` on both task files contains no candidate changes.

### Task 7: HOLA block-fusion candidate

**Files:**
- Modify during investigation; retain only for a winner: `research/kmd2_ablation/qwen_hybrid_hola.py`
- Modify: `tests/ablation/test_qwen_hybrid_chunkwise.py`

- [ ] Proceed only after recurrence/read parity and profiling show `scan_fast` is
  material.  RED: add `test_package_b_hola_fusion_exact_promotion_visibility_vjp`;
  expect missing private fusion.
- [ ] GREEN private prototype preserving staging, read-before-promotion,
  scoring, promotion exactly each 256 valid tokens, sink normalization, epochs,
  invalidation, visible-set ordering, and gradients.  Separately assert CMS-256
  clock behavior and HOLA-256 promotion behavior.
- [ ] Run
  `pytest -q tests/ablation/test_qwen_hybrid_chunkwise.py::test_package_b_hola_fusion_exact_promotion_visibility_vjp`;
  expected GREEN before any benchmark.
- [ ] Run inference/training gates at 32/64/128/256 with nonuniform occupancy and
  packed/reset fallback using
  `python research/kmd2_ablation/scripts/benchmark_qwen_chunkwise.py --mode inference --candidate hola --lengths 32 64 128 256 --warmup 5 --iterations 20 --json`
  and the same command with `--mode training`.  Remove a rejected prototype.
- [ ] For a private winner only, wire fail-closed production integration, repeat
  full parity, and rerun both complete-module gates with
  `--candidate hola --integrated`; expected arm 3 exercises production dispatch
  and the incremental win remains.  Before commit run
  `pytest -q tests/ablation/test_qwen_hybrid_chunkwise.py::test_package_b_hola_fusion_exact_promotion_visibility_vjp tests/ablation/test_qwen_hybrid_hola.py`;
  expected: zero failures.  Commit only:
  `perf: fuse exact Package B HOLA block scan`.  For rejection, verify `git diff`
  on both task files contains no candidate changes.

## Chunk 4: Exhaustive acceptance and compatibility

### Task 8: Full semantic, training, and schema acceptance

**Files:**
- Modify tests only if a demonstrated coverage gap remains.

- [ ] Run FP32 and bf16 complete Package-B tests at T=17/64; valid/invalid tokens;
  arbitrary partitions; CMS 16/64/256 boundaries; separate HOLA promotion at
  valid-token 256; packed documents/resets/fallback; and cached decode.
- [ ] Run independent dense-history reconstruction and compare previous factors,
  history flags/counters, all recurrent/phase carries, `conv_tail`, and every
  HOLA cache/visibility field.
- [ ] Run exhaustive VJPs for input and every named trainable parameter plus the
  two-step optimizer comparison.
- [ ] Assert public APIs/config IDs and cache dataclass are unchanged; compare
  parameter/state-dict names, shapes, dtypes, and counts; save/load a checkpoint
  and resume one training step with identical output/parameter/optimizer state.
  If not already covered, add
  `test_package_b_schema_checkpoint_resume_compatibility` to the chunkwise test
  file.  Run its exact node; expected: pass.
- [ ] Run:
  `pytest -q tests/ablation/test_qwen_hybrid_chunkwise.py tests/ablation/test_qwen_hybrid_four_state.py tests/ablation/test_qwen_hybrid_hola.py tests/ablation/test_qwen_hybrid_triton.py`.
  Expected: all Package-B/chunk/HOLA/Triton tests pass.
- [ ] Run the adjacent compatibility, shared math/campaign, checkpoint/cache, and
  unchanged GDN2 suites:
  `pytest -q tests/ablation/test_qwen_hybrid_components.py tests/ablation/test_qwen_hybrid_math.py tests/ablation/test_qwen_hybrid_shared.py tests/ablation/test_maximum_hybrid_campaign.py tests/ablation/test_architecture.py tests/ablation/test_qwen_architecture.py tests/ablation/test_qwen_backend.py tests/ablation/test_config.py tests/ablation/test_exact_cache_block.py tests/ablation/test_fast_scan_api.py tests/ablation/test_qwen_install.py tests/ablation/test_qwen_gdn2_triton.py`.
  Expected: zero failures and unchanged GDN2 results.
- [ ] Run all three final benchmark modes at 32/64/128/256 with warmup 5 and 20
  interleaved iterations:
  `python research/kmd2_ablation/scripts/benchmark_qwen_chunkwise.py --mode inference --candidate auto --lengths 32 64 128 256 --warmup 5 --iterations 20 --json`,
  the same with `--mode training`, and
  `python research/kmd2_ablation/scripts/benchmark_qwen_chunkwise.py --mode decode --candidate auto --batch-sizes 1 2 --lengths 32 64 128 256 --warmup 5 --iterations 20 --json`.
  Expected: aggregate smoke succeeds.  Then rerun every retained named candidate
  (`rematerialized`, `decode`, `direct-reads`, `mixer`, or `hola`) in its applicable
  mode(s) with the same lengths/warmup/iterations and `--integrated`.  Expected:
  arm 3 call counters prove production selection and every retained dispatch
  still meets its incremental gate against arm 2.
- [ ] Run `git diff --check` and inspect `git diff 2de240d --` to confirm GDN2,
  public/config/schema, and unrelated files are unchanged.
- [ ] Request final spec and code-quality review; fix every blocking issue and
  rerun all commands above from a fresh process.
- [ ] Commit any acceptance-only tests/docs as
  `test: exhaustively validate Package B optimized execution`.
