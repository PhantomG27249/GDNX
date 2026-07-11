# Portable KMD-2 Ablation Suite Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, and package a portable tiny/Qwen ablation suite that can train and evaluate every approved KMD-2 mechanism, including the HOLA-inspired bounded exact-cache lane, on a remote GPU server.

**Architecture:** A standard-library/PyTorch core owns validated configuration, deterministic tasks, variants, result storage, statistics, and packaging. The tiny backend implements the production-native recurrence for causal mechanism screens; the Qwen backend imports the production upgrade path, installs a strict `KMD2ExactCacheAttn` subclass, and runs paired native/cache heal jobs. The current `scan()` API remains unchanged while a separate score-returning fast entry point exposes exact outer-update magnitudes.

**Tech Stack:** Python 3.10+, PyTorch, pytest; optional Transformers/Triton/CUDA for Qwen and fast-scan tests; standard-library JSON/CSV/ZIP/hash tooling for portability.

---

## File map

Create or modify these focused units:

```text
pytest.ini                                      # restrict collection to tests/
gdn3/kmd2_fast_scan.py                         # optional score-returning fast API
research/kmd2_ablation/
|-- __init__.py                                # public version/API
|-- run_ablation.py                            # single CLI entry point
|-- config.py                                  # dataclasses, validation, canonical IDs
|-- inventory.py                               # native feature/source capability manifest
|-- variants.py                                # variant registry and compatibility gates
|-- qwen_variants.py                           # identity-gated Qwen heal recurrence wrappers
|-- exact_cache.py                             # pure-PyTorch recurrence/cache math and state
|-- qwen_exact_cache.py                        # native subclass, strict install/load, call guard
|-- metrics.py                                 # per-run metrics and paired statistics
|-- results.py                                 # schemas, atomic records, deterministic shards
|-- runner.py                                  # job expansion/execution/resume
|-- tiny_backend.py                            # tiny model/training/evaluation
|-- tiny_training.py                           # deterministic optimizer/checkpoint loop
|-- qwen_backend.py                            # Qwen load/heal/save/dry-run adapter
|-- qwen_training.py                           # matched paired-heal optimization loop
|-- qwen_checkpoint.py                         # strict atomic Qwen heal resume format
|-- summarize.py                              # ledgers, CSV/JSON, classifications
|-- bundle.py                                 # deterministic verified archives
|-- tasks/
|   |-- __init__.py                            # task registry
|   |-- mqar.py                                # atomic MQAR/load sweeps
|   |-- state_tracking.py                      # parity/modular/toggle FSM
|   |-- integration.py                         # irregular-time integration and RK4 oracle
|   |-- dynamics.py                            # drift/reversal and trajectory tasks
|   |-- local_binding.py                       # adjacent/separated copy and binding
|   |-- structured.py                          # compressible rule plus exceptions
|   |-- freshness.py                           # rebinding/stale-value task
|   |-- far_surprise.py                        # far fact plus distractor controls
|   |-- affine.py                              # symmetric affine associative regression
|   `-- ruler.py                               # Qwen RULER-style examples/scoring
|-- configs/
|   |-- screening.json                         # quick three-seed matrix
|   |-- promotion.json                         # five-seed/full matrix
|   |-- qwen_exact_cache.json                  # paired Qwen cache jobs
|   `-- smoke.json                             # CPU/offline smoke
|-- scripts/
|   |-- run_remote_tiny.sh                     # extracted-bundle tiny launch
|   `-- run_remote_qwen.sh                     # extracted-bundle Qwen launch
|-- requirements-tiny.txt
|-- requirements-qwen.txt
`-- README.md

tests/ablation/
|-- test_config.py
|-- test_inventory.py
|-- test_exact_cache_math.py
|-- test_exact_cache_block.py
|-- test_fast_scan_api.py
|-- test_qwen_install.py
|-- test_tasks.py
|-- test_tiny_backend.py
|-- test_tiny_training.py
|-- test_metrics.py
|-- test_variants.py
|-- test_results_runner.py
|-- test_cli.py
|-- test_qwen_backend.py
|-- test_summarize.py
|-- test_bundle.py
`-- test_remote_artifacts.py
```

The implementation must not modify `KMD2NativeAttn.forward`, the native recurrence, existing checkpoints, datasets, training results, or the legacy trainer.

---

## Chunk 1: Test foundation, configuration, and exact-cache core

### Task 1: Establish clean pytest collection and package skeleton

**Files:**
- Create: `pytest.ini`
- Create: `research/kmd2_ablation/__init__.py`
- Create: `tests/ablation/__init__.py`
- Create: `tests/ablation/test_config.py`

- [ ] **Step 1: Write the collection/import test**

```python
def test_package_imports_without_qwen_dependencies():
    import research.kmd2_ablation as suite
    assert suite.SUITE_VERSION
```

Run this in a subprocess whose import hook raises immediately for `transformers`
or `triton`. As pure modules are added, extend the subprocess to import
`config`, `inventory`, `exact_cache`, tasks, and `tiny_backend`; production
source inventory must be obtained by reading/hash parsing files, never importing
Qwen modules.

- [ ] **Step 2: Run it and record the expected failure**

Run: `python -m pytest tests/ablation/test_config.py -q`

Expected: FAIL because `research.kmd2_ablation` does not exist.

- [ ] **Step 3: Add `pytest.ini` and the minimal package**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -ra
markers =
    cuda: requires CUDA and Triton
    qwen_assets: requires external Qwen model/checkpoint/data assets
```

Expose only `SUITE_VERSION = "1.0.0"` initially. Do not import Transformers or Triton at package import time.

- [ ] **Step 4: Verify collection and import**

Run: `python -m pytest --collect-only -q`

Expected: archived `research/*_test.py` scripts are absent from collection.

Run: `python -m pytest tests/ablation/test_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pytest.ini research/kmd2_ablation/__init__.py tests/ablation
git commit -m "test: establish ablation suite harness"
```

### Task 2: Implement validated configuration, inventory, and job identity

**Files:**
- Create: `research/kmd2_ablation/config.py`
- Create: `research/kmd2_ablation/inventory.py`
- Modify: `tests/ablation/test_config.py`
- Create: `tests/ablation/test_inventory.py`

- [ ] **Step 1: Write failing configuration tests**

Cover canonical JSON stability, semantic experiment IDs, runtime fields excluded from IDs, every required schema field, threshold relationships, `w=0` only for `chunk_only`, real-eviction requirements, cache dtype/coordinate-frame/read rules, cache optimizer rules, and rejection of invalid Qwen streaming/mask modes. Use one complete `minimal_config_dict()` fixture shared by configuration tests; mutate that fixture rather than using abbreviated mapping placeholders.

```python
def test_semantic_id_ignores_output_path():
    raw = minimal_config_dict()
    a = ExperimentConfig.from_dict(raw | {"runtime": {"out": "a"}})
    b = ExperimentConfig.from_dict(raw | {"runtime": {"out": "b"}})
    assert a.experiment_id == b.experiment_id
```

- [ ] **Step 2: Run tests and confirm missing API failures**

Run: `python -m pytest tests/ablation/test_config.py tests/ablation/test_inventory.py -q`

Expected: FAIL on missing `ExperimentConfig` and `build_inventory`.

- [ ] **Step 3: Implement immutable dataclasses and canonical serialization**

Required interfaces:

```python
@dataclass(frozen=True)
class CacheConfig:
    width: int
    block_size: int
    score: str
    read: str
    read_init: str
    eps_cache: float
    coordinate_frame: str
    storage_dtype: str
    compute_dtype: str = "float32"
    inclusive: bool = True
    tie_policy: str = "score_desc_position_desc"
    lr_cache: float = 1e-3
    weight_decay_cache: float = 0.0

@dataclass(frozen=True)
class PromotionThresholds:
    min_gate_mean: float = 0.005
    min_gate_max: float = 0.02
    min_persistent_hit_rate: float = 0.25
    min_conditional_read_accuracy: float = 0.50
    min_shuffled_cache_dependence: float = 0.05
    min_adjacent_capacity_lcb: float = 0.05

@dataclass(frozen=True)
class ExperimentConfig:
    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        """Validate the complete schema and freeze nested values."""
    def semantic_dict(self) -> dict[str, Any]:
        """Return only fields that define scientific job identity."""
    @property
    def experiment_id(self) -> str:
        """Return SHA-256 of canonical compact semantic JSON."""
```

The complete fixture also declares versions, backend/run mode, baseline,
mechanism/variant, task parameters, seeds, budgets, optimizer/schedule,
model/state/FFN-match bounds, curriculum/extrapolation lengths, primary metric
and direction, addition/reliance/equivalence/synergy thresholds, protected
metrics, device/dtype preferences, and required stage.

Use sorted compact JSON and SHA-256. Validate all promotion thresholds before expanding jobs.

- [ ] **Step 4: Implement source-grounded inventory**

Record source hashes and explicit capabilities for convolution, rotation, shared-query `r_out`, channel decay, decoupled write, native exact-cache absence, inactive legacy `U,Vb` overlap, fast-score support, backend/task compatibility, and required external assets. Tests must fail on a changed expected source hash and verify every positive, negative, and legacy capability without importing Transformers/Triton.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/ablation/test_config.py tests/ablation/test_inventory.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add research/kmd2_ablation/config.py research/kmd2_ablation/inventory.py tests/ablation
git commit -m "feat: add validated ablation configuration"
```

### Task 3: Implement the independent native recurrence and exact-cache math

**Files:**
- Create: `research/kmd2_ablation/exact_cache.py`
- Create: `tests/ablation/test_exact_cache_math.py`
- Create: `tests/ablation/test_exact_cache_block.py`

- [ ] **Step 1: Write failing scalar-oracle tests**

Test forward and gradients for:

```text
S_bar = D*S
m = k^T*S_bar
u = beta_w*v - beta_e*m
S = S_bar + k*u^T
score = ||k||_2*||u||_2
```

Include unit, zero, and sub-epsilon keys; `r_out=1/4`; non-one-hot `out_mix`; fp32 and float64.

- [ ] **Step 2: Write failing cache-selection/read tests**

Cover independent per-head top-`w`, score-desc/position-desc ties, inclusive `j<=t` causality, future-prefix invariance, old-persistent plus current-block visibility, block-end survivor/eviction/clearing, bounded capacity, `w=0` chunk-only behavior, raw-`v` storage, bf16 storage round-trip, sink-only empty reads, non-finite rejection, finite gradients, and exact persistent/block byte formulas.

- [ ] **Step 3: Run tests and confirm failures**

Run: `python -m pytest tests/ablation/test_exact_cache_math.py tests/ablation/test_exact_cache_block.py -q`

Expected: FAIL on missing exact-cache APIs.

- [ ] **Step 4: Implement small pure functions first**

Required interfaces:

```python
def reference_scan_with_scores(q, k, v, decay, beta_e, beta_w, out_mix=None):
    """Return state-read output and detached exact outer-update scores."""

def deterministic_topw(scores, positions, valid, width):
    """Stable score-desc then position-desc indices per batch/head."""

@dataclass(frozen=True)
class ExactCacheState:
    keys: torch.Tensor
    values: torch.Tensor
    scores: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor

def merge_persistent_cache(state, block_k, block_v, block_scores,
                           block_positions, block_valid, width, storage_dtype):
    """Return detached-selection top-w state at the declared storage dtype."""

def cache_read_blocks(q_eff, query_positions, state, block_k, block_v,
                      block_scores, block_positions, block_valid, config,
                      gamma_q, gamma_k, sink_logit):
    """Read persistent plus current entries satisfying position <= query."""
```

Use fp32 compute, explicit storage casts, a manual bounded matmul/softmax, and the declared current-block merge order. Preserve gradients through gathered K/V but detach score/index selection.

- [ ] **Step 5: Implement exact read initialization**

Use gamma-one RMSNorm, `eps_cache`, zero sink, zero amplitude, and fixed-temperature control `sqrt(dk) * cos(q,k)`. Return diagnostics including selected positions, hit-ready indices, entropy, top-1 mass, sink mass, and persistent/block byte counts.

- [ ] **Step 6: Run oracle and block tests**

Run: `python -m pytest tests/ablation/test_exact_cache_math.py tests/ablation/test_exact_cache_block.py -q`

Expected: PASS; float64 `atol=1e-10, rtol=1e-8`, fp32 `atol=1e-6, rtol=1e-5`.

- [ ] **Step 7: Commit**

```bash
git add research/kmd2_ablation/exact_cache.py tests/ablation/test_exact_cache_*.py
git commit -m "feat: add KMD-2 exact-cache reference"
```

### Task 4: Add the score-returning fast-scan API without changing `scan()`

**Files:**
- Modify: `gdn3/kmd2_fast_scan.py`
- Create: `tests/ablation/test_fast_scan_api.py`

- [ ] **Step 1: Write API and parity tests**

The CPU test parses the module AST/source without importing Triton and asserts the original seven-argument `scan` wrapper remains and a separate `scan_with_update_norm` export exists. Executable import/parity tests use `pytest.importorskip("triton")` plus the `cuda` marker. CUDA tests compare state outputs, exact `||k||*||U||` scores, selected indices on separated/tied fixtures, and q/k/v/decay/gate/out-mix gradients.

- [ ] **Step 2: Run the CPU test and confirm the new symbol is missing**

Run: `python -m pytest tests/ablation/test_fast_scan_api.py -q`

Expected: the AST test fails because the new symbol is absent; executable tests SKIP when Triton/CUDA is unavailable rather than failing during collection.

- [ ] **Step 3: Refactor through an internal core while preserving public `scan`**

```python
def _scan_impl(q, k, v, g, beta_e, beta_w, out_mix=None):
    y, _ = _scan_core(q, k, v, g, beta_e, beta_w, out_mix,
                      return_scores=False)
    return y

def _scan_with_update_norm_impl(q, k, v, g, beta_e, beta_w, out_mix=None):
    return _scan_core(q, k, v, g, beta_e, beta_w, out_mix,
                      return_scores=True)

scan = torch.compile(_scan_impl)
scan_with_update_norm = torch.compile(_scan_with_update_norm_impl)
```

Collect score chunks from `k.norm(dim=-1) * U.norm(dim=-1)`, trim padding, reshape to `[B,T,H]`, and detach only the returned score tensor. Do not change operation order on the existing output path.

- [ ] **Step 4: Run local tests and the remote-gated CUDA test when available**

Run: `python -m pytest tests/ablation/test_fast_scan_api.py -q`

Expected locally: PASS with CUDA cases skipped.

Remote command: `python -m pytest tests/ablation/test_fast_scan_api.py -m cuda -q`

Expected: forward relMSE `<2e-3`, gradient relMSE `<1e-2`, exact fixture indices.

- [ ] **Step 5: Commit**

```bash
git add gdn3/kmd2_fast_scan.py tests/ablation/test_fast_scan_api.py
git commit -m "feat: expose KMD-2 update scores"
```

### Task 5: Implement strict Qwen subclass installation and call guards

**Files:**
- Create: `research/kmd2_ablation/qwen_exact_cache.py`
- Create: `tests/ablation/test_qwen_install.py`

- [ ] **Step 1: Write failing subclass/transfer tests**

Use a minimal Qwen-compatible config fixture with optional dependencies installed. Assert `KMD2ExactCacheAttn` subclasses `KMD2NativeAttn`, inherited tensors are exactly transferred, only cache names are new, baseline output/hidden/inherited-parameter gradients match at zero amplitude, amplitude gradient opens, and resume keys are strict. The pure/tiny import subprocess must still pass when this Qwen module is unavailable.

- [ ] **Step 2: Write failing runner-argument guard tests**

Accept only absent/all-one masks, monotonic positions, `use_cache=False`, and empty cache state. Reject padding, packing, resets, decode state, and nonmonotonic positions before the model is called.

- [ ] **Step 3: Implement `KMD2ExactCacheAttn` and installer**

```python
class KMD2ExactCacheAttn(KMD2NativeAttn):
    @classmethod
    def from_native(cls, native, model_config, cache_config):
        """Construct with strict inherited transfer and new cache defaults."""
    def _scan(self, q, k, v, g, beta_e, beta_w):
        """Return native state read plus the bounded exact-cache branch."""

def load_native_then_install(model, manager, model_config, cache_config,
                             native_checkpoint, cache_resume=None):
    """Apply native upgrade/load, replace layers, then optionally load cache."""
def strict_load_cache_resume(model, checkpoint, expected_job_id):
    """Require exact cache names, shapes, schema, and scientific identity."""
def validate_full_recompute_call(**kwargs):
    """Reject padding, packing, resets, decode state, and cross-call cache."""
```

The subclass reuses inherited `forward`; `_scan` obtains state output/scores, computes `q_eff`, adds `lambda*y_cache`, and publishes detached diagnostics for metrics. Tests spy on the exact order: apply native manager, strict-load native checkpoint, instantiate subclass from native/config, strict inherited transfer, then optional strict cache resume. A suite-owned guarded forward wrapper must call `validate_full_recompute_call` before every model forward.

- [ ] **Step 4: Add dedicated cache optimizer helpers**

Return cache parameters with declared `lr_cache`, shared AdamW betas/epsilon/schedule, zero weight decay, and a post-step `[0,1]` projection for amplitudes. Save and restore cache parameters plus optimizer state strictly.

- [ ] **Step 5: Run tests**

Add a nonzero-amplitude reference/fast integration fixture across scan/cache block boundaries for `r_out=1/4` and non-one-hot `out_mix`. Compare storage-cast selected indices, state/cache/final outputs, and gradients for recurrence inputs, RMSNorm gammas, sink, and amplitude; mark the true fast case `cuda`.

Run: `python -m pytest tests/ablation/test_qwen_install.py tests/ablation/test_exact_cache_block.py tests/ablation/test_fast_scan_api.py -q`

Expected: PASS without loading a model snapshot.

- [ ] **Step 6: Commit**

```bash
git add research/kmd2_ablation/qwen_exact_cache.py tests/ablation/test_qwen_install.py tests/ablation/test_fast_scan_api.py
git commit -m "feat: install exact cache into native KMD-2"
```

---

## Chunk 2: Deterministic tasks, tiny training, metrics, and variant screens

### Task 6: Implement deterministic task generators

**Files:**
- Create: `research/kmd2_ablation/tasks/__init__.py`
- Create: `research/kmd2_ablation/tasks/mqar.py`
- Create: `research/kmd2_ablation/tasks/state_tracking.py`
- Create: `research/kmd2_ablation/tasks/integration.py`
- Create: `research/kmd2_ablation/tasks/dynamics.py`
- Create: `research/kmd2_ablation/tasks/local_binding.py`
- Create: `research/kmd2_ablation/tasks/structured.py`
- Create: `research/kmd2_ablation/tasks/freshness.py`
- Create: `research/kmd2_ablation/tasks/far_surprise.py`
- Create: `research/kmd2_ablation/tasks/affine.py`
- Create: `tests/ablation/test_tasks.py`

- [ ] **Step 1 RED: Write `test_episode_contract_and_registry`**

Specify a frozen batch record with batch/time-leading inputs, targets, boolean
loss/query masks, exact source spans, strata tensors and metadata; unknown task
names must fail.

- [ ] **Step 2 RED: Run the contract test**

Run: `python -m pytest tests/ablation/test_tasks.py -k episode_contract -q`

Expected: FAIL importing missing `EpisodeBatch`.

- [ ] **Step 3 GREEN: Implement only the episode record and registry**

- [ ] **Step 4 GREEN: Rerun the contract test**

Expected: the selected contract test PASS.

- [ ] **Step 5 RED: Write `test_state_tracking_exact_ops_and_ood`**

Cover parity, modular counters, and toggle FSM (`HOLD/QUERY`,
`SET0/SET1/TOGGLE/NOOP/QUERY`), balanced exact labels, reset/overwrite, seeds,
and 2x/4x lengths.

- [ ] **Step 6 RED: Run the state-tracking test**

Run: `python -m pytest tests/ablation/test_tasks.py -k state_tracking -q`

Expected: FAIL on missing generator.

- [ ] **Step 7 GREEN: Implement only state tracking**

- [ ] **Step 8 GREEN: Rerun the state-tracking test**

Expected: selected state-tracking tests PASS.

- [ ] **Step 9 RED: Write `test_irregular_integration_matches_rk4`**

Cover the float64 `expm1`-stable analytic solution of `dh/dt=-a*h+u(t)`,
piecewise-linear forcing, tiny/large gaps, curvature, boundaries, seeds and
withheld targets.

- [ ] **Step 10 RED: Run the integration test**

Run: `python -m pytest tests/ablation/test_tasks.py -k irregular_integration -q`

Expected: FAIL on missing analytic generator/oracle.

- [ ] **Step 11 GREEN: Implement analytic integration plus validation-only RK4**

- [ ] **Step 12 GREEN: Rerun the integration test**

Expected: PASS at the declared float64 tolerance.

- [ ] **Step 13 RED: Write separate `drift_reversal` and `trajectory` tests**

Cover balanced reversal/change points, queries before targets, exact
strata and 2x/4x horizons.

- [ ] **Step 14 RED: Run the dynamics tests**

Run: `python -m pytest tests/ablation/test_tasks.py -k "drift_reversal or trajectory" -q`

Expected: two missing-generator failures.

- [ ] **Step 15 GREEN: Implement only the two dynamics generators**

- [ ] **Step 16 GREEN: Rerun the dynamics tests**

Expected: both selected families PASS.

- [ ] **Step 17 RED: Write separate `local_binding` and `mqar` tests**

Cover adjacent/separated binding, motifs, delayed copy, overwrite, exact spans and
distance/load cells below/near/above width.

- [ ] **Step 18 RED: Run the binding/MQAR tests**

Run: `python -m pytest tests/ablation/test_tasks.py -k "local_binding or mqar" -q`

Expected: missing-generator failures.

- [ ] **Step 19 GREEN: Implement only local binding and MQAR**

- [ ] **Step 20 GREEN: Rerun the binding/MQAR tests**

Expected: both selected families PASS.

- [ ] **Step 21 RED: Write separate `structured`, `far_surprise`, `freshness`, and `affine` tests**

Cover rule/exception interventions, queried versus
high-score distractors, latest/historical rebinding, adversarial stale scores,
and symmetric `x,-x` writes with independent intercepts/no constant q/k input.
Every family checks seed identity/variation, visibility, exact spans, balanced
strata and train/2x/4x splits.

- [ ] **Step 22 RED: Run the four remaining task tests**

Run: `python -m pytest tests/ablation/test_tasks.py -k "structured or far_surprise or freshness or affine" -q`

Expected: four missing-generator failures.

- [ ] **Step 23 GREEN: Implement only those four focused generators**

- [ ] **Step 24 GREEN: Rerun the four task tests**

Expected: all four selected families PASS.

- [ ] **Step 25: Run the complete task suite and commit**

Run: `python -m pytest tests/ablation/test_tasks.py -q`

Expected: all named task-family invariants PASS; collection count is asserted
inside the test module so an omitted family cannot appear green.

```bash
git add research/kmd2_ablation/tasks tests/ablation/test_tasks.py
git commit -m "feat: add deterministic KMD-2 tasks"
```

### Task 7: Implement the exact-native tiny backend and trainer

**Files:**
- Create: `research/kmd2_ablation/tiny_backend.py`
- Create: `research/kmd2_ablation/tiny_training.py`
- Create: `tests/ablation/test_tiny_backend.py`
- Create: `tests/ablation/test_tiny_training.py`

- [ ] **Step 1 RED: Write `test_tiny_api_shapes_and_validation`**

The explicit `TinyKMD2Config` constructor declares `d_model`, `heads`, `dk`,
`dv`, `layers`, `vocab_size`, `r_out`, `mimo_rank`, mechanism gates and dtype.
`TinyFactors` holds `q[B,T,H,Q,dk]`, `k[B,T,H,R,dk]`,
`v[B,T,H,R,dv]`, `decay[B,T,H,dk]`, `beta_e/beta_w[B,T,H,R]`,
`out_mix[B,T,H,Q]` for the native/shared-query path or channelwise
`out_mix[B,T,H,R,dv]` plus `read_gate[B,T,H,R,dv]` for true MIMO,
`valid[B,T]`, and `positions[B,T]`. Native shared-query mode has
`R=1,Q=r_out`; true MIMO has `R=Q=mimo_rank`, a base `v/z` projection expanded
by Mamba-3-style `M_V/M_Z` scalings, nonlinear rank gates, and an `M_O`
contraction; simultaneous true MIMO and shared-query widening is rejected.
`state=None` creates fp32 zeros
`[B,H,dk,dv]`; declared boundaries reset it.

`TinyKMD2Cell.forward(factors, state=None, boundaries=None)` returns frozen
`TinyCellOutput(read[B,T,H,dv], final_state[B,H,dk,dv], scores[B,T,H],
state_read, cache_read, selected_positions, sink_mass, state_bytes,
cache_persistent_bytes, cache_block_bytes)`. `TinyModelOutput` contains
`logits[B,T,vocab_or_output]`, scalar optional loss, final per-layer states, and
cell outputs. `TinyKMD2Model.forward(input_ids=None, factors=None,
targets=None, loss_mask=None, boundaries=None)` accepts exactly one of token or
direct-factor input. Write shape, post-update-read orientation, residual flow,
head merge, loss-mask, and invalid-input assertions.

- [ ] **Step 2 RED: Run the tiny API test**

Run: `python -m pytest tests/ablation/test_tiny_backend.py -k api_shapes -q`

Expected: FAIL importing missing `TinyFactors`/`TinyKMD2Cell`.

- [ ] **Step 3 GREEN: Implement only the records and minimal SISO/no-cache cell/model**

- [ ] **Step 4 GREEN: Rerun the tiny API test**

Expected: selected API test PASS.

- [ ] **Step 5 RED: Write independent and production native-parity tests**

First compare forward, hidden/input gradients, and recurrence-parameter
gradients to the independent float64/fp32 oracle for `r_out=1/4` and non-one-hot
mixing. When Transformers is installed, also construct a minimal production
`KMD2NativeAttn` fixture and compare its `_scan`; otherwise skip only this
production-import case explicitly.

- [ ] **Step 6 RED: Run native-parity tests**

Run: `python -m pytest tests/ablation/test_tiny_backend.py -k native_parity -q`

Expected: FAIL on the first recurrence/output/gradient mismatch.

- [ ] **Step 7 GREEN: Implement only exact native recurrence/parity corrections**

- [ ] **Step 8 GREEN: Rerun native-parity tests**

Expected: PASS at float64/fp32 tolerances; optional production case PASS or
explicitly SKIP for missing dependency only.

- [ ] **Step 9 RED: Write disabled-identity tests for optional native features/cache**

For convolution, production data-dependent rotation, constant/non-cumulative/
fixed-RoPE/moving-frame controls, shared-query `r_out`, channel decay, write
offset, and exact cache, require equality to the native cell when disabled. Use
no learned absolute positions on extrapolation tasks; cache normalization stays
branch-local.

- [ ] **Step 10 RED: Run disabled-identity tests**

Run: `python -m pytest tests/ablation/test_tiny_backend.py -k disabled_identity -q`

Expected: FAIL on missing optional feature paths.

- [ ] **Step 11 GREEN: Implement only zero-gated/bypass optional paths**

- [ ] **Step 12 GREEN: Rerun disabled-identity tests**

Expected: all identity cases PASS.

- [ ] **Step 13 RED: Write active-effect/finite-gradient tests**

- [ ] **Step 14 RED: Run active-effect/finite-gradient tests**

Run: `python -m pytest tests/ablation/test_tiny_backend.py -k active_effect -q`

Expected: FAIL because active feature/cache settings do not yet change output.

- [ ] **Step 15 GREEN: Implement only active optional paths**

- [ ] **Step 16 GREEN: Rerun active-effect/finite-gradient tests**

Expected: deterministic output changes and finite mechanism gradients PASS.

- [ ] **Step 17 RED: Write optimizer/cache/checkpoint tests**

Test memory versus cache AdamW groups, dedicated `lr_cache`, shared
betas/epsilon/schedule, cache weight decay zero, exactly-zero amplitude with a
finite opening gradient, post-step `[0,1]` projection, rejection of out-of-range
resume, and strict serialization of cache/model/optimizer/scheduler/RNG/job ID.

- [ ] **Step 18 RED: Run optimizer/cache/checkpoint tests**

Run: `python -m pytest tests/ablation/test_tiny_training.py -k "optimizer or checkpoint" -q`

Expected: FAIL importing missing `TinyTrainer`.

- [ ] **Step 19 GREEN: Implement deterministic optimizer and strict checkpoint semantics**

Use seeded generators, fixed updates/tokens, clipping, finite checks, atomic
checkpoints, and no silent shape/batch/length fallback.

- [ ] **Step 20 GREEN: Rerun optimizer/cache/checkpoint tests**

Expected: selected trainer tests PASS.

- [ ] **Step 21 RED: Write deterministic 10-step/resume learning smoke**

Write a 10-step CPU fixture whose loss decreases, repeat it byte-identically,
resume at step 5 to the same final tensors/metrics, and exercise token and
direct-factor batches through the same trainer.

- [ ] **Step 22 RED: Run deterministic learning smoke**

Run: `python -m pytest tests/ablation/test_tiny_training.py -k learning_smoke -q`

Expected: FAIL before the complete train/evaluate loop exists.

- [ ] **Step 23 GREEN: Implement only the minimal train/evaluate loop**

- [ ] **Step 24 GREEN: Rerun deterministic learning smoke**

Expected: loss decreases and repeat/resume outputs match.

- [ ] **Step 25: Run all tiny tests and commit**

Run: `python -m pytest tests/ablation/test_tiny_backend.py tests/ablation/test_tiny_training.py -q`

Expected: PASS on CPU.

```bash
git add research/kmd2_ablation/tiny_backend.py research/kmd2_ablation/tiny_training.py tests/ablation/test_tiny_backend.py tests/ablation/test_tiny_training.py
git commit -m "feat: add tiny KMD-2 training backend"
```

### Task 8: Implement metrics and statistical decisions

**Files:**
- Create: `research/kmd2_ablation/metrics.py`
- Create: `tests/ablation/test_metrics.py`

- [ ] **Step 1 RED: Write exact base/task metric fixtures**

Use exact numeric fixtures for token/episode/chance-adjusted state accuracy;
integration error by gap/curvature; drift steady-state error, lag, overshoot and
recovery; trajectory smooth/change-point error and phase lag; affine query,
intercept and slope MSE; distance/load bins; queried-span hit/top-1,
conditional-read and cache-only value exact match, wrong-key rate, selector
AUPRC, survival, stale errors, freshness update latency, duplicate occupancy,
old/new mass, explicit rule/exception stratum metrics, entropy/effective
support, sink mass, cache/state norms,
retention/eviction/score statistics, separate persistent/block bytes, latency,
throughput and VRAM. Test both metric directions, empty strata, and non-finite
input.

- [ ] **Step 2 RED: Run base/task metric fixtures**

Run: `python -m pytest tests/ablation/test_metrics.py -k accumulators -q`

Expected: FAIL importing missing accumulators.

- [ ] **Step 3 GREEN: Implement only metric accumulators**

- [ ] **Step 4 GREEN: Rerun base/task metric fixtures**

Expected: selected exact numeric fixtures PASS.

- [ ] **Step 5 RED: Write paired-bootstrap fixtures**

Test deterministic hierarchical resampling of matched seeds then matched
examples, exact matched identities, degenerate/singleton inputs, rejected
unmatched or empty inputs, both directions, and stable interval bytes.

- [ ] **Step 6 RED: Run paired-bootstrap fixtures**

Run: `python -m pytest tests/ablation/test_metrics.py -k paired_bootstrap -q`

Expected: FAIL importing missing bootstrap API.

- [ ] **Step 7 GREEN: Implement only point estimate, interval, counts, and paired deltas**

- [ ] **Step 8 GREEN: Rerun paired-bootstrap fixtures**

Expected: selected bootstrap fixtures PASS deterministically.

- [ ] **Step 9 RED: Write ordered addition/reliance/factorial fixtures**

```python
I = direction * (M11 - M10 - M01 + M00)
d_current = direction * (M11 - M01)
d_feature_off = direction * (M10 - M00)
```

Reject missing/unmatched cells. Add exact fixtures for ordered
failed/harmful/synergistic/incremental/replacement-only/redundant/inconclusive
addition labels and failed/harmful-current/relied-on/dispensable/inconclusive
reliance labels.

- [ ] **Step 10 RED: Run classification fixtures**

Run: `python -m pytest tests/ablation/test_metrics.py -k classification -q`

Expected: FAIL importing missing classifiers.

- [ ] **Step 11 GREEN: Implement only ordered classifiers and factorial checks**

- [ ] **Step 12 GREEN: Rerun classification fixtures**

Expected: all ordered labels and missing-cell rejections PASS.

- [ ] **Step 13 RED: Write Option-3 decision fixtures**

Encode separate surprise and recency fixtures covering long native gain,
surprise-versus-recency, short/8K/exact protections, freshness against both
controls, CE/KL/non-finite limits, `w=64` plus adjacent capacity, amplitude
mean/max, persistent hit, conditional read, shuffled-value dependence, complete
cells, and ordered `surprise` then `recency` then `no_promote`. Assert every
individual rejection code.

Fixtures pin the approved numbers and formulas: long macro and two long cells
have LCB `+0.10`; surprise-versus-recency LCB `+0.05`; 512-4K macro `-0.02`,
individual short and 8K `-0.03`, episode exact `-0.05`; both freshness effects
`-0.02` against both controls; `CE_variant-CE_native` UCB `0.02` and KL UCB
`max(0.005, 0.05*mean(KL_native))`; `w=64` LCB `+0.10` and `w=32|128` LCB
`+0.05`; gate mean/max `0.005/0.02`, hit/read/shuffle LCBs `0.25/0.50/0.05`.

- [ ] **Step 14 RED: Run Option-3 fixtures**

Run: `python -m pytest tests/ablation/test_metrics.py -k option3 -q`

Expected: FAIL importing missing Option-3 decision API.

- [ ] **Step 15 GREEN: Implement only ordered surprise/recency/no-promote gates**

- [ ] **Step 16 GREEN: Rerun Option-3 fixtures**

Expected: every pass and single-gate rejection fixture PASS.

- [ ] **Step 17: Run tests and commit**

Run: `python -m pytest tests/ablation/test_metrics.py -q`

Expected: PASS.

```bash
git add research/kmd2_ablation/metrics.py tests/ablation/test_metrics.py
git commit -m "feat: add paired ablation statistics"
```

### Task 9: Implement variants and staged screen expansion

**Files:**
- Create: `research/kmd2_ablation/variants.py`
- Create: `research/kmd2_ablation/qwen_variants.py`
- Modify: `research/kmd2_ablation/tiny_backend.py`
- Modify: `research/kmd2_ablation/tiny_training.py`
- Create: `tests/ablation/test_variants.py`
- Modify: `tests/ablation/test_tiny_backend.py`

- [ ] **Step 1 RED: Write `test_registry_has_every_declared_arm`**

Require native baseline; rotation and convolution reliance controls; trapezoid;
B/C q/k bias, diagonal-rescale and constant-coordinate oracle; corrected
momentum; causal lookahead; state-size and true-MIMO sweeps; exact-cache off,
current-block-only, exact/coupled/residual/write/recency/reservoir/oracle
selectors, unit/fixed/RMS reads, bf16/fp32 storage, pre-rotation diagnostic,
per-slot-read interaction, unbounded ceiling, width/block sweeps, and
cache-rotation/cache-`r_out` factorials. Metadata declares addition/reliance/
diagnostic, incremental versus replacement, compatible backends/tasks, changed
parameters/state, and required stage.

- [ ] **Step 2 RED: Run registry inventory**

Run: `python -m pytest tests/ablation/test_variants.py -k registry -q`

Expected: FAIL importing missing registry.

- [ ] **Step 3 GREEN: Implement only registry records and lookup**

- [ ] **Step 4 GREEN: Rerun registry inventory**

Expected: all declared entries/metadata PASS.

- [ ] **Step 5 RED: Write trapezoid tiny/Qwen identity, active-effect, and interaction tests**

For trapezoid, implement differentiable factor carry, boundary clearing,
`rho_head=0` native equality, projected `[0,1]` gating, nonzero effect/gradient,
and trapezoid x convolution only after individual promotion. The Qwen wrapper
must subclass native, reuse inherited forward/projections, strict-transfer
tensors, and force the Python loop.

- [ ] **Step 6 RED: Run trapezoid tests**

Run: `python -m pytest tests/ablation/test_variants.py -k trapezoid -q`

Expected: FAIL on missing recurrence/wrapper.

- [ ] **Step 7 GREEN: Implement only tiny/Qwen trapezoid paths**

- [ ] **Step 8 GREEN: Rerun trapezoid tests**

Expected: identity, active effect, boundaries, gradients and compatibility PASS.

- [ ] **Step 9 RED: Write B/C bias tiny/Qwen identity, active-effect, and control tests**

Use separate post-normalization zero amplitudes; diagonal rescaling has equal
parameter count but cannot add a constant coordinate; constant-coordinate arm
is diagnostic. Bias x trapezoid/convolution stays gated on individual wins.

- [ ] **Step 10 RED: Run B/C bias tests**

Run: `python -m pytest tests/ablation/test_variants.py -k bc_bias -q`

Expected: FAIL on missing bias paths.

- [ ] **Step 11 GREEN: Implement only tiny/Qwen B/C bias and controls**

- [ ] **Step 12 GREEN: Rerun B/C bias tests**

Expected: identity, affine active effect, gradients and controls PASS.

- [ ] **Step 13 RED: Write corrected-momentum tiny/Qwen tests**

Require decayed velocity, lookahead state, `gamma=0` native equality, doubled
dynamic-state accounting, active effect/gradient, and decay/erase interaction
gating. A Qwen fixture enables the existing fast scan and spies on dispatch;
the momentum wrapper must force the Python reference recurrence (or reject the
request explicitly) and must never call the existing fast scan.

- [ ] **Step 14 RED: Run corrected-momentum tests**

Run: `python -m pytest tests/ablation/test_variants.py -k momentum -q`

Expected: FAIL on missing momentum paths.

- [ ] **Step 15 GREEN: Implement only tiny/Qwen corrected momentum**

- [ ] **Step 16 GREEN: Rerun corrected-momentum tests**

Expected: identity/equation/state-byte/active-effect cases PASS.

- [ ] **Step 17 RED: Write causal-lookahead tiny/Qwen tests**

Require `v_target=v_t+rho_t*P(v_t-v_prev)`, boundary reset, zero identity,
active effect/gradient, and convolution/trapezoid interaction gating.

- [ ] **Step 18 RED: Run causal-lookahead tests**

Run: `python -m pytest tests/ablation/test_variants.py -k lookahead -q`

Expected: FAIL on missing lookahead paths.

- [ ] **Step 19 GREEN: Implement only tiny/Qwen causal lookahead**

- [ ] **Step 20 GREEN: Rerun causal-lookahead tests**

Expected: identity/boundary/active-effect cases PASS.

- [ ] **Step 21 RED: Write true-MIMO equation/invariance tests**

Test the declared simultaneous rank-`R` equations, exact `R=1` SISO forward/
gradient equality, common slot-permutation invariance, scaling, separation from
shared-query `r_out=4`, and rejection as a Qwen heal arm.

- [ ] **Step 22 RED: Run true-MIMO tests**

Run: `python -m pytest tests/ablation/test_variants.py -k true_mimo -q`

Expected: FAIL on missing MIMO recurrence.

- [ ] **Step 23 GREEN: Implement only true-MIMO tiny recurrence**

- [ ] **Step 24 GREEN: Rerun true-MIMO tests**

Expected: equation, `R=1`, permutation and scaling cases PASS.

- [ ] **Step 25 RED: Write moving-frame equivalence tests**

Require equivalence under pair-tied channel decay and expected non-equivalence
under independently learned channel decay.

- [ ] **Step 26 RED: Run moving-frame tests**

Run: `python -m pytest tests/ablation/test_variants.py -k moving_frame -q`

Expected: FAIL before equivalence helper/control exists.

- [ ] **Step 27 GREEN: Implement only moving-frame controls and equivalence diagnostics**

- [ ] **Step 28 GREEN: Rerun moving-frame tests**

Expected: tied/non-tied cases PASS.

- [ ] **Step 29 RED: Write state-size/MIMO exact parameter-matching tests**

Test raw fixed-FFN and matched state-size, MIMO and factorial arms using exact
instantiated trainable counts/state bytes; search divisible-by-8 finite FFN
bounds and require `max(0.5%,1024)` tolerance plus no-legal-match rejection.

- [ ] **Step 30 RED: Run parameter-matching tests**

Run: `python -m pytest tests/ablation/test_variants.py -k parameter_match -q`

Expected: FAIL on missing matcher.

- [ ] **Step 31 GREEN: Implement only exact instantiated FFN matcher**

- [ ] **Step 32 GREEN: Rerun parameter-matching tests**

Expected: raw, matched, and rejection cases PASS.

- [ ] **Step 33 RED: Write exact-cache versus equal-state-byte control tests**

For every cache width instantiate an equally sized recurrent-state increase and
report unavoidable byte mismatch; generic parameter matching is forbidden.

- [ ] **Step 34 RED: Run equal-state-byte tests**

Run: `python -m pytest tests/ablation/test_variants.py -k equal_state_bytes -q`

Expected: FAIL on missing control constructor.

- [ ] **Step 35 GREEN: Implement only equal-state-byte control construction**

- [ ] **Step 36 GREEN: Rerun equal-state-byte tests**

Expected: exact selected state sizes and byte reports PASS.

- [ ] **Step 37 RED: Write cache/no-op/stage-expansion tests**

Test `w=0` restrictions, at least two blocks and actual eviction, identical
capacity/read/gate/budget controls, diagnostic-only oracle/pre-rotation status,
no post-hoc reliance claims, disabled identity, active output change, and
rejection of native-present/no-op variants. Assert exact serial stage order and
job counts: selector replay; read; `w={0,8,16,32,64,128}`; winner-only
`C_cache={64,128,256}`; five-seed tiny promotion; four-cell interactions; then
paired Qwen native/recency/surprise. Failed gates emit no downstream jobs and
no Cartesian product.

Expansion fixtures pin three paired screening seeds, five paired tiny-promotion
seeds, and three paired Qwen-heal seeds with at least 64 matched RULER episodes
per cell. Tiny selector/read promotion requires five-point LCB; short accuracy
and both freshness comparisons require LCB at least `-0.02`. Exact expected
job IDs/counts are asserted from these inputs.

- [ ] **Step 38 RED: Run cache/no-op/stage tests**

Run: `python -m pytest tests/ablation/test_variants.py -k "cache_compat or stage_expansion" -q`

Expected: FAIL on missing compatibility/expansion functions.

- [ ] **Step 39 GREEN: Implement only compatibility validation and serial expansion**

- [ ] **Step 40 GREEN: Rerun cache/no-op/stage tests**

Expected: exact gates/job IDs/counts and no-Cartesian-product cases PASS.

- [ ] **Step 41: Run tests and commit**

Run: `python -m pytest tests/ablation/test_variants.py tests/ablation/test_tiny_backend.py -q`

Expected: PASS.

```bash
git add research/kmd2_ablation/variants.py research/kmd2_ablation/qwen_variants.py research/kmd2_ablation/tiny_backend.py research/kmd2_ablation/tiny_training.py tests/ablation/test_variants.py tests/ablation/test_tiny_backend.py
git commit -m "feat: define staged KMD-2 ablations"
```

---

## Chunk 3: Runner, result storage, Qwen training/evaluation, and summaries

### Task 10: Implement atomic results, deterministic sharding, and resumable runner

**Files:**
- Create: `research/kmd2_ablation/results.py`
- Create: `research/kmd2_ablation/runner.py`
- Create: `tests/ablation/test_results_runner.py`

- [ ] **Step 1: Add canonical manifest/jobs/sharding tests, run red, and implement**

Test canonical immutable `manifest.json` and `jobs.json`, semantic job IDs,
SHA-256 first-eight-byte unsigned big-endian shard assignment, JSON-key and
`PYTHONHASHSEED` invariance in subprocesses, and disjoint/exhaustive shard union.
The manifest records schema/suite versions, canonical config, source/config/
asset hashes, Git/diff identity, environment/dependency versions, command and
expanded-job digest. Confirm missing APIs, implement canonical writers and
assignment, rerun.

- [ ] **Step 2: Add atomic record/quarantine tests, run red, and implement**

```python
def atomic_write_json(path, record):
    """fsync a unique sibling temp then same-filesystem replace."""
def assign_shard(job_id, num_jobs):
    """Return uint64_be(sha256(job_id)[:8]) modulo num_jobs."""
def validate_completed_run(record, job, provenance):
    """Validate schema, assignment, identity, provenance, and status."""
```

Test concurrent writers, interruption before replace, truncated/stale temp,
conflicting completed files, immutable manifest/jobs, and deterministic
quarantine names under `quarantine/<job_id>/<reason>-<digest>.json`. Implement
same-filesystem `os.replace`, fsync where supported, unique temps, conflict
locking/validation, and quarantine; rerun.

- [ ] **Step 3: Add authoritative resume tests, run red, and implement**

Completion authority is only a valid execution-status `completed` record
matching schema, experiment/job/shard assignment and provenance. `failed` is
the only other persisted execution status; `missing` is derived from absence,
and scientific labels such as `inconclusive` live only in summaries. Test that
missing, failed, truncated, stale-source, stale-config, and interrupted records
rerun; valid completed records skip even when their later scientific
classification is inconclusive; event logs never skip; stale/conflicting records
quarantine; resume never mutates `jobs.json`. Implement `ResultStore` and rerun.

- [ ] **Step 4: Add typed diagnostics/failure tests, run red, and implement runner dispatch**

Every completed record requires metrics and loss curves, non-finite/skip counts,
trainable/total parameters, recurrent-state elements/bytes, wall time,
examples/tokens per second, peak VRAM, checkpoint/data identity, exact command,
canonical config, seed/experiment/job IDs, and Python/PyTorch/CUDA/GPU/dependency
provenance. Completed exact-cache records additionally require width/block size,
score definition, fp32 compute dtype, explicit storage dtype, coordinate frame, inclusive
causality/tie policy, amplitude initial/final, selected-index
digest/sample, score digest/statistics, retention/eviction, hit/conditional-read,
sink/entropy/top-1, stale occupancy/error, cache/state norms, persistent/block
bytes, and implementation paths. Forced OOM records require phase, B/T/H/dk/dv,
width/block, dtype/device, estimated bytes, peak VRAM, unchanged config and a
bounded traceback; non-finite loss/gradient and malformed input have distinct
codes. Test that a failed job is atomic and later jobs still execute. Implement
lazy tiny/Qwen dispatch and typed `completed|failed` execution records; rerun.

- [ ] **Step 5: Run concurrency/resume suite and commit**

Run: `python -m pytest tests/ablation/test_results_runner.py -q`

Expected: PASS, including byte-identical manifests, job tables, and canonical
per-job records; aggregate ledger determinism is tested after Task 13 implements
the ledger.

```bash
git add research/kmd2_ablation/results.py research/kmd2_ablation/runner.py tests/ablation/test_results_runner.py
git commit -m "feat: add resumable ablation runner"
```

### Task 11: Implement CLI and preflight

**Files:**
- Create: `research/kmd2_ablation/run_ablation.py`
- Modify: `research/kmd2_ablation/runner.py`
- Create: `tests/ablation/test_cli.py`

- [ ] **Step 1: Add parser/dispatch tests, run red, and implement lazy CLI**

Test exact `preflight`, `run`, `summarize`, and `bundle` syntax; common
shard/resume flags; Qwen external paths/devices; JSON stdout; and stable exit
codes `0=ok`, `2=config/usage`, `3=preflight`, `4=execution`, `5=summary`,
`6=bundle`. Use an injected handler registry in parser tests so summarize/bundle
dispatch is tested before those modules exist; production handlers remain lazy
imports. Confirm missing module, implement argparse/handler injection, rerun.

- [ ] **Step 2: Add environment/asset preflight tests, run red, and implement**

Test Python/PyTorch/optional dependency versions, CUDA availability/device and
dtype support, writable output, external model/checkpoint/data identity and
optional checksums/tree manifests, source hash staleness, unsupported masks,
packing/decode/streaming/Option-3 modes, and actionable error codes. Qwen dry
run must not import Transformers or load tensors. Implement and rerun.

- [ ] **Step 3: Add scientific/no-op preflight tests, run red, and implement**

Reject backend/task/variant incompatibility, frozen zero gates, identity failure,
missing deterministic active effect, top-surprise with `w<1`, fewer than two
blocks or no possible eviction, incomplete four-cell/Option-3 inputs, and
invalid FFN matching. Report canonical expanded jobs, pairing IDs, trainable/
total parameters, recurrence/cache/block bytes, external assets and exact
commands. On success atomically write immutable `manifest.json`/`jobs.json`
without model tensors. Machine-readable fields are `ok`, `schema_version`,
`codes`, `warnings`, `inventory`, `resources`, `assets`, `jobs`, `commands`,
and `manifest_path`. Implement only after named failures.

- [ ] **Step 4: Run CLI tests and commit**

Run: `python -m pytest tests/ablation/test_cli.py tests/ablation/test_config.py tests/ablation/test_results_runner.py -q`

Expected: PASS.

```bash
git add research/kmd2_ablation/run_ablation.py research/kmd2_ablation/runner.py tests/ablation/test_cli.py
git commit -m "feat: add portable ablation CLI"
```

### Task 12: Implement the Qwen paired-heal backend

**Files:**
- Create: `research/kmd2_ablation/qwen_backend.py`
- Create: `research/kmd2_ablation/qwen_training.py`
- Create: `research/kmd2_ablation/qwen_checkpoint.py`
- Create: `tests/ablation/test_qwen_backend.py`

- [ ] **Step 1: Add lazy load/strict install tests, run red, and implement adapter**

With injected fake model/manager/loaders, test external identity validation,
base model then native upgrade/checkpoint then exact-cache subclass installation,
strict inherited transfer, optional strict cache resume, frozen backbone and
declared trainable names. Confirm missing adapter; implement lazy Transformers
import only inside execution and the exact ordered calls; rerun without assets.

- [ ] **Step 2: Add paired job contract tests, run red, and implement expansion validation**

Require native continuation, capacity/read/gate/budget-matched recency, and
winning-surprise arms from the byte-identical pre-replacement checkpoint with
identical seed, exact example IDs/order, token/update budget, curriculum,
optimizer/schedule, stopping and eval cells. Derive a shared pairing ID and fail
on any mismatch. Run red, implement pairing validation, rerun.

- [ ] **Step 3: Add one-step train-group tests, run red, and implement paired heal**

Use deterministic fake batches to test native memory and cache optimizer groups,
CE/KL/layerwise formulas copied as small suite functions (not
the monolithic trainer), accumulation, checkpointing flag, fixed tokens/updates,
identical windows across arms, finite/non-finite paths, amplitude projection and
structured logs. A missing teacher is allowed only when configuration explicitly
declares the synthetic-only objective; ordinary Qwen heal preflight and runtime
must fail with `teacher_required`. Confirm missing trainer/guard failures;
implement in `qwen_training.py`; rerun.

- [ ] **Step 4: Add atomic checkpoint/resume tests, run red, and implement**

Test all upgraded-layer/cache tensors, optimizer/scheduler/RNG state, step,
job/pair ID, source hashes, data/example identity, promotion config, amplitude
range, exact name/shape/dtype checks, interrupted save and resume mismatch.
Implement `qwen_checkpoint.py` atomically and rerun.

- [ ] **Step 5: Run mocked tests and define the three-arm remote smoke**

Run: `python -m pytest tests/ablation/test_qwen_backend.py tests/ablation/test_qwen_install.py -q`

Expected locally: PASS without model assets.

Remote smoke: native, matched recency, and winning surprise arms, one optimizer
step, `seq_len=64`, identical pairing/seed/examples, finite losses, amplitude
projection, and atomic checkpoints. A native+one-cache smoke is feasibility only.

- [ ] **Step 6: Commit**

```bash
git add research/kmd2_ablation/qwen_backend.py research/kmd2_ablation/qwen_training.py research/kmd2_ablation/qwen_checkpoint.py tests/ablation/test_qwen_backend.py
git commit -m "feat: add paired Qwen heal backend"
```

### Task 13: Implement Qwen RULER evaluation and promotion summaries

**Files:**
- Create: `research/kmd2_ablation/tasks/ruler.py`
- Create: `research/kmd2_ablation/summarize.py`
- Create: `tests/ablation/test_summarize.py`

- [ ] **Step 1: Add exact RULER cell/identity tests, run red, and implement**

Pin `512/2K/4K/8K/16K/32K`, needle/query/depth strata, mandatory
`{16K,32K} x {4q,8q}` long cells, three paired heal seeds, at least 64 matched
episodes per cell, exact source spans, matched native/recency/surprise identities,
teacher-forced labels, and a deterministic free-generation subset. One-seed or
512-only outputs must be labelled feasibility. Confirm missing generator/scorer,
implement, rerun.

- [ ] **Step 2: Add deterministic ledger/summary tests, run red, and implement**

From shuffled input records, require byte-identical `ledger.jsonl`,
`results.json`, and `results.csv`; retain seed/example rows, teacher-forced versus
generation labels, cache diagnostics and paired intervals. Reject unmatched
seeds/examples, duplicate records, missing required arms/cells, or feasibility
data used as promotion evidence. Confirm missing summarizer, implement canonical
ordering/output, rerun.

Integration fixtures must also apply Task 8's ordered addition, reliance, and
factorial classifiers, enforce all four matched interaction cells and protected
metric vetoes, distinguish failed versus derived-missing versus completed runs,
and preserve completed-but-inconclusive results in the ledger.

- [ ] **Step 3: Add every promotion-gate fixture, run red, and implement ordered outcome**

Test all eight surprise gates and recency gates 1/2/4-8: native long macro and
two cells; surprise-versus-recency; 512-4K/8K/episode-exact protections;
freshness accuracy and stale effect against both controls; CE/KL/non-finite/
skipped-step rules; `w=64` plus `w=32|128`; mean/max gate opening; persistent
hit; conditional read; shuffled-value drop; and complete matched cells. Assert
one exact rejection code per failing fixture and ordered outcome `surprise`,
else `recency`, else `no_promote`. Implement by calling the tested metric gate
functions, not duplicating thresholds.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/ablation/test_summarize.py tests/ablation/test_metrics.py -q`

Expected: PASS.

```bash
git add research/kmd2_ablation/tasks/ruler.py research/kmd2_ablation/summarize.py tests/ablation/test_summarize.py
git commit -m "feat: add Qwen retrieval promotion summaries"
```

---

## Chunk 4: Deterministic bundles, documentation, and end-to-end verification

### Task 14: Implement deterministic verified archives

**Files:**
- Create: `research/kmd2_ablation/bundle.py`
- Create: `tests/ablation/test_bundle.py`

- [ ] **Step 1: Add deterministic ZIP byte tests, run red, and implement builder**

Require sorted unique POSIX entries, fixed timestamps/modes/compression,
normalized UTF-8 metadata, and two builds with identical bytes/SHA-256 despite
source mtimes/order. Confirm missing builder, implement the minimal deterministic
ZIP writer, rerun.

- [ ] **Step 2: Add manifest/exclusion tests, run red, and implement bundle plans**

Test embedded SHA-256 manifest, schema/suite/Git/dirty-diff/config/production
source hashes, license/README/config schema/oracle/parity tests, smoke command,
and exclusions for `.git`, `.worktrees`, caches, secrets, model/data/checkpoint/
run artifacts, absolute paths and unsafe traversal. Tiny may contain only standard
library/PyTorch suite imports and must contain `requirements-tiny.txt` plus every
committed tiny config; tests reject Transformers/Triton in tiny requirements.
Qwen additionally contains `requirements-qwen.txt`, every Qwen config, the exact
required `gdn3` modules and an external-assets manifest with logical arguments,
expected identity/size and optional checksum/tree hash; tests verify its
requirements cover every optional import. Run red, implement, rerun.

- [ ] **Step 3: Add standalone verification/extraction smokes, run red, and implement verifier**

Include a standard-library `verify_bundle.py`. Reopen archives, verify every
entry/hash and no unexpected members, reject tampered/duplicate/traversal
archives, extract to a fresh directory, remove checkout from `PYTHONPATH`, run
tiny preflight and CPU smoke, and run Qwen dry-run without model tensors. Confirm
missing verifier/smoke failure, implement, rerun.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/ablation/test_bundle.py -q`

Expected: PASS and two identical archive hashes.

```bash
git add research/kmd2_ablation/bundle.py tests/ablation/test_bundle.py
git commit -m "feat: build verified remote bundles"
```

### Task 15: Add configs, requirements, remote scripts, and operator documentation

**Files:**
- Create: `research/kmd2_ablation/configs/*.json`
- Create: `research/kmd2_ablation/requirements-tiny.txt`
- Create: `research/kmd2_ablation/requirements-qwen.txt`
- Create: `research/kmd2_ablation/scripts/run_remote_tiny.sh`
- Create: `research/kmd2_ablation/scripts/run_remote_qwen.sh`
- Create: `research/kmd2_ablation/README.md`
- Modify: `README.md`
- Modify: `tests/ablation/test_config.py`
- Create: `tests/ablation/test_remote_artifacts.py`

- [ ] **Step 1: Add failing committed-config validation, then write configs**

Extend `test_config.py` to enumerate every committed JSON and fail while absent.
Add CPU smoke, three-seed screen, five-seed promotion, and paired Qwen
native/recency/surprise jobs with complete schemas, all thresholds explicit,
mandatory long RULER cells, and no local asset paths. Rerun to green.

- [ ] **Step 2: Document exact remote workflow**

Document environment creation, dependencies, archive verification, external asset placement, preflight, Slurm/manual sharding, resume, summarize, expected output tree, failure records, and promotion interpretation.

- [ ] **Step 3: Add shell-syntax tests, then copy/paste launch scripts**

Tests inspect scripts for `set -euo pipefail`, relative extracted-bundle paths,
argument parsing for devices/assets/output/shard index/count, preflight before
run, resume plus summarize, and absence of local Windows or `/home/dev` paths.
Implement tiny/Qwen scripts and validate with `bash -n` when Bash is available.

- [ ] **Step 4: Update top-level README**

Point to the canonical trainer and the new research suite without claiming unrun results or streaming support.

- [ ] **Step 5: Validate docs/configs and commit**

Run: `python -m research.kmd2_ablation.run_ablation preflight --backend tiny --config research/kmd2_ablation/configs/smoke.json --out .tmp-preflight --dry-run`

Expected: exit 0 and machine-readable report.

```bash
git add research/kmd2_ablation README.md tests/ablation/test_config.py tests/ablation/test_remote_artifacts.py
git commit -m "docs: add remote ablation workflow"
```

### Task 16: Run the complete verification matrix and create handoff bundles

**Files:**
- Modify only files required by discovered failures
- Produce ignored artifacts under a temporary output directory

- [ ] **Step 1: Run static/repository checks**

```bash
git diff --check
python -m compileall -q research/kmd2_ablation tests/ablation
python -m pytest --collect-only -q
```

Expected: exit 0; no archived research scripts collected.

- [ ] **Step 2: Run all CPU tests**

Run: `python -m pytest tests/ablation -q`

Expected: all CPU tests pass; CUDA/Qwen-asset tests skip with explicit reasons.

- [ ] **Step 3: Run tiny end-to-end smoke twice**

Run the same smoke config twice into separate result roots, summarize both, and compare canonical ledgers byte-for-byte.

- [ ] **Step 4: Exercise resume, concurrent shards, interruption, and forced failure**

Verify no duplicated jobs, deterministic union, atomic/quarantined partials, typed OOM/mask failures, and unchanged configuration.

Also run a mechanism coverage report that requires every committed registry arm
to have an identity test, active-effect test, compatible primary task, metric,
screening config, and result-schema fields; missing coverage is a hard failure.

- [ ] **Step 5: Build and verify both upload archives twice**

Record archive paths, SHA-256, sizes, manifest counts, and extraction smoke results. Ensure large external assets and secrets are absent.

- [ ] **Step 6: Run remote GPU gates when the server is available**

```bash
python -m pytest tests/ablation/test_fast_scan_api.py -m cuda -q
python verify_bundle.py kmd2-qwen.zip
rm -rf /tmp/kmd2-qwen-smoke && mkdir /tmp/kmd2-qwen-smoke
python -m zipfile -e kmd2-qwen.zip /tmp/kmd2-qwen-smoke
cd /tmp/kmd2-qwen-smoke && env -u PYTHONPATH \
  bash research/kmd2_ablation/scripts/run_remote_qwen.sh --smoke \
    --model "$MODEL_PATH" --native-checkpoint "$NATIVE_CHECKPOINT" \
    --data "$DATA_PATH" --out "$OUTPUT_PATH"
```

Expected: fast/reference tolerance gates pass; the verified freshly extracted
archive runs paired one-step native/recency/surprise heals with finite losses and
valid records while the checkout is absent from `PYTHONPATH`. If the local
machine lacks assets/CUDA, preserve and report this exact extraction gate as
unexecuted rather than reporting it locally passed.

- [ ] **Step 7: Final clean-state review and commit fixes**

```bash
git status --short
git diff --check
git log --oneline --decorate -15
```

Expected: only intentional source changes committed; generated outputs ignored.

Commit any verification-only corrections with focused messages, then hand off the exact remote commands and bundle hashes.
