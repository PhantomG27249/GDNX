# Portable KMD-2 Ablation Suite Design

**Date:** 2026-07-09

## Objective

Build a portable, reproducible ablation suite that can be uploaded to a faster
remote GPU server and can test proposed KMD-2 mechanisms in two complementary
settings:

1. a small, self-contained PyTorch backend for clean causal and mechanistic
   experiments; and
2. the current Qwen3.5-0.8B native warm-start KMD-2 backend for checkpoint
   reliance tests and short, controlled heal runs.

The suite must isolate one mechanism at a time, measure the ability that the
mechanism actually claims to improve, and demonstrate incremental value over
the complete current native implementation. A mechanism does not count as a
win merely because it replaces an existing feature that was disabled for the
experiment.

The deliverable is one command-line entry point backed by small focused
modules, configurations, tests, documentation, and verified upload bundles.

## Non-Goals

The initial suite will not:

- modify the production recurrence in `gdn3/kmd2_native.py`;
- change or regenerate the surviving native-heal checkpoint;
- claim that a small synthetic result proves a language-model improvement;
- claim that a Qwen checkpoint intervention proves architectural causality;
- reproduce Mamba-3's 440M pretraining study;
- silently interpret the current shared-query `r_out=4` path as true rank-4
  MIMO;
- treat the user-proposed momentum equations as Nesterov without correcting
  their update and lookahead point;
- make the experimental recurrences compatible with the production fast scan
  before their reference-loop behavior earns promotion; or
- run every pairwise combination. Interaction tests are reserved for features
  that first pass their individual screen.

## Current Native Baseline Contract

The canonical full-model control is `KMD2NativeAttn` selected by
`GDN3_KMD2_NATIVE=1`, with `GDN3_KMD2_ROUT=4` and the reference Python scan.
The fast scan is disabled for recurrence-changing experiments unless a variant
explicitly proves compatibility.

The suite must inventory the current implementation before creating jobs.

### Features already present

| Feature | Current implementation |
|---|---|
| Native Qwen warm load | q/k/v, convolution, gates, norm, and output path |
| Local mixing | Warm-loaded depthwise causal convolution plus SiLU |
| Rotation | Cumulative data-dependent paired q/k rotation |
| Output slots | `r_out=4` scaled copies of a shared query plus learned mixing |
| Decay | Native per-head decay plus learned per-key-channel offsets |
| Write control | Per-head write-beta offset decoupled from erase |
| Recurrent state | Dense `dk x dv` state, zeroed on every forward call |

### Features not currently present

| Feature | Required experimental interpretation |
|---|---|
| Trapezoidal state-input carry | Previous and current write factors blended inside the recurrence |
| B/C-style bias | New gated q/k channel biases after normalization |
| True MIMO | Independent rank-R write inputs and output queries sharing one state |
| Momentum | A second full velocity state with a coherent lookahead update |
| Lookahead target | A causal, identity-gated value-space derivative correction |
| Native state-size knob | A changed native state shape; not warm-load compatible |

The cold `KMD2LinearAttn` and original `GDN3LinearAttn` paths are different
architectures. They are not valid substitutes for the native baseline.

## Redundancy and No-Op Gates

Every experiment must pass the following gates before an expensive job is
launched.

### 1. Feature inventory gate

A versioned manifest records, for each feature:

- whether it is already present;
- its owning production file and parameter names;
- whether it affects projections, recurrence, readout, or dynamic state;
- its parameter and recurrent-state cost; and
- which backends and run modes support it.

Preflight checks the expected production attributes and a source hash. A stale
manifest fails loudly rather than running against a changed implementation.

### 2. Exact baseline gate

- The Qwen backend imports the production `KMD2NativeAttn`; it does not copy the
  production forward method into the suite.
- The tiny backend implements the same scalar gated-delta recurrence and is
  checked against the production `_scan` on deterministic synthetic tensors
  whenever the repository dependencies are available.
- Forward outputs, input gradients, and recurrence-parameter gradients are
  compared.
- The standalone tiny bundle records the production source hash used to
  certify its recurrence.

### 3. Identity gate

New warm-startable mechanisms are parameterized so their disabled or initial
state reproduces the complete current baseline. Before training, the suite
checks output and gradient parity at a declared tolerance.

State-size changes and true rank-R MIMO cannot pass a native warm-start identity
gate. They must be labelled cold/redesigned experiments instead of being
reported as native warm-start ablations.

### 4. Active-effect gate

The suite sets deterministic non-identity test parameters for each variant and
requires its outputs to differ from the baseline on a diagnostic input. A
variant that adds parameters but is disconnected, overwritten, or otherwise a
no-op is rejected.

The changed parameter names, trainable-parameter count, dynamic-state tensors,
and expected output difference are recorded in the run manifest.

### 5. Incremental-value gate

The primary comparison keeps all current native features enabled. A new feature
must improve over that complete baseline. If it helps only after an existing
feature is disabled, it is a replacement candidate, not an incremental win.

Promoted features receive a targeted interaction test with the closest current
or proposed mechanism:

| New feature | Interaction test |
|---|---|
| Trapezoidal carry | trapezoid x existing convolution |
| B/C-style bias | bias x trapezoid, then winning pair x convolution |
| Lookahead | lookahead x convolution; lookahead x trapezoid if both pass |
| Momentum | momentum x current decay/erase behavior |
| Rotation | rotation x pair-tied versus independent channel decay |
| State size / true MIMO | state size x SISO/true-MIMO; current `r_out=4` remains a separate control |

New additions are classified as **incremental**, **replacement-only**,
**redundant**, **synergistic**, **harmful**, or **inconclusive** by the mutually
exclusive rules in "Metrics and Statistical Decisions." Existing rotation and
convolution are not additions; their on/off arms use the separate reliance
labels **relied-on**, **dispensable**, **harmful-current**, or
**inconclusive-reliance**.

## Architecture

The preferred layout is a portable suite directory with one CLI, not one large
monolithic script.

```text
research/kmd2_ablation/
|-- run_ablation.py          # preflight, run, summarize, and bundle commands
|-- config.py                # validated JSON schema and experiment IDs
|-- inventory.py             # current-feature and backend capability manifest
|-- variants.py              # mechanism definitions and compatibility rules
|-- tasks.py                 # deterministic synthetic task generators
|-- metrics.py               # task, efficiency, and statistical metrics
|-- runner.py                # job expansion, execution, resume, atomic output
|-- tiny_backend.py          # exact native-style small PyTorch model
|-- qwen_backend.py          # production import, interventions, and heal adapter
|-- summarize.py             # paired deltas, intervals, classifications
|-- configs/
|   |-- screening.json       # three-seed, short-budget first pass
|   `-- promotion.json       # five-seed and longer extrapolation pass
|-- requirements-tiny.txt
|-- requirements-qwen.txt
`-- README.md

tests/ablation/
|-- test_recurrence_parity.py
|-- test_inventory.py
|-- test_variants.py
|-- test_tasks.py
|-- test_runner_resume.py
`-- test_bundle.py
```

The implementation may consolidate very small modules, but it must preserve
the boundaries between configuration, task generation, model backends,
variants, execution, and result summarization.

## Command-Line Interface

The single entry point exposes four subcommands.

### `preflight`

Validates:

- Python and dependency availability;
- CUDA devices and requested dtypes;
- model, checkpoint, and dataset paths;
- current-feature inventory and source hashes;
- backend/variant/task compatibility;
- identity and active-effect gates;
- expanded job count and estimated state/parameter costs; and
- output-directory writability.

It supports `--dry-run` and produces a machine-readable preflight report.

### `run`

Required common arguments:

```text
--backend tiny|qwen
--config PATH
--out PATH
--job-index N
--num-jobs N
--resume
```

Qwen-only arguments include:

```text
--mode reliance|heal
--model PATH
--checkpoint PATH
--data PATH
--student-device cuda:N
--teacher-device cuda:N
```

The Qwen `reliance` mode does not require a teacher. The `heal` mode requires a
teacher unless a deliberately synthetic-only objective is selected.

### `summarize`

Aggregates completed run records, produces JSON and CSV summaries, computes
paired seed effects and confidence intervals, and applies the preregistered
classification rules.

### `bundle`

Creates one of two verified archives:

- a light tiny bundle containing the suite, configs, tests, requirements, and
  provenance manifest; or
- a Qwen bundle that additionally includes the required repository code but
  excludes model snapshots, datasets, and checkpoints by default.

The Qwen bundle manifest lists every external asset, expected path argument,
size, and optional checksum. The command reopens the completed archive and
verifies required entries, exclusions, and SHA-256 hashes.

## Configuration and Job Identity

Configuration uses JSON so it is available without optional parser packages.
The validated schema includes:

- schema and suite versions;
- backend and Qwen run mode;
- baseline name;
- mechanism and variant;
- task and task-specific parameters;
- seed list;
- training token/update budget;
- optimizer and schedule;
- model/state dimensions and finite `d_ff_match_min`/`d_ff_match_max` bounds;
- context-length curriculum and extrapolation lengths;
- primary metric, metric direction, `min_useful`, `harm_threshold`,
  `min_reliance`, and equivalence band;
- protected metrics with one `max_regression` value each;
- `min_synergy` for a declared four-cell interaction;
- device/dtype preferences; and
- required interaction or promotion stage.

All decision thresholds for one metric use that metric's normalized raw units.
Reliance configurations must satisfy
`min_reliance > equivalence >= 0` and `harm_threshold > equivalence`; preflight
rejects a configuration that could make reliance labels overlap.

A canonical serialization of semantic configuration fields produces the
experiment ID. Runtime-only fields such as output path and device number do not
change the ID. Resume skips only a run with a valid completed record matching
the same experiment ID and code provenance.

Job sharding is deterministic. `--job-index i --num-jobs n` selects a stable
subset, making the suite usable with Slurm arrays or several independent GPU
workers without a shared coordinator.

## Backends

### Tiny backend

The tiny backend depends only on PyTorch and the standard library. It uses:

- a configurable token or continuous-input embedding;
- one or more residual blocks;
- the exact native scalar gated-delta state orientation and post-update read;
- optional current native convolution, rotation, shared-query output slots,
  per-channel decay, and write offset; and
- a task-appropriate classification, token, or regression head.

The default screening baseline includes the current native mechanisms. Each
experiment changes one declared factor. Learned absolute position embeddings
are prohibited in length-extrapolation tasks because they create an unrelated
extrapolation failure and can mask recurrent state tracking.

The tiny backend is the causal evidence backend. It answers whether a
mechanism can learn and extrapolate on its claimed behavior under controlled
capacity and data.

### Qwen backend

The Qwen backend imports the current upgrade manager and
`KMD2NativeAttn`. Experimental wrappers or subclasses live inside the suite;
production KMD-2 source remains unchanged.

It supports two evidence modes.

#### Reliance mode

Loads the existing native-heal checkpoint and performs deterministic
interventions without retraining. Initial interventions include:

- full learned rotation;
- rotation disabled;
- rotation bias only;
- token-shuffled rotation increments;
- cumulative phase reset at configured boundaries; and
- current convolution enabled or bypassed.

Reliance mode determines whether the checkpoint currently uses a feature. It
does not prove that the feature improved training.

#### Heal mode

Starts from native Qwen weights or the declared checkpoint, adds one
identity-gated compatible mechanism, and trains for a fixed paired budget using
the same examples, optimizer settings, and stopping rules as its baseline.

Trapezoid, B/C-style bias, lookahead, and corrected momentum require heal mode.
Recurrence-changing variants use the Python reference loop. The suite must not
fall back silently to the existing fast scan.

Native state-size changes and true MIMO are not valid warm-start heal arms.
They are tiny-backend experiments in the initial suite. A future Qwen cold or
adapter experiment requires a separate promotion design and must not be mixed
into native-heal summaries.

The Qwen backend is the transfer evidence backend. It answers whether a
mechanism remains useful in the hybrid language model without unacceptable
quality or efficiency regression.

## Experimental Mechanisms

### Current data-dependent rotation

The suite tests the existing production rotation; it does not add a second RoPE
implementation to Qwen.

Tiny controls:

- current cumulative data-dependent paired rotation;
- rotation disabled;
- learned constant-rate cumulative rotation;
- parameter-matched non-cumulative rotation;
- standard fixed-frequency RoPE; and
- explicit moving-frame state rotation oracle.

The exact complex-transition equivalence is tested separately under pair-tied
and independently learned channel decay. Success on a task must be described
as phase/state-tracking evidence unless the moving-frame equivalence conditions
also pass.

### Exponential-trapezoidal state-input carry

For one head, let `D_t` be the current per-key-channel decay and define:

```text
S_bar = D_t * S_prev
m_t = k_t^T S_bar
E_t = k_t (beta_e,t * m_t)^T
u_t = beta_w,t * v_t
U_t = k_t u_t^T
U_prev = k_prev u_prev^T
r_t = rho_head * sigmoid(rho_proj(x_t))
S_t = S_bar - E_t + (1 - r_t) U_t + r_t (D_t * U_prev)
```

`rho_head` is a projected per-head parameter constrained to `[0, 1]`, initialized
at exactly zero, and projected back into the interval after each optimizer
step. This gives it a usable boundary gradient while preserving an exact native
identity. The token-dependent projection may begin learning once `rho_head`
moves away from zero.

The carry stores differentiable factors `k_prev` and
`u_prev = beta_w,prev * v_prev`, not a full `dk x dv` matrix. At the first valid
token and every explicit sequence/packed-segment boundary, the carry is cleared
and `r_t` is forced to zero. The previous write is transported by the current
decay `D_t`; the current erase remains based on `beta_e,t` and the decayed
pre-update state. Carry tensors are not detached during training.

Identity proof: with `rho_head=0`, `r_t=0` and the equation reduces to
`S_t = S_bar - E_t + U_t`, exactly the current native update. A classical
equal-endpoint mixture is not native-warm compatible and is not used for the
Qwen heal arm.

This is an adaptation of trapezoidal state-input mixing to a delta-style
associative memory, not a claim of a formally derived second-order KMD-2
integrator. KMD-2 has no exposed continuous-time `Delta_t`, so the suite does
not attach an unsupported second-order accuracy claim to this recurrence.

### B/C-style q/k bias

Adds separate learned head/channel q and k biases after normalization. Qwen
uses an identity gate initialized to zero so the warm start is preserved.

```text
q_biased = q_normalized + a_q,head * b_q,head
k_biased = k_normalized + a_k,head * b_k,head
```

The per-head amplitudes `a_q` and `a_k` are initialized at exactly zero; bias
vectors may be initialized independently because their contribution is then
exactly zero. The active-effect gate sets nonzero amplitudes and bias vectors.
The tiny affine-regression control replaces the additive vectors with an
equal-parameter diagonal rescaling, which cannot supply a constant coordinate.

Bias is screened alone first. It is combined with trapezoidal carry only after
one of the individual arms passes, because their reported language-model role
may overlap with the existing convolution.

### Existing convolution ablation

The suite toggles the current native convolution. It never stacks a duplicate
short convolution.

The primary incremental baseline keeps convolution enabled. Convolution-off
runs answer whether another feature can replace it and are labelled
replacement tests.

### Corrected momentum

The suite does not implement the originally proposed
`W_t = decay(W_{t-1}) + V_t + gamma V_t`, which double-counts the velocity and
does not evaluate a Nesterov lookahead.

For state `S` and velocity `M`, the coherent experimental form is:

```text
S_bar = decay(S_prev)
M_bar = decay(M_prev)
S_look = S_bar + gamma * M_bar
error = beta_w * v - beta_e * (k^T S_look)
G = k error^T
M = gamma * M_bar + G
S = S_bar + M
```

At `gamma=0`, this recovers the current delta update. Momentum doubles the
large dynamic recurrent state and is reported as such. It is reference-loop
only until it demonstrates enough value to justify a new scan derivation.

### Lookahead value target

Uses a causal value-space finite difference rather than calling a projection
of raw hidden-state differences an exact derivative:

```text
v_target = v_t + rho_t * P(v_t - v_prev)
```

`rho` is identity-gated at zero. The mechanism stores previous value factors
and leaves the base scan interface unchanged. It is described as causal
extrapolation, not an implicit solve.

### State-size and true-MIMO sweep

The tiny backend runs two distinct sweeps. The **state-size sweep** fixes
`d_model`, layer count, head count, and MIMO rank at `R=1`, then varies declared
`(dk, dv)` pairs; recurrent capacity is reported as `heads * dk * dv` elements.
The **MIMO-rank sweep** fixes `d_model`, layers, heads, `dk`, `dv`, and therefore
the recurrent-state size, then varies `R`. A declared factorial may vary both,
but it remains labelled as a factorial rather than either one-dimensional
sweep.

Each sweep has a raw fixed-FFN comparison, which keeps the feed-forward hidden
dimension fixed and reports the resulting parameter-count change, and a
separate parameter-matched comparison defined below.

For each head, true MIMO uses row-normalized independent keys
`K_t [R, dk]`, values `V_t [R, dv]`, queries `Q_t [R, dk]`, per-slot erase/write
gates `beta_e [R]` and `beta_w [R]`, and one shared state `S [dk, dv]`.
All slots update simultaneously:

```text
S_bar = D_t * S_prev
K_e = diag(sqrt(beta_e / R)) K_t
S_erase = S_bar - K_e^T (K_e S_bar)
S_t = S_erase + (1 / sqrt(R)) K_t^T diag(beta_w) V_t
Y_t = Q_t S_t
o_t = l2_normalize(o_logits_t), initialized to [1/sqrt(R)] * R
y_t = o_t^T Y_t
```

The erase is the order-invariant average
`sum_r (beta_e,r / R) k_r (k_r^T S_bar)`. Tests require forward and gradient
invariance under a common slot permutation. The `1/R` erase scaling,
`1/sqrt(R)` write scaling, and unit-L2 output mixing prevent an automatic
R-fold magnitude advantage. At `R=1`, `K_e = sqrt(beta_e) k`, the erase is
exactly `S_bar - beta_e k (k^T S_bar)`, the write is exactly
`k (beta_w v)^T`, and the one-slot output mixer is one. The entire recurrence
therefore reduces exactly to the declared native SISO reference with no ridge
or limiting argument.

The current `r_out=4` shared-query slots are another separate control and are
never relabelled as true MIMO.

Parameter matching uses exact instantiated trainable-parameter counts, not a
projection-only estimate:

- **State-size matching:** the target is the configured canonical `R=1` model
  at `(dk_ref, dv_ref)`. Each alternate `(dk, dv)` arm keeps `d_model`, layers,
  heads, and `R=1` fixed and adjusts only the shared per-layer feed-forward
  hidden dimension, upward or downward, to compensate for changed q/k/v,
  gate, rotation, normalization, and output-projection parameters.
- **MIMO-rank matching:** the target is the `R=1` model at the same `(dk, dv)`.
  Each `R>1` arm keeps recurrent state and all other dimensions fixed and
  adjusts only that feed-forward hidden dimension to compensate for the added
  independent slot projections and gates.
- **Declared factorial matching:** the target is the canonical
  `(dk_ref, dv_ref, R=1)` model, and the same feed-forward-only adjustment
  compensates for both changes together.

For every matched arm, preflight instantiates each feed-forward candidate in the
configured finite interval `[d_ff_match_min, d_ff_match_max]` (`d_ff >= 8`, both
bounds and candidates divisible by 8), chooses the one minimizing absolute
count difference, and requires the resulting total trainable-parameter count
to be within the larger of 0.5% of the target or 1,024 parameters. It rejects
the arm when no legal candidate satisfies the tolerance. The bounds, target
count, selected `d_ff`, exact arm count, and residual mismatch are reported.
Thus state-size matching may increase FFN capacity while MIMO matching will
usually reduce it; neither silently changes recurrent-state bytes.

Primary reporting includes quality, state bytes, parameter count, training
throughput, and single-step/reference-loop latency. No automatic 2x efficiency
claim is permitted.

## Task Matrix

Each mechanism has one primary task family chosen for discriminative power.
Shared language-model and retrieval tasks are secondary transfer measures.

| Mechanism | Primary ability | Primary task | Key failure measure |
|---|---|---|---|
| Rotation | cyclic state tracking | parity, modular counter, toggle FSM | OOD length collapse; angle intervention sensitivity |
| Trapezoid | variable-step temporal integration | irregular-time driven decay/integration | error versus time gap and forcing curvature |
| Convolution | local order and binding | adjacent key/value, short motif, delayed copy | accuracy versus local separation |
| Momentum | inertia in a persistent mapping | gradual drift followed by abrupt reversal | adaptation lag and reversal overshoot |
| Lookahead | causal trajectory extrapolation | linear/sinusoidal motion plus change points | smooth-segment gain versus change-point harm |
| State size / MIMO | capacity per state byte | MQAR load and length sweep | quality-state-latency Pareto frontier |
| B/C bias | affine constant-basis memory | symmetric affine associative regression | intercept error; zero-intercept negative control |

### State-tracking tasks

Parity and modular counting include `HOLD` and explicit `QUERY` operations.
Toggle FSM adds `SET0`, `SET1`, `TOGGLE`, `NOOP`, and `QUERY`. These resets and
overwrites distinguish an updateable state from a model that only accumulates
phase to the final token.

Training uses a length curriculum and evaluates held-out 2x and 4x operation
counts. Both raw accuracy and chance-adjusted accuracy are reported.

### Irregular-time integration

Samples the stable scalar/vector system `dh/dt = -a h + u(t)` with `a > 0`,
piecewise-linear forcing between observed endpoint values, and irregular time
gaps `Delta`. Inputs contain the forcing endpoint and elapsed delta. For one
component, with `e = exp(-a Delta)` and
`m = (u_t - u_prev) / Delta`, the exact target is:

```text
h_t = e h_prev
    + u_prev (1 - e) / a
    + m [Delta / a - (1 - e) / a^2]
```

The generator evaluates this in float64 with `expm1`-stable branches for small
`a Delta` and verifies samples against a fixed high-resolution RK4 oracle. The
oracle is validation-only and is shared by every model variant.

Evaluation stratifies error by delta, sequence length, and forcing curvature.
This directly tests temporal integration instead of using general perplexity as
a proxy.

### Drift and reversal

Generates key/value mappings that drift smoothly for a declared interval and
then change abruptly. Queries occur before the next observation so the model
cannot copy the target token.

Metrics include steady-state error, adaptation lag, peak overshoot, recovery
time, and the smooth-drift/reversal tradeoff. Momentum is useful only if its
smooth-regime gain is not purchased with excessive reversal failure.

### Trajectory extrapolation

Generates piecewise linear and sinusoidal value trajectories with withheld
next-step targets. Change-point cases are balanced with smooth cases.

Metrics separately report smooth forecast error, phase lag, change-point
overshoot, and recovery time. This prevents a lookahead mechanism from being
declared a temporal win based only on easy smooth segments.

### Local binding and MQAR

Local binding controls convolution with adjacent and separated key/value
tokens. MQAR varies number of bindings, queries, overwrite frequency, and
distance. Results include token accuracy, episode exact match, and distance/load
bins.

### Affine associative regression

This direct-factor tiny task isolates the constant basis supplied by q/k
biases. Each episode samples a linear map `A` and intercept `b`, observes
write pairs `y_i = A x_i + b`, and predicts a held-out query target. Write keys
are generated in exact `x, -x` pairs so their episode mean is zero and the
intercept cannot be inferred from a spurious key mean.

The task feeds q/k/v factors directly to the memory-cell harness. q/k
projections, readout, and all competing paths are bias-free; there is no MLP,
learned position embedding, constant input coordinate, or token-type feature in
the q/k path. The query/write role is supplied as an external mask rather than
an embedded marker. A parameter-matched diagonal-rescaling control has the same
number of trainable scalars but cannot add a constant basis. An oracle arm
explicitly appends a constant coordinate.

The primary metric is held-out query MSE, with intercept error and slope error
reported separately. Evaluation includes more writes and a wider query range
than training. A balanced `b=0` negative-control split must show no material
bias advantage; nonzero intercepts are sampled symmetrically and independently
of `A`, writes, and queries. Generator tests verify symmetry, independence,
absence of constant q/k inputs, and withheld-query targets.

## Screening and Promotion

### Stage 0: local correctness

- configuration and inventory validation;
- deterministic task tests and answer-leak checks;
- recurrence forward/backward parity;
- identity and active-effect gates;
- CPU tiny smoke run; and
- optional CUDA smoke run.

### Stage 1: tiny screening

- three paired seeds;
- fixed short update/token budget;
- current complete baseline versus one mechanism;
- in-distribution plus 2x/4x extrapolation; and
- primary task and efficiency metrics.

### Stage 2: tiny promotion and redundancy

Only arms exceeding the preregistered useful threshold advance to:

- five paired seeds;
- larger contexts and task loads;
- closest-mechanism interaction tests; and
- secondary retrieval or small-LM transfer tests.

### Stage 3: Qwen reliance

Runs deterministic interventions on the same evaluation examples. Rotation is
evaluated at 512, 2K, 4K, 8K, and 16K where feasible. Results use paired
episode-level intervals.

Teacher-forced evaluation is labelled as such. A material result is confirmed
on a smaller free-generation set before being described as a generation gain.

### Stage 4: Qwen heal

Only compatible tiny winners receive paired baseline/variant heal runs. Runs
use identical data order, update/token budget, learning-rate groups, and
evaluation examples. Full current native features remain enabled unless the
declared experiment is a replacement interaction.

## Metrics and Statistical Decisions

Every run records:

- task primary and secondary metrics;
- loss curves and non-finite/skip counts;
- trainable and total parameter counts;
- recurrent-state elements and bytes per layer/sample;
- wall time, examples/tokens per second, and peak allocated VRAM;
- checkpoint and data identity;
- exact command, canonical configuration, seed, and experiment ID;
- Git revision and relevant source hashes; and
- Python, PyTorch, CUDA, GPU, and dependency versions.

Screening configurations preregister:

- a primary metric;
- minimum useful effect;
- maximum acceptable regression on protected metrics;
- seed count;
- training budget;
- promotion rule; and
- interaction test if promoted.

Summaries show each seed, mean/median, paired deltas, and a paired bootstrap
confidence interval. A best seed is never reported as the aggregate result.

### Normalized effect convention

Every metric declares `direction` as `+1` when larger is better or `-1` when
smaller is better. For a variant `V` and its paired baseline `B`, the effect in
the metric's configured raw units is:

```text
d = direction * (metric(V) - metric(B))
```

The suite computes a paired 95% bootstrap interval `[L, U]` using matched seeds
and matched evaluation examples within each seed. Configurations define
`min_useful > 0`, `harm_threshold > 0`, and, for every protected metric,
`max_regression >= 0`. A protected metric is certified safe only when its lower
interval bound is at least `-max_regression`.

### Promotion rule for new additions

A new addition advances from screening only when:

1. the primary effect has `L >= min_useful`; and
2. every protected metric is certified safe.

Point estimates alone never promote an arm. A replacement-only result does not
enter a native additive heal unless a separate deployment objective explicitly
values removal of the replaced feature.

### Mutually exclusive classification for new additions

Rules are evaluated in this order:

1. **failed/invalid:** the run or a required validity gate failed;
2. **harmful:** `U <= -harm_threshold`, or a protected metric has
   `U < -max_regression` (confident unacceptable regression);
3. **synergistic:** interaction arm only; it is protected-safe and the lower
   interval bound for `I = d_AB - d_A - d_B` is at least the configured
   `min_synergy`;
4. **incremental:** `L >= min_useful` against the complete current baseline and
   all protected metrics are safe;
5. **replacement-only:** not incremental, but `L >= min_useful` in the declared
   existing-feature-off contrast and protected metrics are safe;
6. **redundant:** `U < min_useful`, `L > -harm_threshold`, and no replacement or
   synergy rule applies; or
7. **inconclusive:** every remaining valid result, including an interval that
   still contains both a useful gain and meaningful harm.

All terms in the synergy contrast use the same complete baseline, matched
examples, seed set, metric direction, and budget. Interactions that do not have
all four factorial cells are invalid rather than approximated.

### Reliance semantics for existing features

Rotation and convolution on/off tests measure the effect
`d_reliance = direction * (full_current - ablated)`. They do not use the new
addition labels. Preflight enforces
`min_reliance > equivalence >= 0` and `harm_threshold > equivalence`. The
following ordered rules are therefore mutually exclusive:

1. **failed/invalid:** the run or a required validity gate failed;
2. **harmful-current:** `U <= -harm_threshold`;
3. **relied-on:** `L >= min_reliance`;
4. **dispensable:** the entire interval lies inside the preregistered
   equivalence band `[-equivalence, +equivalence]`; or
5. **inconclusive-reliance:** every remaining valid reliance result.

Reliance results can motivate a later removal design, but cannot be reported as
evidence that an absent feature is incremental.

## Result Storage and Resume

Each job writes to a temporary file and atomically renames it only after a
complete record is serialized. The output tree contains:

```text
results/
|-- manifest.json
|-- jobs.json
|-- runs/<experiment-id>/<seed>.json
|-- checkpoints/<experiment-id>/<seed>/
|-- events/worker-<job-index>-of-<num-jobs>.jsonl
`-- summary/
    |-- ledger.jsonl
    |-- results.json
    `-- results.csv
```

`jobs.json` is an immutable canonical job manifest written once by preflight
before workers start. Workers never append to a shared file. The authoritative
result is the atomic per-run JSON record: a worker writes a uniquely named
temporary file in the destination directory, flushes and closes it, then uses
an atomic same-filesystem rename.

Each shard may append diagnostic events only to its own single-writer event
file. `summarize` reads the immutable manifest and authoritative per-run files,
then deterministically creates `summary/ledger.jsonl`; event logs are not used
to decide completion. Resume validates the result schema, experiment ID, job
assignment, and provenance. A temporary, truncated, stale, or conflicting
record is quarantined and rerun.

Concurrent-writer tests launch at least two shards against one result root,
interrupt one write, and verify that completed run records remain parseable,
the interrupted job is rerunnable, and repeated summarization produces the
same ledger bytes.

## Error Handling

The suite must fail explicitly for:

- missing model, checkpoint, data, or dependency;
- stale current-feature inventory;
- unsupported backend/variant combinations;
- requested Qwen native state-size changes;
- identity or active-effect gate failure;
- unavailable fast-scan support for a recurrence variant;
- non-finite loss or gradients;
- OOM; and
- malformed or conflicting resume records.

OOM is recorded with the requested batch and sequence size. The runner does not
silently reduce the batch, sequence length, dtype, state size, or task load,
because that would invalidate paired comparisons.

One failed job does not corrupt other shards. Summaries distinguish failed,
missing, inconclusive, and completed runs.

## Verification

Implementation is complete only when all of the following are demonstrated
fresh:

1. `preflight --backend tiny` passes on CPU with only tiny requirements.
2. The tiny recurrence matches the production native reference scan in forward
   and backward tests at declared tolerances.
3. The production Qwen backend imports, rather than duplicates,
   `KMD2NativeAttn`.
4. The inventory correctly recognizes existing convolution, rotation,
   shared-query `r_out=4`, channel decay, and write offset.
5. Identity-gated variants match the full baseline before training.
6. Active-effect tests prove every enabled variant changes its intended path.
7. Invalid and redundant feature combinations fail with actionable messages.
8. The trapezoid recurrence resets its factor carry at every declared boundary,
   exactly matches native update at `rho_head=0`, and produces a nonzero active
   effect when the carry gate is enabled.
9. True-MIMO tests prove exact native recurrence equality at `R=1`, simultaneous
   slot-permutation invariance, expected-scale normalization, and the distinct
   MIMO-rank and state-size parameter-match protocols and tolerance.
10. Every task generator is deterministic by seed and passes answer-leak,
    balance, target, and length-split checks; affine regression additionally
    proves symmetric keys, withheld queries, and absence of competing constant
    q/k paths.
11. A short CPU tiny screening matrix completes, resumes without duplicating
   completed jobs, and produces valid JSON and CSV summaries.
12. Statistical classification exercises every mutually exclusive addition and
    reliance label using configured 95% paired intervals, protected-metric
    vetoes, a complete four-cell synergy contrast, and rejected overlapping
    reliance thresholds.
13. Two concurrent shards plus an interrupted writer leave valid atomic run
    records and produce byte-identical ledgers across repeated summarization.
14. Qwen dry-run/preflight accepts explicit model/checkpoint/data paths and
    rejects unsupported native state-size arms without loading large assets.
15. Optional CUDA tests record device, throughput, VRAM, and state bytes without
    silently changing configuration.
16. Tiny and Qwen upload bundles are created, reopened, content-checked, and
    hash-verified; large external assets are absent unless explicitly included.
17. Existing production model, trainer, checkpoint, data, and result files are
    unchanged.
18. `git diff --check` passes for all suite, test, and documentation changes.
