# KMD-2 and GDN3 Future Exploration Backlog

**Date:** 2026-07-10
**Status:** Parked research backlog; not an implementation commitment

## Purpose

Record credible directions for improving KMD-2 and the older GDN3
Kronecker-residual path after completion of the portable ablation suite. This
document preserves the rationale, expected effect, cost, and first
discriminating experiment for each direction so that future work can start
from an explicit research question instead of rediscovering the design space.

This is a selection document, not one monolithic implementation spec. A future
work session must select a bounded tranche, write a dedicated design for it,
and obtain approval before implementation. Every item below is parked unless
the user explicitly promotes it.

Related repository documents:

- `docs/superpowers/specs/2026-07-09-portable-kmd2-ablation-suite-design.md`
- `docs/superpowers/plans/2026-07-09-kmd2-ablation-suite-implementation.md`
- `research/kmd2_ablation/README.md`
- `docs/HANDOFF_chunked_scan.md`
- `docs/COMPACTION_MQAR_RESULTS.md`

## Non-Goals

This backlog does not:

- authorize implementation of any candidate;
- promise that every candidate will be implemented or evaluated;
- combine the candidates into one implementation plan or experiment matrix;
- modify the completed portable ablation suite or its scientific claims;
- claim that cited paper results transfer to KMD-2 or GDN3;
- authorize remote runs, model downloads, training expenditure, or deployment;
- define final resource budgets for an unselected candidate; or
- replace the dedicated design, review, and approval required after selection.

A future design may select one mechanism family or one deliberately small
factorial. Selection of one candidate does not implicitly select its related,
dependent, or production-scale candidates.

## What Already Exists

The completed ablation suite already covers native KMD-2, trapezoidal write
carry, corrected/Nesterov-style momentum, causal lookahead, B/C bias,
state-size and true-MIMO controls, rotation/convolution reliance, and the
HOLA-inspired bounded exact-cache matrix. Those mechanisms remain in the
existing registry and are not duplicated here.

The native recurrence is approximately:

```text
S_t = G_t * S_(t-1)
    - k_t [beta_e,t * (S_(t-1)^T k_t)]^T
    + k_t [beta_w,t * v_t]^T
```

The current production path has positive decay, head-scalar erase and write
strengths, one erase/write address, one delta edit per token, a dense
unnormalized matrix state, cumulative positive q/k rotation increments, and
shared-query output slots. These are the primary openings for future work.

## Backlog Rules

1. No item is a claimed improvement until a preregistered paired experiment
   clears its primary and protected-metric gates.
2. A warm-startable mechanism must reproduce the complete native baseline at
   its disabled identity and must demonstrate a nonzero active effect.
3. A mechanism that changes state shape or cannot reproduce the native point
   is labelled a cold redesign, not an incremental warm-start addition.
4. Reference-loop correctness precedes fast-kernel work.
5. Only individually successful mechanisms receive interaction or factorial
   tests.
6. State bytes, parameter count, training FLOPs, decode latency, and peak
   memory are first-class outcomes, not footnotes.
7. Very recent paper results are hypotheses to replicate, not established
   facts about this codebase.

## Portfolio Overview

The backlog uses three research portfolios plus two enabling tracks so future
work can choose a coherent risk level. Section letters are candidate
namespaces, not portfolio names.

| Portfolio/track | Candidate sections | Goal | Typical compatibility | Main trade-off |
|---|---|---|---|---|
| P-A. Surgical native additions | A1-A11 and E1-E9 | Improve the existing constant-memory update or read geometry | Native warm start or diagnostic identity | New scan algebra, projections, and backward paths |
| P-B. Capacity and memory organization | B1-B9, C1-C13, and D1-D15 | Store more information or organize existing memory better | Mixed warm additions, cold redesigns, cache-only, hybrid cache/state, and GDN3-only work | More state, memory, routing, or compaction cost |
| P-C. Architecture forks | A12, B10, E10, and F1-F8 | Replace or substantially widen the one-step linear fast-weight model | Isolated experimental backend | Highest ceiling and highest systems cost |
| T-G. Training enablers | G1-G14 | Improve a selected owner mechanism without changing its inference equation | Inherits the owner candidate's boundary | Attribution and training-cost risk |
| T-H. Systems enablers | H1-H12 | Make an already successful recurrence correct and efficient | No scientific promotion by itself | Kernel complexity and portability risk |

The recommended starting point is P-A. It gives the cleanest causal
answer about whether KMD-2 is limited by update geometry before increasing raw
capacity.

## Boundary and Promotion Keys

Every candidate has one boundary code and one promotion destination in the
normalized lifecycle registry below.

| Boundary | Meaning | Explicit non-goal |
|---|---|---|
| `N-WARM` | Identity-gated native KMD-2 addition | Does not change persistent state topology |
| `N-WARM-STATE` | Native-output identity is possible but additional dormant state exists | Does not claim equal state bytes |
| `N-COLD` | KMD-2 state shape/topology redesign | Cannot be reported as native warm-start evidence |
| `CACHE` | Bounded exact-cache behavior only | Does not change the recurrent update equation |
| `HYBRID` | Couples bounded exact-cache behavior to a recurrent-state update | Cannot be reported as cache-only or as an ordinary native incremental arm |
| `GDN3` | Older Kronecker-residual path only | Does not modify or establish evidence for native KMD-2 |
| `FORK` | Separate memory architecture or hybrid | Does not enter the native serial registry as an ordinary arm |
| `TRAIN` | Training-only intervention | Does not change inference semantics |
| `SYSTEM` | Kernel/numerical implementation only | Cannot establish a scientific model-quality win |

| Promotion | Required destination |
|---|---|
| `NATIVE` | Tiny mechanism screen, efficiency feasibility, then matched Qwen warm-start/heal |
| `CAPACITY` | Budget-matched Tiny redesign screen, then a dedicated production-install design |
| `CACHE-QWEN` | Tiny cache screen, then the existing Qwen exact-cache workflow |
| `HYBRID-DESIGN` | Tiny hybrid reference/evidence/resource screen, then a dedicated Qwen integration design |
| `GDN3-MODEL` | GDN3 reference/compaction screen, chunk parity, then GDN3 model evaluation |
| `FORK-DESIGN` | Isolated backend evidence, followed by a new architecture design if positive |
| `OWNER` | Follow the selected owner mechanism's promotion route |
| `SYSTEM-TERMINAL` | End after parity, numerical, portability, and speed/resource gates pass |

## Normalized Candidate Lifecycle Registry

This table is the authoritative selection record. `none` is an explicit final
value: no dedicated design, evidence artifact, or superseding candidate exists
yet. Every candidate begins `parked`. A future update changes
one row and links its design/evidence or superseding target; prose alone cannot
change lifecycle state.

The `Depends on` column uses a typed, machine-checkable grammar:

- `none` means that the candidate has no prerequisite;
- `candidate:<ID>@<state>` requires that exact candidate lifecycle state;
- `candidates:<n>-of[<IDs>]@<state>` requires at least `n` listed candidates in
  the named state;
- `evidence:<artifact-kind>@none` is explicitly unsatisfied; satisfying it
  replaces `none` with a repo-relative immutable Markdown link and repeats that
  link in the row's `Evidence` column;
- `owner:none@eligible` is explicitly unsatisfied; satisfying it replaces
  `none` with a concrete owner candidate ID whose current state is exactly one
  of `approved`, `planned`, `active`, or `resolved-positive`, and whose approved
  design is linked;
  and
- ` + ` joins clauses that must all be satisfied.

Free-form prerequisite prose is invalid in this registry. Candidate-state
clauses require the named state exactly. Owner eligibility explicitly excludes
`parked`, `designing`, `resolved-negative`, and `superseded`; there is no
implicit meaning of "approved or later."

| ID | State | Portfolio | Boundary | Depends on | Promotion | Design | Evidence | Superseded by |
|---|---|---|---|---|---|---|---|---|
| A1 | parked | P-A | N-WARM-STATE | none | NATIVE | none | none | none |
| A2 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| A3 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| A4 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| A5 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| A6 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| A7 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| A8 | parked | P-A | N-WARM | candidate:A3@resolved-positive | NATIVE | none | none | none |
| A9 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| A10 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| A11 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| A12 | parked | P-C | FORK | none | FORK-DESIGN | none | none | none |
| B1 | parked | P-B | N-WARM-STATE | none | CAPACITY | none | none | none |
| B2 | parked | P-B | N-WARM-STATE | candidate:B1@resolved-positive | CAPACITY | none | none | none |
| B3 | parked | P-B | N-WARM | none | NATIVE | none | none | none |
| B4 | parked | P-B | N-COLD | none | CAPACITY | none | none | none |
| B5 | parked | P-B | N-COLD | none | CAPACITY | none | none | none |
| B6 | parked | P-B | N-COLD | none | CAPACITY | none | none | none |
| B7 | parked | P-B | N-COLD | candidate:B6@resolved-positive | CAPACITY | none | none | none |
| B8 | parked | P-B | N-COLD | candidate:B6@resolved-positive | CAPACITY | none | none | none |
| B9 | parked | P-B | N-WARM | none | NATIVE | none | none | none |
| B10 | parked | P-C | FORK | evidence:mixer-inventory@none | FORK-DESIGN | none | none | none |
| C1 | parked | P-B | CACHE | evidence:registered-exact-cache-baseline@none | CACHE-QWEN | none | none | none |
| C2 | parked | P-B | CACHE | evidence:registered-exact-cache-baseline@none | CACHE-QWEN | none | none | none |
| C3 | parked | P-B | CACHE | evidence:registered-exact-cache-baseline@none | CACHE-QWEN | none | none | none |
| C4 | parked | P-B | CACHE | evidence:registered-exact-cache-baseline@none | CACHE-QWEN | none | none | none |
| C5 | parked | P-B | CACHE | candidate:C4@resolved-positive | CACHE-QWEN | none | none | none |
| C6 | parked | P-B | CACHE | evidence:registered-exact-cache-baseline@none | CACHE-QWEN | none | none | none |
| C7 | parked | P-B | CACHE | evidence:deterministic-selector-winner@none | CACHE-QWEN | none | none | none |
| C8 | parked | P-B | HYBRID | candidate:C7@resolved-positive | HYBRID-DESIGN | none | none | none |
| C9 | parked | P-B | CACHE | evidence:single-layer-cache-winner@none | CACHE-QWEN | none | none | none |
| C10 | parked | P-B | CACHE | evidence:layerwise-cache-utility@none | CACHE-QWEN | none | none | none |
| C11 | parked | P-B | CACHE | evidence:cache-arm-winner@none | CACHE-QWEN | none | none | none |
| C12 | parked | P-B | CACHE | evidence:cache-lookup-bottleneck@none | CACHE-QWEN | none | none | none |
| C13 | parked | P-B | CACHE | evidence:rotation-cache-factorial@none | CACHE-QWEN | none | none | none |
| D1 | parked | P-B | GDN3 | none | GDN3-MODEL | none | none | none |
| D2 | parked | P-B | GDN3 | none | GDN3-MODEL | none | none | none |
| D3 | parked | P-B | GDN3 | evidence:gdn3-compaction-baseline@none | GDN3-MODEL | none | none | none |
| D4 | parked | P-B | GDN3 | evidence:gdn3-compaction-baseline@none | GDN3-MODEL | none | none | none |
| D5 | parked | P-B | GDN3 | candidate:D3@resolved-positive | GDN3-MODEL | none | none | none |
| D6 | parked | P-B | GDN3 | evidence:gdn3-compaction-baseline@none | GDN3-MODEL | none | none | none |
| D7 | parked | P-B | GDN3 | evidence:gdn3-effective-rank-diagnostics@none | GDN3-MODEL | none | none | none |
| D8 | parked | P-B | GDN3 | evidence:gdn3-exact-svd-baseline@none | GDN3-MODEL | none | none | none |
| D9 | parked | P-B | GDN3 | evidence:gdn3-exact-svd-baseline@none | GDN3-MODEL | none | none | none |
| D10 | parked | P-B | GDN3 | candidate:D4@resolved-positive | GDN3-MODEL | none | none | none |
| D11 | parked | P-B | GDN3 | evidence:gdn3-compaction-baseline@none | GDN3-MODEL | none | none | none |
| D12 | parked | P-B | GDN3 | evidence:gdn3-mimo-lane-diagnostics@none | GDN3-MODEL | none | none | none |
| D13 | parked | P-B | GDN3 | evidence:gdn3-mimo-lane-diagnostics@none | GDN3-MODEL | none | none | none |
| D14 | parked | P-B | GDN3 | evidence:gdn3-coproduct-usage@none | GDN3-MODEL | none | none | none |
| D15 | parked | P-B | GDN3 | evidence:gdn3-nonfinite-reproduction@none | GDN3-MODEL | none | none | none |
| E1 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| E2 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| E3 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| E4 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| E5 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| E6 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| E7 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| E8 | parked | P-A | N-WARM | none | NATIVE | none | none | none |
| E9 | parked | P-A | N-WARM | candidate:E8@resolved-positive | NATIVE | none | none | none |
| E10 | parked | P-C | FORK | none | FORK-DESIGN | none | none | none |
| F1 | parked | P-C | FORK | candidate:A12@resolved-positive | FORK-DESIGN | none | none | none |
| F2 | parked | P-C | FORK | none | FORK-DESIGN | none | none | none |
| F3 | parked | P-C | FORK | candidate:F2@resolved-positive | FORK-DESIGN | none | none | none |
| F4 | parked | P-C | FORK | none | FORK-DESIGN | none | none | none |
| F5 | parked | P-C | FORK | candidate:B5@resolved-positive | FORK-DESIGN | none | none | none |
| F6 | parked | P-C | FORK | candidate:B4@resolved-positive | FORK-DESIGN | none | none | none |
| F7 | parked | P-C | FORK | candidates:2-of[A1,A2,A3,A4,A5,A6,A9,A10,A11,B6,C1]@resolved-positive | FORK-DESIGN | none | none | none |
| F8 | parked | P-C | FORK | candidates:1-of[A2,A9]@resolved-positive | FORK-DESIGN | none | none | none |
| G1 | parked | T-G | TRAIN | owner:none@eligible | OWNER | none | none | none |
| G2 | parked | T-G | TRAIN | owner:none@eligible | OWNER | none | none | none |
| G3 | parked | T-G | TRAIN | owner:none@eligible + evidence:teacher-traces@none | OWNER | none | none | none |
| G4 | parked | T-G | TRAIN | owner:none@eligible | OWNER | none | none | none |
| G5 | parked | T-G | TRAIN | owner:none@eligible | OWNER | none | none | none |
| G6 | parked | T-G | TRAIN | owner:none@eligible | OWNER | none | none | none |
| G7 | parked | T-G | TRAIN | owner:none@eligible | OWNER | none | none | none |
| G8 | parked | T-G | TRAIN | owner:none@eligible | OWNER | none | none | none |
| G9 | parked | T-G | TRAIN | owner:none@eligible | OWNER | none | none | none |
| G10 | parked | T-G | TRAIN | owner:none@eligible + evidence:state-diagnostics@none | OWNER | none | none | none |
| G11 | parked | T-G | TRAIN | owner:none@eligible | OWNER | none | none | none |
| G12 | parked | T-G | TRAIN | owner:none@eligible | OWNER | none | none | none |
| G13 | parked | T-G | TRAIN | owner:none@eligible + evidence:layer-probes@none | OWNER | none | none | none |
| G14 | parked | T-G | TRAIN | owner:none@eligible | OWNER | none | none | none |
| H1 | parked | T-H | SYSTEM | owner:none@eligible | SYSTEM-TERMINAL | none | none | none |
| H2 | parked | T-H | SYSTEM | candidate:A4@resolved-positive | SYSTEM-TERMINAL | none | none | none |
| H3 | parked | T-H | SYSTEM | candidate:A2@resolved-positive | SYSTEM-TERMINAL | none | none | none |
| H4 | parked | T-H | SYSTEM | candidate:A5@resolved-positive | SYSTEM-TERMINAL | none | none | none |
| H5 | parked | T-H | SYSTEM | owner:none@eligible | SYSTEM-TERMINAL | none | none | none |
| H6 | parked | T-H | SYSTEM | owner:none@eligible | SYSTEM-TERMINAL | none | none | none |
| H7 | parked | T-H | SYSTEM | owner:none@eligible | SYSTEM-TERMINAL | none | none | none |
| H8 | parked | T-H | SYSTEM | evidence:h7-quantizer-parity@none | SYSTEM-TERMINAL | none | none | none |
| H9 | parked | T-H | SYSTEM | owner:none@eligible | SYSTEM-TERMINAL | none | none | none |
| H10 | parked | T-H | SYSTEM | owner:none@eligible | SYSTEM-TERMINAL | none | none | none |
| H11 | parked | T-H | SYSTEM | owner:none@eligible | SYSTEM-TERMINAL | none | none | none |
| H12 | parked | T-H | SYSTEM | owner:none@eligible + evidence:parity-oracle@none | SYSTEM-TERMINAL | none | none | none |

## Related and Staged Candidates

These relationships prevent overlapping ideas from being selected as duplicate
independent projects.

| Candidates | Relationship and selection rule |
|---|---|
| A3 and A8 | A3 is the narrow independent-erase-address experiment. A8 is a later generalization to all address projections and requires A3 to resolve positive. |
| A12 and F1 | A12 is the smallest Tiny solver prototype. F1 is the production-scale Mesa-style architecture fork and requires positive A12 evidence. |
| B4 and F6 | B4 asks whether sparse state capacity helps in an isoFLOP Tiny prototype. F6 is the production sparse-KMD-2 fork and is not selectable before B4 succeeds. |
| B5 and F5 | B5 asks whether a dyadic state bank helps in Tiny. F5 is the production log-linear architecture fork and depends on B5. |
| B6 versus E6/E7 | B6 changes persistent topology to independent MIMO states. E6 and E7 retain the shared state and alter only read routing or query projections; they are separate comparisons, not MIMO substitutes. |
| B10 and E10 | B10 selects a layer-level mixer schedule; E10 defines the local-attention hybrid operator used by such a schedule. A design may select E10 alone, while B10 requires an operator inventory. |

## A. Surgical Recurrence Backlog

### A1. Online diagonal preconditioning

**Change:** Maintain a per-key-feature online preconditioner and scale the
write-side key before the delta update. The preconditioner adapts curvature;
it is distinct from corrected momentum, which adapts update velocity.

**What it could do:** Reduce interference caused by a single head-wide
learning rate, accelerate correction on under-trained coordinates, and damp
over-updated coordinates while preserving fixed matrix-state size.

**First experiment:** Native scalar beta versus a frozen all-ones diagonal
control versus an online diagonal preconditioner, followed by adaptive
preconditioner forgetting. Use collision-heavy MQAR, far-surprise, and drift
reversal; match parameters and report added state bytes.

**Risk:** Low-to-medium architecture risk, medium backward/kernel risk.

**Reference:** [OSDN: Improving Delta Rule with Provable Online
Preconditioning](https://arxiv.org/abs/2605.13473)

### A2. Token-dependent channel-wise erase and write gates

**Change:** Replace head-scalar `beta_e` and `beta_w` with a key-channel erase
vector and value-channel write vector. The current static `bw_off` is not a
substitute because it neither depends independently on the token nor varies by
channel.

**What it could do:** Erase only conflicting key coordinates and write only
the value coordinates that need correction, improving repeated-key overwrite
and multi-key retrieval without increasing persistent matrix-state size.

**Implemented first experiment:** The portable tiny suite now includes the
paper-exact both-channel arm `gdn2_decoupled.channelwise` against the existing
scalar-offset native control on MQAR. It uses independent token projections and
the asymmetric `k_t (w_t*v_t - S_bar^T(b_t*k_t))^T` update. The complete
scalar/scalar, erase-only, write-only, both-channel factorial, parameter
matching, and Qwen/kernel port remain follow-up work.

**Risk:** Medium. It requires new projections and a gate-aware chunk scan and
backward implementation.

**References:**

- [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)
- [FG2-GDN](https://arxiv.org/abs/2604.19021)

### A3. Independent erase address

**Change:** Generate an erase direction separate from the write key, erase
stale content at that address, then perform the ordinary corrective delta
write at the current write address.

**What it could do:** Remove obsolete information even when it is stored under
a different address from the new fact, instead of waiting for passive decay.

**First experiment:** Shared-address baseline versus independent erase key,
with the erase branch identity-gated to zero. Target stale-value replacement,
topic switches, and weak-decay regimes.

**Risk:** Medium. Adds a key projection and another rank-one state edit.

**Reference:** [Erase-then-Delta
Attention](https://arxiv.org/abs/2606.26560)

### A4. Signed state transitions

**Change:** Permit selected transition eigenvalues or decay channels to enter
`[-1, 1]` instead of remaining strictly positive. This changes state dynamics;
it is not equivalent to rotating q/k coordinates before the update.

**What it could do:** Represent sign flips, parity, counters, and finite-state
transitions that positive-only contractions cannot express efficiently.

**First experiment:** Reference recurrence only, positive versus signed gate
with identical parameter count. Test parity, toggle FSM, modular counter, and
length extrapolation while recording negative-gate mass, state norm, and
gradient finiteness.

**Risk:** Low parameter cost but medium-to-high kernel risk. The current fast
scan uses positive cumulative-product ratios and needs a separate sign-safe
derivation.

**Reference:** [Unlocking State-Tracking in Linear RNNs Through Negative
Eigenvalues](https://arxiv.org/abs/2411.12537)

### A5. Multiple delta/Householder edits per token

**Change:** Apply a product of two or four generalized Householder/delta edits
per token rather than one rank-one edit.

**What it could do:** Produce a diagonal-plus-higher-rank transition while
retaining the same persistent matrix-state shape, improving state tracking and
length extrapolation.

**First experiment:** `n_h` in `{1, 2, 4}` with equal state size, matched total
parameters, identical data, and state-tracking tasks before language-model
promotion.

**Risk:** Medium-to-high. Projection and mixer FLOPs grow with `n_h`, and an
efficient WY-style scan is required for production viability.

**Reference:** [DeltaProduct](https://arxiv.org/abs/2502.10297)

### A6. Fully dynamic channel decay

**Change:** Replace the current token-dependent head scalar plus static
per-channel offsets with token-dependent decay for every key channel.

**What it could do:** Let one token preserve some address dimensions while
rapidly forgetting others, especially under mixed local and global structure.

**First experiment:** Native decay, static channel offsets, and fully dynamic
channel decay with the same initialization and a projection-matched control.
Use freshness, mixed-timescale recall, and long-context stability.

**Risk:** Medium projection and fast-scan cost.

### A7. Residual-conditioned update strength

**Change:** Condition the recurrent learning rate on the actual prediction
residual `v_t - S^T k_t`, not only on the input hidden state.

**What it could do:** Make large corrections for surprising writes and avoid
damaging already-correct associations.

**First experiment:** Input-only beta versus detached residual-norm correction
versus a small residual-vector gate. Measure overwrite accuracy, redundancy,
and scan approximation error.

**Risk:** Medium-to-high because state-dependent gates complicate chunkwise
parallelization.

### A8. Independent read, write, and erase features

**Change:** Give reading, writing, and erasing distinct feature projections
rather than forcing one key geometry to serve multiple roles.

**What it could do:** Separate retrieval similarity from storage placement and
cleanup, reducing conflicts between good reads and good updates.

**First experiment:** Shared address versus low-rank residual projections for
write only, erase only, and both. Require an exact identity gate and matched
parameters.

**Risk:** Medium; additional freedom can undermine the warm start or make keys
less mutually aligned.

### A9. Relaxed value replacement

**Change:** Let the model interpolate among corrective replacement, additive
write, and retained old value rather than using one fixed delta form.

**What it could do:** Improve repeated facts, accumulators, and cases where a
new value should supplement rather than replace memory.

**First experiment:** Delta replacement versus gated additive/corrective
mixture on repeated-key overwrite, running sums, and structured exceptions.

**Risk:** Medium interaction complexity.

**Reference:** [RWKV-7 Goose](https://arxiv.org/abs/2503.14456)

### A10. State-aware orthogonal novelty writes

**Change:** Remove the component of a proposed write that is redundant with
the addressed state before committing it.

**What it could do:** Preserve state capacity by admitting novel information
instead of repeatedly storing the same direction.

**First experiment:** Equal state bytes and no exact cache: ordinary delta
write versus state-aware orthogonal write. Test repeated redundant inputs,
near-key collisions, and structured exceptions.

**Risk:** High. The update depends on current state and needs careful chunkwise
approximation.

**Reference:** [Lattice: Learning to Efficiently Compress the
Memory](https://arxiv.org/abs/2504.05646)

### A11. State-energy normalization

**Change:** Track state energy and apply a soft norm projection, normalized
write direction, or learned energy gate before non-finite growth occurs.

**What it could do:** Bound long-context drift, reduce BF16 error, and improve
gradient stability without allocating more memory.

**First experiment:** No normalization, diagnostic hard projection oracle,
soft learned projection, and normalized write direction. Measure state norm,
effective rank, retrieval, and protected short-context quality.

**Risk:** Low-to-medium. Over-normalization can erase magnitude-coded
information.

**Reference:** [Variational Linear
Attention](https://arxiv.org/abs/2605.11196)

### A12. Online ridge/RLS memory

**Change:** Maintain sufficient statistics or an adaptive penalty matrix and
solve the associative regression objective more accurately than one online
gradient step.

**What it could do:** Use fixed state capacity more efficiently and reduce
interference near the per-head capacity boundary.

**First experiment:** Tiny one-layer baseline versus one, four, and eight local
solver iterations plus a converged oracle. Report solver residual, state bytes,
latency, and failure codes.

**Risk:** High inference FLOPs and additional Gram/preconditioner state. This
belongs in a separate solver track if the lightweight A1 preconditioner wins.

**References:**

- [MesaNet](https://arxiv.org/abs/2506.05233)
- [Longhorn](https://arxiv.org/abs/2407.14207)

## B. Capacity and State-Organization Backlog

| ID | Direction | What it could do | First discriminating comparison | Risk |
|---|---|---|---|---|
| B1 | Fast and slow KMD-2 states | Separate rapid adaptation from durable memory | Native one-state baseline versus dormant-added two-state identity arm, plus a separately labeled equal-total-byte cold control | Medium |
| B2 | Separate braided-timescale states | Preserve distinct temporal bands instead of averaging their decay | Averaged decay versus routed state bank | Medium-high |
| B3 | Layerwise retention hierarchy | Make lower layers local and upper layers progressively longer-lived | Uniform versus monotone layerwise decay floors | Low-medium |
| B4 | Sparse delta memory | Address a much larger sparse recurrent state at similar active FLOPs | Dense state versus isoFLOP sparse state | High |
| B5 | Logarithmic multiscale state bank | Retain dyadic-age memories with `O(log T)` state/compute growth | Dyadic bank versus equal-byte independent states | High |
| B6 | Production true MIMO | Give Qwen independent memory lanes rather than shared-query output slots | Shared `r_out` versus independent states/queries | High |
| B7 | Content-dependent lane routing | Send tokens and reads to the most relevant memory lanes | B6 true-MIMO baseline versus query-conditioned routing over the same B6 lanes | Medium |
| B8 | Adaptive lane/state rank | Allocate capacity where measured interference is highest | Uniform versus budget-matched adaptive allocation | High |
| B9 | Learned initial state | Store parametric priors in the recurrent memory before context writes | Zero versus learned initial state | Medium |
| B10 | Hybrid mixer schedule | Optimize which layers use KMD-2, local attention, or full attention | Uniform replacement versus sensitivity-selected layout | Medium-high |

Primary references:

- [Sparse Delta Memory](https://arxiv.org/abs/2607.07386)
- [Log-Linear Attention](https://arxiv.org/abs/2506.04761)
- [HGRN](https://arxiv.org/abs/2311.04823) and
  [HGRN2](https://arxiv.org/abs/2404.07904)
- [Griffin](https://arxiv.org/abs/2402.19427)

## C. Exact-Cache Extensions Beyond the Current Matrix

The current suite already covers deterministic selector, read, width, block,
storage, oracle, coordinate-frame, and native-interaction arms. Future cache
work should add new capabilities rather than more names for the same policy.

| ID | Direction | What it could do | First discriminating comparison | Main risk |
|---|---|---|---|---|
| C1 | Query-dependent cache fusion | Trust cache only when its confidence exceeds recurrent-read confidence | Fixed per-head amplitude versus identity-gated query/confidence gate | Learned gate may suppress useful cache reads |
| C2 | Adaptive per-head/layer capacity | Move slots to heads and layers that demonstrate utility | Uniform width versus equal-total-slot adaptive allocation | Budget controller can become unstable or unfair |
| C3 | Hybrid deterministic/learned admission | Preserve surprise score and learn only a bounded utility correction | Registered exact-outer selector versus exact-outer plus bounded learned residual | Overfitting future-use prediction |
| C4 | Duplicate-key consolidation | Merge repeated or near-identical keys instead of wasting slots | No merge versus deterministic similarity-threshold merge on repeated keys | Incorrect merges can blend distinct facts |
| C5 | Versioned facts | Keep position/version chains and retrieve the latest applicable value | Latest-only storage versus bounded version chain at equal bytes | More metadata and stale-value policy complexity |
| C6 | Change-point invalidation | Expire stale regions after topic or regime shifts | No invalidation, causal detector, and oracle boundaries | False resets destroy durable memory |
| C7 | Hierarchical exact/compressed cache | Keep a small exact tier and a larger compressed tier | Single exact tier versus equal-byte exact-plus-compressed tiers | More read paths and accounting complexity |
| C8 | Cache-to-state consolidation | Replay selected exact entries into recurrent state before eviction | Ordinary eviction versus replay-then-evict at the same cache budget | Consolidation may reintroduce interference |
| C9 | Cross-layer cache sharing | Remove duplicate storage across layer groups | Independent per-layer caches versus equal-total-byte shared group cache | Layers may need incompatible feature spaces |
| C10 | Layer-specialized selectors | Assign recency, surprise, or no-cache roles by layer | One selector across layers versus preregistered layer roles | Search space and attribution complexity |
| C11 | Quantized cache | Reduce bandwidth with FP8/INT8 storage and FP32 scoring | BF16 versus quantized storage at identical width and FP32 read compute | Retrieval drift near close keys |
| C12 | Approximate sparse lookup | Scale cache width without scoring every slot | Exhaustive read versus approximate read at equal width and latency budget | Approximation can miss rare needles |
| C13 | Dual-frame phase-aware storage | Preserve enough phase metadata for moving-frame reads | Current rotated-frame cache versus equal-width phase-metadata cache | More state and transform complexity |

The conceptual base remains [HOLA](https://arxiv.org/abs/2607.02303): use a
compressive recurrent state for structure and a bounded exact cache for
associations that should not be forced through that state.

## D. GDN3 Kronecker-Residual Backlog

These items target `GDN3LinearAttn`, not native KMD-2. They must remain a
separate experimental lane because its Kronecker factors, exact residual
buffer, and compaction machinery have different state semantics.

| ID | Direction | What it could do | First experiment | Risk/cost |
|---|---|---|---|---|
| D1 | Decay-consistent factors | Apply `sqrt(gamma)` to each Kronecker factor so their product and residual both decay by `gamma` | Current `gamma^2` factor decay versus matched effective decay | Medium semantic change; low parameter cost |
| D2 | Factor gauge balancing | Equalize factor norms without changing `A kron B` | No balancing versus periodic norm balance | Low compute, medium numerical-validation risk |
| D3 | Error-triggered compaction | Compact only when residual occupancy or reconstruction error demands it | Fixed cadence versus error threshold | Medium control and determinism risk |
| D4 | Adaptive old/new blend | Set compaction blend from discarded energy instead of fixed `slow_decay` | Fixed values versus bounded learned/error-derived gate | Medium attribution risk |
| D5 | Per-head compaction schedule | Spend compression work only on pressured heads | Global versus per-head trigger | High scheduling/kernel complexity |
| D6 | Adaptive residual rank | Give more exact residual slots to high-surprise heads | Uniform versus equal-total-slot adaptive allocation | High dynamic-allocation complexity |
| D7 | Adaptive Kronecker rank | Follow measured effective spectrum | Fixed versus budget-matched adaptive rank | High shape/kernel complexity |
| D8 | Incremental QR/Oja basis | Replace periodic SVD with online low-rank tracking | SVD versus online basis update | High numerical and recurrence complexity |
| D9 | Randomized compaction | Lower SVD cost using a reproducible sketch | Exact versus randomized low-rank approximation | Medium approximation and reproducibility risk |
| D10 | Separate fast/slow factors | Avoid compressing all temporal scales into one factorization | Single versus dual-timescale factors | High state and compute cost |
| D11 | Compaction reconstruction loss | Train factors to preserve pre-compaction reads | Task loss alone versus auxiliary read reconstruction | Low inference cost, medium training attribution risk |
| D12 | Lane diversity regularization | Prevent MIMO lanes from storing the same subspace | No penalty versus correlation/orthogonality penalty | Low inference cost, low-medium tuning risk |
| D13 | Router load balancing | Avoid collapse onto one lane | Native router versus entropy/load-balanced router | Low inference cost, medium tuning risk |
| D14 | Dynamic coproduct rank | Spend coproduct channels only where useful | Fixed versus budget-matched routed rank | Medium-high routing/kernel cost |
| D15 | State-energy projection | Prevent non-finite long-context state growth | No projection versus diagnostic hard and soft projections | Low-medium compute, information-loss risk |

`docs/HANDOFF_chunked_scan.md` records that each Kronecker factor currently
decays by `gamma`, making the product decay by `gamma^2`, while the residual
decays by `gamma`. It also records long-horizon non-finite risk from the
unnormalized state. `docs/COMPACTION_MQAR_RESULTS.md` records the sensitivity
to rank and slow blending. These are evidence for the ablations, not proof of
the proposed remedies.

## E. Rotation, Position, and Local Mixing Backlog

| ID | Direction | What it could do | First discriminating comparison | Distinction from current suite | Risk/cost |
|---|---|---|---|---|---|
| E1 | Signed phase increments | Rotate forward or backward based on content | Positive softplus increment versus identity-gated signed increment | Current increments are positive through softplus | Low parameter cost; medium stability risk |
| E2 | Damped phase memory | Let old phase gradually lose influence | Current cumulative phase versus bounded learned damping | Different from non-cumulative rotation | Low cost; medium long-range-loss risk |
| E3 | Learned phase reset | Reset coordinates at causal event boundaries | No reset, causal learned reset, and oracle boundaries | Different from fixed boundaries and moving-frame oracle | Medium boundary-detection risk |
| E4 | Multi-frequency rotation groups | Track several periodicities at once | One frequency group versus equal-parameter multiple groups | Different from one learned increment per pair | Medium projection/read cost |
| E5 | Value-space rotation | Transform stored value coordinates as well as addresses | q/k-only rotation versus identity-gated q/k/v rotation | Current rotation is primarily q/k addressing | High recurrence and frame-consistency risk |
| E6 | Query-dependent output-slot mix | Route among `r_out` reads per token | Static `out_mix` versus token-conditioned identity-gated residual mix | Current `out_mix` is static per head | Low-medium projection cost |
| E7 | Independent slot projections | Give slots low-rank distinct query directions | Scaled shared q versus equal-parameter low-rank slot residuals | Current slots are scaled copies of one q | Medium parameter and attribution risk |
| E8 | Multi-scale/dilated convolution | Capture several local horizons cheaply | Current kernel versus equal-parameter multi-scale kernels | Current reliance test only toggles one short convolution | Medium local-mixer cost |
| E9 | Dynamic convolution | Make local kernel weights content dependent | Winning fixed convolution versus bounded dynamic residual kernel | New local mixer rather than convolution on/off | High kernel and parameter cost |
| E10 | Local-attention hybrid | Preserve exact nearby interactions while recurrence handles long range | KMD-2-only versus compute-matched KMD-2 plus fixed local window | Requires mixer-layout rather than recurrence-only attribution | High architecture and cache cost |

Relevant broader references include [GateLoop](https://arxiv.org/abs/2311.01927),
[Griffin](https://arxiv.org/abs/2402.19427), and
[Mamba-3](https://arxiv.org/abs/2603.15569).

## F. Architecture-Fork Backlog

These candidates change the memory class enough that they should not enter the
existing serial ablation registry as ordinary warm-start arms.

| ID | Architecture | What it would test | First discriminating comparison | Why isolated |
|---|---|---|---|---|
| F1 | MesaNet-style local solver | Whether near-optimal online regression beats one delta step | Positive A12 prototype versus production-shaped solver at matched iterations | Additional Gram state and iterative inference FLOPs |
| F2 | TTT-Linear comparator | Whether a learned online objective beats fixed delta semantics | Tiny KMD-2 versus state/parameter-matched TTT-Linear | Different training/update abstraction |
| F3 | TTT-MLP state | Whether nonlinear fast weights capture nonlinearly separable associations | Passed TTT-Linear baseline versus closest state/parameter-matched TTT-MLP | Inner-loop differentiation and severe memory I/O |
| F4 | Titans-style neural memory | Whether a deep surprise-updated memory complements local attention/KMD-2 | KMD-2 plus local mixer versus compute-matched neural-memory hybrid | Separate neural-memory module and training system |
| F5 | Log-linear KMD-2 | Whether `O(log T)` state growth closes recall gaps | Positive B5 prototype versus production-shaped dyadic bank | Abandons strict constant memory |
| F6 | Sparse KMD-2 | Whether orders-of-magnitude larger sparse state beats dense recurrence at isoFLOP | Positive B4 prototype versus production-shaped sparse state | New addressing and sparse kernels |
| F7 | Mixture of memory operators | Route among delta state, exact cache, local attention, and neural memory | Best individual operators versus equal-compute routed mixture | Attribution and router-collapse risk |
| F8 | RWKV-7-style generalized update | Combine vector gates, in-context rates, and relaxed replacement | Native KMD-2 versus complete generalized-update replacement and its component controls | Multi-mechanism redesign rather than one isolated addition |

References:

- [MesaNet](https://arxiv.org/abs/2506.05233)
- [Learning to Learn at Test Time](https://arxiv.org/abs/2407.04620)
- [Titans](https://arxiv.org/abs/2501.00663)
- [RWKV-7 Goose](https://arxiv.org/abs/2503.14456)

## G. Training Backlog

These interventions can improve an existing mechanism without changing its
inference equation.

| ID | Training intervention | Intended effect | First discriminating comparison | Risk/cost |
|---|---|---|---|---|
| G1 | Identity-gate curriculum | Open new mechanisms sequentially without destroying the warm start | Owner's fixed-open schedule versus preregistered staged opening | Low inference cost; schedule attribution risk |
| G2 | Mechanism-specific optimizer groups | Give inherited weights, identity gates, cache reads, and new projections appropriate learning rates | One optimizer group versus parameter-count-identical grouped rates | Low inference cost; tuning multiplicity |
| G3 | Teacher memory distillation | Match teacher reads, retrieval rankings, or selected associations in addition to logits | Logit-only distillation versus added memory/read target | Teacher-trace storage and target-mismatch risk |
| G4 | State reconstruction objective | Make recurrent state recover selected prior values or teacher reads | Task loss alone versus bounded reconstruction auxiliary loss | Extra training compute and loss-weight tuning |
| G5 | Length curriculum | Grow context while protecting short-context quality | Fixed maximum length versus matched-token progressive lengths | Low architecture cost; curriculum confounding |
| G6 | Capacity curriculum | Increase competing associations before raw token length | Length-first versus matched-token association-count-first curriculum | Low architecture cost; curriculum confounding |
| G7 | Overwrite curriculum | Train repeated keys, changed values, corrections, and stale facts explicitly | Generic task mixture versus matched-token overwrite-enriched mixture | Data-generation and transfer risk |
| G8 | Collision curriculum | Train on deliberately similar keys and adversarial interference | Random keys versus matched-token controlled-similarity keys | Data-generation and over-specialization risk |
| G9 | Lane/slot diversity loss | Encourage nonredundant MIMO lanes and cache slots | No auxiliary loss versus bounded correlation/orthogonality loss | Low inference cost; useful redundancy may be penalized |
| G10 | State-condition regularization | Penalize exploding norm, poor conditioning, or collapsed effective rank | No regularizer versus one preregistered condition penalty | Extra diagnostics and regularizer tuning |
| G11 | Cache-use calibration | Encourage useful cache dependence without abandoning recurrent memory | Task loss alone versus bounded cache-calibration auxiliary loss | Metric gaming and loss-weight tuning |
| G12 | Mechanism dropout | Prevent cache, convolution, or one lane from becoming the only functional path | No dropout versus one matched-rate path-drop schedule | Training variance and under-use risk |
| G13 | Layerwise sensitivity installation | Upgrade only layers that demonstrate mechanism-specific headroom | Uniform installation versus equal-count probe-selected layers | Probe leakage and selection bias |
| G14 | Progressive unfreezing | Train new memory parameters before inherited projections or full blocks | Immediate full unfreeze versus matched-update staged unfreeze | Longer training protocol and attribution risk |

## H. Kernel and Numerical Backlog

| ID | Systems direction | Intended effect | Required terminal comparison | Risk/cost |
|---|---|---|---|---|
| H1 | Log-space decay products | Prevent underflow in long chunk scans | Direct products versus log-space products across the declared length/dtype grid | Extra transcendental work and kernel complexity |
| H2 | Sign-aware associative scan | Make negative transitions parallel and stable | A4 reference loop versus signed parallel scan, forward and backward | High derivation and zero-crossing risk |
| H3 | Fused channel-gate scan/backward | Make A2 production-feasible | A2 reference loop versus fused scan, forward and backward | High kernel/backward implementation cost |
| H4 | Fused DeltaProduct WY scan | Avoid sequential per-edit overhead | A5 sequential reference versus fused WY scan, forward and backward | High algebra and kernel cost |
| H5 | Selective backward recomputation | Store less gate/intermediate state | Stored-intermediate versus recompute path at parity and matched batch | More backward FLOPs |
| H6 | Selective FP32 accumulation | Protect recurrence, prefix products, and normalization while retaining BF16 projections | Full FP32 oracle, mixed precision, and BF16 across length grid | Bandwidth and register pressure |
| H7 | FP8 or block-floating recurrent state | Reduce state bandwidth and footprint | FP32/BF16 state versus quantized state at fixed model and context | High numerical and hardware portability risk |
| H8 | Error-feedback state quantization | Reinject quantization residuals instead of accumulating bias | Passed H7 quantizer with and without error feedback | Additional residual state and kernel complexity |
| H9 | Persistent decode kernel | Keep state resident and fuse project-update-read-output operations | Eager token decode versus persistent fused decode at output parity | High integration and device-specific cost |
| H10 | Adaptive chunk size | Choose the fastest safe chunk from length and available memory | Fixed chunk grid versus deterministic policy over the same grid | Policy maintenance and benchmark overfitting |
| H11 | Operator-specific checkpointing | Trade compute only where a new memory path is activation-heavy | No checkpointing versus declared checkpoint set at matched batch | Extra backward FLOPs and code paths |
| H12 | Per-recurrence autotuning | Tune blocks separately for native, signed, channel-gated, and multi-edit scans | Shared tuning versus recurrence-specific tuning on frozen parity cases | Compile time and cache/provenance complexity |

Systems work must follow a passed reference-loop mechanism screen. Kernel speed
does not rescue a scientifically unsuccessful recurrence.

## Required Measurements for Future Work

Every new mechanism should report, where applicable:

- primary task metric and paired confidence interval;
- protected validation loss and short-context quality;
- state Frobenius norm over length;
- effective state rank and singular-value spectrum;
- transition/Jacobian spectral-radius estimate;
- memory overwrite half-life;
- read-after-write and read-after-conflicting-write accuracy;
- duplicate-key and near-collision sensitivity;
- per-head and per-layer cache utility;
- MIMO lane correlation and router entropy;
- compaction discarded energy and reconstruction error;
- exact parameter count and recurrent-state bytes;
- training FLOPs, peak memory, tokens per second, and decode latency;
- BF16-versus-FP32 recurrence drift; and
- failure type rather than silent resource reduction.

## Promotion Process

### Stage 0: Select one bounded tranche

Write a dedicated design for one mechanism family or one small factorial. Do
not create a plan that attempts to implement this entire backlog. The design
must set exact caps for added parameters, persistent-state bytes, reference
FLOPs, peak memory, remote-run expenditure, and the number of jobs. It must
also update the selected lifecycle row to `designing` and link the design.

### Workflow routing

The lifecycle registry's promotion value selects the applicable workflow.
Stages not listed are skipped, not silently treated as passed.

| Promotion | Applicable workflow and terminal rule |
|---|---|
| `NATIVE` | Stages 1-6: Tiny reference/evidence, kernel feasibility, matched Qwen warm-start/heal, then gated interactions |
| `CAPACITY` | Stages 1-4 with budget-matched cold/redesigned controls; stop for a dedicated production-install design rather than entering Qwen automatically |
| `CACHE-QWEN` | Use the existing exact-cache Tiny selector/read/capacity gates, then its Qwen preflight/heal workflow; recurrent-update kernel work is out of scope unless the cache design changes it |
| `HYBRID-DESIGN` | Use a Tiny hybrid reference with both cache and recurrent-update parity, mechanism evidence, and resource accounting; stop for a dedicated Qwen integration design |
| `GDN3-MODEL` | GDN3 reference and compaction screen, GDN3 chunk parity/efficiency, then GDN3 model evaluation; skip native KMD-2 Tiny/Qwen heal |
| `FORK-DESIGN` | Isolated backend reference, mechanism evidence, and resource feasibility; stop and write a new architecture design before any production integration |
| `OWNER` | Use the selected owner mechanism's workflow; a training-only intervention skips recurrence-kernel derivation and compares training cost instead |
| `SYSTEM-TERMINAL` | Start only after owner mechanism evidence; finish after forward/backward parity, numerical, portability, and speed/resource gates; do not claim model-quality promotion |

### Stage 1: Mathematical and identity proof

Specify the recurrence, state, parameter shapes, disabled identity, boundary
semantics, and expected computational complexity. Cold redesigns must say so.

### Stage 2: Owning reference screen

Implement the workflow's pure-PyTorch or owning-backend reference,
finite-difference/gradient tests, active-effect proof, deterministic paired
tasks, and state/parameter accounting. `GDN3-MODEL` uses its GDN3 reference;
`FORK-DESIGN` uses an isolated backend.

### Stage 3: Mechanism-specific evidence

Run the smallest task that distinguishes the claimed capability. A mechanism
that does not improve its claimed diagnostic does not advance because of a
generic loss fluctuation.

### Stage 4: Efficiency feasibility

Measure reference cost, derive the parallel form, and establish kernel parity.
No silent changes to batch, sequence, dtype, state, or budget are allowed.

### Stage 5: Destination-specific production validation

For `NATIVE` and `CACHE-QWEN`, promote only after Tiny evidence and a viable
kernel/resource estimate, then run matched Qwen examples and protected metrics
with immutable provenance. `GDN3-MODEL` uses the GDN3 production path.
`CAPACITY`, `HYBRID-DESIGN`, and `FORK-DESIGN` stop for a newly approved
production design.
`OWNER` and `SYSTEM-TERMINAL` follow the routing table.

### Stage 6: Interactions

Only successful individual mechanisms may be crossed with the registered
`exact_cache.selector.exact_outer` arm (or a later explicitly recorded cache
winner), rotation, convolution, MIMO, or one another. Use complete factorial
cells. This does not authorize or claim a separate literal HOLA model.

## Recommended Future Order

### First wave: constant-memory update geometry

1. A1 online diagonal preconditioning.
2. A2 channel-wise erase/write factorial.
3. A4 signed transitions.
4. A5 DeltaProduct with `n_h=2` before `n_h=4`.
5. A3 independent erase address.

These candidates stay closest to native KMD-2, have clear discriminating
tasks, and can reveal whether update geometry is the main limitation before
adding raw capacity.

### Second wave: stability and adaptive capacity

1. A11 lightweight state-energy normalization.
2. A6 fully dynamic channel decay.
3. C1 query-dependent cache fusion.
4. B1 fast/slow states.
5. D1 decay-consistent GDN3 factors and D4 adaptive compaction blend.

### Separate high-risk tracks

- A10 state-aware orthogonal novelty writes.
- A12/F1 local ridge/Mesa solver.
- B4/F6 sparse memory.
- B5/F5 log-linear state banks.
- F3/F4 nonlinear TTT or Titans-style memory.

These must receive independent designs and resource budgets. They should not
be appended to the current exact-cache matrix as ordinary arms.

## Backlog State Model

Each candidate may have exactly one state:

- `parked`: documented only; default for every item in this file;
- `designing`: explicitly selected for a dedicated design;
- `approved`: design reviewed and approved, but no implementation yet;
- `planned`: implementation plan reviewed and approved;
- `active`: implementation or experiment in progress;
- `resolved-positive`: completed evidence supports promotion;
- `resolved-negative`: completed evidence rejects or limits the idea; or
- `superseded`: replaced by a better-defined mechanism.

Changing an item from `parked` requires explicit user direction. Updating this
backlog does not itself authorize code changes, remote runs, or expenditure.
The normalized lifecycle registry is authoritative:

- `designing`, `approved`, `planned`, and `active` require a non-`none` design
  link;
- `resolved-positive` and `resolved-negative` require a non-`none` immutable
  evidence link;
- `superseded` requires a valid candidate ID in `Superseded by`;
- a transition must update exactly one candidate row unless an approved
  factorial design explicitly selects several rows; and
- a `candidate:` or `candidates:` dependency is satisfied only by the named
  lifecycle state, not by an informal result or generic test pass;
- an `evidence:` dependency is satisfied only after `@none` is replaced by the
  immutable artifact link and the same link is recorded in `Evidence`; and
- an `owner:` dependency is satisfied only after `none` is replaced by a
  concrete candidate ID in `approved`, `planned`, `active`, or
  `resolved-positive`, with a linked approved design; `resolved-negative` and
  `superseded` owners are explicitly ineligible.

## Decision

Preserve the full landscape for later, but begin any future work with a small
P-A tranche. The default recommended sequence is online diagonal
preconditioning, channel-wise erase/write gates, signed transitions, and a
two-edit DeltaProduct screen. Capacity expansions and architecture forks wait
until those experiments show whether KMD-2 is limited by optimization
geometry, state-transition expressivity, or raw memory capacity.
