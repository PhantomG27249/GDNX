# Package B Remaining Optimization Design

## Authority and scope

The current `package-b-hola-w64` PyTorch implementation and its tests, plus the
user's explicit compact-lane instruction, are authoritative.  Bible, README,
metadata, and comment drift cannot change the target.  `RTK.md` is absent and
stale.  GDN2, public APIs, parameter/checkpoint/cache schemas, configuration
IDs, masks, resets, document boundaries, and output semantics remain unchanged.
The optimized target is exactly four native compact lanes of width `K/4`, four
writes, sixteen logical cross-lane reads, the dense mixer, CMS periods
`(1, 16, 64, 256)`, and always-enabled HOLA-W64.  Construction/configuration
must fail closed if native `K` cannot be partitioned into four valid compact
complex-pair bands; Package B must never instantiate or execute full-width
lanes.  Other unsupported optimization conditions on an otherwise valid
compact-lane Package B use the authoritative complete PyTorch path.
The previous key/value endpoints and factorized trapezoidal history have exactly
the current representation and update order.

The completed foundation is:

- `a061c62`: independent dense-history oracle and exact generalized-WY
  PyTorch recurrence;
- `cfee867`: fail-closed no-grad CUDA dispatch for the canonical Package B
  prefill gap, after the existing batch-1 Triton backend.

## Remaining architecture

### 1. Exact rematerialized training backward

Wrap the proven generalized-WY recurrence in a private `torch.autograd.Function`.
Forward runs the same exact recurrence without retaining interval products,
triangular factors, or state traces.  It saves only the ten FP32 floating inputs
and the bool/int carry metadata.  Backward recreates differentiable input views,
re-executes the exact forward under `torch.enable_grad()`, and calls
`torch.autograd.grad` with the incoming cotangents for reads and final state/key/
value carries.  HOLA innovation scores stay detached exactly as the baseline.
There is no inverse or reverse-state reconstruction.

Training dispatch is restricted to the same canonical CUDA shape envelope and
must beat or reduce peak memory versus the forced complete PyTorch loop.  The
existing `force_torch_recurrence` attribute bypasses both forward and training
optimized paths.  Existing outer non-reentrant segment checkpointing stays in
place unless benchmarks prove redundant removal is safe.

No candidate may be wired into production until its raw recurrence, complete
module output/cache, and VJP parity are proved first.  Unsupported shape,
native width, dtype, device, mask/reset/document-boundary pattern, nonuniform
HOLA occupancy, or any noncanonical state takes the authoritative complete
PyTorch path.  This applies independently to training, prefill, and decode;
`force_torch_recurrence` is an absolute diagnostic escape hatch.

### 2. Prefill and decode separation

Prefill remains the chunkwise generalized-WY path for eligible 16-64-token
segments.  Token-by-token decoding stays on the current exact one-token path
unless a dedicated vectorized step wins synchronized full-Package-B benchmarks.
The existing batch-1 Triton path retains priority.  No configuration toggle or
alternate model variant is introduced.

### 3. Memory, layout, and workspace investigation

Measure time-first versus current `[B,T,H,R,...]` layouts, shorter internal WY
subchunks, direct read contraction, and trace-free output formation.  Adopt only
changes that improve synchronized complete-model throughput or peak memory.
Reusable module-global mutable workspaces are forbidden because they would break
reentrancy, checkpoint recomputation, and data-parallel safety.  Safe local
preallocation or `out=` buffers may be used only in no-grad code and only after
parity and benchmark evidence.

### 4. Reads, mixer, and HOLA fusion gates

First profile recurrence, normalization/dense mixer, and `HOLA.scan_fast`
separately.  A fusion is implemented only when it is a measured bottleneck and
can preserve all gradients and cache visibility.  HOLA promotion remains every
256 valid tokens with identical staging, read-before-promotion behavior, sink
normalization, epochs, and selection.  Otherwise the already-batched existing
implementation remains authoritative.

### 5. Validation and acceptance

Every new path is compared with `force_torch_recurrence` on the complete
Package-B module with HOLA enabled.  Acceptance includes the entire forward
output and complete cache, including `conv_tail`; all recurrent states, phase,
previous factors, factorized history, history flags, and counters; dense history
reconstruction; and every HOLA tensor, counter, epoch, staged item, visible set,
and score.  CMS clock-boundary checks at 16, 64, and 256 are distinct from HOLA
promotion/visibility checks at valid-token 256.  Tests cover FP32 recurrence
under FP32 and bf16 modules, arbitrary scan partitions, invalid tokens, packed
documents and resets through fail-closed fallback, token-by-token cached decode,
and exhaustive VJPs for the input and every named trainable Package-B/HOLA
parameter.  Multiple optimizer steps compare both parameter and optimizer
state.  Public APIs/configuration IDs, parameter and state-dict names/shapes/
dtypes/counts, checkpoint save/load/resume, and the unchanged GDN2 suites are
compatibility gates.  Benchmarks are synchronized, interleaved, complete-module
comparisons and report median throughput plus peak allocated memory for
32/64/128/256 chunks.

The upstream Liger issue/PR primarily patches Qwen3.5 RMSNorm, SwiGLU, and fused
linear cross entropy; it explicitly notes that upstream
`ChunkGatedDeltaRuleFunction` does not support FP32.  Therefore its optimization
cannot replace Package B's required FP32 recurrent mathematics directly; only
its rematerialization/fusion discipline is applicable.
