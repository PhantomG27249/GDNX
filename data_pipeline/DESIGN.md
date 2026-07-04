# Attention-Rebuild Data Pipeline — Design

**Status (2026-07-02): BUILT & VALIDATED.** Full proof build passed across 3 length
bands × 8 sources. Code gate accepted. See `SOURCES.md` for the source ledger +
operational notes.

A reusable dataset builder for **any** attention retrofit (MLA, CSA, GDN3). It
materialises tokenized, packed, provenance-tracked training blocks from a
**declarative recipe** so each project (and each scale) can pick its own mixture
without touching source code. Adding a project = adding a `recipes/*.yaml`; no code
changes. Run: `python -m data_pipeline.build --recipe R --out D [--scale S --only-band B]`.

## Framing constraints (what makes our case different from the suggestions doc)

- **Online teacher → we cache NO hidden states.** The suggestions doc's "cache
  TB / 1M seq" columns are about storing a teacher's FP16 activation stream; we
  run a frozen teacher forward per step instead. So the *only* thing we write to
  disk is tokenized `int32` packed blocks. At `int32`, 1B tokens ≈ 4 GB. With
  2.4 TB free, **storage is no longer the binding constraint** — token *quality*
  and *coverage* are.
- **One tokenizer across the whole Qwen3.5/3.6 family** (vocab 248320). A mix
  built once is reusable verbatim across 0.8B / 2B / 4B / 27B. Keep the builder
  tokenizer-parametrized, never hardcode vocab.
- **The lever is attention *capacity/behavior*, not context length** (our own
  finding: degradation scales with parallel-retrieval load, not ctx). So the mix
  must *stress* attention: multi-referent retrieval, cross-file code dependency,
  diffuse long-range integration — not just be big.

## Coverage axes (what the student's rebuilt attention must see)

| Axis | Why it matters for a rebuild | Primary sources |
|---|---|---|
| **Backbone web (short-mid)** | don't regress normal-length ability | FineWeb-Edu (+ DCLM-baseline optional) |
| **Code / repo-level (EMPHASIS)** | cross-file long-range exact retrieval — the CSA stress test | StarCoderData / Stack-Edu, **repo-topological concat** |
| **Retrieval + agentic (EMPHASIS)** | parallel multi-referent recall; tool/observation chaining | synthetic multi-query/copy/needle generators + MuSiQue/HotpotQA + agentic traces |
| **Math / local-precision** | dense local syntactic attention (opposite regime) | FineMath-4+, Proof-Pile-2, StackExchange |
| **Diffuse long-range** | narrative/research aggregation — worst regime for sparse selection | PG19 books, arXiv, ProLong-64K |

## Length-band coverage (the quantified lesson)

Mixing length bands beats any single band (Tulu3-32K + LongCite-128K > either
alone). Build explicit bands and use **per-source length-upsampling** (bias toward
long docs *within* a domain, keeping the domain mix fixed) so 32k+ mass appears in
*every* domain, not only books.

Proposed starting split (tunable per recipe):

| Band | Share | Fed by |
|---|---:|---|
| short ≤ 8k | 65% | backbone web, math, short code, short synthetic |
| mid ~32k | 25% | repo-concat code, books, multi-hop, long synthetic retrieval |
| long 128k+ | 10% | ProLong-64K/128K, extreme-position needle/copy |

## Proposed architecture

```
data_pipeline/
  sources/                # each yields raw docs (text + metadata), streaming
    registry.py           #   name -> loader; shared interface
    web.py                #   fineweb-edu, dclm-baseline
    code.py               #   starcoderdata / stack-edu; repo-topological concat
    mathsci.py            #   finemath-4+, proof-pile-2, arxiv, peS2o
    longform.py           #   pg19, prolong-64k (use as-is, keep boundaries)
    multihop.py           #   musique, hotpotqa
    synth_retrieval.py    #   multi-query, copy, needle, assoc-recall, induction,
                          #     bracket/stack, RAFT-style long-position copy
    agentic.py            #   tool-use / observation / multi-step traces  [OPEN]
  recipe.py               # dataclass: {source: (weight, band, preprocess)}
  pack.py                 # tokenize + pack into length bands; length-upsampling
  build.py                # CLI: recipe -> blocks_{band}.pt + manifest + provenance
  recipes/
    mla_27b.yaml  csa.yaml  gdn3.yaml
```

This generalises today's `mla_retrofit/data_mix.py` (which is a single-band,
5-source monolith with the synthetic generator + MuSiQue + code + wiki). That
generator and its packing loop port over as `synth_retrieval.py` + `pack.py`.

## Decisions (resolved 2026-07-02)

1. **Scale**: start small (validated), scale later. Real-build token budget flows
   from the 27B heal plan (steps × 4×B200 throughput × seq_len) — still TBD.
2. **Code source**: `starcoderdata` (inline content, gate accepted). Stack-Edu was
   rejected — it's blob-only (Software-Heritage S3), not inline as first assumed.
3. **Agentic lane**: `nvidia/Nemotron-SFT-Agentic-v2`, serialized via chat template.
4. **Code concat**: simple per-repo first. Topological (dependency-ordered)
   deferred as a later enhancement.

## Backlog / later enhancements

- Topological repo concatenation (import/dep-ordered) for stronger cross-file signal.
- `search` split of Nemotron-Agentic-v2 (needs non-streaming load — shard schema varies).
- Per-source length-upsampling knob (bias toward long docs *within* a domain) — today
  the long bands rely on naturally-long sources (books, per-repo code, long synthetic).
- `recipes/mla_27b.yaml`, `recipes/csa.yaml`, `recipes/gdn3.yaml` at target scale.
