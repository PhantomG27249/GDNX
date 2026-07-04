# Source status — verified against this environment (datasets 5.x, token=senorperez)

Probed 2026-07-02; **full proof build passed across all bands/sources.** `datasets`
5.x **refuses to run dataset scripts**, which kills a lot of classic corpora; and
several code sets ship only Software-Heritage blob_ids, not text. This table is the
ground truth for what the pipeline can actually load.

## Wired & verified — ALL GREEN (full proof build, 3 bands × 8 sources)

| Source name | HF repo | Text via | Notes |
|---|---|---|---|
| `fineweb_edu` | HuggingFaceFW/fineweb-edu | `text` | ✅ config `sample-10BT` |
| `finemath` | HuggingFaceTB/finemath | `text` | ✅ config `finemath-4plus` |
| `open_web_math` | open-web-math/open-web-math | `text` | ✅ ungated parquet |
| `pg19` | **emozilla/pg19** | `text` | ✅ mirror (canonical deepmind/pg19 is a dead script) |
| `musique` | dgslibisey/MuSiQue | 20-para join | ✅ |
| `hotpotqa` | hotpotqa/hotpot_qa | context join | ✅ config `distractor` |
| `nemotron_agentic` | nvidia/Nemotron-SFT-Agentic-v2 | chat template | ✅ tool_calls normalized (JSON-str→dict) → renders real `<tool_call>`; splits `tool_calling`+`interactive_agent` |
| `starcoderdata` | bigcode/starcoderdata | `content`, per-repo concat | ✅ **gate ACCEPTED**; cols `max_stars_repo_{name,path}`; StarCoder sentinels stripped |
| `synth_*` | — | offline generators | ✅ multiquery / copy / needle |

## Ruled out (and why)

| Wanted | Verdict |
|---|---|
| deepmind/pg19, EleutherAI/proof-pile-2, codeparrot/github-code-clean | **dead** — dataset scripts, unsupported in datasets 5.x. (pg19 → emozilla mirror; math covered by finemath+open_web_math) |
| princeton-nlp/prolong-data-{64K,512K} | **unusable** — pre-tokenized *Llama* shards (`{version, shards}`), no raw text, wrong tokenizer |
| HuggingFaceTB/stack-edu, bigcode/the-stack-v2 | **blob-only** — schema is `blob_id/repo_name/path`, contents live in Software-Heritage S3 (separate creds) |
| bigcode/the-stack-smol | **gated=auto** — didn't pursue; starcoderdata covers the lane |

## Code lane — RESOLVED

`bigcode/starcoderdata` gate **accepted on `senorperez`** (2026-07-02) → lane live.
It has inline `content` + `max_stars_repo_{name,path}`, which the `starcoderdata`
source uses for **simple per-repo concatenation** (files from one repo joined with
`# path` headers → cross-file long-range in one window). StarCoder sentinel strings
(`<reponame>`/`<filename>`/`<gh_stars>`/`<fim_*>`) are stripped. Topological
(dependency-ordered) concat is deferred as a later enhancement. If we ever want
fresher/richer code, `the-stack-v2` needs its own gate accept **plus** Software-
Heritage S3 content plumbing.

## Operational notes (learned during the proof build)

- **Budget is enforced mid-document** (`pack.py`). Long-doc sources (PG19 books
  tokenize to >1M tokens each) were overshooting their weighted share massively —
  the `main` band came out ~56% books vs the 22% target — because the budget was
  only checked *after* appending a whole doc. Now the last doc is trimmed to the
  remaining budget, so recipe weights are faithful.
- **`HF_HUB_DISABLE_XET=1` on builds.** The hf_xet download threads throw a
  cosmetic `PyGILState_Release` core dump during interpreter teardown — *after* the
  manifest is written, so output is correct, but it's noisy. Disabling xet silences it.
- **Tokenizer**: use `Qwen/Qwen3.5-2B` (in `hub/`); the 0.8B snapshot lives in the
  cache root, not `hub/`. All Qwen3.5/3.6 share vocab 248320, so any works.
- A transformers warning "Token indices sequence length is longer than 262144" on
  PG19 is harmless — we don't run docs through the model at build time.

## Agentic pick — Nemotron-SFT-Agentic-v2 (over v1)

Chosen for: 5.5× the downloads, newer, and a dedicated **`search`** split (agentic
retrieval-over-long-context) matching our retrieval emphasis. Multi-turn is a plus:
serialized through the chat template, the teacher sees the full call/observation
history, forcing long-range state tracking. `tool_calling` + `interactive_agent`
stream cleanly; `search` needs non-streaming load (inconsistent shard schema).
