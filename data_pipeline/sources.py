"""Natural-data source loaders (HF streaming) for the attention-rebuild pipeline.

Every loader yields raw document *strings*; tokenising/packing is downstream.
Only datasets that (a) load under `datasets` 5.x without a dataset script and
(b) expose reusable text (not Llama-pretokenised shards, not Software-Heritage
blob_ids) are wired here. See DESIGN.md / SOURCES.md for the ones that were ruled
out and why.

Coverage axes:
  web       fineweb-edu            behavior-preserve backbone (short-mid)
  math      finemath-4+, owm       dense local-precision
  books     pg19 (emozilla mirror) diffuse long-range narrative
  multihop  musique, hotpotqa      natural dense multi-referent retrieval
  agentic   nemotron-agentic-v2    multi-turn tool-call/observation state tracking
  code      starcoderdata          repo-level cross-file long-range  [gated=auto]
"""
from __future__ import annotations

from typing import Iterator, Optional

from .registry import source


def _stream(repo, config=None, split="train", **kw):
    from datasets import load_dataset
    return load_dataset(repo, config, split=split, streaming=True, **kw)


# --------------------------------------------------------------------------- #
# web / math / books — simple text-column streamers
# --------------------------------------------------------------------------- #
@source("fineweb_edu")
def fineweb_edu(rng, tokenizer=None, config="sample-10BT", min_score=0.0, **_) -> Iterator[str]:
    for ex in _stream("HuggingFaceFW/fineweb-edu", config):
        if min_score and float(ex.get("score", 0)) < min_score:
            continue
        t = ex.get("text")
        if t:
            yield t


@source("finemath")
def finemath(rng, tokenizer=None, config="finemath-4plus", **_) -> Iterator[str]:
    for ex in _stream("HuggingFaceTB/finemath", config):
        if ex.get("text"):
            yield ex["text"]


@source("open_web_math")
def open_web_math(rng, tokenizer=None, **_) -> Iterator[str]:
    for ex in _stream("open-web-math/open-web-math"):
        if ex.get("text"):
            yield ex["text"]


@source("pg19")
def pg19(rng, tokenizer=None, **_) -> Iterator[str]:
    # emozilla mirror = parquet (the canonical deepmind/pg19 is a dead script dataset)
    for ex in _stream("emozilla/pg19"):
        if ex.get("text"):
            yield ex["text"]


# --------------------------------------------------------------------------- #
# genuine-long English — full papers / full encyclopedia articles (role: stress)
# --------------------------------------------------------------------------- #
@source("arxiv_papers")
def arxiv_papers(rng, tokenizer=None, **_) -> Iterator[str]:
    # full-text arXiv papers (~100k chars), permissive; genuine long-range technical
    for ex in _stream("common-pile/arxiv_papers"):
        if ex.get("text"):
            yield ex["text"]


@source("wikipedia_en")
def wikipedia_en(rng, tokenizer=None, config="20231101.en", **_) -> Iterator[str]:
    for ex in _stream("wikimedia/wikipedia", config):
        t = ex.get("text")
        if t:
            yield f"{ex.get('title','')}\n\n{t}" if ex.get("title") else t


# --------------------------------------------------------------------------- #
# Chinese lane — Qwen is strongest in en + zh
# --------------------------------------------------------------------------- #
@source("chinese_edu")
def chinese_edu(rng, tokenizer=None, min_score=0.0, **_) -> Iterator[str]:
    for ex in _stream("opencsg/Fineweb-Edu-Chinese-V2.1"):
        if min_score and float(ex.get("score", 0)) < min_score:
            continue
        if ex.get("text"):
            yield ex["text"]


@source("wikipedia_zh")
def wikipedia_zh(rng, tokenizer=None, config="20231101.zh", **_) -> Iterator[str]:
    for ex in _stream("wikimedia/wikipedia", config):
        t = ex.get("text")
        if t:
            yield f"{ex.get('title','')}\n\n{t}" if ex.get("title") else t


# --------------------------------------------------------------------------- #
# multi-hop QA — natural multi-referent contexts
# --------------------------------------------------------------------------- #
@source("musique")
def musique(rng, tokenizer=None, **_) -> Iterator[str]:
    for ex in _stream("dgslibisey/MuSiQue"):
        paras = " ".join(p["paragraph_text"] for p in ex["paragraphs"])
        yield f"{paras}\nQuestion: {ex['question']}\nAnswer: {ex.get('answer', '')}"


@source("hotpotqa")
def hotpotqa(rng, tokenizer=None, config="distractor", **_) -> Iterator[str]:
    for ex in _stream("hotpotqa/hotpot_qa", config):
        ctx = ex["context"]
        paras = " ".join("".join(s) for s in ctx["sentences"])
        yield f"{paras}\nQuestion: {ex['question']}\nAnswer: {ex.get('answer', '')}"


# --------------------------------------------------------------------------- #
# agentic — multi-turn tool-call conversations, serialised via the chat template
# --------------------------------------------------------------------------- #
def _normalize_tool_calls(msgs):
    """Nemotron stores tool_call arguments as a JSON *string*; the Qwen chat
    template expects a mapping. Parse it (and drop reasoning_content, which the
    template doesn't consume) so apply_chat_template renders real tool calls."""
    import json
    out = []
    for m in msgs:
        m = dict(m)
        m.pop("reasoning_content", None)
        tcs = m.get("tool_calls")
        if tcs:
            fixed = []
            for tc in tcs:
                tc = dict(tc); fn = dict(tc.get("function", {}))
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except Exception:
                        fn["arguments"] = {}
                tc["function"] = fn
                fixed.append(tc)
            m["tool_calls"] = fixed
        out.append(m)
    return out


@source("nemotron_agentic")
def nemotron_agentic(rng, tokenizer=None,
                     splits=("tool_calling", "interactive_agent"), **_) -> Iterator[str]:
    """Serialise multi-turn tool-call conversations to a flat token stream.

    Uses the model's chat template with the row's `tools` so the teacher sees the
    real function schemas + call/observation history. The `search` split is skipped
    by default (inconsistent schema across shards — needs non-streaming load)."""
    if tokenizer is None:
        raise ValueError("nemotron_agentic needs a tokenizer for the chat template")
    for sp in splits:
        for ex in _stream("nvidia/Nemotron-SFT-Agentic-v2", split=sp):
            msgs, tools = ex.get("messages"), ex.get("tools")
            if not msgs:
                continue
            try:
                text = tokenizer.apply_chat_template(
                    _normalize_tool_calls(msgs), tools=tools,
                    tokenize=False, add_generation_prompt=False)
            except Exception:
                continue  # skip rows the template genuinely can't render
            if text:
                yield text


@source("ultrachat")
def ultrachat(rng, tokenizer=None, split="train_sft", **_) -> Iterator[str]:
    """Multi-turn chat (no tools) → chat template, so the heal doesn't degrade
    plain multi-turn conversational structure before the downstream EBFT stage."""
    if tokenizer is None:
        raise ValueError("ultrachat needs a tokenizer for the chat template")
    for ex in _stream("HuggingFaceH4/ultrachat_200k", split=split):
        msgs = ex.get("messages")
        if not msgs:
            continue
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            continue
        if text:
            yield text


# --------------------------------------------------------------------------- #
# code — per-repo file concatenation (cross-file long-range signal)
# --------------------------------------------------------------------------- #
import re as _re
# StarCoder sentinels that appear as literal text in the raw `content`
_SC_SENTINEL = _re.compile(r"<(reponame|filename|gh_stars|empty_output|"
                           r"fim_prefix|fim_middle|fim_suffix|fim_pad|jupyter_[a-z]+)>[^\n]*")


def _clean_code(s):
    return _SC_SENTINEL.sub("", s)


@source("starcoderdata")
def starcoderdata(rng, tokenizer=None, langs=("python",), repo_chars=48000,
                  content_col="content", repo_col="max_stars_repo_name",
                  path_col="max_stars_repo_path", **_) -> Iterator[str]:
    """Per-repo concatenation: buffer consecutive files sharing a repo, emit a
    single doc with `# path` headers so cross-file dependencies live in one window.

    starcoderdata is gated=auto (one-time accept on the token's HF account). Column
    names differ per language dir; defaults match the python config."""
    from datasets import load_dataset
    for lang in langs:
        ds = load_dataset("bigcode/starcoderdata", data_dir=lang,
                          split="train", streaming=True)
        cur_repo, buf, nchars = None, [], 0
        for ex in ds:
            repo = ex.get(repo_col) or ex.get("repo_name") or ""
            path = ex.get(path_col) or ex.get("path") or ""
            content = _clean_code(ex.get(content_col) or ex.get("content") or "")
            if not content.strip():
                continue
            if repo != cur_repo and buf:
                yield "\n\n".join(buf)
                buf, nchars = [], 0
            cur_repo = repo
            buf.append(f"# {path}\n{content}")
            nchars += len(content)
            if nchars >= repo_chars:      # cap a single mega-repo
                yield "\n\n".join(buf)
                buf, nchars = [], 0
        if buf:
            yield "\n\n".join(buf)


@source("the_stack_smol")
def the_stack_smol(rng, tokenizer=None, **_) -> Iterator[str]:
    """Tiny ungated inline code set — a runnable stand-in for starcoderdata so the
    code lane is exercisable before the gated accept lands."""
    for ex in _stream("bigcode/the-stack-smol", data_dir="data/python"):
        c = ex.get("content")
        if c:
            yield f"# {ex.get('path','')}\n{c}"
