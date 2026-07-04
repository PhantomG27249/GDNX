"""Tokenise source documents and pack them into fixed-length blocks per band.

Packing = concatenate tokenised docs with an EOS separator, then slice into
`seq_len` blocks. Two rules protect long-range structure in the long bands:

  * `min_doc_tokens` floor — a source routed to a 32k/65k band must yield docs that
    are *genuinely* long; short docs are skipped so a long window is never filled by
    gluing many independent short docs together (which would teach the compressor
    that "long context = independent chunks" — the opposite of the goal).
  * documents stay contiguous — a long doc spans whole blocks; block-level shuffle
    happens only *after* packing.

Sources may also be `prepacked` (an existing blocks.pt from a prior mix) — ingested
directly with no retokenisation. A small per-source slice is held out for
per-category eval before the train shuffle.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch

from .registry import get_source


def pack_source(name, tokenizer, budget_tokens, seq_len, seed, config=None,
                max_doc_tokens=None, min_doc_tokens=0, max_skip=20000):
    """Stream `name` until ~budget_tokens are packed into seq_len blocks."""
    cfg = dict(config or {})
    rng = random.Random(seed + sum(ord(c) for c in name))
    it = get_source(name)(rng, tokenizer=tokenizer, **cfg)
    eos = tokenizer.eos_token_id or 0

    buf, got, ndocs, skipped, since_kept = [], 0, 0, 0, 0
    for text in it:
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if min_doc_tokens and len(ids) < min_doc_tokens:
            skipped += 1
            since_kept += 1
            # guard against an infinite generator whose docs never clear the floor
            # (a misconfigured recipe: short source in a high-floor band)
            if since_kept >= max_skip:
                print(f"    !! {name}: {since_kept} consecutive docs below "
                      f"min_doc_tokens={min_doc_tokens}; giving up", flush=True)
                break
            continue
        since_kept = 0
        if max_doc_tokens and len(ids) > max_doc_tokens:
            ids = ids[:max_doc_tokens]
        # enforce the budget mid-document so one huge doc can't skew the mixture
        remaining = budget_tokens - got
        if len(ids) + 1 > remaining:
            ids = ids[:max(0, remaining - 1)]
        buf.extend(ids)
        buf.append(eos)
        got += len(ids) + 1
        ndocs += 1
        if got >= budget_tokens:
            break

    nblk = len(buf) // seq_len
    stats = {"name": name, "docs": ndocs, "skipped_short": skipped,
             "tokens": int(nblk * seq_len), "blocks": nblk}
    if nblk == 0:
        return None, stats
    t = torch.tensor(buf[:nblk * seq_len], dtype=torch.int32).view(nblk, seq_len)
    return t, stats


def pack_prepacked(path, budget_tokens, seq_len):
    """Ingest an existing blocks.pt (same tokenizer). Truncates block width to
    seq_len; a prepacked block narrower than the band's seq_len can't be used."""
    p = Path(path)
    blocks = torch.load(p / "blocks.pt" if p.is_dir() else p).to(torch.int32)
    width = blocks.shape[1]
    if width < seq_len:
        raise ValueError(f"prepacked {path} width {width} < band seq_len {seq_len}")
    if width > seq_len:
        blocks = blocks[:, :seq_len].contiguous()
    nblk = min(blocks.shape[0], budget_tokens // seq_len)
    blocks = blocks[:nblk]
    return blocks, {"name": f"prepacked:{p.name}", "docs": nblk,
                    "skipped_short": 0, "tokens": int(nblk * seq_len), "blocks": nblk}


def build_band(band_name, entries, tokenizer, band_tokens, seq_len, seed,
               min_doc_tokens=0, heldout_blocks=0):
    """Build one length band from its source entries.

    entries: list of {source|prepacked, weight, config?, role?, category?,
    max_doc_tokens?, min_doc_tokens?}. Returns (train, heldout, heldout_cats, stats).
    A genuine-long band passes a default `min_doc_tokens` floor; per-entry override wins.
    """
    tot_w = sum(e["weight"] for e in entries) or 1.0
    train_blocks, held_blocks, held_cats, stats = [], [], [], []
    for e in entries:
        budget = int(band_tokens * e["weight"] / tot_w)
        cat = e.get("category", e.get("source") or "prepacked")
        role = e.get("role", "stress" if min_doc_tokens else "filler")
        if "prepacked" in e:
            t, st = pack_prepacked(e["prepacked"], budget, seq_len)
        else:
            floor = e.get("min_doc_tokens", min_doc_tokens)
            t, st = pack_source(e["source"], tokenizer, budget, seq_len, seed,
                                config=e.get("config"),
                                max_doc_tokens=e.get("max_doc_tokens"),
                                min_doc_tokens=floor)
        st.update(band=band_name, weight=e["weight"], role=role, category=cat)
        stats.append(st)
        print(f"    [{band_name}] {st['name']:>22}: {st['tokens']:>12,} tok "
              f"-> {st['blocks']:>6} x {seq_len}  ({role}"
              + (f", skip<floor {st['skipped_short']}" if st.get('skipped_short') else "")
              + ")", flush=True)
        if t is None:
            continue
        nh = min(heldout_blocks, t.shape[0] // 10) if heldout_blocks else 0
        if nh:
            held_blocks.append(t[:nh]); held_cats += [cat] * nh
            t = t[nh:]
        train_blocks.append(t)

    if not train_blocks:
        return None, None, [], stats
    data = torch.cat(train_blocks, dim=0)
    g = torch.Generator().manual_seed(seed + hash(band_name) % (2**31))
    data = data[torch.randperm(data.shape[0], generator=g)]
    held = torch.cat(held_blocks, dim=0) if held_blocks else None
    return data, held, held_cats, stats
