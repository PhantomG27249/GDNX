"""Reusable multi-referent training mix for self-distillation / attention experiments.

Motivation: the MLA conversion's residual degradation is a function of *parallel
retrieval load* (number of simultaneous queries), not context length. General text
(wikitext) rarely exercises parallel multi-referent retrieval, so KL distillation on
it gives the rebuilt attention little pressure to develop that capacity. This module
builds a mix that does, while keeping a general anchor to avoid forgetting:

  1. synthetic_multiquery : generated key->value bindings (varied surface forms:
       name->code, product->price, var=value, person->city, ...) with explicit
       multi-key queries and teacher-forced answers. The highest-signal, fully
       offline, randomized component — directly drills parallel retrieval.
  2. musique               : MuSiQue multi-hop QA contexts (20 paragraphs each) —
       natural dense multi-referent text.
  3. wikipedia             : entity-dense natural text.
  4. code                  : codeparrot-clean Python — natural variable->use bindings.
  5. wikitext              : general anchor (the original training distribution).

`build_mix(...)` materialises a fixed, packed, shuffled token dataset to disk (int32
blocks + manifest) so runs are reproducible and reusable. `MaterializedMix` loads it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# 1. Synthetic multi-query generator
# --------------------------------------------------------------------------- #
_NAMES = ("Alice Bob Carol Dave Erin Frank Grace Heidi Ivan Judy Karl Lena Mallory "
          "Niaj Olivia Peggy Quinn Rupert Sybil Trent Uma Victor Wendy Xena Yuri Zoe "
          "Amir Bianca Cyrus Dalia Esme Faisal Greta Hugo Inez Jamal").split()
_CITIES = ("Lisbon Cairo Oslo Lima Dhaka Quito Riga Tunis Accra Hanoi Sofia Minsk "
           "Kyoto Perth Davao Cusco Bonn Nantes Leeds Ghent Tartu Pisa Cork Bergen").split()
_THINGS = ("Zephyr Quasar Lumen Vertex Onyx Cobalt Nimbus Falcon Aster Drift Mosaic "
           "Halcyon Pivot Ember Cinder Talon Verve Wisp Strata Vellum").split()
_DISTRACTORS = (
    "The archived note was reviewed but did not change any listed value.",
    "A separate memo mentioned an unrelated schedule update.",
    "The catalog order was shuffled before the final question was written.",
    "One observer copied the list into a temporary notebook.",
    "The summary below may ask about only part of the preceding list.",
)


def _rand_word(rng, n=5):
    c, v = "bcdfghklmnprstv", "aeiou"
    return "".join(rng.choice(c) + rng.choice(v) for _ in range(n // 2 + 1))[:n]


def _surface_forms():
    """Each form: (key_pool_fn, value_fn, statements, queries, answer_fn, name)."""
    return [
        (lambda r: r.choice(_NAMES), lambda r: r.randint(1000, 9999),
         ("The access code for {k} is {v}.", "{k}'s access code is {v}.",
          "For {k}, use access code {v}."),
         ("What are the access codes for {ks}? The codes are",
          "Report each requested access code for {ks}:",
          "For {ks}, list the matching access codes:"),
         lambda pairs: ", ".join(f"{k}: {v}" for k, v in pairs), "name_code"),
        (lambda r: r.choice(_THINGS), lambda r: r.randint(10, 999),
         ("The price of the {k} is {v} dollars.", "{k} costs {v} dollars.",
          "Catalog entry {k}: ${v}."),
         ("What are the prices of {ks}? The prices are",
          "Give the listed prices for {ks}:",
          "For {ks}, return the prices in order:"),
         lambda pairs: ", ".join(f"{k}: {v}" for k, v in pairs), "product_price"),
        (lambda r: "var_" + _rand_word(r, 4), lambda r: r.randint(100, 9999),
         ("{k} = {v}", "set {k} to {v}", "let {k} := {v}"),
         ("Print the values of {ks}. The values are",
          "Resolve these variables in order, {ks}:",
          "What values are bound to {ks}?"),
         lambda pairs: ", ".join(f"{k}={v}" for k, v in pairs), "var_value"),
        (lambda r: r.choice(_NAMES), lambda r: r.choice(_CITIES),
         ("{k} lives in {v}.", "{k}'s city is {v}.",
          "The residence for {k} is {v}."),
         ("Where do {ks} live? They live in",
          "List the cities for {ks}:",
          "Match each person to a city for {ks}:"),
         lambda pairs: ", ".join(f"{k}: {v}" for k, v in pairs), "person_city"),
        (lambda r: _rand_word(r, 6).capitalize(), lambda r: r.randint(1000, 99999),
         ("Record {k}: {v}.", "Identifier {k} maps to {v}.",
          "Ledger item {k} has id {v}."),
         ("List records {ks}. The records are",
          "Return the record ids for {ks}:",
          "For records {ks}, give the ids in order:"),
         lambda pairs: ", ".join(f"{k}: {v}" for k, v in pairs), "record_id"),
    ]


def synthetic_episode(rng, n_bindings, n_queries, distractor_rate=0.25):
    """Return episode text: bindings (shuffled) + a multi-key query + answer."""
    kfn, vfn, stmts, qheads, answer_fn, _ = rng.choice(_surface_forms())
    keys, vals, seen = [], [], set()
    tries = 0
    while len(keys) < n_bindings and tries < n_bindings * 20:
        k = kfn(rng); tries += 1
        if k in seen:
            continue
        seen.add(k); keys.append(k); vals.append(vfn(rng))
    order = list(range(len(keys)))
    rng.shuffle(order)
    lines = []
    for i in order:
        lines.append(rng.choice(stmts).format(k=keys[i], v=vals[i]))
        if rng.random() < distractor_rate:
            lines.append(rng.choice(_DISTRACTORS))
    qidx = rng.sample(range(len(keys)), min(n_queries, len(keys)))
    if rng.random() < 0.5:
        rng.shuffle(qidx)
    ks = ", ".join(keys[i] for i in qidx)
    answer = answer_fn([(keys[i], vals[i]) for i in qidx])
    body = " ".join(lines)
    return f"{body} {rng.choice(qheads).format(ks=ks)} {answer}."


def synthetic_doc(rng, target_chars=4000):
    """Concatenate episodes (varied difficulty) into one document."""
    out, n = [], 0
    while n < target_chars:
        nb = rng.choice([6, 8, 12, 16, 24])
        nq = rng.choice([2, 3, 4, 6, 8])
        ep = synthetic_episode(rng, nb, min(nq, nb))
        out.append(ep); n += len(ep)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 2-5. Natural source text iterators (streaming)
# --------------------------------------------------------------------------- #
def _stream_texts(source: str, rng):
    """Yield text strings from a named source (infinite where possible)."""
    from datasets import load_dataset
    if source == "synthetic":
        while True:
            yield synthetic_doc(rng)
    elif source == "musique":
        ds = load_dataset("dgslibisey/MuSiQue", split="train", streaming=True)
        for ex in ds:
            # join all 20 paragraphs -> one dense multi-referent doc + the question
            paras = " ".join(p["paragraph_text"] for p in ex["paragraphs"])
            yield f"{paras}\nQuestion: {ex['question']}\nAnswer: {ex.get('answer','')}"
    elif source == "wikipedia":
        ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train",
                          streaming=True)
        for ex in ds:
            yield ex["text"]
    elif source == "code":
        ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train",
                          streaming=True)
        for ex in ds:
            yield ex["content"]
    elif source == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        idx = list(range(len(ds)))
        while True:
            rng.shuffle(idx)
            for i in idx:
                if ds[i]["text"]:
                    yield ds[i]["text"]
    else:
        raise ValueError(f"unknown source {source}")


DEFAULT_WEIGHTS = {"synthetic": 0.20, "musique": 0.25, "code": 0.20,
                   "wikipedia": 0.20, "wikitext": 0.15}


def _validate_weights(weights):
    total = sum(weights.values())
    if not weights:
        raise ValueError("weights must not be empty")
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"weights must be non-negative: {weights}")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0, got {total:.6f}: {weights}")


def build_mix(tokenizer, out_dir, total_tokens=40_000_000, seq_len=2048,
              weights=None, seed=0):
    """Materialise a packed, shuffled token dataset to disk.

    Each source contributes ~weight*total_tokens tokens, tokenised and packed into
    seq_len blocks. Writes blocks.pt ([N, seq_len] int32) + manifest.json.
    """
    weights = weights or DEFAULT_WEIGHTS
    _validate_weights(weights)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    eos = tokenizer.eos_token_id or 0
    rng = random.Random(seed)
    all_blocks = []
    all_source_ids = []
    source_to_id = {source: i for i, source in enumerate(weights)}
    manifest = {
        "seq_len": seq_len,
        "weights": weights,
        "source_to_id": source_to_id,
        "per_source": {},
    }

    for source, w in weights.items():
        budget = int(total_tokens * w)
        buf, got = [], 0
        src_rng = random.Random(seed + sum(ord(c) for c in source))  # stable per-source
        for text in _stream_texts(source, src_rng):
            if not text:
                continue
            ids = tokenizer(text, add_special_tokens=False).input_ids
            buf.extend(ids); buf.append(eos); got += len(ids) + 1
            if got >= budget:
                break
        nblk = len(buf) // seq_len
        if nblk:
            t = torch.tensor(buf[:nblk * seq_len], dtype=torch.int32).view(nblk, seq_len)
            all_blocks.append(t)
            all_source_ids.append(torch.full((nblk,), source_to_id[source], dtype=torch.int16))
            manifest["per_source"][source] = {"tokens": int(nblk * seq_len), "blocks": nblk}
        print(f"  {source:>10}: {got:,} tokens -> {nblk} blocks", flush=True)

    data = torch.cat(all_blocks, dim=0)
    source_ids = torch.cat(all_source_ids, dim=0)
    perm = torch.randperm(data.shape[0], generator=torch.Generator().manual_seed(seed))
    data = data[perm]
    source_ids = source_ids[perm]
    torch.save(data, out_dir / "blocks.pt")
    torch.save(source_ids, out_dir / "source_ids.pt")
    manifest["total_blocks"] = int(data.shape[0])
    manifest["total_tokens"] = int(data.numel())
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=1)
    print(f"wrote {out_dir}/blocks.pt  [{data.shape[0]} x {seq_len}] "
          f"= {data.numel():,} tokens", flush=True)
    return manifest


class MaterializedMix(Dataset):
    """Loads a build_mix() output directory as a map-style packed dataset."""

    def __init__(self, path, return_source=False):
        p = Path(path)
        self.data = torch.load(p / "blocks.pt").long()
        self.manifest = json.load(open(p / "manifest.json"))
        self.seq_len = self.manifest["seq_len"]
        self.return_source = return_source
        source_path = p / "source_ids.pt"
        self.source_ids = torch.load(source_path).long() if source_path.exists() else None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        if self.return_source:
            if self.source_ids is None:
                raise RuntimeError("source_ids.pt is missing for this materialized mix")
            return self.data[i], self.source_ids[i]
        return self.data[i]
