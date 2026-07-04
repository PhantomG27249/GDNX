"""Offline synthetic generators that *stress* rebuilt attention.

These force behaviours broad web text under-specifies: parallel multi-referent
retrieval, long-distance exact copy, associative recall, induction. Fully offline,
highly randomised (vocab + layout) so the student learns dependency structure, not
formatting. Ported and extended from `mla_retrofit/data_mix.py`.

Each generator is registered as a source yielding document strings. `target_chars`
scales a document toward a length band (short packs many short episodes; the 32k/65k
bands ask for large target_chars so a single doc spans the whole window with genuine
long-range dependencies rather than packed-independent shorts).
"""
from __future__ import annotations

import random
from typing import Iterator, Optional

from .registry import source

_NAMES = ("Alice Bob Carol Dave Erin Frank Grace Heidi Ivan Judy Karl Lena Mallory "
          "Niaj Olivia Peggy Quinn Rupert Sybil Trent Uma Victor Wendy Xena Yuri Zoe "
          "Amir Bianca Cyrus Dalia Esme Faisal Greta Hugo Inez Jamal").split()
_CITIES = ("Lisbon Cairo Oslo Lima Dhaka Quito Riga Tunis Accra Hanoi Sofia Minsk "
           "Kyoto Perth Davao Cusco Bonn Nantes Leeds Ghent Tartu Pisa Cork Bergen").split()
_THINGS = ("Zephyr Quasar Lumen Vertex Onyx Cobalt Nimbus Falcon Aster Drift Mosaic "
           "Halcyon Pivot Ember Cinder Talon Verve Wisp Strata Vellum").split()


def _rand_word(rng, n=5):
    c, v = "bcdfghklmnprstv", "aeiou"
    return "".join(rng.choice(c) + rng.choice(v) for _ in range(n // 2 + 1))[:n]


def _surface_forms():
    """(key_fn, value_fn, stmt, q_head, name) — varied binding surface forms."""
    return [
        (lambda r: r.choice(_NAMES), lambda r: r.randint(1000, 9999),
         "The access code for {k} is {v}.",
         "What are the access codes for {ks}? The codes are", "name_code"),
        (lambda r: r.choice(_THINGS), lambda r: r.randint(10, 999),
         "The price of the {k} is {v} dollars.",
         "What are the prices of the {ks}? The prices are", "product_price"),
        (lambda r: "var_" + _rand_word(r, 4), lambda r: r.randint(100, 9999),
         "{k} = {v}",
         "Print the values of {ks}. The values are", "var_value"),
        (lambda r: r.choice(_NAMES), lambda r: r.choice(_CITIES),
         "{k} lives in {v}.",
         "Where do {ks} live? They live in", "person_city"),
        (lambda r: _rand_word(r, 6).capitalize(), lambda r: r.randint(1000, 99999),
         "Record {k}: {v}.",
         "List records {ks}. The records are", "record_id"),
    ]


def _multiquery_episode(rng, n_bindings, n_queries):
    kfn, vfn, stmt, qhead, _ = rng.choice(_surface_forms())
    keys, vals, seen = [], [], set()
    tries = 0
    while len(keys) < n_bindings and tries < n_bindings * 20:
        k = kfn(rng); tries += 1
        if k in seen:
            continue
        seen.add(k); keys.append(k); vals.append(vfn(rng))
    order = list(range(len(keys)))
    rng.shuffle(order)
    lines = [stmt.format(k=keys[i], v=vals[i]) for i in order]
    qidx = rng.sample(range(len(keys)), min(n_queries, len(keys)))
    ks = ", ".join(keys[i] for i in qidx)
    answer = ", ".join(str(vals[i]) for i in qidx)
    return f"{' '.join(lines)} {qhead.format(ks=ks)} {answer}."


@source("synth_multiquery")
def synth_multiquery(rng, tokenizer=None, target_chars=4000, min_bindings=6,
                     max_bindings=40, **_) -> Iterator[str]:
    """Parallel multi-referent retrieval: many k->v bindings, multi-key query.

    For long bands raise target_chars *and* max_bindings so the needles are spread
    across the whole window (genuine long-distance parallel retrieval)."""
    while True:
        out, n = [], 0
        while n < target_chars:
            nb = rng.randint(min_bindings, max_bindings)
            nq = rng.choice([2, 3, 4, 6, 8, 12])
            ep = _multiquery_episode(rng, nb, min(nq, nb))
            out.append(ep); n += len(ep)
        yield "\n".join(out)


@source("synth_copy")
def synth_copy(rng, tokenizer=None, target_chars=4000, **_) -> Iterator[str]:
    """Long-distance exact copy: a random token block, distractor gap, then recall.

    Directly probes copy fidelity at range — the RAFT-style signal that guarantees
    the copy behaviour lands at long positions when target_chars fills the band."""
    while True:
        blk = [_rand_word(rng, rng.randint(3, 7)) for _ in range(rng.randint(8, 32))]
        payload = " ".join(blk)
        # distractor filler sized to push the recall far from the payload
        gap_words = max(0, (target_chars - 2 * len(payload)) // 6)
        gap = " ".join(_rand_word(rng, rng.randint(3, 7)) for _ in range(gap_words))
        yield (f"Remember this sequence: {payload}\n{gap}\n"
               f"Repeat the sequence exactly: {payload}")


@source("synth_needle")
def synth_needle(rng, tokenizer=None, target_chars=4000, n_needles=8, **_) -> Iterator[str]:
    """Multi-needle retrieval: N labelled facts hidden in filler, all queried."""
    while True:
        needles = [(f"{_rand_word(rng,5)}-{rng.randint(10,99)}", rng.randint(1000, 9999))
                   for _ in range(n_needles)]
        chunks, n = [], 0
        filler = lambda: " ".join(_rand_word(rng, rng.randint(3, 7))
                                  for _ in range(rng.randint(20, 60)))
        # interleave needles among filler
        seq = list(needles)
        rng.shuffle(seq)
        for key, val in seq:
            chunks.append(filler())
            chunks.append(f"The magic number for {key} is {val}.")
        while sum(len(c) for c in chunks) < target_chars:
            chunks.insert(rng.randint(0, len(chunks)), filler())
        q = ", ".join(k for k, _ in needles)
        a = ", ".join(str(v) for _, v in needles)
        yield f"{' '.join(chunks)}\nWhat are the magic numbers for {q}? {a}."
