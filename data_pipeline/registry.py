"""Source registry for the attention-rebuild data pipeline.

A *source* is a factory that, given an rng (+ optional tokenizer for sources that
need a chat template), returns an iterator of raw document strings. Sources are
registered by name so recipes can reference them declaratively. Packing/tokenising
is handled downstream in `pack.py` — a source only produces text.
"""
from __future__ import annotations

from typing import Callable, Iterator, Optional

# name -> factory(rng, tokenizer=None, **cfg) -> Iterator[str]
SOURCES: dict[str, Callable[..., Iterator[str]]] = {}


def source(name: str):
    def deco(fn):
        if name in SOURCES:
            raise ValueError(f"duplicate source {name!r}")
        SOURCES[name] = fn
        return fn
    return deco


def get_source(name: str) -> Callable[..., Iterator[str]]:
    if name not in SOURCES:
        raise KeyError(f"unknown source {name!r}; registered: {sorted(SOURCES)}")
    return SOURCES[name]
