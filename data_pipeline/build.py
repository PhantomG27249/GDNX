#!/usr/bin/env python3
"""Materialise a recipe into packed, provenance-tracked training blocks.

A recipe (YAML) declares length bands, each with a target seq_len, a token budget,
an optional `min_doc_tokens` floor (genuine-long bands set this so short docs can't
fill a long window), and a weighted list of sources. Per band we write:
  blocks_<band>.pt   train blocks [N, seq_len] int32
  heldout_<band>.pt  a small per-category held-out slice (if heldout_blocks > 0)
plus manifest.json with full provenance (per-source docs/tokens/blocks/role/category)
and a per-band stress/filler token audit.

Usage:
  python -m data_pipeline.build --recipe data_pipeline/recipes/mla_27b.yaml --out data/attn_v2
  python -m data_pipeline.build --recipe ... --out ... --scale 0.01   # smoke build
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml

# ensure source registrations run
from . import sources, synth  # noqa: F401
from .pack import build_band


def load_recipe(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B",
                    help="tokenizer source (same across the Qwen3.5/3.6 family)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every band's token budget (e.g. 0.01 for a smoke build)")
    ap.add_argument("--only-band", default=None, help="build just this band")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    recipe = load_recipe(args.recipe)
    model = recipe.get("model", args.model)
    tok = AutoTokenizer.from_pretrained(model)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = {"recipe": args.recipe, "model": model, "scale": args.scale,
                "seed": args.seed, "bands": {}}

    print(f"building recipe {args.recipe} -> {out}  (scale={args.scale})", flush=True)
    for band_name, band in recipe["bands"].items():
        if args.only_band and band_name != args.only_band:
            continue
        seq_len = int(band["seq_len"])
        band_tokens = int(band["tokens"] * args.scale)
        min_doc = int(band.get("min_doc_tokens", 0))
        heldout = int(band.get("heldout_blocks", 0))
        print(f"\n== band {band_name}: seq_len={seq_len} budget={band_tokens:,} tok"
              f" min_doc_tokens={min_doc} heldout={heldout}", flush=True)
        data, held, held_cats, stats = build_band(
            band_name, band["sources"], tok, band_tokens, seq_len, args.seed,
            min_doc_tokens=min_doc, heldout_blocks=heldout)
        if data is None:
            print(f"   !! band {band_name} produced no blocks", flush=True)
            continue
        torch.save(data, out / f"blocks_{band_name}.pt")

        role_tokens = defaultdict(int)
        for st in stats:
            role_tokens[st["role"]] += st["tokens"]
        entry = {"seq_len": seq_len, "blocks": int(data.shape[0]),
                 "tokens": int(data.numel()), "min_doc_tokens": min_doc,
                 "role_tokens": dict(role_tokens), "sources": stats}
        if held is not None and held.shape[0]:
            torch.save(held, out / f"heldout_{band_name}.pt")
            # per-block category labels (aligned with held rows) for per-category eval
            json.dump(held_cats, open(out / f"heldout_{band_name}_cats.json", "w"))
            hc = defaultdict(int)
            for c in held_cats:
                hc[c] += 1
            entry["heldout"] = {"blocks": int(held.shape[0]), "by_category": dict(hc)}
        manifest["bands"][band_name] = entry

        stress = role_tokens.get("stress", 0); total = sum(role_tokens.values()) or 1
        print(f"   wrote blocks_{band_name}.pt  [{data.shape[0]} x {seq_len}]"
              f"  genuine-long={stress/total:.0%}"
              + (f"  +heldout {held.shape[0]}" if held is not None else ""), flush=True)

    manifest["total_tokens"] = sum(b["tokens"] for b in manifest["bands"].values())
    manifest["total_blocks"] = sum(b["blocks"] for b in manifest["bands"].values())
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    print(f"\ndone: {manifest['total_tokens']:,} tokens across "
          f"{len(manifest['bands'])} band(s) -> {out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
