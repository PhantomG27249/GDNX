#!/usr/bin/env python3
"""RULER-style multi-key/multi-query NIAH for the KMD-2-healed 0.8B.

Adapted from ~/MLA_test/ruler_multikey.py (same task generation + teacher-forced
scoring). Student loading swapped: GDN3_KMD2 env upgrade + gdn3_layers.pt
checkpoint instead of the MLA surgery. Teacher = native Qwen3.5-0.8B.
"""
from __future__ import annotations
import argparse, json, os, random, sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/home/dev/gdn3_fable")

import torch

SNAP = ("/home/dev/.cache/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/"
        "2fc06364715b967f1860aea9cf38778875588b17")

_FILLER = ("The grass is green and the sky is blue. People walk along the river in "
           "the afternoon sun, talking quietly about the weather and their plans. ")
_CONS = "bcdfghklmnprstv"
_VOW = "aeiou"


def _pseudoword(rng):
    return "".join(rng.choice(_CONS) + rng.choice(_VOW) for _ in range(3))


def make_keys_values(rng, k):
    keys, vals = set(), set()
    while len(keys) < k:
        keys.add(_pseudoword(rng))
    while len(vals) < k:
        vals.add(rng.randint(1_000_000, 9_999_999))
    return list(keys), list(vals)


def build_sample(tok, ctx_len, n_needles, n_queries, rng):
    keys, vals = make_keys_values(rng, n_needles)
    needles = [tok(f" One of the special magic numbers for {k} is: {v}.",
                   add_special_tokens=False).input_ids
               for k, v in zip(keys, vals)]
    filler = tok(_FILLER, add_special_tokens=False).input_ids
    reps = max(1, ctx_len // len(filler) + 1)
    hay = (filler * reps)[:ctx_len]
    positions = sorted(rng.sample(range(len(hay)), len(needles)))
    ctx, prev = [], 0
    for pos, needle in zip(positions, needles):
        ctx.extend(hay[prev:pos]); ctx.extend(needle); prev = pos
    ctx.extend(hay[prev:])
    qidx = rng.sample(range(n_needles), n_queries)
    qkeys = [keys[i] for i in qidx]; qvals = [vals[i] for i in qidx]
    if n_queries == 1:
        question = tok(f" What is the special magic number for {qkeys[0]}? "
                       f"The number is:", add_special_tokens=False).input_ids
    else:
        question = tok(f" What are the special magic numbers for {', '.join(qkeys)}? "
                       f"The numbers are:", add_special_tokens=False).input_ids
    input_ids = list(ctx + question)
    val_spans = []
    for i, v in enumerate(qvals):
        input_ids.extend(tok(" " if i == 0 else ", ", add_special_tokens=False).input_ids)
        vtok = tok(str(v), add_special_tokens=False).input_ids
        start = len(input_ids)
        val_spans.append((start, start + len(vtok), vtok))
        input_ids.extend(vtok)
    return input_ids, val_spans


@torch.no_grad()
def score(model, input_ids, val_spans, device):
    ids = torch.tensor([input_ids], device=device)
    h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    n_correct = 0
    for s, e, vtok in val_spans:
        pred = model.lm_head(h[s - 1:e - 1]).argmax(-1).tolist()
        n_correct += int(pred == vtok)
    return n_correct, len(val_spans)


@torch.no_grad()
def run(model, tok, lengths, settings, n_samples, device, label, seed_base=1234):
    print(f"\n=== {label} ===", flush=True)
    print(f"{'ctx':>8} {'needles':>8} {'queries':>8} {'recall':>8} {'exact':>8}")
    results = []
    for L in lengths:
        for (K, Q) in settings:
            rng = random.Random(seed_base + L + K * 100 + Q)
            got = tot = exact_ok = exact_tot = 0
            for _ in range(n_samples):
                ids, spans = build_sample(tok, L, K, Q, rng)
                c, t = score(model, ids, spans, device)
                got += c; tot += t; exact_ok += int(c == t); exact_tot += 1
            rec = got / max(1, tot)
            results.append({"label": label, "ctx": L, "needles": K, "queries": Q,
                            "recall": rec, "n": tot, "got": got,
                            "exact": exact_ok / max(1, exact_tot), "seed_base": seed_base})
            print(f"{L:>8} {K:>8} {Q:>8} {rec:>8.2f} {exact_ok/max(1,exact_tot):>8.2f}", flush=True)
    return results


def load_teacher(device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True)
    m.config.use_cache = False
    return m.to(device).eval()


def load_student(ckpt_dir, rout, device):
    os.environ["GDN3_KMD2"] = "1"
    os.environ["GDN3_KMD2_R"] = "1"
    os.environ["GDN3_KMD2_ROUT"] = str(rout)
    from transformers import AutoModelForCausalLM
    from gdn3.gdn3_upgrade import GDN3UpgradeManager
    m = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32,
                                             low_cpu_mem_usage=True)
    mgr = GDN3UpgradeManager(m); mgr.apply_upgrade()
    sd = torch.load(os.path.join(ckpt_dir, "gdn3_layers.pt"), map_location="cpu")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    loaded = len(sd)
    print(f"[student rout={rout}] loaded {loaded} tensors from {ckpt_dir}", flush=True)
    m.config.use_cache = False
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["teacher", "core", "rout4"])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--lengths", default="1024,2048,4096")
    ap.add_argument("--settings", default="16:1,16:4")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    settings = [tuple(int(y) for y in s.split(":")) for s in args.settings.split(",")]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP)
    if args.which == "teacher":
        model = load_teacher(args.device)
    elif args.which == "core":
        model = load_student("/home/dev/gdn3_fable/runs/kmd2_heal_core/final", 1, args.device)
    else:
        model = load_student("/home/dev/gdn3_fable/runs/kmd2_heal_rout4/final", 4, args.device)
    res = run(model, tok, lengths, settings, args.n_samples, args.device, args.which)
    json.dump(res, open(args.json_out, "w"), indent=1)
    print("wrote", args.json_out)


if __name__ == "__main__":
    main()
