"""
RULER-style retrieval verification for GDN3-Qwen3.5 vs teacher Qwen3.5.
=====================================================================

Tasks (needle-in-a-haystack):
  * S-NIAH   : 1 key->value needle, query it.
  * MK-NIAH-K: K key->value needles in ONE haystack, query each (interference).
               recall = fraction of the K values retrieved correctly.

Scoring is FORCED (teacher-forced), never free generation — the base model is
small and fails open-ended decoding.  For each probe we append the gold value
and check the model's argmax at each answer position:
  * forced_em  : all answer tokens argmax-correct  (primary "recall")
  * tok_acc    : fraction of answer tokens argmax-correct (partial credit)

Needles are placed at randomized depths; recall is averaged over needles/samples.

Usage:
  python verify_ruler.py --lengths 512 1024 2048 4096 --samples 8 \
      --student-ckpt runs/gdn3_distill_full/final/gdn3_layers.pt --discord
"""
from __future__ import annotations
import sys, os, json, time, argparse, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F

MODEL_SNAP = "/home/dev/.cache/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"

FILLER = [
    "The harbor town woke slowly under a pale and even light.",
    "Merchants arranged their stalls while gulls circled the empty pier.",
    "A quiet tram rolled past shuttered cafes on the northern avenue.",
    "Somewhere a clock tower marked the hour without any hurry.",
    "Rain had washed the cobblestones clean during the long night.",
    "Two clerks discussed the ledger totals in low, unremarkable tones.",
    "The bakery released a warm smell that drifted down the lane.",
    "Fishing boats returned with their nets folded and their decks bare.",
    "An old librarian catalogued donations no one had come to read.",
    "The afternoon settled into the steady rhythm of an ordinary day.",
]
NEEDLE_TMPL = "One of the special magic numbers for {key} is: {val}."
QUERY_TMPL = ("\nQuestion: What is the special magic number for {key}?"
              "\nAnswer: The special magic number for {key} is:")

ADJ = "amber brisk cobalt dusky ember frost glim hazel ivory jade "\
      "lunar mossy nimble ochre pewter quartz russet slate umber vivid".split()
NOUN = "lantern harbor cipher meadow falcon quartz anchor beacon cedar willow "\
       "marble canyon ember thicket harbor pylon zephyr cobble trellis onyx".split()


def rand_key(rng):
    return f"{rng.choice(ADJ)}-{rng.choice(NOUN)}"


def rand_val(rng):
    return str(rng.randint(1_000_000, 9_999_999))


# --------------------------------------------------------------------------- #
def build_haystack(tok, ctx_len, n_keys, rng):
    """Return (context_ids[list], needles[list of (key,val)]).

    n_keys distinct key->val needles spliced at random depths in [5%,85%]
    into filler text trimmed to exactly ctx_len tokens.
    """
    keys, vals, seen = [], [], set()
    while len(keys) < n_keys:
        k = rand_key(rng)
        if k in seen:
            continue
        seen.add(k); keys.append(k); vals.append(rand_val(rng))

    # filler tokens ~ctx_len (each FILLER sentence ~10 tokens; +20 slack)
    n_sent = ctx_len // 6 + 20
    filler_txt = " ".join(rng.choice(FILLER) for _ in range(n_sent))
    ftok = tok(filler_txt, add_special_tokens=False).input_ids[:ctx_len]

    needle_tok = [tok(NEEDLE_TMPL.format(key=k, val=v) + " ",
                      add_special_tokens=False).input_ids
                  for k, v in zip(keys, vals)]
    # distinct insertion depths in [0.05, 0.85]
    depths = sorted(rng.uniform(0.05, 0.85) for _ in range(n_keys))
    ctx = list(ftok)
    # insert from the back so earlier positions stay valid
    for d, nt in sorted(zip(depths, needle_tok), key=lambda x: -x[0]):
        pos = int(d * len(ftok))
        ctx[pos:pos] = nt
    ctx = ctx[:ctx_len]
    return ctx, list(zip(keys, vals))


@torch.no_grad()
def eval_config(model, tok, dev, ctx_len, n_keys, n_samples, rng, dtype):
    """Return dict of aggregate metrics over n_samples for one (len, n_keys)."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    em_hits = tok_correct = tok_total = probes = 0
    for _ in range(n_samples):
        ctx, needles = build_haystack(tok, ctx_len, n_keys, rng)
        # one probe per needle; batch them (shared-length via right pad)
        seqs, spans, golds = [], [], []
        for key, val in needles:
            q = tok(QUERY_TMPL.format(key=key), add_special_tokens=False).input_ids
            a = tok(f" {val}", add_special_tokens=False).input_ids
            prefix = ctx + q
            seqs.append(prefix + a)
            spans.append((len(prefix), len(a)))       # answer starts at len(prefix)
            golds.append(a)
        maxlen = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        am = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s); am[i, :len(s)] = 1
        ids = ids.to(dev); am = am.to(dev)
        # Run the backbone once (cheap hidden states), then apply lm_head ONLY at
        # answer positions — avoids materializing full-vocab logits over all T.
        hidden = model.model(input_ids=ids, attention_mask=am).last_hidden_state
        for i, ((astart, alen), gold) in enumerate(zip(spans, golds)):
            h = hidden[i, astart - 1: astart - 1 + alen]      # [alen, hidden]
            pred = model.lm_head(h).float().argmax(-1)         # [alen]
            g = torch.tensor(gold, device=dev)
            nc = int((pred == g).sum()); em = int(nc == alen)
            em_hits += em; tok_correct += nc; tok_total += alen; probes += 1
    return {
        "recall_em": em_hits / max(1, probes),
        "tok_acc": tok_correct / max(1, tok_total),
        "probes": probes,
    }


# --------------------------------------------------------------------------- #
def load_teacher(dev):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL_SNAP, torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True)
    m.config.use_cache = False
    return m.eval().to(dev)


def load_student(ckpt, dev):
    from transformers import AutoModelForCausalLM
    from gdn3.gdn3_upgrade import GDN3UpgradeManager
    m = AutoModelForCausalLM.from_pretrained(MODEL_SNAP, torch_dtype=torch.float32,
                                             low_cpu_mem_usage=True)
    GDN3UpgradeManager(m).apply_upgrade()
    if ckpt and Path(ckpt).exists():
        sd = torch.load(ckpt, map_location="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(unexpected)
        print(f"[student] loaded {loaded}/{len(sd)} GDN3 tensors from {ckpt} "
              f"(unexpected={len(unexpected)})")
        assert len(unexpected) == 0, f"unexpected keys: {unexpected[:5]}"
    else:
        print(f"[student] WARNING: no checkpoint at {ckpt}; using warm-start init")
    m.config.use_cache = False
    return m.eval().to(dev)


def discord(msg):
    try:
        import requests
        tok = (ROOT / "BOT_TOKEN").read_text().strip()
        ch = (ROOT / "CHANNEL_ID").read_text().strip()
        requests.post(f"https://discord.com/api/v10/channels/{ch}/messages",
                      headers={"Authorization": f"Bot {tok}"},
                      json={"content": msg[:1990]}, timeout=10)
    except Exception as e:
        print(f"[discord] {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--keys", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--student-ckpt", default=str(ROOT / "runs/gdn3_twotimescale_heal/final/gdn3_layers.pt"))
    ap.add_argument("--student-dev", default="cuda:1")
    ap.add_argument("--teacher-dev", default="cuda:0")
    ap.add_argument("--out", default=str(ROOT / "runs/ruler_results.json"))
    ap.add_argument("--stop-recall", type=float, default=0.02,
                    help="stop extending lengths once BOTH models fall below this on max-K")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--discord", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_SNAP)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    models = {}
    print("loading teacher..."); models["teacher"] = load_teacher(args.teacher_dev)
    print("loading student..."); models["student"] = load_student(args.student_ckpt, args.student_dev)
    devs = {"teacher": args.teacher_dev, "student": args.student_dev}

    task_name = {1: "S-NIAH", 4: "MK-NIAH-4", 8: "MK-NIAH-8"}
    results = {m: {} for m in models}
    if args.discord:
        discord(f"🔎 **RULER verification** — lengths {args.lengths}, "
                f"K={args.keys}, {args.samples} samples/config. student="
                f"`{Path(args.student_ckpt).parent.name}`")

    for L in args.lengths:
        line = [f"\n=== ctx_len {L} ==="]
        maxk_recalls = []
        for name, model in models.items():
            for K in args.keys:
                rng = random.Random(args.seed + K + L)  # same samples across models
                t0 = time.time()
                r = eval_config(model, tok, devs[name], L, K, args.samples, rng, None)
                dt = time.time() - t0
                results[name][f"L{L}_K{K}"] = r
                line.append(f"  {name:7s} {task_name[K]:10s}: recall_em={r['recall_em']*100:5.1f}%  "
                            f"tok_acc={r['tok_acc']*100:5.1f}%  ({r['probes']} probes, {dt:.0f}s)")
                if K == max(args.keys):
                    maxk_recalls.append(r["recall_em"])
        msg = "\n".join(line)
        print(msg, flush=True)
        if args.discord:
            discord("```" + msg + "```")
        json.dump(results, open(args.out, "w"), indent=2)
        if maxk_recalls and max(maxk_recalls) < args.stop_recall:
            print(f"[stop] both models < {args.stop_recall:.0%} recall on K={max(args.keys)} "
                  f"at L={L}; halting length sweep.")
            break

    # summary table
    print("\n================ SUMMARY (recall_em %) ================")
    hdr = f"{'config':14s}" + "".join(f"{m:>12s}" for m in models)
    print(hdr)
    for L in args.lengths:
        for K in args.keys:
            key = f"L{L}_K{K}"
            if key not in results["teacher"]:
                continue
            row = f"{'L%d ' % L + task_name[K]:14s}"
            for m in models:
                row += f"{results[m][key]['recall_em']*100:11.1f} "
            print(row)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")
    if args.discord:
        discord(f"✅ RULER verification complete → {Path(args.out).name}")


if __name__ == "__main__":
    main()
