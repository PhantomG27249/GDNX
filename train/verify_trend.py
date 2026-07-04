"""Retrieval vs training-step trend: score S-NIAH / MK-4 recall for each GDN3
student checkpoint (250/500/750/final) against the fixed teacher reference.

If forced exact-match stays flat at 0, token-accuracy is the finer signal:
does the value-token accuracy creep up with training at all?
"""
import sys, json, random, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "integration"))
import torch
from transformers import AutoTokenizer
from integration.verify_ruler import eval_config, load_teacher, load_student, MODEL_SNAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--keys", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--dev", default="cuda:1")
    ap.add_argument("--tdev", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--discord", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_SNAP)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ckpt_dir = ROOT / "runs/gdn3_distill_full"
    stages = [("step250", 250), ("step500", 500), ("step750", 750), ("final", 1000)]

    def sweep(model, dev):
        out = {}
        for L in args.lengths:
            for K in args.keys:
                rng = random.Random(args.seed + K + L)
                out[f"L{L}_K{K}"] = eval_config(model, tok, dev, L, K, args.samples, rng, None)
        return out

    rows = {}
    print("== teacher reference ==")
    teach = load_teacher(args.tdev)
    rows["teacher"] = sweep(teach, args.tdev)
    del teach; torch.cuda.empty_cache()

    for name, step in stages:
        ck = ckpt_dir / name / "gdn3_layers.pt"
        if not ck.exists():
            print(f"skip {name} (missing)"); continue
        print(f"== student @ {name} ==")
        m = load_student(str(ck), args.dev)
        rows[name] = sweep(m, args.dev)
        del m; torch.cuda.empty_cache()

    # table
    cfgs = [f"L{L}_K{K}" for L in args.lengths for K in args.keys]
    print("\n===== RECALL (exact-match %) =====")
    print(f"{'stage':9s}" + "".join(f"{c:>10s}" for c in cfgs))
    for name in ["teacher"] + [s for s, _ in stages if s in rows]:
        print(f"{name:9s}" + "".join(f"{rows[name][c]['recall_em']*100:9.1f} " for c in cfgs))
    print("\n===== TOKEN-ACCURACY (%) — finer signal =====")
    print(f"{'stage':9s}" + "".join(f"{c:>10s}" for c in cfgs))
    for name in ["teacher"] + [s for s, _ in stages if s in rows]:
        print(f"{name:9s}" + "".join(f"{rows[name][c]['tok_acc']*100:9.1f} " for c in cfgs))

    json.dump(rows, open(ROOT / "runs/ruler_trend.json", "w"), indent=2)
    print("\nsaved -> runs/ruler_trend.json")

    if args.discord:
        try:
            import requests
            lines = ["**Retrieval vs training step** (exact-match % | token-acc %)"]
            for name in ["teacher"] + [s for s, _ in stages if s in rows]:
                em = ", ".join(f"{c}:{rows[name][c]['recall_em']*100:.0f}/{rows[name][c]['tok_acc']*100:.0f}" for c in cfgs)
                lines.append(f"{name}: {em}")
            tokf = (ROOT / "BOT_TOKEN").read_text().strip(); ch = (ROOT / "CHANNEL_ID").read_text().strip()
            requests.post(f"https://discord.com/api/v10/channels/{ch}/messages",
                          headers={"Authorization": f"Bot {tokf}"},
                          json={"content": "```" + "\n".join(lines) + "```"}, timeout=15)
        except Exception as e:
            print("discord:", e)


if __name__ == "__main__":
    main()
