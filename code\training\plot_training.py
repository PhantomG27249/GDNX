"""Parse the GDN3 distillation log and plot loss/kl/ce/gnorm over steps.

Optionally uploads the figure to the research Discord channel.
Usage: python plot_training.py [--log runs/gdn3_full.log] [--discord]
"""
import re, argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PAT = re.compile(
    r"step (\d+)/\d+ \| loss ([\d.]+) kl ([\d.]+) ce ([\d.]+) \| gnorm ([\d.]+) \| lr_mem ([\d.eE+-]+)")


def parse(log):
    rows = []
    for line in Path(log).read_text().splitlines():
        m = PAT.search(line)
        if m:
            s, loss, kl, ce, gn, lr = m.groups()
            rows.append((int(s), float(loss), float(kl), float(ce), float(gn), float(lr)))
    rows.sort()
    return list(zip(*rows)) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "runs/gdn3_full.log"))
    ap.add_argument("--out", default=str(ROOT / "runs/training_curves.png"))
    ap.add_argument("--discord", action="store_true")
    args = ap.parse_args()

    data = parse(args.log)
    if not data:
        print("no step lines found"); return
    step, loss, kl, ce, gn, lr = data
    print(f"parsed {len(step)} points (steps {step[0]}..{step[-1]})")
    print(f"final: loss={loss[-1]:.3f} kl={kl[-1]:.3f} ce={ce[-1]:.3f} gnorm={gn[-1]:.2f}")
    print(f"min:   loss={min(loss):.3f} kl={min(kl):.3f} ce={min(ce):.3f}")

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    panels = [("total loss", loss, "tab:blue"), ("KL (teacher‖student)", kl, "tab:red"),
              ("CE (next-token)", ce, "tab:green"), ("grad-norm (pre-clip)", gn, "tab:purple")]
    for a, (title, y, c) in zip(ax.flat, panels):
        a.plot(step, y, color=c, lw=1.4)
        a.set_title(title, fontsize=12)
        a.set_xlabel("step"); a.grid(alpha=0.3)
        if title.startswith("grad"):
            a.set_yscale("log")
    fig.suptitle("GDN3 ← Qwen3.5 self-distillation (1000 steps, seq 512, mix_5k_copy)",
                 fontsize=13, y=0.995)
    # LR overlay on KL panel (secondary axis)
    ax2 = ax[0, 1].twinx()
    ax2.plot(step, lr, color="gray", ls="--", lw=0.9, alpha=0.7)
    ax2.set_ylabel("lr_mem", color="gray"); ax2.tick_params(labelcolor="gray")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved -> {args.out}")

    if args.discord:
        try:
            import requests
            tokf = (ROOT / "BOT_TOKEN").read_text().strip()
            ch = (ROOT / "CHANNEL_ID").read_text().strip()
            with open(args.out, "rb") as fh:
                r = requests.post(
                    f"https://discord.com/api/v10/channels/{ch}/messages",
                    headers={"Authorization": f"Bot {tokf}"},
                    data={"content": (f"📈 **Training curves** — final KL {kl[-1]:.2f}, "
                                      f"CE {ce[-1]:.2f}, loss {loss[-1]:.2f} "
                                      f"(min KL {min(kl):.2f}).")},
                    files={"file": ("training_curves.png", fh, "image/png")}, timeout=30)
            print("discord upload:", r.status_code)
        except Exception as e:
            print("discord upload failed:", e)


if __name__ == "__main__":
    main()
