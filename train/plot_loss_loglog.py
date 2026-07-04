"""Log-log loss/KL plots with a smoothed trendline and an offset power-law fit
  L(t) = L_inf + c * t^(-a)
to quantify the descent rate and the irreducible floor.

Usage: python plot_loss_loglog.py [--discord]
"""
import re, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
PAT = re.compile(r"step (\d+)/\d+ \| loss ([\d.]+) kl ([\d.]+) ce ([\d.]+) \| gnorm")


def parse(log):
    rows = []
    for ln in Path(log).read_text().splitlines():
        m = PAT.search(ln)
        if m:
            rows.append(tuple(float(x) for x in m.groups()))
    rows.sort()
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]   # step, loss, kl, ce


def smooth(y, w=9):
    k = np.ones(w) / w
    pad = w // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, k, mode="valid")[:len(y)]


def offset_powerlaw(step, y, fit_from=60):
    """Fit y = L_inf + c*t^(-a) on steps >= fit_from. Returns (params, curve, txt)."""
    mask = step >= fit_from
    t, yy = step[mask], y[mask]
    def f(t, Linf, c, a):
        return Linf + c * np.power(t, -a)
    p0 = [min(yy) * 0.9, yy[0], 0.3]
    try:
        popt, _ = curve_fit(f, t, yy, p0=p0, maxfev=20000,
                            bounds=([0, 0, 0.01], [max(yy), 1e6, 3.0]))
        Linf, c, a = popt
        curve = f(step, *popt)
        txt = f"L∞={Linf:.2f}, a={a:.2f}"
        return popt, curve, txt
    except Exception as e:
        return None, None, f"fit failed: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "runs/gdn3_full.log"))
    ap.add_argument("--out", default=str(ROOT / "runs/loss_loglog.png"))
    ap.add_argument("--discord", action="store_true")
    args = ap.parse_args()

    step, loss, kl, ce = parse(args.log)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, y, col) in zip(axes, [("total loss", loss, "tab:blue"),
                                          ("KL (teacher‖student)", kl, "tab:red")]):
        ax.loglog(step, y, color=col, alpha=0.28, lw=1.0, label="raw (per-10-step avg)")
        ax.loglog(step, smooth(y), color=col, lw=2.2, label="smoothed (window 9)")
        popt, curve, txt = offset_powerlaw(step, y)
        if curve is not None:
            ax.loglog(step, curve, color="black", ls="--", lw=1.6,
                      label=f"fit L∞+c·t⁻ᵃ  ({txt})")
            ax.axhline(popt[0], color="gray", ls=":", lw=1.0, alpha=0.7)
        # pure power-law reference (slope from a log-log lstsq over steps>=100)
        m = step >= 100
        sl, ic = np.polyfit(np.log(step[m]), np.log(y[m]), 1)
        ax.loglog(step, np.exp(ic) * step ** sl, color="orange", ls="-.", lw=1.1,
                  alpha=0.8, label=f"pure power-law (slope {sl:.2f})")
        ax.set_title(f"{name} — log-log", fontsize=12)
        ax.set_xlabel("step"); ax.set_ylabel(name)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("GDN3 distillation — log-log descent & offset power-law fit", y=1.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved -> {args.out}")

    # print interpretation
    for name, y in [("loss", loss), ("kl", kl)]:
        popt, _, txt = offset_powerlaw(step, y)
        if popt is not None:
            frac = (y[-1] - popt[0]) / y[-1]
            print(f"{name}: {txt} | final={y[-1]:.2f}, {frac*100:.0f}% of final is above the L∞ floor")

    if args.discord:
        try:
            import requests
            tokf = (ROOT / "BOT_TOKEN").read_text().strip(); ch = (ROOT / "CHANNEL_ID").read_text().strip()
            pl, _, ltxt = offset_powerlaw(step, loss)
            pk, _, ktxt = offset_powerlaw(step, kl)
            with open(args.out, "rb") as fh:
                r = requests.post(f"https://discord.com/api/v10/channels/{ch}/messages",
                    headers={"Authorization": f"Bot {tokf}"},
                    data={"content": f"📉 **Log-log loss/KL** — offset power-law fit: loss {ltxt}, KL {ktxt}."},
                    files={"file": ("loss_loglog.png", fh, "image/png")}, timeout=30)
            print("discord:", r.status_code)
        except Exception as e:
            print("discord:", e)


if __name__ == "__main__":
    main()
