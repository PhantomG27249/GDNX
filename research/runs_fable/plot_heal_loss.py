#!/usr/bin/env python3
"""Log-log training loss for the two KMD-2 heal runs (core vs r_out=4).

Question: did the 1%-per-100-step plateau stop fire prematurely? On log-log
axes a power-law descent is a straight line; per-window %-improvement of a
power law shrinks with step even while training is still genuinely working —
exactly the failure mode a fixed-% stop has in a noisy regime. We overlay a
power-law fit from steps 200-1000 and extrapolate to the 8000-step budget to
show what the stop plausibly left on the table.
"""
import re
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_CORE, C_ROUT4 = "#0072B2", "#E69F00"   # Okabe-Ito, validated
INK, MUT = "#1a1a1a", "#666666"

RUNS = [("core (r_out=1)", "/home/dev/gdn3_fable/runs/heal_core.log", C_CORE),
        ("r_out=4", "/home/dev/gdn3_fable/runs/heal_rout4.log", C_ROUT4)]

fig, ax = plt.subplots(figsize=(8.5, 5.2))
for label, path, col in RUNS:
    steps, losses = [], []
    for line in open(path):
        m = re.match(r"step (\d+)/8000 \| loss ([\d.]+)", line)
        if m:
            steps.append(int(m.group(1))); losses.append(float(m.group(2)))
    s = np.array(steps, float); l = np.array(losses, float)
    ax.plot(s, l, color=col, lw=2, label=label, zorder=3)
    # power-law fit on steps 200..1000 (post-warmup), extrapolated to 8000
    m_ = (s >= 200) & (s <= 1000)
    b, a = np.polyfit(np.log(s[m_]), np.log(l[m_]), 1)
    xs = np.geomspace(200, 8000, 60)
    ax.plot(xs, np.exp(a) * xs ** b, color=col, lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax.annotate(f"{label}\nfit slope {b:.2f} → {math.exp(a) * 8000 ** b:.2f} @8k",
                xy=(s[-1], l[-1]), xytext=(6, 8), textcoords="offset points",
                fontsize=8.5, color=col)

ax.axvline(1000, color=MUT, lw=1, ls=":", zorder=1)
ax.text(1000, ax.get_ylim()[0] * 1.05, " plateau stop @1000", fontsize=8.5,
        color=MUT, va="bottom")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("step (log)", color=INK)
ax.set_ylabel("distill loss, 10-step mean (log)", color=INK)
ax.set_title("KMD-2 heal: loss vs steps (log-log) — was the 1% plateau stop premature?\n"
             "dashed = power-law fit on steps 200–1000, extrapolated to the 8000-step budget",
             fontsize=10.5, color=INK)
ax.grid(True, which="both", alpha=0.15)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="lower left")
fig.tight_layout()
out = "/home/dev/gdn3_fable/research/runs_fable/heal_loss_loglog.png"
fig.savefig(out, dpi=130)
print("saved", out)
