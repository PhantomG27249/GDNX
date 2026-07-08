#!/usr/bin/env python3
"""RULER n=16 (needles): teacher vs KMD-2 core vs KMD-2 r_out=4.
Adapted from ~/MLA_test/plot_n32_seqlen.py conventions (Okabe-Ito, 95% CI bars,
direct value labels). One bar group per (ctx, queries) cell."""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_T, C_CORE, C_R4 = "#0072B2", "#009E73", "#E69F00"   # validated palette
INK = "#1a1a1a"
BASE = "/home/dev/gdn3_fable/research/runs_fable"

teacher = json.load(open(f"{BASE}/ruler_teacher.json"))
core = json.load(open(f"{BASE}/ruler_core.json"))
rout4 = json.load(open(f"{BASE}/ruler_rout4.json"))

cells = [(1024, 1), (1024, 4), (2048, 1), (2048, 4), (4096, 1), (4096, 4)]


def get(rows, ctx, q):
    for r in rows:
        if r["ctx"] == ctx and r["queries"] == q and r["needles"] == 16:
            return r["recall"], r["n"]
    return None, 0


def ci95(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n) if (p is not None and n) else 0.0


series = [("teacher (native GDN)", C_T, teacher),
          ("KMD-2 core", C_CORE, core),
          ("KMD-2 r_out=4", C_R4, rout4)]

x = np.arange(len(cells))
w = 0.26
fig, ax = plt.subplots(figsize=(10.5, 5.2))
for i, (lab, col, rows) in enumerate(series):
    vals = [get(rows, c, q) for c, q in cells]
    ps = [p if p is not None else 0 for p, _ in vals]
    errs = [ci95(p, n) for p, n in vals]
    off = (i - 1) * w
    ax.bar(x + off, ps, w, yerr=errs, capsize=3, label=lab, color=col, zorder=3)
    for xi, (p, _), e in zip(x, vals, errs):
        if p is not None:
            ax.text(xi + off, p + e + 0.012, f"{p:.2f}", ha="center",
                    fontsize=7.5, color=INK)

ax.set_xticks(x)
ax.set_xticklabels([f"{c} tok / {q}q" for c, q in cells])
ax.set_ylabel("RULER recall (teacher-forced exact value)", color=INK)
ax.set_ylim(0, 1.12)
ax.set_title("Multi-key RULER, 16 needles — teacher vs KMD-2 heals "
             "(plateau-stopped @1000 steps, loss ~3.1)\n"
             "error bars = 95% CI (binomial over queried values)",
             fontsize=10.5, color=INK)
ax.grid(axis="y", alpha=0.15)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, loc="upper right", fontsize=9)

tvals = [get(teacher, c, q)[0] for c, q in cells]
for lab, rows in (("core", core), ("r_out4", rout4)):
    sv = [get(rows, c, q)[0] for c, q in cells]
    rel = [s / t for s, t in zip(sv, tvals) if t]
    print(f"mean recall/teacher {lab}: {sum(rel)/len(rel):.3f}")

fig.tight_layout()
out = f"{BASE}/ruler_kmd2_n16.png"
fig.savefig(out, dpi=130)
print("saved", out)
