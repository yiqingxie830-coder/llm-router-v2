"""plot_rank_rating.py — rank–rating plot (used to fix the strong / weak tier lists)

Ranks by the BT rating of the final context only (= the 90% dev set); the horizontal bars span the lowest to
highest rating across the 6 contexts. Only models with ≥ 300 × (dev share) dev appearances are drawn
(models whose rating range is too wide do not take part in tiering).
Usage: python -I plot_rank_rating.py <repo root>   (run fold_ratings.py first)
Writes: reports/step4_rank_rating.png / .csv
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1])
r = pd.read_csv(root / "reports/step4_ratings_by_context.csv")
N_ALL, N_DEV = 101058, 90923
f = r[r.context == "final"]
f = f[f.n_games >= 300 * N_DEV / N_ALL].sort_values("rating", ascending=False).reset_index(drop=True)
rng = r.pivot(index="model", columns="context", values="rating")
f["lo"], f["hi"] = f.model.map(rng.min(axis=1)), f.model.map(rng.max(axis=1))
f["gap_below"] = f.rating - f.rating.shift(-1)
f["rank"] = f.index + 1
f.round(1).to_csv(root / "reports/step4_rank_rating.csv", index=False)

ink, muted, grid = "#2b2b29", "#6b6a63", "#e1e0d9"
col = ["#2a78d6" if k <= 10 else ("#eb6834" if k >= 41 else "#a8a79f") for k in f["rank"]]
fig, ax = plt.subplots(figsize=(9, 13), dpi=150, facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
y = -f["rank"]
ax.hlines(y, f.lo, f.hi, color=col, lw=2, alpha=.6)
ax.scatter(f.rating, y, s=40, color=col, zorder=3, edgecolor="#fcfcfb", linewidth=1.5)
ax.set_yticks(y)
ax.set_yticklabels([f"{k:>2}  {m}" for k, m in zip(f["rank"], f.model)], fontsize=7.5, color=ink)
for k in [1, 2, 10, 40, 45]:
    yy = -(k + 0.5)
    main = k in (10, 40)
    ax.axhline(yy, color=ink if main else grid, lw=1.2 if main else .8, ls="--")
    ax.text(1142, yy - 0.45, f"gap {f.loc[k - 1, 'gap_below']:.1f} pts", va="top", fontsize=8, color=ink if main else muted)
ax.text(1000, -5.5, "Strong tier (top 10)", color="#2a78d6", fontsize=10)
ax.text(1050, -46, "Weak tier (bottom 11)", color="#eb6834", fontsize=10)
ax.text(1060, -25, "Middle (unused)", color=muted, fontsize=10)
ax.set_xlim(875, 1175)
ax.set_ylim(-52, 0)
ax.set_xlabel("Bradley-Terry rating (90% dev set)\nbar = lowest to highest rating across the 6 contexts", color=muted)
ax.set_title(f"Rank vs rating: {len(f)} models", loc="left", color=ink, fontsize=12)
ax.grid(axis="x", color=grid, lw=.6)
for s in ax.spines.values():
    s.set_color(grid)
ax.tick_params(colors=muted)
fig.tight_layout()
fig.savefig(root / "reports/step4_rank_rating.png", facecolor=fig.get_facecolor())
