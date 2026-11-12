"""plot_rank_rating.py — 名次–分数图 (用于确定强弱档固定名单)

只用 final 情境 (= 90% 开发集) 的 BT 评分排名; 横线是 6 个情境中的最低~最高分。
只画开发集出场 ≥ 300 × 开发集占比 次的模型 (评分区间太宽的不参与定档)。
用法: python -I plot_rank_rating.py <v2 根目录>   (需先跑 fold_ratings.py)
输出: reports/step4_rank_rating.png / .csv
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
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
    ax.text(1142, yy, f"断层 {f.loc[k - 1, 'gap_below']:.1f} 分", va="center", fontsize=8, color=ink if main else muted)
ax.text(1000, -5.5, "强档 (前 10)", color="#2a78d6", fontsize=10)
ax.text(1050, -46, "弱档 (后 11)", color="#eb6834", fontsize=10)
ax.text(1060, -25, "中间带 (不用)", color=muted, fontsize=10)
ax.set_xlim(875, 1175)
ax.set_ylim(-52, 0)
ax.set_xlabel("Bradley-Terry 评分 (只用 90% 开发集拟合); 横线 = 6 个情境中的最低~最高分", color=muted)
ax.set_title(f"名次–分数图: {len(f)} 个模型", loc="left", color=ink, fontsize=12)
ax.grid(axis="x", color=grid, lw=.6)
for s in ax.spines.values():
    s.set_color(grid)
ax.tick_params(colors=muted)
fig.tight_layout()
fig.savefig(root / "reports/step4_rank_rating.png", facecolor=fig.get_facecolor())
