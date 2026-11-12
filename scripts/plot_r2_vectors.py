"""plot_r2_vectors.py — 2-D PCA projection of R2's whitened model vectors (labelled with model names, coloured by the fixed strong / middle / weak tier lists)

Usage: python -I plot_r2_vectors.py <repo root>
Writes: reports/r2_model_vectors_pca.png, reports/r2_model_vectors_pca.csv (PC1/PC2 coordinates per model)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

root = Path(sys.argv[1])
src = (Path(__file__).parent / "fold_ratings.py").read_text()
ns = {}
for line in src.splitlines():
    if line.startswith(("STRONG_MODELS", "WEAK_MODELS")):
        exec(line, ns)
MIN_VOTES = 300

d = pd.read_csv(root / "data/r2_model_vectors_whitened.csv")
d = d[d.n_votes_dev >= MIN_VOTES].reset_index(drop=True).copy()
X = d.filter(regex=r"^w\d+$").values
X = X - X.mean(0)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
pc = X @ Vt[:2].T
ratio = S ** 2 / (S ** 2).sum()
d["tier"] = d.model.map(lambda m: "strong" if m in ns["STRONG_MODELS"] else "weak" if m in ns["WEAK_MODELS"] else "middle")
d["PC1"], d["PC2"] = pc[:, 0], pc[:, 1]
d[["model", "tier", "n_votes_dev", "PC1", "PC2", "bias_u"]].to_csv(root / "reports/r2_model_vectors_pca.csv", index=False)

plt.rcParams["font.family"] = ["DejaVu Sans"]
EN = {"strong": "Strong tier", "middle": "Middle (unused)", "weak": "Weak tier"}
fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
style = {"strong": ("#2a78d6", "o"), "middle": ("#9a9a94", "o"), "weak": ("#eb6834", "s")}
for t, (c, mk) in style.items():
    s = d[d.tier == t]
    ax.scatter(s.PC1, s.PC2, s=55, c=c, marker=mk, edgecolors="white", linewidths=1.5, zorder=3,
               label=f"{EN[t]} ({len(s)})")
texts = [ax.text(r.PC1, r.PC2, r.model, fontsize=7.5, color="#2b2b28" if r.tier != "middle" else "#6b6b66")
         for r in d.itertuples()]
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#b5b5ae", lw=0.6),
            expand=(1.35, 1.8), force_text=(0.8, 1.2), force_static=(0.4, 0.6), iter_lim=2000)
ax.axhline(0, color="#e2e1dc", lw=0.8, zorder=1)
ax.axvline(0, color="#e2e1dc", lw=0.8, zorder=1)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color("#c9c8c1")
ax.tick_params(colors="#6b6b66", labelsize=8)
ax.set_xlabel(f"PC1 ({ratio[0]:.0%} of variance)", color="#3d3d3a")
ax.set_ylabel(f"PC2 ({ratio[1]:.0%} of variance)", color="#3d3d3a")
ax.set_aspect("equal", adjustable="datalim")
ax.set_ylim(d.PC2.min() - 0.03, d.PC2.max() + 0.03)
ax.set_title("R2 matrix factorization: whitened model vectors (which prompts each model is suited to)", loc="left", fontsize=14, color="#1f1f1d", pad=36)
ax.text(0, 1.012, f"Trained on 90,923 dev votes (d = 128); {len(d)} models with ≥ {MIN_VOTES} dev votes. "
        "Distance ≈ how differently two models fit as the prompt changes.\n"
        "Average strength is carried by the rating-gap term, so only the prompt-dependent part is shown.",
        transform=ax.transAxes, fontsize=8.5, color="#6b6b66")
ax.legend(loc="lower right", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(root / "reports/r2_model_vectors_pca.png", facecolor="white")
print(ratio[:3].round(3), len(d))
