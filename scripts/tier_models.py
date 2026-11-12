"""tier_models.py — Step 2 (first half): fit Bradley-Terry ratings on this data and propose tiers

Why tiers
---------
The v1 problem: the label was "did GPT-4 or GPT-3.5 win"; once another model (Claude-v1) was mixed in,
a classifier could guess the label from who the opponent was and never learn prompt difficulty.
v2 follows RouteLLM (Ong et al., 2024): group the 53 models into a strong and a weak tier, and the
label becomes "did the strong-tier model or the weak-tier model win". The router then only has to
answer "is this prompt worth the strong tier".

Key decisions
-------------
D1 Ratings come from this data, not the official leaderboard: the labels come from the same votes,
   so both share one definition. The official board also applies "style control" (regressing out
   answer length and markdown); we do not, because our labels are raw, uncorrected human preference.
D2 Fitting data = cleaned main data (order=1, 101,058 votes). Follow-up votes are excluded, as in Step 1.
D3 Bradley-Terry via logistic regression (the LMArena method):
   one row per vote, +1 at model_a's position and -1 at model_b's, target = did A win.
   Ties (tie and both_bad) are split into "half a win for A + half a win for B": not dropped, not biased.
   Coefficients are mapped to the Elo scale: rating = 1000 + 400 * coef / ln(10).
   Meaning: a 400-point gap = 10:1 odds; a 100-point gap ≈ 64% win rate for the higher-rated side.
D4 Bootstrap (resampling votes 200 times) gives each model's rating a 95% interval, so when setting
   tiers we can see whether a gap between two models is larger than the noise.
D5 Tiers: one-dimensional k-means on the ratings (RouteLLM also clusters into many tiers first and then
   picks), tiers numbered from the highest rating down. Which tiers count as strong or weak is decided
   by a person; this script only outputs the candidates.

Usage: python -I tier_models.py <repo root>
Output: reports/step2_ratings.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

root = Path(sys.argv[1])
df = pd.read_parquet(root / "data/arena_clean.parquet",
                     columns=["model_a", "model_b", "winner"])
models = sorted(set(df.model_a) | set(df.model_b))
idx = {m: i for i, m in enumerate(models)}
SCALE, BASE, INIT = 400, 10, 1000


def fit_bt(d: pd.DataFrame) -> np.ndarray:
    """Return each model's rating on the Elo scale."""
    n = len(d)
    X = np.zeros((n, len(models)))
    X[np.arange(n), d.model_a.map(idx).values] = 1
    X[np.arange(n), d.model_b.map(idx).values] = -1
    a_win = (d.winner == "model_a").values
    b_win = (d.winner == "model_b").values
    tie = ~(a_win | b_win)
    # Split each tie into two rows of weight 0.5: one where A wins, one where B wins
    X2 = np.vstack([X, X[tie]])
    y2 = np.concatenate([a_win | tie, np.zeros(tie.sum(), bool)]).astype(int)
    w2 = np.concatenate([np.where(tie, 0.5, 1.0), np.full(tie.sum(), 0.5)])
    lr = LogisticRegression(fit_intercept=False, C=1e6, max_iter=2000, tol=1e-8)
    lr.fit(X2, y2, sample_weight=w2)
    r = SCALE * lr.coef_[0] / np.log(BASE)
    return INIT + r - r.mean()  # mean fixed at 1000; only differences are meaningful


rating = fit_bt(df)
rng = np.random.default_rng(42)
boots = np.array([fit_bt(df.sample(len(df), replace=True, random_state=rng.integers(1 << 31)))
                  for _ in range(200)])
n_games = pd.concat([df.model_a, df.model_b]).value_counts()
tab = pd.DataFrame({
    "model": models, "rating": rating.round(1),
    "ci_low": np.percentile(boots, 2.5, axis=0).round(1),
    "ci_high": np.percentile(boots, 97.5, axis=0).round(1),
}).assign(n_games=lambda t: t.model.map(n_games)).sort_values("rating", ascending=False)
tab.to_csv(root / "reports/step2_ratings.csv", index=False)

# One-dimensional k-means tiers (several k, fixed seed)
for k in (4, 5, 6, 8):
    km = KMeans(n_clusters=k, n_init=50, random_state=0).fit(tab[["rating"]])
    order = np.argsort(-km.cluster_centers_[:, 0])  # tier 1 = highest rating
    rank = {c: i + 1 for i, c in enumerate(order)}
    tab[f"tier_k{k}"] = [rank[c] for c in km.labels_]
tab.to_csv(root / "reports/step2_ratings.csv", index=False)
print(tab.to_string(index=False))
