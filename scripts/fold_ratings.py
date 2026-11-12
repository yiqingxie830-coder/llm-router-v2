"""fold_ratings.py — Step 4 (second half): fit Bradley-Terry per fold, recompute labels and tiers

Why ("rating leakage", raised 2026-11-09)
-----------------------------------------
"Who is the higher-rated side" and "who is in the strong tier" both come from BT ratings. If the ratings
were fitted on all votes, test votes would take part, and part of the test answers would leak into the
label definition and a feature (the rating gap).
So every training/evaluation context fits ratings only on its own training votes:
  - cv0..cv4: fit on dev votes outside fold k; label fold k (eval) and the other 4 folds (train)
  - final   : fit on the whole dev set (90%); label the test set (eval) and the dev set (train)
In every context the higher-rated side, y_hi, y_hi_route, rating gap, tiers, strong/weak view membership,
y / y_route and the soft labels are all recomputed.

Other decisions
---------------
D1 The BT fit is identical to tier_models.py (logistic regression, ties as half votes, Elo scale, mean 1000).
D2 Strong / weak tiers = fixed lists (decided 2026-11-09), no per-fold k-means:
   - Why: one-dimensional k-means (and RouteLLM's dynamic-programming tiers) merged the top two tiers on
     cv0, so "tiers 1-2" jumped from 10 to 21 models; the strong-vs-weak vote set then differed by fold
     and scores could not be averaged.
   - How: fit BT once on the 90% dev set (= the final context), plot rank vs rating
     (reports/step4_rank_rating.png) and cut at the clear gaps:
       21.4 points between ranks 10 / 11 -> top 10 = strong tier
       24.6 points between ranks 40 / 41 -> bottom 11 = weak tier
       ranks 11-40 (30 models) = middle band, not part of the strong-vs-weak view
     The largest gap inside ranks 11-40 is only 9.1 points; there is no other clear break.
   - Stability: on either side of both cuts, the models' rating ranges across the 6 contexts do not
     overlap, so the lists hold for every fold.
   - The lists were set from the dev set only; the test set played no part.
D3 A model absent from a context's training votes (only a few rare models): rating set to 1000 (the mean),
   no tier; visible in the n_games column.
D4 This script only builds labels; it trains no router.

Usage: python -I fold_ratings.py <repo root>
Output:
  data/fold_labels.parquet           long table: one row per (context, id) with all labels for that context
  data/arena_strong_vs_weak.parquet  strong-vs-weak evaluation view in the final context (test + dev), with fold
  reports/step4_ratings_by_context.csv, reports/step4_fold_labels.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

root = Path(sys.argv[1])
SCALE, BASE, INIT = 400, 10, 1000
STRONG_MODELS = ['gemini-2.5-pro', 'gemini-2.5-pro-preview-03-25', 'grok-4-0709', 'o3-2025-04-16', 'gemini-2.5-pro-preview-05-06', 'deepseek-r1-0528', 'chatgpt-4o-latest-20250326', 'grok-3-preview-02-24', 'gemini-2.5-flash', 'llama-4-maverick-03-26-experimental']
WEAK_MODELS = ['llama-4-scout-17b-16e-instruct', 'llama-4-maverick-17b-128e-instruct', 'claude-3-7-sonnet-20250219-thinking-32k', 'claude-3-5-sonnet-20241022', 'claude-3-7-sonnet-20250219', 'amazon.nova-pro-v1:0', 'llama-3.3-70b-instruct', 'mistral-small-3.1-24b-instruct-2503', 'gpt-4o-mini-2024-07-18', 'magistral-medium-2506', 'claude-3-5-haiku-20241022']

df = pd.read_parquet(root / "data/arena_labeled.parquet")
keep_cols = ["id", "group_id", "split", "fold", "model_a", "model_b", "winner", "is_tie",
             "tie_type", "is_en", "language", "prompt_len"]
df = df[keep_cols]
models = sorted(set(df.model_a) | set(df.model_b))
idx = {m: i for i, m in enumerate(models)}


def fit_bt(d):
    n = len(d)
    X = np.zeros((n, len(models)))
    X[np.arange(n), d.model_a.map(idx).values] = 1
    X[np.arange(n), d.model_b.map(idx).values] = -1
    a_win, b_win = (d.winner == "model_a").values, (d.winner == "model_b").values
    tie = ~(a_win | b_win)
    X2 = np.vstack([X, X[tie]])
    y2 = np.concatenate([a_win | tie, np.zeros(tie.sum(), bool)]).astype(int)
    w2 = np.concatenate([np.where(tie, 0.5, 1.0), np.full(tie.sum(), 0.5)])
    lr = LogisticRegression(fit_intercept=False, C=1e6, max_iter=3000, tol=1e-8)
    lr.fit(X2, y2, sample_weight=w2)
    r = SCALE * lr.coef_[0] / np.log(BASE)
    games = pd.concat([d.model_a, d.model_b]).value_counts().reindex(models).fillna(0).astype(int)
    seen = games.values > 0
    r = INIT + r - r[seen].mean()
    r[~seen] = INIT  # D3
    return pd.DataFrame({"model": models, "rating": r, "n_games": games.values})


def tiers(rt):
    """Fixed lists: strong / weak / mid (middle band or models with too few games)."""
    return rt.model.map(lambda m: "strong" if m in STRONG_MODELS else "weak" if m in WEAK_MODELS else "mid")


contexts = {f"cv{k}": (df.split.eq("dev") & df.fold.ne(k), df.fold.eq(k)) for k in range(5)}
contexts["final"] = (df.split.eq("dev"), df.split.eq("test"))

rows, rtabs = [], []
for ctx, (tr_mask, ev_mask) in contexts.items():
    train = df[tr_mask]
    rt = fit_bt(train)
    rt["tier"] = tiers(rt).values
    rtabs.append(rt.assign(context=ctx))
    r, n, t = (rt.set_index("model")[c] for c in ("rating", "n_games", "tier"))

    d = df[tr_mask | ev_mask].copy()
    d["context"] = ctx
    d["role"] = np.where(ev_mask[d.index], "eval", "train")
    d["rating_a"], d["rating_b"] = d.model_a.map(r), d.model_b.map(r)
    d["n_games_a"], d["n_games_b"] = d.model_a.map(n), d.model_b.map(n)
    d["tier_a"], d["tier_b"] = d.model_a.map(t), d.model_b.map(t)
    a_hi = d.rating_a >= d.rating_b
    d["hi_model"] = np.where(a_hi, d.model_a, d.model_b)
    d["lo_model"] = np.where(a_hi, d.model_b, d.model_a)
    d["rating_gap"] = (d.rating_a - d.rating_b).abs()
    hi_won = np.where(a_hi, d.winner == "model_a", d.winner == "model_b")
    d["y_hi"] = np.where(d.is_tie, np.nan, hi_won.astype(float))
    d["y_hi_route"] = np.where(d.is_tie, 0.0, hi_won.astype(float))
    # Quality score used in evaluation: score of sending to the higher-rated side (win 1 / tie 0.5 / loss 0)
    d["q_hi"] = np.where(d.is_tie, 0.5, hi_won.astype(float))
    sa = d.tier_a.eq("strong") & d.tier_b.eq("weak")
    sb = d.tier_b.eq("strong") & d.tier_a.eq("weak")
    # Safety check: in this context every listed strong model must outrank every weak one, or "strong = higher-rated" fails
    assert r[STRONG_MODELS].min() > r[WEAK_MODELS].max(), ctx
    d["is_strong_vs_weak"] = sa | sb
    strong_won = np.where(sa, d.winner == "model_a", d.winner == "model_b")
    d["y"] = np.where(d.is_strong_vs_weak & ~d.is_tie, strong_won.astype(float), np.nan)
    d["y_route"] = np.where(d.is_strong_vs_weak, (strong_won & ~d.is_tie).astype(float), np.nan)
    # Soft labels: aggregated by group within a context (groups never cross folds, so no eval votes mix in)
    g = d.assign(v=d.y_hi.fillna(0.5)).groupby("group_id")
    d["soft_label_hi"] = g.v.transform("mean")
    d["soft_label_hi_route"] = g.y_hi_route.transform("mean")
    d["n_votes"] = g.v.transform("size")
    sw = d[d.is_strong_vs_weak].assign(v=lambda x: x.y.fillna(0.5)).groupby("group_id")
    d.loc[d.is_strong_vs_weak, "soft_label"] = sw.v.transform("mean")
    d.loc[d.is_strong_vs_weak, "soft_label_route"] = sw.y_route.transform("mean")
    d.loc[d.is_strong_vs_weak, "n_votes_sw"] = sw.v.transform("size")
    rows.append(d)
    print(ctx, "train", len(train), "labeled", len(d), "sw", int(d.is_strong_vs_weak.sum()))

fl = pd.concat(rows, ignore_index=True)
fl.to_parquet(root / "data/fold_labels.parquet", index=False)
rts = pd.concat(rtabs, ignore_index=True)
rts.to_csv(root / "reports/step4_ratings_by_context.csv", index=False)

prompts = pd.read_parquet(root / "data/arena_labeled.parquet", columns=["id", "prompt"])
fin = fl[(fl.context == "final") & fl.is_strong_vs_weak].merge(prompts, on="id")
fin.to_parquet(root / "data/arena_strong_vs_weak.parquet", index=False)

# ---------- report: flip statistics ----------
hi_n = fl.groupby("id").hi_model.nunique()
gap_full = pd.read_parquet(root / "data/arena_labeled.parquet", columns=["id", "rating_gap"]).set_index("id").rating_gap
flipped = hi_n[hi_n > 1].index
sw_n = fl.groupby("id").is_strong_vs_weak.agg(["min", "max"])
sw_changed = (sw_n["min"] != sw_n["max"]).sum()
wide = rts.pivot(index="model", columns="context", values="rating")
gap_bins = pd.cut(gap_full.loc[flipped], [0, 5, 10, 20, 50, 1e9], include_lowest=True).value_counts().sort_index()

md = ["# Step 4 Bradley-Terry fitted per fold", "",
      "## Votes and strong-vs-weak view size per context", "",
      fl.groupby(["context", "role"]).agg(votes=("id", "size"), strong_vs_weak=("is_strong_vs_weak", "sum"),
                                          higher_rated_clear_win=("y_hi_route", "mean")).round(3).to_markdown(), "",
      "## Flips of the higher-rated side", "",
      f"- Of {hi_n.size} votes, the higher-rated side flips between contexts on **{len(flipped)} votes ({len(flipped) / hi_n.size:.2%})**",
      f"- Model pairs involved: {fl[fl.id.isin(flipped)].drop_duplicates('id').apply(lambda x: ' vs '.join(sorted([x.model_a, x.model_b])), axis=1).nunique()}",
      "- Flipped votes by full-data rating gap (flips happen only where the two models were already very close):", "",
      gap_bins.to_frame("votes").to_markdown(), "",
      f"## Strong-vs-weak view membership changes: {sw_changed} votes are in the view in some contexts but not others (fixed lists, should be 0)", "",
      f"Lowest strong-tier rating vs highest weak-tier rating per context: " + ", ".join(
          f"{c}: {g[g.model.isin(STRONG_MODELS)].rating.min():.0f} > {g[g.model.isin(WEAK_MODELS)].rating.max():.0f}"
          for c, g in rts.groupby("context")), "",
      "## Largest rating difference across contexts (same model, max - min)", "",
      (wide.max(axis=1) - wide.min(axis=1)).describe().round(1).to_frame("points").to_markdown()]
(root / "reports/step4_fold_labels.md").write_text("\n".join(md))
print("\n".join(md))
