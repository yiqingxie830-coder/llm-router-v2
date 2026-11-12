"""tie_baselines.py — rating-gap baselines with ties (set 2026-11-10: the baseline and R1 must use the same target, P(win) − P(loss))

Two versions, each fitted only on the context's own train votes (weight 1/n_votes) and predicting that context's eval votes:
  empirical bucket shares (formal): bucket train votes by deciles of the rating gap and compute the weighted
                       win / tie / loss shares per bucket; an eval vote takes the shares of its bucket.
                       No functional form is assumed.
  3-class rating gap (check): multinomial logistic regression whose only feature is gap / 100; the same model
                       family as the 3-class R1, minus the prompt and model identity.
Output p = (1 + P(win) − P(loss)) / 2, so 2p − 1 = expected quality gain; feeds evaluate_cost.py directly.

Usage: python -I tie_baselines.py <repo root> [--final]
Writes: preds/base_tie_bucket.parquet, preds/base_tie_mlogit.parquet (columns context, id, p, p_win, p_tie, p_lose)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

root = Path(sys.argv[1])
use_final = "--final" in sys.argv
NB = 10
fl = pd.read_parquet(root / "data/fold_labels.parquet",
                     columns=["context", "id", "role", "rating_gap", "y_hi_route", "is_tie", "n_votes"])
fl["cls"] = np.where(fl.y_hi_route == 1, 2, np.where(fl.is_tie, 1, 0))
ctxs = ["final"] if use_final else [f"cv{k}" for k in range(5)]
out_b, out_m = [], []
for ctx in ctxs:
    d = fl[fl.context == ctx]
    tr, ev = d[d.role == "train"], d[d.role == "eval"]
    w = 1 / tr.n_votes.values
    # buckets
    edges = np.unique(np.quantile(tr.rating_gap, np.linspace(0, 1, NB + 1)))
    bt = np.clip(np.searchsorted(edges, tr.rating_gap, side="right") - 1, 0, len(edges) - 2)
    be = np.clip(np.searchsorted(edges, ev.rating_gap, side="right") - 1, 0, len(edges) - 2)
    P = np.zeros((len(edges) - 1, 3))
    for b in range(len(edges) - 1):
        m = bt == b
        P[b] = np.bincount(tr.cls.values[m], weights=w[m], minlength=3) / w[m].sum()
    pb = P[be]
    out_b.append(pd.DataFrame({"context": ctx, "id": ev.id.values, "p": (1 + pb[:, 2] - pb[:, 0]) / 2,
                               "p_win": pb[:, 2], "p_tie": pb[:, 1], "p_lose": pb[:, 0]}))
    # 3-class logistic regression
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(tr.rating_gap.values[:, None] / 100, tr.cls.values,
                                                       sample_weight=w)
    pm = clf.predict_proba(ev.rating_gap.values[:, None] / 100)[:, np.searchsorted(clf.classes_, [0, 1, 2])]
    out_m.append(pd.DataFrame({"context": ctx, "id": ev.id.values, "p": (1 + pm[:, 2] - pm[:, 0]) / 2,
                               "p_win": pm[:, 2], "p_tie": pm[:, 1], "p_lose": pm[:, 0]}))
    if ctx == ctxs[0]:
        print(ctx, "bucket edges", edges.round(1).tolist())
        print("loss/tie/win per bucket", P.round(3).tolist())
sfx = "_final" if use_final else ""
pd.concat(out_b).to_parquet(root / f"preds/base_tie_bucket{sfx}.parquet", index=False)
pd.concat(out_m).to_parquet(root / f"preds/base_tie_mlogit{sfx}.parquet", index=False)
print("done")
