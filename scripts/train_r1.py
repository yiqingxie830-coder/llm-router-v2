"""train_r1.py — Stage 2 router R1: prompt vector + rating gap + model identity -> logistic regression

Target y_hi_route (higher-rated model wins outright = 1; ties and lower-rated wins = 0, so ties lean to the small model).
Each context (cv0..cv4, final) trains only on its own train votes and predicts its own eval votes;
rating_gap and the higher/lower-rated sides come from that context's own BT fit (fold_labels.parquet), so nothing leaks.

Features (three nested versions, to split where the gain comes from)
  gap      : rating gap / 100                                       <- equivalent to the rating-gap baseline, just recalibrated
  gap+id   : + model identity (higher-rated +1, lower-rated -1, 53 dims) <- one upper bound for not reading the prompt
  gap+id+e : + e5-large prompt vector (1024 dims, L2-normalized)    <- R1
Sample weight 1 / n_votes (down-weights prompts voted on many times).
Regularization C is chosen by mean log loss over the dev set's 5 folds (the test set is not used); with --final,
the chosen C is trained on the full dev set and predicts the test set.

Usage: python -I train_r1.py <repo root> [--final]
Writes: reports/r1_cv_selection.md, preds/r1_<version>.parquet (context, id, p), ready for evaluate*.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

root = Path(sys.argv[1])
use_final = "--final" in sys.argv
CS = [0.01, 0.03, 0.1, 1.0]
(root / "preds").mkdir(exist_ok=True)

fl = pd.read_parquet(root / "data/fold_labels.parquet",
                     columns=["context", "id", "role", "group_id", "hi_model", "lo_model", "rating_gap",
                              "y_hi_route", "n_votes"])
emb = np.load(root / "data/prompt_emb_e5large.npy").astype(np.float32)
gidx = pd.Series(np.arange(len(emb)), index=pd.read_parquet(root / "data/prompt_emb_index.parquet").group_id)
models = sorted(set(fl.hi_model) | set(fl.lo_model))
mpos = {m: i for i, m in enumerate(models)}


def design(d, version):
    cols = [sparse.csr_matrix(d.rating_gap.values[:, None] / 100)]
    if version in ("gap+id", "gap+id+e"):
        n = len(d)
        r = np.arange(n)
        ident = sparse.csr_matrix((np.r_[np.ones(n), -np.ones(n)],
                                   (np.r_[r, r], np.r_[d.hi_model.map(mpos).values, d.lo_model.map(mpos).values])),
                                  shape=(n, len(models)))
        cols.append(ident)
    if version == "gap+id+e":
        cols.append(sparse.csr_matrix(emb[gidx.loc[d.group_id].values]))
    return sparse.hstack(cols).tocsr()


def fit_predict(ctx, version, C):
    d = fl[fl.context == ctx]
    tr, ev = d[d.role == "train"], d[d.role == "eval"]
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(design(tr, version), tr.y_hi_route.values, sample_weight=1 / tr.n_votes.values)
    p = clf.predict_proba(design(ev, version))[:, 1]
    return ev, p


versions = ["gap", "gap+id", "gap+id+e"]
lines = ["# R1 regularization choice (dev set 5-fold, eval votes, unweighted log loss / AUC)", "",
         "| Version | C | Log loss | AUC |", "|---|---|---|---|"]
best, cache = {}, {}
for v in versions:
    scores = {}
    for C in (CS if v != "gap" else [1.0]):
        t = time.time()
        ys, ps, evs = [], [], []
        for k in range(5):
            ev, p = fit_predict(f"cv{k}", v, C)
            evs.append(ev)
            ys.append(ev.y_hi_route.values)
            ps.append(p)
        cache[(v, C)] = [pd.DataFrame({"context": f"cv{k}", "id": e.id.values, "p": q})
                         for k, (e, q) in enumerate(zip(evs, ps))]
        y, p = np.concatenate(ys), np.concatenate(ps)
        scores[C] = log_loss(y, p)
        lines.append(f"| {v} | {C} | {scores[C]:.4f} | {roc_auc_score(y, p):.4f} |")
        print(v, C, round(scores[C], 4), f"{time.time() - t:.0f}s", flush=True)
    best[v] = min(scores, key=scores.get)
lines += ["", "Chosen: " + ", ".join(f"{v} → C={c}" for v, c in best.items()), ""]
(root / "reports/r1_cv_selection.md").write_text("\n".join(lines))

for v in versions:
    ctxs = ["final"] if use_final else [f"cv{k}" for k in range(5)]
    out = [] if use_final else cache[(v, best[v])]
    for ctx in (ctxs if use_final else []):
        ev, p = fit_predict(ctx, v, best[v])
        out.append(pd.DataFrame({"context": ctx, "id": ev.id.values, "p": p}))
    tag = v.replace("+", "_")
    pd.concat(out).to_parquet(root / f"preds/r1_{tag}{'_final' if use_final else ''}.parquet", index=False)
print("\n".join(lines))
