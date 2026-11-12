"""r1_csweep.py — regularization C for the 3-class R1: nested selection + sensitivity table (item 4 of the 2026-11-10 decisions)

Selection rule (training folds only, the test set is not used):
  for each outer fold cv k, take only that fold's train votes, split them again 80% / 20% by group_id (fixed seed,
  a prompt never straddles the split), train on 80%, compute the 3-class log loss on 20% (weight 1/n_votes) and
  take the C with the lowest loss; then retrain with that C on the whole train part and predict cv k's eval votes.
  Eval votes never take part in choosing C.
Sensitivity: every C is trained on all 5 folds; reports point estimates of strong-vs-weak APGR (call share) and
  main dollar case APGR$.

Usage: python -I r1_csweep.py <repo root> <C>      (one process per C, can run in parallel)
       python -I r1_csweep.py <repo root> --summarize
Writes: preds/r1b_C<C>.parquet, reports/r1_csweep_<C>.json, summary reports/r1_c_selection.md, preds/r1b_nested_eff.parquet
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
CS = [0.01, 0.03, 0.1, 0.3, 1.0]
FINE, GRID = np.linspace(0, 1, 101), np.linspace(0, 1, 11)
FLOOR = 0.25 / 1000


def load_design():
    src = (Path(__file__).parent / "train_r1.py").read_text().split("\nversions = ")[0]
    ns = {}
    sys_argv = sys.argv
    sys.argv = [sys.argv[0], str(root)]
    exec(compile(src, "train_r1.py", "exec"), ns)
    sys.argv = sys_argv
    fl = ns["fl"].merge(pd.read_parquet(root / "data/fold_labels.parquet", columns=["context", "id", "is_tie"]),
                        on=["context", "id"])
    fl["cls"] = np.where(fl.y_hi_route == 1, 2, np.where(fl.is_tie, 1, 0))
    return fl, ns["design"]


def fit3(X, y, w, C):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=3000).fit(X, y, sample_weight=w)


def proba(clf, X):
    return clf.predict_proba(X)[:, np.searchsorted(clf.classes_, [0, 1, 2])]


def run_one(C):
    from sklearn.metrics import log_loss
    fl, design = load_design()
    rows, inner = [], {}
    for k in range(5):
        d = fl[fl.context == f"cv{k}"]
        tr, ev = d[d.role == "train"], d[d.role == "eval"]
        h = tr.group_id.map(lambda g: int(hashlib.md5(f"2026{g}".encode()).hexdigest(), 16) % 5 == 0).values
        a, b = tr[~h], tr[h]
        clf = fit3(design(a, "gap+id+e"), a.cls.values, 1 / a.n_votes.values, C)
        inner[k] = float(log_loss(b.cls.values, proba(clf, design(b, "gap+id+e")), labels=[0, 1, 2],
                                  sample_weight=1 / b.n_votes.values))
        clf = fit3(design(tr, "gap+id+e"), tr.cls.values, 1 / tr.n_votes.values, C)
        p = proba(clf, design(ev, "gap+id+e"))
        rows.append(pd.DataFrame({"context": f"cv{k}", "id": ev.id.values, "p_win": p[:, 2],
                                  "p": (1 + p[:, 2] - p[:, 0]) / 2}))
        print(C, k, inner[k], flush=True)
    pd.concat(rows).to_parquet(root / f"preds/r1b_C{C}.parquet", index=False)
    (root / f"reports/r1_csweep_{C}.json").write_text(json.dumps(inner))


def apgr_frac(q, p):
    o = np.argsort(-p, kind="stable")
    qs = q[o]
    r_w, r_s = (1 - q).mean(), q.mean()
    gain = np.concatenate([[0], np.cumsum(2 * qs - 1)]) / len(q)
    cw = np.arange(len(q) + 1) / len(q)
    pgr = np.interp(GRID, cw, gain / (r_s - r_w))
    return np.trapezoid(pgr, GRID)


def apgr_cost(q, key, extra):
    o = np.argsort(-key, kind="stable")
    qs, es = q[o], extra[o]
    r_w, r_s = (1 - q).mean(), q.mean()
    gain = np.concatenate([[0], np.cumsum(2 * qs - 1)]) / len(q)
    ce = np.maximum.accumulate(np.concatenate([[0], np.cumsum(es)]))
    ce = ce / ce[-1]
    pgr = np.interp(FINE, ce, gain / (r_s - r_w))
    return np.trapezoid(np.interp(GRID, FINE, pgr), GRID), FINE[np.argmax(pgr >= 0.5)]


def summarize():
    fl = pd.read_parquet(root / "data/fold_labels.parquet",
                         columns=["context", "id", "role", "q_hi", "is_strong_vs_weak", "hi_model", "model_a"])
    ev = fl[(fl.role == "eval") & fl.context.str.startswith("cv")]
    vc = pd.read_parquet(root / "data/vote_costs.parquet")
    ev = ev.merge(vc, on="id")
    a = ev.hi_model.values == ev.model_a.values
    ev = ev.assign(cost_hi=np.where(a, ev.cost_a, ev.cost_b), cost_lo=np.where(a, ev.cost_b, ev.cost_a),
                   typ_hi=np.where(a, ev.typ_a, ev.typ_b), typ_lo=np.where(a, ev.typ_b, ev.typ_a))
    sw = ev[ev.is_strong_vs_weak.values]
    main = ev[ev.is_strong_vs_weak.values & (ev.typ_hi > ev.typ_lo).values]
    inner = {C: json.loads((root / f"reports/r1_csweep_{C}.json").read_text()) for C in CS}
    pick = {k: min(CS, key=lambda C: inner[C][str(k)]) for k in range(5)}
    preds = {C: pd.read_parquet(root / f"preds/r1b_C{C}.parquet") for C in CS}
    nested = pd.concat([preds[pick[k]][preds[pick[k]].context == f"cv{k}"] for k in range(5)])
    nested[["context", "id", "p"]].to_parquet(root / "preds/r1b_nested_eff.parquet", index=False)
    preds["nested selection"] = nested
    L = ["# R1 (3-class): choice of regularization C and sensitivity", "",
         "**How C is chosen:** each outer fold uses only its own train votes, splits them 80/20 by prompt group, trains on 80% and computes log loss on 20%, "
         "takes the C with the lowest loss, then retrains on the whole train part and predicts that fold's eval votes. Neither eval votes nor the test set take part in choosing C.", "",
         "Grid C ∈ " + str(CS) + ". Chosen per fold: " + ", ".join(f"cv{k} → {pick[k]}" for k in range(5)), "",
         "| C | Inner validation log loss (5-fold mean) | Strong vs weak APGR (call share) | Main dollar case APGR$ | Main case CPT$(50%) |",
         "|---|---|---|---|---|"]
    for C, pr in preds.items():
        s = sw.merge(pr, on=["context", "id"])
        m = main.merge(pr, on=["context", "id"])
        key = (2 * m.p.values - 1) / np.maximum((m.typ_hi - m.typ_lo).values, FLOOR)
        ap, cpt = apgr_cost(m.q_hi.values, key, (m.cost_hi - m.cost_lo).values)
        il = f"{np.mean(list(inner[C].values())):.4f}" if C in inner else "—"
        L.append(f"| {C} | {il} | {apgr_frac(s.q_hi.values, s.p_win.values):.3f} | {ap:.3f} | {cpt:.2f} |")
    L += ["", "Point estimates, no bootstrap; the sensitivity rows only show whether the conclusion depends on C."]
    (root / "reports/r1_c_selection.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[2] == "--summarize":
        summarize()
    else:
        run_one(float(sys.argv[2]))
