"""train_r1b.py — 3-class version of R1: same features, predicts higher-rated wins / tie / lower-rated wins

Why three classes
-----------------
The dollar view ranks by "expected quality gain ÷ extra cost". Quality is win 1 / tie 0.5 / loss 0, so
  the expected gain of routing to the higher-rated model = P(win) + 0.5·P(tie) − [P(loss) + 0.5·P(tie)] = P(win) − P(loss).
The binary R1's p is P(higher-rated wins outright) (ties count as 0); using 2p − 1 = P(win) − P(loss) − P(tie) directly
undercounts every vote by P(tie) (about 0.27). Votes with small rating gaps would almost all get a "negative gain"
and the ranking would be wrong.
The rating-gap baseline does not have this problem: BT's win rate counts ties as half votes, so 2p − 1 is exactly P(win) − P(loss).

Two prediction outputs:
  p_win = P(higher-rated wins outright)   -> evaluation on the call-share axis (same target as the binary R1)
  p_eff = (1 + P(win) − P(loss)) / 2      -> fed to evaluate_cost.py, so 2·p_eff − 1 = expected quality gain

C reuses 0.1, chosen for the binary R1 on the dev set's 5 folds (saves time; noted in the report).
Usage: python -I train_r1b.py <repo root> [--final]
Writes: preds/r1b_win.parquet, preds/r1b_eff.parquet (columns context, id, p)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

sys.argv = sys.argv[:2] + [a for a in sys.argv[2:]]
root = Path(sys.argv[1])
use_final = "--final" in sys.argv
C = 0.1

# reuse train_r1.py's feature construction (executed only up to the function definitions)
src = (Path(__file__).parent / "train_r1.py").read_text().split("\nversions = ")[0]
ns = {"__file__": str(Path(__file__).parent / "train_r1.py")}
exec(compile(src, "train_r1.py", "exec"), ns)
fl, design = ns["fl"], ns["design"]
fl = fl.merge(pd.read_parquet(root / "data/fold_labels.parquet", columns=["context", "id", "is_tie"]),
              on=["context", "id"])
fl["cls"] = np.where(fl.y_hi_route == 1, 2, np.where(fl.is_tie, 1, 0))  # 2 win / 1 tie / 0 loss

ctxs = ["final"] if use_final else [f"cv{k}" for k in range(5)]
win, eff, ys, ps = [], [], [], []
for ctx in ctxs:
    t = time.time()
    d = fl[fl.context == ctx]
    tr, ev = d[d.role == "train"], d[d.role == "eval"]
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit(design(tr, "gap+id+e"), tr.cls.values, sample_weight=1 / tr.n_votes.values)
    pr = clf.predict_proba(design(ev, "gap+id+e"))
    pr = pr[:, np.searchsorted(clf.classes_, [0, 1, 2])]
    ys.append(ev.cls.values)
    ps.append(pr)
    win.append(pd.DataFrame({"context": ctx, "id": ev.id.values, "p": pr[:, 2]}))
    eff.append(pd.DataFrame({"context": ctx, "id": ev.id.values, "p": (1 + pr[:, 2] - pr[:, 0]) / 2}))
    print(ctx, f"{time.time() - t:.0f}s", flush=True)
sfx = "_final" if use_final else ""
pd.concat(win).to_parquet(root / f"preds/r1b_win{sfx}.parquet", index=False)
pd.concat(eff).to_parquet(root / f"preds/r1b_eff{sfx}.parquet", index=False)
y, p = np.concatenate(ys), np.concatenate(ps)
print("3-class log loss", round(log_loss(y, p, labels=[0, 1, 2]), 4),
      "mean P(tie)", round(p[:, 1].mean(), 3), "actual tie rate", round((y == 1).mean(), 3))
