"""r2_final.py — --final for R2 (MF): with the preregistered settings (reports/final_prereg.md), train on the full dev set and predict the test set once

Settings: d = 128, Adam 1e-3, λ = 1e-5 (chosen in 4 of 5 folds by nested selection on the dev set); epochs = early stopping on an 80/20 split (by prompt-group hash) inside the dev-set train votes.
Usage: python -I r2_final.py <repo root>
Writes: preds/r2_win_final.parquet (p = P(win)), preds/r2_eff_final.parquet (p = (1 + P(win) − P(loss))/2), reports/r2_final_train.json
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
src = (Path(__file__).parent / "train_r2_mf.py").read_text().split("\nlines = [")[0].replace("optax.adam(1e-3)", "optax.adam(LR)")
ns = {"__name__": "mf"}
argv = sys.argv
sys.argv = [argv[0], str(root)]
exec(compile(src, "train_r2_mf.py", "exec"), ns)
sys.argv = argv
D, LR, LAM = 128, 1e-3, 1e-5
ns["K"], ns["LR"] = D, LR
fl, arrays, train, probs = ns["fl"], ns["arrays"], ns["train"], ns["probs"]
d = fl[fl.context == "final"]
tr, ev = d[d.role == "train"], d[d.role == "eval"]
h = tr.group_id.map(lambda g: int(hashlib.md5(f"2026{g}".encode()).hexdigest(), 16) % 5 == 0).values
t = time.time()
_, ep, v = train(arrays(tr[~h]), LAM, val=arrays(tr[h]))
p, _, _ = train(arrays(tr), LAM, epochs=ep)
gi, hi, lo, gap, _, _ = arrays(ev)
pr = np.asarray(probs(p, gi, hi, lo, gap))
pd.DataFrame({"context": "final", "id": ev.id.values, "p": pr[:, 2]}).to_parquet(root / "preds/r2_win_final.parquet", index=False)
pd.DataFrame({"context": "final", "id": ev.id.values, "p": (1 + pr[:, 2] - pr[:, 0]) / 2}).to_parquet(
    root / "preds/r2_eff_final.parquet", index=False)
info = {"d": D, "lr": LR, "lambda": LAM, "inner_val_loss": v, "epochs": ep, "train_votes": int(len(tr)),
        "test_votes": int(len(ev)), "seconds": round(time.time() - t, 1)}
(root / "reports/r2_final_train.json").write_text(json.dumps(info))
print(info)
