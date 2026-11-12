"""r2_final.py — R2 (MF) 的 --final: 按预登记 (reports/final_prereg.md) 的设置, 在整个开发集上训练, 预测测试集一次

设置: d = 128, Adam 1e-3, λ = 1e-5 (开发集 5 折嵌套选择 4/5 折选中); 轮数 = 开发集 train 内部 80/20 (按 prompt 组哈希) 早停。
用法: python -I r2_final.py <v2 根目录>
产出: preds/r2_win_final.parquet (p = P(赢)), preds/r2_eff_final.parquet (p = (1 + P赢 − P输)/2), reports/r2_final_train.json
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
