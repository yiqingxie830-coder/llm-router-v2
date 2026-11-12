"""train_r1b.py — R1 的三分类版本: 同样特征, 预测 高分方赢 / 平局 / 低分方赢

为什么要三分类
--------------
美元口径按 "预期质量提升 ÷ 增量成本" 排序。质量分是 赢 1 / 平 0.5 / 输 0, 所以
  发高分方比发低分方预期多得 = P(赢) + 0.5·P(平) − [P(输) + 0.5·P(平)] = P(赢) − P(输)。
二分类 R1 的 p 是 P(高分方明确赢) (平局记 0), 直接用 2p − 1 = P(赢) − P(输) − P(平),
会把每票少算一个 P(平) (约 0.27)。小评分差的票几乎都被算成"负提升", 排序就错了。
评分差基线没有这个问题: BT 的胜率把平局按半票算, 2p − 1 正好等于 P(赢) − P(输)。

输出两列预测:
  p_win = P(高分方明确赢)               → 按调用比例评估 (和二分类 R1 同口径)
  p_eff = (1 + P(赢) − P(输)) / 2       → 喂给 evaluate_cost.py, 使 2·p_eff − 1 = 预期质量提升

C 沿用二分类 R1 在开发集 5 折上选出的 0.1 (节省时间; 报告里注明)。
用法: python -I train_r1b.py <v2 根目录> [--final]
产出: preds/r1b_win.parquet, preds/r1b_eff.parquet (列 context, id, p)
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

# 复用 train_r1.py 的特征构造 (只执行到函数定义为止)
src = (Path(__file__).parent / "train_r1.py").read_text().split("\nversions = ")[0]
ns = {"__file__": str(Path(__file__).parent / "train_r1.py")}
exec(compile(src, "train_r1.py", "exec"), ns)
fl, design = ns["fl"], ns["design"]
fl = fl.merge(pd.read_parquet(root / "data/fold_labels.parquet", columns=["context", "id", "is_tie"]),
              on=["context", "id"])
fl["cls"] = np.where(fl.y_hi_route == 1, 2, np.where(fl.is_tie, 1, 0))  # 2 赢 / 1 平 / 0 输

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
print("三分类对数损失", round(log_loss(y, p, labels=[0, 1, 2]), 4),
      "平均 P(平)", round(p[:, 1].mean(), 3), "实际平局率", round((y == 1).mean(), 3))
