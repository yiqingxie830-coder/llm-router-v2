"""tie_baselines.py — 带平局的评分差基线 (Yiqing 2026-11-10: 基线和 R1 必须同口径, 都用 P(赢) − P(输))

两个版本, 都只用各情境自己的 train 票拟合 (权重 1/n_votes), 预测该情境的 eval 票:
  分桶经验比例 (正式): 按 train 票评分差的 10 分位分桶, 每桶算加权的 赢/平/输 比例;
                       eval 票落到哪个桶就用哪个桶的比例。不假设任何函数形式。
  三分类评分差 (核对): 多项逻辑回归, 特征只有 评分差/100, 和 R1 三分类是同一类模型、少了 prompt 和模型身份。
输出 p = (1 + P(赢) − P(输)) / 2, 使 2p − 1 = 预期质量提升, 直接给 evaluate_cost.py 的 --base 用。

用法: python -I tie_baselines.py <v2 根目录> [--final]
产出: preds/base_tie_bucket.parquet, preds/base_tie_mlogit.parquet (列 context, id, p, p_win, p_tie, p_lose)
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
    # 分桶
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
    # 三分类逻辑回归
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(tr.rating_gap.values[:, None] / 100, tr.cls.values,
                                                       sample_weight=w)
    pm = clf.predict_proba(ev.rating_gap.values[:, None] / 100)[:, np.searchsorted(clf.classes_, [0, 1, 2])]
    out_m.append(pd.DataFrame({"context": ctx, "id": ev.id.values, "p": (1 + pm[:, 2] - pm[:, 0]) / 2,
                               "p_win": pm[:, 2], "p_tie": pm[:, 1], "p_lose": pm[:, 0]}))
    if ctx == ctxs[0]:
        print(ctx, "分桶边界", edges.round(1).tolist())
        print("各桶 输/平/赢", P.round(3).tolist())
sfx = "_final" if use_final else ""
pd.concat(out_b).to_parquet(root / f"preds/base_tie_bucket{sfx}.parquet", index=False)
pd.concat(out_m).to_parquet(root / f"preds/base_tie_mlogit{sfx}.parquet", index=False)
print("done")
