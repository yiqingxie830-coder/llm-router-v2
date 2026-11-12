"""train_r1.py — 阶段 2 路由器 R1: prompt 向量 + 评分差 + 模型身份 → 逻辑回归

目标 y_hi_route (高分方明确胜出 = 1; 平局和低分方胜 = 0, 平局倾向小模型)。
每个情境 (cv0..cv4, final) 只用该情境的 train 票训练, 预测该情境的 eval 票;
评分差 rating_gap 和高分方/低分方都来自该情境自己的 BT 拟合 (fold_labels.parquet), 不泄漏。

特征 (三个嵌套版本, 用来拆分提升来源)
  gap      : 评分差 / 100                                   ← 等价于评分差基线, 只是重新校准
  gap+id   : + 模型身份 (高分方 +1, 低分方 -1, 共 53 维)       ← 不看 prompt 的上限之一
  gap+id+e : + e5-large prompt 向量 (1024 维, L2 归一化)      ← R1
样本权重 1 / n_votes (同一 prompt 被投多次时降权)。
正则强度 C 用开发集 5 折的平均对数损失选 (测试集不参与); 用 --final 时用选定的 C 在全开发集训练、预测测试集。

用法: python -I train_r1.py <v2 根目录> [--final]
产出: reports/r1_cv_selection.md, preds/r1_<版本>.parquet (context, id, p), 可直接喂 evaluate*.py
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
lines = ["# R1 正则强度选择 (开发集 5 折, eval 票, 不加权的对数损失 / AUC)", "",
         "| 版本 | C | 对数损失 | AUC |", "|---|---|---|---|"]
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
lines += ["", "选定: " + ", ".join(f"{v} → C={c}" for v, c in best.items()), ""]
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
