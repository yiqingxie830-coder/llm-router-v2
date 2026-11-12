"""tier_models.py — Step 2 (上半): 用本数据拟合 Bradley-Terry 评分, 并给出分档候选

为什么要分档 (面试可讲)
----------------------
v1 的问题: 标签是"GPT-4 vs GPT-3.5 谁赢", 一旦混进别的模型 (Claude-v1),
分类器就能靠"对手是谁"猜标签, 学不到 prompt 本身的难度。
v2 参考 RouteLLM (Ong et al., 2024): 把 53 个模型按实力归成"强档 / 弱档",
标签变成"强档模型赢还是弱档模型赢"。这样路由器只需回答"这条 prompt 值不值得上强档"。

关键决策
--------
D1 评分来自本数据, 不抄官方排行榜: 标签也来自同一批投票, 两者口径一致。
   官方榜还做了"风格控制" (对回答长度、markdown 做回归修正), 我们不做,
   因为我们的标签本身就是没有修正过的原始人类偏好。
D2 拟合数据 = 清洗后主数据 (order=1, 101,058 票)。追问票不参与, 原因同 Step 1。
D3 Bradley-Terry 用逻辑回归实现 (LMArena 官方做法):
   每票一行, 特征向量里 model_a 位置 +1, model_b 位置 -1, 目标 = A 是否赢。
   平局 (tie 和 both_bad) 拆成"A 赢半票 + B 赢半票", 不丢掉也不偏向任何一方。
   系数换算成 Elo 刻度: rating = 1000 + 400 * coef / ln(10)。
   含义: 评分差 400 分 = 胜率比 10:1; 差 100 分 ≈ 高分方胜率 64%。
D4 用 bootstrap (按票重采样 200 次) 给每个模型的评分算 95% 区间,
   分档时可以看到两个模型的差距是不是真的大于噪声。
D5 分档: 对评分做一维 k-means (RouteLLM 也是先聚成多层再挑),
   按评分从高到低给层级编号。最终"哪几层算强、哪几层算弱"由 Yiqing 决定,
   这里只输出候选方案和每个方案下剩多少对战、强档胜率多少。

用法: python -I tier_models.py <v2 根目录>
输出: reports/step2_ratings.csv, reports/step2_tiers.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

root = Path(sys.argv[1])
df = pd.read_parquet(root / "data/arena_clean.parquet",
                     columns=["model_a", "model_b", "winner"])
models = sorted(set(df.model_a) | set(df.model_b))
idx = {m: i for i, m in enumerate(models)}
SCALE, BASE, INIT = 400, 10, 1000


def fit_bt(d: pd.DataFrame) -> np.ndarray:
    """返回每个模型的 Elo 刻度评分。"""
    n = len(d)
    X = np.zeros((n, len(models)))
    X[np.arange(n), d.model_a.map(idx).values] = 1
    X[np.arange(n), d.model_b.map(idx).values] = -1
    a_win = (d.winner == "model_a").values
    b_win = (d.winner == "model_b").values
    tie = ~(a_win | b_win)
    # 平局拆两行各 0.5 权重: 一行 A 赢, 一行 B 赢
    X2 = np.vstack([X, X[tie]])
    y2 = np.concatenate([a_win | tie, np.zeros(tie.sum(), bool)]).astype(int)
    w2 = np.concatenate([np.where(tie, 0.5, 1.0), np.full(tie.sum(), 0.5)])
    lr = LogisticRegression(fit_intercept=False, C=1e6, max_iter=2000, tol=1e-8)
    lr.fit(X2, y2, sample_weight=w2)
    r = SCALE * lr.coef_[0] / np.log(BASE)
    return INIT + r - r.mean()  # 平均分固定在 1000, 只有相对差有意义


rating = fit_bt(df)
rng = np.random.default_rng(42)
boots = np.array([fit_bt(df.sample(len(df), replace=True, random_state=rng.integers(1 << 31)))
                  for _ in range(200)])
n_games = pd.concat([df.model_a, df.model_b]).value_counts()
tab = pd.DataFrame({
    "model": models, "rating": rating.round(1),
    "ci_low": np.percentile(boots, 2.5, axis=0).round(1),
    "ci_high": np.percentile(boots, 97.5, axis=0).round(1),
}).assign(n_games=lambda t: t.model.map(n_games)).sort_values("rating", ascending=False)
tab.to_csv(root / "reports/step2_ratings.csv", index=False)

# 一维 k-means 分层 (多试几个 k, 固定种子)
for k in (4, 5, 6, 8):
    km = KMeans(n_clusters=k, n_init=50, random_state=0).fit(tab[["rating"]])
    order = np.argsort(-km.cluster_centers_[:, 0])  # 层 1 = 评分最高
    rank = {c: i + 1 for i, c in enumerate(order)}
    tab[f"tier_k{k}"] = [rank[c] for c in km.labels_]
tab.to_csv(root / "reports/step2_ratings.csv", index=False)
print(tab.to_string(index=False))
