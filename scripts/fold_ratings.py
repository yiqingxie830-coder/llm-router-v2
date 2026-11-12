"""fold_ratings.py — Step 4 (下半): 每折独立拟合 Bradley-Terry, 重算标签与分层

为什么 (Yiqing 2026-11-09 提出的"评分泄漏")
-----------------------------------------
"谁是高分方""谁是强档"都来自 BT 评分。如果评分用全部票拟合, 测试集的票也参与了,
等于测试答案的一部分提前流进了标签定义和特征 (评分差)。
所以每一个"训练/评估情境 (context)"都只用它自己训练部分的票拟合评分:
  - cv0..cv4: 用开发集里除第 k 折之外的票拟合, 给第 k 折 (评估) 和其余 4 折 (训练) 打标签
  - final   : 用全部开发集 (90%) 拟合, 给测试集 (评估) 和开发集 (训练) 打标签
每个情境里, 高分方、y_hi、y_hi_route、评分差、层级、强弱视图成员、y / y_route、软标签全部重算。

其他决策
--------
D1 BT 拟合方法与 tier_models.py 完全相同 (逻辑回归, 平局拆半票, Elo 刻度, 平均 1000)。
D2 强弱档 = 固定名单 (Yiqing 2026-11-09 决定), 不再每折跑 k-means:
   - 原因: 一维 k-means (以及 RouteLLM 的动态规划分层) 在 cv0 上把最高两层合并,
     "层 1-2" 从 10 个模型跳到 21 个, 各折的口径 ② 票集合不同, 分数不能平均。
   - 做法: 只用 90% 开发集拟合一次 BT (= final 情境), 画名次–分数图
     (reports/step4_rank_rating.png), 在明显断层处切:
       第 10 / 11 名之间 21.4 分 -> 前 10 名为强档
       第 40 / 41 名之间 24.6 分 -> 后 11 名为弱档
       第 11-40 名 (30 个模型) 为中间带, 不进口径 ②
     第 11-40 名内部最大分差只有 9.1 分, 没有其他明显断层。
   - 稳定性: 两条切线两侧的模型, 在 6 个情境下的分数区间都不重叠, 所以名单对任何一折都成立。
   - 名单只用开发集决定, 测试集没有参与。
D3 某模型在某情境的训练票里没出现 (只有极少数冷门模型): 评分记为 1000 (平均),
   不进任何档; n_games 列可以看出来。
D4 本脚本只生成标签, 不训练任何路由器。

用法: python -I fold_ratings.py <v2 根目录>
输出:
  data/fold_labels.parquet           长表: 每个 (context, id) 一行, 含该情境下的全部标签
  data/arena_strong_vs_weak.parquet  final 情境下的强弱评估视图 (测试 + 开发), 带 fold
  reports/step4_ratings_by_context.csv, reports/step4_fold_labels.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

root = Path(sys.argv[1])
SCALE, BASE, INIT = 400, 10, 1000
STRONG_MODELS = ['gemini-2.5-pro', 'gemini-2.5-pro-preview-03-25', 'grok-4-0709', 'o3-2025-04-16', 'gemini-2.5-pro-preview-05-06', 'deepseek-r1-0528', 'chatgpt-4o-latest-20250326', 'grok-3-preview-02-24', 'gemini-2.5-flash', 'llama-4-maverick-03-26-experimental']
WEAK_MODELS = ['llama-4-scout-17b-16e-instruct', 'llama-4-maverick-17b-128e-instruct', 'claude-3-7-sonnet-20250219-thinking-32k', 'claude-3-5-sonnet-20241022', 'claude-3-7-sonnet-20250219', 'amazon.nova-pro-v1:0', 'llama-3.3-70b-instruct', 'mistral-small-3.1-24b-instruct-2503', 'gpt-4o-mini-2024-07-18', 'magistral-medium-2506', 'claude-3-5-haiku-20241022']

df = pd.read_parquet(root / "data/arena_labeled.parquet")
keep_cols = ["id", "group_id", "split", "fold", "model_a", "model_b", "winner", "is_tie",
             "tie_type", "is_en", "language", "prompt_len"]
df = df[keep_cols]
models = sorted(set(df.model_a) | set(df.model_b))
idx = {m: i for i, m in enumerate(models)}


def fit_bt(d):
    n = len(d)
    X = np.zeros((n, len(models)))
    X[np.arange(n), d.model_a.map(idx).values] = 1
    X[np.arange(n), d.model_b.map(idx).values] = -1
    a_win, b_win = (d.winner == "model_a").values, (d.winner == "model_b").values
    tie = ~(a_win | b_win)
    X2 = np.vstack([X, X[tie]])
    y2 = np.concatenate([a_win | tie, np.zeros(tie.sum(), bool)]).astype(int)
    w2 = np.concatenate([np.where(tie, 0.5, 1.0), np.full(tie.sum(), 0.5)])
    lr = LogisticRegression(fit_intercept=False, C=1e6, max_iter=3000, tol=1e-8)
    lr.fit(X2, y2, sample_weight=w2)
    r = SCALE * lr.coef_[0] / np.log(BASE)
    games = pd.concat([d.model_a, d.model_b]).value_counts().reindex(models).fillna(0).astype(int)
    seen = games.values > 0
    r = INIT + r - r[seen].mean()
    r[~seen] = INIT  # D3
    return pd.DataFrame({"model": models, "rating": r, "n_games": games.values})


def tiers(rt):
    """固定名单: strong / weak / mid (中间带或出场太少的模型)。"""
    return rt.model.map(lambda m: "strong" if m in STRONG_MODELS else "weak" if m in WEAK_MODELS else "mid")


contexts = {f"cv{k}": (df.split.eq("dev") & df.fold.ne(k), df.fold.eq(k)) for k in range(5)}
contexts["final"] = (df.split.eq("dev"), df.split.eq("test"))

rows, rtabs = [], []
for ctx, (tr_mask, ev_mask) in contexts.items():
    train = df[tr_mask]
    rt = fit_bt(train)
    rt["tier"] = tiers(rt).values
    rtabs.append(rt.assign(context=ctx))
    r, n, t = (rt.set_index("model")[c] for c in ("rating", "n_games", "tier"))

    d = df[tr_mask | ev_mask].copy()
    d["context"] = ctx
    d["role"] = np.where(ev_mask[d.index], "eval", "train")
    d["rating_a"], d["rating_b"] = d.model_a.map(r), d.model_b.map(r)
    d["n_games_a"], d["n_games_b"] = d.model_a.map(n), d.model_b.map(n)
    d["tier_a"], d["tier_b"] = d.model_a.map(t), d.model_b.map(t)
    a_hi = d.rating_a >= d.rating_b
    d["hi_model"] = np.where(a_hi, d.model_a, d.model_b)
    d["lo_model"] = np.where(a_hi, d.model_b, d.model_a)
    d["rating_gap"] = (d.rating_a - d.rating_b).abs()
    hi_won = np.where(a_hi, d.winner == "model_a", d.winner == "model_b")
    d["y_hi"] = np.where(d.is_tie, np.nan, hi_won.astype(float))
    d["y_hi_route"] = np.where(d.is_tie, 0.0, hi_won.astype(float))
    # 评估时用的质量分: 发给高分方的得分 (高分方胜 1 / 平局 0.5 / 输 0)
    d["q_hi"] = np.where(d.is_tie, 0.5, hi_won.astype(float))
    sa = d.tier_a.eq("strong") & d.tier_b.eq("weak")
    sb = d.tier_b.eq("strong") & d.tier_a.eq("weak")
    # 保险: 固定名单里的强档必须在本情境评分上高于弱档, 否则"强档 = 高分方"不成立
    assert r[STRONG_MODELS].min() > r[WEAK_MODELS].max(), ctx
    d["is_strong_vs_weak"] = sa | sb
    strong_won = np.where(sa, d.winner == "model_a", d.winner == "model_b")
    d["y"] = np.where(d.is_strong_vs_weak & ~d.is_tie, strong_won.astype(float), np.nan)
    d["y_route"] = np.where(d.is_strong_vs_weak, (strong_won & ~d.is_tie).astype(float), np.nan)
    # 软标签: 同一情境内按 group 聚合 (group 不跨折, 所以不会混入评估票)
    g = d.assign(v=d.y_hi.fillna(0.5)).groupby("group_id")
    d["soft_label_hi"] = g.v.transform("mean")
    d["soft_label_hi_route"] = g.y_hi_route.transform("mean")
    d["n_votes"] = g.v.transform("size")
    sw = d[d.is_strong_vs_weak].assign(v=lambda x: x.y.fillna(0.5)).groupby("group_id")
    d.loc[d.is_strong_vs_weak, "soft_label"] = sw.v.transform("mean")
    d.loc[d.is_strong_vs_weak, "soft_label_route"] = sw.y_route.transform("mean")
    d.loc[d.is_strong_vs_weak, "n_votes_sw"] = sw.v.transform("size")
    rows.append(d)
    print(ctx, "train", len(train), "labeled", len(d), "sw", int(d.is_strong_vs_weak.sum()))

fl = pd.concat(rows, ignore_index=True)
fl.to_parquet(root / "data/fold_labels.parquet", index=False)
rts = pd.concat(rtabs, ignore_index=True)
rts.to_csv(root / "reports/step4_ratings_by_context.csv", index=False)

prompts = pd.read_parquet(root / "data/arena_labeled.parquet", columns=["id", "prompt"])
fin = fl[(fl.context == "final") & fl.is_strong_vs_weak].merge(prompts, on="id")
fin.to_parquet(root / "data/arena_strong_vs_weak.parquet", index=False)

# ---------- 报告: 翻转统计 ----------
hi_n = fl.groupby("id").hi_model.nunique()
gap_full = pd.read_parquet(root / "data/arena_labeled.parquet", columns=["id", "rating_gap"]).set_index("id").rating_gap
flipped = hi_n[hi_n > 1].index
sw_n = fl.groupby("id").is_strong_vs_weak.agg(["min", "max"])
sw_changed = (sw_n["min"] != sw_n["max"]).sum()
wide = rts.pivot(index="model", columns="context", values="rating")
gap_bins = pd.cut(gap_full.loc[flipped], [0, 5, 10, 20, 50, 1e9], include_lowest=True).value_counts().sort_index()

md = ["# Step 4 每折独立拟合 BT 的结果", "",
      "## 各情境的票数与强弱视图规模", "",
      fl.groupby(["context", "role"]).agg(票数=("id", "size"), 强弱视图=("is_strong_vs_weak", "sum"),
                                          高分方明确胜=("y_hi_route", "mean")).round(3).to_markdown(), "",
      "## 高分方翻转", "",
      f"- 一共 {hi_n.size} 票, 在不同情境之间高分方发生翻转的: **{len(flipped)} 票 ({len(flipped) / hi_n.size:.2%})**",
      f"- 涉及的模型对: {fl[fl.id.isin(flipped)].drop_duplicates('id').apply(lambda x: ' vs '.join(sorted([x.model_a, x.model_b])), axis=1).nunique()} 对",
      "- 翻转票按全量评分差分布 (翻转只发生在两个模型本来就很接近的地方):", "",
      gap_bins.to_frame("票数").to_markdown(), "",
      f"## 强弱视图成员变化: {sw_changed} 票在某些情境属于强弱视图、在另一些不属于 (固定名单, 应为 0)", "",
      f"各情境中强档最低分 vs 弱档最高分: " + ", ".join(
          f"{c}: {g[g.model.isin(STRONG_MODELS)].rating.min():.0f} > {g[g.model.isin(WEAK_MODELS)].rating.max():.0f}"
          for c, g in rts.groupby("context")), "",
      "## 各情境评分的最大差异 (同一模型, max - min)", "",
      (wide.max(axis=1) - wide.min(axis=1)).describe().round(1).to_frame("分").to_markdown()]
(root / "reports/step4_fold_labels.md").write_text("\n".join(md))
print("\n".join(md))
