"""make_labels.py — Step 2 (下半) + Step 3: 打标签、软标签、分组键

数据结构 (Yiqing 2026-11-09 确认, PM 视角: "任意两个模型之间怎么路由")
-------------------------------------------------------------------
1. 全量表 data/arena_labeled.parquet: 清洗后的全部 101,058 票, 每行带两个模型的评分/层级。
   以后训练"带模型身份"的路由器 (RouteLLM 的矩阵分解路由器) 用它, 上线时代入任意两个模型。
2. 评估视图 data/arena_strong_vs_weak.parquet: 只保留"强档 vs 弱档"的票。
   PGR/APGR 必须固定一对强弱才有定义 (RouteLLM 也是这样评估的), 所以评估在这张表上做。
   Step 4 的折号在全量表上按 group_id 生成, 视图直接继承, 保证两张表划分一致。

!! 2026-11-09 Yiqing 补充 (评分泄漏): 本脚本里的评分/层级/y_hi/y 用的是"全部票"拟合的 BT,
!! 测试集的票也参与了拟合, 所以这些列只用于 Step 3 的描述性统计 (重复 prompt、首轮 vs 追问),
!! 不能用于训练或评估。训练/评估用的标签由 fold_ratings.py 按每折独立拟合的评分重新计算。
!! 本脚本对后续步骤真正有用的产出是 group_id、is_tie、tie_type。

关键决策
--------
D1 分档边界: 暂用方案 A (k=5 层 1-2 为强, 层 5 为弱), 等 Yiqing 确认; 改边界只需改 STRONG/WEAK。
   出场 < 300 次的模型评分区间太宽, 不进任何档 (tier 记为空)。
D2 标签 (评估视图): y = 1 强档胜, 0 弱档胜, 平局 y 为空并 is_tie = True。
   平局不归 0: "两个都行"和"弱的更好"对路由的含义不同, 混在一起会低估弱档。
   tie_type 保留原始区分 (tie / both_bad), 以后可以分开研究。
D3 全量表另给一个通用标签 y_hi: 评分高的一方胜 = 1, 低的一方胜 = 0, 平局为空。
   这是"任意两模型"视角下的同一个概念。
D3b 路由目标 (Yiqing 2026-11-09, 成本角度): 用户偏好一样时倾向小模型。
   y_route = 1 只在强档 (或评分高一方) 明确胜出时; 弱档胜或平局 (tie / both_bad) 都 = 0, 即"发弱档就够"。
   数据层仍保留 is_tie / tie_type / y, 不丢信息; y_route 是在它们之上的产品决策规则。
   评估里同样体现: 平局时强弱质量相同 (各 0.5), 发弱档质量不降、成本更低, 所以曲线自然偏向弱档。
D4 分组键: prompt 去首尾空白、连续空白压成一个空格、转小写, 再取 sha1 前 16 位。
   同一 prompt 的所有票必须落在同一折, 否则模型在训练集见过答案, 测试分数虚高。
D5 软标签: 同一 group 内强档胜率, 平局算 0.5 (两边各半)。只出现一次的就是 0 / 1 / 0.5。
   n_votes 记录票数, 票越多软标签越可信。
   重复 prompt 上不同用户的投票不一致, 就是标签噪声上限的直接证据 (见报告)。

用法: python -I make_labels.py <v2 根目录>
"""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
TIER_COL, STRONG, WEAK, MIN_GAMES = "tier_k5", {1, 2}, {5}, 300

rt = pd.read_csv(root / "reports/step2_ratings.csv").set_index("model")
rating = rt.rating
tier = rt[TIER_COL].where(rt.n_games >= MIN_GAMES)  # 出场太少 -> NaN


def norm(p: str) -> str:
    return re.sub(r"\s+", " ", p.strip()).lower()


def gid(p: str) -> str:
    return hashlib.sha1(norm(p).encode("utf-8")).hexdigest()[:16]


def label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["group_id"] = df.prompt.map(gid)
    df["rating_a"], df["rating_b"] = df.model_a.map(rating), df.model_b.map(rating)
    df["tier_a"], df["tier_b"] = df.model_a.map(tier), df.model_b.map(tier)
    df["is_tie"] = df.winner.isin(["tie", "both_bad"])
    df["tie_type"] = df.winner.where(df.is_tie)
    a_hi = df.rating_a >= df.rating_b
    hi_won = np.where(a_hi, df.winner == "model_a", df.winner == "model_b")
    df["y_hi"] = np.where(df.is_tie, np.nan, hi_won.astype(float))
    df["y_hi_route"] = np.where(df.is_tie, 0.0, hi_won.astype(float))  # 平局 -> 发低分方
    df["rating_gap"] = (df.rating_a - df.rating_b).abs()
    sa = df.tier_a.isin(STRONG) & df.tier_b.isin(WEAK)
    sb = df.tier_b.isin(STRONG) & df.tier_a.isin(WEAK)
    df["is_strong_vs_weak"] = sa | sb
    df["strong_model"] = np.where(sa, df.model_a, np.where(sb, df.model_b, None))
    df["weak_model"] = np.where(sa, df.model_b, np.where(sb, df.model_a, None))
    strong_won = np.where(sa, df.winner == "model_a", df.winner == "model_b")
    df["y"] = np.where(df.is_strong_vs_weak & ~df.is_tie, strong_won.astype(float), np.nan)
    df["y_route"] = np.where(df.is_strong_vs_weak, (strong_won & ~df.is_tie).astype(float), np.nan)
    return df


full = label(pd.read_parquet(root / "data/arena_clean.parquet"))
follow = label(pd.read_parquet(root / "data/arena_followup_clean.parquet"))

# 软标签: 全量表上按 group 算"评分高一方胜率"; 视图上按 group 算"强档胜率"
full["vote_hi"] = full.y_hi.fillna(0.5)
g = full.groupby("group_id").vote_hi
full["soft_label_hi"], full["n_votes_all"] = g.transform("mean"), g.transform("size")

sw = full[full.is_strong_vs_weak].copy()
sw["vote"] = sw.y.fillna(0.5)
g = sw.groupby("group_id").vote
sw["soft_label"], sw["n_votes"] = g.transform("mean"), g.transform("size")
# 路由用软标签: 同 prompt 下"强档明确胜出"的比例 (平局算 0)
sw["soft_label_route"] = sw.groupby("group_id").y_route.transform("mean")
full["soft_label_hi_route"] = full.groupby("group_id").y_hi_route.transform("mean")

full.drop(columns="vote_hi").to_parquet(root / "data/arena_labeled.parquet", index=False)
# (评估视图不在这里输出: 正式版由 fold_ratings.py 用开发集评分 + 固定名单生成)


# ---------- 报告 ----------
def three(d, ycol):
    return pd.Series({"票数": len(d), "高/强 胜": (d[ycol] == 1).mean(), "低/弱 胜": (d[ycol] == 0).mean(),
                      "平局": d.is_tie.mean()}).round(3)


def dup_stats(d, vcol, name):
    gr = d.groupby("group_id")[vcol].agg(["size", "mean"])
    rep = gr[gr["size"] >= 2]
    dec = d[~d.is_tie & d.group_id.isin(rep.index)]
    # 噪声上限: 在重复 prompt 的非平局票上, 即使知道每个 prompt 的真实多数, 最多答对多少
    gd = dec.groupby("group_id")[vcol].agg(["size", "mean"])
    ceil = (gd["size"] * np.maximum(gd["mean"], 1 - gd["mean"])).sum() / gd["size"].sum()
    mixed = ((gd["mean"] > 0) & (gd["mean"] < 1) & (gd["size"] >= 2)).mean()
    bins = pd.cut(rep["mean"], [-.01, .001, .25, .499, .5, .75, .999, 1], labels=[
        "0 (弱全胜)", "(0,0.25]", "(0.25,0.5)", "0.5", "(0.5,0.75]", "(0.75,1)", "1 (强全胜)"])
    lines = [f"### {name}", "",
             f"- 不同 prompt 数: {gr.shape[0]}; 出现 ≥2 次的: {rep.shape[0]} 个, 覆盖 {int(rep['size'].sum())} 票 "
             f"({rep['size'].sum() / len(d):.1%})",
             f"- 重复次数分布: {rep['size'].value_counts().sort_index().head(6).to_dict()} (最多 {int(rep['size'].max())})",
             f"- 重复 prompt 中, 至少 2 张非平局票且结果不一致的比例: {mixed:.1%}",
             f"- 噪声上限: 重复 prompt 的非平局票上, 每个 prompt 都猜它的多数结果, 准确率最多 {ceil:.1%}",
             "", "重复 prompt 的胜率 (soft label) 分布:", "",
             bins.value_counts().sort_index().to_frame("prompt 数").to_markdown(), ""]
    return lines


full["vote_hi"] = full.y_hi.fillna(0.5)
sw["vote"] = sw.y.fillna(0.5)
fsw = follow[follow.is_strong_vs_weak]
tiers = rt.assign(tier=tier).dropna(subset=["tier"]).groupby("tier").agg(
    模型数=("rating", "size"), 最低分=("rating", "min"), 最高分=("rating", "max"))

md = ["# Step 2/3 标签与分组结果", "",
      f"分档 (暂定方案 A): 强档 = {TIER_COL} 层 {sorted(STRONG)}, 弱档 = 层 {sorted(WEAK)}; 出场 < {MIN_GAMES} 次不分档。", "",
      tiers.to_markdown(), "",
      "## 标签三类比例", "",
      pd.DataFrame({"全量表 (y_hi: 评分高的一方)": three(full, "y_hi"),
                    "评估视图 (y: 强档 vs 弱档)": three(sw, "y")}).to_markdown(), "",
      "## 路由目标 y_route (平局归弱档, 成本优先)", "",
      pd.DataFrame({"全量表 y_hi_route": full.y_hi_route.value_counts(normalize=True).round(3),
                    "评估视图 y_route": sw.y_route.value_counts(normalize=True).round(3)}).to_markdown(), "",
      "## 重复 prompt (标签噪声的第一个证据)", ""]
md += dup_stats(full, "vote_hi", "全量表 (按评分高一方)")
md += dup_stats(sw, "vote", "评估视图 (强档 vs 弱档)")
md += ["## 首轮 vs 追问轮: 弱档是不是在追问时赢得更多?", "",
       pd.DataFrame({"首轮 order=1": three(sw, "y"), "追问 order>1": three(fsw, "y")}).to_markdown(), "",
       "全量表口径 (评分低一方的胜率; 两组平均评分差接近才可比):", "",
       pd.DataFrame({k: pd.Series({"票数": len(d), "低分方胜": (d.y_hi == 0).mean(), "平局": d.is_tie.mean(),
                                    "平均评分差": d.rating_gap.mean()}).round(3)
                     for k, d in [("首轮", full), ("追问", follow)]}).to_markdown()]
(root / "reports/step3_labels.md").write_text("\n".join(md))
print("\n".join(md))
