"""prompt_axes_report.py — 读 data/r2_prompt_axes.parquet, 出两轴四端的画像表、原文样例、按 t1/t2 五分位的胜负表

用法: python -I prompt_axes_report.py <v2 根目录>
产出: reports/r2_prompt_axes_tables.md (表与样例, 已并入 r2_prompt_axes.md), reports/r2_prompt_axes_ends.csv (四端各 200 条)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
pa = pd.read_parquet(root / "data/r2_prompt_axes.parquet")
fl = pd.read_parquet(root / "data/fold_labels.parquet",
                     columns=["context", "id", "role", "group_id", "rating_gap", "y_hi_route", "is_tie", "is_strong_vs_weak", "q_hi"])
fl["cls"] = np.where(fl.y_hi_route == 1, 2, np.where(fl.is_tie, 1, 0))
o1, o2 = pa.sort_values("t1"), pa.sort_values("t2")
ends = {"PC1 低端": o1.head(200), "PC1 高端": o1.tail(200), "PC2 低端": o2.head(200), "PC2 高端": o2.tail(200)}


def prof(s):
    return {"代码 %": 100 * s.is_code.mean(), "数学 %": 100 * s.math.mean(), "创意写作 %": 100 * s.creative.mean(),
            "指令遵循 %": 100 * s.instr.mean(), "难度标准数 (0–7)": s.n_crit.mean(), "硬题 (≥6 条) %": 100 * (s.n_crit >= 6).mean(),
            "长度 P25 (字符)": s.prompt_len.quantile(.25), "长度中位": s.prompt_len.median(), "长度 P75": s.prompt_len.quantile(.75),
            "英文 %": 100 * (s.language == "en").mean(), "波兰语 %": 100 * (s.language == "pl").mean(),
            "中文 (简+繁) %": 100 * s.language.isin(["zh", "zh-Hant"]).mean(), "日文 %": 100 * (s.language == "ja").mean(),
            "und (未判定) %": 100 * (s.language == "und").mean()}


L = ["## A. 四端各 200 条的画像（开发集 85,835 个不同 prompt）", ""]
T = pd.DataFrame({**{k: prof(v) for k, v in ends.items()}, "全部开发集": prof(pa)})
L += [T.round(1).to_markdown(), ""]
L += ["各端语言前 6: " + "; ".join(f"{k}: " + ", ".join(f"{a} {b}" for a, b in v.language.value_counts().head(6).items())
                               for k, v in ends.items()), ""]


def quint(v, col):
    v = v.assign(q=pd.qcut(v[col], 5, labels=["1 (最低)", "2", "3", "4", "5 (最高)"]))
    rows = {}
    for q, s in v.groupby("q", observed=True):
        sw = s[s.is_strong_vs_weak]
        rows[q] = {"票数": len(s), "高分方 赢 %": 100 * (s.cls == 2).mean(), "平 %": 100 * (s.cls == 1).mean(),
                   "输 %": 100 * (s.cls == 0).mean(), "评分差斜率 (每 100 分)": np.polyfit(s.rating_gap / 100, s.q_hi, 1)[0],
                   "平均评分差": s.rating_gap.mean(), "强弱档票数": len(sw), "强档 赢 %": 100 * (sw.cls == 2).mean(),
                   "强弱 平 %": 100 * (sw.cls == 1).mean(), "弱档 赢 %": 100 * (sw.cls == 0).mean(),
                   "硬题 %": 100 * (s.n_crit >= 6).mean(), "长度中位": s.prompt_len.median()}
    return pd.DataFrame(rows).T


ins = fl[(fl.context == "final") & (fl.role == "train")].merge(pa[["group_id", "t1", "t2", "n_crit", "prompt_len"]], on="group_id")
oof = fl[fl.context.str.startswith("cv") & (fl.role == "eval")].merge(pa[["group_id", "t1_oof", "n_crit", "prompt_len"]], on="group_id")
fmt = lambda df: df.astype(float).round(2).to_markdown()
L += ["## B. 按 t1 五分位（训练集内：最终模型给全部开发集票打分）", "", fmt(quint(ins, "t1")), "",
      "## C. 按 t1 五分位（折外：每折模型只给自己没见过的 eval 票打分）", "", fmt(quint(oof, "t1_oof")), "",
      "## D. 按 t2 五分位（训练集内）", "", fmt(quint(ins, "t2")), ""]
L += ["## E. 原文样例（每端从 200 条里随机抽 10 条，种子 2026，超过 300 字符截断）", ""]
for k, v in ends.items():
    L += [f"### {k}", ""]
    for i, r in enumerate(v.sample(10, random_state=2026).itertuples(), 1):
        t = re.sub(r"\s+", " ", r.prompt).strip()
        t = t[:300] + (" …" if len(t) > 300 else "")
        L.append(f"{i}. `{r.language} · {r.prompt_len} 字符 · t1 = {r.t1:.2f} · t2 = {r.t2:.2f}`  {t.replace('|', '¦')}")
    L.append("")
(root / "reports/r2_prompt_axes_tables.md").write_text("\n".join(L))
pd.concat([v.assign(end=k) for k, v in ends.items()])[
    ["end", "group_id", "t1", "t2", "t1_oof", "language", "prompt_len", "is_code", "math", "creative", "instr", "n_crit", "prompt"]
].to_csv(root / "reports/r2_prompt_axes_ends.csv", index=False)
print("ok")
