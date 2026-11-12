"""prompt_axes_en.py — 语言混淆检查: 只用英文票重做 PC1 (t1) 五分位的强弱档 强胜/平/弱胜 表, 与全语言版并列

英文 = Step 1 的 langdetect 结果为 en (和 R3 用同一语言列); 另附数据集自带 language 列 = en 的版本作核对。
五分位边界用全语言票的 t1 分位点 (两版同一把尺子); 另附英文内部重新切五分位的版本。
主表用折外 t1 (每折模型只给没见过的 eval 票打分)。只用开发集。
用法: python -I prompt_axes_en.py <v2 根目录>    产出: reports/r2_pc1_english_check.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
pa = pd.read_parquet(root / "data/r2_prompt_axes.parquet", columns=["group_id", "t1", "t1_oof", "language"])
ld = pd.read_parquet(root / "data/arena_labeled.parquet", columns=["group_id", "langdetect_lang", "split"])
ld = ld[ld.split != "test"].drop_duplicates("group_id")[["group_id", "langdetect_lang"]]
pa = pa.merge(ld, on="group_id")
fl = pd.read_parquet(root / "data/fold_labels.parquet",
                     columns=["context", "id", "role", "group_id", "y_hi_route", "is_tie", "is_strong_vs_weak"])
fl["cls"] = np.where(fl.y_hi_route == 1, 2, np.where(fl.is_tie, 1, 0))
oof = fl[fl.context.str.startswith("cv") & (fl.role == "eval") & fl.is_strong_vs_weak].merge(pa, on="group_id")
ins = fl[(fl.context == "final") & (fl.role == "train") & fl.is_strong_vs_weak].merge(pa, on="group_id")
Q = ["1 (最低)", "2", "3", "4", "5 (最高)"]


def tab(v, col, edges=None):
    q = pd.cut(v[col], edges, labels=Q, include_lowest=True) if edges is not None else pd.qcut(v[col], 5, labels=Q)
    g = v.assign(q=q).groupby("q", observed=True)
    return pd.DataFrame({"票数": g.size(), "强档 赢 %": g.cls.apply(lambda c: 100 * (c == 2).mean()),
                         "平 %": g.cls.apply(lambda c: 100 * (c == 1).mean()),
                         "弱档 赢 %": g.cls.apply(lambda c: 100 * (c == 0).mean())})


def edges(v, col):   # 全语言票的分位点, 两端放宽到无穷
    e = np.quantile(v[col], np.linspace(0, 1, 6))
    e[0], e[-1] = -np.inf, np.inf
    return e


L = ["# PC1 语言混淆检查：只用英文票重做五分位表", "",
     "强弱档对战票（开发集）。t1 越低，强档越占优（开放式咨询端）；t1 越高越接近代码与确定答案题端。", ""]
for name, v, col in [("折外 t1（主表）", oof, "t1_oof"), ("训练集内 t1", ins, "t1")]:
    e = edges(v, col)
    en = v[v.langdetect_lang == "en"]
    en2 = v[v.language == "en"]
    parts = {"全语言": tab(v, col, e), "英文 (langdetect)": tab(en, col, e),
             "英文 (数据集语言列)": tab(en2, col, e), "英文内部重切五分位": tab(en, col)}
    T = pd.concat(parts, axis=1).round(1)
    T.columns = [f"{a} · {b}" for a, b in T.columns]
    L += [f"## {name}", "", f"英文票占比 {len(en) / len(v):.1%}（langdetect），{len(en2) / len(v):.1%}（数据集语言列）。", "",
          T.T.to_markdown(), ""]
(root / "reports/r2_pc1_english_check.md").write_text("\n".join(L))
print("\n".join(L))
