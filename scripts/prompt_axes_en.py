"""prompt_axes_en.py — language confound check: redo the strong-vs-weak win/tie/loss table by PC1 (t1) quintile on English votes only, next to the all-language version

English = langdetect result "en" from Step 1 (the same language column R3 uses); a version using the dataset's own
language column = en is added as a check. Quintile edges come from the all-language votes' t1 quantiles (same ruler
for both); a version with quintiles re-cut inside English is added too.
The main table uses out-of-fold t1 (each fold's model scores only eval votes it never saw). Dev set only.
Usage: python -I prompt_axes_en.py <repo root>    Writes: reports/r2_pc1_english_check.md
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
Q = ["1 (lowest)", "2", "3", "4", "5 (highest)"]


def tab(v, col, edges=None):
    q = pd.cut(v[col], edges, labels=Q, include_lowest=True) if edges is not None else pd.qcut(v[col], 5, labels=Q)
    g = v.assign(q=q).groupby("q", observed=True)
    return pd.DataFrame({"votes": g.size(), "strong win %": g.cls.apply(lambda c: 100 * (c == 2).mean()),
                         "tie %": g.cls.apply(lambda c: 100 * (c == 1).mean()),
                         "weak win %": g.cls.apply(lambda c: 100 * (c == 0).mean())})


def edges(v, col):   # quantiles of the all-language votes, ends widened to infinity
    e = np.quantile(v[col], np.linspace(0, 1, 6))
    e[0], e[-1] = -np.inf, np.inf
    return e


L = ["# PC1 language confound check: quintile table redone on English votes only", "",
     "Strong-vs-weak votes (dev set). Lower t1 = the strong tier has more of an edge (open-ended advice end); higher t1 = closer to the code and definite-answer end.", ""]
for name, v, col in [("Out-of-fold t1 (main table)", oof, "t1_oof"), ("In-sample t1", ins, "t1")]:
    e = edges(v, col)
    en = v[v.langdetect_lang == "en"]
    en2 = v[v.language == "en"]
    parts = {"all languages": tab(v, col, e), "English (langdetect)": tab(en, col, e),
             "English (dataset language column)": tab(en2, col, e), "English, quintiles re-cut": tab(en, col)}
    T = pd.concat(parts, axis=1).round(1)
    T.columns = [f"{a} · {b}" for a, b in T.columns]
    L += [f"## {name}", "", f"English share of votes {len(en) / len(v):.1%} (langdetect), {len(en2) / len(v):.1%} (dataset language column).", "",
          T.T.to_markdown(), ""]
(root / "reports/r2_pc1_english_check.md").write_text("\n".join(L))
print("\n".join(L))
