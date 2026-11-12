"""prompt_axes_report.py — read data/r2_prompt_axes.parquet; write profiles of the four axis ends, raw prompt samples, and win/loss tables by t1/t2 quintile

Usage: python -I prompt_axes_report.py <repo root>
Writes: reports/r2_prompt_axes_tables.md (tables and samples, merged into r2_prompt_axes.md), reports/r2_prompt_axes_ends.csv (200 prompts per end)
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
ends = {"PC1 low end": o1.head(200), "PC1 high end": o1.tail(200), "PC2 low end": o2.head(200), "PC2 high end": o2.tail(200)}


def prof(s):
    return {"code %": 100 * s.is_code.mean(), "math %": 100 * s.math.mean(), "creative writing %": 100 * s.creative.mean(),
            "instruction following %": 100 * s.instr.mean(), "difficulty criteria met (0–7)": s.n_crit.mean(), "hard (≥6 criteria) %": 100 * (s.n_crit >= 6).mean(),
            "length P25 (chars)": s.prompt_len.quantile(.25), "length median": s.prompt_len.median(), "length P75": s.prompt_len.quantile(.75),
            "English %": 100 * (s.language == "en").mean(), "Polish %": 100 * (s.language == "pl").mean(),
            "Chinese (simpl.+trad.) %": 100 * s.language.isin(["zh", "zh-Hant"]).mean(), "Japanese %": 100 * (s.language == "ja").mean(),
            "und (undetermined) %": 100 * (s.language == "und").mean()}


L = ["## A. Profile of 200 prompts at each of the four ends (85,835 distinct dev-set prompts)", ""]
T = pd.DataFrame({**{k: prof(v) for k, v in ends.items()}, "whole dev set": prof(pa)})
L += [T.round(1).to_markdown(), ""]
L += ["Top 6 languages per end: " + "; ".join(f"{k}: " + ", ".join(f"{a} {b}" for a, b in v.language.value_counts().head(6).items())
                               for k, v in ends.items()), ""]


def quint(v, col):
    v = v.assign(q=pd.qcut(v[col], 5, labels=["1 (lowest)", "2", "3", "4", "5 (highest)"]))
    rows = {}
    for q, s in v.groupby("q", observed=True):
        sw = s[s.is_strong_vs_weak]
        rows[q] = {"votes": len(s), "higher-rated win %": 100 * (s.cls == 2).mean(), "tie %": 100 * (s.cls == 1).mean(),
                   "loss %": 100 * (s.cls == 0).mean(), "rating-gap slope (per 100 pts)": np.polyfit(s.rating_gap / 100, s.q_hi, 1)[0],
                   "mean rating gap": s.rating_gap.mean(), "strong-vs-weak votes": len(sw), "strong win %": 100 * (sw.cls == 2).mean(),
                   "strong-vs-weak tie %": 100 * (sw.cls == 1).mean(), "weak win %": 100 * (sw.cls == 0).mean(),
                   "hard %": 100 * (s.n_crit >= 6).mean(), "length median": s.prompt_len.median()}
    return pd.DataFrame(rows).T


ins = fl[(fl.context == "final") & (fl.role == "train")].merge(pa[["group_id", "t1", "t2", "n_crit", "prompt_len"]], on="group_id")
oof = fl[fl.context.str.startswith("cv") & (fl.role == "eval")].merge(pa[["group_id", "t1_oof", "n_crit", "prompt_len"]], on="group_id")
fmt = lambda df: df.astype(float).round(2).to_markdown()
L += ["## B. By t1 quintile (in-sample: the final model scores all dev-set votes)", "", fmt(quint(ins, "t1")), "",
      "## C. By t1 quintile (out-of-fold: each fold's model scores only eval votes it never saw)", "", fmt(quint(oof, "t1_oof")), "",
      "## D. By t2 quintile (in-sample)", "", fmt(quint(ins, "t2")), ""]
L += ["## E. Raw prompt samples (10 drawn at random from the 200 at each end, seed 2026, cut at 300 characters)", ""]
for k, v in ends.items():
    L += [f"### {k}", ""]
    for i, r in enumerate(v.sample(10, random_state=2026).itertuples(), 1):
        t = re.sub(r"\s+", " ", r.prompt).strip()
        t = t[:300] + (" …" if len(t) > 300 else "")
        L.append(f"{i}. `{r.language} · {r.prompt_len} chars · t1 = {r.t1:.2f} · t2 = {r.t2:.2f}`  {t.replace('|', '¦')}")
    L.append("")
(root / "reports/r2_prompt_axes_tables.md").write_text("\n".join(L))
pd.concat([v.assign(end=k) for k, v in ends.items()])[
    ["end", "group_id", "t1", "t2", "t1_oof", "language", "prompt_len", "is_code", "math", "creative", "instr", "n_crit", "prompt"]
].to_csv(root / "reports/r2_prompt_axes_ends.csv", index=False)
print("ok")
