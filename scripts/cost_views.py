"""cost_views.py — Stage 2 Step 1: vote counts for the dollar view (items 1 and 2 of the 2026-11-09 decisions)

1. Each dev-set vote falls in one of three classes by "typical cost of the higher-rated model vs the lower-rated one":
   higher-rated costs more (enters the dollar curves) / higher-rated costs less or same ("the cheap model is better,
   no routing needed", share only) / no price. Uses each fold's own higher-rated side; eval votes of all folds combined.
2. View ② main dollar case: all strong-vs-weak pairs where the strong model is pricier; reports votes, models
   involved and the distribution of price ratios.

Usage: python -I cost_views.py <repo root>   Writes reports/stage2_step1_cost_views.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
fl = pd.read_parquet(root / "data/fold_labels.parquet",
                     columns=["context", "id", "role", "rating_gap", "is_strong_vs_weak", "hi_model", "lo_model",
                              "model_a", "y_hi_route", "is_tie"])
ev = fl[(fl.role == "eval") & fl.context.str.startswith("cv")].reset_index(drop=True)
mc = pd.read_csv(root / "data/model_cost.csv").set_index("model")
ev["typ_hi"] = ev.hi_model.map(mc.typical_cost)
ev["typ_lo"] = ev.lo_model.map(mc.typical_cost)
ev["cls"] = np.select([ev.typ_hi.isna() | ev.typ_lo.isna(), ev.typ_hi > ev.typ_lo],
                      ["no price", "higher-rated costs more"], "higher-rated costs less or same")
ev["ratio"] = ev.typ_hi / ev.typ_lo
ev["bucket"] = pd.cut(ev.rating_gap, [0, 50, 150, np.inf], right=False, labels=["<50", "50–150", ">150"])

L = ["# Stage 2 Step 1: vote counts for the dollar view (dev set 5-fold, eval votes of all folds combined)", ""]


def share_table(d, title):
    t = d.groupby("cls").agg(votes=("id", "size"), higher_rated_clear_win=("y_hi_route", "mean"), tie_rate=("is_tie", "mean"))
    t["share"] = t.votes / t.votes.sum()
    t = t[["votes", "share", "higher_rated_clear_win", "tie_rate"]]
    return [f"## {title} ({len(d)} votes)", "", t.to_markdown(floatfmt=".3f"), ""]


L += share_table(ev, "① all model pairs")
for b in ["<50", "50–150", ">150"]:
    L += share_table(ev[ev.bucket == b], f"① gap {b}")
sw = ev[ev.is_strong_vs_weak]
L += share_table(sw, "② strong vs weak")

# which model pairs are "the cheap model is better"
ch = ev[ev.cls == "higher-rated costs less or same"]
top = ch.groupby(["hi_model", "lo_model"]).size().sort_values(ascending=False).head(15).rename("votes")
L += ["## The 15 model pairs with the most votes where the higher-rated model is cheaper", "",
      "Higher-rated = the side with the higher rating; typical cost = the model's median cost per request in the dev set.", "",
      top.reset_index().assign(
          higher_rated_per_1k=lambda x: (x.hi_model.map(mc.typical_cost) * 1000).round(2),
          lower_rated_per_1k=lambda x: (x.lo_model.map(mc.typical_cost) * 1000).round(2)).to_markdown(index=False), ""]

# view ② main dollar case
mc_sw = sw[sw.cls == "higher-rated costs more"]
L += [f"## ② Main dollar case: pricier strong model vs cheaper weak model ({len(mc_sw)} votes, {len(mc_sw) / len(sw):.1%} of strong-vs-weak votes)", ""]
q = mc_sw.ratio.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
d_usd = (mc_sw.typ_hi - mc_sw.typ_lo) * 1000
L += ["Typical cost ratio (strong / weak): " + ", ".join(f"P{int(k * 100)} {v:.1f}x" for k, v in q.items()),
      "", "Extra spend per 1,000 requests (strong - weak, USD): " +
      ", ".join(f"P{int(k * 100)} ${v:.2f}" for k, v in d_usd.quantile([0.1, 0.5, 0.9]).items()), "",
      f"Higher-rated clear-win rate {mc_sw.y_hi_route.mean():.3f}, tie rate {mc_sw.is_tie.mean():.3f}", ""]
for side, col in (("strong", "hi_model"), ("weak", "lo_model")):
    t = mc_sw.groupby(col).size().rename("votes").to_frame()
    t["cost_per_1k_usd"] = (t.index.map(mc.typical_cost) * 1000).round(2)
    L += [f"### {side.capitalize()}-tier models involved", "", t.sort_values("votes", ascending=False).to_markdown(), ""]
excl = sw[sw.cls != "higher-rated costs more"].groupby(["hi_model", "lo_model", "cls"]).size().rename("votes")
L += ["### Excluded strong-vs-weak votes (strong model not pricier, or no price)", "",
      excl.sort_values(ascending=False).reset_index().to_markdown(index=False), ""]

(root / "reports/stage2_step1_cost_views.md").write_text("\n".join(L))
print("\n".join(L))
