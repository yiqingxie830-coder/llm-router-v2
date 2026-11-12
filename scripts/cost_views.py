"""cost_views.py — 阶段 2 Step 1: 美元口径的票数统计 (Yiqing 2026-11-09 第 1、2 条)

1. 开发集每票按 "高分方典型成本 vs 低分方" 分三类: 高分方更贵 (进美元曲线) / 高分方更便宜或一样
   ("便宜模型更优、无需路由", 只报比例) / 无价。按 5 折各自的高分方计算, 合并各折 eval 票。
2. 口径 ② 美元主案例: 强档 vs 弱档里 "强档更贵" 的全部配对, 报票数、参与模型、价格倍数分布。

用法: python -I cost_views.py <v2 根目录>   产出 reports/stage2_step1_cost_views.md
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
                      ["无价", "高分方更贵"], "高分方更便宜或一样")
ev["ratio"] = ev.typ_hi / ev.typ_lo
ev["bucket"] = pd.cut(ev.rating_gap, [0, 50, 150, np.inf], right=False, labels=["<50", "50–150", ">150"])

L = ["# 阶段 2 Step 1: 美元口径的票数统计 (开发集 5 折, 合并各折 eval 票)", ""]


def share_table(d, title):
    t = d.groupby("cls").agg(票数=("id", "size"), 高分方明确胜出率=("y_hi_route", "mean"), 平局率=("is_tie", "mean"))
    t["占比"] = t.票数 / t.票数.sum()
    t = t[["票数", "占比", "高分方明确胜出率", "平局率"]]
    return [f"## {title} (共 {len(d)} 票)", "", t.to_markdown(floatfmt=".3f"), ""]


L += share_table(ev, "① 全部模型对")
for b in ["<50", "50–150", ">150"]:
    L += share_table(ev[ev.bucket == b], f"① 评分差 {b}")
sw = ev[ev.is_strong_vs_weak]
L += share_table(sw, "② 强档 vs 弱档")

# 哪些模型对属于"便宜模型更优"
ch = ev[ev.cls == "高分方更便宜或一样"]
top = ch.groupby(["hi_model", "lo_model"]).size().sort_values(ascending=False).head(15).rename("票数")
L += ["## 「高分方反而更便宜」票数最多的 15 个模型对", "",
      "高分方 = 评分更高的一方; 典型成本 = 开发集里该模型单次请求成本的中位数。", "",
      top.reset_index().assign(
          高分方每千次=lambda x: (x.hi_model.map(mc.typical_cost) * 1000).round(2),
          低分方每千次=lambda x: (x.lo_model.map(mc.typical_cost) * 1000).round(2)).to_markdown(index=False), ""]

# 口径 ② 美元主案例
mc_sw = sw[sw.cls == "高分方更贵"]
L += [f"## ② 美元主案例: 贵的强档模型 vs 便宜的弱档模型 (共 {len(mc_sw)} 票, 占强弱对战 {len(mc_sw) / len(sw):.1%})", ""]
q = mc_sw.ratio.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
d_usd = (mc_sw.typ_hi - mc_sw.typ_lo) * 1000
L += ["典型成本倍数 (强档 / 弱档) 分布: " + ", ".join(f"P{int(k * 100)} {v:.1f} 倍" for k, v in q.items()),
      "", "每千次请求多花的钱 (强档 - 弱档, 美元): " +
      ", ".join(f"P{int(k * 100)} ${v:.2f}" for k, v in d_usd.quantile([0.1, 0.5, 0.9]).items()), "",
      f"高分方明确胜出率 {mc_sw.y_hi_route.mean():.3f}, 平局率 {mc_sw.is_tie.mean():.3f}", ""]
for side, col in (("强档", "hi_model"), ("弱档", "lo_model")):
    t = mc_sw.groupby(col).size().rename("票数").to_frame()
    t["每千次成本$"] = (t.index.map(mc.typical_cost) * 1000).round(2)
    L += [f"### 参与的{side}模型", "", t.sort_values("票数", ascending=False).to_markdown(), ""]
excl = sw[sw.cls != "高分方更贵"].groupby(["hi_model", "lo_model", "cls"]).size().rename("票数")
L += ["### 被排除的强弱对战 (强档不比弱档贵, 或无价)", "",
      excl.sort_values(ascending=False).reset_index().to_markdown(index=False), ""]

(root / "reports/stage2_step1_cost_views.md").write_text("\n".join(L))
print("\n".join(L))
