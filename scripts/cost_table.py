"""cost_table.py — 阶段 2 Step 1: 每票美元成本 + "高分方是否更贵" 的筛选

决策 (Yiqing 2026-11-09)
------------------------
1. 价格表 data/model_prices.csv: 第一方官方 API 标价 (USD / 百万 token, 标准档, 不含缓存/批量折扣)。
   开源模型没有第一方价时, Llama 统一用 AWS Bedrock, 其余用 LiteLLM 收录的托管商价。
   实验版/匿名模型 (没公开卖) 标"无价", 不进美元口径。
   o3 在 2025-06-10 降价 80%, 按投票日期取价。
2. 每票成本 (只用来记账, 不当路由特征): 输入价 × 该侧上下文 token + 输出价 × 该侧回答 token。
   token 数来自原始数据 conv_metadata (Arena 用同一个分词器统计, 不是各家自己的分词器;
   思考模型的隐藏推理 token 不在里面, 所以 o3 / R1 / thinking 版的真实成本被低估)。
3. "更贵方" 按模型定义, 不按单票: 每个模型的"典型单次成本" = 开发集 (90%) 里它所有票的成本中位数。
   单票成本里有回答长度 (未来信息), 用它筛票会让评估集本身依赖未来信息, 所以不用。
4. 美元口径只纳入 "高分方 典型成本 > 低分方" 的票; 其余票 (便宜模型反而评分更高) 单独统计, 不进曲线。
5. 口径 ② 美元主案例 = 强档 vs 弱档中满足第 4 条的全部配对 (贵的强档模型 vs 便宜的弱档模型), 作为一组。

用法: python -I cost_table.py <v2 根目录> <原始 140k 目录>
产出: data/vote_costs.parquet, data/model_cost.csv, reports/stage2_step1_costs.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

root, raw = Path(sys.argv[1]), Path(sys.argv[2])
O3_CUT = pd.Timestamp("2025-06-10")

# ---------- 1. 原始 token 数 ----------
clean = pd.read_parquet(root / "data/arena_labeled.parquet",
                        columns=["id", "model_a", "model_b", "timestamp", "split", "fold", "is_strong_vs_weak"])
keep = set(clean.id)
parts = []
for f in sorted(raw.glob("*.parquet")):
    t = pq.read_table(f, columns=["id", "conv_metadata"]).to_pandas()
    t = t[t.id.isin(keep)]
    m = pd.json_normalize(t.conv_metadata.tolist())
    parts.append(pd.DataFrame({"id": t.id.values,
                               "in_a": m["context_a_tokens"].values, "out_a": m["sum_assistant_a_tokens"].values,
                               "in_b": m["context_b_tokens"].values, "out_b": m["sum_assistant_b_tokens"].values,
                               "turns": m["turns"].values}))
tok = pd.concat(parts)
d = clean.merge(tok, on="id", how="left")
assert d.in_a.notna().all(), "部分票在原始数据里找不到 token 数"

# ---------- 2. 每票两侧成本 ----------
pr = pd.read_csv(root / "data/model_prices.csv").set_index("model")
missing = (set(d.model_a) | set(d.model_b)) - set(pr.index)
assert not missing, f"价格表缺模型: {missing}"


def side_cost(model, n_in, n_out, ts):
    pin = model.map(pr.input_usd_mtok).astype(float).values
    pout = model.map(pr.output_usd_mtok).astype(float).values
    o3_old = (model.values == "o3-2025-04-16") & (ts.values < O3_CUT)
    pin = np.where(o3_old, 10.0, pin)
    pout = np.where(o3_old, 40.0, pout)
    return (pin * n_in + pout * n_out) / 1e6


d["cost_a"] = side_cost(d.model_a, d.in_a, d.out_a, d.timestamp)
d["cost_b"] = side_cost(d.model_b, d.in_b, d.out_b, d.timestamp)

# ---------- 3. 每个模型的典型单次成本 (只用开发集) ----------
long = pd.concat([d[["model_a", "cost_a", "out_a", "split"]].set_axis(["model", "cost", "out_tok", "split"], axis=1),
                  d[["model_b", "cost_b", "out_b", "split"]].set_axis(["model", "cost", "out_tok", "split"], axis=1)])
dev = long[long.split != "test"]
mc = dev.groupby("model").agg(n=("cost", "size"), typical_cost=("cost", "median"),
                              median_out_tokens=("out_tok", "median"))
mc = mc.join(pr[["input_usd_mtok", "output_usd_mtok", "check"]])
mc["cost_per_1k_requests"] = mc.typical_cost * 1000
mc.sort_values("typical_cost", ascending=False).to_csv(root / "data/model_cost.csv")

d["typ_a"] = d.model_a.map(mc.typical_cost)
d["typ_b"] = d.model_b.map(mc.typical_cost)
d[["id", "in_a", "out_a", "in_b", "out_b", "turns", "cost_a", "cost_b", "typ_a", "typ_b"]].to_parquet(
    root / "data/vote_costs.parquet", index=False)
print("写出 data/vote_costs.parquet", len(d), "票;  data/model_cost.csv", len(mc), "个模型")
