"""cost_table.py — Stage 2 Step 1: dollar cost per vote + the "is the higher-rated model pricier" filter

Decisions (2026-11-09)
----------------------
1. Price table data/model_prices.csv: first-party official API list prices (USD per million tokens, standard tier,
   no cache / batch discounts). Open models without a first-party price: Llama uses AWS Bedrock, the rest use the
   hosted price listed by LiteLLM. Experimental / anonymous models (not sold publicly) are marked "no price" and
   stay out of the dollar view. o3 dropped its price by 80% on 2025-06-10, so the price follows the vote date.
2. Cost per vote (bookkeeping only, never a routing feature): input price × that side's context tokens
   + output price × that side's answer tokens. Token counts come from the raw conv_metadata (Arena counts them
   with one tokenizer, not each provider's own; hidden reasoning tokens of thinking models are not included,
   so the true cost of o3 / R1 / thinking variants is understated).
3. "Pricier side" is defined per model, not per vote: each model's "typical cost per request" = median cost of
   all its votes in the dev set (90%). Per-vote cost contains the answer length (future information); filtering
   votes on it would make the evaluation set itself depend on future information, so it is not used.
4. The dollar view only includes votes where the higher-rated model's typical cost > the lower-rated model's;
   the rest (the cheaper model is rated higher) are counted separately and stay out of the curves.
5. View ② main dollar case = all strong-vs-weak pairs that satisfy rule 4 (pricier strong model vs cheaper
   weak model), taken as one group.

Usage: python -I cost_table.py <repo root> <raw 140k directory>
Writes: data/vote_costs.parquet, data/model_cost.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

root, raw = Path(sys.argv[1]), Path(sys.argv[2])
O3_CUT = pd.Timestamp("2025-06-10")

# ---------- 1. raw token counts ----------
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
assert d.in_a.notna().all(), "some votes have no token counts in the raw data"

# ---------- 2. cost of both sides per vote ----------
pr = pd.read_csv(root / "data/model_prices.csv").set_index("model")
missing = (set(d.model_a) | set(d.model_b)) - set(pr.index)
assert not missing, f"price table is missing models: {missing}"


def side_cost(model, n_in, n_out, ts):
    pin = model.map(pr.input_usd_mtok).astype(float).values
    pout = model.map(pr.output_usd_mtok).astype(float).values
    o3_old = (model.values == "o3-2025-04-16") & (ts.values < O3_CUT)
    pin = np.where(o3_old, 10.0, pin)
    pout = np.where(o3_old, 40.0, pout)
    return (pin * n_in + pout * n_out) / 1e6


d["cost_a"] = side_cost(d.model_a, d.in_a, d.out_a, d.timestamp)
d["cost_b"] = side_cost(d.model_b, d.in_b, d.out_b, d.timestamp)

# ---------- 3. typical cost per request for each model (dev set only) ----------
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
print("wrote data/vote_costs.parquet", len(d), "votes;  data/model_cost.csv", len(mc), "models")
