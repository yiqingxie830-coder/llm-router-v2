"""overhead_bench.py — measured inference overhead: how much time each router adds per request (cloud container, 4 CPU cores, no GPU)

What is measured:
  1. Prompt vector (multilingual-e5-large, ONNX/fastembed, 4 threads, truncated at 512 tokens):
     a) single-request latency (batch = 1, one at a time), on 200 random view-② prompts (seed 2026), whose length
        distribution matches real requests;
     b) throughput (batch = 32), to convert into compute cost per 1,000 requests.
  2. Router heads: R1 (logistic regression: a 1024+55-dim dot product) and R2 (MF: a 1024×128 matrix product + two
     model-vector dot products), timed per request in numpy.
  3. Baseline (rating gap ÷ price gap): a table lookup, counted as 0.
R3's overhead (Haiku labeling) comes from the measurements in r3/pilot_summary.md and is not measured here.
Usage: python -I overhead_bench.py <repo root> <fastembed cache directory>
Writes: reports/overhead_bench.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

root, cache = Path(sys.argv[1]), sys.argv[2]
g = pd.read_parquet(root / "data/r3_prompts.parquet")
texts = ["query: " + p for p in g.sample(200, random_state=2026).prompt]
from fastembed import TextEmbedding  # noqa: E402

m = TextEmbedding("intfloat/multilingual-e5-large", cache_dir=cache, threads=4)
m.model.tokenizer.enable_truncation(512)
ntok = [len(m.model.tokenizer.encode(t).ids) for t in texts]
list(m.embed(texts[:5], batch_size=1))   # warm-up
lat = []
for t in texts:
    s = time.perf_counter()
    list(m.embed([t], batch_size=1))
    lat.append(time.perf_counter() - s)
s = time.perf_counter()
list(m.embed(texts, batch_size=32))
thr = len(texts) / (time.perf_counter() - s)

rng = np.random.default_rng(0)
e = rng.standard_normal(1024).astype(np.float32)
w1 = rng.standard_normal(1024 + 55).astype(np.float32)
P = dict(np.load(root / "data/r2_final_params.npz"))
W, b, v, w2, u = (P[k].astype(np.float32) for k in ["W", "b", "v", "w2", "u"])
x1 = np.concatenate([e, np.zeros(55, np.float32)])


def r1():
    return 1 / (1 + np.exp(-(x1 @ w1)))


def r2():
    z = e @ W + b
    return (v[3] * z) @ w2 + u[3] - (v[7] * z) @ w2 - u[7]


def timeit(f, n=20000):
    s = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - s) / n


out = {"cpu": "cloud container, 4 cores (fastembed threads=4)", "n_prompts": len(texts),
       "tokens_median": float(np.median(ntok)), "tokens_mean": float(np.mean(ntok)), "tokens_max": int(max(ntok)),
       "embed_latency_ms_p50": 1000 * float(np.percentile(lat, 50)), "embed_latency_ms_p90": 1000 * float(np.percentile(lat, 90)),
       "embed_latency_ms_p99": 1000 * float(np.percentile(lat, 99)), "embed_latency_ms_mean": 1000 * float(np.mean(lat)),
       "embed_throughput_per_s_batch32": thr,
       "r1_head_us": 1e6 * timeit(r1), "r2_head_us": 1e6 * timeit(r2)}
(root / "reports/overhead_bench.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
