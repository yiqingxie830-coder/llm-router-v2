"""overhead_bench.py — 推理开销实测: 每种路由器处理一条请求要多花多少时间 (云端 4 核 CPU, 无 GPU)

测什么:
  1. prompt 向量 (multilingual-e5-large, ONNX/fastembed, 4 线程, 截断 512 token):
     a) 单条请求延迟 (batch = 1, 逐条算), 用口径 ② 的 200 条随机 prompt (种子 2026), 长度分布和真实请求一致;
     b) 吞吐 (batch = 32), 用来折算每千次请求的算力成本。
  2. 路由头: R1 (逻辑回归: 1024+55 维点积) 与 R2 (MF: 1024×128 矩阵乘 + 两个模型向量内积), numpy 单条计时。
  3. 基线 (评分差 ÷ 价格差): 查表, 视为 0。
R3 (Haiku 打标) 的开销来自 r3/pilot_summary.md 的实测, 不在这里测。
用法: python -I overhead_bench.py <v2 根目录> <fastembed 缓存目录>
产出: reports/overhead_bench.json
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
list(m.embed(texts[:5], batch_size=1))   # 预热
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


out = {"cpu": "云端容器 4 核 (fastembed threads=4)", "n_prompts": len(texts),
       "tokens_median": float(np.median(ntok)), "tokens_mean": float(np.mean(ntok)), "tokens_max": int(max(ntok)),
       "embed_latency_ms_p50": 1000 * float(np.percentile(lat, 50)), "embed_latency_ms_p90": 1000 * float(np.percentile(lat, 90)),
       "embed_latency_ms_p99": 1000 * float(np.percentile(lat, 99)), "embed_latency_ms_mean": 1000 * float(np.mean(lat)),
       "embed_throughput_per_s_batch32": thr,
       "r1_head_us": 1e6 * timeit(r1), "r2_head_us": 1e6 * timeit(r2)}
(root / "reports/overhead_bench.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
