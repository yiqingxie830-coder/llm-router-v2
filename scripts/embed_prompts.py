"""embed_prompts.py — 阶段 2 Step 2: 给每个不同的 prompt 算一次向量并缓存

决策
----
- 模型: intfloat/multilingual-e5-large (1024 维, 多语言; Yiqing 2026-11-09 同意 bge-m3 或 e5-large 二选一)。
  bge-m3 的权重在 HF 的 xet 存储上, 云端网络拦了那个域名; e5-large 有 qdrant 提供的 ONNX 版, 用 fastembed
  跑 (只需 onnxruntime, 不用装 torch)。
- 输入: 只用首条用户消息 (和路由时能看到的信息一致), 前面加 e5 要求的 "query: " 前缀, 截断到 512 token。
  9.5 万个 prompt 里超过 512 token 的长文只保留开头。
- 每个 group_id (相同 prompt) 只算一次; 按长度排序后分块计算, 每块存盘, 中断后可以接着跑。
- 池化: 按 attention mask 做 mean pooling (e5 官方做法)。fastembed 对 e5-large 不做归一化,
  所以合并时自己做 L2 归一化 (float32 下算), 再存成 float16 省一半空间。
- 输入文本 = data/prompts_for_embedding.parquet 的 prompt_norm 列 (每个 group_id 取数据里第一次出现的原始 prompt,
  保留大小写和空白, 不是算 group_id 用的小写版), 行顺序 = 按字符长度升序 (稳定排序)。

用法: HF_HUB_DISABLE_XET=1 python -I embed_prompts.py <v2 根目录> <工作目录>
产出: data/prompt_emb_e5large.npy (n_group × 1024, float16) + data/prompt_emb_index.parquet (group_id 顺序)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

root, work = Path(sys.argv[1]), Path(sys.argv[2])
CHUNK, MAXLEN = 2000, 512
(work / "chunks").mkdir(parents=True, exist_ok=True)

d = pd.read_parquet(root / "data/arena_labeled.parquet", columns=["group_id", "prompt"])
u = d.drop_duplicates("group_id").reset_index(drop=True)
u = u.assign(n=u.prompt.str.len()).sort_values("n", kind="stable").reset_index(drop=True)

from fastembed import TextEmbedding  # noqa: E402

m = TextEmbedding("intfloat/multilingual-e5-large", cache_dir=str(work / "cache"), threads=4)
m.model.tokenizer.enable_truncation(MAXLEN)

n_chunks = (len(u) + CHUNK - 1) // CHUNK
for k in range(n_chunks):
    f = work / "chunks" / f"{k:03d}.npy"
    if f.exists():
        continue
    t = time.time()
    texts = ["query: " + x for x in u.prompt.iloc[k * CHUNK:(k + 1) * CHUNK]]
    e = np.array(list(m.embed(texts, batch_size=32)), dtype=np.float16)
    np.save(f.with_suffix(".tmp.npy"), e)
    f.with_suffix(".tmp.npy").rename(f)
    print(f"chunk {k + 1}/{n_chunks}  {time.time() - t:.0f}s", flush=True)

emb = np.concatenate([np.load(work / "chunks" / f"{k:03d}.npy") for k in range(n_chunks)]).astype(np.float32)
emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float16)
assert emb.shape == (len(u), 1024)
np.save(root / "data/prompt_emb_e5large.npy", emb)
u[["group_id"]].to_parquet(root / "data/prompt_emb_index.parquet", index=False)
print("done", emb.shape)
