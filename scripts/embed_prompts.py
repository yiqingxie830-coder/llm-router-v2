"""embed_prompts.py — Stage 2 Step 2: compute one vector per distinct prompt and cache it (CPU path)

Decisions
---------
- Model: intfloat/multilingual-e5-large (1024-d, multilingual; bge-m3 or e5-large were both acceptable, chosen 2026-11-09).
  bge-m3's weights sit on HF's xet storage, whose domain the cloud network blocked; e5-large has an ONNX build from
  qdrant that runs under fastembed (only onnxruntime, no torch needed).
- Input: only the first user message (the same information the router sees), with the "query: " prefix e5 expects,
  truncated to 512 tokens. Of the 95k prompts, long ones over 512 tokens keep only their beginning.
- Each group_id (identical prompt) is encoded once; prompts are sorted by length and processed in chunks, each saved
  to disk, so an interrupted run can resume.
- Pooling: mean pooling over the attention mask (the official e5 recipe). fastembed does not normalize e5-large,
  so the merged result is L2-normalized here (in float32), then stored as float16 to halve the size.
- Input text = the prompt_norm column of data/prompts_for_embedding.parquet (for each group_id, the first raw prompt
  in the data, case and whitespace kept, not the lower-cased form used to build group_id); row order = character
  length ascending (stable sort).

Usage: HF_HUB_DISABLE_XET=1 python -I embed_prompts.py <repo root> <work directory>
Writes: data/prompt_emb_e5large.npy (n_group × 1024, float16) + data/prompt_emb_index.parquet (group_id order)
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
