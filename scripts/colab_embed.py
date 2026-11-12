"""colab_embed.py — Stage 2 Step 2: compute prompt vectors on a Colab GPU (this produced the official results)

Run cell by cell in Google Colab (runtime: T4 GPU). Measured: T4, sentence-transformers, batch 128,
95,373 prompts in 37 min 39 s.

Prepare: run export_prompts_for_embedding.py locally and upload data/prompts_for_embedding.parquet to the Colab working directory.
Model: intfloat/multilingual-e5-large, prefix "query: ", truncate at 512 tokens, mean pooling, L2-normalized, stored as float16.
Note: do not sort; keep the parquet row order (already sorted by length at export) so row numbers line up with prompt_emb_index.parquet.
Writes: prompt_emb_e5large.npy (95373 × 1024) and prompt_emb_index.parquet; download them into the repo's data/.

The two lines below are notebook commands; in a Colab cell, remove the leading "# " and run them:
# !pip -q install sentence-transformers pyarrow
# from google.colab import drive; drive.mount('/content/drive')
"""
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")

# read the data (do not sort; keep the parquet row order)
df = pd.read_parquet("prompts_for_embedding.parquet")
print(df.shape, df.columns.tolist())
texts = ["query: " + str(t) for t in df["prompt_norm"]]

# compute vectors
model = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")
model.max_seq_length = 512
emb = model.encode(texts, batch_size=128, normalize_embeddings=True,
                   show_progress_bar=True, convert_to_numpy=True)
print(emb.shape, np.linalg.norm(emb[:5], axis=1))   # expect (95373, 1024), all norms 1.0

# save to Drive
np.save("/content/drive/MyDrive/prompt_emb_e5large.npy", emb.astype(np.float16))
df[["group_id"]].to_parquet("/content/drive/MyDrive/prompt_emb_index.parquet", index=False)
print("done")
