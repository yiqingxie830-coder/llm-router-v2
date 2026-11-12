"""colab_embed.py — 阶段 2 Step 2: 在 Colab GPU 上算 prompt 向量 (正式结果用的就是这份)

在 Google Colab 里按单元格运行 (运行时选 T4 GPU)。实测: T4, sentence-transformers, batch 128,
95,373 条用时 37 分 39 秒。

准备: 本地先跑 export_prompts_for_embedding.py, 把 data/prompts_for_embedding.parquet 上传到 Colab 当前目录。
模型: intfloat/multilingual-e5-large, 前缀 "query: ", 截断 512 token, mean pooling, L2 归一化, 存 float16。
注意: 不要排序, 保持 parquet 原行序 (export 时已按长度排好), 行号要和 prompt_emb_index.parquet 对齐。
产出: prompt_emb_e5large.npy (95373 × 1024) 和 prompt_emb_index.parquet, 下载后放进仓库 data/。

以下两行是 notebook 命令, 在 Colab 单元格里去掉开头的 "# " 运行:
# !pip -q install sentence-transformers pyarrow
# from google.colab import drive; drive.mount('/content/drive')
"""
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")

# 读数据 (不要排序, 保持 parquet 原行序)
df = pd.read_parquet("prompts_for_embedding.parquet")
print(df.shape, df.columns.tolist())
texts = ["query: " + str(t) for t in df["prompt_norm"]]

# 算向量
model = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")
model.max_seq_length = 512
emb = model.encode(texts, batch_size=128, normalize_embeddings=True,
                   show_progress_bar=True, convert_to_numpy=True)
print(emb.shape, np.linalg.norm(emb[:5], axis=1))   # 预期 (95373, 1024), 模长都是 1.0

# 存到 Drive
np.save("/content/drive/MyDrive/prompt_emb_e5large.npy", emb.astype(np.float16))
df[["group_id"]].to_parquet("/content/drive/MyDrive/prompt_emb_index.parquet", index=False)
print("done")
