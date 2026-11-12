"""export_prompts_for_embedding.py — 阶段 2 Step 2 前置: 导出给 Colab 算向量的 prompt 表

和 embed_prompts.py 用同一套规则: 每个 group_id (相同 prompt) 只取数据里第一次出现的原始 prompt,
按字符长度升序 (稳定排序)。colab_embed.py 不再排序, 直接按这个行序算, 所以向量行号和
data/prompt_emb_index.parquet 的 group_id 一一对应。

用法: python -I export_prompts_for_embedding.py <仓库根目录>
产出: data/prompts_for_embedding.parquet (group_id, prompt_norm; 95,373 行), 上传到 Colab
"""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
d = pd.read_parquet(root / "data/arena_labeled.parquet", columns=["group_id", "prompt"])
u = d.drop_duplicates("group_id").reset_index(drop=True)
u = u.assign(n=u.prompt.str.len()).sort_values("n", kind="stable").reset_index(drop=True)
u.rename(columns={"prompt": "prompt_norm"})[["group_id", "prompt_norm"]].to_parquet(
    root / "data/prompts_for_embedding.parquet", index=False)
print("done", len(u))
