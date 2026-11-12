"""export_prompts_for_embedding.py — before Stage 2 Step 2: export the prompt table for computing vectors on Colab

Same rules as embed_prompts.py: for each group_id (identical prompt) take the first raw prompt in the data,
sorted by character length ascending (stable sort). colab_embed.py does not re-sort; it encodes rows in this order,
so vector row numbers line up one-to-one with the group_ids in data/prompt_emb_index.parquet.

Usage: python -I export_prompts_for_embedding.py <repo root>
Writes: data/prompts_for_embedding.parquet (group_id, prompt_norm; 95,373 rows), to upload to Colab
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
