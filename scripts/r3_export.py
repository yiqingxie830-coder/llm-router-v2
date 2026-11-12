"""r3_export.py — export the R3 labeling scope: every distinct prompt in view ② (fixed-list strong-vs-weak votes), plus a 50-prompt pilot sample

Scope (set 2026-11-10): view ② has 9,332 votes (dev 8,333 + test 999), 9,247 distinct prompts.
Test-set prompts are only labeled, never evaluated; their labels were used once, at --final.
prompt = the raw first user message (the same column the embedding uses); the language column reuses Step 1's
langdetect result and is not left to Haiku.
Pilot sample: a simple random sample of 50 dev-set prompts (seed 2026), for an unbiased estimate of tokens and cost per prompt.

Usage: python -I r3_export.py <repo root>
Writes: r3/r3_prompts.jsonl (9,247 rows), r3/pilot_50.jsonl, data/r3_prompts.parquet
"""
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
d = pd.read_parquet(root / "data/arena_labeled.parquet",
                    columns=["group_id", "prompt", "langdetect_lang", "is_strong_vs_weak", "split", "prompt_len"])
s = d[d.is_strong_vs_weak]
g = (s.groupby("group_id")
     .agg(prompt=("prompt", "first"), langdetect_lang=("langdetect_lang", "first"), split=("split", "first"),
          n_votes=("prompt", "size"), prompt_len=("prompt_len", "first"))
     .reset_index())
assert len(g) == 9247 and s.groupby("group_id").split.nunique().max() == 1
g.to_parquet(root / "data/r3_prompts.parquet", index=False)
out = root / "r3"
out.mkdir(exist_ok=True)
cols = ["group_id", "split", "langdetect_lang", "prompt"]
with open(out / "r3_prompts.jsonl", "w") as f:
    for r in g[cols].to_dict("records"):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
pilot = g[g.split == "dev"].sample(50, random_state=2026)
with open(out / "pilot_50.jsonl", "w") as f:
    for r in pilot[cols].to_dict("records"):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(g), g.split.value_counts().to_dict(), "pilot length median/max", pilot.prompt_len.median(), pilot.prompt_len.max(),
      "full-set length median", g.prompt_len.median())
