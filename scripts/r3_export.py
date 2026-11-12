"""r3_export.py — 导出 R3 打标范围: 口径 ② (固定名单强弱档对战) 的全部不同 prompt, 及 50 条试打样本

范围 (Yiqing 2026-11-10 定): 口径 ② 9,332 票 (开发 8,333 + 测试 999) 去重后 9,247 个 prompt。
测试集 prompt 只打标签, 不做任何评估; 标签之后只在 --final 时用一次。
prompt = 第一轮用户消息原文 (和 embedding 用的同一列); 语言列沿用 Step 1 的 langdetect 结果, 不交给 Haiku 判断。
试打样本: 只从开发集 prompt 里简单随机抽 50 条 (种子 2026), 用来无偏估计每条的平均 token 和费用。

用法: python -I r3_export.py <v2 根目录>
产出: r3/r3_prompts.jsonl (9,247 行), r3/pilot_50.jsonl, data/r3_prompts.parquet
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
print(len(g), g.split.value_counts().to_dict(), "试打长度中位/最大", pilot.prompt_len.median(), pilot.prompt_len.max(),
      "全量长度中位", g.prompt_len.median())
