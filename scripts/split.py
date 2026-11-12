"""split.py — Step 4: 固定划分 (按 group_id 分组)

决策
----
D1 先留出 10% 的 prompt 组作为最终测试集 (split = "test"), 任何模型选择、调参、
   甚至 BT 评分拟合都不许碰它。剩下 90% 为开发集 (split = "dev")。
D2 开发集再按 group_id 做 5 折分组交叉验证 (fold = 0..4; 测试集 fold = -1)。
   之后所有模型都用这同一组划分, 结果才能横向比较。
D3 一律按 group_id 分: 同一 prompt 的所有票 (哪怕对上不同模型) 落在同一侧,
   否则模型在训练时见过同一句话的投票结果, 测试分数会虚高。
D4 固定随机种子 2026, 划分写进表里存档, 不在每次训练时临时生成。
D5 测试集比例按"组"抽 10%; 因为大部分组只有 1 票, 按票算也接近 10%。

用法: python -I split.py <v2 根目录>
输出: data/arena_labeled.parquet 增加 split / fold 两列 (原地更新); reports/step4_split.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

SEED = 2026
root = Path(sys.argv[1])
path = root / "data/arena_labeled.parquet"
df = pd.read_parquet(path).drop(columns=["split", "fold"], errors="ignore")

dev_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
                         .split(df, groups=df.group_id))
df["split"] = "dev"
df.loc[df.index[test_idx], "split"] = "test"
df["fold"] = -1
dev = df[df.split == "dev"]
for k, (_, ev) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                            .split(dev, groups=dev.group_id)):
    df.loc[dev.index[ev], "fold"] = k

# 检查: 任何 group 只能出现在一个 (split, fold) 里
assert df.groupby("group_id")[["split", "fold"]].nunique().max().max() == 1
df.to_parquet(path, index=False)

tab = df.groupby(["split", "fold"]).agg(
    票数=("id", "size"), prompt组数=("group_id", "nunique"),
    平局=("is_tie", "mean"), 英文=("is_en", "mean")).round(3)
tab["占比"] = (tab["票数"] / len(df)).round(3)
md = ["# Step 4 固定划分", "", f"种子 {SEED}; 按 group_id 分组; 已校验没有任何 prompt 组跨折。", "",
      tab.to_markdown()]
(root / "reports/step4_split.md").write_text("\n".join(md))
print("\n".join(md))
