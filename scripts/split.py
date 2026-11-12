"""split.py — Step 4: fixed split (grouped by group_id)

Decisions
---------
D1 First hold out 10% of prompt groups as the final test set (split = "test"); no model selection,
   tuning or even BT rating fit may touch it. The other 90% is the dev set (split = "dev").
D2 The dev set gets 5-fold grouped cross-validation by group_id (fold = 0..4; test set fold = -1).
   Every model afterwards uses this same split, so results are comparable.
D3 Always split by group_id: all votes on one prompt (even against different models) fall on the same
   side; otherwise a model has seen the votes on that exact prompt in training and scores are inflated.
D4 Fixed seed 2026; the split is written into the table, not regenerated at training time.
D5 The test share is 10% of groups; since most groups have a single vote, it is also close to 10% of votes.

Usage: python -I split.py <repo root>
Output: adds split / fold columns to data/arena_labeled.parquet (in place); reports/step4_split.md
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

# Check: every group appears in exactly one (split, fold)
assert df.groupby("group_id")[["split", "fold"]].nunique().max().max() == 1
df.to_parquet(path, index=False)

tab = df.groupby(["split", "fold"]).agg(
    votes=("id", "size"), prompt_groups=("group_id", "nunique"),
    tie=("is_tie", "mean"), english=("is_en", "mean")).round(3)
tab["share"] = (tab["votes"] / len(df)).round(3)
md = ["# Step 4 fixed split", "", f"Seed {SEED}; grouped by group_id; checked that no prompt group crosses folds.", "",
      tab.to_markdown()]
(root / "reports/step4_split.md").write_text("\n".join(md))
print("\n".join(md))
