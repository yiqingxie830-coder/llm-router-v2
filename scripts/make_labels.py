"""make_labels.py — Step 2 (second half) + Step 3: labels, soft labels, grouping key

Data layout (confirmed 2026-11-09; product view: "how to route between any two models")
------------------------------------------------------------------------------------------
1. Full table data/arena_labeled.parquet: all 101,058 cleaned votes, each row carrying both models'
   ratings and tiers. Routers with model identity (RouteLLM's matrix factorization router) train on it
   and can be applied to any two models at serving time.
2. Evaluation view data/arena_strong_vs_weak.parquet: only "strong tier vs weak tier" votes.
   PGR/APGR are defined only for a fixed strong/weak pair (RouteLLM evaluates the same way), so
   evaluation happens on this view. Step 4 assigns folds on the full table by group_id and the view
   inherits them, so both tables share one split.

!! Added 2026-11-09 (rating leakage): the ratings / tiers / y_hi / y in this script come from a BT fit
!! on ALL votes, test votes included, so these columns are only for Step 3's descriptive statistics
!! (repeated prompts, first turn vs follow-ups) and must not be used for training or evaluation.
!! Training and evaluation labels are recomputed by fold_ratings.py from ratings fitted per fold.
!! What later steps actually use from this script is group_id, is_tie and tie_type.

Key decisions
-------------
D1 Tier boundaries: provisional option A (k=5, tiers 1-2 strong, tier 5 weak); change STRONG/WEAK to move them.
   Models with < 300 games have intervals too wide to place and get no tier (tier is empty).
D2 Labels (evaluation view): y = 1 strong wins, 0 weak wins; on a tie y is empty and is_tie = True.
   Ties are not folded into 0: "both are fine" and "the weak one is better" mean different things for
   routing, and mixing them would understate the weak tier.
   tie_type keeps the original distinction (tie / both_bad) for later study.
D3 The full table also gets a general label y_hi: higher-rated side wins = 1, lower-rated side wins = 0,
   tie empty. The same concept from the "any two models" point of view.
D3b Routing target (2026-11-09, cost view): when user preference is equal, prefer the small model.
   y_route = 1 only when the strong tier (or the higher-rated side) clearly wins; a weak win or a tie
   (tie / both_bad) is 0, i.e. "the weak tier is enough".
   The data layer still keeps is_tie / tie_type / y, so nothing is lost; y_route is a product rule on top.
   Evaluation reflects the same: on a tie both sides score 0.5, so sending the weak tier loses no quality
   and costs less, and the curve naturally favours the weak tier.
D4 Grouping key: strip the prompt, collapse runs of whitespace to one space, lowercase, take the first
   16 hex digits of its sha1. All votes on one prompt must land in the same fold; otherwise the model has
   seen the answer in training and test scores are inflated.
D5 Soft labels: the strong-tier win rate within a group, ties counting 0.5 (half each side). A prompt seen
   once gets 0 / 1 / 0.5. n_votes records the count; more votes, more trustworthy soft label.
   Disagreement between different users on the same prompt is direct evidence of the label-noise ceiling
   (see the report).

Usage: python -I make_labels.py <repo root>
"""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
TIER_COL, STRONG, WEAK, MIN_GAMES = "tier_k5", {1, 2}, {5}, 300

rt = pd.read_csv(root / "reports/step2_ratings.csv").set_index("model")
rating = rt.rating
tier = rt[TIER_COL].where(rt.n_games >= MIN_GAMES)  # too few games -> NaN


def norm(p: str) -> str:
    return re.sub(r"\s+", " ", p.strip()).lower()


def gid(p: str) -> str:
    return hashlib.sha1(norm(p).encode("utf-8")).hexdigest()[:16]


def label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["group_id"] = df.prompt.map(gid)
    df["rating_a"], df["rating_b"] = df.model_a.map(rating), df.model_b.map(rating)
    df["tier_a"], df["tier_b"] = df.model_a.map(tier), df.model_b.map(tier)
    df["is_tie"] = df.winner.isin(["tie", "both_bad"])
    df["tie_type"] = df.winner.where(df.is_tie)
    a_hi = df.rating_a >= df.rating_b
    hi_won = np.where(a_hi, df.winner == "model_a", df.winner == "model_b")
    df["y_hi"] = np.where(df.is_tie, np.nan, hi_won.astype(float))
    df["y_hi_route"] = np.where(df.is_tie, 0.0, hi_won.astype(float))  # tie -> send to the lower-rated side
    df["rating_gap"] = (df.rating_a - df.rating_b).abs()
    sa = df.tier_a.isin(STRONG) & df.tier_b.isin(WEAK)
    sb = df.tier_b.isin(STRONG) & df.tier_a.isin(WEAK)
    df["is_strong_vs_weak"] = sa | sb
    df["strong_model"] = np.where(sa, df.model_a, np.where(sb, df.model_b, None))
    df["weak_model"] = np.where(sa, df.model_b, np.where(sb, df.model_a, None))
    strong_won = np.where(sa, df.winner == "model_a", df.winner == "model_b")
    df["y"] = np.where(df.is_strong_vs_weak & ~df.is_tie, strong_won.astype(float), np.nan)
    df["y_route"] = np.where(df.is_strong_vs_weak, (strong_won & ~df.is_tie).astype(float), np.nan)
    return df


full = label(pd.read_parquet(root / "data/arena_clean.parquet"))
follow = label(pd.read_parquet(root / "data/arena_followup_clean.parquet"))

# Soft labels: per group, the higher-rated side's win rate on the full table and the strong-tier win rate on the view
full["vote_hi"] = full.y_hi.fillna(0.5)
g = full.groupby("group_id").vote_hi
full["soft_label_hi"], full["n_votes_all"] = g.transform("mean"), g.transform("size")

sw = full[full.is_strong_vs_weak].copy()
sw["vote"] = sw.y.fillna(0.5)
g = sw.groupby("group_id").vote
sw["soft_label"], sw["n_votes"] = g.transform("mean"), g.transform("size")
# Routing soft label: share of votes on the same prompt where the strong tier clearly won (ties count 0)
sw["soft_label_route"] = sw.groupby("group_id").y_route.transform("mean")
full["soft_label_hi_route"] = full.groupby("group_id").y_hi_route.transform("mean")

full.drop(columns="vote_hi").to_parquet(root / "data/arena_labeled.parquet", index=False)
# (The evaluation view is not written here: the official one is built by fold_ratings.py from dev-set ratings and the fixed lists)


# ---------- report ----------
def three(d, ycol):
    return pd.Series({"votes": len(d), "high/strong wins": (d[ycol] == 1).mean(), "low/weak wins": (d[ycol] == 0).mean(),
                      "tie": d.is_tie.mean()}).round(3)


def dup_stats(d, vcol, name):
    gr = d.groupby("group_id")[vcol].agg(["size", "mean"])
    rep = gr[gr["size"] >= 2]
    dec = d[~d.is_tie & d.group_id.isin(rep.index)]
    # Noise ceiling: on non-tie votes of repeated prompts, the best possible accuracy even if we knew each prompt's true majority
    gd = dec.groupby("group_id")[vcol].agg(["size", "mean"])
    ceil = (gd["size"] * np.maximum(gd["mean"], 1 - gd["mean"])).sum() / gd["size"].sum()
    mixed = ((gd["mean"] > 0) & (gd["mean"] < 1) & (gd["size"] >= 2)).mean()
    bins = pd.cut(rep["mean"], [-.01, .001, .25, .499, .5, .75, .999, 1], labels=[
        "0 (weak always wins)", "(0,0.25]", "(0.25,0.5)", "0.5", "(0.5,0.75]", "(0.75,1)", "1 (strong always wins)"])
    lines = [f"### {name}", "",
             f"- Distinct prompts: {gr.shape[0]}; seen ≥2 times: {rep.shape[0]}, covering {int(rep['size'].sum())} votes "
             f"({rep['size'].sum() / len(d):.1%})",
             f"- Repeat counts: {rep['size'].value_counts().sort_index().head(6).to_dict()} (max {int(rep['size'].max())})",
             f"- Share of repeated prompts with ≥2 non-tie votes that disagree: {mixed:.1%}",
             f"- Noise ceiling: on non-tie votes of repeated prompts, predicting each prompt's majority gives at most {ceil:.1%} accuracy",
             "", "Distribution of win rate (soft label) on repeated prompts:", "",
             bins.value_counts().sort_index().to_frame("prompts").to_markdown(), ""]
    return lines


full["vote_hi"] = full.y_hi.fillna(0.5)
sw["vote"] = sw.y.fillna(0.5)
fsw = follow[follow.is_strong_vs_weak]
tiers = rt.assign(tier=tier).dropna(subset=["tier"]).groupby("tier").agg(
    n_models=("rating", "size"), min_rating=("rating", "min"), max_rating=("rating", "max"))

md = ["# Step 2/3 labels and grouping", "",
      f"Tiers (provisional option A): strong = {TIER_COL} tiers {sorted(STRONG)}, weak = tier {sorted(WEAK)}; models with < {MIN_GAMES} games get no tier.", "",
      tiers.to_markdown(), "",
      "## Label shares", "",
      pd.DataFrame({"full table (y_hi: higher-rated side)": three(full, "y_hi"),
                    "evaluation view (y: strong vs weak)": three(sw, "y")}).to_markdown(), "",
      "## Routing target y_route (ties go to the weak tier, cost first)", "",
      pd.DataFrame({"full table y_hi_route": full.y_hi_route.value_counts(normalize=True).round(3),
                    "evaluation view y_route": sw.y_route.value_counts(normalize=True).round(3)}).to_markdown(), "",
      "## Repeated prompts (first evidence of label noise)", ""]
md += dup_stats(full, "vote_hi", "Full table (by higher-rated side)")
md += dup_stats(sw, "vote", "Evaluation view (strong vs weak)")
md += ["## First turn vs follow-ups: does the weak tier win more on follow-ups?", "",
       pd.DataFrame({"first turn order=1": three(sw, "y"), "follow-up order>1": three(fsw, "y")}).to_markdown(), "",
       "Full-table view (win rate of the lower-rated side; comparable only when the mean rating gaps are close):", "",
       pd.DataFrame({k: pd.Series({"votes": len(d), "lower-rated wins": (d.y_hi == 0).mean(), "tie": d.is_tie.mean(),
                                    "mean rating gap": d.rating_gap.mean()}).round(3)
                     for k, d in [("first turn", full), ("follow-up", follow)]}).to_markdown()]
(root / "reports/step3_labels.md").write_text("\n".join(md))
print("\n".join(md))
