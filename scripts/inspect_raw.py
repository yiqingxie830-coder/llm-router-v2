"""inspect_raw.py — overview of the raw 140k data (step 0, read-only)

Purpose: before any cleaning rule, see what the data actually looks like and check it against the fields the spec assumes.
- conversation_a in the 140k set has the structure [{role, content: [{type, text, image, mimeType}]}];
  "first user message" = all text parts of the first role=='user' turn joined. (Checked: all text, no images.)
- evaluation_order: the n-th vote within a session. >1 means the user continued and was given a new model pair;
  the conversation then carries context left by other models, so it is not "the router sees a new prompt".

Usage: python -I inspect_raw.py <parquet directory> <output md path>
"""
import glob
import sys
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
cols = ["id", "model_a", "model_b", "winner", "evaluation_session_id", "evaluation_order",
        "conversation_a", "language", "is_code", "timestamp"]
df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{src}/*.parquet"))],
               ignore_index=True)


def first_user(conv):
    for t in conv:
        if t["role"] == "user":
            return " ".join(p["text"] for p in t["content"] if p["type"] == "text" and p["text"])
    return None


L = df.conversation_a.map(first_user).str.len()
models = pd.concat([df.model_a, df.model_b])


def share(s):
    return s.value_counts().to_frame().assign(share=lambda t: (t["count"] / len(s)).round(3)).to_markdown()


lines = [
    "# Raw 140k data overview", "",
    f"- Rows: {len(df)} (unique ids: {df.id.nunique()}); sessions: {df.evaluation_session_id.nunique()}",
    f"- evaluation_order=1: {(df.evaluation_order == 1).sum()} rows",
    f"- Models: {models.nunique()}",
    f"- Dates: {df.timestamp.min():%Y-%m-%d} ~ {df.timestamp.max():%Y-%m-%d}",
    "", "## winner distribution (all / order=1)", "", share(df.winner), "",
    share(df.winner[df.evaluation_order == 1]),
    "", "## Top 12 languages (dataset's own language column)", "", share(df.language).split("\n", 14)[:14] and
    "\n".join(share(df.language).split("\n")[:14]),
    f"\n{df.language.nunique()} language tags in total",
    "", "## Length of the first user message (characters)", "",
    L.describe(percentiles=[.05, .1, .25, .5, .75, .9, .95, .99]).round(0).to_frame("chars").to_markdown(),
    f"\n< 16 characters: {(L < 16).sum()} rows",
    "", "## Appearances per model", "", models.value_counts().to_frame().to_markdown(),
]
open(out, "w").write("\n".join(lines))
print("\n".join(lines[:8]))
