"""clean.py — Step 1 cleaning: remove only genuine noise, never touch the human vote labels

Input: the raw 140k parquet directory (output of download.py; treated as untrusted, read with python -I)
Output:
  data/arena_clean.parquet            main data: evaluation_order == 1, cleaned
  data/arena_followup_clean.parquet   follow-up votes: evaluation_order > 1, same rules, not used in this stage
  reports/step1_cleaning.md           rows removed by each rule

Key decisions
-------------
D0 Scope = "first-turn routing" (confirmed 2026-11-07).
   After a vote, Arena swaps in a new model pair and the session continues (evaluation_order 2, 3, ...).
   Those follow-up prompts carry context written by other models; a router that only sees the follow-up
   line would mistake hard tasks for easy ones. So the main data is order == 1 only; follow-up votes are
   kept in a separate file for later work on mid-conversation downgrading.
D1 prompt = the first user message (text parts only). The routing decision happens the moment the user
   sends the first message; later turns, answer content and answer length are "future information" and
   never enter the main table (v1 learned this the hard way).
D2 Removal rules, applied in order; each counts only rows not already removed:
   R1 First message < 16 characters after stripping: "hi", "test", "1+1" carry almost no routing
      information, and those votes are mostly casual clicks (genuine noise).
   R2 Either answer is empty: the user compared "an answer vs nothing", which does not measure the model.
   R3 Either answer was cut by a provider content filter ("The generated text has been blocked by our content filters."):
      the only clear "system error" in the data. The text is inserted by the API gateway, not written by
      the model, and comes almost entirely from the two amazon nova models.
      Note: a model refusing on its own ("I can't help with that") is model behaviour and is kept.
      Answers starting with "TypeError..." (2 rows) were also checked: the prompt itself asked about the
      error, so they are not system errors.
   R4 Exact duplicates of (prompt text, model_a, model_b) keep only the earliest row: we cannot tell
      "the same person resubmitting" from "different people voting independently", so we drop them
      conservatively. The same prompt on a different model pair is unaffected; Step 3 uses those for soft labels.
D3 Language: keep every language, flag only.
   language = the dataset's own tag (detected by LMArena, one consistent source);
   langdetect_lang = our own langdetect pass, used only to report the agreement rate as a cross-check.
   is_en = (language == "en"), for an English-only sensitivity analysis later.
D4 tie / both_bad stay in the winner column as they are, assigned to neither side (Step 2 sets y labels).

Usage: python -I clean.py <raw parquet dir> <repo root>
"""
import glob
import sys
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # langdetect is randomized internally; fix the seed for reproducibility

MIN_PROMPT_CHARS = 16
FILTER_MSG = "The generated text has been blocked by our content filters"

src, root = sys.argv[1], Path(sys.argv[2])
cols = ["id", "evaluation_session_id", "evaluation_order", "model_a", "model_b", "winner",
        "conversation_a", "conversation_b", "language", "is_code", "timestamp"]
raw = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{src}/*.parquet"))],
                ignore_index=True)


def text_of(turn):
    return " ".join(p["text"] for p in turn["content"] if p["type"] == "text" and p["text"])


def first(conv, role):
    """Text of the first turn with the given role; empty string if there is none."""
    return next((text_of(t) for t in conv if t["role"] == role), "")


df = pd.DataFrame({
    "id": raw.id,
    "session_id": raw.evaluation_session_id,
    "evaluation_order": raw.evaluation_order,
    "model_a": raw.model_a,
    "model_b": raw.model_b,
    "winner": raw.winner,
    "prompt": raw.conversation_a.map(lambda c: first(c, "user")),
    "language": raw.language,
    "is_code": raw.is_code,
    "timestamp": raw.timestamp,
})
# Answers are used only for R2/R3 and then dropped; they never reach the output table
resp_a = raw.conversation_a.map(lambda c: first(c, "assistant")).str.strip()
resp_b = raw.conversation_b.map(lambda c: first(c, "assistant")).str.strip()
del raw

rules = {
    "R1 first message < 16 chars": df.prompt.str.strip().str.len() < MIN_PROMPT_CHARS,
    "R2 an answer is empty": (resp_a == "") | (resp_b == ""),
    "R3 answer cut by provider content filter": resp_a.str.contains(FILTER_MSG, regex=False)
                                   | resp_b.str.contains(FILTER_MSG, regex=False),
}


def clean(part: pd.DataFrame, name: str):
    """Apply the rules in order; return (cleaned table, rows removed per rule)."""
    log = [("start", len(part), 0)]
    keep = part
    for rule, mask in rules.items():
        drop = mask.loc[keep.index]
        keep = keep[~drop]
        log.append((rule, len(keep), int(drop.sum())))
    # R4: same prompt + same (model_a, model_b) keeps only the earliest row
    keep = keep.sort_values("timestamp")
    dup = keep.duplicated(["prompt", "model_a", "model_b"], keep="first")
    keep = keep[~dup].sort_index()
    log.append(("R4 exact duplicate prompt+model_a+model_b", len(keep), int(dup.sum())))
    return keep, pd.DataFrame(log, columns=[f"{name}: rule", "rows left", "removed"])


main, log_main = clean(df[df.evaluation_order == 1], "main data order=1")
follow, log_follow = clean(df[df.evaluation_order > 1], "follow-ups order>1")


def lang_detect(s):
    try:
        return detect(s)
    except LangDetectException:  # digits / symbols only, cannot be detected
        return "und"


for part in (main, follow):
    part["langdetect_lang"] = part.prompt.map(lang_detect)
    part["is_en"] = part.language == "en"
    part["prompt_len"] = part.prompt.str.len()

(root / "data").mkdir(exist_ok=True)
main.to_parquet(root / "data/arena_clean.parquet", index=False)
follow.to_parquet(root / "data/arena_followup_clean.parquet", index=False)

# ---------- report ----------
agree = (main.language == main.langdetect_lang)
known = main.language != "und"
lang_tab = main.language.value_counts().head(10).to_frame("rows")
lang_tab["share"] = (lang_tab["rows"] / len(main)).round(3)
win_tab = main.winner.value_counts().to_frame("rows")
win_tab["share"] = (win_tab["rows"] / len(main)).round(3)
pl_week = (main.assign(week=main.timestamp.dt.to_period("W").astype(str), pl=main.language == "pl")
           .groupby("week").pl.mean().round(3))

md = [
    "# Step 1 cleaning results", "",
    f"Raw {len(df)} rows = order=1 {int((df.evaluation_order == 1).sum())} rows + follow-ups {int((df.evaluation_order > 1).sum())} rows (follow-ups stored separately).", "",
    log_main.to_markdown(index=False), "", log_follow.to_markdown(index=False), "",
    "## Main data: winner distribution", "", win_tab.to_markdown(), "",
    "## Main data: top 10 languages (dataset's own tag)", "", lang_tab.to_markdown(), "",
    f"English is_en: {main.is_en.sum()} rows ({main.is_en.mean():.1%})",
    f"Agreement of dataset language tag with langdetect: all rows {agree.mean():.1%}; excluding rows tagged und {agree[known].mean():.1%}",
    f"Agreement on English rows: {agree[main.language == 'en'].mean():.1%}", "",
    "## Share of Polish by week", "", pl_week.to_frame("pl share").to_markdown(), "",
    "## Main data: first message length (characters)", "",
    main.prompt_len.describe(percentiles=[.1, .25, .5, .75, .9, .99]).round(0).to_frame("chars").to_markdown(),
]
(root / "reports/step1_cleaning.md").write_text("\n".join(md))
print("\n".join(md))
