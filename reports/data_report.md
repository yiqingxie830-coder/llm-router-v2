# LLM Router v2 · Stage 1 data report

**In one sentence:** we turned 136k Chatbot Arena human votes into 101k clean votes split by prompt group, and built the evaluation framework. Knowing only "how far apart the two models are rated" already routes well, so every router that reads the prompt from now on has to beat that baseline, not random routing.

## 1. Where the data comes from
- Dataset: lmarena-ai/arena-human-preference-140k, April to July 2025, 135,634 votes, 53 models (Gemini 2.5, o3, Claude 4 and others); files checked against sha256.
- Scope is **first-turn routing**. When a user keeps chatting in the same window, Arena swaps in a new model pair and takes another vote; these follow-up votes carry conversation context from other models. A router decides on the first message only, so the main data uses only the first vote of each session (108,315 votes). Follow-up votes are kept in a separate file for later work on downgrading mid-conversation.

## 2. Cleaning: remove only genuine noise (101,058 votes left, 6.7% removed)
| Rule | Removed | Reason |
|---|---|---|
| First message under 16 characters | 7,029 | "hi", "test" and the like carry no routing information, and the votes are casual |
| One side's answer is empty | 77 | Compares "an answer vs no answer", not model ability |
| Answer cut by a provider content filter | 133 | The text is inserted by the gateway, not written by the model; almost all from amazon nova |
| Identical prompt and model pair | 18 | Cannot tell a resubmission from independent votes, so dropped conservatively |

A model refusing on its own is model behaviour and is **kept**. Language is flagged, not filtered: English is 56%. Polish jumps from 0.5% to 12–24% from the week of June 16, which looks like a promotion campaign; a sensitivity analysis is planned.

## 3. Labels: human preference, never rewritten by AI
- Each vote has three outcomes: the stronger side wins, the weaker side wins, or a tie (including "both bad"). Ties are flagged on their own and never assigned to either side.
- **Routing target y_route**: 1 (send to the strong model) only when the stronger side wins outright; ties are 0. If the user finds both equally good, take the cheaper one. The raw tie information is kept.
- **"Strong" is defined by the data itself**: each model's rating is computed from the votes with a Bradley-Terry model rather than copied from the official leaderboard, so ratings and labels share one definition. The official leaderboard applies "style control" for answer length and formatting; we do not, which is why Claude Opus 4 ranks only 28th here. "Weak" here means "this set of users liked it less", not "less capable".

## 4. Rating leakage and the strong / weak tiers
- **Ratings are fitted per fold**: each fold computes ratings only from its own training votes; test votes are never used. After refitting, 3.9% of votes change which side is higher-rated, all on model pairs less than 20 points apart.
- **Tiers use a fixed list**: automatic k-means tiering was tried first, but in one fold the strong tier jumped from 10 models to 21. So we looked at the rank–rating plot and cut at clear gaps: 21.4 points between ranks 10 and 11, 24.6 points between ranks 40 and 41. **The top 10 are the strong tier and the bottom 11 the weak tier**; the 30 models in between take no part in the strong-vs-weak comparison. Models on either side of both cuts have non-overlapping rating ranges across all 6 fits, so the list holds in every fold.
- Strong vs weak has 9,332 votes: strong wins 61.6%, weak wins 17.4%, ties 21.1%.

## 5. Repeated prompts: noise has a ceiling
- 3,289 prompts were voted on two or more times (8.9% of votes), and **37%** of them have contradictory outcomes. Even guessing the majority vote for every prompt tops out at **79.5%** accuracy. (A conservative figure: repeated prompts often met different model pairs.)
- Repeated prompts are kept: they let soft_label record "how many people preferred the stronger model". Leakage is handled with group_id: all votes on the same prompt always land in the same fold.

## 6. Split
- By prompt group: 10% is held out first as the final test set (10,135 votes, not to be used for any model choice); the remaining 90% is split into 5 cross-validation folds, random seed fixed at 2026, with the fold number stored in the table.

## 7. Evaluation framework (no router is trained in this stage)
- The x-axis is "share of calls sent to the strong model", standing in for cost; the y-axis is PGR, how much of the quality gap between the strong and weak models is recovered. Summary metrics are APGR (area under the curve), CPT(50%) and CPT(80%), each with a 95% interval from 1000 resamples by prompt group.
- On a tie both sides have the same quality, so sending a tie to the small model costs nothing.
- Results on the dev set, 5 folds:

| View | Votes | Random router | Rating-gap baseline (ignores the prompt) |
|---|---|---|---|
| ① All model pairs pooled | 90,923 | 0.497 | **0.668** |
| ① Rating gap < 50 points | 36,414 | 0.485 | 0.645 |
| ① Rating gap 50–150 points | 45,867 | 0.500 | 0.580 |
| ① Rating gap > 150 points | 8,642 | 0.500 | 0.530 |
| ② Strong vs weak | 8,333 | 0.504 | 0.534 |

The random router's APGR is about 0.5, which confirms the framework computes correctly.

> **Note:** the pooled numbers in view ① depend on how often each model pair appears in Arena and do not correspond to any one product scenario. For concrete scenarios, use view ② and the rating-gap buckets.

## 8. What this means for the next stage
1. Every router's gain is measured against the rating-gap baseline. The real test is where the two models are far apart (> 150 points, and strong vs weak), where the baseline is only about 0.53.
2. In training, popular prompts should be down-weighted by vote count, or kept as one row per prompt with soft_label as the target.
3. Cost is approximated for now by the share of calls sent to the strong model; the next stage switches to real API prices.

---
**Files**: data/arena_clean.parquet (full cleaned data), data/arena_labeled.parquet (adds group_id, split, fold), data/fold_labels.parquet (per-fold ratings and labels), data/arena_strong_vs_weak.parquet (the strong-vs-weak evaluation view). Scripts run in order: download → clean → tier_models (ratings on all data, descriptive only) → make_labels → split → fold_ratings → plot_rank_rating → evaluate.
