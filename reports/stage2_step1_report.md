# Stage 2 · Step 1: price table and the dollar cost axis

**In one sentence:** with real prices, in a third of the votes the higher-rated model is actually cheaper, and those votes need no routing. Among the remaining "the stronger one costs more" votes, the rating-gap-only baseline falls to random in the strong-vs-weak main case (APGR$ 0.498); ranking by "rating gap ÷ price gap" reaches 0.623 without reading the prompt, so it becomes the formal baseline on the dollar axis.

## 1. Price table (data/model_prices.csv)
- First-party official API list prices (USD per million tokens, standard tier, no cache or batch discounts). For open models without a first-party price, Llama uses AWS Bedrock throughout and the others use the hosted prices listed by LiteLLM.
- Checks: 28 match LiteLLM's public price list, 18 match official announcements or secondary sources, 1 could not be verified (qwen3-235b-instruct-2507, estimated from the same family's price), 6 have no price (4 experimental versions + 2 models with very few votes) and stay out of the dollar view.
- o3 cut its price by 80% on 2025-06-10, so its price follows the vote date (10/40 before, 2/8 after).
- OpenRouter's current list no longer includes these 2025 models and the Wayback archive was unreachable, so it could not be used as a cross-check.

## 2. Cost per vote
- Cost of each side of a vote = input price × context tokens + output price × answer tokens. Token counts come from the raw data's conv_metadata; this is bookkeeping only and never a routing feature.
- "Which side costs more" is decided per model: the median cost per request of that model in the dev set (data/model_cost.csv), not per vote, so the filter never uses answer length, which is future information.
- **Limitation:** hidden thinking tokens of reasoning models (o3, R1, thinking variants) are not in the data, so their cost is understated. Most of these models are on the higher-rated side, so the share where "the higher-rated model is cheaper" may be overstated.

## 3. Vote counts (90,923 dev votes, reports/stage2_step1_cost_views.md)
| View | Higher-rated costs more (enters the curves) | Higher-rated costs less or same (no routing needed) | No price |
|---|---|---|---|
| ① all | 57.0% | **33.1%** | 9.9% |
| ① rating gap < 50 | 50.7% | 38.7% | 10.6% |
| ① rating gap 50–150 | 56.4% | 33.5% | 10.1% |
| ① rating gap > 150 | 86.6% | 7.2% | 6.2% |
| ② strong vs weak | 82.3% | 8.3% | 9.4% |

The most common "the cheap model is better" pairs are cheap models such as gemini-2.5-flash, qwen3-235b and mistral-medium against Claude Opus 4 and Sonnet 4. In these users' preferences, the higher-rated side costs only 1/2 to 1/70 of its opponent per request.

**② main dollar case:** 9 strong-tier models against 11 weak-tier models, 6,855 votes. The strong tier costs 1.9 / 6.1 / 33 times the weak tier per request (P10 / median / P90), i.e. 1.39 / 10.99 / 18.92 dollars more per 1,000 requests.

## 4. Baselines on the dollar axis (dev set 5-fold, details in reports/eval_demo_dollar_baselines.md / .png)

**Ranking rule (set 2026-11-09):** on the dollar axis every router is ranked by "expected quality gain ÷ extra cost":
- key = (2p − 1) ÷ max(typical cost gap, floor). p is the router's P(higher-rated wins). Routing to the higher-rated instead of the lower-rated model gains p − (1 − p) = 2p − 1 in expected quality.
- Typical cost gap = typical cost per request of the higher-rated model − that of the lower-rated model, computed per model pair, without per-vote answer length.
- **Floor = $0.25 per 1,000 requests**, about the 5th percentile of the cost gap, so two models with nearly equal prices do not produce a huge ratio. Votes clipped by the floor are 5.2% of the pooled view and 3.8% of the main case.
- **Formal baseline = rating gap ÷ price gap**, with p the win rate implied by the rating gap. The old rating-gap baseline (ranks by p only, ignores price) stays in every table as a control row.

| View | Random | Rating-gap baseline (control) | **Rating gap ÷ price gap (formal baseline)** | For reference: rating-gap APGR on call share |
|---|---|---|---|---|
| ① pooled | 0.489 | 0.642 | **0.768** [0.757, 0.778] | 0.665 |
| ① < 50 | 0.469 | 0.652 | **0.814** | 0.638 |
| ① 50–150 | 0.495 | 0.596 | **0.753** | 0.576 |
| ① > 150 | 0.489 | 0.497 | **0.613** | 0.531 |
| ② main dollar case | 0.494 | 0.498 | **0.623** [0.608, 0.638] | 0.531 |

- On the dollar axis the rating-gap baseline falls to random in the > 150 bucket and in the main case. The reason: in pairs with a bigger rating gap, the strong model is usually also pricier. Ranking only by "which is stronger" spends money on the most expensive pairs first.
- The formal baseline is 0.12 to 0.16 above the rating-gap baseline in every view, with 95% intervals excluding 0.

**Sensitivity to the floor (formal baseline APGR$ point estimates):**

| View | floor $0.1/1k | **$0.25/1k (used)** | $1.0/1k | log: (2p−1) ÷ log(1 + cost gap/floor) |
|---|---|---|---|---|
| ① pooled | 0.768 | 0.768 | 0.767 | 0.727 |
| ② main dollar case | 0.623 | 0.623 | 0.623 | 0.621 |

With the floor anywhere from $0.1 to $1.0 the result barely moves, so no log is needed. Taking the log dampens the price gap and actually lowers the pooled view to 0.727.

**② within-pair APGR$ (main dollar case; measures ranking by prompt when the price gap is fixed):**
- Method: group votes by model pair and use only groups with at least 30 votes (68 pairs, 6,559 votes). Compute APGR$ inside each group, then average weighted by votes.
- Groups with a quality gap under 0.05 give distorted APGR, so they are skipped in the weighted mean. A pooled version that cannot be distorted is reported alongside: the sum of each group's area gain divided by the sum of the groups' quality gaps, i.e. weighted by votes × quality gap.

| Router | Vote-weighted | Pooled |
|---|---|---|
| Random router | 0.519 [0.499, 0.536] | 0.517 [0.499, 0.531] |
| Rating-gap baseline (control) | 0.497 [0.475, 0.514] | 0.496 [0.476, 0.510] |
| Rating gap ÷ price gap (formal baseline) | 0.494 [0.474, 0.515] | 0.493 [0.476, 0.511] |

Within a group the models and prices are fixed, so rating-gap baselines only keep cross-fold rating jitter, and all three rows are about 0.5. That is as expected, and shows the metric really measures only the part that comes from reading the prompt. A later router that sits steadily above 0.5 here has understood something about the prompt.

## 5. Limitations
- **Hidden thinking tokens of reasoning models are not counted.** Arena only records the answer tokens the user sees. o3, o3-mini, o4-mini, deepseek-r1, gemini-2.5-pro/flash (thinking on by default), qwen3-235b (thinking version), grok-3-mini, magistral and the thinking versions of Claude actually pay output prices for more tokens, so their cost is understated. Most of them are on the higher-rated side or in the strong tier, with three effects: (1) the 33% where "the higher-rated model is cheaper" may be too high; (2) the extra spend of the strong tier in the main case may be too low; (3) the formal baseline leans slightly towards sending to reasoning models. A next step could be a sensitivity analysis that multiplies reasoning models' output tokens by 2 to 5 and recomputes.
- Token counts come from one Arena tokenizer, not each provider's own, which may differ by 10% to 20%.
- Prices are list prices, without cache, batch or off-peak discounts. The price of qwen3-235b-instruct-2507 could not be verified.
- "Which side costs more" uses typical costs from the dev set. The test-set evaluation reuses the same table without recomputing it.

## Files
scripts/cost_table.py → scripts/cost_views.py → scripts/evaluate_cost.py; outputs data/vote_costs.parquet, data/model_cost.csv, data/model_prices.csv, reports/stage2_step1_*.md, reports/eval_demo_dollar_baselines.*.
