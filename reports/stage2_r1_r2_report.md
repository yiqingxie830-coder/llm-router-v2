# Stage 2 · R1 and R2: routers that read the prompt (dev set 5-fold, test set untouched)

**In one sentence:** reading the prompt helps. On the call-share axis, R2 (matrix factorization) raises strong-vs-weak APGR from 0.534 to 0.588 (+0.054) and R1 (logistic regression) to 0.581 (+0.047), both with intervals excluding 0. Retraining with the prompt vectors randomly shuffled makes the gain disappear, so it really comes from prompt content.

**The gain from reading the prompt is reported as a range (decided 2026-11-10):**
- **Upper bound:** strong vs weak +0.047 (R1) / +0.054 (R2). This ranks 99 model pairs together and includes gains across pairs.
- **Lower bound:** within-pair APGR$ is +0.028 (R1) / +0.022 (R2) above random. This corresponds to a product with a single model pair.

## 1. Common rules (decided 2026-11-10)
- **Expected quality gain = P(win) − P(loss).** Quality is win 1, tie 0.5, loss 0. Routing to the higher-rated instead of the lower-rated model gains [P(win) + 0.5·P(tie)] − [P(loss) + 0.5·P(tie)] in expectation.
- **Dollar ranking** = (P(win) − P(loss)) ÷ max(typical cost gap, $0.25 per 1,000).
- **Routers and baselines both give P(win) and P(loss) from three classes (win / tie / loss).** An unequal comparison such as "3-class router vs binary baseline" is not accepted.

## 2. The formal baseline with ties (scripts/tie_baselines.py)
- **Formal version: empirical bucket shares.** Using only each context's own train votes, bucket by deciles of the rating gap and compute the weighted (1/n_votes) win / tie / loss shares per bucket. For cv0, the smallest-gap bucket has win / tie / loss = 0.36 / 0.30 / 0.34, and the largest-gap bucket 0.63 / 0.20 / 0.17.
- **Check versions:** a 3-class logistic regression on the rating gap only, and the old BT half-vote version.

| Dollar-axis APGR$ | Bucket version (formal) | 3-class logistic version | BT half-vote (old) |
|---|---|---|---|
| ① pooled | 0.768 | 0.768 | 0.768 |
| ① < 50 | 0.819 | 0.820 | 0.814 |
| ② main dollar case | 0.620 | 0.623 | 0.623 |

The three versions differ by at most 0.006, and their differences have intervals containing 0. In other words, the old baseline already treated ties as half votes, and the conclusion does not depend on the baseline version.

## 3. Results

**Call-share axis (against the rating-gap baseline, ΔAPGR [95% interval])**

| View | Rating-gap baseline | R1 3-class | R2 MF | R2 shuffled-vector control |
|---|---|---|---|---|
| ① pooled | 0.668 | 0.683 (+0.015 [0.011, 0.019]) | **0.711** (+0.043 [0.038, 0.049]) | 0.668 (−0.000) |
| ① < 50 | 0.645 | 0.634 (−0.012, not significant) | **0.778** (+0.133 [0.100, 0.165]) | 0.644 |
| ① 50–150 | 0.580 | 0.616 (+0.036) | **0.640** (+0.059) | 0.579 |
| ① > 150 | 0.530 | 0.575 (+0.046) | **0.584** (+0.054) | 0.529 |
| ② strong vs weak | 0.534 | 0.581 (+0.047 [0.038, 0.056]) | **0.588** (+0.054 [0.043, 0.068]) | 0.530 |

**Dollar axis (against the formal baseline with ties, ΔAPGR$)**

| View | Formal baseline | R1 | R2 |
|---|---|---|---|
| ① pooled | 0.768 | 0.757 (−0.011 [−0.016, −0.004]) | 0.774 (+0.006 [−0.001, 0.013]) |
| ① < 50 | 0.819 | 0.757 (−0.062) | **0.866** (+0.046 [0.003, 0.095]) |
| ① 50–150 | 0.753 | 0.750 (−0.003) | 0.751 (−0.002) |
| ① > 150 | 0.613 | 0.619 (+0.006) | 0.616 (+0.003) |
| ② main dollar case | 0.620 | **0.631** (+0.011 [0.002, 0.020]) | 0.628 (+0.008 [−0.005, 0.021]) |

**Within-pair APGR$** (68 model pairs, 6,559 votes; computed inside each pair, then weighted by votes)

| | Formal baseline | R1 | R2 |
|---|---|---|---|
| Point estimate [95% interval] | 0.508 [0.491, 0.530] | **0.528** [0.510, 0.548] | **0.522** [0.506, 0.546] |
| vs random expectation 0.5 | — | +0.028, interval above 0.5 | +0.022, interval above 0.5 |
| vs formal baseline | — | +0.020 [−0.007, 0.045] | +0.015 [−0.015, 0.045] |

Within one model pair, the formal baseline is left with only random tie-breaking (every vote has the same p), so its expectation is 0.5. Subtracting this noisy random line from a router resample by resample adds its noise, so this metric is read mainly from whether the router's own interval sits above 0.5.

## 4. Interpretation (product scenario vs the Arena mix)
1. **In a product with a single model pair, the price gap is constant**, and the dollar curve has the same shape as the call-share curve. The metric closest to the product is then **within-pair APGR$**: R1 0.528 and R2 0.522, both 0.02 to 0.03 better than random, with intervals excluding 0.5. That is the effective gain from reading the prompt in a single pair.
2. **Strong vs weak +0.047 / +0.054 comes from ranking 99 model pairs together.** Besides ranking within a pair, it includes cross-pair gains such as "having judged this prompt hard, send to the strong model first in the pairs with the biggest strong-weak gap". A control that adds model identity without reading the prompt gains almost nothing, so this part still comes from the prompt. But a single-pair product cannot capture the cross-pair part, so +0.047 should be read as the upper bound and the within-pair +0.02 to 0.03 as the lower bound. All external material reports both.
3. **The ±0.01 in the pooled dollar view reflects Arena's mix of many pairs**, not a product reality. There the price gap (1.9 to 33 times) dominates the ranking, and the prompt signal is a small addition on top.
4. **R2 gains a lot in the rating gap < 50 bucket (+0.133).** In these pairs the two models are close in strength, and who wins depends mainly on whether the prompt type matches the model's style, for example writing code vs creative writing. Matrix factorization can learn "this model is good at this kind of prompt"; the linear R1 cannot. With shuffled vectors this gain disappears, so it comes from prompt content, not leakage. This gain is most valuable in a product with "two models at different prices but close in strength".

## 5. Main-case CPT(50%) in dollars (per 1,000 requests)
In the main case, sending everything to the lower-rated model costs $3.55 and everything to the higher-rated model $18.89. Cost at 50% PGR (CPT is on a 1% grid, so the error is about ±$0.15):

| | CPT$(50%) | Cost per 1,000 | Saved vs formal baseline | Saved vs all-strong |
|---|---|---|---|---|
| Formal baseline | 0.37 | $9.23 | — | $9.66 (51%) |
| R1 | 0.32 | $8.46 | $0.77 (8%) | $10.43 (55%) |
| R2 | 0.32 | $8.46 | $0.77 (8%) | $10.43 (55%) |

PGR 50% here means recovering half of the quality gap between the strong and weak models. By that standard, the extra saving from reading the prompt is about $0.8 per 1,000 requests; most of the saving comes from "send first to the pairs that are cheap and clearly better", which the baseline already does.

## 6. How hyperparameters were chosen (inside training folds only; the test set never took part)
**R1's C** (see reports/r1_c_selection.md): each outer fold uses only its own train votes, splits them 80/20 again by prompt group, trains on 80% and chooses C by log loss on 20%, then retrains on the whole train part and predicts that fold's eval votes. Neither eval votes nor the test set take part. All 5 folds chose C = 0.1. (The first version of this report chose C by mean log loss over the 5 folds' eval votes, also landing on 0.1; that was not strict enough and has been replaced.)

| C | Inner validation log loss | Strong vs weak APGR | Main dollar case APGR$ | Main case CPT$(50%) |
|---|---|---|---|---|
| 0.01 | 1.0386 | 0.568 | 0.628 | 0.33 |
| 0.03 | 1.0368 | 0.577 | 0.630 | 0.33 |
| **0.1** | **1.0360** | **0.581** | **0.631** | **0.32** |
| 0.3 | 1.0370 | 0.579 | 0.631 | 0.32 |
| 1.0 | 1.0391 | 0.576 | 0.631 | 0.33 |

For C from 0.03 to 1, strong-vs-weak APGR stays between 0.577 and 0.581 and the main case between 0.630 and 0.631, so the conclusion does not depend on C.

**R2's settings** (reports/r2_selection.md): k = 64, Adam learning rate 1e-3, early stopping and choice of λ on the same inner 80/20 split. Round 1 swept λ over {1e-5, 1e-4, 1e-3} and chose the smallest, 1e-5, at the grid edge (see r2_selection_round1.md). Round 2 widened it to {0, 1e-6, 1e-5, 1e-4}: 4 folds chose 1e-5 and 1 fold 1e-6, none at the edge.

## 7. R3: Haiku labeling cost (view ② only, 9,332 votes)
- After deduplication there are **9,247 prompts** (8,333 dev votes + 999 test votes; labeling only, no evaluation).
- Input per call = v1's 8-dimension scoring instructions (about 1,760 characters, about 450 tokens) + the structured-output format (about 200 tokens) + the prompt itself (about 92 tokens on average; the longest 1% exceed 1,900 tokens). Only 1.7% of prompts exceed 4,000 characters, so the recommendation is **no truncation**; it costs under $0.05 more.
- Output = a JSON of 8 scores (about 60 tokens) + thinking tokens. Haiku 5.5 thinks by default; with effort low, an estimated 0 to 300 extra tokens per call.
- Prices: Claude Haiku 5.5 input $0.10 per million tokens, output $0.50 per million. The scoring instructions are only about 650 tokens, below the minimum cacheable length, so prompt caching is not considered.

| | Input | Output | Total | With the Batch API (half price) |
|---|---|---|---|---|
| Tokens | about 6.9M (up to 1.35× by the Claude tokenizer) | 0.55M to 3.3M | | |
| Cost | $0.69 to $0.93 | $0.28 to $1.65 | **about $1.0 to $2.6** | **about $0.5 to $1.3** |

- For exact figures, first count the input with count_tokens, then run a 50-prompt pilot to measure actual output and thinking tokens. The 50-prompt pilot costs under $0.02. This step needs an API key, which the cloud environment does not have; it is run on the author's own machine.
- With Claude Sonnet 5.5 as in v1 ($2 / $10 per million tokens) the cost would be about 20 times higher, $20 to $50.

## Files
scripts/tie_baselines.py, train_r1.py, train_r1b.py, r1_csweep.py, train_r2_mf.py (supports the --shuffle control), evaluate.py, evaluate_cost.py; preds/ holds each router's per-fold predictions; detailed tables in reports/eval_R1_3class*.md, eval_R2_MF*.md, eval_R2_MF_shuffled_vectors.md.
