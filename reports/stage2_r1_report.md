> **Superseded by stage2_r1_r2_report.md (2026-11-10).** The dollar comparison here used the binary baseline, and C was chosen on the 5 folds' eval votes; both have since been corrected.

# Stage 2 · R1: prompt vector + rating gap + model identity → logistic regression (dev set 5-fold, test set untouched)

**In one sentence:** reading the prompt does help. On the call-share axis, strong-vs-weak APGR rises from 0.534 to 0.581 (+0.047, interval excludes 0), which clears the "worth doing" bar (+0.03). On the dollar axis, the signal from the price gap is far larger than the signal from the prompt: the main case is only 0.008 above the formal baseline, within same-price pairs it is 0.034 higher, and the pooled view is actually 0.011 lower.

## 1. Vectors
- intfloat/multilingual-e5-large, run on Colab, with the "query: " prefix, truncated beyond 512 tokens, mean pooling then L2 normalization: 95,373 × 1024, float16.
- Against 6,000 rows computed earlier in the cloud, the median cosine similarity is 1.0000 and the minimum 0.99996. Another 35 longer prompts (26 to 119,164 characters, including very long truncated ones) were recomputed in the cloud, with a minimum of 0.99997.

## 2. Model
- Logistic regression. Three feature blocks: rating gap ÷ 100; model identity (higher-rated +1, lower-rated −1, 53 dims); the prompt vector. Sample weight = 1 / n_votes. Each context trains only on its own train votes, with ratings from its own BT fit.
- Regularization C is chosen by mean log loss over the 5 folds (reports/r1_cv_selection.md). C = 0.1 was chosen, in the middle of the 0.01 to 1 range.

| Version | Log loss | AUC |
|---|---|---|
| Rating gap only | 0.6796 | 0.584 |
| + model identity | 0.6794 | 0.585 |
| + prompt vector (R1) | **0.6747** | **0.600** |

- **Where the gain comes from:** adding model identity alone moves APGR by at most ±0.006 in every view. So the gains below come entirely from the prompt.
- **3-class fix:** the dollar axis ranks by "expected quality gain ÷ extra cost". Quality is win 1, tie 0.5, loss 0, so routing to the higher-rated instead of the lower-rated model gains P(win) − P(loss) in expectation. The binary R1 outputs p as the probability that the higher-rated model wins outright, with ties as 0. Using 2p − 1 directly undercounts every vote by P(tie), about 0.27, and votes with a small rating gap nearly all come out as a negative gain. The fix is to train a 3-class model (win / tie / loss, scripts/train_r1b.py) on the same features and take P(win) − P(loss). Before the fix the pooled dollar view was 0.624; after it, 0.757. The rating-gap baseline does not have this problem, because BT win rates already count ties as half votes.

## 3. Results (3-class R1, 95% intervals from 1000 bootstrap resamples by prompt group)

**Call-share axis (against the rating-gap baseline)**

| View | Rating-gap baseline | R1 | ΔAPGR |
|---|---|---|---|
| ① pooled | 0.668 | 0.683 | **+0.015** [0.011, 0.019] |
| ① < 50 | 0.645 | 0.634 | −0.012 [−0.037, 0.017] |
| ① 50–150 | 0.580 | 0.616 | **+0.036** [0.029, 0.043] |
| ① > 150 | 0.530 | 0.575 | **+0.046** [0.036, 0.055] |
| ② strong vs weak | 0.534 | 0.581 | **+0.047** [0.038, 0.056] |

CPT(50%): on strong vs weak, the share of strong-model calls needed for 50% PGR drops from 46% to 40%.

**Dollar axis (against the formal baseline "rating gap ÷ price gap"; the rating-gap baseline is still listed in the detailed tables as a control)**

| View | Formal baseline | R1 | ΔAPGR$ |
|---|---|---|---|
| ① pooled | 0.768 | 0.757 | −0.011 [−0.016, −0.004] |
| ① < 50 | 0.814 | 0.757 | −0.057 [−0.094, −0.022] |
| ① 50–150 | 0.753 | 0.750 | −0.003 [−0.010, 0.001] |
| ① > 150 | 0.613 | 0.619 | +0.006 [0.000, 0.013] |
| ② main dollar case | 0.623 | 0.631 | **+0.008** [0.001, 0.014] |
| ② within pair (vote-weighted) | 0.494 | 0.528 | **+0.034** [0.011, 0.059] |
| ② within pair (pooled) | 0.493 | 0.522 | **+0.030** [0.007, 0.047] |

## 4. How to read these numbers
1. **The prompt signal is real but small.** Within one model pair at a fixed price, R1 gets about 0.03 to 0.05 more APGR than random ranking. On strong vs weak, CPT(50%) saves 6 percentage points of strong-model calls.
2. **On the dollar axis the price gap dominates the ranking.** Prices differ by 1.9 to 33 times across model pairs, and the formal baseline already captures most of the gain just by "sending first to the few pairs that are cheap and clearly better". R1's prompt signal adds only 0.008 on top in the main case.
3. **In the rating gap < 50 bucket, R1 is actually worse.** The expected gain for these votes is close to 0, and R1's prediction noise is amplified by the tiny price gaps, which skews the ranking. Two ways to improve: (a) have R1 predict only an "offset from the rating-gap baseline", with stronger shrinkage; (b) matrix factorization (MF, R2), letting the prompt interact with specific model pairs instead of adding one shared "difficulty" offset to all pairs.
4. **Against the success criteria:** the call-share view clears the "worth doing" bar; the main dollar case clears only the "minimum requirement" (interval excludes 0), not +0.03.

## Files
scripts/train_r1.py (binary, and choice of C), scripts/train_r1b.py (3-class); preds/r1_*.parquet, preds/r1b_*.parquet; detailed tables in reports/eval_R1_*.md / .json / .png.
