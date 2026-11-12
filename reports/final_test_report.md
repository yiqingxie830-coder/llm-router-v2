# Test-set results (--final, run once as pre-registered)

**In one sentence:** R2 held up the dev-set conclusion on the test set. On the call-share axis, strong-vs-weak APGR rises from the baseline's 0.529 to 0.612 (+0.083 [0.037, 0.128]) and the rating gap > 150 bucket gains +0.074 [0.031, 0.125], so both pre-registered success criteria are met. R1 is also significant on strong vs weak (+0.046 [0.009, 0.087]), but its interval in the > 150 bucket touches 0. On the main dollar case, neither router differs significantly from the formal baseline, as on the dev set.

Settings are in [final_prereg.md](final_prereg.md). All models were trained on the 90% dev set only. Early stopping inside the dev set chose 13 epochs for R2, which took 39 seconds. The test set has 10,135 votes: 999 in view ② and 840 in the main dollar case. Intervals are 95% intervals from 1000 bootstrap resamples by prompt group.

## 1. Call-share axis (against the rating-gap baseline, ΔAPGR)

| View | Test votes | Rating-gap baseline | R2 (main) | R2 dev Δ | R1 (control) | R1 dev Δ |
|---|---|---|---|---|---|---|
| ① pooled | 10,135 | 0.679 | 0.735 (**+0.056** [0.037, 0.076]) | +0.043 | 0.695 (+0.016 [0.002, 0.031]) | +0.015 |
| ① < 50 | 4,069 | 0.760 | 0.890 (**+0.130** [0.012, 0.292]) | +0.133 | 0.678 (−0.082, not significant) | −0.012 |
| ① 50–150 | 5,038 | 0.569 | 0.644 (**+0.075** [0.049, 0.103]) | +0.059 | 0.617 (+0.048 [0.028, 0.072]) | +0.036 |
| ① > 150 | 1,028 | 0.537 | 0.611 (**+0.074** [0.031, 0.125]) | +0.054 | 0.569 (+0.032 [−0.004, 0.073]) | +0.046 |
| ② strong vs weak | 999 | 0.529 | 0.612 (**+0.083** [0.037, 0.128]) | +0.054 | 0.575 (+0.046 [0.009, 0.087]) | +0.047 |

- **The pre-registered success criterion is that the intervals for view ② and the > 150 bucket both exclude 0.** R2 meets both; R1 meets only view ②, with a lower end of −0.004 in the > 150 bucket.
- Except in the < 50 bucket, R2's test point estimates are 0.01 to 0.03 above the dev set, but every dev number falls inside the test interval. The test set is only about 1/9 the size of the dev set, so its intervals are much wider. This cannot be read as "better on the test set"; it only says the dev conclusion was not overturned.
- R2's large gain in the < 50 bucket (closely matched pairs, +0.130) reappears on the test set, but its interval is wide, from 0.012 to 0.292.

## 2. Dollar axis (against the formal baseline "rating gap ÷ price gap", ΔAPGR$)

| View | Formal baseline | R2 | R2 dev Δ | R1 | R1 dev Δ |
|---|---|---|---|---|---|
| ① pooled | 0.780 | 0.779 (−0.001 [−0.026, 0.024]) | +0.006 | 0.774 (−0.006 [−0.023, 0.013]) | −0.011 |
| ① < 50 | 0.846 | 0.899 (+0.053 [−0.061, 0.259]) | +0.046 | 0.753 (−0.093, not significant) | −0.062 |
| ① 50–150 | 0.778 | 0.754 (−0.025 [−0.048, 0.014]) | −0.002 | 0.777 (−0.001) | −0.003 |
| ① > 150 | 0.631 | 0.633 (+0.002) | +0.003 | 0.643 (+0.012) | +0.006 |
| ② main dollar case (840 votes) | 0.610 | 0.624 (+0.014 [−0.021, 0.068]) | +0.008 | 0.630 (+0.021 [−0.007, 0.051]) | +0.011 |

- The dollar-axis conclusion matches the dev set: the price gap dominates the ranking, the extra gain from reading the prompt is within ±0.02, and every interval contains 0. R1's +0.011 on the main case was significant on the dev set and is not significant on the test set.
- The three versions of the formal baseline (buckets, 3-class logistic, BT half-vote) differ by no more than 0.02 on the test set, so the conclusion does not depend on the baseline version.

## 3. Main-case CPT$(50%) in dollars (per 1,000 requests)

In the test-set main case, sending everything to the weak tier costs $4.04 per 1,000 and sending everything to the strong tier $16.96.

| | CPT$(50%) [95% interval] | Cost per 1,000 | Saved vs formal baseline | Saved vs all-strong |
|---|---|---|---|---|
| Formal baseline | 0.41 [0.30, 0.48] | $9.34 | — | $7.62 (45%) |
| R2 | 0.31 [0.22, 0.40] | $8.05 | $1.29 (14%) | $8.91 (53%) |
| R1 | 0.31 [0.25, 0.39] | $8.05 | $1.29 (14%) | $8.91 (53%) |

After inference overhead (about $0.02 per 1,000, see [overhead_table.md](overhead_table.md)) the net saving is about $1.27. But the baseline's and the routers' intervals overlap, so the $1.29 cannot be treated as a firm number. The same figure on the dev set is $0.77; both should be reported together.

## 4. The within-pair metric (lower bound) cannot be estimated on the test set

In the test-set main case only 3 model pairs have ≥ 30 votes, 98 votes in total, and the interval is as wide as [0.3, 0.9], which carries no information. The lower bound for a product scenario therefore stays at the dev-set figures: R1 0.528 / R2 0.522, 0.02 to 0.03 above random. The pre-registration said so in advance ("the test set will have fewer such pairs than the dev set").

## 5. Files
- Predictions: `preds/r2_win_final.parquet`, `preds/r2_eff_final.parquet`, `preds/r1b_win_final.parquet`, `preds/r1b_eff_final.parquet`, `preds/base_tie_*_final.parquet`
- Detailed tables: `reports/eval_R2_MF_win_final.md` (call share), `reports/eval_R2_MF_final.md` (dollars), `reports/eval_R1_3class_win_final.md`, `reports/eval_R1_3class_final.md`
- R2 training record: `reports/r2_final_train.json`
