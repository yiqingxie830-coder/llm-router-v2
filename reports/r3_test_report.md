# Approach 2 (semantic labels) test-set results: the only run

Run on 2026-11-11, after the author's confirmation that day. The test set was used this once and sealed afterwards.
Full tables are in r3_compare_final.md (pre-registered settings) and r3_compare_final_ep100.md (sensitivity, early-stopping cap 100 epochs). Dev-set results are in r3_dev_report.md.

## How the test set was used (recorded as it happened)

- **This is the second use of the test set.** The first was approach 1's (R1, R2) --final on 2026-11-10, see final_test_report.md. Those results are not rerun or changed.
- **What was fixed in advance:** before the first --final results came out, approach 2's labeling plan was settled on 2026-11-10:
  - scope: the 9,247 prompts in view ②;
  - the 7 fields and the exact labeling prompt;
  - Haiku as the labeling model.
- **What was written afterwards:** the training details are in r3_prereg.md, including router structure, hyperparameter grids, the R2-② control and the decision rules. That file was written after the first --final results came out, but before the full labels were available, and before seeing any R3 result on the dev or test set. So R3's training design could not have been tuned to the test set, but its author already knew approach 1's test-set numbers.
- **Constraints on this run:**
  - settings exactly as in r3_prereg.md, no further tuning;
  - selection used only an 80/20 inner split of the full dev set's ② votes;
  - R2-②, R3 and R2+R3 were run together.
  - The version with the early-stopping cap raised to 100 epochs is the sensitivity check already reported on the dev set; it is reported here too, but not as the main result.
- **Chosen setting:** R3 chose logistic regression (C = 0.01), as in all 5 dev folds.

## Main metric: view ② APGR on the call-share axis (999 test votes, 988 prompts)

| Router | APGR | Δ vs rating-gap baseline |
|---|---|---|
| Rating-gap baseline | 0.529 | — |
| R1 (approach 1, all votes, first --final) | 0.575 | +0.046 [0.008, 0.088] |
| R2 (approach 1, all votes, first --final) | 0.612 | +0.083 [0.038, 0.127] |
| R2-② (approach 1, ② votes only) | 0.562 | +0.033 [0.004, 0.060] |
| R2-②, cap 100 epochs (sensitivity) | 0.577 | +0.048 [0.008, 0.092] |
| **R3 (approach 2)** | **0.592** | **+0.063 [0.017, 0.108]** |
| R2 + R3 (combined) | 0.597 | +0.068 [0.025, 0.115] |

## Pre-registered decisions on the test set

| Decision | Dev set | Test set | Holds on the test set? |
|---|---|---|---|
| Approach 2 is useful: R3's Δ interval excludes 0 | +0.051 [0.039, 0.066] | +0.063 [0.017, 0.108] | **Yes** |
| Approach 2 beats approach 1: R3 − R2-② interval is positive | +0.046 [0.034, 0.057] | +0.030 [−0.010, 0.065] | **No** (positive point estimate, interval contains 0) |
| Same, sensitivity version | +0.026 [0.016, 0.036] | +0.015 [−0.020, 0.042] | No |
| The combination adds value: R2+R3 − R2-② interval is positive | +0.049 [0.038, 0.059] | +0.035 [0.003, 0.071] | **Yes** (lower end close to 0) |
| Same, sensitivity version | +0.029 [0.020, 0.039] | +0.020 [−0.008, 0.047] | No |
| Reference: R3 − full R2 | −0.005 [−0.014, 0.005] | −0.020 [−0.053, 0.011] | Tie (interval contains 0) |

**How to read this:**
- The dev set has 8,333 ② votes and the test set only 999, so intervals are 3 to 4 times wider.
- The test set confirms "labels are useful", and that "trained on few votes, R3 is not far behind full R2".
- But "at equal data, labels beat embeddings" is not confirmed on the test set. It can only be said not to be overturned; the main evidence is from the dev set.
- The report should say: significant on the dev set, same direction but not significant on the test set.

## Dollar axis (secondary)

| Router | Main dollar case ΔAPGR$ vs formal baseline (840 votes) |
|---|---|
| R1 (all votes) | +0.023 [−0.009, 0.054] |
| R2 (all votes) | +0.016 [−0.024, 0.070] |
| R2-② | +0.019 [−0.007, 0.045] |
| R3 | 0.000 [−0.040, 0.043] |
| R2 + R3 | +0.003 [−0.042, 0.051] |

Note: the R1 and R2 rows come from a fresh bootstrap run together with R3 here, and differ by 0.002 from the first --final (final_test_report.md), R1 +0.021 [−0.007, 0.051] and R2 +0.014 [−0.021, 0.068] (both the point estimates and the intervals differ). The two runs used different evaluation scripts (this one r3_eval.py, the first evaluate_cost.py); exactly which step differs was not traced, and the conclusion is unaffected. External reporting always uses the first run's numbers.

Every router's interval contains 0, consistent with the dev-set finding that "routers trained only on ② votes have no edge on the dollar axis". The within-pair metric has only 3 model pairs and 98 votes on the test set; the intervals are too wide to interpret.

## By has_single_answer

The test-set groups have 421 / 382 / 133 / 61 votes; the score-3 group has only 61 votes and an interval wider than 0.6, so it is not interpreted. In the score-0 group, R3 gains +0.106 [0.041, 0.180] over the rating-gap baseline and full R2 +0.098. The direction matches the dev set: open-ended prompts with no single answer gain the most from reading the prompt.

## Sealed

The test set is sealed here. Every number in later reports either comes from the dev set or is copied from these two runs; no new model or setting is run on the test set.
