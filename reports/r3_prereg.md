# Approach 2 (R3 semantic labels) pre-registration: written before the full labels were available (2026-11-10)

> English translation. The original was written in Chinese on 2026-11-10 and is kept unchanged in this repository's git history (commit 3c66d6c, `reports/r3_prereg.md`).

## Background: approach 1 has already used the test set once
The test set was used once by R1 / R2 on 2026-11-10 (final_prereg.md, final_test_report.md). That step should have waited until approach 2 was done, so both could run together. The remedy is as follows:
1. **The formal comparison of the two approaches is on the dev set's 5 folds.** R1, R2, R3 and R2 + R3 use the same folds, the same baselines and the same evaluation code, and each prompt appears in every router's comparison, so differences can be paired in the bootstrap.
2. **Approach 2's design and hyperparameter grid are fixed now** (below), and selection happens only inside the training part of each dev fold. R2's test-set numbers play no part in any choice for approach 2.
3. Approach 2 runs on the test set once, with exactly the same evaluation code as approach 1. R1 / R2's test-set results are neither rerun nor changed.
4. The report will state that the test set was used in two rounds, approach 1 first, and that approach 1's test results had already been seen.

## Data scope
- Labels cover only the 9,247 prompts in view ② (8,259 dev, 988 test), so approach 2 can only train on ② votes: 8,333 dev votes, about 6,700 for training per fold.
- Approach 1's R2 was trained on all 90k votes. For a fair comparison, one more control is added: **R2-②**, the same code and hyperparameters as R2 but trained on ② votes only. The two approaches are compared mainly as R3 vs R2-②; the comparison with full R2 is for reference.

## Labeling settings (fixed 2026-11-10)
claude-haiku-5-5, thinking off, effort low, structured outputs, the 7-field labeling prompt as written, via the Batch API. Failed or refused prompts are recorded as missing, filled with the mode from the dev training part, and the number missing is reported.

## Routers and grids (all 3-class win / tie / loss, weight 1/n_votes)
| Name | Features | Hyperparameter grid (chosen by 80/20 inside each fold's training part only) |
|---|---|---|
| R3-LR | task_type one-hot (8 classes) + one-hot of the 6 scores (each 0–3) + rating gap + model identity | C ∈ {0.01, 0.03, 0.1, 0.3, 1} |
| R3-MF | Same structure as R2, with the 1024-dim prompt vector replaced by the label features above (38 dims) | d ∈ {4, 8, 16}, λ ∈ {1e-5, 1e-4, 1e-3}, learning rate 1e-3, early stopping |
| R2-② | R2 unchanged, trained on ② votes only | d ∈ {16, 64, 128}, λ ∈ {1e-6, 1e-5, 1e-4}, learning rate 1e-3 |
| R2 + R3 | R2 with input = prompt vector concatenated with the label features (1062 dims), ② votes only | same as R2-② |

Whichever of R3-LR and R3-MF has the lower inner validation loss represents "approach 2", chosen separately in each fold.

## Metrics (the same as approach 1)
- **Main metric:** APGR on the call-share axis for ② strong vs weak, ΔAPGR against the rating-gap baseline; differences between approaches (R3 − R2-②, R2+R3 − R2-②) by paired bootstrap, 1000 resamples by prompt group.
- **Secondary metrics:** main dollar case APGR$, within-pair APGR$ (lower bound), CPT$(50%).
- **Added analyses:** APGR for each router by has_single_answer (0 / 1 / 2 / 3); rank correlation of is_open_ended with approach 1's PC1 coordinate t1 (dev set).

## Decision rules (stated in advance)
- Approach 2 is "useful" = R3's ΔAPGR interval on ② excludes 0.
- Approach 2 is "better than approach 1" = the paired interval of R3 − R2-② excludes 0 and is positive.
- "The combination adds value" = the paired interval of R2+R3 − R2-② excludes 0 and is positive.
- In production R3 costs an extra $0.22 per 1,000 requests (overhead_table.md), so even if R3 ties on quality it loses on cost; the report states both.
