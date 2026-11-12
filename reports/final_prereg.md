# Test set --final pre-registration (2026-11-10, written before any test-set result existed)

> English translation. The original was written in Chinese on 2026-11-10 and is kept unchanged in this repository's git history (commit 3c66d6c, `reports/final_prereg.md`).

The test set (10,135 votes, 999 of them in view ②) is evaluated this one time. The settings below were fixed before the run and are not changed afterwards. If the results look bad, nothing is tuned after the fact.

## Training data
- Every model is trained only on the 90% dev set (context = final, role = train). BT ratings are also fitted on dev votes only (already fixed in fold_ratings.py).
- Vectors for the test-set prompts were computed long ago but never took part in training or hyperparameter choice.

## Routers and settings (all from nested selection over the dev set's 5 folds)
| Name | Settings | Source |
|---|---|---|
| Formal baseline | Empirical win / tie / loss shares in 10 rating-gap buckets, divided by the price gap (floor $0.25 per 1,000) | tie_baselines.py --final |
| Control baselines | 3-class logistic version, BT half-vote version, rating gap only (ignores price), random | same |
| **R2 (main)** | Matrix factorization, d = 128, Adam learning rate 1e-3, λ = 1e-5, batch size 1024. Epochs set by early stopping on a 20% split of the dev-set train votes by prompt-group hash (patience 3 epochs, at most 30), then retrained on all dev-set train votes for that many epochs | r2_sweep.md: this combination was chosen in 4 of 5 folds |
| R1 (control) | 3-class logistic regression, features = prompt vector + rating gap + model identity, C = 0.1, weight 1/n_votes | r1_c_selection.md: C = 0.1 chosen in all 5 folds |

## Metrics (same definitions as the dev-set reports, 1000 bootstrap resamples by prompt group)
- **Call-share axis** (evaluate.py, ranked by P(win)): APGR / CPT for ① pooled, the three ① buckets, and ② strong vs weak. ΔAPGR is always against the rating-gap baseline.
- **Dollar axis** (evaluate_cost.py, ranked by (P(win) − P(loss)) ÷ max(typical cost gap, $0.25 per 1,000)): APGR$ / CPT$ for ① pooled, the buckets and the ② main dollar case, with ΔAPGR$ against the formal baseline; within-pair APGR$ for the main case (only model pairs with ≥ 30 votes; the test set will have fewer such pairs than the dev set); main-case CPT$(50%) converted to dollars per 1,000 requests.
- Upper and lower bounds are reported together: the strong-vs-weak Δ is the upper bound, the within-pair difference from 0.5 is the lower bound.

## Expectations stated in advance
- View ② on the test set has only 999 votes, about 1/8 of the dev set's 8,333, so intervals will be roughly 3 times as wide. A dev result like +0.054 [0.043, 0.068] may have a test interval of ±0.03, and its lower end could touch 0.
- The success criterion is stage 2's: the ΔAPGR intervals for view ② and the > 150 bucket exclude 0. If that is not met, it is reported as is, without switching metrics.

## Not included
- R3: to be pre-registered separately once labeling is done; the test set is evaluated once.
- The shuffled-vector control: already done on the dev set, not repeated on the test set.
