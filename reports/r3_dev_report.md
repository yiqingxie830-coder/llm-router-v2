# Approach 2 (semantic labels, R3) dev-set results, compared with approach 1 (embedding matching)

Date 2026-11-11. Run as set out in reports/r3_prereg.md, on the dev set's 5 folds only; the test set was not touched. Full tables are in r3_compare.md (pre-registered settings) and r3_compare_ep100.md (sensitivity check). Scripts: scripts/r3_train.py and scripts/r3_eval.py.

## In one sentence
With the same ~6,700 view-② training votes, semantic labels (R3) clearly beat embeddings (R2-②). R3 ties with the full R2 trained on 90k votes. So the choice between the two approaches is about data and cost: labels save data, embeddings save money on every request. Stacking them (R2 + R3) adds nothing over full R2. On the dollar axis, no router trained only on ② votes beats the formal baseline.

## 1. Main metric: view ② APGR on the call-share axis (8,333 eval votes, 1000 paired bootstrap resamples)

| Router | Training votes | APGR | Δ vs rating-gap baseline |
|---|---|---|---|
| Rating-gap baseline | — | 0.534 | — |
| R1 (approach 1, logistic regression) | all, about 73k per fold | 0.581 | +0.047 [0.036, 0.058] |
| R2 (approach 1, MF) | all | 0.590 | +0.056 [0.042, 0.071] |
| R2-② (approach 1, MF, ② votes only) | about 6,700 per fold | 0.540 | +0.006 [−0.002, 0.014] |
| R2-②, early-stopping cap raised to 100 epochs (sensitivity) | about 6,700 per fold | 0.560 | +0.026 [0.014, 0.037] |
| **R3 (approach 2)** | about 6,700 per fold | **0.585** | **+0.051 [0.039, 0.066]** |
| R2 + R3 (combined) | about 6,700 per fold | 0.588 | +0.054 [0.042, 0.068] |

R3 chose the logistic regression version (R3-LR) in all 5 folds, always with C = 0.01. That is the smallest value in the grid (strongest regularization), which suggests the label features carry limited usable information and the model prefers to stay simple. R3-MF comes out close: +0.049.

## 2. Pre-registered decisions

| Decision | Result | Holds? |
|---|---|---|
| Approach 2 is useful: R3's Δ interval excludes 0 | +0.051 [0.039, 0.066] | Yes |
| Approach 2 beats approach 1: R3 − R2-② interval is positive | pre-registered version +0.046 [0.034, 0.057]; sensitivity version +0.026 [0.016, 0.036] | Yes, under both settings |
| The combination adds value: R2+R3 − R2-② interval is positive | +0.049 [0.038, 0.059]; sensitivity version +0.029 [0.020, 0.039] | Yes |
| Reference: R3 − full R2 | −0.005 [−0.014, 0.005] | Tie |
| Reference: R2+R3 − full R2 | −0.002 [−0.011, 0.008] | Tie |

**On the early-stopping cap:** the pre-registration kept R2's "at most 30 epochs". But with only 6,700 votes each epoch is just 7 steps, and in 3 folds R2-②'s chosen setting stopped at epoch 30, possibly undertrained, which would be unfair to approach 1. So I added a sensitivity check that raises the cap to 100 epochs, with selection still based only on inner validation. With the higher cap R2-② rises from +0.006 to +0.026, but is still clearly behind R3. The conclusion holds; the gap just narrows from 0.046 to 0.026. R3 itself chose logistic regression and is not affected by the cap.

## 3. Dollar axis (secondary)

| Router | Main dollar case ΔAPGR$ vs formal baseline | Within-pair ΔAPGR$ (lower bound) |
|---|---|---|
| R1 (all votes) | +0.011 [0.001, 0.020] | +0.017 [−0.006, 0.046] |
| R2 (all votes) | +0.009 [−0.005, 0.022] | +0.013 [−0.012, 0.042] |
| R2-② | −0.002 [−0.010, 0.005] | −0.022 [−0.050, 0.002] |
| R3 | −0.009 [−0.020, 0.001] | −0.016 [−0.036, 0.013] |
| R2 + R3 | −0.015 [−0.030, −0.003] | −0.023 [−0.045, 0.004] |

None of the three routers trained only on ② votes beats the formal baseline on the dollar axis, and that includes R2-②. So this looks like a problem of "② votes only" rather than of the labels themselves. Dollar ranking relies on the absolute size of "win minus loss" for each model pair divided by its price gap, which needs each pair's strength and tie rate estimated well. 90k votes are enough to estimate them; a few thousand ② votes are not. This is a conjecture and was not tested separately.

## 4. By has_single_answer (ΔAPGR vs rating-gap baseline)

| Router | score 0 (3,588 votes) | score 1 (3,281) | score 2 (1,081) | score 3 (373) |
|---|---|---|---|---|
| R2 (all votes) | +0.058 | +0.052 | +0.046 | −0.081 [−0.195, 0.010] |
| R3 | +0.049 | +0.047 | +0.037 | −0.108 [−0.216, −0.028] |
| R2 + R3 | +0.052 | +0.051 | +0.045 | −0.083 |

- In the three groups with scores 0 to 2, approaches 1 and 2 gain about the same; no kind of prompt is clearly better suited to labels.
- Score 3 (a single clear answer) has only 373 votes. Here the rating-gap baseline is already strong, APGR 0.644, a notch above the other groups, and every router does worse than it, R3 the most. A possible reason: for questions with one right answer, the chance that the strong model gets it right already rises steadily with the rating gap, and reading the prompt only adds noise. The sample is small, so treat this as a hint.

## 5. Labels vs approach 1's PC1 coordinate (8,259 dev-set ② prompts, Spearman)

| Field | with t1 | with t1 (out-of-fold) |
|---|---|---|
| is_open_ended | −0.48 | −0.46 |
| is_creative | +0.24 | +0.24 |
| has_format_constraint | +0.23 | +0.23 |
| expected_length | −0.21 | −0.18 |
| has_single_answer | +0.15 | +0.15 |
| difficulty | −0.11 | −0.08 |

- Mean out-of-fold t1 is monotone in is_open_ended: score 0 +0.67, 1 +0.33, 2 −0.34, 3 −0.69. The more open-ended, the lower t1 and the bigger the strong tier's edge, which matches PC1's name "open-ended advice ↔ code and definite-answer questions".
- By task_type, mean t1 from low to high: advice −0.74, factual_qa −0.25, chitchat +0.25, writing +0.47, other +0.49, translation +0.60, math +0.79, code +0.95.
- difficulty is almost unrelated to PC1 (−0.08) and correlates with PC2 (+0.35), consistent with the earlier finding that "PC1 is not difficulty".

So the 7 hand-defined fields largely reproduce the main axis the embedding learned. That also explains why R3 catches up with R2 on very few votes: it gets the most useful signal, "is this open-ended advice or a question with a definite answer", directly instead of learning it from data.

## 6. Cost

In production R3 calls Haiku once per request: $0.22 extra per 1,000 requests and 1.1 s median latency. R1 and R2 cost about $0.02 per 1,000 with 0.19 s latency (4 CPU cores). With quality tied, R3 is more expensive per request, so its advantage is only when training data is scarce.

## 7. Limitations
- Training used only ② votes, which affects the dollar-axis conclusion (section 3).
- R3-LR's C sits at the grid edge, 0.01. As pre-registered, the grid was not extended; this is only reported.
- The 11 refused prompts were filled with the training-fold mode, 0.1% of prompts.
- has_single_answer = 3 has only 373 votes and very wide intervals.
