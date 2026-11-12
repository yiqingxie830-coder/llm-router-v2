# Approach 1 vs approach 2 (dev set 5-fold, view ②)

8333 ② votes, 8259 prompt groups; main dollar case 6855 votes. Point estimate [95% interval], 1000 paired bootstrap resamples by prompt group.

## Main metric: APGR on call share

| Router | APGR | Δ vs rating-gap baseline | CPT(50%) |
|---|---|---|---|
| rating-gap baseline (BT) | 0.534 [0.523, 0.544] | — | 0.460 [0.440, 0.470] |
| rating-gap baseline with ties (formal) | 0.508 [0.496, 0.518] | -0.026 [-0.039, -0.014] | 0.500 [0.480, 0.520] |
| R1 (approach 1, all votes) | 0.581 [0.569, 0.592] | 0.047 [0.036, 0.058] | 0.390 [0.380, 0.410] |
| R2 (approach 1, all votes) | 0.590 [0.578, 0.601] | 0.056 [0.042, 0.071] | 0.380 [0.360, 0.390] |
| R2-② (approach 1, ② votes only) | 0.540 [0.529, 0.550] | 0.006 [-0.002, 0.014] | 0.460 [0.440, 0.470] |
| R3-LR | 0.585 [0.574, 0.597] | 0.051 [0.039, 0.066] | 0.380 [0.360, 0.390] |
| R3-MF | 0.583 [0.572, 0.595] | 0.049 [0.037, 0.062] | 0.380 [0.370, 0.400] |
| R3 (approach 2, LR/MF chosen per fold) | 0.585 [0.574, 0.597] | 0.051 [0.039, 0.066] | 0.380 [0.360, 0.390] |
| R2 + R3 (combined) | 0.588 [0.577, 0.600] | 0.054 [0.042, 0.068] | 0.380 [0.360, 0.390] |

## Paired differences between approaches

| Comparison | ΔAPGR | ΔAPGR$ (main dollar case) |
|---|---|---|
| R3 (approach 2, LR/MF chosen per fold) − R2-② (approach 1, ② votes only) | 0.046 [0.034, 0.057] | -0.007 [-0.016, 0.003] |
| R2 + R3 (combined) − R2-② (approach 1, ② votes only) | 0.049 [0.038, 0.059] | -0.014 [-0.024, -0.003] |
| R3 (approach 2, LR/MF chosen per fold) − R2 (approach 1, all votes) | -0.005 [-0.014, 0.005] | -0.018 [-0.029, -0.008] |
| R2 + R3 (combined) − R2 (approach 1, all votes) | -0.002 [-0.011, 0.008] | -0.024 [-0.035, -0.015] |
| R2-② (approach 1, ② votes only) − R2 (approach 1, all votes) | -0.050 [-0.062, -0.038] | -0.011 [-0.022, 0.000] |
| R3-LR − R2-② (approach 1, ② votes only) | 0.046 [0.034, 0.057] | -0.007 [-0.017, 0.003] |
| R3-MF − R2-② (approach 1, ② votes only) | 0.043 [0.033, 0.055] | -0.015 [-0.025, -0.005] |

## Main dollar case (pricier strong vs cheaper weak)

| Router | APGR$ | Δ vs formal baseline | CPT$(50%) |
|---|---|---|---|
| rating-gap baseline (BT) | 0.623 [0.609, 0.638] | 0.003 [-0.004, 0.009] | 0.350 [0.330, 0.380] |
| rating-gap baseline with ties (formal) | 0.620 [0.605, 0.636] | — | 0.370 [0.330, 0.400] |
| R1 (approach 1, all votes) | 0.631 [0.615, 0.647] | 0.011 [0.001, 0.020] | 0.320 [0.300, 0.340] |
| R2 (approach 1, all votes) | 0.629 [0.614, 0.646] | 0.009 [-0.005, 0.022] | 0.310 [0.280, 0.340] |
| R2-② (approach 1, ② votes only) | 0.618 [0.603, 0.634] | -0.002 [-0.010, 0.005] | 0.360 [0.330, 0.380] |
| R3-LR | 0.611 [0.596, 0.628] | -0.009 [-0.020, 0.001] | 0.350 [0.320, 0.380] |
| R3-MF | 0.603 [0.587, 0.620] | -0.017 [-0.029, -0.006] | 0.360 [0.320, 0.380] |
| R3 (approach 2, LR/MF chosen per fold) | 0.611 [0.596, 0.628] | -0.009 [-0.020, 0.001] | 0.350 [0.320, 0.380] |
| R2 + R3 (combined) | 0.604 [0.589, 0.622] | -0.015 [-0.030, -0.003] | 0.360 [0.330, 0.390] |

## Within-pair APGR$ (lower bound; 68 model pairs with ≥ 30 votes, 6559 votes)

| Router | within-pair APGR$ | Δ vs formal baseline |
|---|---|---|
| gap ÷ price gap (formal baseline, bucketed win/tie/loss) | 0.511 [0.492, 0.531] | — |
| R1 (approach 1, all votes) | 0.528 [0.511, 0.549] | 0.017 [-0.006, 0.046] |
| R2 (approach 1, all votes) | 0.524 [0.507, 0.544] | 0.013 [-0.012, 0.042] |
| R2-② (approach 1, ② votes only) | 0.489 [0.469, 0.505] | -0.022 [-0.050, 0.002] |
| R3-LR | 0.495 [0.479, 0.517] | -0.016 [-0.036, 0.013] |
| R3-MF | 0.479 [0.464, 0.501] | -0.032 [-0.051, -0.003] |
| R3 (approach 2, LR/MF chosen per fold) | 0.495 [0.479, 0.517] | -0.016 [-0.036, 0.013] |
| R2 + R3 (combined) | 0.488 [0.471, 0.509] | -0.023 [-0.045, 0.004] |

## By has_single_answer (APGR on call share)

| Router | score 0 (3588 votes) | score 1 (3281 votes) | score 2 (1081 votes) | score 3 (373 votes) |
|---|---|---|---|---|
| rating-gap baseline (BT) | 0.526 [0.509, 0.541] | 0.532 [0.517, 0.550] | 0.548 [0.509, 0.583] | 0.644 [0.557, 0.756] |
| rating-gap baseline with ties (formal) | 0.503 [0.487, 0.517] | 0.513 [0.493, 0.527] | 0.504 [0.466, 0.542] | 0.610 [0.503, 0.698] |
| R1 (approach 1, all votes) | 0.570 [0.554, 0.586] | 0.576 [0.558, 0.592] | 0.578 [0.543, 0.621] | 0.576 [0.487, 0.672] |
| R2 (approach 1, all votes) | 0.583 [0.567, 0.599] | 0.584 [0.566, 0.602] | 0.594 [0.555, 0.642] | 0.564 [0.475, 0.656] |
| R2-② (approach 1, ② votes only) | 0.534 [0.518, 0.548] | 0.530 [0.514, 0.547] | 0.553 [0.518, 0.592] | 0.625 [0.543, 0.741] |
| R3-LR | 0.575 [0.560, 0.591] | 0.580 [0.562, 0.597] | 0.585 [0.550, 0.631] | 0.537 [0.445, 0.619] |
| R3-MF | 0.571 [0.556, 0.587] | 0.577 [0.561, 0.594] | 0.584 [0.545, 0.630] | 0.584 [0.506, 0.684] |
| R3 (approach 2, LR/MF chosen per fold) | 0.575 [0.560, 0.591] | 0.580 [0.562, 0.597] | 0.585 [0.550, 0.631] | 0.537 [0.445, 0.619] |
| R2 + R3 (combined) | 0.577 [0.561, 0.593] | 0.583 [0.566, 0.601] | 0.594 [0.557, 0.637] | 0.562 [0.481, 0.652] |

Gain over the rating-gap baseline:

| Router | score 0 | score 1 | score 2 | score 3 |
|---|---|---|---|---|
| rating-gap baseline with ties (formal) | -0.023 [-0.041, -0.006] | -0.019 [-0.042, -0.005] | -0.045 [-0.087, -0.001] | -0.034 [-0.164, 0.053] |
| R1 (approach 1, all votes) | 0.045 [0.030, 0.060] | 0.043 [0.027, 0.061] | 0.030 [-0.005, 0.070] | -0.069 [-0.171, 0.015] |
| R2 (approach 1, all votes) | 0.058 [0.038, 0.077] | 0.052 [0.032, 0.073] | 0.046 [0.006, 0.094] | -0.081 [-0.195, 0.010] |
| R2-② (approach 1, ② votes only) | 0.008 [-0.004, 0.019] | -0.002 [-0.014, 0.011] | 0.005 [-0.022, 0.038] | -0.020 [-0.096, 0.056] |
| R3-LR | 0.049 [0.033, 0.068] | 0.047 [0.028, 0.067] | 0.037 [0.001, 0.084] | -0.107 [-0.215, -0.028] |
| R3-MF | 0.045 [0.028, 0.063] | 0.045 [0.027, 0.064] | 0.036 [0.003, 0.076] | -0.061 [-0.139, 0.009] |
| R3 (approach 2, LR/MF chosen per fold) | 0.049 [0.033, 0.068] | 0.047 [0.028, 0.067] | 0.037 [0.001, 0.084] | -0.108 [-0.216, -0.028] |
| R2 + R3 (combined) | 0.052 [0.034, 0.069] | 0.051 [0.032, 0.071] | 0.045 [0.011, 0.087] | -0.083 [-0.178, -0.007] |

## Rank correlation of labels with approach 1's PC1 coordinate (8259 dev-set ② prompts)

t1 = PC1 coordinate from vectors trained on the full dev set; t1_oof = out-of-fold version (each prompt's coordinate comes from a fold that never saw it).

| Field | with t1 | with t1_oof | with t2 |
|---|---|---|---|
| is_open_ended | -0.479 | -0.462 | 0.175 |
| has_single_answer | 0.152 | 0.151 | 0.034 |
| is_creative | 0.237 | 0.236 | 0.038 |
| has_format_constraint | 0.23 | 0.234 | 0.14 |
| difficulty | -0.106 | -0.082 | 0.349 |
| expected_length | -0.207 | -0.179 | 0.302 |

Mean t1_oof by is_open_ended: score 0: 0.673 (1234 prompts); score 1: 0.334 (2698 prompts); score 2: -0.337 (3172 prompts); score 3: -0.688 (1146 prompts)

Mean t1_oof by task_type: advice -0.735 (2390); factual_qa -0.246 (2357); chitchat 0.248 (317); writing 0.474 (1345); other 0.494 (339); translation 0.598 (128); math 0.793 (326); code 0.947 (1048)