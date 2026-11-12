# Dollar-axis evaluation: R1_3class (test set)

Only votes where the higher-rated model has the higher typical cost. Format: point estimate [95% interval], 1000 bootstrap resamples by prompt group.
The x-axis of APGR$ / CPT$ is normalized dollar cost (0 = all to the lower-rated model, 1 = all to the higher-rated model).

## ① pooled ($)  (5872 votes; per 1,000 requests: all to lower-rated $3.42, all to higher-rated $17.28)

| Router | APGR$ | CPT$(50%) | CPT$(80%) | ΔAPGR$ vs formal baseline | For reference: APGR on call share |
|---|---|---|---|---|---|
| random | 0.517 [0.483, 0.548] | 0.52 [0.43, 0.58] | 0.8 [0.76, 0.84] | -0.263 [-0.31, -0.218] | 0.501 |
| rating-gap baseline (control, ignores price) | 0.647 [0.619, 0.685] | 0.3 [0.23, 0.34] | 0.67 [0.61, 0.72] | -0.133 [-0.165, -0.093] | 0.653 |
| gap ÷ price gap (formal baseline, bucketed win/tie/loss) | 0.78 [0.75, 0.812] | 0.15 [0.11, 0.18] | 0.4 [0.33, 0.45] | — | 0.545 |
| gap ÷ price gap (control: 3-class logistic version) | 0.781 [0.751, 0.814] | 0.14 [0.11, 0.17] | 0.38 [0.32, 0.45] | 0.0 [-0.005, 0.008] | 0.542 |
| gap ÷ price gap (control: BT half-vote version, former formal baseline) | 0.778 [0.747, 0.812] | 0.13 [0.1, 0.17] | 0.38 [0.33, 0.46] | -0.002 [-0.011, 0.008] | 0.559 |
| R1_3class | 0.774 [0.742, 0.807] | 0.13 [0.11, 0.16] | 0.4 [0.33, 0.48] | -0.006 [-0.023, 0.013] | 0.568 |

## ① gap <50 ($)  (2135 votes; per 1,000 requests: all to lower-rated $3.157, all to higher-rated $14.204)

| Router | APGR$ | CPT$(50%) | CPT$(80%) | ΔAPGR$ vs formal baseline | For reference: APGR on call share |
|---|---|---|---|---|---|
| random | 0.502 [0.389, 0.614] | 0.55 [0.25, 0.69] | 0.74 [0.64, 0.87] | -0.344 [-0.507, -0.201] | 0.461 |
| rating-gap baseline (control, ignores price) | 0.711 [0.58, 0.866] | 0.2 [0.12, 0.32] | 0.5 [0.21, 0.81] | -0.135 [-0.264, 0.018] | 0.669 |
| gap ÷ price gap (formal baseline, bucketed win/tie/loss) | 0.846 [0.748, 0.944] | 0.08 [0.02, 0.18] | 0.25 [0.09, 0.51] | — | 0.569 |
| gap ÷ price gap (control: 3-class logistic version) | 0.847 [0.751, 0.945] | 0.09 [0.02, 0.15] | 0.28 [0.09, 0.5] | 0.001 [-0.02, 0.021] | 0.565 |
| gap ÷ price gap (control: BT half-vote version, former formal baseline) | 0.826 [0.727, 0.947] | 0.08 [0.02, 0.18] | 0.28 [0.11, 0.6] | -0.02 [-0.064, 0.044] | 0.577 |
| R1_3class | 0.753 [0.632, 0.896] | 0.07 [0.03, 0.12] | 0.56 [0.08, 0.92] | -0.093 [-0.197, 0.021] | 0.579 |

## ① gap 50–150 ($)  (2816 votes; per 1,000 requests: all to lower-rated $3.382, all to higher-rated $19.537)

| Router | APGR$ | CPT$(50%) | CPT$(80%) | ΔAPGR$ vs formal baseline | For reference: APGR on call share |
|---|---|---|---|---|---|
| random | 0.509 [0.474, 0.549] | 0.53 [0.32, 0.63] | 0.83 [0.78, 0.88] | -0.269 [-0.314, -0.201] | 0.511 |
| rating-gap baseline (control, ignores price) | 0.615 [0.57, 0.659] | 0.3 [0.24, 0.37] | 0.82 [0.65, 0.88] | -0.163 [-0.205, -0.112] | 0.568 |
| gap ÷ price gap (formal baseline, bucketed win/tie/loss) | 0.778 [0.731, 0.809] | 0.13 [0.09, 0.18] | 0.4 [0.32, 0.49] | — | 0.512 |
| gap ÷ price gap (control: 3-class logistic version) | 0.785 [0.733, 0.814] | 0.12 [0.09, 0.18] | 0.4 [0.32, 0.5] | 0.007 [-0.006, 0.016] | 0.511 |
| gap ÷ price gap (control: BT half-vote version, former formal baseline) | 0.784 [0.734, 0.815] | 0.12 [0.09, 0.18] | 0.4 [0.31, 0.5] | 0.005 [-0.008, 0.015] | 0.511 |
| R1_3class | 0.777 [0.737, 0.82] | 0.12 [0.09, 0.18] | 0.41 [0.32, 0.5] | -0.001 [-0.011, 0.025] | 0.53 |

## ① gap >150 ($)  (921 votes; per 1,000 requests: all to lower-rated $4.144, all to higher-rated $17.513)

| Router | APGR$ | CPT$(50%) | CPT$(80%) | ΔAPGR$ vs formal baseline | For reference: APGR on call share |
|---|---|---|---|---|---|
| random | 0.517 [0.471, 0.569] | 0.46 [0.38, 0.55] | 0.81 [0.72, 0.88] | -0.114 [-0.182, -0.04] | 0.498 |
| rating-gap baseline (control, ignores price) | 0.524 [0.479, 0.562] | 0.45 [0.38, 0.54] | 0.78 [0.72, 0.86] | -0.107 [-0.168, -0.044] | 0.533 |
| gap ÷ price gap (formal baseline, bucketed win/tie/loss) | 0.631 [0.587, 0.669] | 0.37 [0.25, 0.44] | 0.65 [0.56, 0.72] | — | 0.491 |
| gap ÷ price gap (control: 3-class logistic version) | 0.639 [0.597, 0.681] | 0.3 [0.25, 0.39] | 0.62 [0.54, 0.71] | 0.008 [-0.008, 0.033] | 0.501 |
| gap ÷ price gap (control: BT half-vote version, former formal baseline) | 0.635 [0.596, 0.68] | 0.3 [0.25, 0.38] | 0.62 [0.54, 0.71] | 0.004 [-0.009, 0.03] | 0.503 |
| R1_3class | 0.643 [0.602, 0.685] | 0.3 [0.24, 0.37] | 0.65 [0.55, 0.72] | 0.012 [-0.013, 0.042] | 0.525 |

## ② pricier strong vs cheaper weak (main dollar case)  (840 votes; per 1,000 requests: all to lower-rated $4.044, all to higher-rated $16.964)

| Router | APGR$ | CPT$(50%) | CPT$(80%) | ΔAPGR$ vs formal baseline | For reference: APGR on call share |
|---|---|---|---|---|---|
| random | 0.533 [0.479, 0.586] | 0.45 [0.37, 0.54] | 0.79 [0.7, 0.86] | -0.077 [-0.15, -0.002] | 0.508 |
| rating-gap baseline (control, ignores price) | 0.509 [0.464, 0.554] | 0.44 [0.38, 0.54] | 0.81 [0.74, 0.88] | -0.1 [-0.174, -0.02] | 0.542 |
| gap ÷ price gap (formal baseline, bucketed win/tie/loss) | 0.61 [0.561, 0.652] | 0.41 [0.3, 0.48] | 0.68 [0.6, 0.73] | — | 0.466 |
| gap ÷ price gap (control: 3-class logistic version) | 0.628 [0.583, 0.667] | 0.32 [0.25, 0.39] | 0.7 [0.58, 0.74] | 0.018 [-0.005, 0.045] | 0.473 |
| gap ÷ price gap (control: BT half-vote version, former formal baseline) | 0.628 [0.584, 0.667] | 0.32 [0.25, 0.39] | 0.7 [0.58, 0.74] | 0.018 [-0.004, 0.045] | 0.473 |
| R1_3class | 0.63 [0.587, 0.673] | 0.31 [0.25, 0.39] | 0.68 [0.59, 0.74] | 0.021 [-0.007, 0.051] | 0.496 |

## ② within-pair APGR$  (3 model pairs with ≥ 30 votes, 98 votes)

APGR$ is computed inside each model pair and then weighted; the price gap is fixed within a pair, so this only measures ranking by prompt. The vote-weighted version skips pairs whose quality gap < 0.05; the pooled version weights by votes × quality gap and cannot blow up.

| Router | vote-weighted | Δ vs formal baseline | pooled | Δ vs formal baseline |
|---|---|---|---|---|
| random | 0.698 [0.375, 0.877] | 0.074 [-0.212, 0.467] | 0.657 [0.419, 0.848] | 0.196 [-0.152, 0.49] |
| rating-gap baseline (control, ignores price) | 0.781 [0.184, 0.836] | 0.156 [-0.346, 0.399] | 0.581 [0.177, 0.773] | 0.12 [-0.318, 0.356] |
| gap ÷ price gap (formal baseline, bucketed win/tie/loss) | 0.624 [0.198, 0.844] | — | 0.461 [0.161, 0.793] | — |
| gap ÷ price gap (control: 3-class logistic version) | 0.651 [0.193, 0.845] | 0.026 [-0.406, 0.41] | 0.497 [0.155, 0.773] | 0.037 [-0.369, 0.404] |
| gap ÷ price gap (control: BT half-vote version, former formal baseline) | 0.729 [0.236, 0.849] | 0.105 [-0.404, 0.406] | 0.502 [0.172, 0.78] | 0.041 [-0.368, 0.391] |
| R1_3class | 0.734 [0.392, 0.889] | 0.11 [-0.223, 0.496] | 0.615 [0.311, 0.841] | 0.154 [-0.198, 0.443] |

## Sensitivity of the formal baseline to the price-gap floor (APGR$ point estimate)

| View | share of votes at the floor | floor $0.1/1k | floor $0.25/1k | floor $1.0/1k | log: (2p-1) / log(1 + cost gap/floor) |
|---|---|---|---|---|---|
| ① pooled | 0.0542 | 0.78 | 0.78 | 0.78 | 0.746 |
| ② main dollar case | 0.0476 | 0.61 | 0.61 | 0.61 | 0.608 |
