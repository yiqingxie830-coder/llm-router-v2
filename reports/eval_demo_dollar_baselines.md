# Dollar-axis evaluation: demo_dollar_baselines (dev set 5-fold, eval votes of all folds combined)

Only votes where the higher-rated model has the higher typical cost. Format: point estimate [95% interval], 1000 bootstrap resamples by prompt group.
The x-axis of APGR$ / CPT$ is normalized dollar cost (0 = all to the lower-rated model, 1 = all to the higher-rated model).

## ① pooled ($)  (51815 votes; per 1,000 requests: all to lower-rated $3.267, all to higher-rated $16.317)

| Router | APGR$ | CPT$(50%) | CPT$(80%) | ΔAPGR$ vs formal baseline | For reference: APGR on call share |
|---|---|---|---|---|---|
| random | 0.489 [0.478, 0.501] | 0.53 [0.51, 0.54] | 0.81 [0.8, 0.82] | -0.278 [-0.295, -0.263] | 0.497 |
| rating-gap baseline (control, ignores price) | 0.642 [0.631, 0.655] | 0.29 [0.27, 0.31] | 0.65 [0.63, 0.67] | -0.125 [-0.137, -0.114] | 0.665 |
| gap ÷ price gap (formal baseline, bucketed win/tie/loss) | 0.768 [0.758, 0.778] | 0.16 [0.14, 0.17] | 0.42 [0.4, 0.43] | — | 0.543 |
| gap ÷ price gap (control: 3-class logistic version) | 0.768 [0.759, 0.779] | 0.15 [0.14, 0.16] | 0.41 [0.39, 0.43] | 0.001 [-0.001, 0.003] | 0.541 |
| gap ÷ price gap (control: BT half-vote version, former formal baseline) | 0.768 [0.758, 0.778] | 0.15 [0.14, 0.16] | 0.41 [0.39, 0.43] | 0.0 [-0.003, 0.003] | 0.557 |

## ① gap <50 ($)  (18463 votes; per 1,000 requests: all to lower-rated $3.154, all to higher-rated $14.221)

| Router | APGR$ | CPT$(50%) | CPT$(80%) | ΔAPGR$ vs formal baseline | For reference: APGR on call share |
|---|---|---|---|---|---|
| random | 0.469 [0.435, 0.508] | 0.54 [0.44, 0.61] | 0.84 [0.79, 0.89] | -0.35 [-0.401, -0.3] | 0.474 |
| rating-gap baseline (control, ignores price) | 0.652 [0.608, 0.702] | 0.26 [0.21, 0.36] | 0.61 [0.52, 0.71] | -0.167 [-0.21, -0.122] | 0.638 |
| gap ÷ price gap (formal baseline, bucketed win/tie/loss) | 0.819 [0.787, 0.852] | 0.09 [0.07, 0.12] | 0.29 [0.24, 0.37] | — | 0.54 |
| gap ÷ price gap (control: 3-class logistic version) | 0.82 [0.79, 0.853] | 0.09 [0.07, 0.12] | 0.29 [0.24, 0.4] | 0.0 [-0.01, 0.011] | 0.529 |
| gap ÷ price gap (control: BT half-vote version, former formal baseline) | 0.814 [0.778, 0.853] | 0.09 [0.06, 0.12] | 0.31 [0.25, 0.4] | -0.006 [-0.019, 0.012] | 0.551 |

## ① gap 50–150 ($)  (25868 votes; per 1,000 requests: all to lower-rated $3.128, all to higher-rated $17.06)

| Router | APGR$ | CPT$(50%) | CPT$(80%) | ΔAPGR$ vs formal baseline | For reference: APGR on call share |
|---|---|---|---|---|---|
| random | 0.495 [0.478, 0.511] | 0.52 [0.5, 0.55] | 0.8 [0.78, 0.82] | -0.258 [-0.279, -0.238] | 0.504 |
| rating-gap baseline (control, ignores price) | 0.596 [0.582, 0.611] | 0.33 [0.3, 0.35] | 0.76 [0.73, 0.79] | -0.157 [-0.172, -0.141] | 0.576 |
| gap ÷ price gap (formal baseline, bucketed win/tie/loss) | 0.753 [0.741, 0.764] | 0.13 [0.12, 0.14] | 0.46 [0.43, 0.49] | — | 0.516 |
| gap ÷ price gap (control: 3-class logistic version) | 0.753 [0.742, 0.764] | 0.13 [0.12, 0.14] | 0.46 [0.44, 0.49] | 0.0 [-0.002, 0.004] | 0.515 |
| gap ÷ price gap (control: BT half-vote version, former formal baseline) | 0.753 [0.742, 0.765] | 0.13 [0.11, 0.14] | 0.46 [0.43, 0.49] | 0.0 [-0.002, 0.004] | 0.516 |

## ① gap >150 ($)  (7484 votes; per 1,000 requests: all to lower-rated $4.029, all to higher-rated $18.918)

| Router | APGR$ | CPT$(50%) | CPT$(80%) | ΔAPGR$ vs formal baseline | For reference: APGR on call share |
|---|---|---|---|---|---|
| random | 0.489 [0.474, 0.503] | 0.52 [0.49, 0.55] | 0.82 [0.81, 0.84] | -0.124 [-0.146, -0.104] | 0.499 |
| rating-gap baseline (control, ignores price) | 0.497 [0.481, 0.515] | 0.52 [0.49, 0.54] | 0.79 [0.77, 0.82] | -0.116 [-0.14, -0.091] | 0.531 |
| gap ÷ price gap (formal baseline, bucketed win/tie/loss) | 0.613 [0.6, 0.628] | 0.36 [0.33, 0.41] | 0.68 [0.65, 0.71] | — | 0.493 |
| gap ÷ price gap (control: 3-class logistic version) | 0.613 [0.599, 0.627] | 0.37 [0.34, 0.39] | 0.67 [0.65, 0.7] | 0.0 [-0.007, 0.006] | 0.499 |
| gap ÷ price gap (control: BT half-vote version, former formal baseline) | 0.613 [0.599, 0.628] | 0.37 [0.34, 0.39] | 0.67 [0.65, 0.69] | 0.001 [-0.007, 0.006] | 0.499 |

## ② pricier strong vs cheaper weak (main dollar case)  (6855 votes; per 1,000 requests: all to lower-rated $3.551, all to higher-rated $18.891)

| Router | APGR$ | CPT$(50%) | CPT$(80%) | ΔAPGR$ vs formal baseline | For reference: APGR on call share |
|---|---|---|---|---|---|
| random | 0.494 [0.479, 0.509] | 0.51 [0.49, 0.54] | 0.82 [0.8, 0.84] | -0.126 [-0.15, -0.105] | 0.501 |
| rating-gap baseline (control, ignores price) | 0.498 [0.483, 0.514] | 0.5 [0.47, 0.53] | 0.8 [0.77, 0.83] | -0.122 [-0.145, -0.097] | 0.531 |
| gap ÷ price gap (formal baseline, bucketed win/tie/loss) | 0.62 [0.605, 0.636] | 0.37 [0.33, 0.4] | 0.67 [0.63, 0.69] | — | 0.495 |
| gap ÷ price gap (control: 3-class logistic version) | 0.623 [0.608, 0.638] | 0.35 [0.33, 0.38] | 0.66 [0.63, 0.69] | 0.003 [-0.004, 0.01] | 0.501 |
| gap ÷ price gap (control: BT half-vote version, former formal baseline) | 0.623 [0.608, 0.638] | 0.35 [0.33, 0.38] | 0.66 [0.63, 0.69] | 0.003 [-0.003, 0.01] | 0.501 |

## ② within-pair APGR$  (68 model pairs with ≥ 30 votes, 6559 votes)

APGR$ is computed inside each model pair and then weighted; the price gap is fixed within a pair, so this only measures ranking by prompt. The vote-weighted version skips pairs whose quality gap < 0.05; the pooled version weights by votes × quality gap and cannot blow up.

| Router | vote-weighted | Δ vs formal baseline | pooled | Δ vs formal baseline |
|---|---|---|---|---|
| random | 0.519 [0.501, 0.535] | 0.011 [-0.019, 0.033] | 0.517 [0.5, 0.53] | 0.011 [-0.015, 0.028] |
| rating-gap baseline (control, ignores price) | 0.497 [0.476, 0.514] | -0.01 [-0.043, 0.01] | 0.496 [0.476, 0.51] | -0.011 [-0.037, 0.006] |
| gap ÷ price gap (formal baseline, bucketed win/tie/loss) | 0.508 [0.491, 0.53] | — | 0.506 [0.493, 0.526] | — |
| gap ÷ price gap (control: 3-class logistic version) | 0.493 [0.472, 0.512] | -0.015 [-0.044, 0.007] | 0.492 [0.473, 0.509] | -0.015 [-0.039, 0.004] |
| gap ÷ price gap (control: BT half-vote version, former formal baseline) | 0.496 [0.475, 0.513] | -0.012 [-0.043, 0.01] | 0.495 [0.476, 0.51] | -0.012 [-0.037, 0.006] |

## Sensitivity of the formal baseline to the price-gap floor (APGR$ point estimate)

| View | share of votes at the floor | floor $0.1/1k | floor $0.25/1k | floor $1.0/1k | log: (2p-1) / log(1 + cost gap/floor) |
|---|---|---|---|---|---|
| ① pooled | 0.0518 | 0.768 | 0.768 | 0.767 | 0.733 |
| ② main dollar case | 0.0381 | 0.62 | 0.62 | 0.62 | 0.622 |
