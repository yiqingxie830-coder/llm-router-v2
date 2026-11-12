# Evaluation: R1_3class_win (test set)

Format: point estimate [95% interval]; 1000 bootstrap resamples by prompt group.

> The pooled view ① depends on Arena's pair sampling and does not represent any single product scenario; view ② and the buckets are the scenario references.

## ① pooled  (10135 votes, 9538 prompt groups; quality all-strong 0.6015, all-weak 0.3985)

| Router | APGR | CPT(50%) | CPT(80%) | ΔAPGR vs rating-gap baseline |
|---|---|---|---|---|
| random | 0.503 [0.479, 0.526] | 0.52 [0.46, 0.55] | 0.8 [0.77, 0.83] | -0.176 [-0.212, -0.143] |
| rating-gap baseline | 0.679 [0.654, 0.709] | 0.29 [0.26, 0.31] | 0.55 [0.51, 0.59] | — |
| R1_3class_win | 0.695 [0.67, 0.724] | 0.25 [0.23, 0.28] | 0.51 [0.46, 0.57] | 0.016 [0.002, 0.031] |

End points: all weak = (0% strong calls, PGR 0); all strong = (100% strong calls, PGR 1).

## ① gap <50  (4069 votes, 3932 prompt groups; quality all-strong 0.5375, all-weak 0.4625)

| Router | APGR | CPT(50%) | CPT(80%) | ΔAPGR vs rating-gap baseline |
|---|---|---|---|---|
| random | 0.496 [0.383, 0.598] | 0.54 [0.32, 0.67] | 0.8 [0.62, 0.9] | -0.264 [-0.475, -0.124] |
| rating-gap baseline | 0.76 [0.642, 0.941] | 0.22 [0.16, 0.31] | 0.35 [0.24, 0.61] | — |
| R1_3class_win | 0.678 [0.574, 0.832] | 0.34 [0.1, 0.48] | 0.58 [0.39, 0.73] | -0.082 [-0.211, 0.029] |

End points: all weak = (0% strong calls, PGR 0); all strong = (100% strong calls, PGR 1).

## ① gap 50–150  (5038 votes, 4847 prompt groups; quality all-strong 0.6338, all-weak 0.3662)

| Router | APGR | CPT(50%) | CPT(80%) | ΔAPGR vs rating-gap baseline |
|---|---|---|---|---|
| random | 0.503 [0.478, 0.528] | 0.52 [0.45, 0.55] | 0.8 [0.77, 0.83] | -0.066 [-0.1, -0.032] |
| rating-gap baseline | 0.569 [0.544, 0.594] | 0.4 [0.37, 0.44] | 0.72 [0.66, 0.78] | — |
| R1_3class_win | 0.617 [0.593, 0.645] | 0.33 [0.3, 0.37] | 0.66 [0.62, 0.71] | 0.048 [0.028, 0.072] |

End points: all weak = (0% strong calls, PGR 0); all strong = (100% strong calls, PGR 1).

## ① gap >150  (1028 votes, 1019 prompt groups; quality all-strong 0.6965, all-weak 0.3035)

| Router | APGR | CPT(50%) | CPT(80%) | ΔAPGR vs rating-gap baseline |
|---|---|---|---|---|
| random | 0.505 [0.468, 0.54] | 0.49 [0.42, 0.56] | 0.82 [0.77, 0.86] | -0.031 [-0.08, 0.014] |
| rating-gap baseline | 0.537 [0.501, 0.571] | 0.48 [0.4, 0.54] | 0.76 [0.71, 0.8] | — |
| R1_3class_win | 0.569 [0.536, 0.603] | 0.39 [0.35, 0.45] | 0.73 [0.68, 0.8] | 0.032 [-0.004, 0.073] |

End points: all weak = (0% strong calls, PGR 0); all strong = (100% strong calls, PGR 1).

## ② strong vs weak  (999 votes, 988 prompt groups; quality all-strong 0.6942, all-weak 0.3058)

| Router | APGR | CPT(50%) | CPT(80%) | ΔAPGR vs rating-gap baseline |
|---|---|---|---|---|
| random | 0.506 [0.473, 0.543] | 0.48 [0.42, 0.56] | 0.82 [0.75, 0.86] | -0.023 [-0.075, 0.03] |
| rating-gap baseline | 0.529 [0.493, 0.566] | 0.46 [0.37, 0.53] | 0.76 [0.7, 0.84] | — |
| R1_3class_win | 0.575 [0.54, 0.615] | 0.39 [0.35, 0.44] | 0.72 [0.65, 0.78] | 0.046 [0.009, 0.087] |

End points: all weak = (0% strong calls, PGR 0); all strong = (100% strong calls, PGR 1).
