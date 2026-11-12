# Evaluation: R2_MF_win (dev set 5-fold CV, eval votes of all folds combined)

Format: point estimate [95% interval]; 1000 bootstrap resamples by prompt group.

> The pooled view ① depends on Arena's pair sampling and does not represent any single product scenario; view ② and the buckets are the scenario references.

## ① pooled  (90923 votes, 85835 prompt groups; quality all-strong 0.6028, all-weak 0.3972)

| Router | APGR | CPT(50%) | CPT(80%) | ΔAPGR vs rating-gap baseline |
|---|---|---|---|---|
| random | 0.497 [0.489, 0.505] | 0.52 [0.51, 0.53] | 0.81 [0.8, 0.82] | -0.171 [-0.183, -0.16] |
| rating-gap baseline | 0.668 [0.66, 0.677] | 0.28 [0.27, 0.29] | 0.57 [0.55, 0.59] | — |
| R2_MF_win | 0.711 [0.702, 0.721] | 0.25 [0.24, 0.26] | 0.5 [0.49, 0.52] | 0.043 [0.038, 0.049] |

End points: all weak = (0% strong calls, PGR 0); all strong = (100% strong calls, PGR 1).

## ① gap <50  (36414 votes, 35240 prompt groups; quality all-strong 0.5457, all-weak 0.4543)

| Router | APGR | CPT(50%) | CPT(80%) | ΔAPGR vs rating-gap baseline |
|---|---|---|---|---|
| random | 0.485 [0.458, 0.511] | 0.53 [0.46, 0.58] | 0.83 [0.8, 0.86] | -0.16 [-0.201, -0.12] |
| rating-gap baseline | 0.645 [0.617, 0.675] | 0.32 [0.28, 0.36] | 0.59 [0.51, 0.66] | — |
| R2_MF_win | 0.778 [0.743, 0.817] | 0.22 [0.2, 0.25] | 0.44 [0.4, 0.48] | 0.133 [0.1, 0.165] |

End points: all weak = (0% strong calls, PGR 0); all strong = (100% strong calls, PGR 1).

## ① gap 50–150  (45867 votes, 44200 prompt groups; quality all-strong 0.6249, all-weak 0.3751)

| Router | APGR | CPT(50%) | CPT(80%) | ΔAPGR vs rating-gap baseline |
|---|---|---|---|---|
| random | 0.5 [0.492, 0.509] | 0.52 [0.5, 0.53] | 0.8 [0.79, 0.81] | -0.08 [-0.092, -0.067] |
| rating-gap baseline | 0.58 [0.571, 0.589] | 0.39 [0.37, 0.4] | 0.71 [0.69, 0.73] | — |
| R2_MF_win | 0.64 [0.63, 0.649] | 0.32 [0.31, 0.33] | 0.62 [0.6, 0.64] | 0.059 [0.051, 0.069] |

End points: all weak = (0% strong calls, PGR 0); all strong = (100% strong calls, PGR 1).

## ① gap >150  (8642 votes, 8568 prompt groups; quality all-strong 0.7264, all-weak 0.2736)

| Router | APGR | CPT(50%) | CPT(80%) | ΔAPGR vs rating-gap baseline |
|---|---|---|---|---|
| random | 0.5 [0.49, 0.51] | 0.51 [0.49, 0.52] | 0.81 [0.79, 0.82] | -0.03 [-0.044, -0.015] |
| rating-gap baseline | 0.53 [0.519, 0.54] | 0.47 [0.45, 0.48] | 0.77 [0.75, 0.78] | — |
| R2_MF_win | 0.584 [0.572, 0.595] | 0.38 [0.36, 0.4] | 0.71 [0.68, 0.73] | 0.054 [0.041, 0.067] |

End points: all weak = (0% strong calls, PGR 0); all strong = (100% strong calls, PGR 1).

## ② strong vs weak  (8333 votes, 8259 prompt groups; quality all-strong 0.7243, all-weak 0.2757)

| Router | APGR | CPT(50%) | CPT(80%) | ΔAPGR vs rating-gap baseline |
|---|---|---|---|---|
| random | 0.504 [0.494, 0.515] | 0.5 [0.48, 0.52] | 0.81 [0.79, 0.82] | -0.03 [-0.045, -0.014] |
| rating-gap baseline | 0.534 [0.523, 0.545] | 0.46 [0.44, 0.47] | 0.77 [0.75, 0.79] | — |
| R2_MF_win | 0.588 [0.578, 0.6] | 0.37 [0.36, 0.39] | 0.7 [0.67, 0.72] | 0.054 [0.043, 0.068] |

End points: all weak = (0% strong calls, PGR 0); all strong = (100% strong calls, PGR 1).
