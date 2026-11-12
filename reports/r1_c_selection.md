# R1 (3-class): choice of regularization C and sensitivity

**How C is chosen:** each outer fold uses only its own train votes, splits them 80/20 by prompt group, trains on 80% and computes log loss on 20%, takes the C with the lowest loss, then retrains on the whole train part and predicts that fold's eval votes. Neither eval votes nor the test set take part in choosing C.

Grid C ∈ [0.01, 0.03, 0.1, 0.3, 1.0]. Chosen per fold: cv0 → 0.1, cv1 → 0.1, cv2 → 0.1, cv3 → 0.1, cv4 → 0.1

| C | Inner validation log loss (5-fold mean) | Strong vs weak APGR (call share) | Main dollar case APGR$ | Main case CPT$(50%) |
|---|---|---|---|---|
| 0.01 | 1.0386 | 0.568 | 0.628 | 0.33 |
| 0.03 | 1.0368 | 0.577 | 0.630 | 0.33 |
| 0.1 | 1.0360 | 0.581 | 0.631 | 0.32 |
| 0.3 | 1.0370 | 0.579 | 0.631 | 0.32 |
| 1.0 | 1.0391 | 0.576 | 0.631 | 0.33 |
| nested selection | — | 0.581 | 0.631 | 0.32 |

Point estimates, no bootstrap; the sensitivity rows only show whether the conclusion depends on C.