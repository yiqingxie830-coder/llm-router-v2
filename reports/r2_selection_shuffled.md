# R2 (MF) hyperparameter choice (each outer fold uses only its own train votes; 80/20 split by prompt group for inner validation)

k = 64, batch size 1024, Adam 1e-3, early-stopping patience 3 epochs, λ ∈ [0.0, 1e-06, 1e-05, 0.0001]

| Context | λ | Inner validation log loss | Early-stop epoch |
|---|---|---|---|
| cv0 | 0.0 | 1.0437 | 8 |
| cv0 | 1e-06 | 1.0437 | 8 |
| cv0 | 1e-05 | 1.0425 | 17 |
| cv0 | 0.0001 | 1.0429 | 18 |
| cv0 chosen | **1e-05** | | 17 |
| cv1 | 0.0 | 1.0424 | 8 |
| cv1 | 1e-06 | 1.0424 | 8 |
| cv1 | 1e-05 | 1.0420 | 13 |
| cv1 | 0.0001 | 1.0419 | 12 |
| cv1 chosen | **0.0001** | | 12 |
| cv2 | 0.0 | 1.0443 | 9 |
| cv2 | 1e-06 | 1.0441 | 11 |
| cv2 | 1e-05 | 1.0438 | 13 |
| cv2 | 0.0001 | 1.0438 | 19 |
| cv2 chosen | **0.0001** | | 19 |
| cv3 | 0.0 | 1.0470 | 10 |
| cv3 | 1e-06 | 1.0468 | 10 |
| cv3 | 1e-05 | 1.0461 | 15 |
| cv3 | 0.0001 | 1.0462 | 23 |
| cv3 chosen | **1e-05** | | 15 |
| cv4 | 0.0 | 1.0422 | 11 |
| cv4 | 1e-06 | 1.0420 | 11 |
| cv4 | 1e-05 | 1.0415 | 13 |
| cv4 | 0.0001 | 1.0416 | 14 |
| cv4 chosen | **1e-05** | | 13 |