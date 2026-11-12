# R2 (MF) hyperparameter choice (each outer fold uses only its own train votes; 80/20 split by prompt group for inner validation)

k = 64, batch size 1024, Adam 1e-3, early-stopping patience 3 epochs, λ ∈ [1e-05, 0.0001, 0.001]

| Context | λ | Inner validation log loss | Early-stop epoch |
|---|---|---|---|
| cv0 | 1e-05 | 1.0328 | 17 |
| cv0 | 0.0001 | 1.0356 | 30 |
| cv0 | 0.001 | 1.0430 | 20 |
| cv0 chosen | **1e-05** | | 17 |
| cv1 | 1e-05 | 1.0332 | 13 |
| cv1 | 0.0001 | 1.0416 | 14 |
| cv1 | 0.001 | 1.0420 | 14 |
| cv1 chosen | **1e-05** | | 13 |
| cv2 | 1e-05 | 1.0360 | 14 |
| cv2 | 0.0001 | 1.0378 | 29 |
| cv2 | 0.001 | 1.0439 | 19 |
| cv2 chosen | **1e-05** | | 14 |
| cv3 | 1e-05 | 1.0362 | 19 |
| cv3 | 0.0001 | 1.0391 | 30 |
| cv3 | 0.001 | 1.0462 | 28 |
| cv3 chosen | **1e-05** | | 19 |
| cv4 | 1e-05 | 1.0323 | 15 |
| cv4 | 0.0001 | 1.0413 | 14 |
| cv4 | 0.001 | 1.0418 | 14 |
| cv4 chosen | **1e-05** | | 15 |