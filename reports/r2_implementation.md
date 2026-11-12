# R2 matrix factorization (MF) router: implementation notes

Code: `scripts/train_r2_mf.py` (model and training), `scripts/r2_sweep.py` (hyperparameter sweep, nested selection, model-vector export). The full sweep table is in `reports/r2_sweep.md`.

## 1. Model structure

For one vote (prompt q, higher-rated model hi, lower-rated model lo):

```
z      = W · e_q + b                      e_q: prompt vector (multilingual-e5-large, 1024 dims, L2-normalized)
f(m,q) = w2 · (v_m ⊙ z) + u_m             how well model m suits this prompt
s      = f(hi,q) − f(lo,q) + a · rating gap/100
```

- **Projection:** a single linear layer W (1024 × d) with bias b, **no non-linearity**. The original R2 used d = 64. A sweep over d was added this time; 4 of 5 folds chose d = 128 (section 4), and the exported model-vector table uses d = 128.
- **Model-vector table:** v has shape 53 × d (53 models = every model seen in the dev set), and w2 has d dims, shared by all models. What actually matters is their element-wise product g_m = v_m ⊙ w2, so f(m,q) = g_m · z + u_m. This is how RouteLLM's MF is written.
- **Per-model bias:** yes, u_m (a scalar). The output only uses f(hi) − f(lo), so only differences between u values matter.
- **How the rating gap is used:** through **one learnable coefficient a**, multiplied by "this fold's BT rating gap ÷ 100" and added directly to s. Initial value 0.5; trained on the full dev set it learns a = 0.55. The rating gap does not interact with the prompt and is not split by model pair.

## 2. Output: ordinal logistic, not softmax

The larger s, the more likely the higher-rated model wins. Two thresholds c1 < c2 cut s into "loss < tie < win":

```
P(loss) = σ(c1 − s)
P(tie)  = σ(c2 − s) − σ(c1 − s)
P(win)  = 1 − σ(c2 − s)
```

To keep c1 < c2 it is parameterized as c2 = c1 + softplus(dc). Trained on the full dev set it learns c1 = −0.64, c2 = 0.56. Routing ranks by P(win) − P(loss), on the same target as the 3-class R1 and the tie-aware baseline.

## 3. Training

| Item | Setting |
|---|---|
| Loss | Weighted negative log likelihood: Σ w·(−log P(true class)) ÷ Σ w, with **w = 1/n_votes** (repeated votes on one prompt add up to 1 vote), probabilities clipped below at 1e-7 |
| Regularization | **Explicit L2** added to the loss: λ·(‖W‖² + ‖v‖² + ‖w2‖²). b, u, a and the thresholds are not regularized. Not AdamW weight decay |
| Optimizer | Adam (optax default β), learning rate swept (1e-3 or 3e-3), batch size 1024, reshuffled every epoch |
| Initialization | W ~ N(0, 0.02²), v, w2 ~ N(0, 0.1²), b = u = 0, a = 0.5, c1 = −0.3, dc = 0.6, random seed 2026 |
| Epochs and early stopping | At most 30 epochs. After each epoch the weighted log loss (without the regularization term) is computed on the **inner validation set** (section 4). A drop of less than 1e-5 counts as "no improvement"; 3 epochs in a row without improvement stops training, keeping the best epoch. Then the same setting is retrained for the best number of epochs on the fold's **whole train part** and predicts that fold's eval votes |
| Sample weights | Still 1/n_votes, for both training and validation |

## 4. Hyperparameter sweep

**How the inner validation set is cut:** each outer fold cv k uses only its own train votes, split by prompt-group hash (groups with md5("2026" + group_id) mod 5 == 0 are validation), giving 80% training and 20% validation. A prompt never lands on both sides. R1 used the same split to choose C. Neither eval votes nor the test set take part in selection.

**Selection rule:** each fold takes the combination with the lowest inner validation loss.

**How the sweep went:**
1. The original R2 swept only λ, with d = 64 and learning rate 1e-3 fixed. Round 1 used λ ∈ {1e-5, 1e-4, 1e-3} and chose 1e-5 at the grid edge. Round 2 used {0, 1e-6, 1e-5, 1e-4}: 4 folds chose 1e-5 and cv1 chose 1e-6.
2. This time the grid was completed: d ∈ {16, 64, 128} × learning rate ∈ {1e-3, 3e-3} × λ ∈ {1e-6, 1e-5, 1e-4}, 18 combinations. All 5 folds chose d = 128, again at the grid edge, so d = 256 was added (learning rate 1e-3, 3 values of λ). Epochs are not swept; early stopping sets them for each combination.

**Chosen per fold (among the 21 combinations including d = 256):**

| Fold | d | Learning rate | λ | Early-stop epoch |
|---|---|---|---|---|
| cv0 | 128 | 1e-3 | 1e-5 | 15 |
| cv1 | 256 | 1e-3 | 1e-5 | 13 |
| cv2 | 128 | 1e-3 | 1e-5 | 14 |
| cv3 | 128 | 1e-3 | 1e-5 | 14 |
| cv4 | 128 | 1e-3 | 1e-6 | 11 |

d = 256 has a worse 5-fold mean inner loss (1.0342) than d = 128 (1.0335), so d is no longer at the edge.

**Sensitivity table** (APGR columns are point estimates on the outer eval votes, used only to check whether the conclusion depends on hyperparameters, never for selection):

| d | Learning rate | λ | Inner validation loss | Early-stop epochs (5 folds) | Strong vs weak APGR | Main case APGR$ | Main case CPT$(50%) |
|---|---|---|---|---|---|---|---|
| 16 | 1e-3 | 1e-6 | 1.0350 | 20/18/20/19/19 | 0.589 | 0.625 | 0.32 |
| 16 | 1e-3 | 1e-5 | 1.0346 | 24/19/20/22/21 | 0.591 | 0.627 | 0.31 |
| 16 | 1e-3 | 1e-4 | 1.0400 | 30/12/30/30/14 | 0.561 | 0.632 | 0.32 |
| 16 | 3e-3 | 1e-6 | 1.0352 | 7/7/6/7/7 | 0.589 | 0.628 | 0.30 |
| 16 | 3e-3 | 1e-5 | 1.0347 | 8/11/9/7/8 | 0.589 | 0.625 | 0.32 |
| 16 | 3e-3 | 1e-4 | 1.0433 | 9/3/6/8/6 | 0.531 | 0.624 | 0.35 |
| 64 | 1e-3 | 1e-6 | 1.0350 | 14/13/14/13/11 | 0.589 | 0.627 | 0.32 |
| 64 | 1e-3 | 1e-5 | 1.0341 | 17/13/14/19/15 | 0.588 | 0.627 | 0.32 |
| 64 | 1e-3 | 1e-4 | 1.0391 | 30/14/29/30/14 | 0.565 | 0.629 | 0.32 |
| 64 | 3e-3 | 1e-6 | 1.0353 | 7/7/5/5/6 | 0.587 | 0.628 | 0.31 |
| 64 | 3e-3 | 1e-5 | 1.0347 | 7/8/9/7/7 | 0.586 | 0.623 | 0.32 |
| 64 | 3e-3 | 1e-4 | 1.0367 | 21/3/18/20/21 | 0.575 | 0.630 | 0.32 |
| 128 | 1e-3 | 1e-6 | 1.0345 | 11/11/10/10/11 | 0.590 | 0.628 | 0.32 |
| 128 | 1e-3 | 1e-5 | **1.0335** | 15/13/14/14/11 | 0.589 | 0.630 | 0.31 |
| 128 | 1e-3 | 1e-4 | 1.0397 | 30/12/30/30/14 | 0.563 | 0.633 | 0.32 |
| 128 | 3e-3 | 1e-6 | 1.0359 | 4/6/5/5/5 | 0.584 | 0.626 | 0.31 |
| 128 | 3e-3 | 1e-5 | 1.0347 | 7/8/5/7/7 | 0.586 | 0.629 | 0.31 |
| 128 | 3e-3 | 1e-4 | 1.0432 | 10/3/5/7/6 | 0.532 | 0.624 | 0.35 |
| 256 | 1e-3 | 1e-6 | 1.0356 | 8/9/8/10/7 | 0.587 | 0.630 | 0.31 |
| 256 | 1e-3 | 1e-5 | 1.0342 | 11/13/10/13/11 | 0.589 | 0.629 | 0.31 |
| 256 | 1e-3 | 1e-4 | 1.0406 | 30/12/13/30/12 | 0.547 | 0.626 | 0.33 |
| **Nested selection** | | | | | **0.590** | **0.629** | **0.31** |

**How to read it:**
- As long as λ ≤ 1e-5, for d from 16 to 256 and both learning rates, strong-vs-weak APGR stays between 0.584 and 0.591 and main-case APGR$ between 0.623 and 0.630; the conclusion does not depend on hyperparameters.
- λ = 1e-4 is too strong and drops strong vs weak to 0.53 to 0.575. But the inner validation loss already rules it out.
- The new nested selection (0.590 / 0.629) differs from the original R2 in the main report (0.588 / 0.628) by at most 0.002, so the main report's numbers need no change. `--final` uses the newly chosen d = 128, learning rate 1e-3, λ = 1e-5.

## 5. Shuffled-vector control (`train_r2_mf.py --shuffle`)

- The 95,373 × 1024 prompt-vector table is **randomly permuted by row** (numpy seed 7). Each prompt gets another real prompt's vector, the same one in every fold. It is **not replaced with random noise**, so the vector distribution (norms, language clusters and so on) is fully kept; only the link "vector ↔ this prompt's votes" is cut.
- Fold split, labels, model identity and rating gap are all unchanged (the same fold_labels.parquet is read). λ is re-chosen on the shuffled data with the same inner rule: 3 folds chose 1e-5, cv1 and cv2 chose 1e-4.
- Result: every view falls back to the rating-gap baseline (strong vs weak 0.530 vs baseline 0.534, pooled 0.668 vs 0.668), so R2's gain comes from prompt content.

## 6. Framework and run time

- The framework is **JAX + optax**, jit-compiled on CPU; not PyTorch and not hand-written numpy. The environment is a 4-core cloud CPU with no GPU. To save memory, XLA matrix ops were set to a single thread.
- Each outer fold (inner early-stopping training + retraining on the whole train part) takes about 26 to 44 seconds, about 35 seconds at d = 128 and learning rate 1e-3. So one combination over 5 folds takes about 3 minutes, and 21 combinations in sequence about 1 hour.
- The final model used for export trains for 14 epochs on all 90,923 dev votes in 18 seconds.
- Parameter count (d = 128): W about 131k, v about 6,800, the rest negligible.
- At inference, the cost is mainly computing the e5-large prompt vector. MF itself is one 1024 × 128 matrix product, almost negligible. The inference overhead table gives the numbers.

## 7. Model-vector table (for the projection plot)

Export command: `r2_sweep.py <root> export 128 0.001 1e-5 14`. Training data is the whole 90% dev set (BT ratings also from the dev set); **no test votes took part, and the test set was not predicted.**

| File | Content |
|---|---|
| `data/r2_model_vectors_whitened.csv` | **Recommended for plotting.** 53 rows × (model, n_votes_dev, bias_u, w0…w127) |
| `data/r2_model_vectors.csv` | Raw effective vectors g = v ⊙ w2. 53 rows × (model, n_votes_dev, bias_u, g0…g127) |
| `data/r2_model_vectors_raw_v.csv` | v before multiplying by w2, for reference only |
| `data/r2_model_vectors_meta.json` | d, learning rate, λ, epochs, a, c1, c2, w2 |

**Why the whitened version is recommended:**
1. g is not unique. Changing W's basis while transforming g back gives exactly the same output. Adding the same vector to every g also leaves the output unchanged (only differences between two models are used). So distances and angles between raw g vectors partly reflect training randomness.
2. The whitened version subtracts the mean over the 53 models, then multiplies by the square root of the covariance of the dev-set prompt projections z. After that, **the Euclidean distance between two models = the standard deviation, over all dev-set prompts, of the difference in how well they suit each prompt**. This quantity is independent of the basis and reads directly as "how different these two models' strengths are". The whitened bias_u also has its mean subtracted.
3. PCA on the whitened version: the first two components explain 65% and 16% of the variance, so a 2-D projection keeps most of the structure.

**Caveats:**
- The table has **53 rows, not 51**. Every model seen in the dev set has a vector, including two with very few votes: qwen3-coder-480b-a35b-instruct has 1 vote and a vector near 0; gemini-2.0-flash-thinking-exp-01-21 has 251 votes. For plotting, filter by n_votes_dev, for example ≥ 300, which leaves 51.
- A model's overall strength is carried mainly by the rating-gap term (a · rating gap). So u and g represent the prompt-dependent part "beyond the rating gap", not a model's overall ability.
