# scripts/

Run each script as `python -I scripts/<name>.py <repo root> [args]`; the header of every file documents its arguments and outputs. The core path in the main README covers stages 0–5.

| Stage | Scripts (run in this order) | Main outputs |
|---|---|---|
| 0 Data | `data/download.py`, `inspect_raw.py` | raw parquet, `reports/step0_raw_overview.md` |
| 1 Clean | `clean.py` | `data/arena_clean.parquet` |
| 2–3 Ratings, labels | `tier_models.py` → `make_labels.py` | `reports/step2_ratings.csv`, `data/arena_labeled.parquet` |
| 4 Split, fold ratings | `split.py` → `fold_ratings.py` → `plot_rank_rating.py` | `data/fold_labels.parquet`, `reports/step4_rank_rating.png` |
| 5 Evaluation | `evaluate.py` | APGR / CPT tables and curves |
| S1 Dollar axis | `cost_table.py raw/` → `cost_views.py` → `tie_baselines.py` → `evaluate_cost.py` | `data/vote_costs.parquet`, dollar baselines |
| S2 Embeddings | `export_prompts_for_embedding.py` → `colab_embed.py` (GPU) or `embed_prompts.py` (CPU) | `data/prompt_emb_e5large.npy` |
| Approach 1 (R1, R2) | `train_r1.py` → `train_r1b.py` → `r1_csweep.py`; `train_r2_mf.py` → `r2_sweep.py` → `r2_final.py` | `preds/`, `reports/stage2_r1_r2_report.md` |
| R2 interpretation | `r2_sweep.py export` → `plot_r2_vectors.py` → `prompt_axes.py raw/` → `prompt_axes_report.py` → `prompt_axes_en.py` | `reports/r2_prompt_axes.md`, PCA figure |
| Approach 2 (R3) | `r3_export.py` → `r3/r3_label.py` (API) → `r3_train.py` → `r3_eval.py` | `reports/r3_dev_report.md` |
| Overhead | `overhead_bench.py` | `reports/overhead_table.md` |

Only the main write-ups are committed in `reports/`. The scripts also write these working files, which the committed reports sometimes cite by name:

- step reports: `step0_raw_overview.md` (`inspect_raw.py`) through `step4_*` (`split.py`, `fold_ratings.py`, `plot_rank_rating.py`), `stage2_step1_cost_views.md` (`cost_views.py`)
- per-router evaluation tables and curves `eval_<name>[_final].md/.png` (`evaluate.py`, `evaluate_cost.py`)
- tuning logs: `r1_c_selection.md` (`r1_csweep.py`), `r2_sweep.md` (`r2_sweep.py`)
- `r3_compare*.md` (`r3_eval.py`), `r2_pc1_english_check.md` (`prompt_axes_en.py`)

A few hand-written working notes (R2 implementation, hyperparameter selection rounds) were removed from `reports/` to keep it short; they remain in the git history (commit 092369f).

Scripts call each other by file name (for example `r2_final.py` loads `train_r2_mf.py`), so keep the names as they are. Running the pipeline rewrites the files in `reports/`.

The test set is sealed. Scripts that take `--final` (R1, R2, R3, baselines, evaluation) were run on it twice under written pre-registrations (`reports/final_prereg.md`, `reports/r3_prereg.md`). The flag stays for transparency, not for new runs.
