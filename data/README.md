# data/

This folder ships no data, with two exceptions listed below. Everything else is rebuilt by the scripts.

## Source

- **lmarena-ai/arena-human-preference-140k** on Hugging Face: <https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k>
  135,634 text-only battles, Apr–Jul 2025, 53 models.
- License (from the dataset card): user prompts are **CC-BY-4.0**; model outputs are governed by each provider's terms of use.
  Example prompts quoted in `reports/` are from this dataset, under CC-BY-4.0.
- Download: `python -I data/download.py raw/` fetches the 7 parquet shards (~1.6 GB) and checks each against the SHA-256 that Hugging Face publishes. Standard library only.

## Exceptions

| File | What | Why it is in the repo |
|---|---|---|
| `data/model_prices.csv` | List prices per model, USD per million input / output tokens, with source and check status for each row | Collected by hand on 2026-11-09 (official pages and the LiteLLM price list); no script can rebuild it. |
| `r3/r3_labels.jsonl` | The 7 semantic labels per prompt (`group_id` + scores only, no prompt text), 9,247 prompts | Re-labeling needs an Anthropic API key (~$1 on the Batch API) and will not reproduce row for row; 11 prompts were refused. |

## Files the scripts write here

| File | Written by |
|---|---|
| `arena_clean.parquet`, `arena_followup_clean.parquet` | `scripts/clean.py` |
| `arena_labeled.parquet` | `scripts/make_labels.py`, then `scripts/split.py` adds split / fold |
| `fold_labels.parquet`, `arena_strong_vs_weak.parquet` | `scripts/fold_ratings.py` |
| `vote_costs.parquet`, `model_cost.csv` | `scripts/cost_table.py` |
| `prompts_for_embedding.parquet` | `scripts/export_prompts_for_embedding.py` |
| `prompt_emb_e5large.npy`, `prompt_emb_index.parquet` | `scripts/colab_embed.py` (GPU) or `scripts/embed_prompts.py` (CPU) |
| `r2_final_params.npz`, `r2_model_vectors*.csv` | `scripts/r2_sweep.py export` |
| `r2_prompt_axes.parquet` | `scripts/prompt_axes.py` |
| `r3_prompts.parquet` | `scripts/r3_export.py` |
