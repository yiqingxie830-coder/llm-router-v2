# Inference overhead: extra cost, latency and net saving per 1,000 requests

**In one sentence:** routing with R1 or R2 costs about $0.02 per 1,000 requests, only 3% of the $0.77 they save over the formal baseline, for a net saving of about $0.75. The real cost is latency: on 4 CPU cores, computing the prompt vector takes 0.19 s at the median and 1.0 s at P90. In production it should run on a GPU or use a smaller vector model. R3 costs about $0.22 more per 1,000 requests with a median latency of 1.1 s, 10 times the cost of R1 and R2 and 6 times as slow. Those numbers come from the 50-prompt pilot (r3/pilot_summary.md).

## 1. Measurements (scripts/overhead_bench.py → reports/overhead_bench.json)

- **Environment:** cloud container, 4 CPU cores, no GPU, fastembed (ONNX) with 4 threads.
- **Sample:** 200 random prompts from view ② (seed 2026), with a length distribution matching real requests. Median 29 tokens, mean 71, truncated at 512.
- **Method:** vectors are computed one at a time (batch = 1), closer to a single production request.

| Step | Median | P90 | P99 | Mean |
|---|---|---|---|---|
| Prompt vector (multilingual-e5-large) | 188 ms | 1,028 ms | 3,963 ms | 462 ms |
| R1 router head (logistic regression, 1,079-dim dot product) | 0.001 ms | | | |
| R2 router head (1024×128 projection + two model-vector dot products) | 0.014 ms | | | |
| Baseline (rating gap ÷ price gap, table lookup) | ≈ 0 | | | |

Almost all the latency is the vector. Long prompts, truncated to 512 tokens, take 1 to 4 seconds each on CPU.

## 2. Overhead and net saving per 1,000 requests (main dollar case, at CPT 50%)

**How cost is converted:** on average each request occupies all 4 CPU cores for 0.46 s, so 1,000 requests use about 0.128 machine-hours. At about $0.17 per hour for a 4-core cloud machine (the order of AWS c7i.xlarge on-demand pricing, not checked live), that is about $0.022 per 1,000 requests. Even at 10 times that price it would be $0.22, and the conclusion would not change.

| Router | LLM call cost per 1,000 | Routing overhead | Total | Saved vs formal baseline | Saved vs all-strong | Added latency (median / P90) |
|---|---|---|---|---|---|---|
| All to the strong model | $18.89 | 0 | $18.89 | — | — | 0 |
| Formal baseline (rating gap ÷ price gap) | $9.23 | ≈ 0 | $9.23 | — | $9.66 (51%) | ≈ 0 |
| R1 (logistic regression) | $8.46 | $0.022 | $8.48 | **$0.75 (8%)** | $10.41 (55%) | 0.19 s / 1.0 s |
| R2 (matrix factorization) | $8.46 | $0.022 | $8.48 | **$0.75 (8%)** | $10.41 (55%) | 0.19 s / 1.0 s |
| R3 (Haiku labels + linear router) | pending R3 evaluation | $0.22 (no thinking) / $0.23 (thinking low) | | | | 1.1 s / 1.4 s (no thinking); 1.1 s / 2.4 s (thinking low) |

- "LLM call cost" is the cost per 1,000 requests at the point where 50% of the strong-weak gap is recovered in the main case, from section 5 of the main report. CPT is on a 1% grid, so it carries about ±$0.15 of error, an order of magnitude more than the routing overhead.
- In production R3 calls Haiku once per request, so the Batch half price does not apply; standard prices are used. The 50-prompt pilot measured about 1,830 input tokens (about 1,600 of them the labeling instructions plus the structured-output format) and about 66 output tokens, $0.00022 per prompt. So $0.22 per 1,000, which would eat 28% of R1/R2's gain (dev set, $0.77) or 17% (test set, $1.29). For R3 to pay off, its CPT has to be at least $0.22 lower than R2's.
- R3 latency: with thinking off, median 1.1 s and P90 1.4 s; with thinking on (effort low), P90 is 2.4 s, because a few requests think for 300 to 400 tokens.

## 3. Interpretation
1. **Money is not the problem; latency is.** Routing overhead is 0.1% of the strong model's answer cost ($18.89 per 1,000). But about 1 s of extra wait at P90 on CPU is noticeable in a chat product.
2. **Ways to cut latency (not yet measured):**
   - Run on a GPU: e5-large typically takes on the order of 10 to 20 ms per request.
   - Use a smaller model: for example multilingual-e5-small, about 1/5 the parameters of large, though quality has to be re-validated.
   - Truncate at 128 instead of 512 tokens: the median length is only 29 tokens, so most requests are unaffected, but APGR has to be re-validated.
   - Run it in parallel with generating the first token to hide the latency.
3. **R1 and R2 have the same overhead**, because they use the same prompt vector. R2's router head adds only 0.013 ms. So choosing R2 for its gains (strong vs weak +0.054, closely matched pairs +0.133) costs nothing extra.
