# Stage 2 Step 1: vote counts for the dollar view (dev set 5-fold, eval votes of all folds combined)

## ① all model pairs (90923 votes)

| cls                             |     votes |   share |   higher_rated_clear_win |   tie_rate |
|:--------------------------------|----------:|--------:|-------------------------:|-----------:|
| higher-rated costs less or same | 30111.000 |   0.331 |                    0.451 |      0.271 |
| higher-rated costs more         | 51815.000 |   0.570 |                    0.478 |      0.271 |
| no price                        |  8997.000 |   0.099 |                    0.455 |      0.276 |

## ① gap <50 (36414 votes)

| cls                             |     votes |   share |   higher_rated_clear_win |   tie_rate |
|:--------------------------------|----------:|--------:|-------------------------:|-----------:|
| higher-rated costs less or same | 14108.000 |   0.387 |                    0.400 |      0.287 |
| higher-rated costs more         | 18463.000 |   0.507 |                    0.400 |      0.295 |
| no price                        |  3843.000 |   0.106 |                    0.398 |      0.294 |

## ① gap 50–150 (45867 votes)

| cls                             |     votes |   share |   higher_rated_clear_win |   tie_rate |
|:--------------------------------|----------:|--------:|-------------------------:|-----------:|
| higher-rated costs less or same | 15385.000 |   0.335 |                    0.493 |      0.261 |
| higher-rated costs more         | 25868.000 |   0.564 |                    0.490 |      0.273 |
| no price                        |  4614.000 |   0.101 |                    0.486 |      0.266 |

## ① gap >150 (8642 votes)

| cls                             |    votes |   share |   higher_rated_clear_win |   tie_rate |
|:--------------------------------|---------:|--------:|-------------------------:|-----------:|
| higher-rated costs less or same |  618.000 |   0.072 |                    0.608 |      0.173 |
| higher-rated costs more         | 7484.000 |   0.866 |                    0.628 |      0.204 |
| no price                        |  540.000 |   0.062 |                    0.594 |      0.230 |

## ② strong vs weak (8333 votes)

| cls                             |    votes |   share |   higher_rated_clear_win |   tie_rate |
|:--------------------------------|---------:|--------:|-------------------------:|-----------:|
| higher-rated costs less or same |  694.000 |   0.083 |                    0.610 |      0.176 |
| higher-rated costs more         | 6855.000 |   0.823 |                    0.628 |      0.206 |
| no price                        |  784.000 |   0.094 |                    0.569 |      0.233 |

## The 15 model pairs with the most votes where the higher-rated model is cheaper

Higher-rated = the side with the higher rating; typical cost = the model's median cost per request in the dev set.

| hi_model                    | lo_model                                |   votes |   higher_rated_per_1k |   lower_rated_per_1k |
|:----------------------------|:----------------------------------------|--------:|----------------------:|---------------------:|
| gemini-2.5-flash            | claude-opus-4-20250514                  |     286 |                  3.48 |                30.45 |
| gemini-2.5-pro              | claude-opus-4-20250514                  |     257 |                 15.6  |                30.45 |
| gemini-2.5-flash            | claude-sonnet-4-20250514                |     248 |                  3.48 |                 5.54 |
| qwen3-235b-a22b-no-thinking | claude-opus-4-20250514                  |     244 |                  2.58 |                30.45 |
| gemini-2.0-flash-001        | claude-3-5-sonnet-20241022              |     237 |                  0.45 |                 4.24 |
| mistral-medium-2505         | claude-opus-4-20250514                  |     236 |                  1.49 |                30.45 |
| mistral-medium-2505         | claude-sonnet-4-20250514                |     229 |                  1.49 |                 5.54 |
| qwen3-235b-a22b-no-thinking | claude-sonnet-4-20250514                |     224 |                  2.58 |                 5.54 |
| gemini-2.5-flash            | command-a-03-2025                       |     214 |                  3.48 |                 7.74 |
| o3-2025-04-16               | claude-opus-4-20250514                  |     212 |                 11.66 |                30.45 |
| deepseek-v3-0324            | claude-opus-4-20250514                  |     208 |                  0.78 |                30.45 |
| gemma-3-27b-it              | claude-opus-4-20250514                  |     205 |                  0.41 |                30.45 |
| deepseek-r1-0528            | gemini-2.5-flash                        |     198 |                  2.45 |                 3.48 |
| qwen3-235b-a22b-no-thinking | claude-3-7-sonnet-20250219-thinking-32k |     198 |                  2.58 |                 3.24 |
| mistral-medium-2505         | claude-3-7-sonnet-20250219-thinking-32k |     197 |                  1.49 |                 3.24 |

## ② Main dollar case: pricier strong model vs cheaper weak model (6855 votes, 82.3% of strong-vs-weak votes)

Typical cost ratio (strong / weak): P10 1.9x, P25 3.6x, P50 6.1x, P75 17.3x, P90 33.0x

Extra spend per 1,000 requests (strong - weak, USD): P10 $1.39, P50 $10.99, P90 $18.92

Higher-rated clear-win rate 0.628, tie rate 0.206

### Strong-tier models involved

| hi_model                     |   votes |   cost_per_1k_usd |
|:-----------------------------|--------:|------------------:|
| gemini-2.5-pro               |    1319 |             15.6  |
| o3-2025-04-16                |    1204 |             11.66 |
| chatgpt-4o-latest-20250326   |    1105 |             12.85 |
| gemini-2.5-flash             |     995 |              3.48 |
| grok-3-preview-02-24         |     803 |             22.16 |
| deepseek-r1-0528             |     517 |              2.45 |
| gemini-2.5-pro-preview-05-06 |     433 |             12.65 |
| gemini-2.5-pro-preview-03-25 |     255 |             12.23 |
| grok-4-0709                  |     224 |             22.01 |

### Weak-tier models involved

| lo_model                                |   votes |   cost_per_1k_usd |
|:----------------------------------------|--------:|------------------:|
| claude-3-5-haiku-20241022               |     921 |              1.31 |
| claude-3-7-sonnet-20250219-thinking-32k |     914 |              3.24 |
| amazon.nova-pro-v1:0                    |     903 |              2.09 |
| llama-4-maverick-17b-128e-instruct      |     764 |              0.67 |
| claude-3-5-sonnet-20241022              |     727 |              4.24 |
| llama-3.3-70b-instruct                  |     725 |              0.49 |
| claude-3-7-sonnet-20250219              |     721 |              3.56 |
| mistral-small-3.1-24b-instruct-2503     |     421 |              0.21 |
| llama-4-scout-17b-16e-instruct          |     356 |              0.44 |
| magistral-medium-2506                   |     299 |              2.32 |
| gpt-4o-mini-2024-07-18                  |     104 |              0.3  |

### Excluded strong-vs-weak votes (strong model not pricier, or no price)

| hi_model                            | lo_model                                | cls                             |   votes |
|:------------------------------------|:----------------------------------------|:--------------------------------|--------:|
| gemini-2.5-flash                    | claude-3-7-sonnet-20250219              | higher-rated costs less or same |     194 |
| gemini-2.5-flash                    | claude-3-5-sonnet-20241022              | higher-rated costs less or same |     157 |
| deepseek-r1-0528                    | claude-3-7-sonnet-20250219-thinking-32k | higher-rated costs less or same |     135 |
| llama-4-maverick-03-26-experimental | claude-3-7-sonnet-20250219-thinking-32k | no price                        |     128 |
| deepseek-r1-0528                    | claude-3-7-sonnet-20250219              | higher-rated costs less or same |     124 |
| llama-4-maverick-03-26-experimental | claude-3-5-haiku-20241022               | no price                        |     118 |
| llama-4-maverick-03-26-experimental | amazon.nova-pro-v1:0                    | no price                        |     106 |
| llama-4-maverick-03-26-experimental | claude-3-5-sonnet-20241022              | no price                        |      92 |
| llama-4-maverick-03-26-experimental | claude-3-7-sonnet-20250219              | no price                        |      90 |
| deepseek-r1-0528                    | claude-3-5-sonnet-20241022              | higher-rated costs less or same |      84 |
| llama-4-maverick-03-26-experimental | llama-4-maverick-17b-128e-instruct      | no price                        |      82 |
| llama-4-maverick-03-26-experimental | llama-3.3-70b-instruct                  | no price                        |      53 |
| llama-4-maverick-03-26-experimental | mistral-small-3.1-24b-instruct-2503     | no price                        |      45 |
| llama-4-maverick-03-26-experimental | llama-4-scout-17b-16e-instruct          | no price                        |      40 |
| llama-4-maverick-03-26-experimental | magistral-medium-2506                   | no price                        |      30 |
