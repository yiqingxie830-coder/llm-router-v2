# Step 2/3 labels and grouping

Tiers (provisional option A): strong = tier_k5 tiers [1, 2], weak = tier [5]; models with < 300 games get no tier.

|   tier |   n_models |   min_rating |   max_rating |
|-------:|-----------:|-------------:|-------------:|
|      1 |          1 |       1136.8 |       1136.8 |
|      2 |          9 |       1064.8 |       1103.8 |
|      3 |         17 |       1002.1 |       1042.6 |
|      4 |         13 |        960.3 |        997.8 |
|      5 |         11 |        882.1 |        930.4 |

## Label shares

|                  |   full table (y_hi: higher-rated side) |   evaluation view (y: strong vs weak) |
|:-----------------|---------------------------------------:|--------------------------------------:|
| votes            |                             101058     |                              9332     |
| high/strong wins |                                  0.467 |                                 0.616 |
| low/weak wins    |                                  0.261 |                                 0.174 |
| tie              |                                  0.272 |                                 0.211 |

## Routing target y_route (ties go to the weak tier, cost first)

|    |   full table y_hi_route |   evaluation view y_route |
|---:|------------------------:|--------------------------:|
|  0 |                   0.533 |                     0.384 |
|  1 |                   0.467 |                     0.616 |

## Repeated prompts (first evidence of label noise)

### Full table (by higher-rated side)

- Distinct prompts: 95373; seen ≥2 times: 3289, covering 8974 votes (8.9%)
- Repeat counts: {2: 2137, 3: 596, 4: 263, 5: 131, 6: 77, 7: 46} (max 19)
- Share of repeated prompts with ≥2 non-tie votes that disagree: 37.0%
- Noise ceiling: on non-tie votes of repeated prompts, predicting each prompt's majority gives at most 79.5% accuracy

Distribution of win rate (soft label) on repeated prompts:

| mean                   |   prompts |
|:-----------------------|----------:|
| 0 (weak always wins)   |       181 |
| (0,0.25]               |       348 |
| (0.25,0.5)             |       182 |
| 0.5                    |      1116 |
| (0.5,0.75]             |       835 |
| (0.75,1)               |       140 |
| 1 (strong always wins) |       487 |

### Evaluation view (strong vs weak)

- Distinct prompts: 9247; seen ≥2 times: 82, covering 167 votes (1.8%)
- Repeat counts: {2: 79, 3: 3} (max 3)
- Share of repeated prompts with ≥2 non-tie votes that disagree: 14.7%
- Noise ceiling: on non-tie votes of repeated prompts, predicting each prompt's majority gives at most 91.5% accuracy

Distribution of win rate (soft label) on repeated prompts:

| mean                   |   prompts |
|:-----------------------|----------:|
| 0 (weak always wins)   |         5 |
| (0,0.25]               |         5 |
| (0.25,0.5)             |         0 |
| 0.5                    |        24 |
| (0.5,0.75]             |        17 |
| (0.75,1)               |         0 |
| 1 (strong always wins) |        31 |

## First turn vs follow-ups: does the weak tier win more on follow-ups?

|                  |   first turn order=1 |   follow-up order>1 |
|:-----------------|---------------------:|--------------------:|
| votes            |             9332     |            2269     |
| high/strong wins |                0.616 |               0.533 |
| low/weak wins    |                0.174 |               0.221 |
| tie              |                0.211 |               0.245 |

Full-table view (win rate of the lower-rated side; comparable only when the mean rating gaps are close):

|                  |   first turn |   follow-up |
|:-----------------|-------------:|------------:|
| votes            |   101058     |   25222     |
| lower-rated wins |        0.261 |       0.285 |
| tie              |        0.272 |       0.296 |
| mean rating gap  |       71.927 |      71.642 |