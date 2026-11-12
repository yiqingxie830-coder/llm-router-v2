# Step 1 cleaning results

Raw 135634 rows = order=1 108315 rows + follow-ups 27319 rows (follow-ups stored separately).

| main data order=1: rule                   |   rows left |   removed |
|:------------------------------------------|------------:|----------:|
| start                                     |      108315 |         0 |
| R1 first message < 16 chars               |      101286 |      7029 |
| R2 an answer is empty                     |      101209 |        77 |
| R3 answer cut by provider content filter  |      101076 |       133 |
| R4 exact duplicate prompt+model_a+model_b |      101058 |        18 |

| follow-ups order>1: rule                  |   rows left |   removed |
|:------------------------------------------|------------:|----------:|
| start                                     |       27319 |         0 |
| R1 first message < 16 chars               |       25303 |      2016 |
| R2 an answer is empty                     |       25251 |        52 |
| R3 answer cut by provider content filter  |       25226 |        25 |
| R4 exact duplicate prompt+model_a+model_b |       25222 |         4 |

## Main data: winner distribution

| winner   |   rows |   share |
|:---------|-------:|--------:|
| model_b  |  37294 |   0.369 |
| model_a  |  36303 |   0.359 |
| tie      |  15465 |   0.153 |
| both_bad |  11996 |   0.119 |

## Main data: top 10 languages (dataset's own tag)

| language   |   rows |   share |
|:-----------|-------:|--------:|
| en         |  56703 |   0.561 |
| pl         |  10405 |   0.103 |
| ru         |   7261 |   0.072 |
| und        |   5932 |   0.059 |
| zh         |   3891 |   0.039 |
| de         |   3414 |   0.034 |
| fr         |   1703 |   0.017 |
| ja         |   1650 |   0.016 |
| ko         |   1588 |   0.016 |
| es         |   1350 |   0.013 |

English is_en: 56703 rows (56.1%)
Agreement of dataset language tag with langdetect: all rows 85.7%; excluding rows tagged und 90.9%
Agreement on English rows: 95.3%

## Share of Polish by week

| week                  |   pl share |
|:----------------------|-----------:|
| 2025-04-14/2025-04-20 |      0.005 |
| 2025-04-21/2025-04-27 |      0.004 |
| 2025-04-28/2025-05-04 |      0.005 |
| 2025-05-05/2025-05-11 |      0.007 |
| 2025-05-12/2025-05-18 |      0.005 |
| 2025-05-19/2025-05-25 |      0.022 |
| 2025-05-26/2025-06-01 |      0.011 |
| 2025-06-02/2025-06-08 |      0.014 |
| 2025-06-09/2025-06-15 |      0.063 |
| 2025-06-16/2025-06-22 |      0.244 |
| 2025-06-23/2025-06-29 |      0.201 |
| 2025-06-30/2025-07-06 |      0.185 |
| 2025-07-07/2025-07-13 |      0.155 |
| 2025-07-14/2025-07-20 |      0.125 |
| 2025-07-21/2025-07-27 |      0.122 |

## Main data: first message length (characters)

|       |   chars |
|:------|--------:|
| count |  101058 |
| mean  |     421 |
| std   |    2073 |
| min   |      16 |
| 10%   |      28 |
| 25%   |      45 |
| 50%   |      89 |
| 75%   |     216 |
| 90%   |     606 |
| 99%   |    6747 |
| max   |  119164 |