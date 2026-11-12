# Step 2 tier candidates (k=5 one-dimensional k-means, ratings from Bradley-Terry on this data)

Rating scale: mean 1000; a 100-point gap ≈ 64% win rate for the higher-rated model, 175 points ≈ 73%. Intervals = 95% from 200 bootstrap resamples.

Models with < 300 appearances are left out of tiering (rating interval too wide): qwen3-coder-480b (1 appearance), gemini-2.0-flash-thinking-exp (286).

## Layer 1 (ratings 1137–1137, 1 model)

| model          |   rating |   ci_low |   ci_high |   n_games |
|:---------------|---------:|---------:|----------:|----------:|
| gemini-2.5-pro |   1136.8 |   1127.8 |      1144 |      6862 |

## Layer 2 (ratings 1065–1104, 9 models)

| model                               |   rating |   ci_low |   ci_high |   n_games |
|:------------------------------------|---------:|---------:|----------:|----------:|
| gemini-2.5-pro-preview-03-25        |   1103.8 |   1085.7 |    1118.9 |      1067 |
| grok-4-0709                         |   1094.7 |   1081.2 |    1111.2 |      1196 |
| o3-2025-04-16                       |   1087.3 |   1080   |    1095   |      6316 |
| gemini-2.5-pro-preview-05-06        |   1085.8 |   1073.6 |    1098   |      2452 |
| deepseek-r1-0528                    |   1081.9 |   1073.1 |    1090.1 |      4837 |
| chatgpt-4o-latest-20250326          |   1075.1 |   1066.6 |    1082.2 |      5757 |
| grok-3-preview-02-24                |   1074   |   1063.9 |    1082.7 |      4437 |
| gemini-2.5-flash                    |   1069.8 |   1062.2 |    1078.2 |      7199 |
| llama-4-maverick-03-26-experimental |   1064.8 |   1056.5 |    1073.7 |      4364 |

## Layer 3 (ratings 1002–1043, 19 models)

| model                                        |   rating |   ci_low |   ci_high |   n_games |
|:---------------------------------------------|---------:|---------:|----------:|----------:|
| gemini-2.5-flash-preview-04-17               |   1042.6 |   1031.7 |    1051.2 |      3760 |
| qwen3-235b-a22b-no-thinking                  |   1041.1 |   1032.8 |    1047.2 |      6805 |
| hunyuan-turbos-20250416                      |   1037.1 |   1017.5 |    1054.3 |      1163 |
| gpt-4.1-2025-04-14                           |   1033   |   1024.8 |    1041.2 |      4844 |
| kimi-k2-0711-preview                         |   1032.7 |   1022.1 |    1047.5 |      2013 |
| o4-mini-2025-04-16                           |   1031.3 |   1024.1 |    1039.2 |      4871 |
| gemini-2.5-flash-lite-preview-06-17-thinking |   1028.7 |   1018.5 |    1037.7 |      3916 |
| gemini-2.0-flash-thinking-exp-01-21          |   1026.9 |    997.7 |    1061.9 |       286 |
| qwen3-235b-a22b                              |   1026.3 |   1018   |    1033.5 |      3756 |
| deepseek-v3-0324                             |   1026.3 |   1016.3 |    1034.4 |      4576 |
| qwen-max-2025-01-25                          |   1023.6 |   1012.5 |    1033   |      3264 |
| mistral-medium-2505                          |   1020.4 |   1012.4 |    1027.6 |      6824 |
| gemma-3-27b-it                               |   1016.3 |   1009.3 |    1025.2 |      5464 |
| qwen3-coder-480b-a35b-instruct               |   1013.4 |    928.2 |    1201.4 |         1 |
| qwen3-235b-a22b-instruct-2507                |   1013.3 |    985.5 |    1041.4 |       457 |
| minimax-m1                                   |   1011.8 |   1000.7 |    1019.9 |      3894 |
| grok-3-mini-beta                             |   1010.6 |   1000.4 |    1019.6 |      4453 |
| claude-opus-4-20250514-thinking-16k          |   1002.4 |    994.7 |    1011   |      4593 |
| grok-3-mini-high                             |   1002.1 |    990.6 |    1015.1 |      2086 |

## Layer 4 (ratings 960–998, 13 models)

| model                                 |   rating |   ci_low |   ci_high |   n_games |
|:--------------------------------------|---------:|---------:|----------:|----------:|
| claude-opus-4-20250514                |    997.8 |    990.4 |    1004.2 |      7604 |
| gemini-2.0-flash-001                  |    995.3 |    986.5 |    1003.2 |      4574 |
| qwq-32b                               |    987.2 |    976.1 |     995.7 |      3107 |
| gpt-4.1-mini-2025-04-14               |    981.1 |    971.6 |     989.8 |      4604 |
| qwen3-30b-a3b                         |    978.9 |    969.8 |     988.6 |      4098 |
| claude-sonnet-4-20250514-thinking-32k |    977.6 |    969   |     986.5 |      4197 |
| gpt-4o-2024-11-20                     |    977.2 |    945.6 |    1003.9 |       430 |
| mistral-small-2506                    |    976.8 |    961.8 |     991.1 |      1741 |
| command-a-03-2025                     |    970.4 |    962.2 |     977.6 |      5184 |
| claude-sonnet-4-20250514              |    965.8 |    956.7 |     972.4 |      6198 |
| o3-mini                               |    963.8 |    955   |     971.7 |      4913 |
| gemma-3n-e4b-it                       |    961.2 |    948.3 |     975.7 |      1947 |
| amazon-nova-experimental-chat-05-14   |    960.3 |    949.2 |     971   |      2347 |

## Layer 5 (ratings 882–930, 11 models)

| model                                   |   rating |   ci_low |   ci_high |   n_games |
|:----------------------------------------|---------:|---------:|----------:|----------:|
| llama-4-maverick-17b-128e-instruct      |    930.4 |    921.5 |     938.4 |      3993 |
| llama-4-scout-17b-16e-instruct          |    929.8 |    915.8 |     941.9 |      2074 |
| claude-3-7-sonnet-20250219-thinking-32k |    922.3 |    913.3 |     930.7 |      5378 |
| claude-3-7-sonnet-20250219              |    920.6 |    911.4 |     929.3 |      5174 |
| claude-3-5-sonnet-20241022              |    920.2 |    910.1 |     927.8 |      5163 |
| amazon.nova-pro-v1:0                    |    908.3 |    898.6 |     917.2 |      4728 |
| llama-3.3-70b-instruct                  |    904.1 |    893.7 |     913.3 |      3682 |
| mistral-small-3.1-24b-instruct-2503     |    901.4 |    888.8 |     912.2 |      2416 |
| claude-3-5-haiku-20241022               |    887.7 |    878.1 |     895.7 |      4828 |
| magistral-medium-2506                   |    885.7 |    872.5 |     896.2 |      1779 |
| gpt-4o-mini-2024-07-18                  |    882.1 |    851.3 |     912.4 |       444 |

## Candidate schemes (only models with ≥ 300 appearances)

| Scheme | Strong tier | Weak tier | Battles | Strong wins | Weak wins | Ties | Strong wins among non-ties | Mean gap |
|---|---|---|---|---|---|---|---|---|
| A | layers 1-2 (10) | layer 5 (11) | 9,332 | 61.6% | 17.4% | 21.1% | 78.0% | 175 |
| B | layers 1-2 (10) | layers 4-5 (24) | 21,081 | 56.7% | 20.1% | 23.2% | 73.9% | 138 |
| C | layers 1-2 (10) | layer 3 (17) | 14,857 | 45.4% | 26.8% | 27.8% | 62.9% | 63 |
| D | layers 1-3 (27) | layer 5 (11) | 22,881 | 56.3% | 20.1% | 23.7% | 73.7% | 138 |

With k=6, layers 1-2 and layer 6 split exactly as in scheme A, so A's boundaries are not sensitive to the choice of k.