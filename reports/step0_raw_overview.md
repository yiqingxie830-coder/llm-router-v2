# 140k 原始数据概况

- 行数: 135634 (id 唯一: 135634); 会话数: 115372
- evaluation_order=1: 108315 行
- 模型数: 53
- 时间: 2025-04-17 ~ 2025-07-24

## winner 分布 (全部 / order=1)

| winner   |   count |   share |
|:---------|--------:|--------:|
| model_b  |   49785 |   0.367 |
| model_a  |   48563 |   0.358 |
| tie      |   21532 |   0.159 |
| both_bad |   15754 |   0.116 |

| winner   |   count |   share |
|:---------|--------:|--------:|
| model_b  |   39968 |   0.369 |
| model_a  |   39105 |   0.361 |
| tie      |   16589 |   0.153 |
| both_bad |   12653 |   0.117 |

## 语言 (自带 language 列) 前 12

| language   |   count |   share |
|:-----------|--------:|--------:|
| en         |   71175 |   0.525 |
| pl         |   13813 |   0.102 |
| und        |   11835 |   0.087 |
| ru         |    9263 |   0.068 |
| zh         |    6501 |   0.048 |
| de         |    4311 |   0.032 |
| ko         |    2486 |   0.018 |
| ja         |    2444 |   0.018 |
| fr         |    2128 |   0.016 |
| pt         |    1684 |   0.012 |
| es         |    1672 |   0.012 |
| fa         |    1599 |   0.012 |

共 126 种语言标记

## 第一条用户消息字符长度

|       |   chars |
|:------|--------:|
| count |  135634 |
| mean  |     361 |
| std   |    1863 |
| min   |       1 |
| 5%    |      13 |
| 10%   |      20 |
| 25%   |      38 |
| 50%   |      77 |
| 75%   |     186 |
| 90%   |     514 |
| 95%   |    1106 |
| 99%   |    5540 |
| max   |  119164 |

< 16 字符: 8923 行

## 每个模型出场次数

|                                              |   count |
|:---------------------------------------------|--------:|
| claude-opus-4-20250514                       |   10092 |
| gemini-2.5-flash                             |    9668 |
| gemini-2.5-pro                               |    9219 |
| mistral-medium-2505                          |    9135 |
| qwen3-235b-a22b-no-thinking                  |    9076 |
| o3-2025-04-16                                |    8529 |
| claude-sonnet-4-20250514                     |    8295 |
| chatgpt-4o-latest-20250326                   |    7650 |
| gemma-3-27b-it                               |    7326 |
| claude-3-7-sonnet-20250219-thinking-32k      |    7149 |
| claude-3-7-sonnet-20250219                   |    6852 |
| command-a-03-2025                            |    6838 |
| claude-3-5-sonnet-20241022                   |    6817 |
| o3-mini                                      |    6624 |
| deepseek-r1-0528                             |    6554 |
| gpt-4.1-2025-04-14                           |    6540 |
| o4-mini-2025-04-16                           |    6477 |
| claude-3-5-haiku-20241022                    |    6466 |
| amazon.nova-pro-v1:0                         |    6383 |
| claude-opus-4-20250514-thinking-16k          |    6157 |
| gpt-4.1-mini-2025-04-14                      |    6121 |
| deepseek-v3-0324                             |    6117 |
| gemini-2.0-flash-001                         |    6074 |
| grok-3-preview-02-24                         |    6040 |
| grok-3-mini-beta                             |    5991 |
| llama-4-maverick-03-26-experimental          |    5806 |
| claude-sonnet-4-20250514-thinking-32k        |    5613 |
| qwen3-30b-a3b                                |    5498 |
| minimax-m1                                   |    5469 |
| llama-4-maverick-17b-128e-instruct           |    5332 |
| gemini-2.5-flash-lite-preview-06-17-thinking |    5232 |
| qwen3-235b-a22b                              |    5115 |
| gemini-2.5-flash-preview-04-17               |    5050 |
| llama-3.3-70b-instruct                       |    4905 |
| qwen-max-2025-01-25                          |    4379 |
| qwq-32b                                      |    4259 |
| mistral-small-3.1-24b-instruct-2503          |    3271 |
| amazon-nova-experimental-chat-05-14          |    3267 |
| gemini-2.5-pro-preview-05-06                 |    3255 |
| llama-4-scout-17b-16e-instruct               |    2805 |
| grok-3-mini-high                             |    2801 |
| kimi-k2-0711-preview                         |    2743 |
| gemma-3n-e4b-it                              |    2611 |
| magistral-medium-2506                        |    2606 |
| mistral-small-2506                           |    2373 |
| hunyuan-turbos-20250416                      |    1585 |
| grok-4-0709                                  |    1561 |
| gemini-2.5-pro-preview-03-25                 |    1389 |
| qwen3-235b-a22b-instruct-2507                |     622 |
| gpt-4o-mini-2024-07-18                       |     590 |
| gpt-4o-2024-11-20                            |     573 |
| gemini-2.0-flash-thinking-exp-01-21          |     366 |
| qwen3-coder-480b-a35b-instruct               |       2 |