# 候选数据集对比 (2026-11-07)

两份都已下载并通过官方 sha256 校验。

| | 100k (lmarena-ai/arena-human-preference-100k) | 140k (lmarena-ai/arena-human-preference-140k) |
|---|---|---|
| 时间 | 2024-06-20 ~ 2024-08-19 | 2025-04-17 ~ 2025-07-24 |
| 行数 | 106,134 | 135,634 |
| 模型数 | 55 (gpt-4o, claude-3.5-sonnet, llama-3, gemma-2…) | 53 (claude-opus-4, gemini-2.5, o3, gpt-4.1, deepseek-r1…) |
| winner 取值 | model_a / model_b / tie / tie (bothbad) | model_a / model_b / tie / both_bad |
| 平局比例 | 38.4% | 27.5% (order=1 时 27.0%) |
| 语言 | 90 种, 英文 54.3%, 俄 10.9%, 中 9.5% | 126 种, 英文 52.5%, 波兰 10.2%, und 8.7%, 俄 6.8% |
| 首条消息长度 (字符) | 中位 109, P90 1,162 | 中位 77, P90 514 |
| 首条 < 16 字符 | 7,664 | 8,923 |
| 多轮 | 16.5% | 14.2% |
| 特有字段 | is_refusal, dedup_tag | evaluation_session_id, evaluation_order |
| 同会话连投 | 无 | 有: 115,372 个会话; evaluation_order=1 的 108,315 行, 其余 27,319 行是追问后换新模型再投 |

## evaluation_order 的问题
140k 里同一会话投完一票会换一对新模型继续聊。order>1 的票, 用户第一条消息前面还有别的模型的上下文, 不是"路由器看到一条新 prompt"的场景。建议只用 order=1。
