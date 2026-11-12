# Candidate dataset comparison (2026-11-07)

Both were downloaded and passed the official sha256 checks.

| | 100k (lmarena-ai/arena-human-preference-100k) | 140k (lmarena-ai/arena-human-preference-140k) |
|---|---|---|
| Dates | 2024-06-20 ~ 2024-08-19 | 2025-04-17 ~ 2025-07-24 |
| Rows | 106,134 | 135,634 |
| Models | 55 (gpt-4o, claude-3.5-sonnet, llama-3, gemma-2…) | 53 (claude-opus-4, gemini-2.5, o3, gpt-4.1, deepseek-r1…) |
| winner values | model_a / model_b / tie / tie (bothbad) | model_a / model_b / tie / both_bad |
| Tie share | 38.4% | 27.5% (27.0% at order=1) |
| Languages | 90, English 54.3%, Russian 10.9%, Chinese 9.5% | 126, English 52.5%, Polish 10.2%, und 8.7%, Russian 6.8% |
| First message length (chars) | median 109, P90 1,162 | median 77, P90 514 |
| First message < 16 chars | 7,664 | 8,923 |
| Multi-turn | 16.5% | 14.2% |
| Fields unique to the set | is_refusal, dedup_tag | evaluation_session_id, evaluation_order |
| Repeat votes in a session | no | yes: 115,372 sessions; 108,315 rows with evaluation_order=1, the other 27,319 are votes after a follow-up with a new model pair |

## The evaluation_order issue
In the 140k set, after a vote the session continues with a new model pair. For votes with order>1 the user's message is preceded by context from other models, so it is not the "router sees a new prompt" scenario. Recommendation: use order=1 only.
