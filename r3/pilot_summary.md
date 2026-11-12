# R3 pilot on 50 prompts: cost and agreement

Model claude-haiku-5-5; price input $0.10 / output $0.50 per million tokens; structured outputs (JSON schema).
Output tokens include thinking tokens (thinking is billed as output). Full-run extrapolation = mean cost per prompt × 9,247; input length is adjusted to the full-set mean length with the chars→tokens linear fit from the 50 prompts.

| Setting | OK | Failed/refused | Mean input tokens | Mean output tokens (P90) | Cost per prompt | All 9,247 (standard) | All (Batch, half price) | Latency P50 / P90 (s) |
|---|---|---|---|---|---|---|---|---|
| A_think_low | 50 | 0 | 1831 | 92 (73) | $0.00023 | $2.12 | $1.06 | 1.1 / 2.4 |
| B_no_think | 50 | 0 | 1830 | 66 (68) | $0.00022 | $1.99 | $1.00 | 1.1 / 1.4 |

Failed or refused calls:

- none

## Agreement between the two settings (50 prompts where both succeeded)

| Field | Exact match | Within 1 |
|---|---|---|
| task_type | 94% |  |
| has_single_answer | 82% | 98% |
| is_open_ended | 86% | 100% |
| is_creative | 84% | 100% |
| has_format_constraint | 92% | 100% |
| difficulty | 78% | 100% |
| expected_length | 78% | 100% |

## First 10 labels from setting A (for manual spot checks)

The prompts are quoted from the dataset (CC-BY-4.0) in their original languages.

- `{"task_type": "chitchat", "has_single_answer": 0, "is_open_ended": 0, "is_creative": 1, "has_format_constraint": 1, "difficulty": 0, "expected_length": 0}`  zgraj ze mną w grę, podam Ci nazwę kraju, a Ty podaj mi nazwę innego kraju na ostanią literę kraju który podam, następnie ja podam nazwę kraju na ostatnią liter
- `{"task_type": "advice", "has_single_answer": 0, "is_open_ended": 3, "is_creative": 2, "has_format_constraint": 1, "difficulty": 2, "expected_length": 1}`  Give me a truly unique web app idea for an MVP I can build with lovable in 6 weeks, that would get traction and paid users fast because there is a real need and
- `{"task_type": "factual_qa", "has_single_answer": 0, "is_open_ended": 2, "is_creative": 0, "has_format_constraint": 0, "difficulty": 1, "expected_length": 1}`  какие есть площадки аналоги https://kemono.su https://f95zone.to
- `{"task_type": "advice", "has_single_answer": 0, "is_open_ended": 2, "is_creative": 2, "has_format_constraint": 1, "difficulty": 1, "expected_length": 2}`  I would like to paint the outside of a Victorian terraced house including the bricks and the exterior windows.what light colours would you suggest for exterior 
- `{"task_type": "writing", "has_single_answer": 0, "is_open_ended": 1, "is_creative": 3, "has_format_constraint": 0, "difficulty": 0, "expected_length": 1}`  nu metal song about a maniac who feeds on other people's smells
- `{"task_type": "chitchat", "has_single_answer": 0, "is_open_ended": 2, "is_creative": 0, "has_format_constraint": 0, "difficulty": 0, "expected_length": 1}`  Varför ser alla PNS cyklister bra ut men har så mjuka ben?
- `{"task_type": "writing", "has_single_answer": 0, "is_open_ended": 2, "is_creative": 1, "has_format_constraint": 3, "difficulty": 1, "expected_length": 3}`  ## (该括号填写该单词) 英音:(该括号填写英音);美音:(该括号填写美音) ### 词根词缀 #### 词源 #### 词根词缀解析 #### 派生词 ### 语境:XXX(概括语境) #### (n./adj./v./...)(概括含义)词性及含义 - **英中例句:** - - **衍生短语:** - - **
- `{"task_type": "advice", "has_single_answer": 0, "is_open_ended": 3, "is_creative": 0, "has_format_constraint": 0, "difficulty": 2, "expected_length": 3}`  写好了目前两个比较耗时的工作流程，考虑进行优化，需要的话做一些基础建设，如图数据库等。 先深度思考给我提一些下一步进行的建议。 然后从软件工程的角度思考，目前还缺的哪一些文档，我应该继续撰写哪几类文档？ 下面‘---’包围的是是已经写好的两个 as-is 文档： --- ## 真人认证审核工作流 1. **流程名称**
- `{"task_type": "advice", "has_single_answer": 0, "is_open_ended": 1, "is_creative": 0, "has_format_constraint": 0, "difficulty": 0, "expected_length": 1}`  jak rozweselić kota
- `{"task_type": "factual_qa", "has_single_answer": 2, "is_open_ended": 0, "is_creative": 0, "has_format_constraint": 0, "difficulty": 1, "expected_length": 0}`  which company is largest mortgage servicer in the US