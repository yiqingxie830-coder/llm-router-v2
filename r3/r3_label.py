"""r3_label.py — R3 语义标签: 用 Claude Haiku 5.5 给口径 ② 的 prompt 打 7 个字段 (打标 prompt 由 Yiqing 2026-11-10 定)

在你自己的电脑上跑 (需要 API key)。只依赖 anthropic SDK:
    pip install -U anthropic
    python r3_label.py pilot --env "<module 4 目录>/.env"      # 试打 50 条, 两种设置各 50 次, 共 100 次调用, 预计 < $0.05
    python r3_label.py count --env ...                          # (可选) 用 count_tokens 数全部 9,247 条的准确输入 token, 不收费
    python r3_label.py submit --env ... [--arm B_no_think]      # 全量 9,247 条提交到 Batch API (五折), 记下 batch id
    python r3_label.py collect --env ...                        # 等 batch 跑完 (通常 < 1 小时), 取回结果写 r3_labels.jsonl

key 的读取顺序: 已设置的环境变量 ANTHROPIC_API_KEY > --env 指定的 .env 文件。脚本不会打印或保存 key。

试打的两种设置 (都用结构化输出保证 JSON 合法):
  A_think_low : 自适应思考 (Haiku 5.5 默认开启), effort = low
  B_no_think  : 关闭思考 (thinking disabled, effort = low)
产出: pilot_results.jsonl (每次调用的标签、token、耗时), pilot_summary.md (费用外推、两种设置的一致率、延迟)
全量: submit 提交 Batch (五折, 默认用关思考 B 设置), collect 取回 r3_labels.jsonl + r3_labels_summary.txt, 两个文件传回 v2/r3/。
collect 可以中断后重跑: batch id 存在 batch_id.json 里, 结果保留 29 天。
"""
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6       # 美元 / token, Haiku 5.5, prompt ≤ 100K token
N_ALL = 9247

SYSTEM = """You label a single user message sent to a chatbot. Judge the task itself from the message alone.
Return ONLY a JSON object with these 7 fields.

task_type — one of:
  "code" (write / fix / explain code), "math" (calculation, proof, logic puzzle),
  "factual_qa" (a fact with a checkable answer), "writing" (essay, story, poem, copy, email),
  "advice" (recommendation, how-to, opinion, planning), "translation" (translate or rewrite in another language),
  "chitchat" (greeting, small talk, roleplay), "other"

has_single_answer — 0 to 3: is there one verifiably correct answer?
  0 = no correct answer exists (write a poem, what do you think of X)
  1 = loosely right/wrong, many acceptable answers (explain recursion, summarize this)
  2 = mostly determinate, small room for variation (fix this bug, convert units with explanation)
  3 = exactly one correct answer, checkable (what is 17×23, capital of Peru, output must be "yes" or "no")

is_open_ended — 0 to 3: how much does answer quality depend on depth and organization rather than correctness?
  0 = not at all, a one-line reply fully answers it
  1 = a short, standard reply suffices
  2 = a good answer needs some structure or several considerations
  3 = answer quality is mainly about depth, breadth, judgment (recommend books for X, how should I plan my career, compare approaches)

is_creative — 0 to 3: does it ask for original creative content?
  0 = none
  1 = minor stylistic choice (rephrase politely)
  2 = some creativity within constraints (write a product description)
  3 = fully creative (story, poem, song, roleplay scene)

has_format_constraint — 0 to 3: how strict are the explicit output-format requirements?
  0 = none
  1 = mild (keep it short, use bullet points)
  2 = specific (answer in JSON, exactly 3 items, in Spanish)
  3 = very strict and easy to violate (one word only, no punctuation, exact template)

difficulty — 0 to 3: how capable a model is needed?
  0 = trivial, any small model (greetings, simple fact, simple rewrite)
  1 = routine, a 7B–13B model handles it well
  2 = needs a solid mid-size model (multi-step reasoning, non-trivial code)
  3 = frontier-level (research-level math, long complex code, subtle expert domain)

expected_length — 0 to 3: how long should a good answer be?
  0 = one word to one sentence
  1 = a short paragraph
  2 = several paragraphs or a code block
  3 = long-form (essay, full program, detailed plan)

Rules: judge the task, not whether it is safe or appropriate. Do not guess the language; ignore it.
If the message is empty or meaningless, set task_type "other" and all scores 0.
Output JSON only, no explanation."""

SCORES = ["has_single_answer", "is_open_ended", "is_creative", "has_format_constraint", "difficulty", "expected_length"]
TASKS = ["code", "math", "factual_qa", "writing", "advice", "translation", "chitchat", "other"]
SCHEMA = {
    "type": "object",
    "properties": {"task_type": {"type": "string", "enum": TASKS},
                   **{k: {"type": "integer", "enum": [0, 1, 2, 3]} for k in SCORES}},
    "required": ["task_type"] + SCORES,
    "additionalProperties": False,
}
ARMS = {
    "A_think_low": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
    "B_no_think": {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}},
}


def user_content(prompt):
    # 消息原文整段放进标签里, 不截断
    return f"<message>\n{prompt}\n</message>"


def load_key(env_path):
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if env_path and Path(env_path).exists():
        for line in Path(env_path).read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                return
    sys.exit("没找到 ANTHROPIC_API_KEY: 请设环境变量, 或用 --env 指向含这一行的 .env 文件")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def label_one(client, row, arm):
    cfg = ARMS[arm]
    t = time.time()
    try:
        r = client.messages.create(
            model=MODEL, max_tokens=4000, system=SYSTEM,
            messages=[{"role": "user", "content": user_content(row["prompt"])}],
            thinking=cfg["thinking"],
            output_config={**cfg["output_config"], "format": {"type": "json_schema", "schema": SCHEMA}},
        )
    except anthropic.APIStatusError as e:
        return {"group_id": row["group_id"], "arm": arm, "error": f"{e.status_code} {e.message}"}
    except anthropic.APIConnectionError as e:
        return {"group_id": row["group_id"], "arm": arm, "error": f"connection {e}"}
    sec = time.time() - t
    out = {"group_id": row["group_id"], "arm": arm, "stop_reason": r.stop_reason, "seconds": round(sec, 2),
           "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens,
           "prompt_chars": len(row["prompt"]), "request_id": r._request_id}
    if r.stop_reason == "refusal":
        out["error"] = "refusal"
        return out
    text = next((b.text for b in r.content if b.type == "text"), "")
    try:
        out["labels"] = json.loads(text)
    except json.JSONDecodeError:
        out["error"] = "bad_json"
        out["raw"] = text[:500]
    return out


def pilot(args):
    client = anthropic.Anthropic(max_retries=5)
    rows = read_jsonl(HERE / "pilot_50.jsonl")
    jobs = [(r, a) for a in ARMS for r in rows]
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda j: label_one(client, *j), jobs))
    with open(HERE / "pilot_results.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summarize(res, rows)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summarize(res, rows):
    L = ["# R3 试打 50 条: 费用与一致性", "",
         f"模型 {MODEL}; 单价 输入 ${PRICE_IN * 1e6:.2f} / 输出 ${PRICE_OUT * 1e6:.2f} 每百万 token; 结构化输出 (JSON schema)。",
         "输出 token 含思考 token (思考按输出计费)。全量外推 = 每条平均费用 × 9,247; 输入长度按 50 条的 字符→token 线性关系校正到全量平均长度。", "",
         "| 设置 | 成功 | 失败/拒答 | 平均输入 token | 平均输出 token (P90) | 每条费用 | 全量 9,247 条 (标准) | 全量 (Batch 五折) | 延迟 P50 / P90 (秒) |",
         "|---|---|---|---|---|---|---|---|---|"]
    all_chars = None
    p_all = HERE / "r3_prompts.jsonl"
    if p_all.exists():
        all_chars = statistics.mean(len(r["prompt"]) for r in read_jsonl(p_all))
    by_arm = {}
    for arm in ARMS:
        ok = [r for r in res if r["arm"] == arm and "labels" in r]
        bad = [r for r in res if r["arm"] == arm and "labels" not in r]
        by_arm[arm] = {r["group_id"]: r["labels"] for r in ok}
        if not ok:
            L.append(f"| {arm} | 0 | {len(bad)} | | | | | | |")
            continue
        ins = [r["input_tokens"] for r in ok]
        outs = [r["output_tokens"] for r in ok]
        mean_in = statistics.mean(ins)
        if all_chars is not None and len(ok) > 2:   # 输入 token ≈ a + b × 字符数, 外推到全量平均字符数
            xs = [r["prompt_chars"] for r in ok]
            mx, my = statistics.mean(xs), mean_in
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ins)) / max(sum((x - mx) ** 2 for x in xs), 1)
            mean_in = my + b * (all_chars - mx)
        cost = mean_in * PRICE_IN + statistics.mean(outs) * PRICE_OUT
        secs = [r["seconds"] for r in ok]
        L.append(f"| {arm} | {len(ok)} | {len(bad)} | {mean_in:.0f} | {statistics.mean(outs):.0f} ({pct(outs, .9)}) | "
                 f"${cost:.5f} | ${cost * N_ALL:.2f} | ${cost * N_ALL / 2:.2f} | {pct(secs, .5):.1f} / {pct(secs, .9):.1f} |")
    fails = [f"- {r['arm']} {r['group_id']}: {r.get('error')}" for r in res if "labels" not in r]
    L += ["", "失败或拒答的调用:", ""] + (fails or ["- 无"])
    a, b = by_arm.get("A_think_low", {}), by_arm.get("B_no_think", {})
    common = sorted(set(a) & set(b))
    if common:
        L += ["", f"## 两种设置的一致率 ({len(common)} 条都成功)", "", "| 字段 | 完全一致 | 相差 ≤ 1 |", "|---|---|---|"]
        for k in ["task_type"] + SCORES:
            same = sum(a[g][k] == b[g][k] for g in common) / len(common)
            near = "" if k == "task_type" else f"{sum(abs(a[g][k] - b[g][k]) <= 1 for g in common) / len(common):.0%}"
            L.append(f"| {k} | {same:.0%} | {near} |")
    text = {r["group_id"]: r for r in rows}
    if a:
        L += ["", "## 设置 A 的前 10 条标签 (人工抽查用)", ""]
        for g in list(a)[:10]:
            p = " ".join(text[g]["prompt"].split())[:160]
            L.append(f"- `{json.dumps(a[g], ensure_ascii=False)}`  {p}")
    (HERE / "pilot_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def count(args):
    client = anthropic.Anthropic(max_retries=8)
    rows = read_jsonl(HERE / "r3_prompts.jsonl")

    def one(r):
        c = client.messages.count_tokens(
            model=MODEL, system=SYSTEM, messages=[{"role": "user", "content": user_content(r["prompt"])}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
        return c.input_tokens
    with ThreadPoolExecutor(args.workers) as ex:
        toks = list(ex.map(one, rows))
    tot = sum(toks)
    msg = (f"{len(rows)} 条, 输入 token 合计 {tot:,} (平均 {tot / len(rows):.0f}, 最大 {max(toks):,}); "
           f"输入费用 标准 ${tot * PRICE_IN:.2f} / Batch ${tot * PRICE_IN / 2:.2f}")
    (HERE / "count_tokens_all.txt").write_text(msg, encoding="utf-8")
    print(msg)


def _req(row, i, arm):
    cfg = ARMS[arm]
    return {"custom_id": f"p{i:05d}",   # custom_id 只能用字母数字, 序号对应 r3_prompts.jsonl 的行号
            "params": {"model": MODEL, "max_tokens": 4000, "system": SYSTEM,
                       "messages": [{"role": "user", "content": user_content(row["prompt"])}],
                       "thinking": cfg["thinking"],
                       "output_config": {**cfg["output_config"], "format": {"type": "json_schema", "schema": SCHEMA}}}}


def submit(args):
    client = anthropic.Anthropic(max_retries=5)
    rows = read_jsonl(HERE / "r3_prompts.jsonl")
    assert len(rows) == N_ALL, len(rows)
    b = client.messages.batches.create(requests=[_req(r, i, args.arm) for i, r in enumerate(rows)])
    (HERE / "batch_id.json").write_text(json.dumps({"batch_id": b.id, "arm": args.arm, "n": len(rows),
                                                    "created": time.strftime("%Y-%m-%d %H:%M:%S")}))
    print("已提交", b.id, b.processing_status, "设置", args.arm)


def collect(args):
    client = anthropic.Anthropic(max_retries=5)
    meta = json.loads((HERE / "batch_id.json").read_text())
    rows = read_jsonl(HERE / "r3_prompts.jsonl")
    while True:
        b = client.messages.batches.retrieve(meta["batch_id"])
        if b.processing_status == "ended":
            break
        c = b.request_counts
        print(time.strftime("%H:%M:%S"), b.processing_status, f"处理中 {c.processing} 成功 {c.succeeded} 失败 {c.errored}", flush=True)
        time.sleep(60)
    out, bad, tin, tout = [], [], 0, 0
    for r in client.messages.batches.results(meta["batch_id"]):
        row = rows[int(r.custom_id[1:])]
        rec = {"group_id": row["group_id"], "split": row["split"], "langdetect_lang": row["langdetect_lang"], "arm": meta["arm"]}
        if r.result.type != "succeeded":
            bad.append({**rec, "error": r.result.type})
            continue
        m = r.result.message
        tin, tout = tin + m.usage.input_tokens, tout + m.usage.output_tokens
        if m.stop_reason == "refusal":
            bad.append({**rec, "error": "refusal"})
            continue
        try:
            rec["labels"] = json.loads(next((x.text for x in m.content if x.type == "text"), ""))
            out.append(rec)
        except json.JSONDecodeError:
            bad.append({**rec, "error": "bad_json"})
    with open(HERE / "r3_labels.jsonl", "w", encoding="utf-8") as f:
        for r in out + bad:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cost = (tin * PRICE_IN + tout * PRICE_OUT) / 2
    msg = (f"成功 {len(out)} / 失败 {len(bad)}; 输入 {tin:,} / 输出 {tout:,} token; Batch 价约 ${cost:.2f}")
    (HERE / "r3_labels_summary.txt").write_text(msg, encoding="utf-8")
    print(msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "count", "submit", "collect"])
    ap.add_argument("--arm", default="B_no_think", choices=list(ARMS))
    ap.add_argument("--env", default=None, help="含 ANTHROPIC_API_KEY= 的 .env 文件路径")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    load_key(args.env)
    {"pilot": pilot, "count": count, "submit": submit, "collect": collect}[args.mode](args)
