"""clean.py — Step 1 清洗: 只去掉"真噪声", 不碰人类投票标签

输入: 140k 原始 parquet 目录 (download.py 的输出, 视为不可信文件, 用 python -I 读)
输出:
  data/arena_clean.parquet            主数据: evaluation_order == 1, 清洗后
  data/arena_followup_clean.parquet   追问票子集: evaluation_order > 1, 同样规则清洗, 本阶段不用
  reports/step1_cleaning.md           每条规则删了多少行

关键决策 (面试可讲)
------------------
D0 范围 = "首轮路由" (Yiqing 2026-11-07 确认)。
   Arena 同一会话里投完一票会换一对新模型继续聊 (evaluation_order 2, 3, ...)。
   这些追问票的 prompt 前面有别的模型留下的上下文, 路由器只看这一句追问会把难题误判成简单题。
   所以主数据只用 order == 1; 追问票不删, 单独存, 以后研究"中途降级"用。
D1 prompt = 第一条用户消息 (只取 text 段)。路由决策发生在用户发出第一句话的那一刻,
   后续轮次、回答内容、回答长度都是"未来信息", 一律不进主表 (v1 吃过这个亏)。
D2 删除规则, 按顺序执行, 每条只统计"前面规则没删掉的行":
   R1 第一条消息去空白后 < 16 字符: "hi", "test", "1+1" 这类几乎没有路由信息,
      而且投票多半是随手点的 (真噪声)。
   R2 任一侧回答为空: 用户在比"一个回答 vs 什么都没有", 测的不是模型能力。
   R3 任一侧回答被服务商内容过滤截断 ("The generated text has been blocked by our content filters."):
      数据里只有这一种明确的"系统报错"。这是 API 网关插入的文字, 不是模型写的。
      实际几乎全来自 amazon nova 两个模型。
      注意: 模型自己拒答 ("I can't help with that") 是模型表现, 不删。
      也检查过 "TypeError..." 开头的回答 (2 条), 是 prompt 本身在问报错, 不算系统错误。
   R4 完全重复的 (prompt 原文, model_a, model_b) 组合只保留时间最早的一条:
      按需求文档执行。原因是无法区分"同一人重复提交"和"不同人独立投票",
      保守起见去掉。不同模型对上的同一 prompt 不受影响, Step 3 会用它们算软标签。
D3 语言: 保留全部语言, 只打标记, 不删。
   language = 数据集自带标记 (LMArena 统一检测, 来源一致);
   langdetect_lang = 我们用 langdetect 再测一遍, 只用来报告两者一致率, 作为交叉检查。
   is_en = (language == "en"), 后面做"只看英文"的敏感性分析。
D4 tie / both_bad 原样保留在 winner 列, 不归到任何一方 (Step 2 再打 y 标签)。

用法: python -I clean.py <原始 parquet 目录> <v2 根目录>
"""
import glob
import sys
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # langdetect 内部有随机性, 固定种子保证可复现

MIN_PROMPT_CHARS = 16
FILTER_MSG = "The generated text has been blocked by our content filters"

src, root = sys.argv[1], Path(sys.argv[2])
cols = ["id", "evaluation_session_id", "evaluation_order", "model_a", "model_b", "winner",
        "conversation_a", "conversation_b", "language", "is_code", "timestamp"]
raw = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{src}/*.parquet"))],
                ignore_index=True)


def text_of(turn):
    return " ".join(p["text"] for p in turn["content"] if p["type"] == "text" and p["text"])


def first(conv, role):
    """对话里第一个指定角色的文本; 没有则返回空串。"""
    return next((text_of(t) for t in conv if t["role"] == role), "")


df = pd.DataFrame({
    "id": raw.id,
    "session_id": raw.evaluation_session_id,
    "evaluation_order": raw.evaluation_order,
    "model_a": raw.model_a,
    "model_b": raw.model_b,
    "winner": raw.winner,
    "prompt": raw.conversation_a.map(lambda c: first(c, "user")),
    "language": raw.language,
    "is_code": raw.is_code,
    "timestamp": raw.timestamp,
})
# 回答只用于判断 R2/R3, 用完即丢, 不写进输出表
resp_a = raw.conversation_a.map(lambda c: first(c, "assistant")).str.strip()
resp_b = raw.conversation_b.map(lambda c: first(c, "assistant")).str.strip()
del raw

rules = {
    "R1 首条消息 < 16 字符": df.prompt.str.strip().str.len() < MIN_PROMPT_CHARS,
    "R2 有一侧回答为空": (resp_a == "") | (resp_b == ""),
    "R3 回答被服务商内容过滤截断": resp_a.str.contains(FILTER_MSG, regex=False)
                                   | resp_b.str.contains(FILTER_MSG, regex=False),
}


def clean(part: pd.DataFrame, name: str):
    """按顺序套规则, 返回 (清洗后表, 每条规则的删除记录)。"""
    log = [("起始", len(part), 0)]
    keep = part
    for rule, mask in rules.items():
        drop = mask.loc[keep.index]
        keep = keep[~drop]
        log.append((rule, len(keep), int(drop.sum())))
    # R4: 同 prompt + 同 (model_a, model_b) 只留最早一条
    keep = keep.sort_values("timestamp")
    dup = keep.duplicated(["prompt", "model_a", "model_b"], keep="first")
    keep = keep[~dup].sort_index()
    log.append(("R4 完全重复的 prompt+model_a+model_b", len(keep), int(dup.sum())))
    return keep, pd.DataFrame(log, columns=[f"{name}: 规则", "剩余行数", "本条删除"])


main, log_main = clean(df[df.evaluation_order == 1], "主数据 order=1")
follow, log_follow = clean(df[df.evaluation_order > 1], "追问票 order>1")


def lang_detect(s):
    try:
        return detect(s)
    except LangDetectException:  # 纯数字/符号等无法判断
        return "und"


for part in (main, follow):
    part["langdetect_lang"] = part.prompt.map(lang_detect)
    part["is_en"] = part.language == "en"
    part["prompt_len"] = part.prompt.str.len()

(root / "data").mkdir(exist_ok=True)
main.to_parquet(root / "data/arena_clean.parquet", index=False)
follow.to_parquet(root / "data/arena_followup_clean.parquet", index=False)

# ---------- 报告 ----------
agree = (main.language == main.langdetect_lang)
known = main.language != "und"
lang_tab = main.language.value_counts().head(10).to_frame("行数")
lang_tab["占比"] = (lang_tab["行数"] / len(main)).round(3)
win_tab = main.winner.value_counts().to_frame("行数")
win_tab["占比"] = (win_tab["行数"] / len(main)).round(3)
pl_week = (main.assign(week=main.timestamp.dt.to_period("W").astype(str), pl=main.language == "pl")
           .groupby("week").pl.mean().round(3))

md = [
    "# Step 1 清洗结果", "",
    f"原始 {len(df)} 行 = order=1 {int((df.evaluation_order == 1).sum())} 行 + 追问票 {int((df.evaluation_order > 1).sum())} 行 (追问票单独存)。", "",
    log_main.to_markdown(index=False), "", log_follow.to_markdown(index=False), "",
    "## 主数据 winner 分布", "", win_tab.to_markdown(), "",
    "## 主数据语言前 10 (自带 language)", "", lang_tab.to_markdown(), "",
    f"英文 is_en: {main.is_en.sum()} 行 ({main.is_en.mean():.1%})",
    f"自带 language 与 langdetect 一致率: 全部 {agree.mean():.1%}; 去掉自带标记为 und 的行 {agree[known].mean():.1%}",
    f"英文行一致率: {agree[main.language == 'en'].mean():.1%}", "",
    "## 波兰语占比按周", "", pl_week.to_frame("pl 占比").to_markdown(), "",
    "## 主数据首条消息长度 (字符)", "",
    main.prompt_len.describe(percentiles=[.1, .25, .5, .75, .9, .99]).round(0).to_frame("chars").to_markdown(),
]
(root / "reports/step1_cleaning.md").write_text("\n".join(md))
print("\n".join(md))
