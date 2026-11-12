"""inspect_raw.py — 140k 原始数据概况 (第 0 步, 只读不改)

目的: 动任何清洗规则之前, 先看清数据实际长什么样, 和需求文档假设的字段对一遍。
- 140k 的 conversation_a 是 [{role, content: [{type, text, image, mimeType}]}] 结构;
  "第一条用户消息" = 第一个 role=='user' 的所有 text 段拼起来。(已核对: 全部是纯文本, 无图片)
- evaluation_order: 同一会话里第几次投票。>1 表示用户追问后换了一对新模型再投,
  此时对话里有别的模型留下的上下文, 不是"路由器看到一条新 prompt"。

用法: python -I inspect_raw.py <parquet 目录> <输出 md 路径>
"""
import glob
import sys
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
cols = ["id", "model_a", "model_b", "winner", "evaluation_session_id", "evaluation_order",
        "conversation_a", "language", "is_code", "timestamp"]
df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{src}/*.parquet"))],
               ignore_index=True)


def first_user(conv):
    for t in conv:
        if t["role"] == "user":
            return " ".join(p["text"] for p in t["content"] if p["type"] == "text" and p["text"])
    return None


L = df.conversation_a.map(first_user).str.len()
models = pd.concat([df.model_a, df.model_b])


def share(s):
    return s.value_counts().to_frame().assign(share=lambda t: (t["count"] / len(s)).round(3)).to_markdown()


lines = [
    "# 140k 原始数据概况", "",
    f"- 行数: {len(df)} (id 唯一: {df.id.nunique()}); 会话数: {df.evaluation_session_id.nunique()}",
    f"- evaluation_order=1: {(df.evaluation_order == 1).sum()} 行",
    f"- 模型数: {models.nunique()}",
    f"- 时间: {df.timestamp.min():%Y-%m-%d} ~ {df.timestamp.max():%Y-%m-%d}",
    "", "## winner 分布 (全部 / order=1)", "", share(df.winner), "",
    share(df.winner[df.evaluation_order == 1]),
    "", "## 语言 (自带 language 列) 前 12", "", share(df.language).split("\n", 14)[:14] and
    "\n".join(share(df.language).split("\n")[:14]),
    f"\n共 {df.language.nunique()} 种语言标记",
    "", "## 第一条用户消息字符长度", "",
    L.describe(percentiles=[.05, .1, .25, .5, .75, .9, .95, .99]).round(0).to_frame("chars").to_markdown(),
    f"\n< 16 字符: {(L < 16).sum()} 行",
    "", "## 每个模型出场次数", "", models.value_counts().to_frame().to_markdown(),
]
open(out, "w").write("\n".join(lines))
print("\n".join(lines[:8]))
