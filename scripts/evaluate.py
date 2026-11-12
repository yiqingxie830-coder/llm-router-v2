"""evaluate.py — Step 5: 路由器评估框架 (成本–质量曲线, PGR / APGR / CPT, bootstrap)

输入: 任意路由器的预测文件 (parquet/csv), 列 = context, id, p
      p = 该情境下 P(高分方明确胜出), 只用来排序, 不要求校准。
      context ∈ {cv0..cv4} (开发集 5 折交叉验证, 每折只评估自己的 eval 票) 或 final (测试集)。
标签来自 data/fold_labels.parquet (每折独立拟合 BT 的评分, 见 fold_ratings.py)。

核心定义 (面试可讲)
------------------
质量分 (每票): 发给高分方 → 高分方胜 1 / 平局 0.5 / 高分方输 0; 发给低分方 → 1 减去上述值。
   平局时两边都是 0.5: 发便宜的不掉质量 → 曲线天然奖励"平局发小模型" (Yiqing 的成本偏好)。
   注意: 这是"用户偏好"质量, 不是正确率, 与 RouteLLM 在 MT-Bench/MMLU 上的打分不同。
路由: 把 p 从高到低排, 前 f 比例发给高分方 (f = 成本代理: 发给强模型的比例)。
   p 相同的票随机打散 (固定种子), 所以常数预测 = 随机路由。
PGR(f) = (r(f) - r_弱) / (r_强 - r_弱): 在强弱差距里收回了多少。
APGR = PGR 曲线下面积, 用 f = 0, 0.1, ..., 1 共 10 个区间的梯形近似 (需求: 10 个阈值)。
   随机路由的期望 APGR = 0.5; 完美路由可以 > 0.5 很多 (平局和弱方胜的票都不用发强)。
CPT(x) = 达到 PGR ≥ x 所需的最小强模型调用比例 (1% 网格)。
Bootstrap: 1000 次, 按 group_id 整组重采样 (同一 prompt 的票不独立), 每次重算 r_强 / r_弱 / 全部指标。
   给出 95% 百分位区间; 相对基线的提升用"同一次重采样"配对计算 ΔAPGR。

两个口径 (Yiqing 2026-11-09)
--------------------------
① 合并口径: 所有模型对一起算。另按评分差分桶 (<50, 50–150, >150 分) 各报一组。
   !! 合并数字受 Arena 的配对分布影响 (哪些模型对被抽到得多), 不代表任何单一产品场景;
   !! 口径 ② 和分桶结果才是具体场景的参考。
② 强弱档口径: 只看强档 vs 弱档的票 (强档一定是高分方, 所以 p 直接可用)。

基线
----
- 全发弱 (f=0, PGR=0)、全发强 (f=1, PGR=1): 两个端点
- 随机路由: p = 随机数
- 评分差基线: p = BT 预测的高分方胜率 = 1 / (1 + 10^(-评分差/400)), 不看 prompt。
  所有"看 prompt"的路由器都要相对它报告提升 —— 否则分不清提升来自读懂 prompt 还是只是知道谁更强。

用法:
  python -I evaluate.py <v2 根目录> --demo            # 用随机路由和评分差基线跑通框架 (开发集 5 折)
  python -I evaluate.py <v2 根目录> <预测文件> <名字>   # 评估任意路由器 (默认开发集 5 折)
  加 --final 才在测试集上评估 (只在最终模型选定后用一次)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT, SEED = 1000, 2026
GRID = np.linspace(0, 1, 11)             # APGR 的 10 个区间
FINE = np.linspace(0, 1, 101)            # CPT 用 1% 网格
BUCKETS = [(0, 50, "<50"), (50, 150, "50–150"), (150, np.inf, ">150")]

root = Path(sys.argv[1])
args = sys.argv[2:]
use_final = "--final" in args
args = [a for a in args if a != "--final"]
contexts = ["final"] if use_final else [f"cv{k}" for k in range(5)]

fl = pd.read_parquet(root / "data/fold_labels.parquet",
                     columns=["context", "id", "role", "group_id", "q_hi", "rating_gap", "is_strong_vs_weak"])
ev = fl[(fl.role == "eval") & fl.context.isin(contexts)].reset_index(drop=True)


def curve(q, p, w, rng):
    """给定权重 w (bootstrap 次数), 返回 FINE 网格上的 PGR。q = 发高分方的质量分。"""
    order = np.lexsort((rng.random(len(p)), -p))  # p 降序, 同分随机
    qs, ws = q[order], w[order]
    W = ws.sum()
    r_w = (ws * (1 - qs)).sum() / W                      # 全发低分方
    r_s = (ws * qs).sum() / W                            # 全发高分方
    cw = np.concatenate([[0], np.cumsum(ws)]) / W         # 已发强的比例
    gain = np.concatenate([[0], np.cumsum(ws * (2 * qs - 1))]) / W  # 每多发一票强, 质量增加 q-(1-q)
    r = r_w + np.interp(FINE, cw, gain)
    return (r - r_w) / (r_s - r_w), r_s, r_w


def metrics(pgr):
    g = np.interp(GRID, FINE, pgr)
    apgr = np.trapezoid(g, GRID)
    cpt = {x: FINE[np.argmax(pgr >= x)] if (pgr >= x).any() else np.nan for x in (0.5, 0.8)}
    return apgr, cpt[0.5], cpt[0.8]


def evaluate(d: pd.DataFrame, routers: dict) -> dict:
    """routers: 名字 -> p 数组 (与 d 对齐)。返回点估计、区间、配对提升。"""
    q = d.q_hi.values
    groups, gidx = np.unique(d.group_id.values, return_inverse=True)
    rng = np.random.default_rng(SEED)
    out = {"n_votes": int(len(d)), "n_groups": int(len(groups))}
    point = {}
    for name, p in routers.items():
        pgr, r_s, r_w = curve(q, p, np.ones(len(q)), np.random.default_rng(SEED))
        point[name] = metrics(pgr)
        out.setdefault("curves", {})[name] = pgr.round(4).tolist()
    out["r_strong"], out["r_weak"] = round(float(r_s), 4), round(float(r_w), 4)
    boots = {n: [] for n in routers}
    for _ in range(N_BOOT):
        cnt = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        w = cnt[gidx].astype(float)
        for name, p in routers.items():
            boots[name].append(metrics(curve(q, p, w, rng)[0]))
    out["metrics"] = {}
    base = np.array(boots["评分差基线"]) if "评分差基线" in boots else None
    for name in routers:
        b = np.array(boots[name])
        row = {}
        for j, m in enumerate(["APGR", "CPT50", "CPT80"]):
            lo, hi = np.nanpercentile(b[:, j], [2.5, 97.5])
            row[m] = [round(float(point[name][j]), 3), round(float(lo), 3), round(float(hi), 3)]
        if base is not None and name != "评分差基线":
            dlt = b[:, 0] - base[:, 0]
            row["ΔAPGR_vs_评分差基线"] = [round(float(point[name][0] - point["评分差基线"][0]), 3),
                                     *np.percentile(dlt, [2.5, 97.5]).round(3).tolist()]
        out["metrics"][name] = row
    return out


def run_all(d: pd.DataFrame, preds: dict) -> dict:
    res = {"① 合并": evaluate(d, preds)}
    for lo, hi, lab in BUCKETS:
        m = (d.rating_gap >= lo) & (d.rating_gap < hi)
        res[f"① 评分差 {lab}"] = evaluate(d[m], {k: v[m.values] for k, v in preds.items()})
    m = d.is_strong_vs_weak.values
    res["② 强弱档"] = evaluate(d[m], {k: v[m] for k, v in preds.items()})
    return res


def report(res: dict, name: str):
    lines = [f"# 评估: {name} ({'测试集' if use_final else '开发集 5 折交叉验证, 合并各折 eval 票'})", "",
             "格式: 点估计 [95% 区间]; bootstrap 1000 次按 prompt 组重采样。", "",
             "> 口径 ① 合并数字受 Arena 配对分布影响, 不代表任何单一产品场景; 口径 ② 和分桶结果才是具体场景的参考。", ""]
    for view, r in res.items():
        lines += [f"## {view}  (票数 {r['n_votes']}, prompt 组 {r['n_groups']}; "
                  f"全发强质量 {r['r_strong']}, 全发弱质量 {r['r_weak']})", "",
                  "| 路由器 | APGR | CPT(50%) | CPT(80%) | ΔAPGR vs 评分差基线 |", "|---|---|---|---|---|"]
        fmt = lambda v: f"{v[0]} [{v[1]}, {v[2]}]"
        for rn, m in r["metrics"].items():
            lines.append(f"| {rn} | {fmt(m['APGR'])} | {fmt(m['CPT50'])} | {fmt(m['CPT80'])} | "
                         f"{fmt(m['ΔAPGR_vs_评分差基线']) if 'ΔAPGR_vs_评分差基线' in m else '—'} |")
        lines += ["", "端点基线: 全发弱 = (强调用 0%, PGR 0); 全发强 = (强调用 100%, PGR 1)。", ""]
    return "\n".join(lines)


def plot(res: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#2b2b29", "#6b6a63", "#e1e0d9"
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # 分类色 1-3 (全对验证通过)
    views = ["① 合并", "② 强弱档"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, facecolor="#fcfcfb")
    for ax, v in zip(axes, views):
        ax.set_facecolor("#fcfcfb")
        ax.plot([0, 1], [0, 1], color=muted, lw=1, ls="--", label="随机期望 (对角线)")
        for (rn, c), col in zip(res[v]["curves"].items(), colors):
            ax.plot(FINE, c, color=col, lw=2, label=f"{rn}  APGR {res[v]['metrics'][rn]['APGR'][0]}")
        ax.scatter([0, 1], [0, 1], s=50, color=ink, zorder=5)
        ax.annotate("全发弱", (0, 0), (6, 8), textcoords="offset points", color=ink, fontsize=9)
        ax.annotate("全发强", (1, 1), (-40, -14), textcoords="offset points", color=ink, fontsize=9)
        ax.set_title(f"{v} (n={res[v]['n_votes']})", color=ink, fontsize=11, loc="left")
        ax.set_xlabel("发给强 (高分) 模型的比例", color=muted)
        ax.set_ylabel("PGR", color=muted)
        ax.grid(color=grid, lw=0.6)
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=muted)
        ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    import matplotlib.font_manager as fm
    if args and args[0] == "--demo":
        name = "demo_随机与评分差基线"
        preds = {}
    else:
        pf, name = args[0], args[1]
        pr = pd.read_parquet(pf) if pf.endswith(".parquet") else pd.read_csv(pf)
        ev = ev.merge(pr[["context", "id", "p"]], on=["context", "id"], how="left")
        assert ev.p.notna().all(), "预测文件缺少部分评估票"
        preds = {name: ev.p.values}
    rng = np.random.default_rng(SEED)
    preds = {"随机路由": rng.random(len(ev)),
             "评分差基线": 1 / (1 + 10 ** (-ev.rating_gap.values / 400)), **preds}
    res = run_all(ev, preds)
    out = root / "reports" / f"eval_{name}{'_final' if use_final else ''}"
    Path(f"{out}.json").write_text(json.dumps(res, ensure_ascii=False))
    Path(f"{out}.md").write_text(report(res, name))
    try:
        import matplotlib
        cjk = [f.name for f in fm.fontManager.ttflist if any(k in f.name for k in ("CJK", "Hei", "WenQuanYi", "Noto Sans SC"))]
        if cjk:
            matplotlib.rcParams["font.sans-serif"] = [cjk[0]]
        plot(res, Path(f"{out}.png"))
    except Exception as e:  # 画图失败不影响指标
        print("plot failed:", e)
    print(report(res, name))
