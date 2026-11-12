"""evaluate_cost.py — 阶段 2 Step 1: 美元成本口径的评估 (在 evaluate.py 的 PGR 定义上把横轴换成美元)

和 evaluate.py 的区别
--------------------
- 只用 "高分方典型成本 > 低分方" 的票 (cost_table.py 第 4 条); 其余票只统计比例。
- 横轴 = 归一化美元成本: (这一批请求花的钱 - 全发低分方的钱) / (全发高分方的钱 - 全发低分方的钱)。
  每票多花的钱 = 高分方实际成本 - 低分方实际成本, 不同票不一样 (长回答、贵模型多花得多)。
  同时报每千次请求的绝对美元数。
- 排序规则 (Yiqing 2026-11-09 定): 美元口径下所有路由器统一按 "预期质量提升 ÷ 增量成本" 排序,
  key = (2p - 1) / max(典型成本差, FLOOR)。p = 路由器输出的 P(高分方赢), 发高分方比发低分方
  预期多得 p - (1 - p) = 2p - 1 分质量; 典型成本差 = 高分方与低分方各自典型单次成本之差 (见 cost_table.py)。
  FLOOR = 每千次请求 0.25 美元 (约为成本差的第 5 百分位), 防止价格几乎一样的模型对比值爆炸;
  报告附 FLOOR 取 0.1 / 1.0 和取对数的敏感性。
- 正式基线 = 评分差 ÷ 价格差 (p 取 BT 评分差推出的胜率)。"评分差基线" (只按 p 排序, 不看价格) 保留为对照行。
- 配对内指标 (仅美元主案例): 按模型对分组, 每组内单独算 APGR$, 再按票数加权平均。组内价格差固定,
  所以这个数只衡量路由器"在同一对模型里靠 prompt 排序"的能力。评分差类基线在组内只剩跨折的评分抖动,
  预期 ≈ 随机。只用票数 ≥ 30 的模型对; 组内质量差距 (全发强 - 全发弱) < 0.05 的组 APGR 不稳定, 加权平均时跳过,
  同时报不会爆炸的合并版 (各组曲线下增益之和 ÷ 各组质量差距之和, 等价于按 票数×质量差距 加权)。
- 单票实际多花的钱偶尔为负 (贵模型这次答得短), 累计成本取"到目前为止的最大值"保证横轴单调 (PGR 本身不做单调化)。
- APGR$ = PGR 对归一化美元成本的曲线下面积; CPT$(50%) = PGR 到 50% 时花掉的归一化成本。
- 每个口径同时给出"按调用比例"的旧 APGR, 方便和阶段 1 对照。

用法: python -I evaluate_cost.py <v2 根目录> [--demo | <预测文件> <名字>] [--final]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT, SEED = 1000, 2026
GRID = np.linspace(0, 1, 11)
FINE = np.linspace(0, 1, 101)
BUCKETS = [(0, 50, "<50"), (50, 150, "50–150"), (150, np.inf, ">150")]
FLOOR = 0.25 / 1000          # 美元 / 单次请求
BASE = "评分差÷价格差 (正式基线, 分桶赢/平/输)"
MIN_PAIR, MIN_GAP = 30, 0.05

root = Path(sys.argv[1])
args = sys.argv[2:]
use_final = "--final" in args
args = [a for a in args if a != "--final"]
contexts = ["final"] if use_final else [f"cv{k}" for k in range(5)]

fl = pd.read_parquet(root / "data/fold_labels.parquet",
                     columns=["context", "id", "role", "group_id", "q_hi", "rating_gap", "is_strong_vs_weak",
                              "hi_model", "lo_model", "model_a", "tier_a", "tier_b"])
ev = fl[(fl.role == "eval") & fl.context.isin(contexts)].reset_index(drop=True)
vc = pd.read_parquet(root / "data/vote_costs.parquet")
ev = ev.merge(vc, on="id", how="left")
hi_is_a = ev.hi_model.values == ev.model_a.values
ev["cost_hi"] = np.where(hi_is_a, ev.cost_a, ev.cost_b)
ev["cost_lo"] = np.where(hi_is_a, ev.cost_b, ev.cost_a)
ev["typ_hi"] = np.where(hi_is_a, ev.typ_a, ev.typ_b)
ev["typ_lo"] = np.where(hi_is_a, ev.typ_b, ev.typ_a)
ev["cost_class"] = np.select([ev.typ_hi.isna() | ev.typ_lo.isna(), ev.typ_hi > ev.typ_lo],
                             ["无价", "高分方更贵"], "高分方更便宜或一样")


def curve(q, p, w, extra, rng):
    """返回 (按美元的 PGR 网格, 按调用比例的 PGR 网格, r_s, r_w, 全发低分方成本, 全发高分方成本)。"""
    order = np.lexsort((rng.random(len(p)), -p))
    qs, ws, es = q[order], w[order], extra[order]
    W = ws.sum()
    r_w = (ws * (1 - qs)).sum() / W
    r_s = (ws * qs).sum() / W
    gain = np.concatenate([[0], np.cumsum(ws * (2 * qs - 1))]) / W
    pgr_pts = gain / (r_s - r_w)
    cw = np.concatenate([[0], np.cumsum(ws)]) / W
    ce = np.concatenate([[0], np.cumsum(ws * es)])
    ce = np.maximum.accumulate(ce) / ce[-1]
    ce[-1] = 1.0
    pgr_cost = np.interp(FINE, ce, pgr_pts)
    pgr_frac = np.interp(FINE, cw, pgr_pts)
    return pgr_cost, pgr_frac, r_s, r_w


def metrics(pgr):
    g = np.interp(GRID, FINE, pgr)
    apgr = np.trapezoid(g, GRID)
    cpt = {x: FINE[np.argmax(pgr >= x)] if (pgr >= x).any() else np.nan for x in (0.5, 0.8)}
    return apgr, cpt[0.5], cpt[0.8]


def evaluate(d, routers):
    q = d.q_hi.values
    extra = (d.cost_hi - d.cost_lo).values
    groups, gidx = np.unique(d.group_id.values, return_inverse=True)
    rng = np.random.default_rng(SEED)
    out = {"n_votes": int(len(d)), "n_groups": int(len(groups)),
           "usd_per_1k_all_lo": round(float(d.cost_lo.mean() * 1000), 3),
           "usd_per_1k_all_hi": round(float(d.cost_hi.mean() * 1000), 3)}
    point, frac_apgr = {}, {}
    for name, p in routers.items():
        pc, pf, r_s, r_w = curve(q, p, np.ones(len(q)), extra, np.random.default_rng(SEED))
        point[name] = metrics(pc)
        frac_apgr[name] = round(float(metrics(pf)[0]), 3)
        out.setdefault("curves", {})[name] = pc.round(4).tolist()
    out["r_strong"], out["r_weak"] = round(float(r_s), 4), round(float(r_w), 4)
    boots = {n: [] for n in routers}
    for _ in range(N_BOOT):
        cnt = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        w = cnt[gidx].astype(float)
        for name, p in routers.items():
            boots[name].append(metrics(curve(q, p, w, extra, rng)[0]))
    out["metrics"] = {}
    base = np.array(boots[BASE])
    for name in routers:
        b = np.array(boots[name])
        row = {"APGR_按调用比例": frac_apgr[name]}
        for j, m in enumerate(["APGR$", "CPT$50", "CPT$80"]):
            lo, hi = np.nanpercentile(b[:, j], [2.5, 97.5])
            row[m] = [round(float(point[name][j]), 3), round(float(lo), 3), round(float(hi), 3)]
        if name != BASE:
            dlt = b[:, 0] - base[:, 0]
            row["ΔAPGR$_vs_正式基线"] = [round(float(point[name][0] - point[BASE][0]), 3),
                                     *np.percentile(dlt, [2.5, 97.5]).round(3).tolist()]
        out["metrics"][name] = row
    return out


def within_pairs(d, routers):
    """配对内 APGR$: 每个模型对单独算, 再加权。返回点估计、区间和相对正式基线的提升。"""
    pair = (d.hi_model + " | " + d.lo_model).values
    names, pidx = np.unique(pair, return_inverse=True)
    keep = np.bincount(pidx) >= MIN_PAIR
    m = keep[pidx]
    d, pidx = d[m], pidx[m]
    routers = {k: v[m] for k, v in routers.items()}
    q, extra = d.q_hi.values, (d.cost_hi - d.cost_lo).values
    groups, gidx = np.unique(d.group_id.values, return_inverse=True)
    members = [np.where(pidx == k)[0] for k in np.unique(pidx)]

    def score(w, rng):
        res = {}
        for name, p in routers.items():
            num_w = den_w = num_p = den_p = 0.0
            for idx in members:
                wk = w[idx]
                n = wk.sum()
                if n == 0:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    pc, _, r_s, r_w = curve(q[idx], p[idx], wk, extra[idx], rng)
                gap = r_s - r_w
                if abs(gap) < 1e-12:
                    continue
                apgr = metrics(pc)[0]
                num_p += n * apgr * gap
                den_p += n * gap
                if gap >= MIN_GAP:
                    num_w += n * apgr
                    den_w += n
            res[name] = (num_w / den_w, num_p / den_p)
        return res

    point = score(np.ones(len(q)), np.random.default_rng(SEED))
    rng = np.random.default_rng(SEED)
    boots = {n: [] for n in routers}
    for _ in range(N_BOOT):
        cnt = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        r = score(cnt[gidx].astype(float), rng)
        for n in routers:
            boots[n].append(r[n])
    out = {"n_votes": int(len(d)), "n_pairs": int(len(members)), "metrics": {}}
    base = np.array(boots[BASE])
    for name in routers:
        b = np.array(boots[name])
        row = {}
        for j, lab in enumerate(["配对内APGR$ (按票数加权)", "配对内APGR$ (合并版)"]):
            row[lab] = [round(float(point[name][j]), 3), *np.percentile(b[:, j], [2.5, 97.5]).round(3).tolist()]
            if name != BASE:
                row["Δ " + lab] = [round(float(point[name][j] - point[BASE][j]), 3),
                                   *np.percentile(b[:, j] - base[:, j], [2.5, 97.5]).round(3).tolist()]
        out["metrics"][name] = row
    return out


def sensitivity(d, p_bt):  # p_bt: 正式基线的 p (2p−1 = 预期质量提升)
    """正式基线对 FLOOR 和取对数的敏感性 (只报点估计)。"""
    dt = (d.typ_hi - d.typ_lo).values
    q, extra = d.q_hi.values, (d.cost_hi - d.cost_lo).values
    variants = {f"下限 ${f}/千次": (2 * p_bt - 1) / np.maximum(dt, f / 1000) for f in (0.1, 0.25, 1.0)}
    variants["取对数: (2p-1) / log(1 + 成本差/下限)"] = (2 * p_bt - 1) / np.log1p(dt / FLOOR)
    out = {"触到下限的票占比": round(float((dt < FLOOR).mean()), 4)}
    for k, v in variants.items():
        out[k] = round(float(metrics(curve(q, v, np.ones(len(q)), extra, np.random.default_rng(SEED))[0])[0]), 3)
    return out


def run_all(d, preds):
    m0 = (d.cost_class == "高分方更贵").values
    d, preds = d[m0], {k: v[m0] for k, v in preds.items()}
    res = {"① 合并 (美元)": evaluate(d, preds)}
    for lo, hi, lab in BUCKETS:
        m = ((d.rating_gap >= lo) & (d.rating_gap < hi)).values
        res[f"① 评分差 {lab} (美元)"] = evaluate(d[m], {k: v[m] for k, v in preds.items()})
    m = d.is_strong_vs_weak.values
    res["② 贵强档 vs 便宜弱档 (美元主案例)"] = evaluate(d[m], {k: v[m] for k, v in preds.items()})
    res["② 配对内"] = within_pairs(d[m], {k: v[m] for k, v in preds.items()})
    p_bt = 1 / (1 + 10 ** (-d.rating_gap.values / 400))
    pb = d.p_tie_bucket.values
    res["正式基线敏感性"] = {"① 合并": sensitivity(d, pb), "② 美元主案例": sensitivity(d[m], pb[m])}
    return res


def report(res, name):
    lines = [f"# 美元口径评估: {name} ({'测试集' if use_final else '开发集 5 折, 合并各折 eval 票'})", "",
             "只含高分方典型成本更高的票。格式: 点估计 [95% 区间], bootstrap 1000 次按 prompt 组重采样。",
             "APGR$ / CPT$ 的横轴是归一化美元成本 (0 = 全发低分方, 1 = 全发高分方)。", ""]
    fmt = lambda v: f"{v[0]} [{v[1]}, {v[2]}]"
    for view, r in res.items():
        if view == "② 配对内":
            labs = ["配对内APGR$ (按票数加权)", "配对内APGR$ (合并版)"]
            lines += [f"## ② 配对内 APGR$  (票数 ≥ {MIN_PAIR} 的 {r['n_pairs']} 个模型对, 共 {r['n_votes']} 票)", "",
                      "每个模型对内单独算 APGR$ 再加权; 组内价格差固定, 只看靠 prompt 排序的能力。"
                      f"按票数加权版跳过组内质量差距 < {MIN_GAP} 的组; 合并版按 票数×质量差距 加权, 不会爆炸。", "",
                      "| 路由器 | 按票数加权 | Δ vs 正式基线 | 合并版 | Δ vs 正式基线 |", "|---|---|---|---|---|"]
            for rn, m in r["metrics"].items():
                d1 = fmt(m["Δ " + labs[0]]) if "Δ " + labs[0] in m else "—"
                d2 = fmt(m["Δ " + labs[1]]) if "Δ " + labs[1] in m else "—"
                lines.append(f"| {rn} | {fmt(m[labs[0]])} | {d1} | {fmt(m[labs[1]])} | {d2} |")
            lines.append("")
            continue
        if view == "正式基线敏感性":
            lines += ["## 正式基线对价格差下限的敏感性 (APGR$ 点估计)", "",
                      "| 口径 | " + " | ".join(k for k in r["① 合并"]) + " |",
                      "|---" * (len(r["① 合并"]) + 1) + "|"]
            for v, row in r.items():
                lines.append(f"| {v} | " + " | ".join(str(x) for x in row.values()) + " |")
            lines.append("")
            continue
        lines += [f"## {view}  (票数 {r['n_votes']}; 每千次请求: 全发低分方 ${r['usd_per_1k_all_lo']}, "
                  f"全发高分方 ${r['usd_per_1k_all_hi']})", "",
                  "| 路由器 | APGR$ | CPT$(50%) | CPT$(80%) | ΔAPGR$ vs 正式基线 | 对照: 按调用比例的 APGR |",
                  "|---|---|---|---|---|---|"]
        for rn, m in r["metrics"].items():
            dl = fmt(m["ΔAPGR$_vs_正式基线"]) if "ΔAPGR$_vs_正式基线" in m else "—"
            lines.append(f"| {rn} | {fmt(m['APGR$'])} | {fmt(m['CPT$50'])} | {fmt(m['CPT$80'])} | {dl} | "
                         f"{m['APGR_按调用比例']} |")
        lines.append("")
    return "\n".join(lines)


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    cjk = [f.name for f in fm.fontManager.ttflist if any(k in f.name for k in ("CJK", "Hei", "WenQuanYi", "Noto Sans SC"))]
    if cjk:
        matplotlib.rcParams["font.sans-serif"] = [cjk[0]]
    ink, muted, grid = "#2b2b29", "#6b6a63", "#e1e0d9"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59b6", "#888888", "#d4a017"]
    views = ["① 合并 (美元)", "② 贵强档 vs 便宜弱档 (美元主案例)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, facecolor="#fcfcfb")
    for ax, v in zip(axes, views):
        r = res[v]
        lo, hi = r["usd_per_1k_all_lo"], r["usd_per_1k_all_hi"]
        x = lo + FINE * (hi - lo)
        ax.set_facecolor("#fcfcfb")
        ax.plot([lo, hi], [0, 1], color=muted, lw=1, ls="--", label="随机期望")
        for (rn, c), col in zip(r["curves"].items(), colors):
            ax.plot(x, c, color=col, lw=2, label=f"{rn}  APGR$ {r['metrics'][rn]['APGR$'][0]}")
        ax.scatter([lo, hi], [0, 1], s=50, color=ink, zorder=5)
        ax.annotate("全发低分方", (lo, 0), (6, 8), textcoords="offset points", color=ink, fontsize=9)
        ax.annotate("全发高分方", (hi, 1), (-60, -14), textcoords="offset points", color=ink, fontsize=9)
        ax.set_title(f"{v} (n={r['n_votes']})", color=ink, fontsize=11, loc="left")
        ax.set_xlabel("每千次请求成本 (美元)", color=muted)
        ax.set_ylabel("PGR", color=muted)
        ax.grid(color=grid, lw=0.6)
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=muted)
        ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    if args and args[0] == "--demo":
        name, preds = "demo_美元口径基线", {}
    else:
        pf, name = args[0], args[1]
        pr = pd.read_parquet(pf) if pf.endswith(".parquet") else pd.read_csv(pf)
        ev = ev.merge(pr[["context", "id", "p"]], on=["context", "id"], how="left")
        assert ev.p.notna().all(), "预测文件缺少部分评估票"
        preds = {name: ev.p.values}
    rng = np.random.default_rng(SEED)
    p_bt = 1 / (1 + 10 ** (-ev.rating_gap.values / 400))
    dt = np.maximum((ev.typ_hi - ev.typ_lo).values, FLOOR)
    key = lambda p: (2 * p - 1) / dt       # 预期质量提升 ÷ 增量成本
    # 正式基线 (Yiqing 2026-11-10): 带平局的评分差基线, 预期提升 = P(赢) − P(输), 来自 tie_baselines.py
    # (分桶经验比例, 只用各情境 train 票拟合)。BT 半票版和三分类逻辑回归版作为对照行保留。
    sfx = "_final" if use_final else ""
    for tag, col in (("bucket", "p_tie_bucket"), ("mlogit", "p_tie_mlogit")):
        b = pd.read_parquet(root / f"preds/base_tie_{tag}{sfx}.parquet")[["context", "id", "p"]].rename(columns={"p": col})
        ev = ev.merge(b, on=["context", "id"], how="left")
        assert ev[col].notna().all()
    preds = {"随机路由": rng.random(len(ev)),
             "评分差基线 (对照, 不看价格)": p_bt,
             BASE: key(ev.p_tie_bucket.values),
             "评分差÷价格差 (对照: 三分类逻辑回归版)": key(ev.p_tie_mlogit.values),
             "评分差÷价格差 (对照: BT 半票版, 旧正式基线)": key(p_bt),
             **{k: key(v) for k, v in preds.items()}}
    res = run_all(ev, preds)
    out = root / "reports" / f"eval_{name}{'_final' if use_final else ''}"
    Path(f"{out}.json").write_text(json.dumps(res, ensure_ascii=False))
    Path(f"{out}.md").write_text(report(res, name))
    try:
        plot(res, Path(f"{out}.png"))
    except Exception as e:
        print("plot failed:", e)
    print(report(res, name))
