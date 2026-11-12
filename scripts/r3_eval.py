"""r3_eval.py — 方案一 vs 方案二 的正式对比 (开发集 5 折, 口径 ②), 按 reports/r3_prereg.md

同一批 ② eval 票、同一次 bootstrap 重采样 (1000 次, 按 prompt 组) 上同时算所有路由器, 所以路由器之间的差是配对的。
- 主指标: 按调用比例的 APGR (用 P(赢) 排序), 和评分差基线比, 以及方案间的配对差。
- 次要: 美元主案例 (高分方更贵的 ② 票) APGR$ 和 CPT$(50%), 排序键 = (P赢 − P输) / max(典型成本差, $0.25/千次);
        配对内 APGR$ (下限) 复用 evaluate_cost.within_pairs。
- Yiqing 加的: 按 has_single_answer 分组的 APGR; is_open_ended 等字段与方案一 PC1 坐标 t1 的秩相关。

用法: python -I r3_eval.py <v2 根目录> [--final]
产出: reports/r3_compare[_final].md / .json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

root = Path(sys.argv[1])
use_final = "--final" in sys.argv
sfx = "_final" if use_final else ""
import os
VAR = os.environ.get("R3_VARIANT", "")   # 例如 "_ep100": MF 类路由器换成最多 100 轮早停的敏感性版本

src = (Path(__file__).parent / "evaluate_cost.py").read_text().split('\nif __name__ == "__main__":')[0]
ns = {"__name__": "evc"}
argv = sys.argv
sys.argv = [argv[0], str(root)] + (["--final"] if use_final else [])
exec(compile(src, "evaluate_cost.py", "exec"), ns)
sys.argv = argv
curve, metrics, within_pairs, BASE, FLOOR = ns["curve"], ns["metrics"], ns["within_pairs"], ns["BASE"], ns["FLOOR"]
N_BOOT, SEED, GRID, FINE = ns["N_BOOT"], ns["SEED"], ns["GRID"], ns["FINE"]

ev = ns["ev"]
ev = ev[ev.is_strong_vs_weak.values].reset_index(drop=True)

# ---- 路由器: 名字 -> 预测文件 (列 p_win = P(赢), p = (1 + P赢 − P输)/2) ----
P = root / "preds"
files = {
    "R1 (方案一, 全量票)": (P / f"r1b_win{sfx}.parquet", P / f"r1b_eff{sfx}.parquet"),
    "R2 (方案一, 全量票)": (P / ("r2_nested.parquet" if not use_final else "r2_win_final.parquet"),
                         P / ("r2_nested.parquet" if not use_final else "r2_eff_final.parquet")),
    "R2-② (方案一, 只用 ② 票)": P / f"r3/r2sw{sfx}{VAR}.parquet",
    "R3-LR": P / f"r3/r3lr{sfx}.parquet",
    "R3-MF": P / f"r3/r3mf{sfx}{VAR}.parquet",
    "R3 (方案二, 每折选 LR/MF)": P / f"r3/r3{sfx}.parquet",
    "R2 + R3 (组合)": P / f"r3/r2r3{sfx}{VAR}.parquet",
}
pw, pe = {}, {}
for name, f in files.items():
    if isinstance(f, tuple):
        a = pd.read_parquet(f[0])
        b = pd.read_parquet(f[1])
        a = a.rename(columns={"p_win": "pw"}) if "p_win" in a else a.rename(columns={"p": "pw"})
        d = ev[["context", "id"]].merge(a[["context", "id", "pw"]], on=["context", "id"], how="left") \
                                 .merge(b[["context", "id", "p"]], on=["context", "id"], how="left")
    else:
        d = ev[["context", "id"]].merge(pd.read_parquet(f).rename(columns={"p_win": "pw"})[["context", "id", "pw", "p"]],
                                        on=["context", "id"], how="left")
    assert d.pw.notna().all() and d.p.notna().all(), name
    pw[name], pe[name] = d.pw.values, d.p.values

p_bt = 1 / (1 + 10 ** (-ev.rating_gap.values / 400))
tb = ev[["context", "id"]].merge(pd.read_parquet(P / f"base_tie_bucket{sfx}.parquet"), on=["context", "id"], how="left")
BT, TIE = "评分差基线 (BT)", "带平局评分差基线 (正式)"
pw = {BT: p_bt, TIE: tb.p_win.values, **pw}
pe = {BT: p_bt, TIE: tb.p.values, **pe}
names = list(pw)
R2SW, R3, R2R3, R2 = "R2-② (方案一, 只用 ② 票)", "R3 (方案二, 每折选 LR/MF)", "R2 + R3 (组合)", "R2 (方案一, 全量票)"
PAIRS = [(R3, R2SW), (R2R3, R2SW), (R3, R2), (R2R3, R2), (R2SW, R2), ("R3-LR", R2SW), ("R3-MF", R2SW)]

# ---- 标签 和 PC1 坐标 ----
lab = []
for line in open(root / "r3/r3_labels.jsonl", encoding="utf-8"):
    r = json.loads(line)
    lab.append({"group_id": r["group_id"], **(r.get("labels") or {})})
lab = pd.DataFrame(lab)
ev = ev.merge(lab, on="group_id", how="left")

q = ev.q_hi.values
main = (ev.cost_class == "高分方更贵").values
dt = np.maximum((ev.typ_hi - ev.typ_lo).values, FLOOR)
extra = (ev.cost_hi - ev.cost_lo).values
groups, gidx = np.unique(ev.group_id.values, return_inverse=True)
hsa = ev.has_single_answer.values  # 缺失 (拒答) 的票不进分组表


def all_scores(w, rng):
    """一次重采样 (权重 w) 下所有路由器的: 调用比例 APGR / CPT50, 美元主案例 APGR$ / CPT$50, 分组 APGR。"""
    out = {}
    for n in names:
        pf = curve(q, pw[n], w, extra, rng)[1]
        a, c50, _ = metrics(pf)
        pc = curve(q[main], (2 * pe[n][main] - 1) / dt[main], w[main], extra[main], rng)[0]
        a2, c2, _ = metrics(pc)
        grp = []
        for g in range(4):
            m = hsa == g
            grp.append(metrics(curve(q[m], pw[n][m], w[m], extra[m], rng)[1])[0])
        out[n] = [a, c50, a2, c2, *grp]
    return out


COLS = ["APGR", "CPT50", "APGR$", "CPT$50", "hsa0", "hsa1", "hsa2", "hsa3"]
point = all_scores(np.ones(len(q)), np.random.default_rng(SEED))
rng = np.random.default_rng(SEED)
boots = {n: [] for n in names}
for i in range(N_BOOT):
    cnt = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = all_scores(cnt[gidx].astype(float), rng)
    for n in names:
        boots[n].append(r[n])
    if i % 200 == 0:
        print("bootstrap", i, flush=True)
boots = {n: np.array(v) for n, v in boots.items()}


def ci(n, j):
    lo, hi = np.nanpercentile(boots[n][:, j], [2.5, 97.5])
    return [round(float(point[n][j]), 3), round(float(lo), 3), round(float(hi), 3)]


def dci(a, b, j):
    d = boots[a][:, j] - boots[b][:, j]
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return [round(float(point[a][j] - point[b][j]), 3), round(float(lo), 3), round(float(hi), 3)]


res = {"n_votes": int(len(q)), "n_groups": int(len(groups)), "n_main": int(main.sum()),
       "routers": {n: {c: ci(n, j) for j, c in enumerate(COLS)} for n in names},
       "vs_BT": {n: {c: dci(n, BT, j) for j, c in enumerate(COLS)} for n in names if n != BT},
       "vs_TIE$": {n: dci(n, TIE, 2) for n in names if n != TIE},
       "pairs": {f"{a} − {b}": {c: dci(a, b, j) for j, c in enumerate(COLS)} for a, b in PAIRS},
       "hsa_n": {g: int((hsa == g).sum()) for g in range(4)}}

# ---- 配对内 APGR$ (下限) ----
wp_routers = {BASE: (2 * pe[TIE] - 1) / dt, **{n: (2 * pe[n] - 1) / dt for n in names if n not in (BT, TIE)}}
d_main = ev[main].reset_index(drop=True)
res["within_pairs"] = within_pairs(d_main, {k: v[main] for k, v in wp_routers.items()})

# ---- 标签 与 PC1 (t1) 的秩相关 (开发集 ② prompt, 每个 prompt 一行) ----
if not use_final:
    ax = pd.read_parquet(root / "data/r2_prompt_axes.parquet", columns=["group_id", "t1", "t2", "t1_oof"])
    pr = lab.merge(ax.drop_duplicates("group_id"), on="group_id")
    corr = {}
    for f in ["is_open_ended", "has_single_answer", "is_creative", "has_format_constraint", "difficulty", "expected_length"]:
        m = pr[f].notna()
        corr[f] = {k: round(float(spearmanr(pr.loc[m, f], pr.loc[m, k]).statistic), 3) for k in ["t1", "t1_oof", "t2"]}
    res["spearman"] = {"n_prompts": int(len(pr)), "by_field": corr,
                       "t1_mean_by_open_ended": pr.groupby("is_open_ended").t1_oof.agg(["mean", "count"]).round(3).reset_index().values.tolist(),
                       "t1_mean_by_task": pr.groupby("task_type").t1_oof.agg(["mean", "count"]).round(3).sort_values("mean").reset_index().values.tolist()}

(root / f"reports/r3_compare{sfx}{VAR}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))

# ---- 报告 ----
f = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
L = [f"# 方案一 vs 方案二 ({'测试集' if use_final else '开发集 5 折'}, 口径 ②)", "",
     f"② 票 {res['n_votes']} 张, prompt 组 {res['n_groups']}; 美元主案例 {res['n_main']} 张。点估计 [95% 区间], 1000 次按 prompt 组配对 bootstrap。", "",
     "## 主指标: 按调用比例 APGR", "", "| 路由器 | APGR | Δ vs 评分差基线 | CPT(50%) |", "|---|---|---|---|"]
for n in names:
    L.append(f"| {n} | {f(res['routers'][n]['APGR'])} | {f(res['vs_BT'][n]['APGR']) if n != BT else '—'} | {f(res['routers'][n]['CPT50'])} |")
L += ["", "## 方案之间的配对差", "", "| 对比 | ΔAPGR | ΔAPGR$ (美元主案例) |", "|---|---|---|"]
for k, v in res["pairs"].items():
    L.append(f"| {k} | {f(v['APGR'])} | {f(v['APGR$'])} |")
L += ["", "## 美元主案例 (贵强档 vs 便宜弱档)", "", "| 路由器 | APGR$ | Δ vs 正式基线 | CPT$(50%) |", "|---|---|---|---|"]
for n in names:
    L.append(f"| {n} | {f(res['routers'][n]['APGR$'])} | {f(res['vs_TIE$'][n]) if n != TIE else '—'} | {f(res['routers'][n]['CPT$50'])} |")
wp = res["within_pairs"]
L += ["", f"## 配对内 APGR$ (下限; 票数 ≥ 30 的 {wp['n_pairs']} 个模型对, {wp['n_votes']} 票)", "",
      "| 路由器 | 配对内 APGR$ | Δ vs 正式基线 |", "|---|---|---|"]
for n, m in wp["metrics"].items():
    k = "配对内APGR$ (按票数加权)"
    L.append(f"| {n} | {f(m[k])} | {f(m['Δ ' + k]) if 'Δ ' + k in m else '—'} |")
L += ["", "## 按 has_single_answer 分组 (按调用比例 APGR)", "",
      "| 路由器 | " + " | ".join(f"{g} 分 ({res['hsa_n'][g]} 票)" for g in range(4)) + " |", "|---|---|---|---|---|"]
for n in names:
    L.append(f"| {n} | " + " | ".join(f(res['routers'][n][f'hsa{g}']) for g in range(4)) + " |")
L += ["", "相对评分差基线的提升:", "", "| 路由器 | " + " | ".join(f"{g} 分" for g in range(4)) + " |", "|---|---|---|---|---|"]
for n in names:
    if n != BT:
        L.append(f"| {n} | " + " | ".join(f(res['vs_BT'][n][f'hsa{g}']) for g in range(4)) + " |")
if "spearman" in res:
    s = res["spearman"]
    L += ["", f"## 标签与方案一 PC1 坐标的秩相关 (开发集 ② 的 {s['n_prompts']} 个 prompt)", "",
          "t1 = 用全开发集向量算的 PC1 坐标; t1_oof = 折外版本 (每个 prompt 的坐标来自没见过它的折)。", "",
          "| 字段 | 与 t1 | 与 t1_oof | 与 t2 |", "|---|---|---|---|"]
    for k, v in s["by_field"].items():
        L.append(f"| {k} | {v['t1']} | {v['t1_oof']} | {v['t2']} |")
    L += ["", "is_open_ended 各档的 t1_oof 均值: " + "; ".join(f"{int(a)} 分 {b} ({int(c)} 条)" for a, b, c in s["t1_mean_by_open_ended"]),
          "", "各 task_type 的 t1_oof 均值: " + "; ".join(f"{a} {b} ({int(c)})" for a, b, c in s["t1_mean_by_task"])]
(root / f"reports/r3_compare{sfx}{VAR}.md").write_text("\n".join(L))
print("\n".join(L))
