"""prompt_axes.py — 把开发集 prompt 投到 R2 模型向量图的 PC1 / PC2 轴上, 用来给两轴命名并检验 PC1 是否是"难度"

轴的定义 (和 plot_r2_vectors.py 同一套 PCA):
  模型 m 随 prompt 变化的适合度 = g_m · (z_q − z̄) = g_w,m · ẑ_q,  g_w = Σ^(1/2) g,  ẑ_q = Σ^(−1/2) (z_q − z̄)
  PC k 方向 V_k (白化空间), 模型坐标 = g_w,m · V_k, prompt 坐标 t_k(q) = V_k · ẑ_q (在开发集 prompt 上方差为 1)
  所以 t_k(q) 高的 prompt 给 PC k 坐标高的模型加分。图上强档在 PC1 负端, 所以 t1 高 → 强弱差距缩小, t1 低 → 强档更占优。
折外版本: 每个 cv 折只用 train 票训练同设置模型、自己做 PCA (符号按和最终模型的模型坐标相关性对齐), 再给该折 eval 票的 prompt 打 t1,
  用来检查 "t1 两端胜率不同" 不是训练集内的过拟合。只用开发集; 测试集不参与。

用法: python -I prompt_axes.py <v2 根目录> <原始 140k 目录>
产出: data/r2_prompt_axes.parquet, reports/r2_prompt_axes.md
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

root, raw = Path(sys.argv[1]), Path(sys.argv[2])
MIN_VOTES, N_END, N_SHOW = 300, 200, 10
argv = sys.argv
src = (Path(__file__).parent / "train_r2_mf.py").read_text().split("\nlines = [")[0].replace("optax.adam(1e-3)", "optax.adam(LR)")
ns = {"__name__": "mf"}
sys.argv = [argv[0], str(root)]
exec(compile(src, "train_r2_mf.py", "exec"), ns)
sys.argv = argv
ns["K"], ns["LR"] = 128, 1e-3
fl, E, gpos, names = ns["fl"], np.asarray(ns["E"]), ns["gpos"], ns["models"]


def axes(P, groups, train_votes):
    """返回 (模型 PC 坐标 DataFrame, 函数: group_id 数组 → t1,t2,t3)。PCA 只用训练票数 ≥ MIN_VOTES 的模型。"""
    g = np.asarray(P["v"]) * np.asarray(P["w2"])
    Z = E[gpos.loc[groups].values] @ np.asarray(P["W"], dtype=np.float64) + np.asarray(P["b"])
    zbar = Z.mean(0)
    ev, evec = np.linalg.eigh(np.cov(Z, rowvar=False))
    keep = ev > ev.max() * 1e-8
    S_half = (evec[:, keep] * np.sqrt(ev[keep])) @ evec[:, keep].T
    S_mhalf = (evec[:, keep] / np.sqrt(ev[keep])) @ evec[:, keep].T
    cnt = pd.concat([train_votes.hi_model, train_votes.lo_model]).value_counts()
    ok = np.array([cnt.get(m, 0) >= MIN_VOTES for m in names])
    gw = (g[ok] - g[ok].mean(0)) @ S_half
    _, S, Vt = np.linalg.svd(gw, full_matrices=False)
    coords = pd.DataFrame(gw @ Vt[:3].T, columns=["PC1", "PC2", "PC3"], index=np.array(names)[ok])
    proj = lambda gr: ((E[gpos.loc[gr].values] @ np.asarray(P["W"], dtype=np.float64) + np.asarray(P["b"]) - zbar)
                       @ S_mhalf @ Vt[:3].T)
    return coords, proj, S ** 2 / (S ** 2).sum()


# ---- 最终模型 (整个开发集 90%) ----
P = dict(np.load(root / "data/r2_final_params.npz"))
dev = fl[(fl.context == "final") & (fl.role == "train")]
groups = dev.group_id.unique()
coords, proj, ratio = axes(P, groups, dev)
ref = pd.read_csv(root / "reports/r2_model_vectors_pca.csv").set_index("model")
for k in ["PC1", "PC2"]:   # SVD 符号与画图脚本一致 (同一输入); 不一致就翻转
    if np.corrcoef(coords.loc[ref.index, k], ref[k])[0, 1] < 0:
        coords[k] *= -1
        flip = {"PC1": 0, "PC2": 1}[k]
        proj0 = proj
        proj = (lambda f, j: (lambda gr: f(gr) * np.where(np.arange(3) == j, -1, 1)))(proj0, flip)
assert np.allclose(coords.loc[ref.index, ["PC1", "PC2"]].values, ref[["PC1", "PC2"]].values, atol=1e-6)
T = proj(groups)
pa = pd.DataFrame({"group_id": groups, "t1": T[:, 0], "t2": T[:, 1], "t3": T[:, 2]})

# ---- prompt 属性: 原文、长度、语言、代码; Arena 自带的 LLM 标签 (category_tag) ----
lab = pd.read_parquet(root / "data/arena_labeled.parquet",
                      columns=["id", "group_id", "prompt", "language", "is_code", "prompt_len", "split"])
lab = lab[lab.split != "test"].drop_duplicates("group_id")
tags = []
for f in sorted(glob.glob(str(raw / "*.parquet"))):
    t = pq.read_table(f, columns=["id", "category_tag"]).to_pandas()
    tags.append(t[t.id.isin(lab.id)])
tags = pd.concat(tags).set_index("id").category_tag
crit = ["complexity", "creativity", "domain_knowledge", "problem_solving", "real_world", "specificity", "technical_accuracy"]
lab = lab.assign(
    math=lab.id.map(lambda i: bool(tags[i]["math_v0.1"]["math"])),
    creative=lab.id.map(lambda i: bool(tags[i]["creative_writing_v0.1"]["creative_writing"])),
    instr=lab.id.map(lambda i: bool(tags[i]["if_v0.1"]["if"])),
    n_crit=lab.id.map(lambda i: int(sum(bool(tags[i]["criteria_v0.1"][c]) for c in crit))))
pa = pa.merge(lab.drop(columns="split"), on="group_id", how="left")
assert pa.prompt.notna().all()


# ---- 折外 t1: 每折只用 train 票训练, 给 eval 票的 prompt 打分 ----
oof = []
for k in range(5):
    d = fl[fl.context == f"cv{k}"]
    tr, ev = d[d.role == "train"], d[d.role == "eval"]
    Pk, _, _ = ns["train"](ns["arrays"](tr), 1e-5, epochs=14)
    Pk = {a: np.asarray(b) for a, b in Pk.items()}
    ck, projk, _ = axes(Pk, tr.group_id.unique(), tr)
    common = ck.index.intersection(coords.index)
    r = [np.corrcoef(ck.loc[common, c], coords.loc[common, "PC1"])[0, 1] for c in ["PC1", "PC2", "PC3"]]
    sign = np.sign(r[0])
    eg = ev.group_id.unique()
    oof.append(pd.DataFrame({"group_id": eg, "t1_oof": projk(eg)[:, 0] * sign, "fold_r": abs(r[0])}))
    print(f"cv{k} 折内 PC1 与最终 PC1 的模型坐标相关", np.round(r, 3), flush=True)
oof = pd.concat(oof)
pa = pa.merge(oof, on="group_id", how="left")
pa.to_parquet(root / "data/r2_prompt_axes.parquet", index=False)
coords.to_csv(root / "reports/r2_model_vectors_pca3.csv")
print(ratio[:3].round(3), len(pa))
