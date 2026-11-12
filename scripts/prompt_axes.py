"""prompt_axes.py — project dev-set prompts onto the PC1 / PC2 axes of the R2 model-vector plot, to name the two axes and test whether PC1 is "difficulty"

Axis definition (the same PCA as plot_r2_vectors.py):
  prompt-dependent fit of model m = g_m · (z_q − z̄) = g_w,m · ẑ_q,  g_w = Σ^(1/2) g,  ẑ_q = Σ^(−1/2) (z_q − z̄)
  PC k direction V_k (whitened space), model coordinate = g_w,m · V_k, prompt coordinate t_k(q) = V_k · ẑ_q (variance 1 over dev prompts)
  So a prompt with high t_k(q) favours models with a high PC k coordinate. In the plot the strong tier sits at the
  negative end of PC1, so high t1 -> the strong-weak gap narrows, low t1 -> the strong tier has more of an edge.
Out-of-fold version: each cv fold trains a model with the same settings on its train votes only, runs its own PCA
  (sign aligned by correlation of model coordinates with the final model), then scores t1 for the prompts of that
  fold's eval votes; this checks that "different win rates at the two ends of t1" is not in-sample overfitting.
  Dev set only; the test set is not used.

Usage: python -I prompt_axes.py <repo root> <raw 140k directory>
Writes: data/r2_prompt_axes.parquet, reports/r2_model_vectors_pca3.csv
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
    """Return (DataFrame of model PC coordinates, function: group_id array -> t1,t2,t3, variance ratios). PCA uses only models with ≥ MIN_VOTES training votes."""
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


# ---- final model (the full 90% dev set) ----
P = dict(np.load(root / "data/r2_final_params.npz"))
dev = fl[(fl.context == "final") & (fl.role == "train")]
groups = dev.group_id.unique()
coords, proj, ratio = axes(P, groups, dev)
ref = pd.read_csv(root / "reports/r2_model_vectors_pca.csv").set_index("model")
for k in ["PC1", "PC2"]:   # SVD signs should match the plotting script (same input); flip if not
    if np.corrcoef(coords.loc[ref.index, k], ref[k])[0, 1] < 0:
        coords[k] *= -1
        flip = {"PC1": 0, "PC2": 1}[k]
        proj0 = proj
        proj = (lambda f, j: (lambda gr: f(gr) * np.where(np.arange(3) == j, -1, 1)))(proj0, flip)
assert np.allclose(coords.loc[ref.index, ["PC1", "PC2"]].values, ref[["PC1", "PC2"]].values, atol=1e-6)
T = proj(groups)
pa = pd.DataFrame({"group_id": groups, "t1": T[:, 0], "t2": T[:, 1], "t3": T[:, 2]})

# ---- prompt attributes: text, length, language, code; Arena's own LLM tags (category_tag) ----
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


# ---- out-of-fold t1: each fold trains on its train votes only and scores its eval prompts ----
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
    print(f"cv{k} correlation of fold PC1..3 model coordinates with final PC1", np.round(r, 3), flush=True)
oof = pd.concat(oof)
pa = pa.merge(oof, on="group_id", how="left")
pa.to_parquet(root / "data/r2_prompt_axes.parquet", index=False)
coords.to_csv(root / "reports/r2_model_vectors_pca3.csv")
print(ratio[:3].round(3), len(pa))
