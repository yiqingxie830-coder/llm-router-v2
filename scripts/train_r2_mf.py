"""train_r2_mf.py — 阶段 2 路由器 R2: 矩阵分解 (MF), 参考 RouteLLM 的 MF 路由器

和 R1 的区别
------------
R1 是线性的: prompt 对所有模型对加同一个"难度偏移"。MF 让 prompt 和具体模型交互:
  z      = W · e + b                     (prompt 向量 1024 维 → k 维)
  f(m,q) = w2 · (v_m ⊙ z) + u_m           (模型 m 在这个 prompt 上的"适合度")
  s      = f(高分方) − f(低分方) + a · 评分差/100
输出用有序逻辑 (输 < 平 < 赢, 两个切点 c1 < c2):
  P(输) = σ(c1 − s),  P(输或平) = σ(c2 − s),  P(赢) = 1 − σ(c2 − s)
所以 P(赢) − P(输) 直接可得, 和三分类 R1、带平局基线同口径。

训练: 加权负对数似然 (权重 1/n_votes) + L2 (λ), Adam, 批大小 1024。
选超参 (只在训练折内): 每个外层折的 train 票按 prompt 组切 80/20, 在 80% 上训练, 20% 上做早停并选 λ;
再用选中的 λ 和轮数在整份 train 上重训, 预测该折 eval 票。eval 票、测试集都不参与。

用法: python -I train_r2_mf.py <v2 根目录> [--final] [--shuffle]
产出: preds/r2_win.parquet (p = P(赢)), preds/r2_eff.parquet (p = (1 + P赢 − P输)/2), reports/r2_selection.md
"""
import hashlib
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

root = Path(sys.argv[1])
use_final = "--final" in sys.argv
K, BATCH, MAX_EP, PATIENCE = 64, 1024, 30, 3
LAMBDAS = [0.0, 1e-6, 1e-5, 1e-4]  # 第一轮 [1e-5, 1e-4, 1e-3] 选中网格边缘 1e-5, 见 r2_selection_round1.md
SEED = 2026

fl = pd.read_parquet(root / "data/fold_labels.parquet",
                     columns=["context", "id", "role", "group_id", "hi_model", "lo_model", "rating_gap",
                              "y_hi_route", "is_tie", "n_votes"])
fl["cls"] = np.where(fl.y_hi_route == 1, 2, np.where(fl.is_tie, 1, 0)).astype(np.int32)
emb = np.load(root / "data/prompt_emb_e5large.npy").astype(np.float32)
SHUFFLE = "--shuffle" in sys.argv   # 对照: 把 prompt 向量在 prompt 之间随机打乱, 检查提升是否真来自 prompt 内容
if SHUFFLE:
    emb = emb[np.random.default_rng(7).permutation(len(emb))]
gpos = pd.Series(np.arange(len(emb)), index=pd.read_parquet(root / "data/prompt_emb_index.parquet").group_id)
models = sorted(set(fl.hi_model) | set(fl.lo_model))
mpos = {m: i for i, m in enumerate(models)}
M = len(models)


def arrays(d):
    return (jnp.asarray(gpos.loc[d.group_id].values), jnp.asarray(d.hi_model.map(mpos).values),
            jnp.asarray(d.lo_model.map(mpos).values), jnp.asarray(d.rating_gap.values / 100, dtype=jnp.float32),
            jnp.asarray(d.cls.values), jnp.asarray(1 / d.n_votes.values, dtype=jnp.float32))


E = jnp.asarray(emb)


def init(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return {"W": jax.random.normal(k1, (1024, K)) * 0.02, "b": jnp.zeros(K),
            "v": jax.random.normal(k2, (M, K)) * 0.1, "u": jnp.zeros(M), "w2": jax.random.normal(k3, (K,)) * 0.1,
            "a": jnp.array(0.5), "c1": jnp.array(-0.3), "dc": jnp.array(0.6)}


def score(p, gi, hi, lo, gap):
    z = E[gi] @ p["W"] + p["b"]
    f = lambda m: (p["v"][m] * z) @ p["w2"] + p["u"][m]
    return f(hi) - f(lo) + p["a"] * gap


def probs(p, gi, hi, lo, gap):
    s = score(p, gi, hi, lo, gap)
    c1, c2 = p["c1"], p["c1"] + jax.nn.softplus(p["dc"])
    pl = jax.nn.sigmoid(c1 - s)
    ple = jax.nn.sigmoid(c2 - s)
    return jnp.stack([pl, ple - pl, 1 - ple], axis=1)


def nll(p, batch):
    gi, hi, lo, gap, y, w = batch
    pr = jnp.clip(probs(p, gi, hi, lo, gap), 1e-7, 1)
    return -(w * jnp.log(pr[jnp.arange(len(y)), y])).sum() / w.sum()


def make_step(lam, opt):
    def loss(p, batch):
        reg = lam * (jnp.sum(p["W"] ** 2) + jnp.sum(p["v"] ** 2) + jnp.sum(p["w2"] ** 2))
        return nll(p, batch) + reg

    @jax.jit
    def step(p, s, batch):
        g = jax.grad(loss)(p, batch)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s
    return step


def train(data, lam, epochs=None, val=None):
    """epochs=None 时用 val 早停, 返回 (参数, 最佳轮数, 最佳验证损失)。"""
    opt = optax.adam(1e-3)
    p = init(jax.random.PRNGKey(SEED))
    s = opt.init(p)
    step = make_step(lam, opt)
    n = len(data[0])
    rng = np.random.default_rng(SEED)
    best, best_ep, best_p, bad = np.inf, 0, p, 0
    for ep in range(1, (epochs or MAX_EP) + 1):
        order = rng.permutation(n)
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            p, s = step(p, s, tuple(x[idx] for x in data))
        if val is not None:
            v = float(nll(p, val))
            if v < best - 1e-5:
                best, best_ep, best_p, bad = v, ep, p, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    return (best_p, best_ep, best) if val is not None else (p, epochs, None)


lines = ["# R2 (MF) 超参选择 (每个外层折只用自己的 train 票; 按 prompt 组切 80/20 做内层验证)", "",
         f"k = {K}, 批大小 {BATCH}, Adam 1e-3, 早停耐心 {PATIENCE} 轮, λ ∈ {LAMBDAS}", "",
         "| 情境 | λ | 内层验证对数损失 | 早停轮数 |", "|---|---|---|---|"]
ctxs = ["final"] if use_final else [f"cv{k}" for k in range(5)]
win, eff = [], []
for ctx in ctxs:
    d = fl[fl.context == ctx]
    tr, ev = d[d.role == "train"], d[d.role == "eval"]
    h = tr.group_id.map(lambda g: int(hashlib.md5(f"2026{g}".encode()).hexdigest(), 16) % 5 == 0).values
    a_data, b_data = arrays(tr[~h]), arrays(tr[h])
    res = {}
    for lam in LAMBDAS:
        t = time.time()
        _, ep, v = train(a_data, lam, val=b_data)
        res[lam] = (v, ep)
        lines.append(f"| {ctx} | {lam} | {v:.4f} | {ep} |")
        print(ctx, lam, round(v, 4), ep, f"{time.time() - t:.0f}s", flush=True)
    lam = min(res, key=lambda k: res[k][0])
    p, _, _ = train(arrays(tr), lam, epochs=res[lam][1])
    gi, hi, lo, gap, _, _ = arrays(ev)
    pr = np.asarray(probs(p, gi, hi, lo, gap))
    win.append(pd.DataFrame({"context": ctx, "id": ev.id.values, "p": pr[:, 2]}))
    eff.append(pd.DataFrame({"context": ctx, "id": ev.id.values, "p": (1 + pr[:, 2] - pr[:, 0]) / 2}))
    lines.append(f"| {ctx} 选中 | **{lam}** | | {res[lam][1]} |")
sfx = ("_final" if use_final else "") + ("_shuffled" if SHUFFLE else "")
pd.concat(win).to_parquet(root / f"preds/r2_win{sfx}.parquet", index=False)
pd.concat(eff).to_parquet(root / f"preds/r2_eff{sfx}.parquet", index=False)
(root / f"reports/r2_selection{sfx}.md").write_text("\n".join(lines))
print("\n".join(lines))
