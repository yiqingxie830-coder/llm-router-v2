"""r3_train.py — 方案二 (R3 语义标签) 和对照的训练, 按 reports/r3_prereg.md 执行

路由器 (都只用口径 ② 的票训练, 三分类 赢/平/输):
  r3lr   : task_type 独热 (8) + 6 个分数独热 (6×4) + 评分差 + 模型身份 → 逻辑回归, C 网格
  r3mf   : R2 的 MF 结构, prompt 输入换成标签特征 38 维 (32 维独热 + 6 个分数 /3 的数值版), d × λ 网格
  r2sw   : R2-②, R2 原样 (1024 维 e5 向量), 只用 ② 的票
  r2r3   : R2 + R3, 输入 = e5 向量 和 38 维标签特征 拼接 (1062 维), 只用 ② 的票
选参: 每个外层折 cv k 只用自己的 train 票, 按 prompt 组哈希切 80/20 (和方案一同一个哈希), 20% 上的加权对数损失最小者入选;
再用入选设置 (MF 还有早停轮数) 在整份 train 上重训, 预测该折 eval 票。eval 票和测试集不参与任何选择。
方案二代表 "r3": 每折在 r3lr 和 r3mf 的入选者里取内层验证损失更低的那个。
标签缺失 (Haiku 拒答 11 条): 用该折 train 票所在 prompt 的众数填补。
权重: 1 / n_votes_sw (该 prompt 在 ② 里的票数; 只用 ② 训练时与方案一 "1/n_votes" 同义)。

用法: python -I r3_train.py <v2 根目录> <路由器> [--final]
      python -I r3_train.py <v2 根目录> combine [--final]     (生成 r3 = 每折 r3lr / r3mf 中内层损失低者)
产出: preds/r3/<路由器>[_final].parquet (context, id, p_win, p), preds/r3/<路由器>[_final].json (网格、内层损失、入选)
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
which = sys.argv[2]
use_final = "--final" in sys.argv
ctxs = ["final"] if use_final else [f"cv{k}" for k in range(5)]
sfx = "_final" if use_final else ""
out_dir = root / "preds/r3"
out_dir.mkdir(parents=True, exist_ok=True)

TASKS = ["code", "math", "writing", "factual_qa", "advice", "translation", "chitchat", "other"]
SCORES = ["has_single_answer", "is_open_ended", "is_creative", "has_format_constraint", "difficulty", "expected_length"]
GRIDS = {
    "r3lr": [{"C": c} for c in [0.01, 0.03, 0.1, 0.3, 1.0]],
    "r3mf": [{"d": d, "lam": l} for d in [4, 8, 16] for l in [1e-5, 1e-4, 1e-3]],
    "r2sw": [{"d": d, "lam": l} for d in [16, 64, 128] for l in [1e-6, 1e-5, 1e-4]],
    "r2r3": [{"d": d, "lam": l} for d in [16, 64, 128] for l in [1e-6, 1e-5, 1e-4]],
}
import os
BATCH, PATIENCE, LR, SEED = 1024, 3, 1e-3, 2026
MAX_EP = int(os.environ.get("R3_MAX_EP", 30))   # 预登记沿用 R2 的 30; 敏感性检查用 R3_MAX_EP=100
sfx += "" if MAX_EP == 30 else f"_ep{MAX_EP}"

fl = pd.read_parquet(root / "data/fold_labels.parquet",
                     columns=["context", "id", "role", "group_id", "hi_model", "lo_model", "rating_gap",
                              "y_hi_route", "is_tie", "n_votes_sw", "is_strong_vs_weak"])
fl = fl[fl.is_strong_vs_weak & fl.context.isin(ctxs)].reset_index(drop=True)
fl["cls"] = np.where(fl.y_hi_route == 1, 2, np.where(fl.is_tie, 1, 0)).astype(np.int32)
fl["w"] = 1.0 / fl.n_votes_sw.values
models = sorted(set(fl.hi_model) | set(fl.lo_model))
mpos = {m: i for i, m in enumerate(models)}

lab = []
for line in open(root / "r3/r3_labels.jsonl", encoding="utf-8"):
    r = json.loads(line)
    lab.append({"group_id": r["group_id"], **(r.get("labels") or {})})
lab = pd.DataFrame(lab).set_index("group_id")
gids = lab.index.values
gpos = pd.Series(np.arange(len(gids)), index=gids)
fl["g"] = gpos.loc[fl.group_id].values


def label_matrix(train_gids):
    """按训练折众数填补缺失, 返回 (独热 32 维, 数值 6 维) 两个矩阵, 行与 gids 对齐。"""
    L = lab.copy()
    tr = L.loc[np.unique(train_gids)]
    n_miss = int(L.task_type.isna().sum())
    for c in ["task_type"] + SCORES:
        L[c] = L[c].fillna(tr[c].mode().iloc[0])
    oh = [(L.task_type.to_numpy(dtype=object)[:, None] == np.array(TASKS)[None]).astype(np.float32)]
    for c in SCORES:
        oh.append((L[c].to_numpy(dtype=float).astype(int)[:, None] == np.arange(4)[None]).astype(np.float32))
    num = L[SCORES].to_numpy(dtype=np.float32) / 3
    return np.hstack(oh), num, n_miss


def inner_split(tr):
    return tr.group_id.map(lambda g: int(hashlib.md5(f"2026{g}".encode()).hexdigest(), 16) % 5 == 0).values


# ---------------- 逻辑回归 (R3-LR) ----------------
def lr_design(d, OH):
    from scipy import sparse
    n = len(d)
    r = np.arange(n)
    ident = sparse.csr_matrix((np.r_[np.ones(n), -np.ones(n)],
                               (np.r_[r, r], np.r_[d.hi_model.map(mpos).values, d.lo_model.map(mpos).values])),
                              shape=(n, len(models)))
    return sparse.hstack([sparse.csr_matrix(d.rating_gap.values[:, None] / 100), ident,
                          sparse.csr_matrix(OH[d.g.values])]).tocsr()


def lr_fit_proba(tr, ev, OH, C):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, max_iter=3000).fit(lr_design(tr, OH), tr.cls.values, sample_weight=tr.w.values)
    return clf.predict_proba(lr_design(ev, OH))[:, np.searchsorted(clf.classes_, [0, 1, 2])]


def wnll(y, pr, w):
    pr = np.clip(pr, 1e-7, 1)
    return float(-(w * np.log(pr[np.arange(len(y)), y])).sum() / w.sum())


# ---------------- MF (R2 结构, 输入矩阵可换) ----------------
def mf_tools(E_np, K, lam):
    import jax
    import jax.numpy as jnp
    import optax
    E = jnp.asarray(E_np)
    F, M = E_np.shape[1], len(models)

    def init(key):
        k1, k2, k3 = jax.random.split(key, 3)
        return {"W": jax.random.normal(k1, (F, K)) * 0.02, "b": jnp.zeros(K),
                "v": jax.random.normal(k2, (M, K)) * 0.1, "u": jnp.zeros(M), "w2": jax.random.normal(k3, (K,)) * 0.1,
                "a": jnp.array(0.5), "c1": jnp.array(-0.3), "dc": jnp.array(0.6)}

    def probs(p, gi, hi, lo, gap):
        z = E[gi] @ p["W"] + p["b"]
        f = lambda m: (p["v"][m] * z) @ p["w2"] + p["u"][m]
        s = f(hi) - f(lo) + p["a"] * gap
        c1, c2 = p["c1"], p["c1"] + jax.nn.softplus(p["dc"])
        pl, ple = jax.nn.sigmoid(c1 - s), jax.nn.sigmoid(c2 - s)
        return jnp.stack([pl, ple - pl, 1 - ple], axis=1)

    def nll(p, b):
        gi, hi, lo, gap, y, w = b
        pr = jnp.clip(probs(p, gi, hi, lo, gap), 1e-7, 1)
        return -(w * jnp.log(pr[jnp.arange(len(y)), y])).sum() / w.sum()

    opt = optax.adam(LR)

    def loss(p, b):
        return nll(p, b) + lam * (jnp.sum(p["W"] ** 2) + jnp.sum(p["v"] ** 2) + jnp.sum(p["w2"] ** 2))

    @jax.jit
    def step(p, s, b):
        g = jax.grad(loss)(p, b)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s

    nll_j = jax.jit(nll)
    probs_j = jax.jit(probs)

    def arrays(d):
        return (jnp.asarray(d.g.values), jnp.asarray(d.hi_model.map(mpos).values),
                jnp.asarray(d.lo_model.map(mpos).values), jnp.asarray(d.rating_gap.values / 100, dtype=jnp.float32),
                jnp.asarray(d.cls.values), jnp.asarray(d.w.values, dtype=jnp.float32))

    def train(data, epochs=None, val=None):
        p = init(jax.random.PRNGKey(SEED))
        s = opt.init(p)
        n = len(data[0])
        rng = np.random.default_rng(SEED)
        best, best_ep, best_p, bad = np.inf, 0, p, 0
        for ep in range(1, (epochs or MAX_EP) + 1):
            order = rng.permutation(n)
            for i in range(0, n, BATCH):
                idx = order[i:i + BATCH]
                p, s = step(p, s, tuple(x[idx] for x in data))
            if val is not None:
                v = float(nll_j(p, val))
                if v < best - 1e-5:
                    best, best_ep, best_p, bad = v, ep, p, 0
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
        return (best_p, best_ep, best) if val is not None else (p, epochs, None)

    def predict(p, d):
        gi, hi, lo, gap, _, _ = arrays(d)
        return np.asarray(probs_j(p, gi, hi, lo, gap))

    return arrays, train, predict


def emb_rows():
    emb = np.load(str(root / "data/prompt_emb_e5large.npy"))
    idx = pd.Series(np.arange(len(emb)), index=pd.read_parquet(root / "data/prompt_emb_index.parquet").group_id)
    return np.asarray(emb[idx.loc[gids].values], dtype=np.float32)


def run(which):
    E_emb = emb_rows() if which in ("r2sw", "r2r3") else None
    rows, info = [], {}
    for ctx in ctxs:
        d = fl[fl.context == ctx]
        tr, ev = d[d.role == "train"], d[d.role == "eval"]
        OH, NUM, n_miss = label_matrix(tr.group_id.values)
        h = inner_split(tr)
        a, b = tr[~h], tr[h]
        res = []
        for cfg in GRIDS[which]:
            t = time.time()
            if which == "r3lr":
                v = wnll(b.cls.values, lr_fit_proba(a, b, OH, cfg["C"]), b.w.values)
                res.append({**cfg, "val_loss": v, "sec": round(time.time() - t, 1)})
            else:
                E = {"r3mf": np.hstack([OH, NUM]), "r2sw": E_emb,
                     "r2r3": None if E_emb is None else np.hstack([E_emb, OH, NUM])}[which]
                arrays, train, _ = mf_tools(E, cfg["d"], cfg["lam"])
                _, ep, v = train(arrays(a), val=arrays(b))
                res.append({**cfg, "val_loss": v, "epochs": ep, "sec": round(time.time() - t, 1)})
            print(which, ctx, res[-1], flush=True)
        best = min(res, key=lambda r: r["val_loss"])
        if which == "r3lr":
            pr = lr_fit_proba(tr, ev, OH, best["C"])
        else:
            E = {"r3mf": np.hstack([OH, NUM]), "r2sw": E_emb, "r2r3": np.hstack([E_emb, OH, NUM]) if E_emb is not None else None}[which]
            arrays, train, predict = mf_tools(E, best["d"], best["lam"])
            p, _, _ = train(arrays(tr), epochs=best["epochs"])
            pr = predict(p, ev)
        rows.append(pd.DataFrame({"context": ctx, "id": ev.id.values, "p_win": pr[:, 2],
                                  "p": (1 + pr[:, 2] - pr[:, 0]) / 2,
                                  "p_tie": pr[:, 1], "p_lose": pr[:, 0]}))
        info[ctx] = {"grid": res, "pick": best, "n_train": int(len(tr)), "n_eval": int(len(ev)),
                     "n_label_missing_total": n_miss}
    pd.concat(rows).to_parquet(out_dir / f"{which}{sfx}.parquet", index=False)
    (out_dir / f"{which}{sfx}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))


def combine():
    a = json.loads((out_dir / f"r3lr{sfx}.json").read_text())
    b = json.loads((out_dir / f"r3mf{sfx}.json").read_text())
    pa, pb = pd.read_parquet(out_dir / f"r3lr{sfx}.parquet"), pd.read_parquet(out_dir / f"r3mf{sfx}.parquet")
    rows, pick = [], {}
    for ctx in ctxs:
        use_lr = a[ctx]["pick"]["val_loss"] <= b[ctx]["pick"]["val_loss"]
        pick[ctx] = {"chosen": "r3lr" if use_lr else "r3mf", "r3lr": a[ctx]["pick"], "r3mf": b[ctx]["pick"]}
        src = pa if use_lr else pb
        rows.append(src[src.context == ctx])
    pd.concat(rows).to_parquet(out_dir / f"r3{sfx}.parquet", index=False)
    (out_dir / f"r3{sfx}.json").write_text(json.dumps(pick, ensure_ascii=False, indent=1))
    print(json.dumps(pick, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    combine() if which == "combine" else run(which)
