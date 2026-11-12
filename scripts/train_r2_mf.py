"""train_r2_mf.py — Stage 2 router R2: matrix factorization (MF), after RouteLLM's MF router

How it differs from R1
----------------------
R1 is linear: a prompt adds the same "difficulty offset" to every model pair. MF lets the prompt interact with each model:
  z      = W · e + b                     (prompt vector, 1024 dims -> k dims)
  f(m,q) = w2 · (v_m ⊙ z) + u_m           (how well model m suits this prompt)
  s      = f(higher-rated) − f(lower-rated) + a · gap/100
Output is ordinal logistic (loss < tie < win, two cut points c1 < c2):
  P(loss) = σ(c1 − s),  P(loss or tie) = σ(c2 − s),  P(win) = 1 − σ(c2 − s)
so P(win) − P(loss) comes out directly, on the same target as the 3-class R1 and the tie-aware baseline.

Training: weighted negative log likelihood (weight 1/n_votes) + L2 (λ), Adam, batch size 1024.
Hyperparameters (inside training folds only): each outer fold's train votes are split 80/20 by prompt group; train on
80%, early-stop and choose λ on 20%; then retrain on the whole train part with the chosen λ and epoch count and predict
that fold's eval votes. Neither eval votes nor the test set take part.

Usage: python -I train_r2_mf.py <repo root> [--final] [--shuffle]
Writes: preds/r2_win.parquet (p = P(win)), preds/r2_eff.parquet (p = (1 + P(win) − P(loss))/2), reports/r2_selection.md
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
LAMBDAS = [0.0, 1e-6, 1e-5, 1e-4]  # round 1 [1e-5, 1e-4, 1e-3] picked the grid edge 1e-5, see r2_selection_round1.md
SEED = 2026

fl = pd.read_parquet(root / "data/fold_labels.parquet",
                     columns=["context", "id", "role", "group_id", "hi_model", "lo_model", "rating_gap",
                              "y_hi_route", "is_tie", "n_votes"])
fl["cls"] = np.where(fl.y_hi_route == 1, 2, np.where(fl.is_tie, 1, 0)).astype(np.int32)
emb = np.load(root / "data/prompt_emb_e5large.npy").astype(np.float32)
SHUFFLE = "--shuffle" in sys.argv   # control: shuffle prompt vectors across prompts, to check the gain really comes from prompt content
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
    """With epochs=None, early-stop on val and return (params, best epoch, best validation loss)."""
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


lines = ["# R2 (MF) hyperparameter choice (each outer fold uses only its own train votes; 80/20 split by prompt group for inner validation)", "",
         f"k = {K}, batch size {BATCH}, Adam 1e-3, early-stopping patience {PATIENCE} epochs, λ ∈ {LAMBDAS}", "",
         "| Context | λ | Inner validation log loss | Early-stop epoch |", "|---|---|---|---|"]
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
    lines.append(f"| {ctx} chosen | **{lam}** | | {res[lam][1]} |")
sfx = ("_final" if use_final else "") + ("_shuffled" if SHUFFLE else "")
pd.concat(win).to_parquet(root / f"preds/r2_win{sfx}.parquet", index=False)
pd.concat(eff).to_parquet(root / f"preds/r2_eff{sfx}.parquet", index=False)
(root / f"reports/r2_selection{sfx}.md").write_text("\n".join(lines))
print("\n".join(lines))
