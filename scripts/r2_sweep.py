"""r2_sweep.py — R2 (MF) hyperparameter sweep, nested selection, sensitivity table, and export of the model-vector table

Sweep: d (prompt projection size) × learning rate × λ (explicit L2); each combination, on each of the 5 outer folds:
  inner: split the outer train votes 80/20 by prompt group (same hash as train_r2_mf.py), train on 80%, early-stop
         on 20% (patience 3 epochs, at most 30), record the best epoch and its validation log loss;
  outer: retrain on the whole train part with that combination and epoch count, predict that fold's eval votes
         (used only for the sensitivity table, never for selection).
Nested selection: each fold takes the combination with the lowest inner validation loss. Neither eval votes nor the test set take part.

Usage: python -I r2_sweep.py <repo root> run <d> [<lr> <λ>]   (can run a single combination; with tight memory run one process per combination)
       python -I r2_sweep.py <repo root> summarize
       python -I r2_sweep.py <repo root> export <d> <lr> <λ> <epochs>   (train once on the full 90% dev set and export the model-vector table; does not predict the test set)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
mode = sys.argv[2]
DS, LRS, LAMS = [16, 64, 128], [1e-3, 3e-3], [1e-6, 1e-5, 1e-4]
out_dir = root / "preds/r2_sweep"
out_dir.mkdir(parents=True, exist_ok=True)


def load_mf():
    src = (Path(__file__).parent / "train_r2_mf.py").read_text().split("\nlines = [")[0]
    src = src.replace("optax.adam(1e-3)", "optax.adam(LR)")
    ns = {"__name__": "mf"}
    argv = sys.argv
    sys.argv = [argv[0], str(root)]
    exec(compile(src, "train_r2_mf.py", "exec"), ns)
    sys.argv = argv
    return ns


def tag(d, lr, lam):
    return f"d{d}_lr{lr:g}_lam{lam:g}"


def run(d, only=None):
    import hashlib
    ns = load_mf()
    ns["K"] = d
    fl, arrays, train, probs = ns["fl"], ns["arrays"], ns["train"], ns["probs"]
    for lr in LRS:
        for lam in LAMS:
            ns["LR"] = lr
            if (out_dir / f"{tag(d, lr, lam)}.json").exists() or (only and only != (lr, lam)):   # resume / run a single combination
                continue
            rows, info = [], {}
            for k in range(5):
                t = time.time()
                dd = fl[fl.context == f"cv{k}"]
                tr, ev = dd[dd.role == "train"], dd[dd.role == "eval"]
                h = tr.group_id.map(lambda g: int(hashlib.md5(f"2026{g}".encode()).hexdigest(), 16) % 5 == 0).values
                _, ep, v = train(arrays(tr[~h]), lam, val=arrays(tr[h]))
                t_inner = time.time() - t
                p, _, _ = train(arrays(tr), lam, epochs=ep)
                gi, hi, lo, gap, _, _ = arrays(ev)
                pr = np.asarray(probs(p, gi, hi, lo, gap))
                rows.append(pd.DataFrame({"context": f"cv{k}", "id": ev.id.values, "p_win": pr[:, 2],
                                          "p": (1 + pr[:, 2] - pr[:, 0]) / 2}))
                info[f"cv{k}"] = {"val_loss": v, "epochs": ep, "sec_inner": round(t_inner, 1),
                                  "sec_total": round(time.time() - t, 1)}
                print(tag(d, lr, lam), k, round(v, 4), ep, info[f"cv{k}"]["sec_total"], flush=True)
            pd.concat(rows).to_parquet(out_dir / f"{tag(d, lr, lam)}.parquet", index=False)
            (out_dir / f"{tag(d, lr, lam)}.json").write_text(json.dumps(info))


def summarize():
    src = (Path(__file__).parent / "r1_csweep.py").read_text().split("\ndef summarize")[0]
    ns = {"__name__": "x"}
    argv = sys.argv
    sys.argv = [argv[0], str(root), "x"]
    exec(compile(src, "r1_csweep.py", "exec"), ns)
    sys.argv = argv
    apgr_frac, apgr_cost, FLOOR = ns["apgr_frac"], ns["apgr_cost"], ns["FLOOR"]
    fl = pd.read_parquet(root / "data/fold_labels.parquet",
                         columns=["context", "id", "role", "q_hi", "is_strong_vs_weak", "hi_model", "model_a"])
    ev = fl[(fl.role == "eval") & fl.context.str.startswith("cv")]
    vc = pd.read_parquet(root / "data/vote_costs.parquet")
    ev = ev.merge(vc, on="id")
    a = ev.hi_model.values == ev.model_a.values
    ev = ev.assign(cost_hi=np.where(a, ev.cost_a, ev.cost_b), cost_lo=np.where(a, ev.cost_b, ev.cost_a),
                   typ_hi=np.where(a, ev.typ_a, ev.typ_b), typ_lo=np.where(a, ev.typ_b, ev.typ_a))
    sw = ev[ev.is_strong_vs_weak.values]
    main = ev[ev.is_strong_vs_weak.values & (ev.typ_hi > ev.typ_lo).values]
    combos = [(d, lr, lam) for d in DS + [256] for lr in LRS for lam in LAMS
              if (out_dir / f"{tag(d, lr, lam)}.json").exists()]   # d=256 was only added for lr=1e-3 (after d=128 landed on the grid edge)
    info = {c: json.loads((out_dir / f"{tag(*c)}.json").read_text()) for c in combos}
    preds = {c: pd.read_parquet(out_dir / f"{tag(*c)}.parquet") for c in combos}
    pick = {k: min(combos, key=lambda c: info[c][f"cv{k}"]["val_loss"]) for k in range(5)}
    nested = pd.concat([preds[pick[k]][preds[pick[k]].context == f"cv{k}"] for k in range(5)])
    nested.to_parquet(root / "preds/r2_nested.parquet", index=False)

    def scores(pr):
        s = sw.merge(pr, on=["context", "id"])
        m = main.merge(pr, on=["context", "id"])
        key = (2 * m.p.values - 1) / np.maximum((m.typ_hi - m.typ_lo).values, FLOOR)
        ap, cpt = apgr_cost(m.q_hi.values, key, (m.cost_hi - m.cost_lo).values)
        return apgr_frac(s.q_hi.values, s.p_win.values), ap, cpt

    L = ["# R2 (MF) hyperparameter sweep and nested selection", "",
         f"Grid: d ∈ {DS} (plus d = 256, learning rate 1e-3 only), learning rate ∈ {LRS}, λ ∈ {LAMS} ({len(combos)} combinations); batch size 1024, Adam, early-stopping patience 3 epochs, at most 30.",
         "Inner validation = the 20% of each outer fold's train votes split off by prompt-group hash. APGR columns are point estimates on the outer eval votes, for sensitivity only, not used in selection.", "",
         "| d | Learning rate | λ | Inner validation loss (5-fold mean) | Early-stop epochs (5 folds) | Seconds per fold (mean) | Strong vs weak APGR | Main case APGR$ | Main case CPT$(50%) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in combos:
        vl = np.mean([info[c][f"cv{k}"]["val_loss"] for k in range(5)])
        eps = "/".join(str(info[c][f"cv{k}"]["epochs"]) for k in range(5))
        sec = np.mean([info[c][f"cv{k}"]["sec_total"] for k in range(5)])
        a1, a2, cpt = scores(preds[c])
        L.append(f"| {c[0]} | {c[1]:g} | {c[2]:g} | {vl:.4f} | {eps} | {sec:.0f} | {a1:.3f} | {a2:.3f} | {cpt:.2f} |")
    a1, a2, cpt = scores(nested)
    L += [f"| **nested selection** | | | | | | **{a1:.3f}** | **{a2:.3f}** | **{cpt:.2f}** |", "",
          "Chosen per fold: " + "; ".join(f"cv{k} → d={pick[k][0]}, lr={pick[k][1]:g}, λ={pick[k][2]:g}, "
                                 f"{info[pick[k]][f'cv{k}']['epochs']} epochs" for k in range(5))]
    (root / "reports/r2_sweep.md").write_text("\n".join(L))
    print("\n".join(L))


def export(d, lr, lam, epochs):
    ns = load_mf()
    ns["K"], ns["LR"] = d, lr
    fl, arrays, train = ns["fl"], ns["arrays"], ns["train"]
    dd = fl[(fl.context == "final") & (fl.role == "train")]   # = the full 90% dev set, ratings from the dev-set BT; test votes not used
    t = time.time()
    p, _, _ = train(arrays(dd), lam, epochs=epochs)
    sec = time.time() - t
    v, w2, u = np.asarray(p["v"]), np.asarray(p["w2"]), np.asarray(p["u"])
    g = v * w2                      # effective model vector: f(m, q) = g_m · z + u_m (the split between v and w2 is not identifiable; combining into g removes it)
    # g itself is only determined up to "a global shift + a linear transform" (change W's basis, transform g back, same output).
    # Whitened version: with Σ the covariance of the dev-set prompt projections z, g_w = Σ^(1/2) (g − mean over models).
    # Then the Euclidean distance between two models' g_w = the standard deviation over prompts of their prompt-dependent fit difference, independent of the basis; suitable for a projection plot.
    gi = np.unique(np.asarray(arrays(dd)[0]))
    Z = np.asarray(ns["E"][gi] @ p["W"] + p["b"], dtype=np.float64)
    ev, evec = np.linalg.eigh(np.cov(Z, rowvar=False))
    S_half = (evec * np.sqrt(np.clip(ev, 0, None))) @ evec.T
    gw = (g - g.mean(0)) @ S_half
    names = ns["models"]
    cnt = pd.concat([dd.hi_model, dd.lo_model]).value_counts()
    df = pd.DataFrame(g, columns=[f"g{i}" for i in range(d)])
    df.insert(0, "model", names)
    df.insert(1, "n_votes_dev", [int(cnt.get(m, 0)) for m in names])
    df.insert(2, "bias_u", u)
    df.to_csv(root / "data/r2_model_vectors.csv", index=False)
    dw = pd.DataFrame(gw, columns=[f"w{i}" for i in range(d)])
    dw.insert(0, "model", names)
    dw.insert(1, "n_votes_dev", df.n_votes_dev.values)
    dw.insert(2, "bias_u", u - u.mean())
    dw.to_csv(root / "data/r2_model_vectors_whitened.csv", index=False)
    raw = pd.DataFrame(v, columns=[f"v{i}" for i in range(d)])
    raw.insert(0, "model", names)
    raw.to_csv(root / "data/r2_model_vectors_raw_v.csv", index=False)
    np.savez(root / "data/r2_final_params.npz", **{k: np.asarray(x) for k, x in p.items()}, models=np.array(names))
    meta = {"d": d, "lr": lr, "lambda": lam, "epochs": epochs, "train_votes": int(len(dd)), "seconds": round(sec, 1),
            "a_gap_coef": float(p["a"]), "c1": float(p["c1"]),
            "c2": float(p["c1"] + np.log1p(np.exp(float(p["dc"])))), "w2": w2.tolist()}
    (root / "data/r2_model_vectors_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(meta["seconds"], "s", len(names), "models")


if __name__ == "__main__":
    if mode == "run":
        run(int(sys.argv[3]), (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else None)
    elif mode == "summarize":
        summarize()
    else:
        export(int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]))
