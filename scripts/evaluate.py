"""evaluate.py — Step 5: router evaluation framework (cost–quality curve, PGR / APGR / CPT, bootstrap)

Input: a prediction file from any router (parquet/csv) with columns context, id, p
      p = P(higher-rated model wins outright) in that context; used only for ranking, need not be calibrated.
      context in {cv0..cv4} (dev set 5-fold CV, each fold evaluates only its own eval votes) or final (test set).
Labels come from data/fold_labels.parquet (BT ratings fitted independently per fold, see fold_ratings.py).

Core definitions
----------------
Quality (per vote): routed to the higher-rated model -> 1 if it wins / 0.5 on a tie / 0 if it loses;
   routed to the lower-rated model -> 1 minus that.
   On a tie both sides score 0.5, so sending the cheap model loses no quality: the curve naturally rewards
   "send ties to the small model" (the project's cost preference).
   Note: this is user-preference quality, not accuracy; it differs from RouteLLM's MT-Bench/MMLU scoring.
Routing: sort p from high to low and send the top fraction f to the higher-rated model
   (f = cost proxy: share of calls to the strong model).
   Ties in p are shuffled randomly (fixed seed), so a constant prediction = random routing.
PGR(f) = (r(f) - r_weak) / (r_strong - r_weak): how much of the strong-weak gap is recovered.
APGR = area under the PGR curve, trapezoid over f = 0, 0.1, ..., 1 (10 intervals, as the spec asks).
   Random routing has expected APGR = 0.5; a perfect router can be well above 0.5
   (votes that tie or that the weak side wins never need the strong model).
CPT(x) = smallest share of strong-model calls that reaches PGR >= x (1% grid).
Bootstrap: 1000 resamples of whole group_id groups (votes on the same prompt are not independent);
   r_strong / r_weak and every metric are recomputed on each resample.
   Reports 95% percentile intervals; gains over the baseline (ΔAPGR) are paired on the same resample.

Two views (set 2026-11-09)
--------------------------
① pooled: all model pairs together. Also reported per rating-gap bucket (<50, 50–150, >150 points).
   !! The pooled number depends on Arena's pair sampling (which pairs were drawn most often) and does not
   !! represent any single product scenario; view ② and the buckets are the scenario references.
② strong vs weak: only votes between the strong tier and the weak tier (the strong side is always the
   higher-rated one, so p can be used directly).

Baselines
---------
- All weak (f=0, PGR=0) and all strong (f=1, PGR=1): the two end points
- Random router: p = random number
- Rating-gap baseline: p = BT-predicted win rate of the higher-rated model = 1 / (1 + 10^(-gap/400)),
  ignores the prompt. Every router that reads the prompt reports its gain over this baseline; otherwise
  we cannot tell whether a gain comes from understanding the prompt or just from knowing which model is stronger.

Usage:
  python -I evaluate.py <repo root> --demo                    # run the framework with the random and rating-gap baselines (dev 5-fold)
  python -I evaluate.py <repo root> <prediction file> <name>  # evaluate any router (dev 5-fold by default)
  add --final to evaluate on the test set (only once, after the final model is chosen)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT, SEED = 1000, 2026
GRID = np.linspace(0, 1, 11)             # the 10 APGR intervals
FINE = np.linspace(0, 1, 101)            # 1% grid for CPT
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
    """Given weights w (bootstrap counts), return PGR on the FINE grid. q = quality when routed to the higher-rated model."""
    order = np.lexsort((rng.random(len(p)), -p))  # p descending, ties random
    qs, ws = q[order], w[order]
    W = ws.sum()
    r_w = (ws * (1 - qs)).sum() / W                      # all to the lower-rated model
    r_s = (ws * qs).sum() / W                            # all to the higher-rated model
    cw = np.concatenate([[0], np.cumsum(ws)]) / W         # share sent to strong so far
    gain = np.concatenate([[0], np.cumsum(ws * (2 * qs - 1))]) / W  # each extra vote sent to strong adds q-(1-q)
    r = r_w + np.interp(FINE, cw, gain)
    return (r - r_w) / (r_s - r_w), r_s, r_w


def metrics(pgr):
    g = np.interp(GRID, FINE, pgr)
    apgr = np.trapezoid(g, GRID)
    cpt = {x: FINE[np.argmax(pgr >= x)] if (pgr >= x).any() else np.nan for x in (0.5, 0.8)}
    return apgr, cpt[0.5], cpt[0.8]


def evaluate(d: pd.DataFrame, routers: dict) -> dict:
    """routers: name -> p array (aligned with d). Returns point estimates, intervals and paired gains."""
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
    base = np.array(boots["rating-gap baseline"]) if "rating-gap baseline" in boots else None
    for name in routers:
        b = np.array(boots[name])
        row = {}
        for j, m in enumerate(["APGR", "CPT50", "CPT80"]):
            lo, hi = np.nanpercentile(b[:, j], [2.5, 97.5])
            row[m] = [round(float(point[name][j]), 3), round(float(lo), 3), round(float(hi), 3)]
        if base is not None and name != "rating-gap baseline":
            dlt = b[:, 0] - base[:, 0]
            row["ΔAPGR_vs_gap_baseline"] = [round(float(point[name][0] - point["rating-gap baseline"][0]), 3),
                                     *np.percentile(dlt, [2.5, 97.5]).round(3).tolist()]
        out["metrics"][name] = row
    return out


def run_all(d: pd.DataFrame, preds: dict) -> dict:
    res = {"① pooled": evaluate(d, preds)}
    for lo, hi, lab in BUCKETS:
        m = (d.rating_gap >= lo) & (d.rating_gap < hi)
        res[f"① gap {lab}"] = evaluate(d[m], {k: v[m.values] for k, v in preds.items()})
    m = d.is_strong_vs_weak.values
    res["② strong vs weak"] = evaluate(d[m], {k: v[m] for k, v in preds.items()})
    return res


def report(res: dict, name: str):
    lines = [f"# Evaluation: {name} ({'test set' if use_final else 'dev set 5-fold CV, eval votes of all folds combined'})", "",
             "Format: point estimate [95% interval]; 1000 bootstrap resamples by prompt group.", "",
             "> The pooled view ① depends on Arena's pair sampling and does not represent any single product scenario; view ② and the buckets are the scenario references.", ""]
    for view, r in res.items():
        lines += [f"## {view}  ({r['n_votes']} votes, {r['n_groups']} prompt groups; "
                  f"quality all-strong {r['r_strong']}, all-weak {r['r_weak']})", "",
                  "| Router | APGR | CPT(50%) | CPT(80%) | ΔAPGR vs rating-gap baseline |", "|---|---|---|---|---|"]
        fmt = lambda v: f"{v[0]} [{v[1]}, {v[2]}]"
        for rn, m in r["metrics"].items():
            lines.append(f"| {rn} | {fmt(m['APGR'])} | {fmt(m['CPT50'])} | {fmt(m['CPT80'])} | "
                         f"{fmt(m['ΔAPGR_vs_gap_baseline']) if 'ΔAPGR_vs_gap_baseline' in m else '—'} |")
        lines += ["", "End points: all weak = (0% strong calls, PGR 0); all strong = (100% strong calls, PGR 1).", ""]
    return "\n".join(lines)


def plot(res: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#2b2b29", "#6b6a63", "#e1e0d9"
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical colours 1-3 (all pairs pass the contrast check)
    views = ["① pooled", "② strong vs weak"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, facecolor="#fcfcfb")
    for ax, v in zip(axes, views):
        ax.set_facecolor("#fcfcfb")
        ax.plot([0, 1], [0, 1], color=muted, lw=1, ls="--", label="random expectation (diagonal)")
        for (rn, c), col in zip(res[v]["curves"].items(), colors):
            ax.plot(FINE, c, color=col, lw=2, label=f"{rn}  APGR {res[v]['metrics'][rn]['APGR'][0]}")
        ax.scatter([0, 1], [0, 1], s=50, color=ink, zorder=5)
        ax.annotate("all weak", (0, 0), (6, 8), textcoords="offset points", color=ink, fontsize=9)
        ax.annotate("all strong", (1, 1), (-54, -14), textcoords="offset points", color=ink, fontsize=9)
        ax.set_title(f"{v} (n={res[v]['n_votes']})", color=ink, fontsize=11, loc="left")
        ax.set_xlabel("share routed to the strong (higher-rated) model", color=muted)
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
        name = "demo_random_and_gap_baselines"
        preds = {}
    else:
        pf, name = args[0], args[1]
        pr = pd.read_parquet(pf) if pf.endswith(".parquet") else pd.read_csv(pf)
        ev = ev.merge(pr[["context", "id", "p"]], on=["context", "id"], how="left")
        assert ev.p.notna().all(), "prediction file is missing some eval votes"
        preds = {name: ev.p.values}
    rng = np.random.default_rng(SEED)
    preds = {"random": rng.random(len(ev)),
             "rating-gap baseline": 1 / (1 + 10 ** (-ev.rating_gap.values / 400)), **preds}
    res = run_all(ev, preds)
    out = root / "reports" / f"eval_{name}{'_final' if use_final else ''}"
    Path(f"{out}.json").write_text(json.dumps(res, ensure_ascii=False))
    Path(f"{out}.md").write_text(report(res, name))
    try:
        plot(res, Path(f"{out}.png"))
    except Exception as e:  # a plotting failure does not affect the metrics
        print("plot failed:", e)
    print(report(res, name))
