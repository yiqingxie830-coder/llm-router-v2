"""evaluate_cost.py — Stage 2 Step 1: evaluation on the dollar axis (evaluate.py's PGR, with dollars on the x-axis)

Differences from evaluate.py
----------------------------
- Uses only votes where the higher-rated model's typical cost > the lower-rated model's (rule 4 in cost_table.py);
  the other votes are only counted.
- x-axis = normalized dollar cost: (spend of this batch - spend if all go to the lower-rated model)
  / (spend if all go to the higher-rated model - spend if all go to the lower-rated model).
  The extra spend per vote = actual cost of the higher-rated model - actual cost of the lower-rated model,
  which differs by vote (long answers and pricey models cost more). Absolute dollars per 1,000 requests are also reported.
- Ranking rule (set 2026-11-09): on the dollar axis every router is ranked by "expected quality gain ÷ extra cost",
  key = (2p - 1) / max(typical cost gap, FLOOR). p = the router's P(higher-rated model wins); routing to the
  higher-rated model instead of the lower-rated one gains p - (1 - p) = 2p - 1 in expected quality; typical cost gap
  = difference of the two models' typical cost per request (see cost_table.py).
  FLOOR = $0.25 per 1,000 requests (about the 5th percentile of the cost gap), so near-equal prices do not blow up
  the ratio; the report adds sensitivity for FLOOR = 0.1 / 1.0 and for a log version.
- Formal baseline = rating gap ÷ price gap (p from the BT rating gap). The "rating-gap baseline" (ranks by p only,
  ignores price) is kept as a control row.
- Within-pair metric (main dollar case only): group by model pair, compute APGR$ inside each group, then average
  weighted by votes. The price gap is fixed inside a group, so this only measures how well the router ranks prompts
  within one pair of models. Rating-gap baselines only keep cross-fold rating jitter inside a group, so they are
  expected to be ≈ random. Only pairs with ≥ 30 votes; groups whose quality gap (all strong - all weak) < 0.05 have
  unstable APGR and are skipped in the weighted mean. A pooled version that cannot blow up is also reported
  (sum of each group's area gain ÷ sum of each group's quality gap, i.e. weighted by votes × quality gap).
- The extra spend of a single vote is occasionally negative (the pricey model answered briefly this time); the
  cumulative cost uses the running maximum so the x-axis is monotone (PGR itself is not made monotone).
- APGR$ = area under PGR against normalized dollar cost; CPT$(50%) = normalized cost spent when PGR reaches 50%.
- Each view also reports the old APGR on the call-share axis, for comparison with stage 1.

Usage: python -I evaluate_cost.py <repo root> [--demo | <prediction file> <name>] [--final]
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
FLOOR = 0.25 / 1000          # dollars per request
BASE = "gap ÷ price gap (formal baseline, bucketed win/tie/loss)"
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
                             ["no price", "higher-rated costs more"], "higher-rated costs less or same")


def curve(q, p, w, extra, rng):
    """Return (PGR grid on the dollar axis, PGR grid on the call-share axis, r_s, r_w)."""
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
        row = {"APGR_by_call_share": frac_apgr[name]}
        for j, m in enumerate(["APGR$", "CPT$50", "CPT$80"]):
            lo, hi = np.nanpercentile(b[:, j], [2.5, 97.5])
            row[m] = [round(float(point[name][j]), 3), round(float(lo), 3), round(float(hi), 3)]
        if name != BASE:
            dlt = b[:, 0] - base[:, 0]
            row["ΔAPGR$_vs_formal_baseline"] = [round(float(point[name][0] - point[BASE][0]), 3),
                                     *np.percentile(dlt, [2.5, 97.5]).round(3).tolist()]
        out["metrics"][name] = row
    return out


def within_pairs(d, routers):
    """Within-pair APGR$: computed per model pair, then weighted. Returns point estimates, intervals and gains over the formal baseline."""
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
        for j, lab in enumerate(["within-pair APGR$ (vote-weighted)", "within-pair APGR$ (pooled)"]):
            row[lab] = [round(float(point[name][j]), 3), *np.percentile(b[:, j], [2.5, 97.5]).round(3).tolist()]
            if name != BASE:
                row["Δ " + lab] = [round(float(point[name][j] - point[BASE][j]), 3),
                                   *np.percentile(b[:, j] - base[:, j], [2.5, 97.5]).round(3).tolist()]
        out["metrics"][name] = row
    return out


def sensitivity(d, p_bt):  # p_bt: the formal baseline's p (2p−1 = expected quality gain)
    """Sensitivity of the formal baseline to FLOOR and to a log version (point estimates only)."""
    dt = (d.typ_hi - d.typ_lo).values
    q, extra = d.q_hi.values, (d.cost_hi - d.cost_lo).values
    variants = {f"floor ${f}/1k": (2 * p_bt - 1) / np.maximum(dt, f / 1000) for f in (0.1, 0.25, 1.0)}
    variants["log: (2p-1) / log(1 + cost gap/floor)"] = (2 * p_bt - 1) / np.log1p(dt / FLOOR)
    out = {"share of votes at the floor": round(float((dt < FLOOR).mean()), 4)}
    for k, v in variants.items():
        out[k] = round(float(metrics(curve(q, v, np.ones(len(q)), extra, np.random.default_rng(SEED))[0])[0]), 3)
    return out


def run_all(d, preds):
    m0 = (d.cost_class == "higher-rated costs more").values
    d, preds = d[m0], {k: v[m0] for k, v in preds.items()}
    res = {"① pooled ($)": evaluate(d, preds)}
    for lo, hi, lab in BUCKETS:
        m = ((d.rating_gap >= lo) & (d.rating_gap < hi)).values
        res[f"① gap {lab} ($)"] = evaluate(d[m], {k: v[m] for k, v in preds.items()})
    m = d.is_strong_vs_weak.values
    res["② pricier strong vs cheaper weak (main dollar case)"] = evaluate(d[m], {k: v[m] for k, v in preds.items()})
    res["② within pair"] = within_pairs(d[m], {k: v[m] for k, v in preds.items()})
    p_bt = 1 / (1 + 10 ** (-d.rating_gap.values / 400))
    pb = d.p_tie_bucket.values
    res["formal baseline sensitivity"] = {"① pooled": sensitivity(d, pb), "② main dollar case": sensitivity(d[m], pb[m])}
    return res


def report(res, name):
    lines = [f"# Dollar-axis evaluation: {name} ({'test set' if use_final else 'dev set 5-fold, eval votes of all folds combined'})", "",
             "Only votes where the higher-rated model has the higher typical cost. Format: point estimate [95% interval], 1000 bootstrap resamples by prompt group.",
             "The x-axis of APGR$ / CPT$ is normalized dollar cost (0 = all to the lower-rated model, 1 = all to the higher-rated model).", ""]
    fmt = lambda v: f"{v[0]} [{v[1]}, {v[2]}]"
    for view, r in res.items():
        if view == "② within pair":
            labs = ["within-pair APGR$ (vote-weighted)", "within-pair APGR$ (pooled)"]
            lines += [f"## ② within-pair APGR$  ({r['n_pairs']} model pairs with ≥ {MIN_PAIR} votes, {r['n_votes']} votes)", "",
                      "APGR$ is computed inside each model pair and then weighted; the price gap is fixed within a pair, so this only measures ranking by prompt. "
                      f"The vote-weighted version skips pairs whose quality gap < {MIN_GAP}; the pooled version weights by votes × quality gap and cannot blow up.", "",
                      "| Router | vote-weighted | Δ vs formal baseline | pooled | Δ vs formal baseline |", "|---|---|---|---|---|"]
            for rn, m in r["metrics"].items():
                d1 = fmt(m["Δ " + labs[0]]) if "Δ " + labs[0] in m else "—"
                d2 = fmt(m["Δ " + labs[1]]) if "Δ " + labs[1] in m else "—"
                lines.append(f"| {rn} | {fmt(m[labs[0]])} | {d1} | {fmt(m[labs[1]])} | {d2} |")
            lines.append("")
            continue
        if view == "formal baseline sensitivity":
            lines += ["## Sensitivity of the formal baseline to the price-gap floor (APGR$ point estimate)", "",
                      "| View | " + " | ".join(k for k in r["① pooled"]) + " |",
                      "|---" * (len(r["① pooled"]) + 1) + "|"]
            for v, row in r.items():
                lines.append(f"| {v} | " + " | ".join(str(x) for x in row.values()) + " |")
            lines.append("")
            continue
        lines += [f"## {view}  ({r['n_votes']} votes; per 1,000 requests: all to lower-rated ${r['usd_per_1k_all_lo']}, "
                  f"all to higher-rated ${r['usd_per_1k_all_hi']})", "",
                  "| Router | APGR$ | CPT$(50%) | CPT$(80%) | ΔAPGR$ vs formal baseline | For reference: APGR on call share |",
                  "|---|---|---|---|---|---|"]
        for rn, m in r["metrics"].items():
            dl = fmt(m["ΔAPGR$_vs_formal_baseline"]) if "ΔAPGR$_vs_formal_baseline" in m else "—"
            lines.append(f"| {rn} | {fmt(m['APGR$'])} | {fmt(m['CPT$50'])} | {fmt(m['CPT$80'])} | {dl} | "
                         f"{m['APGR_by_call_share']} |")
        lines.append("")
    return "\n".join(lines)


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#2b2b29", "#6b6a63", "#e1e0d9"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59b6", "#888888", "#d4a017"]
    views = ["① pooled ($)", "② pricier strong vs cheaper weak (main dollar case)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), dpi=150, facecolor="#fcfcfb")
    for ax, v in zip(axes, views):
        r = res[v]
        lo, hi = r["usd_per_1k_all_lo"], r["usd_per_1k_all_hi"]
        x = lo + FINE * (hi - lo)
        ax.set_facecolor("#fcfcfb")
        ax.plot([lo, hi], [0, 1], color=muted, lw=1, ls="--", label="random expectation")
        for (rn, c), col in zip(r["curves"].items(), colors):
            ax.plot(x, c, color=col, lw=2, label=f"{rn}  APGR$ {r['metrics'][rn]['APGR$'][0]}")
        ax.scatter([lo, hi], [0, 1], s=50, color=ink, zorder=5)
        ax.annotate("all lower-rated", (lo, 0), (6, 8), textcoords="offset points", color=ink, fontsize=9)
        ax.annotate("all higher-rated", (hi, 1), (-80, -14), textcoords="offset points", color=ink, fontsize=9)
        ax.set_title(f"{v.replace('pricier strong vs cheaper weak (main dollar case)', 'main dollar case')} (n={r['n_votes']})", color=ink, fontsize=11, loc="left")
        ax.set_xlabel("cost per 1,000 requests (USD)", color=muted)
        ax.set_ylabel("PGR", color=muted)
        ax.grid(color=grid, lw=0.6)
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=muted)
        ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc="upper left", bbox_to_anchor=(0, -0.14))
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    if args and args[0] == "--demo":
        name, preds = "demo_dollar_baselines", {}
    else:
        pf, name = args[0], args[1]
        pr = pd.read_parquet(pf) if pf.endswith(".parquet") else pd.read_csv(pf)
        ev = ev.merge(pr[["context", "id", "p"]], on=["context", "id"], how="left")
        assert ev.p.notna().all(), "prediction file is missing some eval votes"
        preds = {name: ev.p.values}
    rng = np.random.default_rng(SEED)
    p_bt = 1 / (1 + 10 ** (-ev.rating_gap.values / 400))
    dt = np.maximum((ev.typ_hi - ev.typ_lo).values, FLOOR)
    key = lambda p: (2 * p - 1) / dt       # expected quality gain ÷ extra cost
    # Formal baseline (set 2026-11-10): rating-gap baseline with ties, expected gain = P(win) − P(loss), from tie_baselines.py
    # (empirical bucket shares, fitted on each context's train votes only). The BT half-vote and 3-class logistic versions stay as control rows.
    sfx = "_final" if use_final else ""
    for tag, col in (("bucket", "p_tie_bucket"), ("mlogit", "p_tie_mlogit")):
        b = pd.read_parquet(root / f"preds/base_tie_{tag}{sfx}.parquet")[["context", "id", "p"]].rename(columns={"p": col})
        ev = ev.merge(b, on=["context", "id"], how="left")
        assert ev[col].notna().all()
    preds = {"random": rng.random(len(ev)),
             "rating-gap baseline (control, ignores price)": p_bt,
             BASE: key(ev.p_tie_bucket.values),
             "gap ÷ price gap (control: 3-class logistic version)": key(ev.p_tie_mlogit.values),
             "gap ÷ price gap (control: BT half-vote version, former formal baseline)": key(p_bt),
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
