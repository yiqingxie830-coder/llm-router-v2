"""r3_eval.py — formal comparison of approach 1 vs approach 2 (dev set 5-fold, view ②), following reports/r3_prereg.md

All routers are scored on the same ② eval votes and the same bootstrap resamples (1000, by prompt group),
so differences between routers are paired.
- Main metric: APGR on the call-share axis (ranked by P(win)), compared with the rating-gap baseline, plus paired
  differences between approaches.
- Secondary: main dollar case (② votes where the higher-rated model costs more) APGR$ and CPT$(50%),
        ranking key = (P(win) − P(loss)) / max(typical cost gap, $0.25 per 1,000);
        within-pair APGR$ (lower bound) reuses evaluate_cost.within_pairs.
- Added on request: APGR by has_single_answer group; rank correlation of is_open_ended and other fields with
  approach 1's PC1 coordinate t1.

Usage: python -I r3_eval.py <repo root> [--final]
Writes: reports/r3_compare[_final].md / .json
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
VAR = os.environ.get("R3_VARIANT", "")   # e.g. "_ep100": MF routers switch to the sensitivity version with early stopping capped at 100 epochs

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

# ---- routers: name -> prediction file (columns p_win = P(win), p = (1 + P(win) − P(loss))/2) ----
P = root / "preds"
files = {
    "R1 (approach 1, all votes)": (P / f"r1b_win{sfx}.parquet", P / f"r1b_eff{sfx}.parquet"),
    "R2 (approach 1, all votes)": (P / ("r2_nested.parquet" if not use_final else "r2_win_final.parquet"),
                         P / ("r2_nested.parquet" if not use_final else "r2_eff_final.parquet")),
    "R2-② (approach 1, ② votes only)": P / f"r3/r2sw{sfx}{VAR}.parquet",
    "R3-LR": P / f"r3/r3lr{sfx}.parquet",
    "R3-MF": P / f"r3/r3mf{sfx}{VAR}.parquet",
    "R3 (approach 2, LR/MF chosen per fold)": P / f"r3/r3{sfx}.parquet",
    "R2 + R3 (combined)": P / f"r3/r2r3{sfx}{VAR}.parquet",
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
BT, TIE = "rating-gap baseline (BT)", "rating-gap baseline with ties (formal)"
pw = {BT: p_bt, TIE: tb.p_win.values, **pw}
pe = {BT: p_bt, TIE: tb.p.values, **pe}
names = list(pw)
R2SW, R3, R2R3, R2 = "R2-② (approach 1, ② votes only)", "R3 (approach 2, LR/MF chosen per fold)", "R2 + R3 (combined)", "R2 (approach 1, all votes)"
PAIRS = [(R3, R2SW), (R2R3, R2SW), (R3, R2), (R2R3, R2), (R2SW, R2), ("R3-LR", R2SW), ("R3-MF", R2SW)]

# ---- labels and PC1 coordinates ----
lab = []
for line in open(root / "r3/r3_labels.jsonl", encoding="utf-8"):
    r = json.loads(line)
    lab.append({"group_id": r["group_id"], **(r.get("labels") or {})})
lab = pd.DataFrame(lab)
ev = ev.merge(lab, on="group_id", how="left")

q = ev.q_hi.values
main = (ev.cost_class == "higher-rated costs more").values
dt = np.maximum((ev.typ_hi - ev.typ_lo).values, FLOOR)
extra = (ev.cost_hi - ev.cost_lo).values
groups, gidx = np.unique(ev.group_id.values, return_inverse=True)
hsa = ev.has_single_answer.values  # votes with missing labels (refused) stay out of the group table


def all_scores(w, rng):
    """For one resample (weights w), every router's: call-share APGR / CPT50, main dollar case APGR$ / CPT$50, APGR per group."""
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

# ---- within-pair APGR$ (lower bound) ----
wp_routers = {BASE: (2 * pe[TIE] - 1) / dt, **{n: (2 * pe[n] - 1) / dt for n in names if n not in (BT, TIE)}}
d_main = ev[main].reset_index(drop=True)
res["within_pairs"] = within_pairs(d_main, {k: v[main] for k, v in wp_routers.items()})

# ---- rank correlation of labels with PC1 (t1) (dev set ② prompts, one row per prompt) ----
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

# ---- report ----
f = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
L = [f"# Approach 1 vs approach 2 ({'test set' if use_final else 'dev set 5-fold'}, view ②)", "",
     f"{res['n_votes']} ② votes, {res['n_groups']} prompt groups; main dollar case {res['n_main']} votes. Point estimate [95% interval], 1000 paired bootstrap resamples by prompt group.", "",
     "## Main metric: APGR on call share", "", "| Router | APGR | Δ vs rating-gap baseline | CPT(50%) |", "|---|---|---|---|"]
for n in names:
    L.append(f"| {n} | {f(res['routers'][n]['APGR'])} | {f(res['vs_BT'][n]['APGR']) if n != BT else '—'} | {f(res['routers'][n]['CPT50'])} |")
L += ["", "## Paired differences between approaches", "", "| Comparison | ΔAPGR | ΔAPGR$ (main dollar case) |", "|---|---|---|"]
for k, v in res["pairs"].items():
    L.append(f"| {k} | {f(v['APGR'])} | {f(v['APGR$'])} |")
L += ["", "## Main dollar case (pricier strong vs cheaper weak)", "", "| Router | APGR$ | Δ vs formal baseline | CPT$(50%) |", "|---|---|---|---|"]
for n in names:
    L.append(f"| {n} | {f(res['routers'][n]['APGR$'])} | {f(res['vs_TIE$'][n]) if n != TIE else '—'} | {f(res['routers'][n]['CPT$50'])} |")
wp = res["within_pairs"]
L += ["", f"## Within-pair APGR$ (lower bound; {wp['n_pairs']} model pairs with ≥ 30 votes, {wp['n_votes']} votes)", "",
      "| Router | within-pair APGR$ | Δ vs formal baseline |", "|---|---|---|"]
for n, m in wp["metrics"].items():
    k = "within-pair APGR$ (vote-weighted)"
    L.append(f"| {n} | {f(m[k])} | {f(m['Δ ' + k]) if 'Δ ' + k in m else '—'} |")
L += ["", "## By has_single_answer (APGR on call share)", "",
      "| Router | " + " | ".join(f"score {g} ({res['hsa_n'][g]} votes)" for g in range(4)) + " |", "|---|---|---|---|---|"]
for n in names:
    L.append(f"| {n} | " + " | ".join(f(res['routers'][n][f'hsa{g}']) for g in range(4)) + " |")
L += ["", "Gain over the rating-gap baseline:", "", "| Router | " + " | ".join(f"score {g}" for g in range(4)) + " |", "|---|---|---|---|---|"]
for n in names:
    if n != BT:
        L.append(f"| {n} | " + " | ".join(f(res['vs_BT'][n][f'hsa{g}']) for g in range(4)) + " |")
if "spearman" in res:
    s = res["spearman"]
    L += ["", f"## Rank correlation of labels with approach 1's PC1 coordinate ({s['n_prompts']} dev-set ② prompts)", "",
          "t1 = PC1 coordinate from vectors trained on the full dev set; t1_oof = out-of-fold version (each prompt's coordinate comes from a fold that never saw it).", "",
          "| Field | with t1 | with t1_oof | with t2 |", "|---|---|---|---|"]
    for k, v in s["by_field"].items():
        L.append(f"| {k} | {v['t1']} | {v['t1_oof']} | {v['t2']} |")
    L += ["", "Mean t1_oof by is_open_ended: " + "; ".join(f"score {int(a)}: {b} ({int(c)} prompts)" for a, b, c in s["t1_mean_by_open_ended"]),
          "", "Mean t1_oof by task_type: " + "; ".join(f"{a} {b} ({int(c)})" for a, b, c in s["t1_mean_by_task"])]
(root / f"reports/r3_compare{sfx}{VAR}.md").write_text("\n".join(L))
print("\n".join(L))
