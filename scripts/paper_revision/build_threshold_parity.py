#!/usr/bin/env python
"""Analyse the threshold-parity + probability-metric run (reviewers #2, #5, #8).

Reads results/paper_revision/threshold_parity/*.parquet and produces:
  (A) THRESHOLD PARITY (reviewer #2): does oversampling still lose once every
      strategy gets the same out-of-fold threshold-tuning opportunity? Reports,
      on XGBoost binary datasets: threshold-moving (baseline+tuned) vs SMOTE at
      default and at tuned thresholds; how much tuning alone helps SMOTE; and the
      best-non-generative vs best-generative dominance at TUNED threshold.
  (B) MODEL-CONTROLLED (reviewer #5): the same best-gen-vs-best-non-gen-at-parity
      comparison per base learner (xgboost / rf / logreg) -> table_parity.tex.
  (C) NEW METRICS (reviewer #8): per-strategy PR-AUC, ROC-AUC, Brier, ECE,
      recall@FPR (XGBoost) -> table_metrics.tex; shows the SMOTE family is worse
      calibrated than the baseline/threshold strategies.

Writes a macros file (paper_v2/tables/parity_macros.tex) so the prose can cite
exact numbers. Safe to run on partial output (uses whatever cells are present).

    python -m scripts.paper_revision.build_threshold_parity
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.paper_revision.config import RESULTS_DIR

CELLS = RESULTS_DIR / "threshold_parity"
TAB = Path("paper_v2/tables")
GEN = ["smote", "borderline_smote", "adasyn", "safe_level_smote"]
NONGEN = ["baseline", "cost", "balanced_rf", "easy_ensemble"]
LEARNERS = ["xgboost", "rf", "logreg"]


def _load():
    files = sorted(CELLS.glob("*.parquet"))
    if not files:
        raise SystemExit("no threshold_parity cells yet.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "error" in df.columns:
        df = df[df["error"].isna()] if df["error"].notna().any() else df
    # per (dataset, base_learner, strategy): mean over folds
    metrics = [c for c in df.columns if c not in
               ("dataset", "roster", "base_learner", "strategy", "repeat", "fold", "error", "n_train")]
    agg = df.groupby(["dataset", "base_learner", "strategy"])[metrics].mean().reset_index()
    return agg


def _ci(a, rng, n=10000):
    a = np.asarray(a, float)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    boot = [rng.choice(a, len(a), replace=True).mean() for _ in range(n)]
    return tuple(np.percentile(boot, [2.5, 97.5]))


def _paired(a, b):
    """mean (a-b) in pp, win rate, wilcoxon p over common datasets."""
    d = (a - b).dropna()
    if len(d) < 3:
        return float("nan"), float("nan"), float("nan"), len(d)
    try:
        p = wilcoxon(d).pvalue
    except ValueError:
        p = float("nan")
    return 100 * d.mean(), 100 * (d > 0).mean(), p, len(d)


def main():
    rng = np.random.default_rng(0)
    agg = _load()
    learners_present = [l for l in LEARNERS if l in set(agg.base_learner)]
    print(f"datasets={agg.dataset.nunique()}  learners={learners_present}")
    macros = {}

    def piv(learner, value):
        s = agg[agg.base_learner == learner]
        return s.pivot_table(index="dataset", columns="strategy", values=value)

    # ---------- (A) threshold parity on XGBoost ----------
    print("\n=== (A) THRESHOLD PARITY (XGBoost binary) ===")
    bd = piv("xgboost", "bacc_default"); bt = piv("xgboost", "bacc_tuned")
    have = [m for m in GEN + NONGEN if m in bt.columns]
    bestgen_t = bt[[m for m in GEN if m in bt.columns]].max(axis=1)
    bestng_t = bt[[m for m in NONGEN if m in bt.columns]].max(axis=1)
    bestgen_d = bd[[m for m in GEN if m in bd.columns]].max(axis=1)
    comparisons = []
    if "baseline" in bd.columns and "smote" in bd.columns:
        comparisons.append(("threshold-move vs SMOTE@default", bt["baseline"], bd["smote"]))
    if "baseline" in bt.columns and "smote" in bt.columns:
        comparisons.append(("threshold-move vs SMOTE@tuned", bt["baseline"], bt["smote"]))
        comparisons.append(("SMOTE@tuned vs SMOTE@default", bt["smote"], bd["smote"]))
    comparisons.append(("best non-gen@tuned vs best gen@tuned", bestng_t, bestgen_t))
    comparisons.append(("best non-gen@tuned vs best gen@default", bestng_t, bestgen_d))
    for name, a, b in comparisons:
        m, wr, p, n = _paired(a, b)
        print(f"  {name:42s} d={m:+.2f}pp WR={wr:.0f}% p={p:.3g} (n={n})")
        macros[name] = (m, wr, p, n)

    # ---------- (B) model-controlled parity table ----------
    print("\n=== (B) best non-gen vs best gen @ TUNED threshold, per base learner ===")
    rows = []
    for lr in learners_present:
        bt_l = piv(lr, "bacc_tuned")
        g = [m for m in GEN if m in bt_l.columns]; ng = [m for m in NONGEN if m in bt_l.columns]
        if not g or not ng:
            continue
        bg = bt_l[g].max(axis=1); bn = bt_l[ng].max(axis=1)
        d = (bn - bg).dropna()
        lo, hi = _ci(d.values * 100, rng)
        try:
            p = wilcoxon(d).pvalue
        except ValueError:
            p = float("nan")
        # best single fixed names
        bestgen_name = bt_l[g].mean().idxmax(); bestng_name = bt_l[ng].mean().idxmax()
        print(f"  {lr:8s} n={len(d)}  NG>=G WR={100*(d>=0).mean():.0f}%  adv={100*d.mean():+.2f}pp "
              f"[{lo:+.2f},{hi:+.2f}] p={p:.3g}  (top gen={bestgen_name}, top non-gen={bestng_name})")
        rows.append(dict(learner=lr, n=len(d), wr=100 * (d >= 0).mean(),
                         adv=100 * d.mean(), lo=lo, hi=hi, p=p,
                         best_gen=bt_l[g].mean().max() * 100, best_ng=bt_l[ng].mean().max() * 100))
    if rows:
        _write_parity_table(rows)

    # ---------- (C) probability metrics on XGBoost ----------
    print("\n=== (C) probability metrics by strategy (XGBoost, mean over datasets) ===")
    order = ["baseline", "cost", "smote", "borderline_smote", "adasyn",
             "safe_level_smote", "balanced_rf", "easy_ensemble"]
    metric_rows = []
    for value, lab in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
                       ("brier", "Brier"), ("ece", "ECE"),
                       ("recall_fpr10", "Rec@FPR10")]:
        p = piv("xgboost", value)
        means = {m: p[m].mean() for m in order if m in p.columns}
        metric_rows.append((lab, means))
    present = [m for m in order if any(m in mr[1] for mr in metric_rows)]
    hdr = "  " + "metric".ljust(11) + "".join(f"{m[:9]:>10s}" for m in present)
    print(hdr)
    for lab, means in metric_rows:
        print("  " + lab.ljust(11) + "".join(f"{means.get(m, float('nan')):>10.3f}" for m in present))
    _write_metrics_table(metric_rows, present)

    # ---------- macros ----------
    TAB.mkdir(parents=True, exist_ok=True)
    def fmt(key, idx, dp=2):
        v = macros.get(key, (float("nan"),) * 4)[idx]
        return f"{v:+.{dp}f}" if idx == 0 else f"{v:.0f}"
    lines = []
    if "threshold-move vs SMOTE@tuned" in macros:
        m, wr, p, n = macros["threshold-move vs SMOTE@tuned"]
        lines.append(rf"\newcommand{{\ThreshVsSmoteTuned}}{{{m:+.2f}}}")
        lines.append(rf"\newcommand{{\ThreshVsSmoteTunedWR}}{{{wr:.0f}}}")
    if "SMOTE@tuned vs SMOTE@default" in macros:
        m, wr, p, n = macros["SMOTE@tuned vs SMOTE@default"]
        lines.append(rf"\newcommand{{\SmoteTuningGain}}{{{m:+.2f}}}")
    if "best non-gen@tuned vs best gen@tuned" in macros:
        m, wr, p, n = macros["best non-gen@tuned vs best gen@tuned"]
        lines.append(rf"\newcommand{{\NgDomTunedWR}}{{{wr:.0f}}}")
        lines.append(rf"\newcommand{{\NgDomTunedAdv}}{{{m:+.2f}}}")
    (TAB / "parity_macros.tex").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {TAB/'parity_macros.tex'}, {TAB/'table_parity.tex'}, {TAB/'table_metrics.tex'}")


def _write_parity_table(rows):
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Base learner & $n$ & NG$\ge$G & non-gen advantage & paired $p$ \\",
             r"\midrule"]
    name = {"xgboost": "XGBoost", "rf": "Random forest", "logreg": "Logistic regression"}
    for r in rows:
        lines.append(f"{name.get(r['learner'], r['learner'])} & {int(r['n'])} & "
                     f"{r['wr']:.0f}\\% & ${r['adv']:+.2f}$ "
                     f"[{r['lo']:+.2f},{r['hi']:+.2f}] & ${r['p']:.1g}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "table_parity.tex").write_text("\n".join(lines) + "\n")


def _write_metrics_table(metric_rows, present):
    short = {"baseline": "none", "cost": "cost", "smote": "SMOTE",
             "borderline_smote": "Border", "adasyn": "ADASYN",
             "safe_level_smote": "SafeLvl", "balanced_rf": "bRF", "easy_ensemble": "EasyEns"}
    cols = "l" + "r" * len(present)
    head = " & ".join([""] + [short.get(m, m) for m in present])
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", head + r" \\", r"\midrule"]
    for lab, means in metric_rows:
        lines.append(" & ".join([lab] + [f"{means.get(m, float('nan')):.3f}" for m in present]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "table_metrics.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
