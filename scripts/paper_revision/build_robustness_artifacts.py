#!/usr/bin/env python
"""Build the dose-response figure and the base-learner dominance analysis.
Run AFTER run_dose_response and run_frontier_clf (rf, logreg) complete.

  python -m scripts.paper_revision.build_robustness_artifacts
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

from scripts.paper_revision.config import RESULTS_DIR
from scripts.paper_revision.datasets import DATASETS

FIG = Path("paper_v2/figures")
FDIR = RESULTS_DIR / "frontier_benchmark"
NONGEN = ["baseline", "threshold_moved", "cost_sensitive", "balanced_rf", "easy_ensemble", "triage_weighting"]
GEN = ["smote", "clean_masked_smote", "napierala_guided_smote"]


def dose_response_fig():
    p = RESULTS_DIR / "dose_response_arms.parquet"
    if not p.exists():
        print("dose_response_arms.parquet missing"); return
    df = pd.read_parquet(p)
    per = df.groupby(["dataset", "arm", "dose"])[["accuracy", "balanced_accuracy"]].mean().reset_index()
    print(f"\n=== Dose-response, 2-arm (n={per.dataset.nunique()} datasets) ===")
    out = {}
    for arm in ["boundary", "safe"]:
        a = per[per.arm == arm]
        g = a.groupby("dose")[["accuracy", "balanced_accuracy"]].agg(["mean", "sem"])
        out[arm] = g
        doses = g.index.values
        acc0, acc1 = g.loc[doses[0], ("accuracy", "mean")], g.loc[doses[-1], ("accuracy", "mean")]
        bac0, bac1 = g.loc[doses[0], ("balanced_accuracy", "mean")], g.loc[doses[-1], ("balanced_accuracy", "mean")]
        rhos = [spearmanr(s.dose, s.accuracy)[0] for _, s in a.groupby("dataset") if s.dose.nunique() > 2]
        rhos = [r for r in rhos if not np.isnan(r)]
        print(f"  [{arm}] accuracy {acc0:.4f}->{acc1:.4f} ({100*(acc1-acc0):+.2f}pp); "
              f"bacc {bac0:.4f}->{bac1:.4f} ({100*(bac1-bac0):+.2f}pp); "
              f"median within-dataset rho(dose,acc)={np.median(rhos):+.2f}, "
              f"{np.mean([r<0 for r in rhos])*100:.0f}% negative")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    col = {"boundary": "C3", "safe": "C0"}
    for metric, ax, ttl in [("accuracy", axes[0], "Accuracy"),
                            ("balanced_accuracy", axes[1], "Balanced accuracy")]:
        for arm in ["boundary", "safe"]:
            g = out[arm]
            ax.errorbar(g.index.values, g[(metric, "mean")], yerr=g[(metric, "sem")],
                        marker="o", color=col[arm], capsize=3,
                        label=f"{arm} seeds")
        ax.set_xlabel("dose: synthetic minority added (fraction of \\#errors)")
        ax.set_ylabel(f"mean {metric.replace('_',' ')}"); ax.set_title(ttl)
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Dose-response: boundary generation trades accuracy for balanced accuracy; "
                 "safe generation does not", fontsize=11)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(FIG / "fig_dose_response.pdf", bbox_inches="tight"); plt.close()
    print(f"wrote {FIG / 'fig_dose_response.pdf'}")


def _read_mean(path, metric):
    return float(pd.read_parquet(path)[metric].mean()) if path.exists() else np.nan


def _clf_metric(clf, method, ds, metric):
    if method in ("balanced_rf", "easy_ensemble"):
        return _read_mean(FDIR / f"original__{method}__{ds}.parquet", metric)       # learner-agnostic
    if method in ("cost_sensitive", "threshold_moved"):
        return _read_mean(FDIR / clf / f"original__{method}__{ds}.parquet", metric)  # this script's cells
    return _read_mean(RESULTS_DIR / "main_benchmark" / f"{clf}__{method}__{ds}.parquet", metric)


def baselearner_dominance(metric="balanced_accuracy"):
    roster = [d.name for d in DATASETS]
    for clf in ["rf", "logreg"]:
        rows = []
        for ds in roster:
            ng = {m: _clf_metric(clf, m, ds, metric) for m in NONGEN}
            gn = {m: _clf_metric(clf, m, ds, metric) for m in GEN}
            ng = {k: v for k, v in ng.items() if not np.isnan(v)}
            gn = {k: v for k, v in gn.items() if not np.isnan(v)}
            if not ng or not gn:
                continue
            rows.append({"dataset": ds, "best_nongen": max(ng.values()),
                         "best_gen": max(gn.values()),
                         **{f"ng_{k}": v for k, v in ng.items()},
                         **{f"gn_{k}": v for k, v in gn.items()}})
        d = pd.DataFrame(rows)
        if not len(d):
            print(f"\n[{clf}] no data yet"); continue
        wins = (d.best_nongen >= d.best_gen).mean()
        adv = 100 * (d.best_nongen - d.best_gen).mean()
        print(f"\n=== base learner = {clf.upper()} | {metric} | n={len(d)} OpenML datasets ===")
        print(f"  best non-generative >= best generative in {wins*100:.0f}% of datasets; "
              f"mean advantage = {adv:+.2f} pp")
        try:
            p = wilcoxon(d.best_nongen - d.best_gen, alternative="greater").pvalue
            print(f"  paired Wilcoxon (non-gen > gen): p={p:.2e}")
        except Exception:
            pass
        means = {m: d[f"ng_{m}"].mean() for m in NONGEN if f"ng_{m}" in d}
        means.update({m: d[f"gn_{m}"].mean() for m in GEN if f"gn_{m}" in d})
        for m, v in sorted(means.items(), key=lambda kv: -kv[1]):
            tag = "non-gen" if m in NONGEN else "gen"
            print(f"    [{tag:7s}] {m:24s} mean {metric}={v:.4f}")
        d.to_parquet(RESULTS_DIR / f"baselearner_dominance_{clf}.parquet")


if __name__ == "__main__":
    dose_response_fig()
    baselearner_dominance("balanced_accuracy")
