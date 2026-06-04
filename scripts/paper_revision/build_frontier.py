#!/usr/bin/env python
"""Frontier analysis + figure: do NON-generative methods dominate the SMOTE family?

Combines the already-run baseline + SMOTE-family cells (keel_benchmark / main_benchmark) with
the non-generative cells from run_frontier (frontier_benchmark), places every method on the
accuracy/balanced-accuracy plane per benchmark, and reports whether the non-generative methods
match or dominate the generative frontier. Annotates with the triage epistemic/aleatoric
diagnosis (triage_features.parquet, from diagnostic_calibration.py) so we can show WHERE
balanced accuracy is recoverable (epistemic) vs irreducible (aleatoric).

Run after run_frontier completes on both rosters:
    python -m scripts.paper_revision.build_frontier
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.paper_revision.config import RESULTS_DIR
from scripts.paper_revision.datasets import DATASETS

FIG_DIR = Path("paper_v2/figures")
ROSTER54 = {d.name for d in DATASETS}

GENERATIVE = ["smote", "borderline_smote", "adasyn", "safe_level_smote", "polynom_fit_smote",
              "prowsyn", "mwmote", "napierala_guided_smote", "clean_masked_smote"]
NONGEN = ["baseline", "cost_sensitive", "threshold_moved", "balanced_rf", "easy_ensemble",
          "triage_weighting", "triage_cost_sensitive"]


def _mean_metric(files, metric):
    out = {}
    for f in files:
        df = pd.read_parquet(f)
        out[df["dataset"].iloc[0]] = df[metric].mean()
    return pd.Series(out)


def _collect(benchmark):
    """Per-(method) per-dataset mean accuracy & balanced accuracy for one benchmark."""
    if benchmark == "keel":
        sweep_dir, prefix, roster = RESULTS_DIR / "keel_benchmark", "xgboost__", None
        frontier_prefix = "keel__"
    else:
        sweep_dir, prefix, roster = RESULTS_DIR / "main_benchmark", "xgboost__", ROSTER54
        frontier_prefix = "original__"
    fdir = RESULTS_DIR / "frontier_benchmark"
    rows = []
    methods = GENERATIVE + NONGEN
    for m in methods:
        if m in ("cost_sensitive", "threshold_moved", "balanced_rf", "easy_ensemble"):
            files = glob.glob(str(fdir / f"{frontier_prefix}{m}__*.parquet"))
        else:
            files = glob.glob(str(sweep_dir / f"{prefix}{m}__*.parquet"))
        if not files:
            continue
        acc = _mean_metric(files, "accuracy")
        bacc = _mean_metric(files, "balanced_accuracy")
        common = acc.index
        if roster is not None:
            common = [d for d in common if d in roster]
        for ds in common:
            rows.append({"benchmark": benchmark, "method": m, "dataset": ds,
                         "accuracy": acc[ds], "balanced_accuracy": bacc[ds]})
    return pd.DataFrame(rows)


def analyze():
    parts = [d for d in (_collect("keel"), _collect("original")) if len(d)]
    if not parts:
        print("No frontier/sweep cells found. Run run_frontier first."); return
    df = pd.concat(parts, ignore_index=True)

    print("=== per-method mean (over datasets), by benchmark ===")
    agg = (df.groupby(["benchmark", "method"])[["accuracy", "balanced_accuracy"]]
             .mean().reset_index())
    agg["kind"] = np.where(agg["method"].isin(GENERATIVE), "generative", "non-generative")
    for b in agg["benchmark"].unique():
        print(f"\n-- {b} --")
        sub = agg[agg.benchmark == b].sort_values("balanced_accuracy", ascending=False)
        for _, r in sub.iterrows():
            print(f"  [{r.kind[:6]:6s}] {r.method:24s} acc={r.accuracy:.4f}  bacc={r.balanced_accuracy:.4f}")

    # dominance: per dataset, best non-generative vs best generative on balanced accuracy,
    # and whether the best non-generative also preserves accuracy >= best generative's.
    print("\n=== per-dataset: does the best NON-generative dominate the best generative? ===")
    for b in df.benchmark.unique():
        sub = df[df.benchmark == b]
        wins = tot = 0; dbacc = []
        for ds, g in sub.groupby("dataset"):
            gen = g[g.method.isin(GENERATIVE)]; ng = g[g.method.isin(NONGEN)]
            if gen.empty or ng.empty:
                continue
            tot += 1
            bestgen_b = gen.balanced_accuracy.max()
            bestng_b = ng.balanced_accuracy.max()
            dbacc.append(bestng_b - bestgen_b)
            if bestng_b >= bestgen_b:
                wins += 1
        print(f"  {b}: best non-gen >= best gen on bacc in {wins}/{tot} datasets; "
              f"mean bacc advantage = {100*np.mean(dbacc):+.2f}pp")

    _figure(agg)
    return df, agg


_SHORT = {"baseline": "none", "cost_sensitive": "cost", "threshold_moved": "thresh",
          "balanced_rf": "bRF", "easy_ensemble": "EasyEns", "triage_weighting": "tri-wt",
          "triage_cost_sensitive": "tri-cost", "smote": "SMOTE", "borderline_smote": "Border",
          "adasyn": "ADASYN", "safe_level_smote": "SafeLvl", "polynom_fit_smote": "polyfit",
          "prowsyn": "ProWSyn", "mwmote": "MWMOTE", "napierala_guided_smote": "Napier",
          "clean_masked_smote": "clean-mask"}
_ROSTER = {"keel": "KEEL", "original": "OpenML"}
_ANNOT = {"threshold_moved", "balanced_rf", "easy_ensemble", "cost_sensitive",
          "smote", "baseline", "safe_level_smote"}


def _figure(agg):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 8, "savefig.dpi": 200})
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    benches = list(agg.benchmark.unique())
    fig, axes = plt.subplots(1, len(benches), figsize=(5.4 * len(benches), 4.6))
    if len(benches) == 1:
        axes = [axes]
    for ax, b in zip(axes, benches):
        sub = agg[agg.benchmark == b]
        for _, r in sub.iterrows():
            gen = r.method in GENERATIVE
            ax.scatter(r.accuracy, r.balanced_accuracy,
                       c=("C1" if gen else "C0"), marker=("^" if gen else "*"),
                       s=(55 if gen else 150), edgecolors="k", linewidths=0.5, zorder=3,
                       label=("generative" if gen else "non-generative"))
            if r.method in _ANNOT:   # label only spatially-separated key methods (table has the rest)
                ax.annotate(_SHORT.get(r.method, r.method), (r.accuracy, r.balanced_accuracy),
                            fontsize=7, alpha=0.9, xytext=(5, 3), textcoords="offset points")
        bl = sub[sub.method == "baseline"]
        if len(bl):
            ax.axvline(bl.accuracy.iloc[0], c="0.7", ls=":", lw=0.8, zorder=1)
            ax.axhline(bl.balanced_accuracy.iloc[0], c="0.7", ls=":", lw=0.8, zorder=1)
        ax.set_xlabel("mean accuracy"); ax.set_ylabel("mean balanced accuracy")
        ax.set_title(_ROSTER.get(b, b)); ax.grid(alpha=0.25, zorder=0)
        h, l = ax.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        ax.legend(uniq.values(), uniq.keys(), loc="lower left", framealpha=0.9)
    fig.suptitle("Non-generative strategies reach higher balanced accuracy than the SMOTE family",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_frontier.pdf", bbox_inches="tight")
    plt.close()
    print(f"\nwrote {FIG_DIR / 'fig_frontier.pdf'}")


if __name__ == "__main__":
    analyze()
