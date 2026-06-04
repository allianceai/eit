#!/usr/bin/env python
"""Regime map: which STRATEGY FAMILY does a dataset need, as a function of its
error-structure + complexity meta-features?

Turns the "five negatives" into a positive prescription. Reads the per-dataset
selections written by meta_selection.py (--menu full) and the meta-feature table,
groups the 16 strategies into 5 families, fits a shallow interpretable decision tree
(meta-features -> oracle-best family), and draws a 2-D regime map.

Families:
  do_nothing       : baseline
  threshold_move   : threshold_moved
  cost_sensitive   : cost_sensitive, triage_weighting
  oversample       : smote, borderline_smote, adasyn, safe_level_smote, prowsyn, mwmote,
                     polynom_fit_smote, clean_masked_smote, clean_masked_balanced,
                     napierala_guided_smote
  balanced_ensemble: balanced_rf, easy_ensemble

Run after: python -m scripts.paper_revision.meta_selection --menu full --metric balanced_accuracy
    python -m scripts.paper_revision.build_regime_map [--metric balanced_accuracy]
"""
from __future__ import annotations
import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.paper_revision.config import RESULTS_DIR

FIG_DIR = Path("paper_v2/figures")

FAMILY = {
    "baseline": "do_nothing",
    "threshold_moved": "threshold_move",
    "cost_sensitive": "cost_sensitive", "triage_weighting": "cost_sensitive",
    "smote": "oversample", "borderline_smote": "oversample", "adasyn": "oversample",
    "safe_level_smote": "oversample", "prowsyn": "oversample", "mwmote": "oversample",
    "polynom_fit_smote": "oversample", "clean_masked_smote": "oversample",
    "clean_masked_balanced": "oversample", "napierala_guided_smote": "oversample",
    "balanced_rf": "balanced_ensemble", "easy_ensemble": "balanced_ensemble",
}
FAM_ORDER = ["do_nothing", "threshold_move", "cost_sensitive", "oversample", "balanced_ensemble"]
TREE_FEATS = ["ir", "F1", "N1", "N3", "minority_error_rate", "cat2_mass", "cat3_mass",
              "nap_safe", "nap_outlier", "n", "n_classes"]


def _family_best(row, method_cols):
    """For one dataset row, the best realized metric per family (max over family members)."""
    out = {}
    for fam in FAM_ORDER:
        vals = [row[f"m_{m}"] for m, f in FAMILY.items() if f == fam and f"m_{m}" in method_cols
                and not pd.isna(row.get(f"m_{m}", np.nan))]
        out[fam] = max(vals) if vals else np.nan
    return out


def analyze(metric):
    sel_path = RESULTS_DIR / f"meta_selection_full_{metric}.parquet"
    if not sel_path.exists():
        print(f"missing {sel_path}; run meta_selection --menu full --metric {metric} first.")
        return
    sel = pd.read_parquet(sel_path)
    feats = pd.read_parquet(RESULTS_DIR / "meta_features.parquet").drop_duplicates(["dataset", "benchmark"])
    df = sel.merge(feats, on=["dataset", "benchmark"], how="left", suffixes=("", "_f"))
    method_cols = [c for c in sel.columns if c.startswith("m_")]

    # oracle-best family per dataset + family-level realized
    fam_best = df.apply(lambda r: _family_best(r, method_cols), axis=1, result_type="expand")
    df["oracle_family"] = df["oracle_method"].map(FAMILY)
    df["best_family"] = fam_best[FAM_ORDER].idxmax(axis=1)

    print(f"=== regime map | metric={metric} | {len(df)} datasets ===")
    print("\noracle-best FAMILY distribution:")
    print(df["best_family"].value_counts().reindex(FAM_ORDER).fillna(0).astype(int).to_string())

    # family-level headroom: best-per-dataset family vs best fixed family
    fam_means = fam_best[FAM_ORDER].mean()
    print("\nmean realized by family (over datasets where applicable):")
    print((100 * fam_means).round(2).to_string())
    base = fam_best["do_nothing"]
    print("\nfamily mean advantage over do_nothing (pp):")
    print((100 * (fam_best[FAM_ORDER].sub(base, axis=0)).mean()).round(2).to_string())

    # interpretable tree: meta-features -> best family
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import cross_val_score, LeaveOneOut
    from sklearn.preprocessing import LabelEncoder
    feat_cols = [c for c in TREE_FEATS if c in df.columns]
    X = df[feat_cols].fillna(df[feat_cols].median()).values.astype(float)
    y = df["best_family"].astype(str).values
    ye = LabelEncoder().fit_transform(y)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0).fit(X, ye)
    print(f"\n=== regime tree (depth<=3) on {feat_cols} ===")
    print(export_text(tree, feature_names=feat_cols,
                      class_names=list(LabelEncoder().fit(y).classes_)))
    try:
        acc = cross_val_score(DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0),
                              X, ye, cv=LeaveOneOut()).mean()
        mode_acc = float(np.bincount(ye).max()) / len(ye)   # predict-majority baseline
        print(f"LOO family-prediction accuracy: {acc:.3f}  (predict-majority baseline: {mode_acc:.3f})")
    except Exception as e:
        print("LOO eval skipped:", e)

    _figure(df, metric)
    df.to_parquet(RESULTS_DIR / f"regime_map_{metric}.parquet")
    print(f"\nwrote regime_map_{metric}.parquet")
    return df


def _figure(df, metric):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 8, "savefig.dpi": 200})
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = {f: c for f, c in zip(FAM_ORDER, ["0.55", "C0", "C2", "C1", "C3"])}
    marker = {f: m for f, m in zip(FAM_ORDER, ["o", "s", "D", "^", "P"])}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    panels = [("ir", "F1", True, "imbalance ratio (log scale)", "F1 (class overlap; higher = harder)"),
              ("cat3_mass", "cat2_mass", False, "Cat\\,3 (irreducible) error mass",
               "Cat\\,2 (data-limited) error mass")]
    for ax, (xc, yc, logx, xl, yl) in zip(axes, panels):
        for fam in FAM_ORDER:
            s = df[df["best_family"] == fam]
            if len(s):
                ax.scatter(s[xc], s[yc], c=colors[fam], marker=marker[fam],
                           label=fam.replace("_", " "), s=46, alpha=0.85,
                           edgecolors="k", linewidths=0.4, zorder=3)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=0.25, zorder=0)
    axes[0].legend(fontsize=8, title="oracle-best family", framealpha=0.9, loc="best")
    fig.suptitle("Regime map: which strategy family wins (balanced accuracy)", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_regime_map.pdf", bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG_DIR / 'fig_regime_map.pdf'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="balanced_accuracy")
    analyze(ap.parse_args().metric)
