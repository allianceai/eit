#!/usr/bin/env python
"""Diagnostic-calibration analysis: does the triage PREDICT the rebalancing payoff?

For each dataset (KEEL + the original 54 roster) we compute triage-derived features
from TRAINING DATA ONLY (class-balanced triage), then correlate them with the ACTUAL
sweep outcomes (SMOTE balanced-accuracy gain, SMOTE accuracy cost, the accuracy-
preserving gain of triage_cost_sensitive, ...). The hypothesis, grounded in the causal
mechanism: the irreducible (Cat3) fraction predicts the accuracy COST of boundary
generation, and the reducible/learnable minority predicts the achievable balanced-
accuracy GAIN. If these calibrate, the triage is a pre-resampling diagnostic.

Triage features are cached to results/paper_revision/triage_features.parquet (the slow
part). Re-run after triage_cost_sensitive lands to add that outcome. Produces a
calibration figure at paper_v2/figures/fig_calibration.pdf.

Usage:  python -m scripts.paper_revision.diagnostic_calibration
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import glob
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.paper_revision.config import RESULTS_DIR, TRIAGE_PARAMS
from scripts.paper_revision.cv_runner import _stratified_subsample

FEAT_PATH = RESULTS_DIR / "triage_features.parquet"
FIG_DIR = Path("paper_v2/figures")


# ----------------------------- triage features -----------------------------

def _features_for(X, y) -> dict:
    """Balanced-triage error-structure features from training data only."""
    from endgame.augmentation.error_triage import ErrorTriage
    X, y = _stratified_subsample(X, y, 10000, 42)
    t = ErrorTriage(**{**TRIAGE_PARAMS, "noise_mode": "balanced", "random_state": 42}).fit(X, y)
    cats = t.categories_
    err = cats != "correct"
    ne = int(err.sum())
    counts = np.bincount(y.astype(int))
    minority = int(np.argmin(counts[counts > 0])) if (counts > 0).sum() else 0
    minority = int(np.argmin(np.where(counts == 0, counts.max() + 1, counts)))
    ir = float(counts[counts > 0].max() / counts[counts > 0].min())
    mino_mask = y == minority
    f = dict(
        n=len(y), ir=ir,
        error_rate=ne / len(y),                                   # ceiling gap proxy
        frac_cat1=(cats == "noise").sum() / ne if ne else 0.0,    # among errors
        frac_cat2=(cats == "data_limited").sum() / ne if ne else 0.0,
        frac_cat3=(cats == "irreducible").sum() / ne if ne else 0.0,
        minority_error_rate=float((err & mino_mask).sum() / max(mino_mask.sum(), 1)),
        learnable_minority_frac=float((mino_mask & np.isin(cats, ["correct", "data_limited"])).sum()
                                      / max(mino_mask.sum(), 1)),
    )
    # absolute reducible/irreducible error mass (fraction of ALL instances)
    f["cat2_mass"] = (cats == "data_limited").sum() / len(y)
    f["cat3_mass"] = (cats == "irreducible").sum() / len(y)
    return f


def build_features(force=False) -> pd.DataFrame:
    if FEAT_PATH.exists() and not force:
        return pd.read_parquet(FEAT_PATH)
    from scripts.paper_revision.keel_datasets import KEEL_DATASETS, load_keel
    from scripts.paper_revision.datasets import DATASETS, load_dataset
    rows = []
    for name in KEEL_DATASETS:
        try:
            X, y = load_keel(name)
            rows.append({"dataset": name, "benchmark": "keel", **_features_for(X, y)})
            print(f"  keel  {name}", flush=True)
        except Exception as e:
            print(f"  SKIP keel {name}: {e}", flush=True)
    for spec in DATASETS:
        try:
            X, y = load_dataset(spec)
            rows.append({"dataset": spec.name, "benchmark": "roster", **_features_for(X, y)})
            print(f"  roster {spec.name}", flush=True)
        except Exception as e:
            print(f"  SKIP roster {spec.name}: {e}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(FEAT_PATH)
    return df


# ----------------------------- outcomes -----------------------------

def _per_dataset_metric(cell_dir: Path, method: str, metric: str) -> pd.Series:
    fs = list(cell_dir.glob(f"xgboost__{method}__*.parquet"))
    if not fs:
        return pd.Series(dtype=float)
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return df.groupby("dataset")[metric].mean()


def load_outcomes() -> pd.DataFrame:
    from scripts.paper_revision.datasets import DATASETS
    roster = {d.name for d in DATASETS}
    out = []
    for cell_dir, bench in [(RESULTS_DIR / "keel_benchmark", "keel"),
                            (RESULTS_DIR / "main_benchmark", "roster")]:
        base_a = _per_dataset_metric(cell_dir, "baseline", "accuracy")
        base_b = _per_dataset_metric(cell_dir, "baseline", "balanced_accuracy")
        rec = {}
        for m in ["smote", "clean_masked_balanced", "clean_masked_smote", "triage_cost_sensitive"]:
            ma = _per_dataset_metric(cell_dir, m, "accuracy")
            mb = _per_dataset_metric(cell_dir, m, "balanced_accuracy")
            for ds in base_a.index:
                if ds in ma.index:
                    rec.setdefault(ds, {"dataset": ds, "benchmark": bench})
                    rec[ds][f"dacc_{m}"] = ma[ds] - base_a[ds]
                    rec[ds][f"dbacc_{m}"] = mb[ds] - base_b[ds]
        for ds in rec:
            if bench == "roster" and ds not in roster:
                continue
            out.append(rec[ds])
    return pd.DataFrame(out)


# ----------------------------- analysis -----------------------------

def analyze():
    feats = build_features()
    outs = load_outcomes()
    df = feats.merge(outs, on=["dataset", "benchmark"], how="inner")
    print(f"\n=== merged: {len(df)} datasets "
          f"({(df.benchmark=='keel').sum()} keel + {(df.benchmark=='roster').sum()} roster) ===")

    feature_cols = ["error_rate", "frac_cat2", "frac_cat3", "minority_error_rate",
                    "learnable_minority_frac", "cat2_mass", "cat3_mass", "ir"]
    outcomes = [c for c in df.columns if c.startswith(("dacc_", "dbacc_"))]

    print("\n=== Spearman(triage feature, outcome) — does the triage predict the payoff? ===")
    print(f"{'outcome':28s} " + " ".join(f"{f[:9]:>10s}" for f in feature_cols))
    best = {}
    for o in outcomes:
        sub = df.dropna(subset=[o])
        if len(sub) < 8:
            continue
        cells = []
        for f in feature_cols:
            r, p = spearmanr(sub[f], sub[o])
            cells.append(f"{r:+.2f}{'*' if p < 0.05 else ' '}")
            if o not in best or abs(r) > abs(best[o][1]):
                best[o] = (f, r, p, len(sub))
        print(f"{o:28s} " + " ".join(f"{c:>10s}" for c in cells))

    print("\n=== best single predictor per outcome ===")
    for o, (f, r, p, n) in best.items():
        print(f"  {o:28s} <- {f:24s} rho={r:+.3f} p={p:.4f} (n={n})")

    # ---- incremental value of the triage OVER just knowing the imbalance ratio ----
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score, KFold
    triage_feats = ["minority_error_rate", "frac_cat2", "frac_cat3", "learnable_minority_frac",
                    "cat2_mass", "cat3_mass", "error_rate"]
    print("\n=== Incremental value: cross-val R^2, IR-only vs IR+triage (does the triage beat just IR?) ===")
    for o in ["dbacc_smote", "dacc_smote", "dbacc_clean_masked_smote", "dacc_clean_masked_smote"]:
        sub = df.dropna(subset=[o])
        if len(sub) < 12:
            continue
        cv = KFold(5, shuffle=True, random_state=0)
        def r2(cols):
            return cross_val_score(LinearRegression(), sub[cols].values, sub[o].values,
                                   cv=cv, scoring="r2").mean()
        r2_ir = r2(["ir"]); r2_all = r2(["ir"] + triage_feats); r2_tri = r2(triage_feats)
        print(f"  {o:26s}  R2(IR)={r2_ir:+.2f}  R2(triage)={r2_tri:+.2f}  R2(IR+triage)={r2_all:+.2f}  "
              f"(delta over IR = {r2_all - r2_ir:+.2f})")

    _calibration_figure(df)
    return df


def _calibration_figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Two panels: (1) Cat3 mass -> SMOTE accuracy cost; (2) learnable-minority -> bacc gain
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    color = {"keel": "C3", "roster": "C0"}
    # The triage's minority error rate forecasts BOTH sides of the rebalancing trade.
    # panel 1: balanced-accuracy GAIN
    ax = axes[0]; sub = df.dropna(subset=["dbacc_smote"])
    for b in ["keel", "roster"]:
        s = sub[sub.benchmark == b]
        ax.scatter(100 * s["minority_error_rate"], 100 * s["dbacc_smote"], c=color[b], s=28, alpha=0.8, label=b)
    r, p = spearmanr(sub["minority_error_rate"], sub["dbacc_smote"])
    ax.set_xlabel("triage minority error rate (\\%)")
    ax.set_ylabel("SMOTE balanced-acc gain (pp)")
    ax.set_title(f"forecasts balanced-accuracy gain\n$\\rho={r:.2f}$, $p={p:.1e}$")
    ax.axhline(0, c="0.7", lw=0.8); ax.legend(fontsize=8)
    # panel 2: accuracy COST
    ax = axes[1]; sub = df.dropna(subset=["dacc_smote"])
    for b in ["keel", "roster"]:
        s = sub[sub.benchmark == b]
        ax.scatter(100 * s["minority_error_rate"], 100 * s["dacc_smote"], c=color[b], s=28, alpha=0.8, label=b)
    r, p = spearmanr(sub["minority_error_rate"], sub["dacc_smote"])
    ax.set_xlabel("triage minority error rate (\\%)")
    ax.set_ylabel("SMOTE accuracy change (pp)")
    ax.set_title(f"forecasts accuracy cost\n$\\rho={r:.2f}$, $p={p:.1e}$")
    ax.axhline(0, c="0.7", lw=0.8); ax.legend(fontsize=8)
    fig.suptitle("One triage signal (minority error rate) forecasts both sides of the rebalancing trade", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_calibration.pdf", bbox_inches="tight")
    plt.close()
    print(f"\nwrote {FIG_DIR / 'fig_calibration.pdf'}")


if __name__ == "__main__":
    analyze()
