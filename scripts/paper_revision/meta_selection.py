#!/usr/bin/env python
"""Meta-selection: prescribe the best imbalance-handling STRATEGY per dataset from
error-structure + complexity meta-features, validated leave-one-dataset-out, against
fixed baselines and an IR-only / separability-only rule (the arXiv:2604.04541 proxy).

PRE-REGISTERED protocol (docs/superpowers/specs/2026-05-31-prescriptive-imbalance-selection-design.md):
  primary selector = per-method RandomForestRegressor(all meta-features) -> argmax predicted
  primary metric   = balanced_accuracy   (secondary: mcc, g_mean; companion: accuracy cost)
  validation       = leave-one-dataset-out (LOO) across datasets
  baselines        = oracle, always-SMOTE, best-fixed, IR-only, separability-only, random
  significance     = paired Wilcoxon across datasets + bootstrap CI on the mean gain

Reads per-dataset, per-method metrics from the benchmark parquets already on disk:
  SMOTE family / baseline / triage  -> {keel,main}_benchmark/xgboost__{method}__{ds}.parquet
  frontier (cost/threshold/BRF/EE)  -> frontier_benchmark/{roster}__{method}__{ds}.parquet
Meta-features from meta_features.parquet if present, else triage_features.parquet.

Usage:
  python -m scripts.paper_revision.meta_selection --menu oversamplers
  python -m scripts.paper_revision.meta_selection --menu full --metric balanced_accuracy
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.paper_revision.config import RESULTS_DIR

OVERSAMPLER_MENU = ["baseline", "smote", "borderline_smote", "adasyn", "safe_level_smote",
                    "prowsyn", "mwmote", "polynom_fit_smote", "clean_masked_smote",
                    "clean_masked_balanced", "napierala_guided_smote", "triage_weighting"]
FRONTIER_METHODS = ["cost_sensitive", "threshold_moved", "balanced_rf", "easy_ensemble"]
FULL_MENU = OVERSAMPLER_MENU + FRONTIER_METHODS

# pre-specified feature sets (the 21-feature 'all' overfits LOO at n~79; 'core' is parsimonious)
CORE_FEATS = ["ir", "minority_error_rate", "cat2_mass", "cat3_mass", "F1", "N1", "nap_safe"]
TRIAGE_ONLY = ["n", "ir", "error_rate", "frac_cat1", "frac_cat2", "frac_cat3",
               "minority_error_rate", "learnable_minority_frac", "cat2_mass", "cat3_mass"]

# columns in the feature table that are identifiers or would leak the outcome
_NON_FEATURES = {"dataset", "benchmark", "baseline_acc", "baseline_bacc",
                 "baseline_accuracy", "baseline_balanced_accuracy"}


# --------------------------- metric lookup ---------------------------

def _cell_mean(path: Path, metric: str) -> float:
    if not path.exists():
        return np.nan
    try:
        return float(pd.read_parquet(path)[metric].mean())
    except Exception:
        return np.nan


def metric_for(dataset: str, benchmark: str, method: str, metric: str) -> float:
    """Mean (over folds) of `metric` for (dataset, method); NaN if the cell is absent."""
    if method in FRONTIER_METHODS:
        roster = "keel" if benchmark == "keel" else "original"
        return _cell_mean(RESULTS_DIR / "frontier_benchmark" / f"{roster}__{method}__{dataset}.parquet", metric)
    bench_dir = "keel_benchmark" if benchmark == "keel" else "main_benchmark"
    return _cell_mean(RESULTS_DIR / bench_dir / f"xgboost__{method}__{dataset}.parquet", metric)


# --------------------------- assemble matrices ---------------------------

def load_feature_table() -> pd.DataFrame:
    mf = RESULTS_DIR / "meta_features.parquet"
    tf = RESULTS_DIR / "triage_features.parquet"
    path = mf if mf.exists() else tf
    print(f"[features] {path.name}")
    return pd.read_parquet(path).drop_duplicates(["dataset", "benchmark"])


def build_matrices(menu, metric):
    """Return (feats_df, B [n_datasets x n_methods], feat_cols, kept_methods).

    B[i, j] = mean `metric` of method j on dataset i (NaN if the cell is absent, e.g.
    a binary-only frontier method on a multiclass dataset)."""
    from scripts.paper_revision.datasets import DATASETS
    roster = {d.name for d in DATASETS}
    feats = load_feature_table()
    rows, Brows = [], []
    for _, r in feats.iterrows():
        ds, bench = r["dataset"], r["benchmark"]
        if bench == "roster" and ds not in roster:
            continue
        vals = [metric_for(ds, bench, m, metric) for m in menu]
        # require the dataset to have the non-binary-only core present (baseline+smote at least)
        if np.isnan(vals[menu.index("baseline")]) or np.isnan(vals[menu.index("smote")]):
            continue
        rows.append(r)
        Brows.append(vals)
    feats_df = pd.DataFrame(rows).reset_index(drop=True)
    B = np.array(Brows, dtype=float)
    # methods present on ALL kept datasets (well-defined "fixed" baselines / oracle core)
    feat_cols = [c for c in feats_df.columns
                 if c not in _NON_FEATURES and pd.api.types.is_numeric_dtype(feats_df[c])]
    return feats_df, B, feat_cols, menu


# --------------------------- selectors ---------------------------

def _loo_regressor_selector(X, B, menu, feat_idx, n_estimators=200, seed=0):
    """Per-method RF regressor on features X[:, feat_idx] -> argmax predicted metric,
    LOO across datasets. NaN-aware: a method is a candidate for dataset i only if B[i,m]
    is observed; each method's regressor trains on datasets where that method is observed."""
    from sklearn.ensemble import RandomForestRegressor
    Xf = X[:, feat_idx]
    N, M = B.shape
    realized = np.full(N, np.nan)
    chosen = np.empty(N, dtype=object)
    for i in range(N):
        cand = [m for m in range(M) if not np.isnan(B[i, m])]
        preds = {}
        for m in cand:
            obs = np.array([j for j in range(N) if j != i and not np.isnan(B[j, m])])
            if len(obs) < 5:
                preds[m] = -np.inf
                continue
            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                                       n_jobs=1).fit(Xf[obs], B[obs, m])
            preds[m] = rf.predict(Xf[i:i + 1])[0]
        sel = max(preds, key=preds.get)
        realized[i] = B[i, sel]
        chosen[i] = menu[sel]
    return realized, chosen


def _loo_classifier_selector(X, B, menu, feat_idx, seed=0):
    """Secondary robustness check: a single RF CLASSIFIER predicts the oracle method."""
    from sklearn.ensemble import RandomForestClassifier
    Xf = X[:, feat_idx]
    N, M = B.shape
    # label = argmax over OBSERVED methods (NaN-safe)
    y = np.array([int(np.nanargmax(B[i])) for i in range(N)])
    realized = np.full(N, np.nan)
    for i in range(N):
        tr = [j for j in range(N) if j != i]
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1).fit(Xf[tr], y[tr])
        pred = int(clf.predict(Xf[i:i + 1])[0])
        if np.isnan(B[i, pred]):                       # predicted an inapplicable method
            pred = int(np.nanargmax([p if not np.isnan(B[i, k]) else -np.inf
                                     for k, p in enumerate(clf.predict_proba(Xf[i:i + 1])[0])]))
        realized[i] = B[i, pred]
    return realized


# --------------------------- significance ---------------------------

def _paired(sel, base):
    """paired Wilcoxon (sel vs base) + mean gain (pp) + bootstrap 95% CI on the mean gain."""
    d = (sel - base)
    d = d[~np.isnan(d)]
    mean_gain = 100 * d.mean()
    try:
        p = wilcoxon(d, zero_method="wilcox", alternative="greater").pvalue if np.any(d != 0) else 1.0
    except Exception:
        p = np.nan
    rng = np.random.default_rng(0)
    boots = [100 * rng.choice(d, len(d), replace=True).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    wr = 100 * np.mean(d > 0)
    return mean_gain, (lo, hi), p, wr


# --------------------------- main ---------------------------

def run(menu_name, metric, features="all"):
    menu = {"oversamplers": OVERSAMPLER_MENU, "full": FULL_MENU}[menu_name]
    feats, B, feat_cols, menu = build_matrices(menu, metric)
    N, M = B.shape
    print(f"\n=== meta-selection | menu={menu_name} ({M} methods) | metric={metric} "
          f"| {N} datasets ({(feats.benchmark=='keel').sum()} keel + {(feats.benchmark=='roster').sum()} roster) ===")
    coverage = {menu[m]: int(np.sum(~np.isnan(B[:, m]))) for m in range(M)}
    print(f"method coverage (datasets with a run cell): {coverage}")
    print(f"selector features ({len(feat_cols)}): {feat_cols}")

    X = feats[feat_cols].values.astype(float)
    chosen_feats = {"all": feat_cols,
                    "core": [c for c in CORE_FEATS if c in feat_cols],
                    "triage": [c for c in TRIAGE_ONLY if c in feat_cols]}[features]
    prim_idx = [feat_cols.index(c) for c in chosen_feats]
    print(f"primary selector feature set '{features}' ({len(prim_idx)}): {chosen_feats}")
    ir_idx = [feat_cols.index("ir")] if "ir" in feat_cols else [0]
    sep_idx = [feat_cols.index(c) for c in ("F1", "N1", "N3") if c in feat_cols]

    # ---- ground-truth references ----
    oracle = np.nanmax(B, axis=1)
    # "best-fixed" must be a strategy applicable to EVERY dataset (a real "always use X"
    # baseline) -> restrict to methods with full coverage; binary-only methods are excluded.
    full_cov = np.array([np.sum(~np.isnan(B[:, m])) == N for m in range(M)])
    masked_means = np.where(full_cov, np.nanmean(B, axis=0), -np.inf)
    best_fixed_idx = int(np.argmax(masked_means))
    best_fixed = B[:, best_fixed_idx]
    smote = B[:, menu.index("smote")]
    baseline = B[:, menu.index("baseline")]
    rand = np.nanmean(B, axis=1)

    # ---- selectors ----
    sel_all, chosen_all = _loo_regressor_selector(X, B, menu, prim_idx)
    sel_ir, _ = _loo_regressor_selector(X, B, menu, ir_idx)
    sel_clf = _loo_classifier_selector(X, B, menu, prim_idx)
    sel_sep = None
    if sep_idx:
        sel_sep, _ = _loo_regressor_selector(X, B, menu, sep_idx)

    def line(name, vec):
        return f"  {name:34s} {np.nanmean(vec):.4f}"

    print("\n-- mean realized {0} over datasets --".format(metric))
    print(line(f"ORACLE (best-per-dataset)", oracle))
    print(line(f"triage/meta selector (LOO)  *PRIMARY*", sel_all))
    print(line(f"RF-classifier selector (LOO)", sel_clf))
    if sel_sep is not None:
        print(line("separability-only selector (LOO)", sel_sep))
    print(line("IR-only selector (LOO)", sel_ir))
    print(line(f"best-fixed ({menu[best_fixed_idx]})", best_fixed))
    print(line("always-SMOTE", smote))
    print(line("always-baseline", baseline))
    print(line("random method (mean)", rand))
    head = float(np.nanmean(oracle) - np.nanmean(best_fixed))
    print(f"  --> oracle headroom over best-fixed = {100*head:+.2f} pp")

    # ---- significance: primary selector vs each baseline ----
    print("\n-- primary selector vs baselines (paired, one-sided 'selector > baseline') --")
    print(f"  {'baseline':22s} {'mean gain(pp)':>13s}  {'95% CI':>16s}  {'p(Wilcoxon)':>11s}  {'win%':>6s}")
    for name, base in [("oracle (ceiling)", oracle), ("best-fixed", best_fixed),
                       ("always-SMOTE", smote), ("IR-only selector", sel_ir),
                       ("always-baseline", baseline)]:
        g, (lo, hi), p, wr = _paired(sel_all, base)
        print(f"  {name:22s} {g:>+12.2f}  [{lo:+.2f},{hi:+.2f}]  {p:>11.4f}  {wr:>5.0f}%")
    if np.nanmean(oracle) > np.nanmean(best_fixed):
        frac = 100 * (np.nanmean(sel_all) - np.nanmean(best_fixed)) / head
        print(f"  fraction of oracle headroom captured by primary selector: {frac:.0f}%")

    # every selector variant vs the two key baselines (does ANY principled selector beat best-fixed?)
    print("\n-- each selector vs best-fixed / always-SMOTE (paired, one-sided 'selector > baseline') --")
    sels = {"primary (regressor)": sel_all, "RF-classifier": sel_clf, "IR-only": sel_ir}
    if sel_sep is not None:
        sels["separability-only"] = sel_sep
    for sname, svec in sels.items():
        capt = 100 * (np.nanmean(svec) - np.nanmean(best_fixed)) / head if head > 0 else float("nan")
        for bname, base in [("best-fixed", best_fixed), ("always-SMOTE", smote)]:
            g, (lo, hi), p, wr = _paired(svec, base)
            tag = " <== beats best-fixed (p<.05)" if (bname == "best-fixed" and p < 0.05) else ""
            print(f"  {sname:20s} vs {bname:12s}: {g:>+6.2f}pp [{lo:+.2f},{hi:+.2f}] p={p:.4f} "
                  f"win={wr:.0f}% headroom={capt:.0f}%{tag}")

    # ---- companion: accuracy cost of the selected strategy (only meaningful when metric!=accuracy)
    if metric != "accuracy":
        accB = np.array([[metric_for(r["dataset"], r["benchmark"], m, "accuracy") for m in menu]
                         for _, r in feats.iterrows()], dtype=float)
        base_acc = accB[:, menu.index("baseline")]
        sel_acc = np.array([accB[i, menu.index(chosen_all[i])] for i in range(N)])
        print(f"\n-- companion accuracy: selected strategy vs baseline = "
              f"{100*np.nanmean(sel_acc-base_acc):+.2f} pp (always-SMOTE: "
              f"{100*np.nanmean(accB[:, menu.index('smote')]-base_acc):+.2f} pp) --")

    # ---- winner distribution + persist per-dataset selections (for the regime map) ----
    winners = pd.Series([menu[int(np.nanargmax(B[i]))] for i in range(N)]).value_counts()
    print(f"\nper-dataset oracle-winner distribution: {dict(winners)}")
    out = feats[["dataset", "benchmark"]].copy()
    out["oracle"] = oracle
    out["oracle_method"] = [menu[int(np.nanargmax(B[i]))] for i in range(N)]
    out["selector_method"] = chosen_all
    out["selector_realized"] = sel_all
    for m in range(M):
        out[f"m_{menu[m]}"] = B[:, m]
    out_path = RESULTS_DIR / f"meta_selection_{menu_name}_{metric}.parquet"
    out.to_parquet(out_path)
    print(f"\nwrote per-dataset selections -> {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu", choices=["oversamplers", "full"], default="oversamplers")
    ap.add_argument("--metric", default="balanced_accuracy")
    ap.add_argument("--features", choices=["all", "core", "triage"], default="all")
    args = ap.parse_args()
    run(args.menu, args.metric, args.features)


if __name__ == "__main__":
    main()
