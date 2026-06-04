#!/usr/bin/env python
"""Interventional augmentation experiment (§5.4 causal evidence).

For each dataset:
  1. Fit triage on full training set; get error masks per category.
  2. For each augmentation strategy {augment_cat1, augment_cat2, augment_cat3,
     augment_all_errors, augment_random, remove_random, remove_cat1}:
     generate the modified training set, train XGBoost, evaluate.
  3. Compare to no-augmentation baseline.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from imblearn.metrics import geometric_mean_score
from xgboost import XGBClassifier

from scripts.paper_revision.datasets import DATASETS, load_dataset
from scripts.paper_revision.config import (
    RESULTS_DIR, TRIAGE_PARAMS, XGB_PARAMS, MAX_INSTANCES, N_REPEATS, N_FOLDS, RANDOM_STATE,
)
from scripts.paper_revision.cv_runner import _stratified_subsample
from endgame.augmentation.error_triage import ErrorTriage

OUT = RESULTS_DIR / "interventional.parquet"
STRATEGIES = ["baseline", "augment_cat1", "augment_cat2", "augment_cat3",
              "augment_all_errors", "augment_random", "remove_random", "remove_cat1"]


def _augment_around_seeds(X, y, seed_mask, budget, random_state):
    """Append ``budget`` synthetic points by SMOTE-interpolating the targeted
    error subset (``seed_mask``), with each synthetic point taking the class of
    its seed. Keeps ALL original training data.

    Paper Sec.5.3: "we generate synthetic samples via SMOTE of the targeted error
    subset." Synthetic take the SEED'S OWN class (not a forced minority label),
    and the budget is split across classes in proportion to the seed subset's
    class distribution. This makes ``augment_random`` a proper null control: a
    class-proportional seed set yields class-proportional synthetic, leaving the
    class balance (and thus balanced accuracy) unchanged, while category-targeted
    augmentation concentrates synthetic where those errors actually live.
    Interpolation is within-class (SMOTE-style). Returns the data unchanged when
    budget <= 0 or no class has >=2 seeds.
    """
    from scipy.spatial import KDTree
    seeds = np.where(seed_mask)[0]
    if len(seeds) < 1 or budget <= 0:
        return X, y
    rng = np.random.RandomState(random_state)
    seed_classes = y[seeds]
    classes, counts = np.unique(seed_classes, return_counts=True)
    total = int(counts.sum())

    X_syn_all, y_syn_all = [], []
    for cls, cnt in zip(classes, counts):
        idx_c = seeds[seed_classes == cls]
        n_c = int(round(budget * cnt / total))  # class-proportional allocation
        if len(idx_c) < 2 or n_c <= 0:
            continue
        Xc = X[idx_c]
        k = min(5, len(idx_c) - 1)
        nn = KDTree(Xc).query(Xc, k=k + 1)[1][:, 1:]  # same-class neighbors, drop self
        syn = np.empty((n_c, X.shape[1]))
        for j in range(n_c):
            a = rng.randint(len(idx_c))
            b = nn[a][rng.randint(k)]
            lam = rng.uniform()
            syn[j] = Xc[a] + lam * (Xc[b] - Xc[a])
        X_syn_all.append(syn)
        y_syn_all.append(np.full(n_c, cls, dtype=y.dtype))

    if not X_syn_all:
        return X, y
    return (np.vstack([X, *X_syn_all]),
            np.concatenate([y, *y_syn_all]))


def _modify(X, y, strategy, triage, random_state):
    cat1 = triage.get_category_mask("noise")
    cat2 = triage.get_category_mask("data_limited")
    cat3 = triage.get_category_mask("irreducible")
    err = triage.error_mask_
    rng = np.random.default_rng(random_state)
    # Fixed augmentation budget, identical across strategies, so the comparison
    # isolates WHERE the synthetic points are placed (which error category), not
    # how many are added.
    budget = int(err.sum())

    if strategy == "baseline":
        return X.copy(), y.copy()
    if strategy == "augment_cat1":
        return _augment_around_seeds(X, y, cat1 & err, budget, random_state)
    if strategy == "augment_cat2":
        return _augment_around_seeds(X, y, cat2 & err, budget, random_state)
    if strategy == "augment_cat3":
        return _augment_around_seeds(X, y, cat3 & err, budget, random_state)
    if strategy == "augment_all_errors":
        return _augment_around_seeds(X, y, err, budget, random_state)
    if strategy == "augment_random":
        rand_mask = np.zeros(len(X), dtype=bool)
        rand_mask[rng.choice(len(X), size=min(budget, len(X)), replace=False)] = True
        return _augment_around_seeds(X, y, rand_mask, budget, random_state)
    if strategy == "remove_random":
        n_err = err.sum()
        drop = rng.choice(len(X), size=min(n_err, len(X)//2), replace=False)
        keep = np.setdiff1d(np.arange(len(X)), drop)
        return X[keep], y[keep]
    if strategy == "remove_cat1":
        keep = ~(cat1 & err)
        return X[keep], y[keep]
    raise ValueError(strategy)


def main():
    import sys
    from datetime import datetime
    from rich.progress import track

    def _ts():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for spec in track(DATASETS, description="interventional datasets"):
        try:
            X, y = load_dataset(spec)
            X, y = _stratified_subsample(X, y, MAX_INSTANCES, RANDOM_STATE)
        except Exception as e:
            print(f"[{_ts()}] skip {spec.name}: {e}", file=sys.stderr)
            continue

        rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS,
                                       random_state=RANDOM_STATE)
        for split_i, (tr, te) in enumerate(rskf.split(X, y)):
            seed = RANDOM_STATE + split_i
            Xtr, ytr = X[tr], y[tr]
            triage = ErrorTriage(**{**TRIAGE_PARAMS, "random_state": seed}).fit(Xtr, ytr)
            for strat in STRATEGIES:
                try:
                    Xm, ym = _modify(Xtr, ytr, strat, triage, seed)
                    # XGBoost requires consecutive class labels; pre-encode both
                    # train and test using the union of classes seen in either.
                    all_classes = np.union1d(np.unique(ym), np.unique(y[te]))
                    le = LabelEncoder().fit(all_classes)
                    ym_enc = le.transform(ym)
                    yte_enc = le.transform(y[te])
                    clf = XGBClassifier(**{**XGB_PARAMS, "random_state": seed})
                    clf.fit(Xm, ym_enc)
                    yp = clf.predict(X[te])
                    rows.append({
                        "dataset": spec.name,
                        "strategy": strat,
                        "repeat": split_i // N_FOLDS,
                        "fold": split_i % N_FOLDS,
                        "accuracy": accuracy_score(yte_enc, yp),
                        "balanced_accuracy": balanced_accuracy_score(yte_enc, yp),
                        "f1_macro": f1_score(yte_enc, yp, average="macro", zero_division=0),
                        "mcc": matthews_corrcoef(yte_enc, yp),
                        "g_mean": geometric_mean_score(yte_enc, yp, average="macro", correction=0.001),
                    })
                except Exception as e:
                    print(f"[{_ts()}]   ERROR {spec.name} {strat}: {e}", file=sys.stderr)
        print(f"[{_ts()}] done {spec.name}  rows_so_far={len(rows)}", file=sys.stderr)
        pd.DataFrame(rows).to_parquet(OUT)  # checkpoint

    df = pd.DataFrame(rows)
    df.to_parquet(OUT)

    # Compute mean Δ per strategy for fig1
    base = (df[df.strategy=="baseline"].groupby("dataset")
              [["accuracy","balanced_accuracy","f1_macro","mcc","g_mean"]].mean())
    deltas = []
    for strat in STRATEGIES:
        if strat == "baseline": continue
        m = (df[df.strategy==strat].groupby("dataset")
             [["accuracy","balanced_accuracy","f1_macro","mcc","g_mean"]].mean())
        common = base.index.intersection(m.index)
        for metric in ["accuracy","balanced_accuracy","f1_macro","mcc","g_mean"]:
            d = (m.loc[common, metric] - base.loc[common, metric]).mean()
            deltas.append({"strategy": strat, "metric": metric, "mean_delta": float(d)})
    pd.DataFrame(deltas).to_parquet(RESULTS_DIR / "interventional_deltas.parquet")


if __name__ == "__main__":
    main()
