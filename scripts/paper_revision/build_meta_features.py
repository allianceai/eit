#!/usr/bin/env python
"""Build the meta-feature table for prescriptive strategy selection.

Extends the cached error-structure (triage) features with theory-grounded
data-complexity / separability measures (Lorena et al., "How complex is your
classification problem?") and Napierala minority-type fractions, plus basics
(dimensionality, #classes) and baseline classifier performance.

Output: results/paper_revision/meta_features.parquet  (one row per dataset)

Complexity measures (feature-normalised to [0,1] for the distance-based ones):
  max_fdr : maximum Fisher discriminant ratio over features (higher = more separable)
  F1      : 1/(1+max_fdr)  (Lorena F1; higher = more overlap / harder)
  N1      : fraction of points on the class boundary (MST edges to a different class)
  N2      : intra/inter nearest-neighbour distance ratio, mapped to [0,1)
  N3      : leave-one-out 1-NN error rate
Napierala minority fractions: nap_safe, nap_borderline, nap_rare, nap_outlier.

Usage:  python -m scripts.paper_revision.build_meta_features [--force]
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

from scripts.paper_revision.config import RESULTS_DIR, RANDOM_STATE
from scripts.paper_revision.cv_runner import _stratified_subsample

OUT_PATH = RESULTS_DIR / "meta_features.parquet"
TRIAGE_PATH = RESULTS_DIR / "triage_features.parquet"
COMPLEXITY_CAP = 3000  # subsample for the O(n^2) distance-based measures (MST / 1-NN)


def _clean(X):
    X = np.asarray(X, dtype=float)
    if not np.isfinite(X).all():
        col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        idx = np.where(~np.isfinite(X))
        X[idx] = np.take(col_mean, idx[1])
    return X


def _minmax(X):
    lo = X.min(0); hi = X.max(0); rng = np.where(hi > lo, hi - lo, 1.0)
    return (X - lo) / rng


def _max_fisher_ratio(X, y):
    """Maximum over features of between-class / within-class variance (multiclass FDR)."""
    classes = np.unique(y)
    mu = X.mean(0)
    num = np.zeros(X.shape[1]); den = np.zeros(X.shape[1])
    for c in classes:
        Xc = X[y == c]
        num += len(Xc) * (Xc.mean(0) - mu) ** 2
        den += ((Xc - Xc.mean(0)) ** 2).sum(0)
    den = np.where(den > 0, den, np.nan)
    fdr = num / den
    fdr = fdr[np.isfinite(fdr)]
    return float(fdr.max()) if len(fdr) else 0.0


def _n1_boundary(Xn, y):
    """Fraction of points incident to an MST edge joining two different classes."""
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import minimum_spanning_tree
    D = squareform(pdist(Xn))
    mst = minimum_spanning_tree(D).tocoo()
    on_boundary = np.zeros(len(Xn), dtype=bool)
    for i, j in zip(mst.row, mst.col):
        if y[i] != y[j]:
            on_boundary[i] = on_boundary[j] = True
    return float(on_boundary.mean())


def _n2_n3(Xn, y):
    """N2 (intra/inter NN distance ratio -> [0,1)) and N3 (LOO 1-NN error)."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(len(Xn), 50)).fit(Xn)
    dist, ind = nn.kneighbors(Xn)
    intra = np.full(len(Xn), np.nan); inter = np.full(len(Xn), np.nan)
    n3_err = 0
    for i in range(len(Xn)):
        nb = ind[i, 1:]; dd = dist[i, 1:]
        same = y[nb] == y[i]
        if same.any():
            intra[i] = dd[same][0]
        if (~same).any():
            inter[i] = dd[~same][0]
        n3_err += int(y[nb[0]] != y[i])          # nearest (non-self) neighbour label
    intra_sum = np.nansum(intra); inter_sum = np.nansum(inter)
    r = intra_sum / inter_sum if inter_sum > 0 else 0.0
    n2 = r / (1 + r)
    n3 = n3_err / len(Xn)
    return float(n2), float(n3)


def _napierala_fractions(X, y):
    from endgame.augmentation.napierala_categorizer import NapieralaCategorizer
    cats = NapieralaCategorizer(k=5).fit(X, y).categories_
    minority = cats != "majority"
    nm = max(int(minority.sum()), 1)
    return {f"nap_{t}": float((cats == t).sum()) / nm
            for t in ("safe", "borderline", "rare", "outlier")}


def _complexity_for(X, y):
    X = _clean(X)
    y = np.asarray(y).astype(int)
    Xs, ys = _stratified_subsample(X, y, COMPLEXITY_CAP, RANDOM_STATE)
    Xn = _minmax(Xs)
    max_fdr = _max_fisher_ratio(Xs, ys)
    n1 = _n1_boundary(Xn, ys)
    n2, n3 = _n2_n3(Xn, ys)
    feats = dict(n_features=X.shape[1], n_classes=int(len(np.unique(y))),
                 max_fdr=max_fdr, F1=1.0 / (1.0 + max_fdr), N1=n1, N2=n2, N3=n3)
    feats.update(_napierala_fractions(Xs, ys))
    return feats


# ---------------- baseline performance (from benchmark parquets) ----------------

def _baseline_perf(dataset, benchmark):
    bench_dir = "keel_benchmark" if benchmark == "keel" else "main_benchmark"
    p = RESULTS_DIR / bench_dir / f"xgboost__baseline__{dataset}.parquet"
    if not p.exists():
        return {"baseline_acc": np.nan, "baseline_bacc": np.nan}
    df = pd.read_parquet(p)
    return {"baseline_acc": float(df["accuracy"].mean()),
            "baseline_bacc": float(df["balanced_accuracy"].mean())}


def build(force=False):
    if OUT_PATH.exists() and not force:
        print(f"{OUT_PATH} exists; use --force to rebuild.")
        return pd.read_parquet(OUT_PATH)
    base = pd.read_parquet(TRIAGE_PATH).drop_duplicates(["dataset", "benchmark"])
    from scripts.paper_revision.keel_datasets import load_keel
    from scripts.paper_revision.datasets import DATASETS, load_dataset
    loaders = {("keel", n): (lambda n=n: load_keel(n)) for n in
               base.loc[base.benchmark == "keel", "dataset"]}
    spec_by_name = {d.name: d for d in DATASETS}

    rows = []
    for _, r in base.iterrows():
        ds, bench = r["dataset"], r["benchmark"]
        try:
            if bench == "keel":
                X, y = load_keel(ds)
            else:
                if ds not in spec_by_name:
                    print(f"  SKIP {bench} {ds}: not in active roster", flush=True); continue
                X, y = load_dataset(spec_by_name[ds])
            feats = _complexity_for(X, y)
            feats.update(_baseline_perf(ds, bench))
            rows.append({**r.to_dict(), **feats})
            print(f"  {bench:6s} {ds:28s} F1={feats['F1']:.3f} N1={feats['N1']:.3f} "
                  f"N3={feats['N3']:.3f} nap_safe={feats['nap_safe']:.2f}", flush=True)
        except Exception as e:
            print(f"  ERROR {bench} {ds}: {e}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH)
    print(f"\nwrote {OUT_PATH}  ({len(df)} datasets, {df.shape[1]} columns)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    build(ap.parse_args().force)
