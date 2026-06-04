#!/usr/bin/env python
"""Interventional causal experiment with OUT-OF-FOLD triage seeds -- RUN MANUALLY.

Robustness re-run for reviewer #3. The published interventional result
(run_interventional.py / interventional.parquet) seeds the targeted augmentation
from triage categories whose model-derived signals are already out-of-bag. Here we
go further: within each outer training fold the seed categories are obtained by a
STRICT inner cross-validation (forests fit on inner-train categorize held-out
inner-val via ErrorTriage.categorize_heldout), so the seeds cannot be an artifact
of in-sample optimism. If `augment_cat3` still reproduces the SMOTE signature
(balanced-accuracy up, accuracy flat/down) while `augment_random` stays null, the
causal claim survives the out-of-fold concern.

To stay tractable with the extra inner-CV cost we use a 3x3 outer protocol (vs the
headline 5x5) -- a robustness check, reported as such. Parallel per-dataset cells,
resume-safe, mirrors run_frontier hardening.

Usage:
    python -m scripts.paper_revision.run_interventional_oof --workers 4
    python -m scripts.paper_revision.run_interventional_oof --combine-only
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.paper_revision.config import (
    RESULTS_DIR, TRIAGE_PARAMS, XGB_PARAMS, MAX_INSTANCES, RANDOM_STATE,
)
from scripts.paper_revision.cv_runner import _stratified_subsample
from scripts.paper_revision.run_interventional import _augment_around_seeds, STRATEGIES

OUT_DIR = RESULTS_DIR / "interventional_oof"
COMBINED = RESULTS_DIR / "interventional_oof.parquet"
N_REPEATS, N_FOLDS, N_INNER = 3, 3, 3
CELL_TIMEOUT_S = 5400


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _oof_masks(Xtr, ytr, seed):
    """Inner-CV out-of-fold triage category + error masks for the training fold."""
    from sklearn.model_selection import StratifiedKFold
    from endgame.augmentation.error_triage import ErrorTriage
    cats = np.empty(len(ytr), dtype=object)
    err = np.zeros(len(ytr), dtype=bool)
    n_splits = max(2, min(N_INNER, int(np.bincount(ytr.astype(int)).min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for itr, iva in skf.split(Xtr, ytr):
        t = ErrorTriage(**{**TRIAGE_PARAMS, "random_state": seed, "n_jobs": 1}).fit(Xtr[itr], ytr[itr])
        out = t.categorize_heldout(Xtr[iva], ytr[iva])
        cats[iva] = out["categories"]
        err[iva] = out["error_mask"]
    return cats, err


def _modify(X, y, strategy, cats, err, random_state):
    cat1 = (cats == "noise")
    cat2 = (cats == "data_limited")
    cat3 = (cats == "irreducible")
    rng = np.random.default_rng(random_state)
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
        rand = np.zeros(len(X), dtype=bool)
        rand[rng.choice(len(X), size=min(budget, len(X)), replace=False)] = True
        return _augment_around_seeds(X, y, rand, budget, random_state)
    if strategy == "remove_random":
        drop = rng.choice(len(X), size=min(int(err.sum()), len(X) // 2), replace=False)
        keep = np.setdiff1d(np.arange(len(X)), drop)
        return X[keep], y[keep]
    if strategy == "remove_cat1":
        keep = ~(cat1 & err)
        return X[keep], y[keep]
    raise ValueError(strategy)


def _evaluate(name, X, y):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 matthews_corrcoef)
    from imblearn.metrics import geometric_mean_score
    from xgboost import XGBClassifier
    X, y = _stratified_subsample(X, y, MAX_INSTANCES, RANDOM_STATE)
    counts = np.bincount(np.asarray(y).astype(int))
    splits = max(2, min(N_FOLDS, int(counts[counts > 0].min())))
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    rows = []
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        seed = RANDOM_STATE + i
        Xtr, ytr = X[tr], y[tr]
        cats, err = _oof_masks(Xtr, ytr, seed)
        for strat in STRATEGIES:
            try:
                Xm, ym = _modify(Xtr, ytr, strat, cats, err, seed)
                all_classes = np.union1d(np.unique(ym), np.unique(y[te]))
                le = LabelEncoder().fit(all_classes)
                clf = XGBClassifier(**{**XGB_PARAMS, "n_jobs": 1, "random_state": seed})
                clf.fit(Xm, le.transform(ym))
                yp = clf.predict(X[te]); yte = le.transform(y[te])
                rows.append({"dataset": name, "strategy": strat,
                             "repeat": i // splits, "fold": i % splits,
                             "accuracy": accuracy_score(yte, yp),
                             "balanced_accuracy": balanced_accuracy_score(yte, yp),
                             "f1_macro": f1_score(yte, yp, average="macro", zero_division=0),
                             "mcc": matthews_corrcoef(yte, yp),
                             "g_mean": geometric_mean_score(yte, yp, average="macro", correction=0.001)})
            except Exception:
                pass
    return pd.DataFrame(rows)


def _items():
    from scripts.paper_revision.datasets import DATASETS, load_dataset
    skip = {"USPS", "webpage"}
    return [(d.name, (lambda d=d: load_dataset(d))) for d in DATASETS if d.name not in skip]


def _out_path(name):
    return OUT_DIR / f"{name}.parquet"


class _CellTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise _CellTimeout()


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"


def _run_cell(name):
    out = _out_path(name)
    if out.exists():
        return {"name": name, "status": "skip"}
    import signal
    try:
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(CELL_TIMEOUT_S)
    except (ValueError, AttributeError):
        pass
    try:
        loader = dict(_items())[name]
        X, y = loader()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            df = _evaluate(name, X, y)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        signal.alarm(0)
        return {"name": name, "status": "ok", "rows": len(df)}
    except Exception:
        try:
            signal.alarm(0)
        except Exception:
            pass
        return {"name": name, "status": "error", "msg": traceback.format_exc()}


def _combine():
    files = sorted(OUT_DIR.glob("*.parquet"))
    if not files:
        print("  nothing to combine."); return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.to_parquet(COMBINED)
    base = df[df.strategy == "baseline"].groupby("dataset")[["accuracy", "balanced_accuracy"]].mean()
    from scipy.stats import wilcoxon
    print(f"[{_ts()}] combined {df.dataset.nunique()} datasets -> {COMBINED}")
    for strat in ["augment_cat3", "augment_cat2", "augment_random", "augment_all_errors"]:
        m = df[df.strategy == strat].groupby("dataset")[["accuracy", "balanced_accuracy"]].mean()
        common = base.index.intersection(m.index)
        db = (m.loc[common, "balanced_accuracy"] - base.loc[common, "balanced_accuracy"])
        da = (m.loc[common, "accuracy"] - base.loc[common, "accuracy"])
        try:
            p = wilcoxon(db).pvalue
        except Exception:
            p = float("nan")
        print(f"  {strat:20s} dBAcc={100*db.mean():+.2f}pp  dAcc={100*da.mean():+.2f}pp  "
              f"WR={100*(db>0).mean():.0f}%  p(bacc)={p:.4g}  (n={len(common)})")


def main():
    import signal as _sig
    try:
        _sig.signal(_sig.SIGHUP, _sig.SIG_IGN)
    except (ValueError, AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--combine-only", action="store_true")
    args = ap.parse_args()
    if args.combine_only:
        _combine(); return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = [n for n, _ in _items()]
    pending = [n for n in names if not _out_path(n).exists()]
    print(f"[{_ts()}] interventional_oof: {len(names)} datasets, {len(pending)} pending.", flush=True)
    if pending:
        err_log = RESULTS_DIR / "interventional_oof_errors.log"
        done = 0
        ex = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker)
        futs = {ex.submit(_run_cell, n): n for n in pending}
        remaining = set(futs)
        try:
            while remaining:
                finished, remaining = wait(remaining, timeout=6000, return_when=FIRST_COMPLETED)
                if not finished:
                    print(f"[{_ts()}] STALL -- hard-exit; re-run to resume.", flush=True)
                    sys.stdout.flush(); os._exit(2)
                for fut in finished:
                    r = fut.result(); done += 1
                    if r["status"] == "ok":
                        print(f"[{_ts()}] ok ({done}/{len(pending)}) {r['name']}  rows={r['rows']}", flush=True)
                    elif r["status"] == "skip":
                        print(f"[{_ts()}] skip {r['name']}", flush=True)
                    else:
                        print(f"[{_ts()}] ERR ({done}/{len(pending)}) {r['name']}", flush=True)
                        with open(err_log, "a") as f:
                            f.write(f"[{_ts()}] {r['name']}\n{r['msg']}\n{'='*60}\n")
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    _combine()


if __name__ == "__main__":
    main()
