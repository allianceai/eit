#!/usr/bin/env python
"""Frontier benchmark — RUN MANUALLY. Non-generative methods vs the SMOTE family.

Adds the NON-GENERATIVE methods on both rosters so build_frontier.py can place them on
the accuracy/balanced-accuracy plane against the already-run SMOTE family + baseline:
  - cost_sensitive   : XGBoost + class-balanced sample weights (any task)
  - threshold_moved  : XGBoost + decision threshold chosen on out-of-fold TRAIN predictions
                       to maximise balanced accuracy (BINARY only; no test leakage)
  - balanced_rf      : BalancedRandomForest (undersampling ensemble; any task)
  - easy_ensemble    : EasyEnsemble (undersampling ensemble; BINARY only)

Tests the thesis: non-generative methods match/dominate the generative frontier WITHOUT the
accuracy cost (which comes from boundary generation). The SMOTE family + baseline are already
on disk (keel_benchmark / main_benchmark) and are NOT re-run here.

One parquet PER CELL in results/paper_revision/frontier_benchmark/, skip-if-exists resume,
per-cell SIGALRM timeout (a hung cell self-aborts; the pool never deadlocks). Stop with
Ctrl-C and re-run the same command to continue.

Usage:
    python -m scripts.paper_revision.run_frontier --roster keel     --workers 4
    python -m scripts.paper_revision.run_frontier --roster original --workers 6
    python -m scripts.paper_revision.run_frontier --roster keel --dry-run
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.paper_revision.config import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "frontier_benchmark"
METHODS = ["cost_sensitive", "threshold_moved", "balanced_rf", "easy_ensemble"]
BINARY_ONLY = {"threshold_moved", "easy_ensemble"}
CELL_TIMEOUT_S = 900
N_REPEATS, N_FOLDS = 5, 5


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _out_path(roster, method, dataset):
    return OUT_DIR / f"{roster}__{method}__{dataset}.parquet"


# ---------------------------------------------------------------------------
# Roster loaders
# ---------------------------------------------------------------------------

def _roster_items(roster):
    """Return list of (name, loader_callable). Loaders return (X, y)."""
    if roster == "keel":
        from scripts.paper_revision.keel_datasets import KEEL_DATASETS, load_keel
        return [(n, (lambda n=n: load_keel(n))) for n in KEEL_DATASETS]
    if roster == "original":
        from scripts.paper_revision.datasets import DATASETS, load_dataset
        skip = {"USPS", "webpage"}  # see run_parallel.SWEEP_EXCLUDE
        return [(d.name, (lambda d=d: load_dataset(d))) for d in DATASETS if d.name not in skip]
    raise ValueError(roster)


# ---------------------------------------------------------------------------
# Per-method CV evaluation
# ---------------------------------------------------------------------------

def _metrics(yte, yp):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 matthews_corrcoef)
    from imblearn.metrics import geometric_mean_score
    return dict(
        accuracy=accuracy_score(yte, yp),
        balanced_accuracy=balanced_accuracy_score(yte, yp),
        f1_macro=f1_score(yte, yp, average="macro", zero_division=0),
        mcc=matthews_corrcoef(yte, yp),
        g_mean=geometric_mean_score(yte, yp, average="macro", correction=0.001),
    )


def _fit_predict(method, Xtr, ytr, Xte):
    """Train `method` on (Xtr, ytr) and return predictions on Xte (original label space)."""
    from sklearn.preprocessing import LabelEncoder
    from scripts.paper_revision.config import XGB_PARAMS
    le = LabelEncoder().fit(ytr)
    ytr_e = le.transform(ytr)
    xgb_params = {**XGB_PARAMS, "n_jobs": 1}

    if method == "cost_sensitive":
        from xgboost import XGBClassifier
        from sklearn.utils.class_weight import compute_sample_weight
        w = compute_sample_weight("balanced", ytr_e).astype(float)
        clf = XGBClassifier(**xgb_params).fit(Xtr, ytr_e, sample_weight=w)
        return le.inverse_transform(clf.predict(Xte))

    if method == "threshold_moved":
        # BINARY only. Choose the decision threshold on OUT-OF-FOLD train predictions
        # (no test leakage), then refit on full train and apply it to the test fold.
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        from sklearn.metrics import balanced_accuracy_score
        pos = 1  # encoded minority/positive class label
        inner = StratifiedKFold(min(3, np.bincount(ytr_e).min()), shuffle=True, random_state=0)
        oof = cross_val_predict(XGBClassifier(**xgb_params), Xtr, ytr_e, cv=inner,
                                method="predict_proba")[:, pos]
        best_tau, best_b = 0.5, -1.0
        for tau in np.unique(np.quantile(oof, np.linspace(0.02, 0.98, 60))):
            b = balanced_accuracy_score(ytr_e, (oof >= tau).astype(int))
            if b > best_b:
                best_b, best_tau = b, tau
        clf = XGBClassifier(**xgb_params).fit(Xtr, ytr_e)
        pte = clf.predict_proba(Xte)[:, pos]
        return le.inverse_transform((pte >= best_tau).astype(int))

    if method == "balanced_rf":
        from imblearn.ensemble import BalancedRandomForestClassifier
        clf = BalancedRandomForestClassifier(n_estimators=300, sampling_strategy="all",
                                             replacement=True, bootstrap=False,
                                             random_state=0, n_jobs=1).fit(Xtr, ytr_e)
        return le.inverse_transform(clf.predict(Xte))

    if method == "easy_ensemble":
        from imblearn.ensemble import EasyEnsembleClassifier
        clf = EasyEnsembleClassifier(n_estimators=10, random_state=0, n_jobs=1).fit(Xtr, ytr_e)
        return le.inverse_transform(clf.predict(Xte))

    raise ValueError(method)


def evaluate(method, X, y, dataset):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from scripts.paper_revision.cv_runner import _stratified_subsample
    from scripts.paper_revision.config import MAX_INSTANCES, RANDOM_STATE
    X, y = _stratified_subsample(X, y, MAX_INSTANCES, RANDOM_STATE)
    counts = np.bincount(np.asarray(y).astype(int))
    splits = max(2, min(N_FOLDS, int(counts[counts > 0].min())))
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    rows = []
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        yp = _fit_predict(method, X[tr], y[tr], X[te])
        rows.append({"dataset": dataset, "method": method, "repeat": i // splits,
                     "fold": i % splits, "n_train": len(tr), **_metrics(y[te], yp)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Worker + driver (mirrors run_keel hardening)
# ---------------------------------------------------------------------------

class _CellTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise _CellTimeout()


def _init_worker():
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"


def _run_cell(cell):
    roster, method, dataset, = cell
    out = _out_path(roster, method, dataset)
    if out.exists():
        return {"cell": cell, "status": "skip", "msg": ""}
    import signal
    try:
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(CELL_TIMEOUT_S)
    except (ValueError, AttributeError):
        pass
    try:
        loader = dict((n, l) for n, l in _roster_items(roster))[dataset]
        X, y = loader()
        if method in BINARY_ONLY and len(np.unique(y)) != 2:
            signal.alarm(0)
            return {"cell": cell, "status": "skip", "msg": "non-binary; method binary-only"}
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            df = evaluate(method, X, y, dataset)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        signal.alarm(0)
        return {"cell": cell, "status": "ok", "acc": float(df.accuracy.mean()),
                "bacc": float(df.balanced_accuracy.mean()), "msg": ""}
    except Exception:
        try:
            signal.alarm(0)
        except Exception:
            pass
        return {"cell": cell, "status": "error", "msg": traceback.format_exc()}


def build_cells(roster):
    items = _roster_items(roster)
    return [(roster, m, n) for m in METHODS for n, _ in items]


def main():
    import signal as _sig
    try:
        _sig.signal(_sig.SIGHUP, _sig.SIG_IGN)  # survive terminal/SSH close (likely cause of the silent 11/108 death)
    except (ValueError, AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser(description="Frontier benchmark (non-generative methods).")
    ap.add_argument("--roster", choices=["keel", "original"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    all_cells = build_cells(args.roster)
    pending = [c for c in all_cells if not _out_path(*c).exists()]
    print(f"[{_ts()}] frontier/{args.roster}: {len(all_cells)} cells, {len(pending)} pending, "
          f"{len(all_cells) - len(pending)} done (skip-if-exists resume).")
    if args.dry_run:
        from collections import Counter
        print("  pending by method:", dict(Counter(c[1] for c in pending)))
        return
    if not pending:
        print("  nothing to do."); return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    err_log = RESULTS_DIR / "frontier_errors.log"
    done = 0
    watchdog = 600  # no-completion window -> hard-exit + resume (cells finish in seconds-to-low-minutes)
    # NOTE: do NOT set max_tasks_per_child — with N fast cells, all workers hit the recycle
    # limit at ~the same task and simultaneous mass-recycling deadlocks ProcessPoolExecutor
    # (observed: keel froze at cell 40=4x10, original at 60=6x10). Memory is not a constraint.
    ex = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker)
    futs = {ex.submit(_run_cell, c): c for c in pending}
    remaining = set(futs)
    try:
        while remaining:
            finished, remaining = wait(remaining, timeout=watchdog, return_when=FIRST_COMPLETED)
            if not finished:
                print(f"[{_ts()}] STALL: no cell in {watchdog}s — hard-exit; re-run to resume.", flush=True)
                sys.stdout.flush(); os._exit(2)
            for fut in finished:
                try:
                    r = fut.result()
                except Exception as exc:
                    print(f"[{_ts()}] pool error: {exc}", flush=True); continue
                done += 1
                roster, method, dataset = r["cell"]
                label = f"{roster}__{method}__{dataset}"
                if r["status"] == "ok":
                    print(f"[{_ts()}] ok  ({done}/{len(pending)}) {label}  "
                          f"acc={r['acc']:.4f} bacc={r['bacc']:.4f}", flush=True)
                elif r["status"] == "skip":
                    print(f"[{_ts()}] skip {label}  {r['msg']}", flush=True)
                else:
                    print(f"[{_ts()}] ERR ({done}/{len(pending)}) {label}", flush=True)
                    with open(err_log, "a") as f:
                        f.write(f"[{_ts()}] {label}\n{r['msg']}\n{'='*60}\n")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    print(f"[{_ts()}] done. Errors (if any) in {err_log}")


if __name__ == "__main__":
    main()
