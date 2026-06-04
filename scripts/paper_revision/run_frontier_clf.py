#!/usr/bin/env python
"""Base-learner robustness for the frontier (rebut the 'XGBoost artifact' objection).

Runs the two base-learner-dependent non-generative strategies (cost_sensitive,
threshold_moved) with a NON-XGBoost base learner (random forest or logistic
regression) on the OpenML roster. Combined with the existing rf__/logreg__
SMOTE-family + baseline cells in main_benchmark and the learner-agnostic
balanced_rf/easy_ensemble cells, this lets us show non-generative dominance is
not specific to XGBoost. Single-process, per-cell resume (skip-if-exists).

Writes to results/paper_revision/frontier_benchmark/{classifier}/.
Usage:
  python -m scripts.paper_revision.run_frontier_clf --roster original --classifier rf
  python -m scripts.paper_revision.run_frontier_clf --roster original --classifier logreg
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import signal as _sig
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score

from scripts.paper_revision.config import RESULTS_DIR, MAX_INSTANCES, RANDOM_STATE
from scripts.paper_revision.cv_runner import _stratified_subsample
from scripts.paper_revision.run_frontier import _metrics, _roster_items

METHODS = ["cost_sensitive", "threshold_moved"]
BINARY_ONLY = {"threshold_moved"}
N_REPEATS, N_FOLDS = 5, 5


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _make(clf, cost):
    cw = "balanced" if cost else None
    if clf == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=0, class_weight=cw)
    if clf == "logreg":
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=0, class_weight=cw))
    raise ValueError(clf)


def _fit_predict(method, Xtr, ytr, Xte, clf):
    le = LabelEncoder().fit(ytr)
    ytr_e = le.transform(ytr)
    if method == "cost_sensitive":
        m = _make(clf, cost=True).fit(Xtr, ytr_e)
        return le.inverse_transform(m.predict(Xte))
    if method == "threshold_moved":
        pos = 1
        inner = StratifiedKFold(min(3, int(np.bincount(ytr_e).min())), shuffle=True, random_state=0)
        oof = cross_val_predict(_make(clf, cost=False), Xtr, ytr_e, cv=inner, method="predict_proba")[:, pos]
        best_tau, best_b = 0.5, -1.0
        for tau in np.unique(np.quantile(oof, np.linspace(0.02, 0.98, 60))):
            b = balanced_accuracy_score(ytr_e, (oof >= tau).astype(int))
            if b > best_b:
                best_b, best_tau = b, tau
        m = _make(clf, cost=False).fit(Xtr, ytr_e)
        pte = m.predict_proba(Xte)[:, pos]
        return le.inverse_transform((pte >= best_tau).astype(int))
    raise ValueError(method)


def evaluate(method, X, y, dataset, clf):
    X, y = _stratified_subsample(X, y, MAX_INSTANCES, RANDOM_STATE)
    counts = np.bincount(np.asarray(y).astype(int))
    splits = max(2, min(N_FOLDS, int(counts[counts > 0].min())))
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    rows = []
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        yp = _fit_predict(method, X[tr], y[tr], X[te], clf)
        rows.append({"dataset": dataset, "method": method, "classifier": clf,
                     "repeat": i // splits, "fold": i % splits, **_metrics(y[te], yp)})
    return pd.DataFrame(rows)


def main():
    try:
        _sig.signal(_sig.SIGHUP, _sig.SIG_IGN)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", choices=["keel", "original"], default="original")
    ap.add_argument("--classifier", choices=["rf", "logreg"], required=True)
    args = ap.parse_args()
    out_dir = RESULTS_DIR / "frontier_benchmark" / args.classifier
    out_dir.mkdir(parents=True, exist_ok=True)
    items = _roster_items(args.roster)
    cells = [(m, n, l) for m in METHODS for n, l in items]
    print(f"[{_ts()}] frontier_clf/{args.classifier}/{args.roster}: {len(cells)} cells", flush=True)
    for k, (method, name, loader) in enumerate(cells, 1):
        out = out_dir / f"{args.roster}__{method}__{name}.parquet"
        if out.exists():
            continue
        try:
            X, y = loader()
            if method in BINARY_ONLY and len(np.unique(y)) != 2:
                print(f"[{_ts()}] skip ({k}/{len(cells)}) {method}/{name} (non-binary)", flush=True)
                continue
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):
                df = evaluate(method, X, y, name, args.classifier)
            df.to_parquet(out)
            print(f"[{_ts()}] ok  ({k}/{len(cells)}) {args.classifier}/{method}/{name}  "
                  f"acc={df.accuracy.mean():.4f} bacc={df.balanced_accuracy.mean():.4f}", flush=True)
        except Exception as e:
            print(f"[{_ts()}] ERR ({k}/{len(cells)}) {method}/{name}: {e}", flush=True)
    print(f"[{_ts()}] DONE {args.classifier}/{args.roster}", flush=True)


if __name__ == "__main__":
    main()
