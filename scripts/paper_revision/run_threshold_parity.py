#!/usr/bin/env python
"""Threshold-parity + probability-metric benchmark -- RUN MANUALLY.

Reviewer de-risking (the single most important addition). Two gaps in the headline
balanced-accuracy comparison:

  (1) THRESHOLD PARITY. `threshold_moved` tunes its decision threshold on
      out-of-fold train predictions to maximise balanced accuracy; every other
      strategy (SMOTE family, cost-sensitive, balanced ensembles) is scored at the
      default 0.5 threshold. A reviewer can call that "metric-tuned threshold vs
      default threshold", not "non-generative vs generative". Here EVERY strategy
      gets the SAME out-of-fold threshold-tuning opportunity, so we can ask: does
      oversampling still lose after it also receives the threshold optimisation?

  (2) METRICS. The benchmark saved only label-based metrics (acc/bacc/mcc/g-mean).
      We add the threshold-free probability metrics reviewers in this area expect:
      ROC-AUC, PR-AUC (average precision), Brier score + expected calibration
      error (ECE), and recall at fixed false-positive rate (5%, 10%).

Binary datasets only (threshold-moving, PR-AUC, recall@FPR all need a positive
class). Positive class = the minority class.

Menu per cell = (roster, dataset, base_learner):
  base_learner in {xgboost, rf, logreg} x resampler in
      {baseline, smote, borderline_smote, adasyn, safe_level_smote, cost}
  and base_learner == "ensemble" -> strategy in {balanced_rf, easy_ensemble}.
"cost" = class-balanced sample weights on the base learner. Each strategy is
scored at BOTH the default 0.5 threshold and an out-of-fold-tuned threshold
(inner StratifiedKFold; the resampler is applied INSIDE each inner fold, so the
tuned threshold has no test leakage), plus the threshold-free probability metrics.

One parquet PER CELL in results/paper_revision/threshold_parity/, skip-if-exists
resume, per-cell SIGALRM timeout, watchdog hard-exit + resume. Mirrors
run_frontier.py hardening. Stop with Ctrl-C and re-run the same command to resume.

Usage:
    python -m scripts.paper_revision.run_threshold_parity --roster keel     --workers 6
    python -m scripts.paper_revision.run_threshold_parity --roster original --workers 6
    python -m scripts.paper_revision.run_threshold_parity --roster keel --dry-run
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

from scripts.paper_revision.config import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "threshold_parity"
BASE_LEARNERS = ["xgboost", "rf", "logreg"]
RESAMPLERS = ["baseline", "smote", "borderline_smote", "adasyn", "safe_level_smote", "cost"]
ENSEMBLES = ["balanced_rf", "easy_ensemble"]
LEARNER_AXIS = BASE_LEARNERS + ["ensemble"]
CELL_TIMEOUT_S = 3600
N_REPEATS, N_FOLDS = 5, 5
N_INNER = 3            # inner CV folds for the out-of-fold threshold
QUANTILE_GRID = 60     # threshold candidates (matches run_frontier.threshold_moved)


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _out_path(roster, learner, dataset):
    return OUT_DIR / f"{roster}__{learner}__{dataset}.parquet"


# ---------------------------------------------------------------------------
# Roster loaders (reuse run_frontier's so the dataset set is identical)
# ---------------------------------------------------------------------------

def _roster_items(roster):
    from scripts.paper_revision.run_frontier import _roster_items as _ri
    return _ri(roster)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _make_model(learner):
    from scripts.paper_revision.config import XGB_PARAMS, RF_PARAMS, LR_PARAMS
    if learner == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**{**XGB_PARAMS, "n_jobs": 1})
    if learner == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**{**RF_PARAMS, "n_jobs": 1})
    if learner == "logreg":
        # scaled pipeline (matches run_frontier_clf; LR needs scaled inputs)
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(),
                             LogisticRegression(**{**LR_PARAMS, "n_jobs": 1}))
    raise ValueError(learner)


def _make_ensemble(name):
    if name == "balanced_rf":
        from imblearn.ensemble import BalancedRandomForestClassifier
        return BalancedRandomForestClassifier(n_estimators=300, sampling_strategy="all",
                                              replacement=True, bootstrap=False,
                                              random_state=0, n_jobs=1)
    if name == "easy_ensemble":
        from imblearn.ensemble import EasyEnsembleClassifier
        return EasyEnsembleClassifier(n_estimators=10, random_state=0, n_jobs=1)
    raise ValueError(name)


def _resample(resampler, Xtr, ytr, rs):
    """Return (Xr, yr, w) in the SAME (encoded) label space as ytr.

    `ytr` is already integer-encoded {0,1}; the endgame resamplers are
    label-agnostic, so we resample directly in encoded space.
    """
    from scripts.paper_revision.methods import run_method
    if resampler == "baseline":
        return Xtr, ytr, None
    if resampler == "cost":
        Xr, yr, w, _ = run_method("class_balanced_weights", Xtr, ytr, rs)
        return Xr, yr, w
    Xr, yr, w, _ = run_method(resampler, Xtr, ytr, rs)
    return Xr, yr, w


def _fit_proba(model, Xr, yr, w, Xte, pos):
    """Fit `model` on (Xr, yr[, w]) and return P(pos) on Xte."""
    if w is not None:
        if hasattr(model, "steps"):  # sklearn Pipeline -> route weight to final step
            last = model.steps[-1][0]
            model.fit(Xr, yr, **{f"{last}__sample_weight": w})
        else:
            model.fit(Xr, yr, sample_weight=w)
    else:
        model.fit(Xr, yr)
    classes = list(model.classes_)
    col = classes.index(pos) if pos in classes else (len(classes) - 1)
    return model.predict_proba(Xte)[:, col]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _ece(y_pos, p, n_bins=10):
    """Expected calibration error of the positive-class probability."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            continue
        ece += (c / n) * abs(y_pos[m].mean() - p[m].mean())
    return float(ece)


def _recall_at_fpr(y_pos, p, target):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_pos, p)
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def _prob_metrics(yte, pte, pos):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss)
    y_pos = (yte == pos).astype(int)
    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_pos, pte))
    except ValueError:
        out["roc_auc"] = float("nan")
    out["pr_auc"] = float(average_precision_score(y_pos, pte))
    out["brier"] = float(brier_score_loss(y_pos, pte))
    out["ece"] = _ece(y_pos, pte)
    out["recall_fpr05"] = _recall_at_fpr(y_pos, pte, 0.05)
    out["recall_fpr10"] = _recall_at_fpr(y_pos, pte, 0.10)
    return out


def _thr_metrics(yte, pte, pos, neg, tau, suffix):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 matthews_corrcoef, recall_score, precision_score)
    from imblearn.metrics import geometric_mean_score
    yp = np.where(pte >= tau, pos, neg)
    return {
        f"acc_{suffix}": accuracy_score(yte, yp),
        f"bacc_{suffix}": balanced_accuracy_score(yte, yp),
        f"mcc_{suffix}": matthews_corrcoef(yte, yp),
        f"gmean_{suffix}": geometric_mean_score(yte, yp, average="macro", correction=0.001),
        f"recall_pos_{suffix}": recall_score(yte, yp, pos_label=pos, zero_division=0),
        f"precision_pos_{suffix}": precision_score(yte, yp, pos_label=pos, zero_division=0),
    }


def _tune_tau(oof, ytr, pos):
    """Threshold on P(pos) maximising balanced accuracy on out-of-fold train preds."""
    from sklearn.metrics import balanced_accuracy_score
    yb = (ytr == pos).astype(int)
    finite = oof[np.isfinite(oof)]
    if len(finite) == 0:
        return 0.5
    best_tau, best_b = 0.5, -1.0
    for tau in np.unique(np.quantile(finite, np.linspace(0.02, 0.98, QUANTILE_GRID))):
        b = balanced_accuracy_score(yb, (oof >= tau).astype(int))
        if b > best_b:
            best_b, best_tau = b, float(tau)
    return best_tau


# ---------------------------------------------------------------------------
# Per-strategy out-of-fold probabilities for the threshold (resample inside)
# ---------------------------------------------------------------------------

def _oof_proba(make_fn, resampler, Xtr, ytr, pos, seed):
    """Inner-CV out-of-fold P(pos) for every training instance.

    `make_fn` builds a fresh estimator; `resampler` (or None for ensembles) is
    applied to each inner-train fold ONLY, so the OOF probabilities -- and hence
    the tuned threshold -- never see the data they are evaluated on.
    """
    from sklearn.model_selection import StratifiedKFold
    n_splits = max(2, min(N_INNER, int(np.bincount(ytr).min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof = np.full(len(ytr), np.nan, dtype=float)
    for itr, iva in skf.split(Xtr, ytr):
        if resampler is None:
            Xr, yr, w = Xtr[itr], ytr[itr], None
        else:
            Xr, yr, w = _resample(resampler, Xtr[itr], ytr[itr], seed)
        try:
            oof[iva] = _fit_proba(make_fn(), Xr, yr, w, Xtr[iva], pos)
        except Exception:
            oof[iva] = 0.5
    return oof


# ---------------------------------------------------------------------------
# Cell evaluation
# ---------------------------------------------------------------------------

def _strategies_for(learner):
    if learner == "ensemble":
        return [(s, None) for s in ENSEMBLES]            # (strategy, resampler=None)
    return [(r, r) for r in RESAMPLERS]                   # strategy name == resampler


def _make_for(learner, strategy):
    if learner == "ensemble":
        return lambda: _make_ensemble(strategy)
    return lambda: _make_model(learner)


def evaluate(roster, learner, dataset, X, y):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from scripts.paper_revision.cv_runner import _stratified_subsample
    from scripts.paper_revision.config import MAX_INSTANCES, RANDOM_STATE

    X, y = _stratified_subsample(X, y, MAX_INSTANCES, RANDOM_STATE)
    le = LabelEncoder().fit(y)
    y = le.transform(y)
    counts = np.bincount(y)
    pos = int(np.argmin(counts))      # positive = minority
    neg = 1 - pos
    splits = max(2, min(N_FOLDS, int(counts[counts > 0].min())))
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    strategies = _strategies_for(learner)
    rows = []
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        repeat, fold = i // splits, i % splits
        seed = RANDOM_STATE + i
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        for strategy, resampler in strategies:
            try:
                # test-fold probabilities (resampler applied to full train fold)
                if resampler is None:
                    Xr, yr, w = Xtr, ytr, None
                else:
                    Xr, yr, w = _resample(resampler, Xtr, ytr, seed)
                make_fn = _make_for(learner, strategy)
                pte = _fit_proba(make_fn(), Xr, yr, w, Xte, pos)
                # out-of-fold threshold
                oof = _oof_proba(make_fn, resampler, Xtr, ytr, pos, seed)
                tau = _tune_tau(oof, ytr, pos)
                row = {"dataset": dataset, "roster": roster, "base_learner": learner,
                       "strategy": strategy, "repeat": repeat, "fold": fold,
                       "n_train": len(tr), "tuned_tau": tau}
                row.update(_thr_metrics(yte, pte, pos, neg, 0.5, "default"))
                row.update(_thr_metrics(yte, pte, pos, neg, tau, "tuned"))
                row.update(_prob_metrics(yte, pte, pos))
                rows.append(row)
            except Exception:
                rows.append({"dataset": dataset, "roster": roster, "base_learner": learner,
                             "strategy": strategy, "repeat": repeat, "fold": fold,
                             "error": traceback.format_exc(limit=2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Worker + driver (mirrors run_frontier hardening)
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
    roster, learner, dataset = cell
    out = _out_path(roster, learner, dataset)
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
        if len(np.unique(y)) != 2:
            signal.alarm(0)
            return {"cell": cell, "status": "skip", "msg": "non-binary"}
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            df = evaluate(roster, learner, dataset, X, y)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        signal.alarm(0)
        bacc_d = float(df.get("bacc_default", pd.Series(dtype=float)).mean())
        bacc_t = float(df.get("bacc_tuned", pd.Series(dtype=float)).mean())
        return {"cell": cell, "status": "ok", "bacc_d": bacc_d, "bacc_t": bacc_t, "msg": ""}
    except Exception:
        try:
            signal.alarm(0)
        except Exception:
            pass
        return {"cell": cell, "status": "error", "msg": traceback.format_exc()}


def build_cells(roster):
    items = _roster_items(roster)
    return [(roster, lr, n) for lr in LEARNER_AXIS for n, _ in items]


def main():
    import signal as _sig
    try:
        _sig.signal(_sig.SIGHUP, _sig.SIG_IGN)
    except (ValueError, AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser(description="Threshold-parity + probability-metric benchmark.")
    ap.add_argument("--roster", choices=["keel", "original"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    all_cells = build_cells(args.roster)
    pending = [c for c in all_cells if not _out_path(*c).exists()]
    print(f"[{_ts()}] threshold_parity/{args.roster}: {len(all_cells)} cells, "
          f"{len(pending)} pending, {len(all_cells) - len(pending)} done.", flush=True)
    if args.dry_run:
        from collections import Counter
        print("  pending by base_learner:", dict(Counter(c[1] for c in pending)))
        return
    if not pending:
        print("  nothing to do."); return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    err_log = RESULTS_DIR / "threshold_parity_errors.log"
    done = 0
    watchdog = 5400  # cells are minutes-long here; allow a generous no-completion window
    ex = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker)
    futs = {ex.submit(_run_cell, c): c for c in pending}
    remaining = set(futs)
    try:
        while remaining:
            finished, remaining = wait(remaining, timeout=watchdog, return_when=FIRST_COMPLETED)
            if not finished:
                print(f"[{_ts()}] STALL: no cell in {watchdog}s -- hard-exit; re-run to resume.", flush=True)
                sys.stdout.flush(); os._exit(2)
            for fut in finished:
                try:
                    r = fut.result()
                except Exception as exc:
                    print(f"[{_ts()}] pool error: {exc}", flush=True); continue
                done += 1
                roster, learner, dataset = r["cell"]
                label = f"{roster}__{learner}__{dataset}"
                if r["status"] == "ok":
                    print(f"[{_ts()}] ok  ({done}/{len(pending)}) {label}  "
                          f"bacc_default={r['bacc_d']:.4f} bacc_tuned={r['bacc_t']:.4f}", flush=True)
                elif r["status"] == "skip":
                    print(f"[{_ts()}] skip {label}  {r['msg']}", flush=True)
                else:
                    print(f"[{_ts()}] ERR ({done}/{len(pending)}) {label}", flush=True)
                    with open(err_log, "a") as f:
                        f.write(f"[{_ts()}] {label}\n{r['msg']}\n{'='*60}\n")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    print(f"[{_ts()}] done. Errors (if any) in {err_log}", flush=True)


if __name__ == "__main__":
    main()
