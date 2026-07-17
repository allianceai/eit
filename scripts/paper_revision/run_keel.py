#!/usr/bin/env python
"""KEEL/HDDT external imbalanced-benchmark sweep — RUN MANUALLY.

External validation on the 27 canonical imbalanced-learn datasets
(`keel_datasets.py`), complementing the original 54-roster headline. Mirrors
`run_parallel` but on the KEEL roster, writing one parquet PER CELL to
`results/paper_revision/keel_benchmark/` with skip-if-exists resume: stop with
Ctrl-C any time and re-run the same command to continue exactly where it left off.

Usage:
    python -m scripts.paper_revision.run_keel --workers 6
    python -m scripts.paper_revision.run_keel --workers 6 --classifiers rf,lgbm,logreg,svm
    python -m scripts.paper_revision.run_keel --dry-run        # count cells only

Progress is printed per cell (timestamp, status, metrics). Pipe to a log and tail it:
    python -m scripts.paper_revision.run_keel --workers 6 \\
        > results/paper_revision/keel_run.log 2>&1 &
    tail -f results/paper_revision/keel_run.log
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.paper_revision.config import RESULTS_DIR
from scripts.paper_revision.keel_datasets import KEEL_DATASETS

OUT_DIR = RESULTS_DIR / "keel_benchmark"

# XGBoost: the full broad-method comparison (matches the original-roster sweep).
XGBOOST_METHODS = [
    "baseline", "smote", "borderline_smote", "adasyn", "safe_level_smote",
    "polynom_fit_smote", "prowsyn", "mwmote",
    "clean_masked_smote", "triage_weighting", "napierala_guided_smote",
    "napierala_weighting_rare", "napierala_weighting_rare_outlier",
    "napierala_weighting_borderline", "napierala_weighting_nonsafe",
    # Imbalance-aware noise-detection variants (KEEL-motivated): the 15 above are
    # skipped on resume; these 6 are the new cells (6 x 27 = 162).
    "clean_masked_class_conditional", "clean_masked_balanced", "clean_masked_protect_minority",
    "triage_weighting_class_conditional", "triage_weighting_balanced", "triage_weighting_protect_minority",
    # The formalized improved weighter: aggressive on learnable minority, not the boundary.
    "triage_cost_sensitive",
    # Neurocomputing R7: modern geometric variant (Douzas & Bacao 2019), authors'
    # maintained implementation. All earlier cells skip on resume; this adds 27 cells.
    "gsmote",
]
# Cross-classifier robustness: the core tradeoff / masking / weighting methods.
ABLATION_METHODS = ["baseline", "smote", "triage_weighting",
                    "clean_masked_smote", "napierala_guided_smote"]


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _out_path(clf: str, method: str, dataset: str) -> Path:
    return OUT_DIR / f"{clf}__{method}__{dataset}.parquet"


def _init_worker():
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"


class _CellTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _CellTimeout()


# Per-cell hard cap (seconds): any cell exceeding this self-aborts and is logged as an
# error so a single pathological cell (e.g. an oversampler stuck on high-dim data)
# cannot deadlock the whole pool. Catches Python-level hangs; combined with
# skip-if-exists resume, the run is robust to stalls.
CELL_TIMEOUT_S = 900


def _run_cell(cell):
    clf, method, dataset = cell
    out_path = _out_path(clf, method, dataset)
    if out_path.exists():
        return {"cell": cell, "status": "skip", "elapsed": 0.0, "msg": ""}

    from scripts.paper_revision.config import XGB_PARAMS, RF_PARAMS, LGBM_PARAMS, LR_PARAMS
    XGB_PARAMS["n_jobs"] = 1; RF_PARAMS["n_jobs"] = 1
    LGBM_PARAMS["n_jobs"] = 1; LR_PARAMS["n_jobs"] = 1

    from scripts.paper_revision.keel_datasets import load_keel
    from scripts.paper_revision.cv_runner import evaluate_method_on_dataset

    import signal
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(CELL_TIMEOUT_S)
    except (ValueError, AttributeError):
        pass  # not in main thread / unsupported — proceed without the alarm

    t0 = time.perf_counter()
    try:
        X, y = load_keel(dataset)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            df = evaluate_method_on_dataset(method, X, y, dataset_name=dataset,
                                            classifier=clf, n_repeats=5, n_folds=5)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        signal.alarm(0)
        return {"cell": cell, "status": "ok", "elapsed": time.perf_counter() - t0,
                "acc": float(df.accuracy.mean()), "bacc": float(df.balanced_accuracy.mean()),
                "msg": ""}
    except _CellTimeout:
        signal.alarm(0)
        return {"cell": cell, "status": "error", "elapsed": time.perf_counter() - t0,
                "msg": f"cell exceeded {CELL_TIMEOUT_S}s hard cap — aborted to protect the pool"}
    except Exception:
        signal.alarm(0)
        return {"cell": cell, "status": "error", "elapsed": time.perf_counter() - t0,
                "msg": traceback.format_exc()}
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass


def build_cells(classifiers):
    cells = [("xgboost", m, d) for m in XGBOOST_METHODS for d in KEEL_DATASETS]
    for clf in classifiers:
        cells += [(clf, m, d) for m in ABLATION_METHODS for d in KEEL_DATASETS]
    return cells


def main():
    ap = argparse.ArgumentParser(description="KEEL external benchmark sweep (resume-safe).")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--classifiers", default="",
                    help="comma-separated extra downstream classifiers "
                         "(e.g. rf,lgbm,logreg,svm); xgboost always runs the full method set")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    extra = [c for c in args.classifiers.split(",") if c]
    all_cells = build_cells(extra)
    pending = [c for c in all_cells if not _out_path(*c).exists()]
    print(f"[{_ts()}] KEEL sweep: {len(all_cells)} cells total, {len(pending)} pending, "
          f"{len(all_cells) - len(pending)} already done (skip-if-exists resume).")
    if args.dry_run:
        from collections import Counter
        by_clf = Counter(c[0] for c in pending)
        print("  pending by classifier:", dict(by_clf))
        return
    if not pending:
        print("  nothing to do — all cells present.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    err_log = RESULTS_DIR / "keel_errors.log"
    done = 0
    # Watchdog: if no cell finishes within this window the pool has deadlocked
    # (e.g. an OOM-killed worker, which the per-cell SIGALRM cannot catch). Hard-exit
    # so the run never hangs indefinitely; re-running resumes (done cells are skipped).
    from concurrent.futures import wait, FIRST_COMPLETED
    watchdog_s = CELL_TIMEOUT_S + 300
    ex = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             max_tasks_per_child=50)
    futs = {ex.submit(_run_cell, c): c for c in pending}
    remaining = set(futs)
    try:
        while remaining:
            finished, remaining = wait(remaining, timeout=watchdog_s,
                                       return_when=FIRST_COMPLETED)
            if not finished:
                print(f"[{_ts()}] STALL: no cell completed in {watchdog_s}s — the worker "
                      f"pool has deadlocked. Hard-exiting; RE-RUN the same command to "
                      f"resume ({done} done this run, skip-if-exists).", flush=True)
                sys.stdout.flush(); sys.stderr.flush()
                os._exit(2)
            for fut in finished:
                try:
                    r = fut.result()
                except Exception as exc:
                    print(f"[{_ts()}] pool error: {exc}", flush=True)
                    continue
                done += 1
                clf, method, dataset = r["cell"]
                label = f"{clf}__{method}__{dataset}"
                if r["status"] == "ok":
                    print(f"[{_ts()}] ok  ({done}/{len(pending)}) {label}  "
                          f"{r['elapsed']:.1f}s  acc={r['acc']:.4f} bacc={r['bacc']:.4f}",
                          flush=True)
                elif r["status"] == "skip":
                    print(f"[{_ts()}] skip {label}", flush=True)
                else:
                    print(f"[{_ts()}] ERR ({done}/{len(pending)}) {label}", flush=True)
                    with open(err_log, "a") as f:
                        f.write(f"[{_ts()}] {label}\n{r['msg']}\n{'='*60}\n")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    print(f"[{_ts()}] done. Errors (if any) in {err_log}")


if __name__ == "__main__":
    main()
