#!/usr/bin/env python
"""Out-of-fold triage validation -- RUN MANUALLY (reviewer #3).

A skeptical reviewer worries the Error-Instance-Triage categories are assigned
from IN-SAMPLE predictions and are therefore optimistic. (In fact the category
assignment already rests on out-of-bag model signals + data geometry; only the
reported aleatoric/epistemic uncertainties were in-bag.) To remove all doubt we
re-derive the categories under a STRICT inner cross-validation: forests fit on
inner-training folds categorize held-out instances, so every model-derived signal
-- error indicator, true-class probability, aleatoric/epistemic uncertainty, noise
consensus -- is genuinely out-of-fold.

For each dataset we report the in-sample vs out-of-fold:
  - Cat1/2/3 fractions among errors (the "Cat3 ~ 85%" claim),
  - minority error rate,
  - mean aleatoric / epistemic uncertainty per category (the decomposition that
    underwrites "Cat3 = irreducible / Cat2 = data-limited").

Writes one parquet per dataset to results/paper_revision/triage_oof/ then combines
to triage_oof_features.parquet. Resume-safe, per-cell timeout, mirrors run_frontier.

Usage:
    python -m scripts.paper_revision.run_triage_oof --workers 8
    python -m scripts.paper_revision.run_triage_oof --combine-only
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

from scripts.paper_revision.config import RESULTS_DIR, TRIAGE_PARAMS, MAX_INSTANCES, RANDOM_STATE
from scripts.paper_revision.cv_runner import _stratified_subsample

OUT_DIR = RESULTS_DIR / "triage_oof"
COMBINED = RESULTS_DIR / "triage_oof_features.parquet"
CELL_TIMEOUT_S = 2400
N_INNER = 5


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _cat_stats(cats, al, ep, y, prefix):
    """Fractions among errors + per-category mean aleatoric/epistemic."""
    err = cats != "correct"
    ne = int(err.sum())
    counts = np.bincount(y.astype(int))
    minority = int(np.argmin(np.where(counts == 0, counts.max() + 1, counts)))
    mino = y == minority
    out = {
        f"{prefix}_n_err": ne,
        f"{prefix}_frac_cat1": (cats == "noise").sum() / ne if ne else 0.0,
        f"{prefix}_frac_cat2": (cats == "data_limited").sum() / ne if ne else 0.0,
        f"{prefix}_frac_cat3": (cats == "irreducible").sum() / ne if ne else 0.0,
        f"{prefix}_minority_error_rate": float((err & mino).sum() / max(mino.sum(), 1)),
    }
    for cat in ("data_limited", "irreducible"):
        m = cats == cat
        out[f"{prefix}_aleatoric_{cat}"] = float(al[m].mean()) if m.any() else float("nan")
        out[f"{prefix}_epistemic_{cat}"] = float(ep[m].mean()) if m.any() else float("nan")
    return out


def _features_for(name, benchmark, X, y):
    from sklearn.model_selection import StratifiedKFold
    from endgame.augmentation.error_triage import ErrorTriage
    X, y = _stratified_subsample(X, y, MAX_INSTANCES, RANDOM_STATE)
    y = y.astype(int)
    params = {**TRIAGE_PARAMS, "noise_mode": "balanced", "random_state": 42, "n_jobs": 1}

    # in-sample (the reported configuration)
    t = ErrorTriage(**params).fit(X, y)
    row = {"dataset": name, "benchmark": benchmark, "n": len(y)}
    row.update(_cat_stats(t.categories_, t.aleatoric_, t.epistemic_, y, "insample"))

    # strict inner-CV out-of-fold
    oof_cat = np.empty(len(y), dtype=object)
    oof_al = np.zeros(len(y)); oof_ep = np.zeros(len(y))
    n_splits = max(2, min(N_INNER, int(np.bincount(y).min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for itr, iva in skf.split(X, y):
        tt = ErrorTriage(**params).fit(X[itr], y[itr])
        out = tt.categorize_heldout(X[iva], y[iva])
        oof_cat[iva] = out["categories"]
        oof_al[iva] = out["aleatoric"]; oof_ep[iva] = out["epistemic"]
    row.update(_cat_stats(oof_cat, oof_al, oof_ep, y, "oof"))
    return row


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

def _all_items():
    from scripts.paper_revision.keel_datasets import KEEL_DATASETS, load_keel
    from scripts.paper_revision.datasets import DATASETS, load_dataset
    items = [("keel", n, (lambda n=n: load_keel(n))) for n in KEEL_DATASETS]
    items += [("roster", d.name, (lambda d=d: load_dataset(d))) for d in DATASETS]
    return items


def _out_path(benchmark, name):
    return OUT_DIR / f"{benchmark}__{name}.parquet"


class _CellTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise _CellTimeout()


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"


def _run_cell(cell):
    benchmark, name = cell
    out = _out_path(benchmark, name)
    if out.exists():
        return {"cell": cell, "status": "skip"}
    import signal
    try:
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(CELL_TIMEOUT_S)
    except (ValueError, AttributeError):
        pass
    try:
        loader = dict(((b, n), l) for b, n, l in _all_items())[cell]
        X, y = loader()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            row = _features_for(name, benchmark, X, y)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_parquet(out)
        signal.alarm(0)
        return {"cell": cell, "status": "ok",
                "ins": row["insample_frac_cat3"], "oof": row["oof_frac_cat3"]}
    except Exception:
        try:
            signal.alarm(0)
        except Exception:
            pass
        return {"cell": cell, "status": "error", "msg": traceback.format_exc()}


def _combine():
    files = sorted(OUT_DIR.glob("*.parquet"))
    if not files:
        print("  no per-dataset files to combine."); return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.to_parquet(COMBINED)
    print(f"[{_ts()}] combined {len(df)} datasets -> {COMBINED}")
    d = df.dropna(subset=["insample_frac_cat3", "oof_frac_cat3"])
    print(f"  Cat3-among-errors  in-sample mean={d.insample_frac_cat3.mean():.3f}  "
          f"OOF mean={d.oof_frac_cat3.mean():.3f}  "
          f"mean|delta|={np.abs(d.insample_frac_cat3 - d.oof_frac_cat3).mean():.3f}")


def main():
    import signal as _sig
    try:
        _sig.signal(_sig.SIGHUP, _sig.SIG_IGN)
    except (ValueError, AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--combine-only", action="store_true")
    args = ap.parse_args()
    if args.combine_only:
        _combine(); return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = [(b, n) for b, n, _ in _all_items()]
    pending = [c for c in cells if not _out_path(*c).exists()]
    print(f"[{_ts()}] triage_oof: {len(cells)} datasets, {len(pending)} pending.", flush=True)
    if pending:
        err_log = RESULTS_DIR / "triage_oof_errors.log"
        done = 0
        ex = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker)
        futs = {ex.submit(_run_cell, c): c for c in pending}
        remaining = set(futs)
        try:
            while remaining:
                finished, remaining = wait(remaining, timeout=3000, return_when=FIRST_COMPLETED)
                if not finished:
                    print(f"[{_ts()}] STALL -- hard-exit; re-run to resume.", flush=True)
                    sys.stdout.flush(); os._exit(2)
                for fut in finished:
                    r = fut.result(); done += 1
                    b, n = r["cell"]; label = f"{b}__{n}"
                    if r["status"] == "ok":
                        print(f"[{_ts()}] ok ({done}/{len(pending)}) {label}  "
                              f"cat3 insample={r['ins']:.3f} oof={r['oof']:.3f}", flush=True)
                    elif r["status"] == "skip":
                        print(f"[{_ts()}] skip {label}", flush=True)
                    else:
                        print(f"[{_ts()}] ERR ({done}/{len(pending)}) {label}", flush=True)
                        with open(err_log, "a") as f:
                            f.write(f"[{_ts()}] {label}\n{r['msg']}\n{'='*60}\n")
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    _combine()


if __name__ == "__main__":
    main()
