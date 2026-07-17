#!/usr/bin/env python
"""Remedy-mapped reducibility decomposition of the minority deficit (HEADLINE instrument).

For each dataset we fit the triage ensemble and, using its OUT-OF-BAG predicted
minority probability (held-out by construction), decompose the set of
minority-class instances that are MISCLASSIFIED at the default 0.5 threshold into
three REMEDY classes:

  THRESHOLD-RECOVERABLE : the ensemble ranks the instance correctly -- it is
      classified correctly once the decision threshold is moved to the
      balanced-accuracy-optimal value on out-of-fold predictions. Remedy: move the
      threshold (no data needed). This is the novel axis: it conditions on
      already-wrong-at-0.5 and asks whether a threshold move alone fixes it.
  DATA-REDUCIBLE (epistemic) : not threshold-recoverable, and the local same-class
      evidence is sparse (triage Cat2 / low class-ratio). Remedy: collect/generate
      minority data there.
  IRREDUCIBLE (aleatoric) : not threshold-recoverable and not data-limited --
      genuine class overlap (triage Cat3). Remedy: none.
  (Cat1 noise is reported separately and excluded from the three remedy classes.)

Output: per-dataset fractions of the minority *error* set in each remedy class +
the OOF-optimal threshold, to results/paper_revision/reducibility.parquet, and a
LaTeX summary table. The headline number is the across-dataset mean fraction that
is THRESHOLD-RECOVERABLE -- the mechanistic explanation for why "just move the
threshold" (Elor 2022; Provost 2000) works.

    python -m scripts.paper_revision.build_reducibility --workers 8
    python -m scripts.paper_revision.build_reducibility --combine-only
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

OUT_DIR = RESULTS_DIR / "reducibility"
COMBINED = RESULTS_DIR / "reducibility.parquet"
TAB = Path("paper_v2/tables")
CELL_TIMEOUT_S = 2400


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _oob_minority_prob(triage, minority_label):
    """Mean out-of-bag P(minority class) per instance across the triage forests."""
    n = len(triage._y_fit)
    acc = np.zeros(n); cnt = np.zeros(n)
    for rf in triage.forests_:
        oob = rf.oob_decision_function_              # (n, n_classes), NaN where no OOB
        if minority_label not in list(rf.classes_):
            continue
        col = list(rf.classes_).index(minority_label)
        has = ~np.isnan(oob[:, 0])
        acc[has] += oob[has, col]; cnt[has] += 1
    cnt = np.maximum(cnt, 1)
    return acc / cnt


def _best_tau_bacc(p_minority, is_minority):
    """Threshold on P(minority) maximising balanced accuracy (OOB predictions)."""
    from sklearn.metrics import balanced_accuracy_score
    finite = p_minority[np.isfinite(p_minority)]
    if len(finite) == 0:
        return 0.5
    best_tau, best_b = 0.5, -1.0
    for tau in np.unique(np.quantile(finite, np.linspace(0.02, 0.98, 80))):
        b = balanced_accuracy_score(is_minority, (p_minority >= tau).astype(int))
        if b > best_b:
            best_b, best_tau = b, float(tau)
    return best_tau


def _decompose(name, benchmark, X, y, triage_overrides=None):
    from endgame.augmentation.error_triage import ErrorTriage
    X, y = _stratified_subsample(X, y, MAX_INSTANCES, RANDOM_STATE)
    y = y.astype(int)
    counts = np.bincount(y)
    minority = int(np.argmin(np.where(counts == 0, counts.max() + 1, counts)))
    params = {**TRIAGE_PARAMS, "noise_mode": "balanced",
              "random_state": 42, "n_jobs": 1}
    if triage_overrides:
        params.update(triage_overrides)
    t = ErrorTriage(**params).fit(X, y)
    cats = t.categories_
    p_min = _oob_minority_prob(t, minority)
    is_min = (y == minority).astype(int)
    tau = _best_tau_bacc(p_min, is_min)

    # minority instances misclassified at the DEFAULT 0.5 threshold (OOB argmax wrong)
    err_default = is_min.astype(bool) & (p_min < 0.5)
    n_err = int(err_default.sum())

    recovered = err_default & (p_min >= tau)                 # fixed by the threshold move
    not_rec = err_default & ~(p_min >= tau)
    data_reducible = not_rec & (cats == "data_limited")
    irreducible = not_rec & (cats == "irreducible")
    noise = not_rec & (cats == "noise")
    other = not_rec & ~np.isin(cats, ["data_limited", "irreducible", "noise"])

    def frac(m):
        return float(m.sum() / n_err) if n_err else 0.0

    ir = float(counts[counts > 0].max() / counts[counts > 0].min())
    err_all = cats != "correct"
    n_err_all = max(int(err_all.sum()), 1)
    return {
        "frac_errors_cat1": float((cats == "noise").sum() / n_err_all),
        "frac_errors_cat2": float((cats == "data_limited").sum() / n_err_all),
        "frac_errors_cat3": float((cats == "irreducible").sum() / n_err_all),
        "n_errors_all": int(err_all.sum()),
        "dataset": name, "benchmark": benchmark, "n": len(y), "ir": ir,
        "n_classes": len(counts[counts > 0]),
        "minority_size": int(is_min.sum()), "n_minority_err_default": n_err,
        "tau": tau,
        "frac_threshold_recoverable": frac(recovered),
        "frac_data_reducible": frac(data_reducible),
        "frac_irreducible": frac(irreducible),
        "frac_noise": frac(noise),
        "frac_other_unrecovered": frac(other),
    }


# --------------------------- roster + worker plumbing ---------------------------

def _all_items():
    from scripts.paper_revision.keel_datasets import KEEL_DATASETS, load_keel
    from scripts.paper_revision.datasets import DATASETS, load_dataset
    items = [("keel", n, (lambda n=n: load_keel(n))) for n in KEEL_DATASETS]
    items += [("roster", d.name, (lambda d=d: load_dataset(d))) for d in DATASETS]
    return items


def _out_path(b, n):
    return OUT_DIR / f"{b}__{n}.parquet"


class _CellTimeout(Exception):
    pass


def _alarm(s, f):
    raise _CellTimeout()


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"


def _run_cell(cell):
    b, n = cell
    out = _out_path(b, n)
    if out.exists():
        return {"cell": cell, "status": "skip"}
    import signal
    try:
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(CELL_TIMEOUT_S)
    except (ValueError, AttributeError):
        pass
    try:
        loader = dict(((bb, nn), l) for bb, nn, l in _all_items())[cell]
        X, y = loader()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            row = _decompose(n, b, X, y)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_parquet(out)
        signal.alarm(0)
        return {"cell": cell, "status": "ok", "tr": row["frac_threshold_recoverable"],
                "ne": row["n_minority_err_default"]}
    except Exception:
        try:
            signal.alarm(0)
        except Exception:
            pass
        return {"cell": cell, "status": "error", "msg": traceback.format_exc()}


def _combine():
    files = sorted(OUT_DIR.glob("*.parquet"))
    if not files:
        print("  nothing to combine."); return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.to_parquet(COMBINED)
    # exclude degenerate datasets (no minority errors at default) from the headline mean
    d = df[df.n_minority_err_default >= 5]
    print(f"[{_ts()}] combined {len(df)} datasets ({len(d)} with >=5 minority errors) -> {COMBINED}")
    for col, lab in [("frac_threshold_recoverable", "threshold-recoverable"),
                     ("frac_data_reducible", "data-reducible (epistemic)"),
                     ("frac_irreducible", "irreducible (aleatoric)"),
                     ("frac_noise", "noise"),
                     ("frac_other_unrecovered", "other-unrecovered")]:
        print(f"  {lab:30s} mean={100*d[col].mean():5.1f}%  median={100*d[col].median():5.1f}%")
    # imbalanced subset (IR>3)
    di = d[d.ir > 3]
    print(f"  [IR>3, n={len(di)}] threshold-recoverable mean={100*di.frac_threshold_recoverable.mean():.1f}%")
    _table(d)


def _table(d):
    TAB.mkdir(parents=True, exist_ok=True)
    rows = [("All datasets", d), ("$\\mathrm{IR}>3$", d[d.ir > 3]),
            ("$\\mathrm{IR}>10$", d[d.ir > 10]), ("Binary", d[d.n_classes == 2])]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Subset & threshold-recoverable & data-reducible & irreducible & noise \\",
             r"\midrule"]
    for lab, sub in rows:
        if len(sub) == 0:
            continue
        lines.append(f"{lab} & {100*sub.frac_threshold_recoverable.mean():.0f}\\% & "
                     f"{100*sub.frac_data_reducible.mean():.0f}\\% & "
                     f"{100*sub.frac_irreducible.mean():.0f}\\% & "
                     f"{100*sub.frac_noise.mean():.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "table_reducibility.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {TAB/'table_reducibility.tex'}")


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
    print(f"[{_ts()}] reducibility: {len(cells)} datasets, {len(pending)} pending.", flush=True)
    if pending:
        ex = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker)
        futs = {ex.submit(_run_cell, c): c for c in pending}
        remaining = set(futs); done = 0
        try:
            while remaining:
                fin, remaining = wait(remaining, timeout=3000, return_when=FIRST_COMPLETED)
                if not fin:
                    print(f"[{_ts()}] STALL -- re-run to resume.", flush=True); os._exit(2)
                for fut in fin:
                    r = fut.result(); done += 1
                    if r["status"] == "ok":
                        print(f"[{_ts()}] ok ({done}/{len(pending)}) {r['cell'][1]}  "
                              f"thresh-recov={100*r['tr']:.0f}% (n_err={r['ne']})", flush=True)
                    elif r["status"] == "error":
                        print(f"[{_ts()}] ERR {r['cell']}\n{r['msg']}", flush=True)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    _combine()


if __name__ == "__main__":
    main()
