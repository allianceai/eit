#!/usr/bin/env python
"""Remedy decomposition v2: multiclass-valid error set + cross-fitted threshold.

Fixes two methodological gaps in build_reducibility.py (v1), raised in external
review of the Neurocomputing revision:

  (1) MULTICLASS ERROR SET. v1 counted a minority instance as a default error
      whenever OOB P(minority) < 0.5. Valid for binary (argmax <=> p >= 0.5) but
      NOT multiclass: an instance can be argmax-CORRECT with p_min < 0.5, and v1
      counted it as an error that a low tau then trivially "recovers" -- inflating
      the multiclass threshold-recoverable fraction (v1: multiclass TR 82% vs
      binary 61%). v2 defines the error set from the ensemble's ACTUAL default
      prediction: argmax over the mean OOB probability vector.

  (2) CROSS-FITTED THRESHOLD. v1 selected tau on the same OOB predictions used
      to score recovery (model-held-out, but not selection-held-out). v2
      cross-fits: stratified K folds over instances; tau_k is selected on the
      other folds' OOB predictions and applied to fold k only. Every reported
      recovery decision uses a tau chosen without that instance.

  The threshold intervention itself is stated multiclass-valid (one-vs-rest):
      predict the focal minority class iff P(minority) >= tau, else argmax over
      the remaining classes.
  A minority error is threshold-recoverable iff P(minority) >= tau (the OvR rule
  then predicts its true class). tau maximises balanced accuracy of the
  minority-vs-rest indicator, as in v1.

  For attribution, v2 also emits the v1 definitions per dataset:
      frac_tr_insample_tau  -- argmax error set, single in-sample tau (isolates
                               the cross-fitting effect)
      frac_tr_v1            -- v1's p<0.5 error set + in-sample tau (reproduces
                               the published number; isolates the error-set effect)

Output: results/paper_revision/reducibility_v2/<benchmark>__<name>.parquet,
combined to reducibility_v2.parquet; prints the summary table (all / IR>3 /
IR>10 / binary / multiclass) for v2-primary and both diagnostics.

    python -m scripts.paper_revision.build_reducibility_v2 --workers 8
    python -m scripts.paper_revision.build_reducibility_v2 --combine-only
    # unweighted-instrument robustness pass (separate out dir):
    python -m scripts.paper_revision.build_reducibility_v2 --noise-mode global --workers 8
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime

import numpy as np
import pandas as pd

from scripts.paper_revision.config import RESULTS_DIR, TRIAGE_PARAMS, MAX_INSTANCES, RANDOM_STATE
from scripts.paper_revision.cv_runner import _stratified_subsample
from scripts.paper_revision.build_reducibility import (
    _best_tau_bacc, _all_items, _CellTimeout, _alarm, _init_worker, _ts,
)

CELL_TIMEOUT_S = 2400
N_TAU_FOLDS = 5


def _dirs(noise_mode, standardize=False):
    suffix = "" if noise_mode == "balanced" else f"_{noise_mode}"
    if standardize:
        suffix += "_std"
    return (RESULTS_DIR / f"reducibility_v2{suffix}",
            RESULTS_DIR / f"reducibility_v2{suffix}.parquet")


def _oob_prob_matrix(triage):
    """Mean out-of-bag probability vector per instance across the triage forests."""
    classes = triage.classes_
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n = len(triage._y_fit)
    acc = np.zeros((n, len(classes))); cnt = np.zeros(n)
    for rf in triage.forests_:
        oob = rf.oob_decision_function_          # (n, n_forest_classes), NaN if no OOB
        has = ~np.isnan(oob[:, 0])
        fmap = np.array([class_to_idx[c] for c in rf.classes_])
        acc[np.ix_(has, fmap)] += oob[has]
        cnt[has] += 1
    cnt = np.maximum(cnt, 1)
    return acc / cnt[:, None]


def _crossfit_recovered(p_min, is_min, err_mask, seed=0):
    """Per-instance recovery under a tau selected WITHOUT that instance.

    Stratified K folds over (is_min); tau_k comes from the other folds' OOB
    predictions and is applied to fold k. Returns (recovered_mask, mean_tau).
    """
    from sklearn.model_selection import StratifiedKFold
    n = len(p_min)
    recovered = np.zeros(n, dtype=bool)
    taus = []
    n_splits = min(N_TAU_FOLDS, max(2, int(is_min.sum())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, te_idx in skf.split(np.zeros(n), is_min):
        tau_k = _best_tau_bacc(p_min[tr_idx], is_min[tr_idx])
        taus.append(tau_k)
        te = np.zeros(n, dtype=bool); te[te_idx] = True
        recovered |= te & err_mask & (p_min >= tau_k)
    return recovered, float(np.mean(taus))


def _decompose_v2(name, benchmark, X, y, noise_mode="balanced", triage_overrides=None,
                  standardize=False):
    from endgame.augmentation.error_triage import ErrorTriage
    X, y = _stratified_subsample(X, y, MAX_INSTANCES, RANDOM_STATE)
    if standardize:
        # robustness pass: unit-invariant k-NN geometry (forest signals are
        # scale-invariant; only the class-ratio / noise-neighbour signals change)
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
    y = y.astype(int)
    counts = np.bincount(y)
    minority = int(np.argmin(np.where(counts == 0, counts.max() + 1, counts)))
    params = {**TRIAGE_PARAMS, "noise_mode": noise_mode,
              "random_state": 42, "n_jobs": 1}
    if triage_overrides:
        params.update(triage_overrides)
    t = ErrorTriage(**params).fit(X, y)
    cats = t.categories_

    P = _oob_prob_matrix(t)                       # (n, C) mean OOB probabilities
    classes = t.classes_
    min_col = int(np.where(classes == minority)[0][0])
    p_min = P[:, min_col]
    is_min = (y == minority).astype(int)

    # ---- v2 primary: argmax error set + cross-fitted tau -------------------
    pred_default = classes[np.argmax(P, axis=1)]
    err_argmax = is_min.astype(bool) & (pred_default != y)
    n_err = int(err_argmax.sum())

    recovered_cf, tau_cf = _crossfit_recovered(p_min, is_min, err_argmax,
                                               seed=RANDOM_STATE)
    not_rec = err_argmax & ~recovered_cf
    data_reducible = not_rec & (cats == "data_limited")
    irreducible = not_rec & (cats == "irreducible")
    noise = not_rec & (cats == "noise")
    other = not_rec & ~np.isin(cats, ["data_limited", "irreducible", "noise"])

    def frac(m, denom):
        return float(m.sum() / denom) if denom else 0.0

    # ---- diagnostics: isolate each correction ------------------------------
    tau_is = _best_tau_bacc(p_min, is_min)        # in-sample tau (v1 selection)
    tr_insample = frac(err_argmax & (p_min >= tau_is), n_err)

    err_v1 = is_min.astype(bool) & (p_min < 0.5)  # v1 error set
    n_err_v1 = int(err_v1.sum())
    tr_v1 = frac(err_v1 & (p_min >= tau_is), n_err_v1)

    ir = float(counts[counts > 0].max() / counts[counts > 0].min())
    err_all = cats != "correct"
    n_err_all = max(int(err_all.sum()), 1)
    return {
        # triage category shares over ALL errors (instrument-level; used by the
        # M-sensitivity analysis — unchanged by the remedy-definition fix)
        "frac_errors_cat1": float((cats == "noise").sum() / n_err_all),
        "frac_errors_cat2": float((cats == "data_limited").sum() / n_err_all),
        "frac_errors_cat3": float((cats == "irreducible").sum() / n_err_all),
        "n_errors_all": int(err_all.sum()),
        "dataset": name, "benchmark": benchmark, "n": len(y), "ir": ir,
        "n_classes": len(counts[counts > 0]), "noise_mode": noise_mode,
        "minority_size": int(is_min.sum()),
        # v2 primary
        "n_minority_err_argmax": n_err,
        "tau_crossfit_mean": tau_cf,
        "frac_threshold_recoverable": frac(recovered_cf, n_err),
        "frac_data_reducible": frac(data_reducible, n_err),
        "frac_irreducible": frac(irreducible, n_err),
        "frac_noise": frac(noise, n_err),
        "frac_other_unrecovered": frac(other, n_err),
        # diagnostics
        "tau_insample": float(tau_is),
        "frac_tr_insample_tau": tr_insample,
        "n_minority_err_v1": n_err_v1,
        "frac_tr_v1": tr_v1,
    }


def _out_path(out_dir, b, n):
    return out_dir / f"{b}__{n}.parquet"


def _run_cell(args_tuple):
    cell, noise_mode, standardize = args_tuple
    b, n = cell
    out_dir, _ = _dirs(noise_mode, standardize)
    out = _out_path(out_dir, b, n)
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
            row = _decompose_v2(n, b, X, y, noise_mode=noise_mode,
                                standardize=standardize)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_parquet(out)
        signal.alarm(0)
        return {"cell": cell, "status": "ok", "tr": row["frac_threshold_recoverable"],
                "ne": row["n_minority_err_argmax"]}
    except Exception:
        try:
            signal.alarm(0)
        except Exception:
            pass
        return {"cell": cell, "status": "error", "msg": traceback.format_exc()}


def _paper_roster(df):
    """Filter to the paper's 79-dataset roster (drop OpenML USPS + webpage;
    the KEEL suite's own webpage stays)."""
    return df[~((df.benchmark == "roster") & df.dataset.isin(["USPS", "webpage"]))]


def _table(d):
    """Write tables/table_reducibility.tex from the primary (unweighted) run,
    paper roster, >=5 minority argmax errors."""
    from pathlib import Path
    TAB = Path("paper_v2/tables")
    TAB.mkdir(parents=True, exist_ok=True)
    rows = [("All datasets", d), ("$\\mathrm{IR}>3$", d[d.ir > 3]),
            ("$\\mathrm{IR}>10$", d[d.ir > 10]),
            ("Binary", d[d.n_classes == 2]),
            ("Multiclass", d[d.n_classes > 2])]
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


def _combine(noise_mode, standardize=False, write_table=False):
    out_dir, combined = _dirs(noise_mode, standardize)
    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        print("  nothing to combine."); return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.to_parquet(combined)
    d = df[df.n_minority_err_argmax >= 5]
    print(f"[{_ts()}] combined {len(df)} datasets ({len(d)} with >=5 minority argmax errors)"
          f" -> {combined}")
    subsets = [("All", d), ("IR>3", d[d.ir > 3]), ("IR>10", d[d.ir > 10]),
               ("Binary", d[d.n_classes == 2]), ("Multiclass", d[d.n_classes > 2])]
    print(f"  {'subset':10s} {'n':>4s} {'TR-v2(crossfit)':>16s} {'TR(insample tau)':>17s} "
          f"{'TR-v1(p<.5)':>12s} {'data-red':>9s} {'irred':>7s} {'noise':>7s}")
    for lab, sub in subsets:
        if not len(sub):
            continue
        print(f"  {lab:10s} {len(sub):>4d} {100*sub.frac_threshold_recoverable.mean():>15.1f}% "
              f"{100*sub.frac_tr_insample_tau.mean():>16.1f}% "
              f"{100*sub.frac_tr_v1.mean():>11.1f}% "
              f"{100*sub.frac_data_reducible.mean():>8.1f}% "
              f"{100*sub.frac_irreducible.mean():>6.1f}% "
              f"{100*sub.frac_noise.mean():>6.1f}%")
    if write_table:
        pr = _paper_roster(df)
        _table(pr[pr.n_minority_err_argmax >= 5])


def main():
    import signal as _sig
    try:
        _sig.signal(_sig.SIGHUP, _sig.SIG_IGN)
    except (ValueError, AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--noise-mode", default="balanced",
                    choices=["balanced", "global", "class_conditional", "protect_minority"],
                    help="Triage ensemble mode; 'global' = unweighted forests "
                         "(instrument-robustness pass, separate output dir).")
    ap.add_argument("--standardize", action="store_true",
                    help="Standardize features before the triage (unit-invariant "
                         "k-NN geometry robustness pass, separate output dir).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Restrict to these dataset names (smoke tests).")
    ap.add_argument("--write-table", action="store_true",
                    help="Also write paper_v2/tables/table_reducibility.tex "
                         "(paper roster; use with the primary --noise-mode global run).")
    args = ap.parse_args()
    if args.combine_only:
        _combine(args.noise_mode, args.standardize, args.write_table); return
    out_dir, _ = _dirs(args.noise_mode, args.standardize)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = [(b, n) for b, n, _ in _all_items()]
    if args.only:
        cells = [c for c in cells if c[1] in set(args.only)]
    pending = [c for c in cells if not _out_path(out_dir, *c).exists()]
    print(f"[{_ts()}] reducibility v2 (noise_mode={args.noise_mode}, "
          f"standardize={args.standardize}): "
          f"{len(cells)} datasets, {len(pending)} pending.", flush=True)
    if pending:
        ex = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker)
        futs = {ex.submit(_run_cell, (c, args.noise_mode, args.standardize)): c
                for c in pending}
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
                              f"TR-v2={100*r['tr']:.0f}% (n_err={r['ne']})", flush=True)
                    elif r["status"] == "error":
                        print(f"[{_ts()}] ERR {r['cell']}\n{r['msg']}", flush=True)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    _combine(args.noise_mode, args.standardize, args.write_table)


if __name__ == "__main__":
    main()
