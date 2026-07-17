#!/usr/bin/env python
"""Remedy decomposition with NON-TREE instruments (Neurocomputing R5 / E1).

The headline remedy decomposition (build_reducibility.py) derives every
model-based signal from a class-balanced random-forest ensemble. The reviewer
asks whether the result is an artefact of tree-based uncertainty. Here the
SAME decomposition is re-derived with bagged non-tree ensembles:

  mlp_bag    : B=25 bootstrap-bagged scaled MLPClassifier(64,32) members.
  logreg_bag : B=25 bootstrap-bagged scaled LogisticRegression members.

Members are deliberately UNWEIGHTED (plain bootstrap, no class_weight): the
decomposition targets the default-threshold minority deficit of a default
classifier. Class-balancing the members enacts the operating-point shift
inside the instrument (for logistic regression, completely -- exactly the
weighting result of the parity analysis) and dissolves the deficit being
decomposed. The RF instrument's class_weight="balanced" perturbs leaf
probabilities only mildly, so its deficit survives; an unweighted non-tree
ensemble is the comparable design.

Per instance, all signals are OUT-OF-BAG: only members whose bootstrap did not
contain the instance contribute. Signals mirror the RF instrument:
  - OOB mean class probabilities  -> error set (argmax != y), TCP, P(minority)
  - Shaker-style decomposition over OOB members -> aleatoric / epistemic
  - local class ratio: identical k-NN (k=10) data-geometry computation
Categories use the same thresholds (tau_noise=0.12 on TCP; class_ratio<0.4 with
the TCP<1/n_classes gate for Cat2). DOCUMENTED DIVERGENCE: the RF noise gate
additionally uses a forest-consensus noise score, which has no non-tree
analogue; here Cat1 is gated by TCP alone. This affects only the small noise
share (~7%), not the threshold-recoverable headline.

The remedy decomposition itself (threshold-recoverability at the OOF
balanced-accuracy-optimal tau, then the triage split of the residue) is
identical code to build_reducibility.py.

Outputs one parquet per (instrument, dataset) to
results/paper_revision/reducibility_nontree/, then a combined parquet +
comparison against the RF-instrument results (reducibility.parquet).

    python -m scripts.paper_revision.run_reducibility_nontree --workers 8
    python -m scripts.paper_revision.run_reducibility_nontree --instruments mlp_bag
    python -m scripts.paper_revision.run_reducibility_nontree --combine-only
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

from scripts.paper_revision.config import (RESULTS_DIR, TRIAGE_PARAMS,
                                           MAX_INSTANCES, RANDOM_STATE, MLP_PARAMS)
from scripts.paper_revision.cv_runner import _stratified_subsample
from scripts.paper_revision.build_reducibility import _best_tau_bacc, _all_items

# v2 dirs (post external review: argmax error set + cross-fitted tau).
# v1-definition results remain in reducibility_nontree/ + .parquet.
OUT_DIR = RESULTS_DIR / "reducibility_nontree_v2"
COMBINED = RESULTS_DIR / "reducibility_nontree_v2.parquet"
RF_COMBINED = RESULTS_DIR / "reducibility_v2_global.parquet"
INSTRUMENTS = ["mlp_bag", "logreg_bag"]
N_MEMBERS = 25
CELL_TIMEOUT_S = 2400
TAU_NOISE = TRIAGE_PARAMS["noise_tcp_threshold"]           # 0.12
TAU_CAT2 = TRIAGE_PARAMS["cat2_class_ratio_threshold"]     # 0.4


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _make_member(instrument, seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if instrument == "mlp_bag":
        from sklearn.neural_network import MLPClassifier
        return make_pipeline(StandardScaler(),
                             MLPClassifier(**{**MLP_PARAMS, "random_state": seed}))
    if instrument == "logreg_bag":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=seed))
    raise ValueError(instrument)


def _bootstrap_idx(instrument, y, rng):
    """Plain bootstrap for every instrument (see module docstring: weighted or
    class-balanced members would enact the operating-point shift inside the
    instrument and dissolve the default-threshold deficit being decomposed)."""
    n = len(y)
    return rng.choice(n, size=n, replace=True)


def _oob_signals(instrument, X, y, n_classes):
    """OOB mean probabilities, aleatoric/epistemic (over OOB members), coverage."""
    n = len(y)
    rng = np.random.default_rng(RANDOM_STATE)
    probs = np.full((N_MEMBERS, n, n_classes), np.nan, dtype=float)
    inbag = np.zeros((N_MEMBERS, n), dtype=bool)
    for b in range(N_MEMBERS):
        idx = _bootstrap_idx(instrument, y, rng)
        inbag[b, np.unique(idx)] = True
        m = _make_member(instrument, RANDOM_STATE + b)
        # a bootstrap can miss a class entirely on extreme IR; resample until all present
        tries = 0
        while len(np.unique(y[idx])) < n_classes and tries < 20:
            idx = _bootstrap_idx(instrument, y, rng); tries += 1
        m.fit(X[idx], y[idx])
        p = m.predict_proba(X)
        cols = {c: j for j, c in enumerate(m.classes_)}
        full = np.zeros((n, n_classes))
        for c, j in cols.items():
            full[:, int(c)] = p[:, j]
        probs[b] = full

    oob = ~inbag                                     # (B, n) member b is OOB for i
    eps = 1e-12
    mean_p = np.zeros((n, n_classes))
    aleatoric = np.zeros(n)
    coverage = oob.sum(axis=0)
    for i in range(n):
        mask = oob[:, i]
        if not mask.any():                            # ~0.632^B, essentially never
            mask = np.ones(N_MEMBERS, dtype=bool)
        pi = probs[mask, i, :]
        mean_p[i] = pi.mean(axis=0)
        aleatoric[i] = float((-pi * np.log(pi + eps)).sum(axis=1).mean())
    total = (-mean_p * np.log(mean_p + eps)).sum(axis=1)
    epistemic = total - aleatoric
    return mean_p, aleatoric, epistemic, coverage


def _class_ratio(X, y, k=10, n_jobs=1):
    """Identical semantics to ErrorTriage._compute_class_ratio (k-NN same-class
    fraction normalized by global class frequency)."""
    from sklearn.neighbors import NearestNeighbors
    n = len(X)
    k = min(k, n - 1)
    if k < 1:
        return np.ones(n)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=n_jobs).fit(X)
    _, idx = nn.kneighbors(X)
    same = (y[idx[:, 1:]] == y[:, None]).mean(axis=1)
    freq = {c: max((y == c).mean(), 1e-10) for c in np.unique(y)}
    expected = np.array([freq[yi] for yi in y])
    return same / expected


def _decompose_nontree(instrument, name, benchmark, X, y):
    X = np.asarray(X, dtype=float)
    X, y = _stratified_subsample(X, y, MAX_INSTANCES, RANDOM_STATE)
    y = y.astype(int)
    counts = np.bincount(y)
    n_classes = int((counts > 0).sum())
    # remap labels to 0..n_classes-1 for the prob matrix
    classes = np.flatnonzero(counts)
    remap = {int(c): j for j, c in enumerate(classes)}
    y = np.array([remap[int(v)] for v in y])
    counts = np.bincount(y)
    minority = int(np.argmin(counts))

    mean_p, aleatoric, epistemic, coverage = _oob_signals(instrument, X, y, n_classes)
    ratio = _class_ratio(X, y)

    pred = mean_p.argmax(axis=1)
    err = pred != y
    tcp = mean_p[np.arange(len(y)), y]

    # categories on errors (same thresholds; TCP-only noise gate, see module docstring)
    cats = np.full(len(y), "correct", dtype=object)
    tcp_cat2 = 1.0 / n_classes
    for i in np.flatnonzero(err):
        if tcp[i] < TAU_NOISE:
            cats[i] = "noise"
        elif ratio[i] < TAU_CAT2 and tcp[i] < tcp_cat2:
            cats[i] = "data_limited"
        else:
            cats[i] = "irreducible"

    # remedy decomposition (v2 definitions, identical to build_reducibility_v2:
    # argmax-based multiclass-valid error set + cross-fitted tau)
    from scripts.paper_revision.build_reducibility_v2 import _crossfit_recovered
    p_min = mean_p[:, minority]
    is_min = (y == minority).astype(int)
    tau = _best_tau_bacc(p_min, is_min)          # in-sample tau (diagnostic only)
    err_default = is_min.astype(bool) & err      # argmax-wrong minority instances
    n_err = int(err_default.sum())
    recovered, tau_cf = _crossfit_recovered(p_min, is_min, err_default,
                                            seed=RANDOM_STATE)
    not_rec = err_default & ~recovered

    def frac(m):
        return float(m.sum() / n_err) if n_err else 0.0

    # aleatoric>epistemic sanity: Cat3 errors should carry more aleatoric mass than Cat2
    al_c3 = float(np.nanmean(aleatoric[(cats == "irreducible")])) if (cats == "irreducible").any() else np.nan
    al_c2 = float(np.nanmean(aleatoric[(cats == "data_limited")])) if (cats == "data_limited").any() else np.nan

    ir = float(counts[counts > 0].max() / counts[counts > 0].min())
    return {
        "instrument": instrument, "dataset": name, "benchmark": benchmark,
        "n": len(y), "ir": ir, "n_classes": n_classes,
        "minority_size": int(is_min.sum()), "n_minority_err_default": n_err,
        "tau": tau, "tau_crossfit_mean": tau_cf,
        "oob_coverage_mean": float(coverage.mean()),
        "frac_threshold_recoverable": frac(recovered),
        "frac_data_reducible": frac(not_rec & (cats == "data_limited")),
        "frac_irreducible": frac(not_rec & (cats == "irreducible")),
        "frac_noise": frac(not_rec & (cats == "noise")),
        "frac_other_unrecovered": frac(not_rec & ~np.isin(cats, ["data_limited", "irreducible", "noise"])),
        "aleatoric_cat3_mean": al_c3, "aleatoric_cat2_mean": al_c2,
    }


# --------------------------- worker plumbing (mirrors build_reducibility) ---------------------------

def _out_path(instrument, b, n):
    return OUT_DIR / f"{instrument}__{b}__{n}.parquet"


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"


class _CellTimeout(Exception):
    pass


def _alarm(s, f):
    raise _CellTimeout()


def _run_cell(cell):
    instrument, b, n = cell
    out = _out_path(instrument, b, n)
    if out.exists():
        return {"cell": cell, "status": "skip"}
    import signal
    try:
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(CELL_TIMEOUT_S)
    except (ValueError, AttributeError):
        pass
    try:
        loader = dict(((bb, nn), l) for bb, nn, l in _all_items())[(b, n)]
        X, y = loader()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            row = _decompose_nontree(instrument, n, b, X, y)
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
    d = df[df.n_minority_err_default >= 5]
    print(f"[{_ts()}] combined {len(df)} rows ({len(d)} with >=5 minority errors) -> {COMBINED}")
    for inst in sorted(d.instrument.unique()):
        s = d[d.instrument == inst]
        si = s[s.ir > 3]
        print(f"  [{inst}, n={len(s)}] thresh-recov mean={100*s.frac_threshold_recoverable.mean():.1f}% "
              f"median={100*s.frac_threshold_recoverable.median():.1f}%  "
              f"data-red={100*s.frac_data_reducible.mean():.1f}%  "
              f"irred={100*s.frac_irreducible.mean():.1f}%  noise={100*s.frac_noise.mean():.1f}%  "
              f"| IR>3 thresh-recov={100*si.frac_threshold_recoverable.mean():.1f}% (n={len(si)})")
    if RF_COMBINED.exists():
        rf = pd.read_parquet(RF_COMBINED)
        rf = rf[rf.n_minority_err_argmax >= 5][["dataset", "benchmark", "frac_threshold_recoverable"]]
        rf = rf.rename(columns={"frac_threshold_recoverable": "tr_rf"})
        for inst in sorted(d.instrument.unique()):
            s = d[d.instrument == inst][["dataset", "benchmark", "frac_threshold_recoverable"]].merge(
                rf, on=["dataset", "benchmark"])
            if len(s) > 2:
                r = np.corrcoef(s.frac_threshold_recoverable, s.tr_rf)[0, 1]
                print(f"  per-dataset corr(thresh-recov, RF instrument) [{inst}]: "
                      f"r={r:.2f} over {len(s)} datasets")


def main():
    import signal as _sig
    try:
        _sig.signal(_sig.SIGHUP, _sig.SIG_IGN)
    except (ValueError, AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--instruments", default=",".join(INSTRUMENTS),
                    help=f"comma-separated subset of {INSTRUMENTS}")
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.combine_only:
        _combine(); return
    instruments = [i for i in args.instruments.split(",") if i]
    unknown = set(instruments) - set(INSTRUMENTS)
    if unknown:
        ap.error(f"unknown instruments {sorted(unknown)}; choose from {INSTRUMENTS}")
    cells = [(inst, b, n) for inst in instruments for b, n, _ in _all_items()]
    pending = [c for c in cells if not _out_path(*c).exists()]
    print(f"[{_ts()}] reducibility_nontree: {len(cells)} cells, {len(pending)} pending.", flush=True)
    if args.dry_run:
        return
    if pending:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
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
                    inst, b, n = r["cell"]
                    if r["status"] == "ok":
                        print(f"[{_ts()}] ok ({done}/{len(pending)}) {inst}:{n}  "
                              f"thresh-recov={100*r['tr']:.0f}% (n_err={r['ne']})", flush=True)
                    elif r["status"] == "error":
                        print(f"[{_ts()}] ERR {r['cell']}\n{r['msg']}", flush=True)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    _combine()


if __name__ == "__main__":
    main()
