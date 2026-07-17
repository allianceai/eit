#!/usr/bin/env python
"""Ensemble-size (M) sensitivity of the triage + remedy decomposition (R6/E3b).

The uncertainty instrument uses M=5 random forests x T=100 trees. The reviewer
asks whether M=5 suffices. Here the full decomposition (triage category shares
among errors + remedy shares of the minority deficit) is recomputed for
M in {1, 2, 5, 10, 20} on a 10-dataset IR-spanning subset, everything else
fixed (T=100, thresholds, seed).

Output: results/paper_revision/m_sensitivity.parquet, plus a stability summary
(per-quantity max absolute deviation from the M=5 reference).

    python -m scripts.paper_revision.run_m_sensitivity --workers 8
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

from scripts.paper_revision.config import RESULTS_DIR

OUT = RESULTS_DIR / "m_sensitivity.parquet"
M_GRID = [1, 2, 5, 10, 20]
# IR-spanning, small-to-medium subset (fast enough for M=20): KEEL names.
DATASETS = ["ecoli", "yeast_me2", "abalone_19", "car_eval_34", "us_crime",
            "scene", "thyroid_sick", "wine_quality", "optical_digits", "satimage"]
QUANTITIES = ["frac_errors_cat1", "frac_errors_cat2", "frac_errors_cat3",
              "frac_threshold_recoverable", "frac_data_reducible",
              "frac_irreducible", "frac_noise"]


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"


def _run_cell(cell):
    ds, m = cell
    try:
        from scripts.paper_revision.keel_datasets import load_keel
        from scripts.paper_revision.build_reducibility import _decompose
        from threadpoolctl import threadpool_limits
        X, y = load_keel(ds)
        with threadpool_limits(limits=1):
            row = _decompose(ds, "keel", X, y, triage_overrides={"n_forests": m})
        row["n_forests"] = m
        return {"cell": cell, "status": "ok", "row": row}
    except Exception:
        return {"cell": cell, "status": "error", "msg": traceback.format_exc()}


def _summary(df):
    ref = df[df.n_forests == 5].set_index("dataset")
    print(f"\n[{_ts()}] stability vs the M=5 reference (max |dev| across datasets, pp):")
    for m in sorted(df.n_forests.unique()):
        if m == 5:
            continue
        cur = df[df.n_forests == m].set_index("dataset")
        common = ref.index.intersection(cur.index)
        devs = {q: float((cur.loc[common, q] - ref.loc[common, q]).abs().max()) * 100
                for q in QUANTITIES}
        worst = max(devs, key=devs.get)
        print(f"  M={m:2d}: " + "  ".join(f"{q.replace('frac_', '')}={v:.1f}"
                                          for q, v in devs.items())
              + f"   (worst: {worst})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if args.summary_only:
        _summary(pd.read_parquet(OUT)); return

    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    have = set(zip(done.dataset, done.n_forests)) if len(done) else set()
    cells = [(d, m) for d in DATASETS for m in M_GRID if (d, m) not in have]
    print(f"[{_ts()}] m_sensitivity: {len(DATASETS) * len(M_GRID)} cells, {len(cells)} pending.")
    rows = done.to_dict("records")
    if cells:
        ex = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker)
        try:
            futs = {ex.submit(_run_cell, c): c for c in cells}
            n = 0
            for fut in as_completed(futs):
                r = fut.result(); n += 1
                if r["status"] == "ok":
                    rows.append(r["row"])
                    pd.DataFrame(rows).to_parquet(OUT)   # checkpoint
                    print(f"[{_ts()}] ok ({n}/{len(cells)}) {r['cell']}  "
                          f"TR={r['row']['frac_threshold_recoverable']:.2f}", flush=True)
                else:
                    print(f"[{_ts()}] ERR {r['cell']}\n{r['msg']}", flush=True)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    _summary(df)


if __name__ == "__main__":
    main()
