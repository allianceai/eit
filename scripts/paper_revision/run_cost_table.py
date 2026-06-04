#!/usr/bin/env python
"""Measure triage + downstream classifier time as a function of dataset size."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.paper_revision.datasets import DATASETS, load_dataset
from scripts.paper_revision.config import RESULTS_DIR, TRIAGE_PARAMS, XGB_PARAMS
from endgame.augmentation.error_triage import ErrorTriage
from xgboost import XGBClassifier

OUT = RESULTS_DIR / "cost_table.parquet"


def time_pipeline(X, y) -> dict:
    t0 = time.perf_counter()
    ErrorTriage(**TRIAGE_PARAMS).fit(X, y)
    t_triage = time.perf_counter() - t0

    t0 = time.perf_counter()
    XGBClassifier(**XGB_PARAMS).fit(X, y)
    t_xgb = time.perf_counter() - t0

    return {"n": len(X), "d": X.shape[1],
            "triage_s": t_triage, "xgboost_s": t_xgb,
            "overhead_pct": 100 * t_triage / max(t_xgb, 1e-9)}


def main():
    import sys
    from datetime import datetime
    from rich.progress import track

    def _ts():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    # Sample one dataset per order of magnitude
    bins = {"<1k": [], "1k-10k": [], "10k-100k": [], ">100k": []}
    for spec in DATASETS:
        if spec.n_samples < 1000: bins["<1k"].append(spec)
        elif spec.n_samples < 10_000: bins["1k-10k"].append(spec)
        elif spec.n_samples < 100_000: bins["10k-100k"].append(spec)
        else: bins[">100k"].append(spec)

    # Flatten to at most 3 per bin for progress tracking
    to_run = []
    for bin_name, specs in bins.items():
        for spec in specs[:3]:
            to_run.append((bin_name, spec))

    for bin_name, spec in track(to_run, description="cost table"):
        try:
            X, y = load_dataset(spec)
            row = time_pipeline(X, y)
            row.update({"dataset": spec.name, "bin": bin_name})
            rows.append(row)
            print(f"[{_ts()}] {spec.name}  n={row['n']}  triage={row['triage_s']:.2f}s  "
                  f"xgb={row['xgboost_s']:.2f}s  overhead={row['overhead_pct']:.1f}%",
                  file=sys.stderr)
        except Exception as e:
            print(f"[{_ts()}] ERROR {spec.name}: {e}", file=sys.stderr)

    pd.DataFrame(rows).to_parquet(OUT)
    print(f"[{_ts()}] wrote {OUT}", file=sys.stderr)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
