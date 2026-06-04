#!/usr/bin/env python
"""Verify stratified 10K subsampling preserves triage category distribution."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.paper_revision.datasets import DATASETS, load_dataset
from scripts.paper_revision.config import RESULTS_DIR, TRIAGE_PARAMS
from endgame.augmentation.error_triage import ErrorTriage

OUT = RESULTS_DIR / "subsampling_control.parquet"


def category_fractions(X, y) -> dict[str, float]:
    t = ErrorTriage(**TRIAGE_PARAMS).fit(X, y)
    n = len(y)
    return {k: float((t.categories_ == k).mean())
            for k in ("correct", "noise", "data_limited", "irreducible")}


def main():
    import sys
    from datetime import datetime
    from rich.progress import track

    def _ts():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    big = sorted([d for d in DATASETS if d.n_samples > 20_000],
                 key=lambda d: -d.n_samples)[:5]
    for spec in track(big, description="subsampling control"):
        X, y = load_dataset(spec)
        full = category_fractions(X, y)

        rng = np.random.default_rng(0)
        idx = []
        for cls in np.unique(y):
            ci = np.where(y == cls)[0]
            n_keep = max(2, int(len(ci) * (10_000 / len(X))))
            idx.append(rng.choice(ci, size=n_keep, replace=False))
        idx = np.concatenate(idx)
        sub = category_fractions(X[idx], y[idx])

        for cat, f_full in full.items():
            rows.append({"dataset": spec.name, "category": cat,
                         "fraction_full": f_full,
                         "fraction_subsample": sub[cat],
                         "abs_diff_pct": 100 * abs(f_full - sub[cat])})
        print(f"[{_ts()}] {spec.name}  full={full}  sub={sub}", file=sys.stderr)

    pd.DataFrame(rows).to_parquet(OUT)
    print(f"[{_ts()}] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
