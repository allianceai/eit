#!/usr/bin/env python
"""Overlap-synthetic boundary-crossing, extended method set (Neurocomputing R7).

Same PRE-REGISTERED geometry as run_overlap_synthetic.py (2D Gaussians, known
balanced Bayes boundary x = 0, 9:1 imbalance, separation sweep, 20 seeds), with
the crossing measurement extended to the boundary-targeting variants and to
Geometric SMOTE (Douzas & Bacao 2019, authors' implementation).

Mechanism predictions (Proposition 1), stated before running:
  - Borderline-SMOTE / ADASYN raise the boundary-seed proportion, so their
    wrong-side fraction should be >= standard SMOTE's (clause iii).
  - G-SMOTE's 'combined' selection bounds the generation hypersphere by the
    nearest majority neighbour, so its wrong-side fraction should sit BETWEEN
    clean-masked (~0) and SMOTE.

Writes results/paper_revision/overlap_synthetic_gsmote.parquet (new file; the
original overlap_synthetic.parquet is left untouched). Resume-safe per
separation.

Usage:  python -m scripts.paper_revision.run_overlap_gsmote
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd

from scripts.paper_revision.config import RESULTS_DIR
from scripts.paper_revision.run_overlap_synthetic import (
    SEPARATIONS, N_SEEDS, _make, _wrong_side_frac)

OUT = RESULTS_DIR / "overlap_synthetic_gsmote.parquet"
METHODS = ["smote", "clean_masked_smote", "gsmote", "borderline_smote", "adasyn"]


def main():
    done_seps = set()
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        done_seps = set(prev["separation"].unique())
        print(f"resume: separations already done: {sorted(done_seps)}")
        rows = prev.to_dict("records")
    else:
        rows = []

    for s in SEPARATIONS:
        if s in done_seps:
            continue
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(1000 * int(s * 100) + seed)
            X, y = _make(rng, s)
            row = {"separation": s, "seed": seed}
            for m in METHODS:
                row[f"wrong_side_{m}"] = _wrong_side_frac(X, y, m, seed)
            rows.append(row)
        pd.DataFrame(rows).to_parquet(OUT)   # checkpoint after each separation
        sub = pd.DataFrame(rows).query("separation == @s")
        print(f"s={s}: " + "  ".join(
            f"{m}={sub[f'wrong_side_{m}'].mean():.3f}" for m in METHODS), flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
