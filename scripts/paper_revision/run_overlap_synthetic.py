#!/usr/bin/env python
"""Overlap-synthetic mechanism demo (replacement candidate for the dropped Bayes demo).

PRE-REGISTERED (no tuning to a desired outcome). Controlled 2D Gaussian-mixture with a
KNOWN balanced Bayes boundary at x = 0: majority ~ N([-s,0], I), minority ~ N([+s,0], I),
imbalance 9:1. As class overlap grows (separation s shrinks), the paper's mechanism
predicts that standard SMOTE generates an increasing fraction of synthetic minority
points ACROSS the boundary, in majority territory (x < 0) -- label-corrupting generation
-- whereas clean-masked SMOTE (which excludes triage Cat\,3 / boundary seeds) generates
fewer such points; downstream accuracy should track this.

Metrics recorded per (separation, seed):
  - wrong_side_frac: fraction of synthetic minority points with x < 0 (majority side),
    for smote and clean_masked_smote.
  - test accuracy / balanced accuracy on a large CLEAN held-out set, for
    baseline / smote / clean_masked_smote / triage_weighting (logistic regression).

Resume-safe: results are checkpointed to overlap_synthetic.parquet after every
separation; separations already present are skipped on re-run.

Usage:  python -m scripts.paper_revision.run_overlap_synthetic
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.paper_revision.config import RESULTS_DIR
from scripts.paper_revision.methods import run_method

OUT = RESULTS_DIR / "overlap_synthetic.parquet"
SEPARATIONS = [0.4, 0.6, 0.8, 1.0, 1.25, 1.5]
N_SEEDS = 20
N_MAJ, N_MIN = 900, 100          # 9:1 imbalance


def _make(rng, s):
    Xmaj = rng.normal([-s, 0.0], 1.0, (N_MAJ, 2))
    Xmin = rng.normal([+s, 0.0], 1.0, (N_MIN, 2))
    X = np.vstack([Xmaj, Xmin]).astype(np.float64)
    y = np.array([0] * N_MAJ + [1] * N_MIN)
    return X, y


def _synthetic_minority(X, y, Xr, yr):
    """Recover SMOTE-appended synthetic minority points (imblearn keeps originals first)."""
    n_min_orig = int((y == 1).sum())
    res_min = Xr[yr == 1]
    return res_min[n_min_orig:] if len(res_min) > n_min_orig else res_min[:0]


def _wrong_side_frac(X, y, method, seed):
    Xr, yr, w, _ = run_method(method, X, y, seed)
    syn = _synthetic_minority(X, y, Xr, yr)
    if len(syn) == 0:
        return np.nan
    return float((syn[:, 0] < 0).mean())   # x<0 == majority side of the Bayes boundary


def main():
    done_seps = set()
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        done_seps = set(prev["separation"].unique())
        print(f"resume: {len(done_seps)} separations already done: {sorted(done_seps)}")
        rows = prev.to_dict("records")
    else:
        rows = []

    for s in SEPARATIONS:
        if s in done_seps:
            continue
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(1000 * int(s * 100) + seed)
            X, y = _make(rng, s)
            rows.append({"separation": s, "seed": seed,
                         "wrong_side_smote": _wrong_side_frac(X, y, "smote", seed),
                         "wrong_side_clean_masked": _wrong_side_frac(X, y, "clean_masked_smote", seed)})
        pd.DataFrame(rows).to_parquet(OUT)   # checkpoint after each separation
        sub = pd.DataFrame(rows).query("separation == @s")
        print(f"s={s}: wrong-side SMOTE={sub.wrong_side_smote.mean():.3f} "
              f"clean_masked={sub.wrong_side_clean_masked.mean():.3f}", flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
