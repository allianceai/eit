#!/usr/bin/env python
"""Synthetic + semi-synthetic triage validation (§5.1, §5.3 v2 numbering)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.paper_revision.datasets import DATASETS, load_dataset
from scripts.paper_revision.config import RESULTS_DIR, TRIAGE_PARAMS
from endgame.augmentation.error_triage import ErrorTriage

OUT = RESULTS_DIR / "synthetic_validation.parquet"


def _cat2_data_limited_among_errors(n_per_class, sep, std, seed):
    """Fraction of *errors* the triage flags as data-limited (Cat2), for two
    overlapping 2D Gaussians sampled at n_per_class points each."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal([-sep, 0], std, (n_per_class, 2)),
                   rng.normal([+sep, 0], std, (n_per_class, 2))])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    t = ErrorTriage(**TRIAGE_PARAMS).fit(X, y)
    err = t.error_mask_
    if err.sum() == 0:
        return float("nan"), 0
    dl = t.get_category_mask("data_limited")
    return float((dl & err).sum() / err.sum()), int(err.sum())


def cat2_discrimination(sep=2.0, std=1.0, n_seeds=20):
    """Paper Sec.5.1: dense (900/class, well-separated) vs sparse (100/class,
    isolated) 2D Gaussians; compare the data-limited (Cat2) fraction AMONG ERRORS.

    Same well-separated Gaussian geometry (centers +/-sep, std), differing ONLY in
    sampling density -- the sparse condition is "isolated" because few same-class
    points sit near each error, lowering its local class ratio. Errors are pooled
    across seeds to stabilise the (small) sparse-condition error count. Reported
    as-is; not tuned to the paper's 3.5x ratio.
    """
    def pooled_frac(n_per_class, seeds):
        cat2_count = err_count = 0
        for s in seeds:
            f, ne = _cat2_data_limited_among_errors(n_per_class, sep, std, s)
            if ne > 0:
                cat2_count += f * ne  # f*ne = number of Cat2 errors
                err_count += ne
        return (cat2_count / err_count if err_count else float("nan")), err_count

    cat2_dense, n_err_dense = pooled_frac(900, range(n_seeds))
    cat2_sparse, n_err_sparse = pooled_frac(100, range(100, 100 + n_seeds))
    return {"experiment": "cat2_discrim",
            "separation": sep,
            "cat2_dense": cat2_dense, "cat2_sparse": cat2_sparse,
            "n_err_dense": int(n_err_dense), "n_err_sparse": int(n_err_sparse),
            "ratio": cat2_sparse / max(cat2_dense, 1e-9)}


def sparsity_induction(n_datasets=24, removal_frac=0.9, k_remove=20,
                       random_state=0):
    """Paper Sec.5.2: removing 90% of the same-class neighbors around dense-region
    (Cat3) errors should raise the local data-limited (Cat2) fraction among the
    surviving errors -- a direct test that the triage responds to induced sparsity.

    For each dataset: triage -> Cat2-fraction-among-errors (before); thin 90% of the
    same-class neighbors of the Cat3 ("dense-region") errors; re-triage -> Cat2
    fraction (after). Report delta per dataset. Not tuned to the paper's +0.183.
    """
    from scipy.spatial import KDTree
    rows = []
    for spec in DATASETS[:n_datasets]:
        try:
            X, y = load_dataset(spec)
        except Exception:
            continue
        if len(X) > 5000:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(X), 5000, replace=False)
            X, y = X[idx], y[idx]

        t0 = ErrorTriage(**TRIAGE_PARAMS).fit(X, y)
        err0 = t0.error_mask_
        if err0.sum() == 0:
            continue
        c2_before = float((t0.get_category_mask("data_limited") & err0).sum() / err0.sum())

        # Dense-region errors = Cat3 (locally well-represented) errors.
        dense_err = np.where(t0.get_category_mask("irreducible") & err0)[0]
        if len(dense_err) == 0:
            continue

        rng = np.random.default_rng(random_state)
        to_remove: set[int] = set()
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            de_c = [i for i in dense_err if y[i] == c]
            if len(ci) < 2 or not de_c:
                continue
            kk = min(k_remove, len(ci) - 1)
            _, nn = KDTree(X[ci]).query(X[de_c], k=kk + 1)
            for row in nn:
                neigh = ci[row[1:]]  # same-class neighbors (drop self)
                n_rm = int(removal_frac * len(neigh))
                if n_rm > 0:
                    to_remove.update(rng.choice(neigh, size=n_rm, replace=False).tolist())

        keep = np.array([i for i in range(len(y)) if i not in to_remove])
        if len(keep) < 10 or len(np.unique(y[keep])) < 2:
            continue
        t1 = ErrorTriage(**TRIAGE_PARAMS).fit(X[keep], y[keep])
        err1 = t1.error_mask_
        if err1.sum() == 0:
            continue
        c2_after = float((t1.get_category_mask("data_limited") & err1).sum() / err1.sum())
        rows.append({"experiment": "sparsity_induction", "dataset": spec.name,
                     "cat2_before": c2_before, "cat2_after": c2_after,
                     "delta": c2_after - c2_before, "n_removed": len(to_remove)})
    return rows


def cat3_detection_sweep():
    rows = []
    for sep in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        rng = np.random.default_rng(int(100*sep))
        X = np.vstack([rng.normal([-sep, 0], 1.0, (500, 2)),
                       rng.normal([+sep, 0], 1.0, (500, 2))])
        y = np.array([0]*500 + [1]*500)
        t = ErrorTriage(**TRIAGE_PARAMS).fit(X, y)
        err = t.error_mask_
        if err.sum() == 0:
            cat3_frac = 0.0
        else:
            cat3_frac = float((t.get_category_mask("irreducible") & err).sum() / err.sum())
        rows.append({"experiment": "cat3_sweep", "separation": sep,
                     "cat3_frac_among_errors": cat3_frac, "n_errors": int(err.sum())})
    return rows


def semi_synthetic_noise_injection(n_datasets: int = 24):
    """For each of the first n_datasets, flip 5% of labels and check triage recall."""
    rows = []
    for spec in DATASETS[:n_datasets]:
        try:
            X, y = load_dataset(spec)
        except Exception:
            continue
        if len(X) > 5000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X), 5000, replace=False)
            X, y = X[idx], y[idx]

        rng = np.random.default_rng(0)
        flip_idx = rng.choice(len(y), size=max(1, int(0.05*len(y))), replace=False)
        y_noisy = y.copy()
        classes = np.unique(y)
        for i in flip_idx:
            other = classes[classes != y_noisy[i]]
            y_noisy[i] = rng.choice(other)
        injected_mask = np.zeros(len(y), dtype=bool)
        injected_mask[flip_idx] = True

        t = ErrorTriage(**TRIAGE_PARAMS).fit(X, y_noisy)
        detected = t.get_category_mask("noise")

        if injected_mask.sum() and detected.sum():
            precision = float((detected & injected_mask).sum() / detected.sum())
            recall = float((detected & injected_mask).sum() / injected_mask.sum())
        else:
            precision = recall = float("nan")
        rows.append({"experiment": "noise_injection", "dataset": spec.name,
                     "precision": precision, "recall": recall,
                     "n_injected": int(injected_mask.sum()),
                     "n_detected": int(detected.sum())})
    return rows


def main():
    import sys
    from datetime import datetime
    from rich.progress import track

    def _ts():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out = []

    print(f"[{_ts()}] running cat2 discrimination", file=sys.stderr)
    out.append(cat2_discrimination())
    print(f"[{_ts()}] done cat2 discrimination", file=sys.stderr)

    sep_vals = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    for sep in track(sep_vals, description="cat3 sweep"):
        rng = np.random.default_rng(int(100 * sep))
        X = np.vstack([rng.normal([-sep, 0], 1.0, (500, 2)),
                       rng.normal([+sep, 0], 1.0, (500, 2))])
        y = np.array([0] * 500 + [1] * 500)
        from endgame.augmentation.error_triage import ErrorTriage
        t = ErrorTriage(**TRIAGE_PARAMS).fit(X, y)
        err = t.error_mask_
        if err.sum() == 0:
            cat3_frac = 0.0
        else:
            cat3_frac = float((t.get_category_mask("irreducible") & err).sum() / err.sum())
        row = {"experiment": "cat3_sweep", "separation": sep,
               "cat3_frac_among_errors": cat3_frac, "n_errors": int(err.sum())}
        out.append(row)
        print(f"[{_ts()}] cat3_sweep sep={sep:.2f}  cat3_frac={cat3_frac:.3f}  n_errors={err.sum()}", file=sys.stderr)

    n_datasets = 24
    for spec in track(DATASETS[:n_datasets], description="noise injection"):
        try:
            X, y = load_dataset(spec)
        except Exception:
            continue
        if len(X) > 5000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X), 5000, replace=False)
            X, y = X[idx], y[idx]

        rng = np.random.default_rng(0)
        flip_idx = rng.choice(len(y), size=max(1, int(0.05 * len(y))), replace=False)
        y_noisy = y.copy()
        classes = np.unique(y)
        for i in flip_idx:
            other = classes[classes != y_noisy[i]]
            y_noisy[i] = rng.choice(other)
        injected_mask = np.zeros(len(y), dtype=bool)
        injected_mask[flip_idx] = True

        from endgame.augmentation.error_triage import ErrorTriage
        t = ErrorTriage(**TRIAGE_PARAMS).fit(X, y_noisy)
        detected = t.get_category_mask("noise")

        if injected_mask.sum() and detected.sum():
            precision = float((detected & injected_mask).sum() / detected.sum())
            recall = float((detected & injected_mask).sum() / injected_mask.sum())
        else:
            precision = recall = float("nan")
        row = {"experiment": "noise_injection", "dataset": spec.name,
               "precision": precision, "recall": recall,
               "n_injected": int(injected_mask.sum()),
               "n_detected": int(detected.sum())}
        out.append(row)
        print(f"[{_ts()}] noise_injection {spec.name}  prec={precision:.3f}  rec={recall:.3f}", file=sys.stderr)

    print(f"[{_ts()}] running sparsity induction", file=sys.stderr)
    out.extend(sparsity_induction())
    print(f"[{_ts()}] done sparsity induction", file=sys.stderr)

    pd.DataFrame(out).to_parquet(OUT)
    print(f"[{_ts()}] wrote {OUT}", file=sys.stderr)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
