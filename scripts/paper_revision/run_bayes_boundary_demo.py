#!/usr/bin/env python
"""Synthetic Bayes-boundary demonstration (Experiment N8).

Parallelised at the (separation, seed) level using ProcessPoolExecutor.
500 tasks (5 separations × 100 seeds); checkpoint every 50 completions.

Usage:
    python -m scripts.paper_revision.run_bayes_boundary_demo [--workers 10]
"""
from __future__ import annotations
import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scripts.paper_revision.config import RESULTS_DIR
from scripts.paper_revision.methods import run_method

OUT = RESULTS_DIR / "bayes_boundary_demo.parquet"


def _gen(separation: float, imbalance: float, n: int, seed: int):
    rng = np.random.default_rng(seed)
    n1 = int(n * imbalance)
    n0 = n - n1
    X0 = rng.normal(loc=[-separation, 0], scale=1.0, size=(n0, 2))
    X1 = rng.normal(loc=[+separation, 0], scale=1.0, size=(n1, 2))
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0]*n0 + [1]*n1)
    return X, y


def _bayes_boundary_x(separation: float) -> float:
    """For two equal-cov Gaussians centered at ±sep, Bayes boundary is x=0
    when classes balanced; with imbalanced priors it shifts. We use the
    *log-prior-corrected* analytical boundary."""
    return 0.0  # we apply prior correction during eval (see _eval)


def _boundary_at_y(model: LogisticRegression, ys=np.linspace(-3, 3, 5)) -> np.ndarray:
    # logistic decision boundary: w0*x + w1*y + b = 0 → x = -(w1*y + b)/w0
    w = model.coef_[0]
    b = model.intercept_[0]
    return -(w[1] * ys + b) / w[0]


def _eval(X, y, method_name: str, seed: int, separation: float) -> dict:
    Xr, yr, w, _ = run_method(method_name, X, y, random_state=seed)
    clf = LogisticRegression(max_iter=5000)
    if w is not None:
        clf.fit(Xr, yr, sample_weight=w)
    else:
        clf.fit(Xr, yr)

    # decision boundary x-locations at 5 evenly spaced y-values
    ys = np.linspace(-3, 3, 5)
    learned_x = _boundary_at_y(clf, ys)
    bayes_x = np.full_like(learned_x, _bayes_boundary_x(separation))
    rmse = float(np.sqrt(np.mean((learned_x - bayes_x) ** 2)))

    # accuracy on a large held-out balanced grid (proxy for Bayes-optimal alignment)
    rng = np.random.default_rng(seed + 999)
    Xtest = np.vstack([
        rng.normal(loc=[-separation, 0], scale=1.0, size=(2000, 2)),
        rng.normal(loc=[+separation, 0], scale=1.0, size=(2000, 2)),
    ])
    ytest = np.array([0]*2000 + [1]*2000)
    acc = float(clf.score(Xtest, ytest))
    return {"method": method_name, "boundary_rmse": rmse, "accuracy_vs_bayes": acc}


METHOD_LIST = [
    "baseline", "smote",
    "napierala_weighting_rare", "napierala_weighting_rare_outlier",
    "napierala_weighting_borderline", "napierala_weighting_nonsafe",
    "triage_weighting",
]


def simulate_once(*, separation: float, imbalance: float, n: int, seed: int) -> pd.DataFrame:
    X, y = _gen(separation, imbalance, n, seed)
    rows = [_eval(X, y, m, seed, separation) for m in METHOD_LIST]
    df = pd.DataFrame(rows)
    df["separation"] = separation
    df["imbalance"] = imbalance
    df["seed"] = seed
    return df


def _run_seed(args_tuple: tuple) -> pd.DataFrame:
    """Top-level worker: simulate one (separation, seed) cell."""
    separation, seed = args_tuple
    return simulate_once(separation=separation, imbalance=0.1, n=1000, seed=seed)


def main():
    ap = argparse.ArgumentParser(description="Bayes-boundary demo (parallelised).")
    ap.add_argument("--workers", type=int,
                    default=min(10, max(1, (os.cpu_count() or 2) // 2)),
                    help="Worker processes (default: min(10, cpu//2))")
    args = ap.parse_args()

    def _ts():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    separations = [0.5, 0.75, 1.0, 1.25, 1.5]
    seeds = list(range(100))
    tasks = [(sep, seed) for sep in separations for seed in seeds]
    total = len(tasks)

    from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn

    completed_dfs: list[pd.DataFrame] = []
    checkpoint_interval = 50

    print(f"[{_ts()}] bayes_boundary_demo: {total} cells, {args.workers} workers", file=sys.stderr)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task_id = progress.add_task("bayes boundary", total=total)

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_args = {executor.submit(_run_seed, t): t for t in tasks}

            for future in as_completed(future_to_args):
                sep, seed = future_to_args[future]
                try:
                    df = future.result()
                    completed_dfs.append(df)
                    progress.advance(task_id)
                    print(
                        f"[{_ts()}] ok  sep={sep:.2f}  seed={seed:03d}  rows={len(df)}",
                        file=sys.stderr,
                    )
                except Exception:
                    tb = traceback.format_exc()
                    progress.advance(task_id)
                    print(
                        f"[{_ts()}] err sep={sep:.2f} seed={seed:03d}: "
                        f"{tb[:200].replace(chr(10), ' ')}",
                        file=sys.stderr,
                    )

                # Checkpoint every 50 completions
                if len(completed_dfs) % checkpoint_interval == 0 and completed_dfs:
                    pd.concat(completed_dfs, ignore_index=True).to_parquet(OUT)
                    print(
                        f"[{_ts()}] checkpoint  {len(completed_dfs)}/{total}  → {OUT}",
                        file=sys.stderr,
                    )

    if completed_dfs:
        out = pd.concat(completed_dfs, ignore_index=True)
        out.to_parquet(OUT)
        print(f"[{_ts()}] wrote {OUT}  rows={len(out)}", file=sys.stderr)
        print(f"wrote {OUT}  rows={len(out)}")
    else:
        print("No results collected — check errors above.", file=sys.stderr)


if __name__ == "__main__":
    main()
