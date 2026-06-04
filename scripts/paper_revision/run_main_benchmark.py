#!/usr/bin/env python
"""Run every method on every dataset; write one parquet per (method, dataset)."""
from __future__ import annotations
import argparse
import sys
import traceback
from pathlib import Path
import pandas as pd

from scripts.paper_revision.datasets import DATASETS, load_dataset
from scripts.paper_revision.cv_runner import evaluate_method_on_dataset
from scripts.paper_revision.config import RESULTS_DIR
from scripts.paper_revision.methods import METHODS

OUT_DIR = RESULTS_DIR / "main_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--dataset", default=None, help="run a single dataset by name")
    ap.add_argument("--classifier", default="xgboost",
                    choices=["xgboost", "rf", "lgbm", "logreg"])
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    specs = [d for d in DATASETS if args.dataset is None or d.name == args.dataset]
    for spec in specs:
        out_path = OUT_DIR / f"{args.classifier}__{args.method}__{spec.name}.parquet"
        if out_path.exists():
            print(f"skip {out_path.name}")
            continue
        try:
            X, y = load_dataset(spec)
        except Exception as e:
            print(f"ERROR load {spec.name}: {e}", file=sys.stderr)
            continue
        try:
            df = evaluate_method_on_dataset(args.method, X, y,
                                            dataset_name=spec.name,
                                            classifier=args.classifier,
                                            n_repeats=args.n_repeats,
                                            n_folds=args.n_folds)
            df.to_parquet(out_path)
            print(f"wrote {out_path.name}  rows={len(df)}  "
                  f"acc={df.accuracy.mean():.4f}  bacc={df.balanced_accuracy.mean():.4f}")
        except Exception as e:
            print(f"ERROR run {args.method} on {spec.name}: {e}", file=sys.stderr)
            traceback.print_exc()


if __name__ == "__main__":
    main()
