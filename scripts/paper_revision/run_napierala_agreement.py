"""For each dataset, compute the cross-tab of Napierala vs triage categories."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.paper_revision.datasets import DATASETS, load_dataset
from scripts.paper_revision.config import RESULTS_DIR, TRIAGE_PARAMS, MAX_INSTANCES
from endgame.augmentation.error_triage import ErrorTriage
from endgame.augmentation.napierala_categorizer import NapieralaCategorizer

OUT_DIR = RESULTS_DIR / "napierala_agreement"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_agreement_matrix(X, y, *, random_state: int = 42) -> pd.DataFrame:
    # subsample for speed
    if len(X) > MAX_INSTANCES:
        rng = np.random.default_rng(random_state)
        idx = []
        for cls in np.unique(y):
            ci = np.where(y == cls)[0]
            frac = MAX_INSTANCES / len(X)
            n = max(2, int(len(ci) * frac))
            idx.append(rng.choice(ci, size=n, replace=False))
        idx = np.concatenate(idx)
        X, y = X[idx], y[idx]

    triage = ErrorTriage(**{**TRIAGE_PARAMS, "random_state": random_state}).fit(X, y)
    napierala = NapieralaCategorizer(k=5).fit(X, y)
    tri_cat = triage.categories_
    nap_cat = napierala.categories_

    df = pd.DataFrame({"napierala": nap_cat, "triage": tri_cat})
    out = (df.groupby(["napierala", "triage"]).size()
             .rename("count").reset_index())
    return out


def main():
    import sys
    from datetime import datetime
    from rich.progress import track

    def _ts():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()

    specs = [s for s in DATASETS if not args.dataset or s.name == args.dataset]
    for spec in track(specs, description="napierala agreement"):
        out_path = OUT_DIR / f"{spec.name}.parquet"
        if out_path.exists():
            print(f"[{_ts()}] skip {spec.name}", file=sys.stderr)
            continue
        try:
            X, y = load_dataset(spec)
            df = compute_agreement_matrix(X, y)
            df["dataset"] = spec.name
            df.to_parquet(out_path)
            print(f"[{_ts()}] wrote {out_path.name}", file=sys.stderr)
        except Exception as e:
            print(f"[{_ts()}] ERROR {spec.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
