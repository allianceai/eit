#!/usr/bin/env python
"""Dose-response causal evidence (corrected 2-arm design).

For a dose d in [0,1], we ADD d*B synthetic minority points (B = #training errors)
seeded from ONE region, and measure downstream accuracy/balanced accuracy:
  - BOUNDARY arm: seeds = Cat3 (Bayes-boundary) minority errors
  - SAFE arm:     seeds = non-boundary minority (correct + data-limited)
Both arms rebalance by the same amount at each dose; only the LOCATION differs.
Causal prediction: as the dose rises, the BOUNDARY arm loses accuracy (and gains
balanced accuracy) while the SAFE arm gains balanced accuracy WITHOUT the accuracy
cost. The contrast is dose-response evidence that boundary placement -- not
rebalancing -- causes the accuracy cost.

Resume-safe (per-dataset checkpoint). Output: results/paper_revision/dose_response_arms.parquet
Usage:  python -m scripts.paper_revision.run_dose_response
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import signal as _sig
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from scripts.paper_revision.datasets import DATASETS, load_dataset
from scripts.paper_revision.config import (RESULTS_DIR, TRIAGE_PARAMS, XGB_PARAMS,
                                           MAX_INSTANCES, RANDOM_STATE)
from scripts.paper_revision.cv_runner import _stratified_subsample
from scripts.paper_revision.run_interventional import _augment_around_seeds
from endgame.augmentation.error_triage import ErrorTriage

OUT = RESULTS_DIR / "dose_response_arms.parquet"
DOSES = [0.0, 0.25, 0.5, 0.75, 1.0]
N_REPEATS, N_FOLDS = 2, 5


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    try:
        _sig.signal(_sig.SIGHUP, _sig.SIG_IGN)
    except Exception:
        pass
    rows, done = [], set()
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        rows = prev.to_dict("records")
        done = set(prev.dataset.unique())
        print(f"[{_ts()}] resume: {len(done)} datasets done", flush=True)
    for spec in DATASETS:
        if spec.name in done:
            continue
        try:
            X, y = load_dataset(spec)
            X, y = _stratified_subsample(X, y, MAX_INSTANCES, RANDOM_STATE)
        except Exception as e:
            print(f"[{_ts()}] skip {spec.name}: {e}", flush=True)
            continue
        counts = np.bincount(np.asarray(y).astype(int))
        minority = int(np.argmin(np.where(counts == 0, counts.max() + 1, counts)))
        rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
        ds_rows = []
        for si, (tr, te) in enumerate(rskf.split(X, y)):
            seed = RANDOM_STATE + si
            Xtr, ytr = X[tr], y[tr]
            try:
                t = ErrorTriage(**{**TRIAGE_PARAMS, "noise_mode": "balanced",
                                   "random_state": seed}).fit(Xtr, ytr)
            except Exception as e:
                print(f"[{_ts()}]  triage err {spec.name}: {e}", flush=True)
                continue
            cats, err = t.categories_, t.error_mask_
            mino = (ytr == minority)
            arms = {"boundary": (cats == "irreducible") & err & mino,
                    "safe": mino & ~((cats == "irreducible") & err) & (cats != "noise")}
            B = int(err.sum())
            allcls = np.union1d(np.unique(ytr), np.unique(y[te]))
            le = LabelEncoder().fit(allcls)
            yte = le.transform(y[te])
            for arm, seeds in arms.items():
                for d in DOSES:
                    n = int(round(d * B))
                    Xa, ya = _augment_around_seeds(Xtr, ytr, seeds, n, seed)
                    try:
                        clf = XGBClassifier(**{**XGB_PARAMS, "random_state": seed, "n_jobs": 1})
                        clf.fit(Xa, le.transform(ya))
                        yp = clf.predict(X[te])
                        ds_rows.append({"dataset": spec.name, "arm": arm, "dose": d,
                                        "repeat": si // N_FOLDS, "fold": si % N_FOLDS,
                                        "n_added": len(ya) - len(ytr),
                                        "accuracy": accuracy_score(yte, yp),
                                        "balanced_accuracy": balanced_accuracy_score(yte, yp)})
                    except Exception as e:
                        print(f"[{_ts()}]  fit err {spec.name} {arm} d={d}: {e}", flush=True)
        rows += ds_rows
        pd.DataFrame(rows).to_parquet(OUT)
        print(f"[{_ts()}] done {spec.name}  rows={len(rows)}", flush=True)
    print(f"[{_ts()}] DONE total rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
