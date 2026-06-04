"""Authoritative dataset roster for the Error-Instance-Triage / Bayes-boundary paper.

Roster and loader restored from the original code in ``/home/cameron/endgame_backup``
(``scripts/error_triage_experiments.py``), which reproduces the paper. Datasets keep
their NATIVE class structure (no binary collapse); categoricals are ordinal-encoded;
data is capped at 50k rows on load. Downstream experiments apply their own stratified
10k subsample (cv_runner) per the paper.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np
import pandas as pd

# Fresh cache dir: the previous reconstruction cached binary-collapsed data under
# data/paper_revision_cache; do NOT reuse it. (Old cache left in place, not deleted.)
CACHE_DIR = Path(os.environ.get("PAPER_REVISION_CACHE", "data/original_roster_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES_ON_LOAD = 50_000  # matches the original loader


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str          # "openml" or "sklearn"
    identifier: str | int  # OpenML id, or negative sentinel for sklearn builtins
    n_samples: int       # approximate (actual comes from load)
    n_features: int      # approximate
    task: str            # native task label (descriptive; loader does NOT collapse)


def _S(name, ident, n, d, task):
    return DatasetSpec(name, "sklearn" if ident < 0 else "openml", ident, n, d, task)


# Original TIER1 + TIER2 + TIER3 roster (endgame_backup error_triage_experiments.py).
DATASETS: list[DatasetSpec] = [
    # --- TIER 1 ---
    _S("phoneme", 1489, 5404, 5, "binary"),
    _S("electricity", 151, 45312, 8, "binary"),
    _S("bank-marketing", 1461, 45211, 16, "binary"),
    _S("MagicTelescope", 1120, 19020, 10, "binary"),
    _S("MiniBooNE", 41150, 130064, 50, "binary"),
    _S("jannis", 41168, 83733, 54, "multiclass"),
    _S("covertype", 1596, 581012, 54, "multiclass"),
    _S("credit-g", 31, 1000, 20, "binary"),
    _S("vehicle", 54, 846, 18, "multiclass"),
    _S("segment", 36, 2310, 19, "multiclass"),
    _S("kc1", 1067, 2109, 21, "binary"),
    _S("pc4", 1049, 1458, 37, "binary"),
    # --- TIER 2 ---
    _S("ozone-level-8hr", 1487, 2534, 72, "binary"),
    _S("sick", 38, 3772, 29, "binary"),
    _S("abalone", 183, 4177, 8, "multiclass"),
    _S("yeast", 181, 1484, 8, "multiclass"),
    _S("oil_spill", 1018, 937, 49, "binary"),
    _S("mammography", 310, 11183, 6, "binary"),
    _S("wine_quality", 287, 6497, 11, "multiclass"),
    _S("thyroid", 40474, 7200, 21, "multiclass"),
    _S("satimage", 182, 6430, 36, "multiclass"),
    _S("creditcard", 1597, 284807, 30, "binary"),
    _S("webpage", 350, 34780, 300, "binary"),
    _S("solar-flare", 40687, 1066, 12, "multiclass"),
    # --- TIER 3 ---
    _S("iris", -1, 150, 4, "multiclass"),
    _S("wine", -2, 178, 13, "multiclass"),
    _S("breast_cancer", -3, 569, 30, "binary"),
    _S("digits", -4, 1797, 64, "multiclass"),
    _S("glass", 41, 214, 9, "multiclass"),
    _S("vowel", 307, 990, 12, "multiclass"),
    _S("letter", 6, 20000, 16, "multiclass"),
    _S("optdigits", 28, 5620, 64, "multiclass"),
    _S("page-blocks", 30, 5473, 10, "multiclass"),
    _S("sonar", 40, 208, 60, "binary"),
    _S("eeg-eye-state", 1471, 14980, 14, "binary"),
    _S("waveform", 60, 5000, 21, "multiclass"),
    _S("madelon", 1485, 2600, 500, "binary"),
    _S("GesturePhaseSegmentationProcessed", 4538, 9873, 32, "multiclass"),
    _S("JapaneseVowels", 375, 9961, 14, "multiclass"),
    _S("heart-statlog", 53, 270, 13, "binary"),
    _S("ionosphere", 59, 351, 34, "binary"),
    _S("wine-quality-white", 40498, 4898, 11, "multiclass"),
    _S("wine-quality-red", 40691, 1599, 11, "multiclass"),
    _S("mfeat-factors", 12, 2000, 216, "multiclass"),
    _S("mfeat-fourier", 14, 2000, 76, "multiclass"),
    _S("mfeat-karhunen", 16, 2000, 64, "multiclass"),
    _S("mfeat-morphological", 18, 2000, 6, "multiclass"),
    _S("mfeat-pixel", 20, 2000, 240, "multiclass"),
    _S("mfeat-zernike", 22, 2000, 47, "multiclass"),
    _S("semeion", 1501, 1593, 256, "multiclass"),
    _S("synthetic_control", 377, 600, 60, "multiclass"),
    _S("adult", 179, 48842, 14, "binary"),
    _S("splice", 46, 3190, 60, "multiclass"),
    _S("USPS", 41082, 9298, 256, "multiclass"),
]
# ROSTER REALIGNMENT (paper v2): the additive sweep now uses EXACTLY the original
# paper's 54-dataset headline roster, recovered verbatim from the `dataset` column of
# results/original_study/sample_weighting_results.parquet (== masked_rebalancing). The
# earlier reconstruction had drifted: it dropped USPS (re-added above; OpenML 41082) and
# added five datasets not in the headline 54 — artificial-characters (in no original
# result at all) and arrhythmia/balance-scale/dna/libras (present only in the
# interventional-50, which is reused as-is, NOT in the headline 54). Those five are
# removed so every additive experiment sits on the same benchmark as the headline,
# resolving the 50/54/57 dataset-count inconsistency the reviewers flagged. webpage
# remains (id 350) but fails-fast on load (sparse ARFF); it is covered by the reused
# original results, so the extended sweep effectively spans 53 of the 54.
assert len(DATASETS) == 54, f"expected the headline 54-dataset roster, got {len(DATASETS)}"


def load_dataset(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    """Load a dataset with the ORIGINAL paper's semantics (endgame_backup).

    Native classes are preserved (no binary collapse). Categoricals are
    ordinal-encoded; NaN/inf are zeroed; targets are label-encoded; data is capped
    at 50k rows (seed 42). Results are cached as npz under ``CACHE_DIR``.
    """
    cache_path = CACHE_DIR / f"{spec.name}.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        return z["X"], z["y"]

    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

    ident = spec.identifier
    if ident == -1:
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True); X = X.astype(np.float64)
    elif ident == -2:
        from sklearn.datasets import load_wine
        X, y = load_wine(return_X_y=True); X = X.astype(np.float64)
    elif ident == -3:
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True); X = X.astype(np.float64)
    elif ident == -4:
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True); X = X.astype(np.float64)
    else:
        # NOTE: webpage (350) cannot be parsed by the installed openml library (sparse
        # ARFF -> pandas.factorize TypeError) and the openml API is currently flaky, so
        # the fetch_openml fallback hangs on network retries. Fail FAST here instead:
        # run_parallel logs the load error and moves on; webpage is excluded from the
        # broad sweep (57/58) but is present in the reused original_study results.
        import openml
        ds = openml.datasets.get_dataset(ident)
        X_df, y_series, cat_indicator, _ = ds.get_data(
            dataset_format="dataframe", target=ds.default_target_attribute)
        if X_df is None:
            raise ValueError(f"Dataset {ident} returned no data")
        if cat_indicator is not None:
            cat_cols = [c for c, is_cat in zip(X_df.columns, cat_indicator) if is_cat]
        else:
            cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            oe = OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1, encoded_missing_value=-2)
            X_df[cat_cols] = oe.fit_transform(X_df[cat_cols].astype(str))
        X = np.nan_to_num(X_df.values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        y = LabelEncoder().fit_transform(y_series)

    if MAX_SAMPLES_ON_LOAD is not None and len(X) > MAX_SAMPLES_ON_LOAD:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), MAX_SAMPLES_ON_LOAD, replace=False)
        X, y = X[idx], y[idx]
        from sklearn.preprocessing import LabelEncoder as _LE
        y = _LE().fit_transform(y)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y
