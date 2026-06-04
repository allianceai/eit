"""KEEL/HDDT external imbalanced benchmark — the 27 canonical datasets bundled by
imbalanced-learn's ``fetch_datasets`` (Zenodo collection; imbalance ratio ~8 to ~130).

This is the pre-specified, citable, genuinely-imbalanced benchmark the ECML reviews
asked for (R3: "select only imbalanced datasets from established benchmarks, e.g.
KEEL"). It complements the recovered 54-dataset original roster (see
ANALYSIS_NOTES.md): the original roster is the headline; KEEL is external validation.

Targets arrive as +1 (minority) / -1 (majority); we map to 1 / 0. Data is cached by
imbalanced-learn under ~/scikit_learn_data on first fetch.
"""
from __future__ import annotations
import numpy as np

# The 27 names, in imbalanced-learn / Zenodo order (ascending-ish imbalance ratio).
KEEL_DATASETS = [
    "ecoli", "optical_digits", "satimage", "pen_digits", "abalone",
    "sick_euthyroid", "spectrometer", "car_eval_34", "isolet", "us_crime",
    "yeast_ml8", "scene", "libras_move", "thyroid_sick", "coil_2000",
    "arrhythmia", "solar_flare_m0", "oil", "car_eval_4", "wine_quality",
    "letter_img", "yeast_me2", "webpage", "ozone_level", "mammography",
    "protein_homo", "abalone_19",
]


def load_keel(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one KEEL/HDDT benchmark dataset as (X float64, y in {0,1})."""
    from imblearn.datasets import fetch_datasets
    bunch = fetch_datasets(filter_data=(name,))[name]
    X = np.asarray(bunch.data, dtype=np.float64)
    y = (np.asarray(bunch.target) > 0).astype(int)  # +1 minority -> 1, -1 majority -> 0
    return X, y
