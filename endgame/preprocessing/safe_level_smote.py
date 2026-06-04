"""Safe-Level-SMOTE (Bunkhumpornpat, Sinapiromsaran, Lursinsap 2009).

Thin wrapper around the smote-variants library implementation, providing
an imbalanced-learn-style fit_resample API consistent with the rest of
the preprocessing module.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator


class SafeLevelSMOTEResampler(BaseEstimator):
    """Safe-Level-SMOTE wrapper."""

    def __init__(self, random_state: int | None = None,
                 n_neighbors: int = 5,
                 proportion: float = 1.0):
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.proportion = proportion

    def fit_resample(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        import smote_variants as sv
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        sampler = sv.Safe_Level_SMOTE(proportion=self.proportion,
                                      n_neighbors=self.n_neighbors,
                                      random_state=self.random_state)
        Xr, yr = sampler.sample(X, y)
        return Xr, yr

    def fit(self, X, y):
        return self
