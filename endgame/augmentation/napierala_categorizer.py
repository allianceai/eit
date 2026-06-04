"""Napierala & Stefanowski (2016) instance categorization.

Categorizes minority instances by the proportion of same-class neighbors
among the k nearest neighbors:
    safe       : >= 4 same-class out of 5
    borderline : 2-3 same-class out of 5
    rare       : 1 same-class out of 5
    outlier    : 0 same-class out of 5

Majority instances are labeled "majority" (not in the four categories).

Reference
---------
Napierała, K., Stefanowski, J. Types of minority class examples and their
influence on learning classifiers from imbalanced data. J Intell Inf Syst 46,
563-597 (2016).
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y, check_is_fitted

_CATEGORIES = ("safe", "borderline", "rare", "outlier", "majority")


class NapieralaCategorizer(BaseEstimator):
    """k-NN-based minority instance categorization (Napierala & Stefanowski 2016).

    Parameters
    ----------
    k : int, default=5
        Number of nearest neighbors. The original paper uses k=5.
    minority_class : int or None, default=None
        Class label considered minority. If None, the least frequent class is used.
    """

    def __init__(self, k: int = 5, minority_class: int | None = None):
        self.k = k
        self.minority_class = minority_class

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        # In multiclass: each non-majority class treated as a minority pass.
        # Simplification per Napierala: identify the global minority(s) by frequency.
        counts = np.bincount(y.astype(int))
        if self.minority_class is None:
            minority = int(np.argmin(counts))
        else:
            minority = int(self.minority_class)

        self.categories_ = np.full(len(X), "majority", dtype=object)
        minority_mask = y == minority
        minority_idx = np.where(minority_mask)[0]

        if len(minority_idx) == 0:
            return self

        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        _, ind = nn.kneighbors(X[minority_idx])
        # drop self (the first column)
        ind = ind[:, 1:]
        same_class_counts = (y[ind] == minority).sum(axis=1)

        cats = np.empty(len(minority_idx), dtype=object)
        cats[same_class_counts >= 4] = "safe"
        cats[(same_class_counts >= 2) & (same_class_counts <= 3)] = "borderline"
        cats[same_class_counts == 1] = "rare"
        cats[same_class_counts == 0] = "outlier"

        self.categories_[minority_idx] = cats
        self.minority_class_ = minority
        return self

    def get_category_mask(self, category):
        check_is_fitted(self, "categories_")
        if isinstance(category, str):
            return self.categories_ == category
        # set / list / tuple
        return np.isin(self.categories_, list(category))
