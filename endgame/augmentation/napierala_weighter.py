"""Napierala-categorization-based sample weighting.

Mirrors the TriageSampleWeighter algorithm (Algorithm 2 in the paper) but
substitutes Napierala & Stefanowski categories for the uncertainty-based
triage. Used for the killer ablation N7.

Mappings choose which Napierala categories play the "Cat2" role
(error-instances whose correct same-class neighbors get upweighted):

    rare            : {rare}
    rare_outlier    : {rare, outlier}
    borderline      : {borderline}
    nonsafe         : {borderline, rare, outlier}
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_X_y, check_is_fitted
from scipy.spatial import KDTree

from endgame.augmentation.napierala_categorizer import NapieralaCategorizer

_VALID_MAPPINGS = {
    "rare": ("rare",),
    "rare_outlier": ("rare", "outlier"),
    "borderline": ("borderline",),
    "nonsafe": ("borderline", "rare", "outlier"),
}


class NapieralaSampleWeighter(BaseEstimator):
    """Per-instance sample weights derived from Napierala categorization.

    Errors are identified by majority vote of an out-of-bag random forest
    (matching the v1 paper's triage error-mask convention). For each error
    instance whose Napierala category is in the mapping set, the k nearest
    correctly-classified same-class neighbors are upweighted by `weight`.

    Parameters
    ----------
    weight : float, default=2.0
        Upweight value.
    mapping : str, default="rare"
        Which Napierala categories trigger upweighting. See module docstring.
    k_neighbors : int, default=5
        Neighbors to upweight per qualifying error.
    napierala_k : int, default=5
        k for the Napierala categorization rule.
    rf_trees : int, default=300
        Trees in the OOB error-detection forest.
    random_state : int or None, default=None
    """

    def __init__(self, weight: float = 2.0, mapping: str = "rare",
                 k_neighbors: int = 5, napierala_k: int = 5,
                 rf_trees: int = 300, random_state: int | None = None):
        self.weight = weight
        self.mapping = mapping
        self.k_neighbors = k_neighbors
        self.napierala_k = napierala_k
        self.rf_trees = rf_trees
        self.random_state = random_state

    def fit(self, X, y):
        if self.mapping not in _VALID_MAPPINGS:
            raise ValueError(
                f"mapping must be one of {sorted(_VALID_MAPPINGS)}, got {self.mapping!r}"
            )
        X, y = check_X_y(X, y)

        # Error mask via OOB RF (mirror v1 triage convention)
        rf = RandomForestClassifier(n_estimators=self.rf_trees, oob_score=True,
                                    bootstrap=True, n_jobs=-1,
                                    random_state=self.random_state)
        rf.fit(X, y)
        oob_pred = np.argmax(rf.oob_decision_function_, axis=1)
        classes = rf.classes_
        error_mask = classes[oob_pred] != y

        # Napierala categorization on the same data
        cat = NapieralaCategorizer(k=self.napierala_k).fit(X, y)
        cat2_like_set = _VALID_MAPPINGS[self.mapping]
        cat2_like_mask = cat.get_category_mask(cat2_like_set)
        qualifying_error = error_mask & cat2_like_mask

        correct_mask = ~error_mask
        sw = np.ones(len(X), dtype=float)

        for cls in np.unique(y):
            cls_qual = qualifying_error & (y == cls)
            cls_correct = correct_mask & (y == cls)
            if cls_qual.sum() == 0 or cls_correct.sum() < 2:
                continue
            correct_idx = np.where(cls_correct)[0]
            tree = KDTree(X[correct_idx])
            k = min(self.k_neighbors, len(correct_idx))
            _, nn = tree.query(X[cls_qual], k=k)
            sw[correct_idx[nn.flatten()]] = self.weight

        self.sample_weights_ = sw
        self.error_mask_ = error_mask
        self.categorizer_ = cat
        return self

    def get_sample_weights(self):
        check_is_fitted(self, "sample_weights_")
        return self.sample_weights_
