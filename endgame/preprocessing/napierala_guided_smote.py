"""Napierala-guided SMOTE: restrict SMOTE seeds to {safe, borderline}.

The closest Napierala-style analogue to our Clean-Masked SMOTE: exclude
"rare" and "outlier" minority instances from the seed pool, then run
standard SMOTE on the remaining minority subset.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y

from endgame.augmentation.napierala_categorizer import NapieralaCategorizer


class NapieralaGuidedSMOTE(BaseEstimator):
    def __init__(self, k_neighbors: int = 5, napierala_k: int = 5,
                 random_state: int | None = None):
        self.k_neighbors = k_neighbors
        self.napierala_k = napierala_k
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        cat = NapieralaCategorizer(k=self.napierala_k).fit(X, y)
        # Keep safe + borderline as seeds (exclude rare + outlier)
        keep = cat.get_category_mask(("safe", "borderline"))
        # Always include majority
        keep |= cat.get_category_mask("majority")
        self.seed_mask_ = keep
        self.categorizer_ = cat
        self.X_ = X
        self.y_ = y
        return self

    def fit_resample(self, X, y):
        """Append synthetic minority points to the FULL original set.

        Rare/outlier instances are excluded only from the SMOTE *seed* pool
        (they do not originate synthetic points); they are NOT removed from the
        training data. This mirrors clean_masked SMOTE (TriageMaskedSMOTE) so the
        Napierala-vs-triage comparison isolates the categorization scheme alone.
        Dropping the hard minority originals (the previous behavior) collapsed
        minority recall and made the baseline an unfair comparison.
        """
        from scipy.spatial import KDTree
        self.fit(X, y)
        X, y = self.X_, self.y_

        classes, counts = np.unique(y, return_counts=True)
        max_count = int(counts.max())
        rng = np.random.RandomState(self.random_state)

        X_synth_all, y_synth_all = [], []
        for cls, cnt in zip(classes, counts):
            cnt = int(cnt)
            if cnt >= max_count:
                continue  # majority class
            cls_mask = y == cls
            seed_mask = cls_mask & self.seed_mask_  # safe + borderline seeds only
            if int(seed_mask.sum()) < 2:
                seed_mask = cls_mask  # fallback: all class instances
            n_seeds = int(seed_mask.sum())
            if n_seeds < 2:
                continue

            X_seeds = X[seed_mask]
            k = min(self.k_neighbors, n_seeds - 1)
            _, nn_idx = KDTree(X_seeds).query(X_seeds, k=k + 1)
            nn_idx = nn_idx[:, 1:]  # drop self

            n_needed = max_count - cnt
            synthetic = np.empty((n_needed, X.shape[1]))
            for s in range(n_needed):
                idx = rng.randint(n_seeds)
                nn = nn_idx[idx][rng.randint(k)]
                lam = rng.uniform()
                synthetic[s] = X_seeds[idx] + lam * (X_seeds[nn] - X_seeds[idx])
            X_synth_all.append(synthetic)
            y_synth_all.append(np.full(n_needed, cls, dtype=y.dtype))

        if X_synth_all:
            return (np.vstack([X, *X_synth_all]),
                    np.concatenate([y, *y_synth_all]))
        return X.copy(), y.copy()
