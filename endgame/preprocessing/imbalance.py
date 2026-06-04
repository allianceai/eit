"""Class imbalance handling: SMOTE variants, under-sampling, and combined methods.

This module provides sklearn-compatible wrappers around imbalanced-learn with
competition-tuned defaults and additional utilities for handling class imbalance.

Example
-------
>>> from endgame.preprocessing import SMOTEResampler, AutoBalancer
>>>
>>> # Simple SMOTE resampling
>>> smote = SMOTEResampler(sampling_strategy='auto')
>>> X_resampled, y_resampled = smote.fit_resample(X, y)
>>>
>>> # Auto-select best strategy based on imbalance ratio
>>> balancer = AutoBalancer(strategy='auto')
>>> X_balanced, y_balanced = balancer.fit_resample(X, y)
>>>
>>> # Use in sklearn pipeline with imblearn's Pipeline
>>> from imblearn.pipeline import Pipeline
>>> pipe = Pipeline([
...     ('balance', SMOTEResampler()),
...     ('clf', RandomForestClassifier())
... ])
"""

from __future__ import annotations

from typing import Literal, Any
import warnings

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_X_y, check_is_fitted

# Lazy imports for imblearn
_IMBLEARN_AVAILABLE = None


def _check_imblearn():
    """Check if imbalanced-learn is available."""
    global _IMBLEARN_AVAILABLE
    if _IMBLEARN_AVAILABLE is None:
        try:
            import imblearn
            _IMBLEARN_AVAILABLE = True
        except ImportError:
            _IMBLEARN_AVAILABLE = False
    if not _IMBLEARN_AVAILABLE:
        raise ImportError(
            "imbalanced-learn is required for class balancing. "
            "Install with: pip install imbalanced-learn"
        )


# =============================================================================
# Over-sampling Methods
# =============================================================================

class SMOTEResampler(BaseEstimator):
    """SMOTE (Synthetic Minority Over-sampling Technique) wrapper.

    Creates synthetic samples by interpolating between minority class instances
    and their k-nearest neighbors.

    Parameters
    ----------
    sampling_strategy : float, str, dict, or callable, default='auto'
        Sampling information:
        - 'auto': Resample all classes but the majority
        - 'minority': Resample only the minority class
        - 'not majority': Resample all classes but the majority
        - 'all': Resample all classes
        - float: Ratio of minority to majority (0 < ratio <= 1)
        - dict: {class_label: n_samples} for each class

    k_neighbors : int, default=5
        Number of nearest neighbors used to construct synthetic samples.

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=-1
        Number of parallel jobs for neighbor search.

    Attributes
    ----------
    sampler_ : imblearn.over_sampling.SMOTE
        The fitted SMOTE sampler.

    sampling_strategy_ : dict
        The computed sampling strategy.

    Examples
    --------
    >>> from endgame.preprocessing import SMOTEResampler
    >>> smote = SMOTEResampler(k_neighbors=5, random_state=42)
    >>> X_res, y_res = smote.fit_resample(X, y)
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = 'auto',
        k_neighbors: int = 5,
        random_state: int | None = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SMOTEResampler":
        """Fit the SMOTE sampler.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : SMOTEResampler
            Fitted sampler.
        """
        _check_imblearn()
        from imblearn.over_sampling import SMOTE

        X, y = check_X_y(X, y)

        self.sampler_ = SMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        X_resampled : ndarray of shape (n_samples_new, n_features)
            Resampled training data.
        y_resampled : ndarray of shape (n_samples_new,)
            Resampled target values.
        """
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


class BorderlineSMOTEResampler(BaseEstimator):
    """Borderline-SMOTE wrapper focusing on difficult borderline samples.

    Only generates synthetic samples from minority instances that are
    near the decision boundary (borderline instances).

    Parameters
    ----------
    sampling_strategy : float, str, dict, or callable, default='auto'
        See SMOTEResampler for details.

    k_neighbors : int, default=5
        Number of nearest neighbors for SMOTE interpolation.

    m_neighbors : int, default=10
        Number of nearest neighbors to determine if instance is borderline.

    kind : {'borderline-1', 'borderline-2'}, default='borderline-1'
        - 'borderline-1': Only use borderline minority instances
        - 'borderline-2': Use borderline minority + their majority neighbors

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=-1
        Number of parallel jobs.
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = 'auto',
        k_neighbors: int = 5,
        m_neighbors: int = 10,
        kind: Literal['borderline-1', 'borderline-2'] = 'borderline-1',
        random_state: int | None = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.kind = kind
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> "BorderlineSMOTEResampler":
        """Fit the BorderlineSMOTE sampler."""
        _check_imblearn()
        from imblearn.over_sampling import BorderlineSMOTE

        X, y = check_X_y(X, y)

        self.sampler_ = BorderlineSMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            m_neighbors=self.m_neighbors,
            kind=self.kind,
            random_state=self.random_state,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


class ADASYNResampler(BaseEstimator):
    """ADASYN (Adaptive Synthetic Sampling) wrapper.

    Generates synthetic samples adaptively based on local density -
    more samples are generated in regions where minority class is sparse.

    Parameters
    ----------
    sampling_strategy : float, str, dict, or callable, default='auto'
        See SMOTEResampler for details.

    n_neighbors : int, default=5
        Number of nearest neighbors.

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=-1
        Number of parallel jobs.
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = 'auto',
        n_neighbors: int = 5,
        random_state: int | None = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ADASYNResampler":
        """Fit the ADASYN sampler."""
        _check_imblearn()
        from imblearn.over_sampling import ADASYN

        X, y = check_X_y(X, y)

        self.sampler_ = ADASYN(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


class SVMSMOTEResampler(BaseEstimator):
    """SVM-SMOTE wrapper using SVM to identify borderline samples.

    Uses SVM to identify support vectors (borderline samples) and
    generates synthetic samples only from those.

    Parameters
    ----------
    sampling_strategy : float, str, dict, or callable, default='auto'
        See SMOTEResampler for details.

    k_neighbors : int, default=5
        Number of nearest neighbors for SMOTE.

    m_neighbors : int, default=10
        Number of nearest neighbors for borderline detection.

    svm_estimator : estimator or None, default=None
        SVM classifier. If None, uses SVC with default parameters.

    out_step : float, default=0.5
        Step size for generating samples outside the decision boundary.

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=-1
        Number of parallel jobs.
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = 'auto',
        k_neighbors: int = 5,
        m_neighbors: int = 10,
        svm_estimator: Any = None,
        out_step: float = 0.5,
        random_state: int | None = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.svm_estimator = svm_estimator
        self.out_step = out_step
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SVMSMOTEResampler":
        """Fit the SVM-SMOTE sampler."""
        _check_imblearn()
        from imblearn.over_sampling import SVMSMOTE

        X, y = check_X_y(X, y)

        self.sampler_ = SVMSMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            m_neighbors=self.m_neighbors,
            svm_estimator=self.svm_estimator,
            out_step=self.out_step,
            random_state=self.random_state,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


class KMeansSMOTEResampler(BaseEstimator):
    """K-Means SMOTE wrapper for cluster-based oversampling.

    Applies k-means clustering before SMOTE, generating synthetic
    samples in under-represented clusters.

    Parameters
    ----------
    sampling_strategy : float, str, dict, or callable, default='auto'
        See SMOTEResampler for details.

    k_neighbors : int, default=2
        Number of nearest neighbors for SMOTE.

    kmeans_estimator : estimator or int, default=None
        KMeans instance or number of clusters. If None, uses n_classes.

    cluster_balance_threshold : float, default=0.1
        Threshold for considering clusters as imbalanced.

    density_exponent : float or 'auto', default='auto'
        Exponent for density-based sample allocation.

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=-1
        Number of parallel jobs.
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = 'auto',
        k_neighbors: int = 2,
        kmeans_estimator: Any = None,
        cluster_balance_threshold: float = 0.1,
        density_exponent: float | str = 'auto',
        random_state: int | None = None,
        n_jobs: int = -1,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.kmeans_estimator = kmeans_estimator
        self.cluster_balance_threshold = cluster_balance_threshold
        self.density_exponent = density_exponent
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KMeansSMOTEResampler":
        """Fit the K-Means SMOTE sampler."""
        _check_imblearn()
        from imblearn.over_sampling import KMeansSMOTE

        X, y = check_X_y(X, y)

        self.sampler_ = KMeansSMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            kmeans_estimator=self.kmeans_estimator,
            cluster_balance_threshold=self.cluster_balance_threshold,
            density_exponent=self.density_exponent,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


class RandomOverSampler(BaseEstimator):
    """Random over-sampling wrapper (duplicates minority samples).

    Simply duplicates random minority class samples. Fast but may
    lead to overfitting.

    Parameters
    ----------
    sampling_strategy : float, str, dict, or callable, default='auto'
        See SMOTEResampler for details.

    random_state : int or None, default=None
        Random seed for reproducibility.

    shrinkage : float or dict, default=None
        If not None, apply smoothed bootstrap with this shrinkage factor.
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = 'auto',
        random_state: int | None = None,
        shrinkage: float | dict | None = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.shrinkage = shrinkage

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RandomOverSampler":
        """Fit the random over-sampler."""
        _check_imblearn()
        from imblearn.over_sampling import RandomOverSampler as _ROS

        X, y = check_X_y(X, y)

        self.sampler_ = _ROS(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            shrinkage=self.shrinkage,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


# =============================================================================
# Under-sampling Methods
# =============================================================================

class EditedNearestNeighbours(BaseEstimator):
    """Edited Nearest Neighbours (ENN) under-sampling.

    Removes samples whose class label differs from the majority of
    their k-nearest neighbors (noise removal).

    Parameters
    ----------
    sampling_strategy : str, list, or callable, default='auto'
        Classes to be under-sampled.

    n_neighbors : int, default=3
        Number of nearest neighbors for majority voting.

    kind_sel : {'all', 'mode'}, default='all'
        - 'all': Sample removed if any neighbor is from different class
        - 'mode': Sample removed if majority of neighbors are different

    n_jobs : int, default=-1
        Number of parallel jobs.
    """

    def __init__(
        self,
        sampling_strategy: str | list = 'auto',
        n_neighbors: int = 3,
        kind_sel: Literal['all', 'mode'] = 'all',
        n_jobs: int = -1,
    ):
        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike) -> "EditedNearestNeighbours":
        """Fit the ENN sampler."""
        _check_imblearn()
        from imblearn.under_sampling import EditedNearestNeighbours as _ENN

        X, y = check_X_y(X, y)

        self.sampler_ = _ENN(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.n_neighbors,
            kind_sel=self.kind_sel,
            n_jobs=self.n_jobs,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


class AllKNNUnderSampler(BaseEstimator):
    """AllKNN under-sampling (multiple passes of ENN).

    Applies ENN repeatedly with increasing k values until no more
    samples are removed.

    Parameters
    ----------
    sampling_strategy : str, list, or callable, default='auto'
        Classes to be under-sampled.

    n_neighbors : int, default=3
        Starting number of nearest neighbors.

    kind_sel : {'all', 'mode'}, default='all'
        Selection strategy (see EditedNearestNeighbours).

    allow_minority : bool, default=False
        If True, allow removal of minority samples.

    n_jobs : int, default=-1
        Number of parallel jobs.
    """

    def __init__(
        self,
        sampling_strategy: str | list = 'auto',
        n_neighbors: int = 3,
        kind_sel: Literal['all', 'mode'] = 'all',
        allow_minority: bool = False,
        n_jobs: int = -1,
    ):
        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.allow_minority = allow_minority
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike) -> "AllKNNUnderSampler":
        """Fit the AllKNN sampler."""
        _check_imblearn()
        from imblearn.under_sampling import AllKNN

        X, y = check_X_y(X, y)

        self.sampler_ = AllKNN(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.n_neighbors,
            kind_sel=self.kind_sel,
            allow_minority=self.allow_minority,
            n_jobs=self.n_jobs,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


class TomekLinksUnderSampler(BaseEstimator):
    """Tomek Links under-sampling.

    Removes Tomek links - pairs of instances from different classes
    that are each other's nearest neighbor. Cleans the decision boundary.

    Parameters
    ----------
    sampling_strategy : str, list, or callable, default='auto'
        Classes to be under-sampled.

    n_jobs : int, default=-1
        Number of parallel jobs.
    """

    def __init__(
        self,
        sampling_strategy: str | list = 'auto',
        n_jobs: int = -1,
    ):
        self.sampling_strategy = sampling_strategy
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike) -> "TomekLinksUnderSampler":
        """Fit the Tomek Links sampler."""
        _check_imblearn()
        from imblearn.under_sampling import TomekLinks

        X, y = check_X_y(X, y)

        self.sampler_ = TomekLinks(
            sampling_strategy=self.sampling_strategy,
            n_jobs=self.n_jobs,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


class RandomUnderSampler(BaseEstimator):
    """Random under-sampling (removes random majority samples).

    Randomly removes majority class samples. Fast but may lose
    important information.

    Parameters
    ----------
    sampling_strategy : float, str, dict, or callable, default='auto'
        Sampling information.

    random_state : int or None, default=None
        Random seed for reproducibility.

    replacement : bool, default=False
        Whether to sample with replacement.
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = 'auto',
        random_state: int | None = None,
        replacement: bool = False,
    ):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.replacement = replacement

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RandomUnderSampler":
        """Fit the random under-sampler."""
        _check_imblearn()
        from imblearn.under_sampling import RandomUnderSampler as _RUS

        X, y = check_X_y(X, y)

        self.sampler_ = _RUS(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            replacement=self.replacement,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


class NearMissUnderSampler(BaseEstimator):
    """NearMiss under-sampling using nearest neighbor heuristics.

    Selects majority samples based on their distance to minority samples.

    Parameters
    ----------
    sampling_strategy : float, str, dict, or callable, default='auto'
        Sampling information.

    version : {1, 2, 3}, default=1
        Version of NearMiss algorithm:
        - 1: Select majority samples with smallest average distance to k nearest minority
        - 2: Select majority samples with smallest average distance to k farthest minority
        - 3: Select majority samples with smallest distance to each minority sample

    n_neighbors : int, default=3
        Number of nearest neighbors.

    n_neighbors_ver3 : int, default=3
        Number of neighbors for version 3.

    n_jobs : int, default=-1
        Number of parallel jobs.
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = 'auto',
        version: Literal[1, 2, 3] = 1,
        n_neighbors: int = 3,
        n_neighbors_ver3: int = 3,
        n_jobs: int = -1,
    ):
        self.sampling_strategy = sampling_strategy
        self.version = version
        self.n_neighbors = n_neighbors
        self.n_neighbors_ver3 = n_neighbors_ver3
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike) -> "NearMissUnderSampler":
        """Fit the NearMiss sampler."""
        _check_imblearn()
        from imblearn.under_sampling import NearMiss

        X, y = check_X_y(X, y)

        self.sampler_ = NearMiss(
            sampling_strategy=self.sampling_strategy,
            version=self.version,
            n_neighbors=self.n_neighbors,
            n_neighbors_ver3=self.n_neighbors_ver3,
            n_jobs=self.n_jobs,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


class CondensedNearestNeighbour(BaseEstimator):
    """Condensed Nearest Neighbour (CNN) under-sampling.

    Iteratively selects samples that are misclassified by 1-NN on the
    current condensed set. Finds a minimal consistent subset.

    Parameters
    ----------
    sampling_strategy : str, list, or callable, default='auto'
        Classes to be under-sampled.

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_neighbors : int, default=1
        Number of nearest neighbors.

    n_seeds_S : int, default=1
        Number of samples to start the condensing.

    n_jobs : int, default=-1
        Number of parallel jobs.
    """

    def __init__(
        self,
        sampling_strategy: str | list = 'auto',
        random_state: int | None = None,
        n_neighbors: int = 1,
        n_seeds_S: int = 1,
        n_jobs: int = -1,
    ):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_seeds_S = n_seeds_S
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike) -> "CondensedNearestNeighbour":
        """Fit the CNN sampler."""
        _check_imblearn()
        from imblearn.under_sampling import CondensedNearestNeighbour as _CNN

        X, y = check_X_y(X, y)

        self.sampler_ = _CNN(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
            n_seeds_S=self.n_seeds_S,
            n_jobs=self.n_jobs,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


class OneSidedSelectionUnderSampler(BaseEstimator):
    """One-Sided Selection (OSS) under-sampling.

    Combines Tomek links removal with CNN to remove noisy and
    redundant majority samples.

    Parameters
    ----------
    sampling_strategy : str, list, or callable, default='auto'
        Classes to be under-sampled.

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_neighbors : int, default=1
        Number of nearest neighbors for CNN step.

    n_seeds_S : int, default=1
        Number of samples to start CNN condensing.

    n_jobs : int, default=-1
        Number of parallel jobs.
    """

    def __init__(
        self,
        sampling_strategy: str | list = 'auto',
        random_state: int | None = None,
        n_neighbors: int = 1,
        n_seeds_S: int = 1,
        n_jobs: int = -1,
    ):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_seeds_S = n_seeds_S
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike) -> "OneSidedSelectionUnderSampler":
        """Fit the OSS sampler."""
        _check_imblearn()
        from imblearn.under_sampling import OneSidedSelection

        X, y = check_X_y(X, y)

        self.sampler_ = OneSidedSelection(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
            n_seeds_S=self.n_seeds_S,
            n_jobs=self.n_jobs,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


class NeighbourhoodCleaningRule(BaseEstimator):
    """Neighbourhood Cleaning Rule (NCR) under-sampling.

    Uses ENN to clean the data and then removes majority samples
    whose nearest neighbors are mostly minority.

    Parameters
    ----------
    sampling_strategy : str, list, or callable, default='auto'
        Classes to be under-sampled.

    n_neighbors : int, default=3
        Number of nearest neighbors.

    threshold_cleaning : float, default=0.5
        Threshold for cleaning majority samples.

    n_jobs : int, default=None
        Number of parallel jobs.
    """

    def __init__(
        self,
        sampling_strategy: str | list = 'auto',
        n_neighbors: int = 3,
        threshold_cleaning: float = 0.5,
        n_jobs: int | None = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.threshold_cleaning = threshold_cleaning
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike) -> "NeighbourhoodCleaningRule":
        """Fit the NCR sampler."""
        _check_imblearn()
        from imblearn.under_sampling import NeighbourhoodCleaningRule as _NCR

        X, y = check_X_y(X, y)

        self.sampler_ = _NCR(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.n_neighbors,
            threshold_cleaning=self.threshold_cleaning,
            n_jobs=self.n_jobs,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


class InstanceHardnessThresholdSampler(BaseEstimator):
    """Instance Hardness Threshold (IHT) under-sampling.

    Removes samples that are hard to classify based on a classifier's
    predicted probabilities.

    Parameters
    ----------
    sampling_strategy : str, list, or callable, default='auto'
        Classes to be under-sampled.

    estimator : estimator or None, default=None
        Classifier for computing instance hardness. If None, uses
        RandomForestClassifier.

    cv : int, default=5
        Number of cross-validation folds.

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=-1
        Number of parallel jobs.
    """

    def __init__(
        self,
        sampling_strategy: str | list = 'auto',
        estimator: Any = None,
        cv: int = 5,
        random_state: int | None = None,
        n_jobs: int = -1,
    ):
        self.sampling_strategy = sampling_strategy
        self.estimator = estimator
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike) -> "InstanceHardnessThresholdSampler":
        """Fit the IHT sampler."""
        _check_imblearn()
        from imblearn.under_sampling import InstanceHardnessThreshold

        X, y = check_X_y(X, y)

        self.sampler_ = InstanceHardnessThreshold(
            sampling_strategy=self.sampling_strategy,
            estimator=self.estimator,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


class ClusterCentroidsUnderSampler(BaseEstimator):
    """Cluster Centroids under-sampling.

    Replaces majority samples with cluster centroids from k-means.

    Parameters
    ----------
    sampling_strategy : float, str, dict, or callable, default='auto'
        Sampling information.

    random_state : int or None, default=None
        Random seed for reproducibility.

    estimator : estimator or None, default=None
        Clustering estimator. If None, uses KMeans.

    voting : {'hard', 'soft'}, default='auto'
        Voting strategy for cluster assignment.
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = 'auto',
        random_state: int | None = None,
        estimator: Any = None,
        voting: Literal['hard', 'soft', 'auto'] = 'auto',
    ):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.estimator = estimator
        self.voting = voting

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ClusterCentroidsUnderSampler":
        """Fit the Cluster Centroids sampler."""
        _check_imblearn()
        from imblearn.under_sampling import ClusterCentroids

        X, y = check_X_y(X, y)

        self.sampler_ = ClusterCentroids(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            estimator=self.estimator,
            voting=self.voting,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


# =============================================================================
# Combined Methods (Over + Under)
# =============================================================================

class SMOTEENNResampler(BaseEstimator):
    """SMOTE + Edited Nearest Neighbours combined resampling.

    Applies SMOTE over-sampling followed by ENN cleaning to remove
    noisy synthetic samples.

    Parameters
    ----------
    sampling_strategy : float, str, dict, or callable, default='auto'
        Sampling strategy for SMOTE.

    k_neighbors : int, default=5
        Number of nearest neighbors used by the underlying SMOTE.

    smote : SMOTEResampler or dict, default=None
        SMOTE instance or parameters.  If provided, k_neighbors is ignored.

    enn : EditedNearestNeighbours or dict, default=None
        ENN instance or parameters.

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=-1
        Number of parallel jobs.
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = 'auto',
        k_neighbors: int = 5,
        smote: Any = None,
        enn: Any = None,
        random_state: int | None = None,
        n_jobs: int = -1,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.smote = smote
        self.enn = enn
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SMOTEENNResampler":
        """Fit the SMOTE-ENN sampler."""
        _check_imblearn()
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE

        X, y = check_X_y(X, y)

        smote_instance = self.smote
        if smote_instance is None:
            smote_instance = SMOTE(
                k_neighbors=self.k_neighbors,
                random_state=self.random_state,
            )

        self.sampler_ = SMOTEENN(
            sampling_strategy=self.sampling_strategy,
            smote=smote_instance,
            enn=self.enn,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


class SMOTETomekResampler(BaseEstimator):
    """SMOTE + Tomek Links combined resampling.

    Applies SMOTE over-sampling followed by Tomek links removal to
    clean the decision boundary.

    Parameters
    ----------
    sampling_strategy : float, str, dict, or callable, default='auto'
        Sampling strategy for SMOTE.

    k_neighbors : int, default=5
        Number of nearest neighbors used by the underlying SMOTE.

    smote : SMOTEResampler or dict, default=None
        SMOTE instance or parameters.  If provided, k_neighbors is ignored.

    tomek : TomekLinksUnderSampler or dict, default=None
        Tomek Links instance or parameters.

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=-1
        Number of parallel jobs.
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = 'auto',
        k_neighbors: int = 5,
        smote: Any = None,
        tomek: Any = None,
        random_state: int | None = None,
        n_jobs: int = -1,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.smote = smote
        self.tomek = tomek
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SMOTETomekResampler":
        """Fit the SMOTE-Tomek sampler."""
        _check_imblearn()
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE

        X, y = check_X_y(X, y)

        smote_instance = self.smote
        if smote_instance is None:
            smote_instance = SMOTE(
                k_neighbors=self.k_neighbors,
                random_state=self.random_state,
            )

        self.sampler_ = SMOTETomek(
            sampling_strategy=self.sampling_strategy,
            smote=smote_instance,
            tomek=self.tomek,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.sampler_.fit(X, y)
        self.sampling_strategy_ = self.sampler_.sampling_strategy_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self.sampler_.fit_resample(X, y)


# =============================================================================
# Triage-Masked SMOTE
# =============================================================================


class TriageMaskedSMOTE(BaseEstimator):
    """SMOTE resampling with error-triage-based seed masking.

    Runs ErrorTriage internally to identify noise (Cat1) and irreducible
    boundary (Cat3) error instances, then excludes them from the SMOTE
    seed pool.  Synthetic minority samples are generated only from "safe"
    seeds — correctly classified instances and data-limited (Cat2) errors.

    Parameters
    ----------
    k_neighbors : int, default=5
        Number of nearest neighbors for SMOTE interpolation.
    n_forests : int, default=5
        Number of Random Forests for the ErrorTriage ensemble.
    n_trees_per_forest : int, default=100
        Trees per forest in the ErrorTriage ensemble.
    noise_tcp_threshold : float, default=0.12
        True-class probability threshold for noise detection.
    cat2_class_ratio_threshold : float, default=0.4
        Local class ratio threshold for data-limited classification.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    triage_ : ErrorTriage
        Fitted ErrorTriage instance (accessible for diagnostics).

    Examples
    --------
    >>> from endgame.preprocessing import TriageMaskedSMOTE
    >>> resampler = TriageMaskedSMOTE(random_state=42)
    >>> X_res, y_res = resampler.fit_resample(X, y)
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        n_forests: int = 5,
        n_trees_per_forest: int = 100,
        noise_tcp_threshold: float = 0.12,
        cat2_class_ratio_threshold: float = 0.4,
        random_state: int | None = None,
        noise_mode: str = "global",
    ):
        self.k_neighbors = k_neighbors
        self.n_forests = n_forests
        self.n_trees_per_forest = n_trees_per_forest
        self.noise_tcp_threshold = noise_tcp_threshold
        self.cat2_class_ratio_threshold = cat2_class_ratio_threshold
        self.random_state = random_state
        self.noise_mode = noise_mode

    def fit(self, X: ArrayLike, y: ArrayLike) -> "TriageMaskedSMOTE":
        """Run ErrorTriage and build the exclude mask.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        from endgame.augmentation.error_triage import ErrorTriage

        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        self.triage_ = ErrorTriage(
            n_forests=self.n_forests,
            n_trees_per_forest=self.n_trees_per_forest,
            noise_tcp_threshold=self.noise_tcp_threshold,
            cat2_class_ratio_threshold=self.cat2_class_ratio_threshold,
            random_state=self.random_state,
            noise_mode=self.noise_mode,
        )
        self.triage_.fit(X, y)

        # Exclude mask: Cat1 (noise) + Cat3 (irreducible) errors
        cat1 = self.triage_.get_category_mask("noise")
        cat3 = self.triage_.get_category_mask("irreducible")
        self.exclude_mask_ = (cat1 | cat3) & self.triage_.error_mask_
        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit triage and resample using masked SMOTE.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        X_resampled : ndarray of shape (n_samples_new, n_features)
        y_resampled : ndarray of shape (n_samples_new,)
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
                continue  # majority class — skip

            n_needed = max_count - cnt
            cls_mask = y == cls

            # Seeds: class instances NOT in exclude mask
            seed_mask = cls_mask & ~self.exclude_mask_
            n_seeds = int(seed_mask.sum())

            if n_seeds < 2:
                # Fallback: use all class instances
                seed_mask = cls_mask
                n_seeds = int(seed_mask.sum())

            if n_seeds < 2:
                continue

            X_seeds = X[seed_mask]
            k = min(self.k_neighbors, n_seeds - 1)

            tree = KDTree(X_seeds)
            _, nn_idx = tree.query(X_seeds, k=k + 1)
            nn_idx = nn_idx[:, 1:]  # exclude self

            synthetic = np.empty((n_needed, X.shape[1]))
            for s in range(n_needed):
                idx = rng.randint(n_seeds)
                nn = nn_idx[idx][rng.randint(k)]
                lam = rng.uniform()
                synthetic[s] = X_seeds[idx] + lam * (X_seeds[nn] - X_seeds[idx])

            X_synth_all.append(synthetic)
            y_synth_all.append(np.full(n_needed, cls, dtype=y.dtype))

        if X_synth_all:
            X_synthetic = np.vstack(X_synth_all)
            y_synthetic = np.concatenate(y_synth_all)
            return np.vstack([X, X_synthetic]), np.concatenate([y, y_synthetic])

        return X.copy(), y.copy()


# =============================================================================
# Auto-Balancer with Strategy Selection
# =============================================================================

# Available algorithms for easy access
OVER_SAMPLERS = {
    'smote': SMOTEResampler,
    'borderline_smote': BorderlineSMOTEResampler,
    'adasyn': ADASYNResampler,
    'svm_smote': SVMSMOTEResampler,
    'kmeans_smote': KMeansSMOTEResampler,
    'random_over': RandomOverSampler,
    'triage_masked_smote': TriageMaskedSMOTE,
}

UNDER_SAMPLERS = {
    'enn': EditedNearestNeighbours,
    'allknn': AllKNNUnderSampler,
    'tomek': TomekLinksUnderSampler,
    'random_under': RandomUnderSampler,
    'nearmiss': NearMissUnderSampler,
    'cnn': CondensedNearestNeighbour,
    'oss': OneSidedSelectionUnderSampler,
    'ncr': NeighbourhoodCleaningRule,
    'iht': InstanceHardnessThresholdSampler,
    'cluster_centroids': ClusterCentroidsUnderSampler,
}

COMBINED_SAMPLERS = {
    'smoteenn': SMOTEENNResampler,
    'smotetomek': SMOTETomekResampler,
}

# Modern algorithm categories (populated by submodules)
GEOMETRIC_SAMPLERS: dict[str, type] = {}
GENERATIVE_SAMPLERS: dict[str, type] = {}
LLM_SAMPLERS: dict[str, type] = {}

ALL_SAMPLERS = {**OVER_SAMPLERS, **UNDER_SAMPLERS, **COMBINED_SAMPLERS}


class AutoBalancer(BaseEstimator):
    """Automatic class balancing with strategy selection.

    Automatically selects and applies the best resampling strategy
    based on the imbalance ratio and data characteristics.

    Parameters
    ----------
    strategy : str, default='auto'
        Balancing strategy:
        - 'auto': Automatically select based on imbalance ratio
        - 'oversample': Use SMOTE-based oversampling
        - 'undersample': Use ENN-based undersampling
        - 'combine': Use SMOTE + ENN
        - 'geometric': Use MultivariateGaussianSMOTE (from geometric module)
        - 'generative': Use ForestFlowResampler (from generative module)
        - Any key from ALL_SAMPLERS (e.g., 'smote', 'borderline_smote', etc.)

    sampling_strategy : float, str, dict, or callable, default='auto'
        Target class distribution.

    imbalance_threshold : float, default=0.5
        Ratio below which data is considered imbalanced.

    severe_imbalance_threshold : float, default=0.1
        Ratio below which imbalance is considered severe.

    random_state : int or None, default=None
        Random seed for reproducibility.

    include_generative : bool, default=False
        If True, include generative samplers (from ``imbalance_generative``)
        in the auto-selection pool.

    n_jobs : int, default=-1
        Number of parallel jobs.

    **kwargs : dict
        Additional parameters passed to the selected sampler.

    Attributes
    ----------
    sampler_ : BaseEstimator
        The fitted sampler.

    imbalance_ratio_ : float
        Computed imbalance ratio (minority / majority).

    selected_strategy_ : str
        The strategy that was selected.

    Examples
    --------
    >>> from endgame.preprocessing import AutoBalancer
    >>> balancer = AutoBalancer(strategy='auto', random_state=42)
    >>> X_balanced, y_balanced = balancer.fit_resample(X, y)
    >>> print(f"Selected: {balancer.selected_strategy_}")
    """

    def __init__(
        self,
        strategy: str = 'auto',
        sampling_strategy: str | float | dict = 'auto',
        imbalance_threshold: float = 0.5,
        severe_imbalance_threshold: float = 0.1,
        include_generative: bool = False,
        random_state: int | None = None,
        n_jobs: int = -1,
        **kwargs,
    ):
        self.strategy = strategy
        self.sampling_strategy = sampling_strategy
        self.imbalance_threshold = imbalance_threshold
        self.severe_imbalance_threshold = severe_imbalance_threshold
        self.include_generative = include_generative
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def _compute_imbalance_ratio(self, y: np.ndarray) -> float:
        """Compute imbalance ratio (minority / majority)."""
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            return 1.0
        return counts.min() / counts.max()

    def _select_strategy(self, X: np.ndarray, y: np.ndarray) -> str:
        """Auto-select the best strategy based on data characteristics."""
        self.imbalance_ratio_ = self._compute_imbalance_ratio(y)
        n_samples, n_features = X.shape

        # If data is balanced, no resampling needed
        if self.imbalance_ratio_ >= self.imbalance_threshold:
            return 'none'

        # Severe imbalance: use combined approach
        if self.imbalance_ratio_ < self.severe_imbalance_threshold:
            return 'smoteenn'

        # Moderate imbalance with small dataset: oversample
        if n_samples < 1000:
            return 'borderline_smote'

        # Large dataset: undersample to save computation
        if n_samples > 10000:
            return 'random_under'

        # Default: SMOTE
        return 'smote'

    def fit(self, X: ArrayLike, y: ArrayLike) -> "AutoBalancer":
        """Fit the auto-balancer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : AutoBalancer
            Fitted balancer.
        """
        _check_imblearn()
        X, y = check_X_y(X, y)

        # Determine strategy
        if self.strategy == 'auto':
            self.selected_strategy_ = self._select_strategy(X, y)
        elif self.strategy == 'oversample':
            self.selected_strategy_ = 'smote'
        elif self.strategy == 'undersample':
            self.selected_strategy_ = 'enn'
        elif self.strategy == 'combine':
            self.selected_strategy_ = 'smoteenn'
        elif self.strategy == 'geometric':
            self.selected_strategy_ = 'multivariate_gaussian_smote'
        elif self.strategy == 'generative':
            self.selected_strategy_ = 'forest_flow'
        else:
            self.selected_strategy_ = self.strategy

        # Handle 'none' strategy
        if self.selected_strategy_ == 'none':
            self.sampler_ = None
            return self

        if self.selected_strategy_ not in ALL_SAMPLERS:
            raise ValueError(
                f"Unknown strategy '{self.selected_strategy_}'. "
                f"Available: {list(ALL_SAMPLERS.keys())}"
            )
        else:
            SamplerClass = ALL_SAMPLERS[self.selected_strategy_]

        # Create sampler with appropriate parameters
        sampler_params = {
            'sampling_strategy': self.sampling_strategy,
            'random_state': self.random_state,
        }

        # Add n_jobs if supported
        import inspect
        sig = inspect.signature(SamplerClass.__init__)
        if 'n_jobs' in sig.parameters:
            sampler_params['n_jobs'] = self.n_jobs

        # Add extra kwargs
        sampler_params.update(self.kwargs)

        # Filter to valid parameters
        valid_params = set(sig.parameters.keys()) - {'self'}
        sampler_params = {k: v for k, v in sampler_params.items() if k in valid_params}

        self.sampler_ = SamplerClass(**sampler_params)
        self.sampler_.fit(X, y)

        return self

    def fit_resample(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        X_resampled : ndarray of shape (n_samples_new, n_features)
            Resampled training data.
        y_resampled : ndarray of shape (n_samples_new,)
            Resampled target values.
        """
        self.fit(X, y)

        if self.sampler_ is None:
            # No resampling needed
            return np.asarray(X), np.asarray(y)

        return self.sampler_.fit_resample(X, y)

    def get_sampler(self) -> BaseEstimator | None:
        """Get the underlying sampler.

        Returns
        -------
        sampler : BaseEstimator or None
            The fitted sampler, or None if no resampling was needed.
        """
        check_is_fitted(self, 'sampler_')
        return self.sampler_


def get_imbalance_ratio(y: ArrayLike) -> float:
    """Compute the imbalance ratio of a target array.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target values.

    Returns
    -------
    ratio : float
        Imbalance ratio (minority_count / majority_count).
        Returns 1.0 if all classes have the same count.

    Examples
    --------
    >>> y = [0, 0, 0, 0, 0, 1, 1]
    >>> get_imbalance_ratio(y)
    0.4
    """
    y = np.asarray(y)
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        return 1.0
    return counts.min() / counts.max()


def get_class_distribution(y: ArrayLike) -> dict[Any, int]:
    """Get the class distribution of a target array.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target values.

    Returns
    -------
    distribution : dict
        Dictionary mapping class labels to counts.

    Examples
    --------
    >>> y = [0, 0, 0, 1, 1, 2]
    >>> get_class_distribution(y)
    {0: 3, 1: 2, 2: 1}
    """
    y = np.asarray(y)
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))
