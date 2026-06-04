"""Kovács 2019 top-ranked SMOTE variants.

Wraps polynom_fit_SMOTE, ProWSyn, and MWMOTE from smote-variants.
Selected by mean rank in Kovács 2019 (Table 4) as broadly-effective
variants not yet represented in our baseline set.

Note: smote-variants 1.0.1 exposes polynom_fit_SMOTE as four sub-variants
(bus, mesh, poly, star).  We use the *_poly variant, which applies
polynomial fitting along synthetic interpolation directions and is the
closest analogue to the method described in the Kovács ranking.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator


class _BaseSV(BaseEstimator):
    _sv_cls_name: str = ""

    def __init__(self, random_state: int | None = None,
                 proportion: float = 1.0):
        self.random_state = random_state
        self.proportion = proportion

    def _sampler_cls(self):
        """The smote-variants sampler class to use (overridable)."""
        import smote_variants as sv
        return getattr(sv, self._sv_cls_name)

    def fit_resample(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        sampler = self._sampler_cls()(proportion=self.proportion,
                                      random_state=self.random_state)
        Xr, yr = sampler.sample(X, y)
        return Xr, yr

    def fit(self, X, y):
        return self


class PolynomFitSMOTEResampler(_BaseSV):
    # smote-variants 1.0.1 splits the original method into four sub-variants;
    # polynom_fit_SMOTE_poly is the canonical polynomial-fitting approach.
    _sv_cls_name = "polynom_fit_SMOTE_poly"


def _bounded_prowsyn_cls():
    """Build a ProWSyn subclass that bounds the per-cluster simplex neighborhood.

    smote-variants 1.0.1 generates synthetic points inside each proximity
    cluster by enumerating *all-pairs* simplices: it passes
    ``indices=circulant(arange(len(cluster)))`` to the simplex sampler, so every
    one of a cluster's N members is treated as a neighbor of every other. The
    simplex enumerator then materialises an ``(N*(N-1), N)`` int64 array, i.e.
    ~8*N**3 bytes. On large, near-balanced datasets one proximity cluster holds
    a few thousand minority points (electricity: ~2850), which needs 80-400 GiB
    and raises ``numpy ArrayMemoryError``.

    This override replaces the full circulant with a bounded k-nearest-neighbour
    graph within each cluster, using the same ``n_neighbors + 1`` convention as
    smote-variants' own SMOTE (kneighbors returns self in column 0). Memory then
    scales linearly in cluster size and the proximity-weighted cluster-selection
    logic is unchanged. This is also closer to Barua et al. (2013), which
    interpolates each minority point with its k nearest minority neighbours
    rather than with the entire partition.
    """
    import smote_variants as sv
    from smote_variants.base import NearestNeighborsWithMetricTensor

    class _BoundedProWSyn(sv.ProWSyn):
        def generate_samples_in_clusters(self, X, Ps, weights, n_to_sample):
            clusters_selected = self.random_state.choice(
                len(Ps), n_to_sample, p=weights)
            cluster_unique, cluster_count = np.unique(
                clusters_selected, return_counts=True)

            nn_params = {**self.nn_params}
            nn_params["metric_tensor"] = self.metric_tensor_from_nn_params(
                nn_params, X, None)

            samples = []
            for idx, cluster in enumerate(cluster_unique):
                cluster_vectors = X[Ps[cluster]]
                n_neighbors = min(len(cluster_vectors), self.n_neighbors + 1)
                nnmt = NearestNeighborsWithMetricTensor(
                    n_neighbors=n_neighbors, n_jobs=self.n_jobs, **nn_params)
                nnmt.fit(cluster_vectors)
                indices = nnmt.kneighbors(cluster_vectors, return_distance=False)
                samples.append(self.sample_simplex(
                    X=cluster_vectors, indices=indices,
                    n_to_sample=cluster_count[idx]))

            return np.vstack(samples)

    return _BoundedProWSyn


class ProWSynResampler(_BaseSV):
    _sv_cls_name = "ProWSyn"

    def _sampler_cls(self):
        return _bounded_prowsyn_cls()


class MWMOTEResampler(_BaseSV):
    _sv_cls_name = "MWMOTE"
