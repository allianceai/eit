"""Error Instance Triage: Three-way decomposition of model errors.

Decomposes misclassified training instances into:
- **Noise** (Cat1): Mislabeled instances — remove to improve accuracy.
- **Data-limited** (Cat2): Epistemic uncertainty — augment to help.
- **Irreducible** (Cat3): Aleatoric uncertainty — leave alone.

Uses an ensemble of Random Forests for aleatoric/epistemic uncertainty
decomposition, combined with noise detection heuristics.

Example
-------
>>> from endgame.augmentation.error_triage import ErrorTriage
>>> triage = ErrorTriage(random_state=42)
>>> triage.fit(X_train, y_train)
>>> summary = triage.get_triage_summary()
>>> noise_mask = triage.get_category_mask('noise')
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y, check_is_fitted


class ErrorTriage(BaseEstimator):
    """Three-way error instance triage via uncertainty decomposition.

    Trains an ensemble of Random Forests and uses out-of-bag predictions
    to decompose prediction uncertainty into aleatoric (irreducible) and
    epistemic (data-limited) components.  Combines this with noise
    detection heuristics, OOB true-class probability, and local class
    representation to classify each misclassified instance as noise,
    data-limited, or irreducibly hard.

    Parameters
    ----------
    n_forests : int, default=5
        Number of independent Random Forests in the ensemble.
    n_trees_per_forest : int, default=100
        Trees per forest (500 total by default).
    noise_threshold : float, default=0.8
        Noise score above this classifies an error as Cat1 (noise).
    noise_tcp_threshold : float, default=0.12
        Maximum mean true-class probability for noise classification.
        Noise instances have very low tcp because the model assigns
        negligible probability to the *wrong* (given) label.  This
        gate prevents sparse-boundary errors (tcp ≈ 0.2–0.3) from
        being misclassified as noise.
    cat2_class_ratio_threshold : float, default=0.4
        Maximum class ratio for data-limited (Cat2) classification.
        The class ratio measures how well the instance's class is
        represented locally: same-class k-NN fraction divided by
        global class frequency.  Values below this threshold indicate
        the instance's class is severely underrepresented locally.
    confident_error_ratio : float, default=1.0
        Fraction of random-chance probability below which an error
        may be classified as data-limited (secondary gate).  At the
        default of 1.0, this equals the baseline probability and is
        satisfied by virtually all errors.
    sparsity_threshold : float, default=2.0
        Local/global all-class k-NN distance ratio above this.
        Used only for the diagnostic ``sparsity_score_`` attribute,
        not for category assignment.
    sparsity_k : int, default=10
        Number of neighbours for sparsity scoring (all classes).
    misclass_freq_threshold : float, default=0.5
        Minimum OOB misclassification frequency to be considered an
        error instance.
    noise_neighbor_k : int, default=10
        Number of neighbours for noise detection.
    noise_same_class_threshold : float, default=0.2
        Same-class neighbour fraction below this (combined with high
        confidence) signals label noise.
    noise_confidence_threshold : float, default=0.8
        Confidence threshold for noise detection signals.
    min_samples_leaf : int, default=5
        Minimum samples per leaf in each tree — prevents degenerate
        0/1 probabilities.
    random_state : int or None, default=None
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Parallel jobs for forest fitting.

    Attributes
    ----------
    categories_ : ndarray of shape (n_samples,)
        Per-instance category: ``'correct'``, ``'noise'``,
        ``'data_limited'``, or ``'irreducible'``.
    aleatoric_ : ndarray of shape (n_samples,)
        Per-instance aleatoric uncertainty (mean individual-tree entropy).
    epistemic_ : ndarray of shape (n_samples,)
        Per-instance epistemic uncertainty (total − aleatoric).
    total_uncertainty_ : ndarray of shape (n_samples,)
        Per-instance total uncertainty (entropy of ensemble mean).
    noise_score_ : ndarray of shape (n_samples,)
        Per-instance noise score (0–1).
    mean_true_class_prob_ : ndarray of shape (n_samples,)
        Mean probability assigned to the true class across all
        individual trees.  Low values indicate trees are confidently
        wrong (data-limited); values near 1/n_classes indicate
        genuine uncertainty (irreducible).
    sparsity_score_ : ndarray of shape (n_samples,)
        Per-instance sparsity score — local/global all-class k-NN
        distance ratio.  Higher means sparser local region.
        Diagnostic only; not used in category assignment.
    misclass_freq_ : ndarray of shape (n_samples,)
        Per-instance OOB misclassification frequency.
    error_mask_ : ndarray of shape (n_samples,)
        Boolean mask — True for misclassified instances.
    forests_ : list of RandomForestClassifier
        Fitted forests (retained for tree-level access).
    classes_ : ndarray
        Unique class labels.
    """

    def __init__(
        self,
        n_forests: int = 5,
        n_trees_per_forest: int = 100,
        noise_threshold: float = 0.8,
        noise_tcp_threshold: float = 0.12,
        cat2_class_ratio_threshold: float = 0.4,
        confident_error_ratio: float = 1.0,
        sparsity_threshold: float = 2.0,
        sparsity_k: int = 10,
        misclass_freq_threshold: float = 0.5,
        noise_neighbor_k: int = 10,
        noise_same_class_threshold: float = 0.2,
        noise_confidence_threshold: float = 0.8,
        min_samples_leaf: int = 5,
        random_state: int | None = None,
        n_jobs: int = -1,
        noise_mode: str = "global",
    ):
        # noise_mode controls how the Cat1 (noise) TCP gate handles class imbalance:
        #   "global"            : original behaviour — absolute TCP < noise_tcp_threshold.
        #   "class_conditional" : TCP threshold scaled by the instance's own-class median
        #                         TCP, so minority instances (whose TCP is suppressed by a
        #                         majority-biased ensemble) are not spuriously flagged.
        #   "balanced"          : compute the ensemble with class_weight="balanced" so
        #                         minority TCP is not suppressed in the first place.
        #   "protect_minority"  : never flag non-majority-class instances as noise.
        self.noise_mode = noise_mode
        self.n_forests = n_forests
        self.n_trees_per_forest = n_trees_per_forest
        self.noise_threshold = noise_threshold
        self.noise_tcp_threshold = noise_tcp_threshold
        self.cat2_class_ratio_threshold = cat2_class_ratio_threshold
        self.confident_error_ratio = confident_error_ratio
        self.sparsity_threshold = sparsity_threshold
        self.sparsity_k = sparsity_k
        self.misclass_freq_threshold = misclass_freq_threshold
        self.noise_neighbor_k = noise_neighbor_k
        self.noise_same_class_threshold = noise_same_class_threshold
        self.noise_confidence_threshold = noise_confidence_threshold
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit the triage model: train forests, decompose uncertainty, assign categories.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self._y_fit = y  # retained for class-aware noise gating in _assign_categories
        self._X_fit = X  # retained for held-out geometry signals (categorize_heldout)
        n_samples = len(X)

        # Step 1: Train ensemble of forests with OOB enabled
        self.forests_ = self._train_forests(X, y)

        # Step 2: Compute OOB misclassification frequency and confidence
        self.misclass_freq_, oob_confidence = self._compute_oob_misclass(X, y)
        self.error_mask_ = self.misclass_freq_ >= self.misclass_freq_threshold

        # Step 3: Decompose uncertainty (aleatoric / epistemic)
        self.aleatoric_, self.epistemic_, self.total_uncertainty_ = (
            self._compute_uncertainty_decomposition(X)
        )

        # Step 3b: Compute OOB-based mean true-class probability
        # (uses OOB predictions only, not in-bag, for a clean signal)
        self.mean_true_class_prob_ = self._compute_oob_true_class_prob(y)

        # Step 4: Compute noise scores
        self.noise_score_ = self._compute_noise_scores(X, y, oob_confidence)

        # Step 5: Compute sparsity scores (local/global all-class k-NN ratio)
        self.sparsity_score_ = self._compute_sparsity_scores(X, y)

        # Step 5b: Compute local same-class error rate
        # (used to gate noise detection: true noise is isolated,
        # data-limited errors are clustered)
        self.local_error_rate_ = self._compute_local_error_rate(X, y)

        # Step 5c: Compute local class ratio
        # (same_class_frac / expected_class_freq — values < 1.0 mean
        # the instance's class is locally underrepresented)
        self.class_ratio_ = self._compute_class_ratio(X, y)

        # Step 6: Assign categories
        self.categories_ = self._assign_categories()

        return self

    def _train_forests(self, X, y):
        """Train n_forests independent Random Forests with OOB."""
        forests = []
        for i in range(self.n_forests):
            seed = (
                self.random_state * 100 + i
                if self.random_state is not None
                else None
            )
            rf = RandomForestClassifier(
                n_estimators=self.n_trees_per_forest,
                min_samples_leaf=self.min_samples_leaf,
                oob_score=True,
                n_jobs=self.n_jobs,
                random_state=seed,
                # "balanced" mode: class-balanced instance weights (inverse class
                # frequency) so minority class probabilities are not suppressed by
                # a majority-biased ensemble. NB this is sklearn class_weight
                # reweighting on a standard bootstrap, not a balanced bootstrap.
                class_weight=("balanced" if self.noise_mode == "balanced" else None),
            )
            rf.fit(X, y)
            forests.append(rf)
        return forests

    def _compute_oob_misclass(self, X, y):
        """Compute per-instance OOB misclassification frequency and confidence."""
        n_samples = len(X)
        misclass_counts = np.zeros(n_samples)
        confidence_accum = np.zeros(n_samples)
        valid_counts = np.zeros(n_samples)

        for rf in self.forests_:
            oob_preds = rf.oob_decision_function_
            has_oob = ~np.isnan(oob_preds[:, 0])

            oob_labels = rf.classes_[np.argmax(oob_preds, axis=1)]
            misclass_counts[has_oob] += (
                oob_labels[has_oob] != y[has_oob]
            ).astype(np.float64)
            confidence_accum[has_oob] += np.max(oob_preds[has_oob], axis=1)
            valid_counts[has_oob] += 1

        valid_counts = np.maximum(valid_counts, 1)
        freq = misclass_counts / valid_counts
        confidence = confidence_accum / valid_counts

        return freq, confidence

    def _compute_uncertainty_decomposition(self, X):
        """Decompose total uncertainty into aleatoric and epistemic components.

        For T total trees across all forests, each producing probability
        p_t(y|x):
          - Total = H(mean_t(p_t))       — entropy of ensemble average
          - Aleatoric = mean_t(H(p_t))   — average entropy of individual trees
          - Epistemic = Total - Aleatoric — inter-tree disagreement

        Non-negative by Jensen's inequality (H is concave).
        """
        n_samples = len(X)
        n_classes = len(self.classes_)

        # Build class index mapping for each forest
        # (some trees may have seen only a subset of classes)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        # Collect per-tree probabilities: shape (T, n_samples, n_classes)
        all_probs = []
        for rf in self.forests_:
            # Map forest classes to global class set
            forest_map = np.array([
                class_to_idx[c] for c in rf.classes_
            ])
            for tree in rf.estimators_:
                p = tree.predict_proba(X)  # (n_samples, n_forest_classes)
                # Map to full class set
                full_p = np.zeros((n_samples, n_classes))
                full_p[:, forest_map] = p
                all_probs.append(full_p)

        all_probs = np.array(all_probs)  # (T, n_samples, n_classes)
        T = all_probs.shape[0]

        # Ensemble mean probability
        mean_p = all_probs.mean(axis=0)  # (n_samples, n_classes)

        # Total uncertainty: H(mean_p)
        total = self._entropy(mean_p)  # (n_samples,)

        # Aleatoric uncertainty: mean_t(H(p_t))
        per_tree_entropy = np.array([
            self._entropy(all_probs[t]) for t in range(T)
        ])  # (T, n_samples)
        aleatoric = per_tree_entropy.mean(axis=0)  # (n_samples,)

        # Epistemic: Total - Aleatoric (non-negative by Jensen)
        epistemic = np.maximum(total - aleatoric, 0.0)

        return aleatoric, epistemic, total

    @staticmethod
    def _entropy(p):
        """Shannon entropy of probability vectors (base e).

        Parameters
        ----------
        p : ndarray of shape (n_samples, n_classes)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        p_safe = np.clip(p, 1e-15, 1.0)
        return -np.sum(p_safe * np.log(p_safe), axis=1)

    def _compute_oob_true_class_prob(self, y):
        """Mean OOB probability for the true class across forests.

        For each instance, averages the OOB-predicted probability of
        the true label across all forests.  OOB predictions exclude
        in-bag trees (which memorise training instances), giving a
        clean signal of how confidently the model predicts the wrong
        class:

        - **Low** true-class prob → trees are confidently wrong
          (data-limited: the local feature space is dominated by the
          opposite class due to sparsity).
        - **Near baseline** (1/n_classes) true-class prob → trees are
          genuinely uncertain (irreducible: classes overlap).
        """
        n_samples = len(y)
        prob_accum = np.zeros(n_samples)
        count = np.zeros(n_samples)

        for rf in self.forests_:
            oob = rf.oob_decision_function_  # (n_samples, n_forest_classes)
            has_oob = ~np.isnan(oob[:, 0])

            # Build mapping from true class label → column index in
            # this forest's OOB decision function
            forest_class_to_idx = {c: i for i, c in enumerate(rf.classes_)}

            for i in range(n_samples):
                if has_oob[i] and y[i] in forest_class_to_idx:
                    idx = forest_class_to_idx[y[i]]
                    prob_accum[i] += oob[i, idx]
                    count[i] += 1

        count = np.maximum(count, 1)
        return prob_accum / count

    def _compute_noise_scores(self, X, y, oob_confidence):
        """Compute per-instance noise score combining three signals.

        Signal 1 (weight=0.4): Confident learning — misclass_freq * I(confidence > threshold).
        Signal 2 (weight=0.3): Neighbour-based — few same-class neighbours + high confidence.
        Signal 3 (weight=0.3): Multi-forest consensus — all forests predict same wrong class.
        """
        n_samples = len(X)
        noise_score = np.zeros(n_samples)

        # --- Signal 1: Confident learning (0.4) ---
        confident_wrong = (
            self.misclass_freq_
            * (oob_confidence > self.noise_confidence_threshold).astype(float)
        )
        noise_score += 0.4 * confident_wrong

        # --- Signal 2: Neighbour-based (0.3) ---
        k = min(self.noise_neighbor_k, n_samples - 1)
        if k > 0:
            nn = NearestNeighbors(
                n_neighbors=k + 1, metric="euclidean", n_jobs=self.n_jobs,
            )
            nn.fit(X)
            _, indices = nn.kneighbors(X)
            neighbor_labels = y[indices[:, 1:]]  # (n_samples, k)
            same_class_frac = (
                neighbor_labels == y[:, np.newaxis]
            ).mean(axis=1)  # (n_samples,)

            # Low same-class neighbours + high confidence → noise
            neighbour_noise = (
                (1.0 - same_class_frac / max(self.noise_same_class_threshold, 1e-10))
                * (oob_confidence > self.noise_confidence_threshold).astype(float)
            )
            neighbour_noise = np.clip(neighbour_noise, 0.0, 1.0)
            noise_score += 0.3 * neighbour_noise

        # --- Signal 3: Multi-forest consensus wrong class (0.3) ---
        forest_preds = []
        for rf in self.forests_:
            oob = rf.oob_decision_function_
            pred = rf.classes_[np.argmax(oob, axis=1)]
            forest_preds.append(pred)
        forest_preds = np.array(forest_preds)  # (n_forests, n_samples)

        # Check: all forests predict the same class AND it's wrong
        consensus_pred = forest_preds[0]
        all_agree = np.all(forest_preds == consensus_pred[np.newaxis, :], axis=0)
        all_wrong = consensus_pred != y
        # Combine with mean confidence across forests
        mean_conf = np.mean([
            np.max(rf.oob_decision_function_, axis=1) for rf in self.forests_
        ], axis=0)
        consensus_noise = (
            all_agree.astype(float)
            * all_wrong.astype(float)
            * (mean_conf > self.noise_confidence_threshold).astype(float)
        )
        noise_score += 0.3 * consensus_noise

        return noise_score

    def _compute_sparsity_scores(self, X, y):
        """Compute per-instance sparsity via all-class local density ratio.

        Uses the ratio of local k-NN distance (all classes) to the
        global mean.  A high ratio means the instance is in a genuinely
        sparse region of feature space, as opposed to a class-boundary
        instance which has plenty of nearby neighbours (just of the
        wrong class).
        """
        n_samples = len(X)
        k = min(self.sparsity_k, n_samples - 1)
        if k < 1:
            return np.ones(n_samples)

        nn = NearestNeighbors(
            n_neighbors=k + 1, metric="euclidean", n_jobs=self.n_jobs,
        )
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        # distances[:, 0] is self-distance (0); use 1:
        local_mean_dist = distances[:, 1:].mean(axis=1)  # (n_samples,)

        global_mean = local_mean_dist.mean()
        if global_mean < 1e-15:
            return np.ones(n_samples)

        return local_mean_dist / global_mean

    def _compute_local_error_rate(self, X, y):
        """Fraction of k nearest neighbours (all classes) that are errors.

        Used to distinguish isolated noise (low local error rate) from
        systematic data-limited failures (high local error rate).

        Uses all-class k-NN rather than same-class k-NN because noise
        instances with flipped labels cluster together in feature
        space (they share the same true-class region), giving a
        misleadingly high same-class error rate.  With all-class
        k-NN, a noise instance's neighbours are mostly true-class
        instances (correctly classified → low error rate), while a
        data-limited instance's neighbours are genuinely misclassified.
        """
        n_samples = len(X)
        k = min(self.noise_neighbor_k, n_samples - 1)
        if k < 1:
            return np.zeros(n_samples)

        nn = NearestNeighbors(
            n_neighbors=k + 1, metric="euclidean", n_jobs=self.n_jobs,
        )
        nn.fit(X)
        _, indices = nn.kneighbors(X)

        # indices[:, 1:] are the k nearest neighbours (excluding self)
        neighbor_errors = self.error_mask_[indices[:, 1:]]
        return neighbor_errors.mean(axis=1)

    def _compute_class_ratio(self, X, y):
        """Local class representation ratio.

        For each instance, computes the fraction of k nearest
        neighbours that share the same label, divided by the global
        class frequency.  Values below 1.0 indicate the instance's
        class is locally underrepresented — a hallmark of data-limited
        regions where the opposite class dominates locally.

        At a dense boundary: both classes are present in proportion
        → class_ratio ≈ 1.0.  At a sparse boundary: the minority
        class is locally overwhelmed → class_ratio < 1.0.
        """
        n_samples = len(X)
        k = min(self.noise_neighbor_k, n_samples - 1)
        if k < 1:
            return np.ones(n_samples)

        nn = NearestNeighbors(
            n_neighbors=k + 1, metric="euclidean", n_jobs=self.n_jobs,
        )
        nn.fit(X)
        _, indices = nn.kneighbors(X)

        neighbor_labels = y[indices[:, 1:]]
        same_class_frac = (
            neighbor_labels == y[:, np.newaxis]
        ).mean(axis=1)

        # Normalise by global class frequency
        class_freq = {}
        for c in self.classes_:
            class_freq[c] = max((y == c).mean(), 1e-10)

        expected = np.array([class_freq[yi] for yi in y])
        return same_class_frac / expected

    def _assign_categories(self):
        """Assign three-way categories using noise score, tcp, class ratio.

        Decision logic for each error instance:

        1. **Noise** — noise_score > threshold AND local_error_rate < 0.5
           AND mean_true_class_prob < noise_tcp_threshold.  Noise
           instances have *very low* tcp because the model assigns
           negligible probability to the wrong (given) label.  The tcp
           gate prevents sparse-boundary errors (tcp ≈ 0.2–0.3) from
           being stolen by the noise detector.

        2. **Data-limited** — class_ratio < cat2_class_ratio_threshold
           AND mean_true_class_prob < confident_error_ratio × baseline.
           The instance's class is *severely underrepresented* locally.
           This is the primary Cat2 signal — at a dense boundary both
           classes are well-represented (class_ratio ≈ 1.0), while at
           a sparse boundary the minority class is overwhelmed
           (class_ratio < 0.4).

        3. **Irreducible** — the instance's class is adequately
           represented locally (class_ratio ≥ threshold).  The error
           arises from genuine class overlap at the Bayes boundary.
        """
        return self._assign_from_signals(
            error_mask=self.error_mask_,
            noise_score=self.noise_score_,
            local_error_rate=self.local_error_rate_,
            mean_tcp=self.mean_true_class_prob_,
            class_ratio=self.class_ratio_,
            y=self._y_fit,
        )

    def _assign_from_signals(self, *, error_mask, noise_score, local_error_rate,
                             mean_tcp, class_ratio, y):
        """Shared category-assignment logic over pre-computed per-instance signals.

        Used both for the in-sample fit (``_assign_categories``) and for held-out
        instances (``categorize_heldout``), so the two paths apply IDENTICAL
        decision rules and thresholds.
        """
        n_samples = len(error_mask)
        categories = np.full(n_samples, "correct", dtype=object)
        error_idx = np.where(error_mask)[0]

        # Secondary tcp gate for Cat2 (permissive at default ratio=1.0)
        n_classes = max(len(self.classes_), 1)
        baseline_prob = 1.0 / n_classes
        tcp_cat2 = self.confident_error_ratio * baseline_prob

        # Imbalance-aware noise gating (see noise_mode in __init__). For "global" and
        # "balanced" these reduce to the original absolute threshold for all instances
        # (balanced changes only the ensemble, not this logic).
        uniq, counts = np.unique(self._y_fit, return_counts=True)
        majority_cls = uniq[int(np.argmax(counts))]
        noise_tcp_thr = np.full(n_samples, self.noise_tcp_threshold, dtype=float)
        noise_allowed = np.ones(n_samples, dtype=bool)
        if self.noise_mode == "class_conditional":
            for c in uniq:
                m = y == c
                med = float(np.median(mean_tcp[m])) if m.any() else 1.0
                noise_tcp_thr[m] = self.noise_tcp_threshold * med
        elif self.noise_mode == "protect_minority":
            noise_allowed = (y == majority_cls)

        for i in error_idx:
            is_noise = (
                noise_allowed[i]
                and noise_score[i] > self.noise_threshold
                and local_error_rate[i] < 0.5
                and mean_tcp[i] < noise_tcp_thr[i]
            )
            if is_noise:
                categories[i] = "noise"
            elif (
                class_ratio[i] < self.cat2_class_ratio_threshold
                and mean_tcp[i] < tcp_cat2
            ):
                categories[i] = "data_limited"
            else:
                categories[i] = "irreducible"

        return categories

    # ------------------------------------------------------------------
    # Held-out (out-of-fold) categorization
    # ------------------------------------------------------------------
    def categorize_heldout(self, X_new, y_new):
        """Categorize instances the forests were NOT trained on.

        Every model-derived signal (error indicator, mean true-class probability,
        aleatoric/epistemic uncertainty, noise consensus) is computed by predicting
        the held-out points with forests that never saw them -- i.e. genuinely
        out-of-fold, a strictly stronger guarantee than the OOB estimates used
        within :meth:`fit`. Geometry signals (local class ratio, local error rate,
        neighbour noise) are measured against the fitted TRAINING set. Used by the
        inner-CV out-of-fold robustness check (run_triage_oof.py) so the reported
        Cat1/2/3 fractions and the aleatoric/epistemic separation can be shown to
        hold on held-out data.

        Returns a dict with keys: categories, error_mask, aleatoric, epistemic,
        total_uncertainty, mean_true_class_prob.
        """
        check_is_fitted(self, "forests_")
        X_new = np.asarray(X_new)
        y_new = np.asarray(y_new)
        n = len(X_new)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        forest_probs, tree_probs = [], []
        for rf in self.forests_:
            fmap = np.array([class_to_idx[c] for c in rf.classes_])
            fp = np.zeros((n, n_classes)); fp[:, fmap] = rf.predict_proba(X_new)
            forest_probs.append(fp)
            for tree in rf.estimators_:
                tp = np.zeros((n, n_classes)); tp[:, fmap] = tree.predict_proba(X_new)
                tree_probs.append(tp)
        forest_probs = np.array(forest_probs)   # (F, n, C)
        tree_probs = np.array(tree_probs)       # (T, n, C)

        forest_pred = self.classes_[np.argmax(forest_probs, axis=2)]   # (F, n)
        misclass_freq = (forest_pred != y_new[None, :]).mean(axis=0)   # (n,)
        error_mask = misclass_freq >= self.misclass_freq_threshold
        conf = forest_probs.max(axis=2).mean(axis=0)                   # (n,)
        true_idx = np.array([class_to_idx.get(c, 0) for c in y_new])
        mean_tcp = forest_probs[:, np.arange(n), true_idx].mean(axis=0)

        mean_p = tree_probs.mean(axis=0)                               # (n, C)
        total = self._entropy(mean_p)
        per_tree_ent = np.array([self._entropy(tree_probs[t]) for t in range(len(tree_probs))])
        aleatoric = per_tree_ent.mean(axis=0)
        epistemic = np.maximum(total - aleatoric, 0.0)

        class_ratio = self._heldout_class_ratio(X_new, y_new)
        local_error_rate = self._heldout_local_error_rate(X_new)
        noise_score = self._heldout_noise_score(X_new, y_new, misclass_freq, conf, forest_pred)

        categories = self._assign_from_signals(
            error_mask=error_mask, noise_score=noise_score,
            local_error_rate=local_error_rate, mean_tcp=mean_tcp,
            class_ratio=class_ratio, y=y_new,
        )
        return dict(categories=categories, error_mask=error_mask,
                    aleatoric=aleatoric, epistemic=epistemic,
                    total_uncertainty=total, mean_true_class_prob=mean_tcp)

    def _heldout_neighbors(self, X_new):
        k = min(self.noise_neighbor_k, len(self._X_fit))
        nn = NearestNeighbors(n_neighbors=max(1, k), metric="euclidean", n_jobs=self.n_jobs)
        nn.fit(self._X_fit)
        _, idx = nn.kneighbors(X_new)
        return idx

    def _heldout_class_ratio(self, X_new, y_new):
        idx = self._heldout_neighbors(X_new)
        same = (self._y_fit[idx] == y_new[:, None]).mean(axis=1)
        class_freq = {c: max((self._y_fit == c).mean(), 1e-10) for c in self.classes_}
        expected = np.array([class_freq.get(yi, 1e-10) for yi in y_new])
        return same / expected

    def _heldout_local_error_rate(self, X_new):
        idx = self._heldout_neighbors(X_new)
        return self.error_mask_[idx].mean(axis=1)

    def _heldout_noise_score(self, X_new, y_new, misclass_freq, conf, forest_pred):
        n = len(X_new)
        hi_conf = (conf > self.noise_confidence_threshold).astype(float)
        score = 0.4 * (misclass_freq * hi_conf)
        idx = self._heldout_neighbors(X_new)
        same = (self._y_fit[idx] == y_new[:, None]).mean(axis=1)
        nb = (1.0 - same / max(self.noise_same_class_threshold, 1e-10)) * hi_conf
        score = score + 0.3 * np.clip(nb, 0.0, 1.0)
        all_agree = np.all(forest_pred == forest_pred[0][None, :], axis=0)
        all_wrong = forest_pred[0] != y_new
        score = score + 0.3 * (all_agree.astype(float) * all_wrong.astype(float) * hi_conf)
        return score

    def get_category_mask(self, category):
        """Return boolean mask for instances in the given category.

        Parameters
        ----------
        category : str
            One of ``'correct'``, ``'noise'``, ``'data_limited'``,
            ``'irreducible'``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Boolean mask.
        """
        check_is_fitted(self, "categories_")
        valid = {"correct", "noise", "data_limited", "irreducible"}
        if category not in valid:
            raise ValueError(
                f"Unknown category '{category}'. Expected one of {valid}."
            )
        return self.categories_ == category

    def get_triage_summary(self):
        """Return summary statistics of the triage.

        Returns
        -------
        dict
            Keys: ``n_samples``, ``n_errors``, ``error_rate``,
            per-category counts/fractions, mean uncertainties,
            ``error_ceiling``.
        """
        check_is_fitted(self, "categories_")
        n = len(self.categories_)
        n_errors = int(self.error_mask_.sum())
        error_rate = n_errors / n if n > 0 else 0.0

        summary = {
            "n_samples": n,
            "n_errors": n_errors,
            "error_rate": error_rate,
        }

        for cat in ("noise", "data_limited", "irreducible"):
            mask = self.categories_ == cat
            count = int(mask.sum())
            summary[f"n_{cat}"] = count
            summary[f"frac_{cat}"] = count / n if n > 0 else 0.0
            if count > 0:
                summary[f"mean_aleatoric_{cat}"] = float(
                    self.aleatoric_[mask].mean()
                )
                summary[f"mean_epistemic_{cat}"] = float(
                    self.epistemic_[mask].mean()
                )
            else:
                summary[f"mean_aleatoric_{cat}"] = 0.0
                summary[f"mean_epistemic_{cat}"] = 0.0

        summary["error_ceiling"] = self.error_ceiling()

        return summary

    def error_ceiling(self):
        """Estimate accuracy ceiling: 1 − irreducible_frac × error_rate.

        This estimates the best achievable accuracy assuming all noise
        can be removed and all data-limited errors can be resolved, but
        irreducible errors remain.

        Returns
        -------
        float
        """
        check_is_fitted(self, "categories_")
        n = len(self.categories_)
        if n == 0:
            return 1.0

        n_errors = int(self.error_mask_.sum())
        n_irreducible = int((self.categories_ == "irreducible").sum())

        if n_errors == 0:
            return 1.0

        irreducible_frac = n_irreducible / n_errors
        error_rate = n_errors / n

        return 1.0 - irreducible_frac * error_rate


class TriageSampleWeighter(BaseEstimator):
    """Per-instance sample weights based on error triage.

    Runs ErrorTriage to identify data-limited (Cat2) errors, then
    upweights correct-class neighbors of those errors.  The intuition:
    Cat2 errors occur where the model lacks training signal, so boosting
    nearby correctly-classified instances reinforces the decision boundary
    in those sparse regions.

    Parameters
    ----------
    weight : float, default=2.0
        Weight to assign to neighbors of Cat2 errors.
    k_neighbors : int, default=5
        Number of same-class correct neighbors to upweight per Cat2 error.
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
    sample_weights_ : ndarray of shape (n_samples,)
        Per-instance weights (1.0 for most, ``weight`` for upweighted).

    Examples
    --------
    >>> from endgame.augmentation import TriageSampleWeighter
    >>> weighter = TriageSampleWeighter(weight=2.0, random_state=42)
    >>> weighter.fit(X_train, y_train)
    >>> model.fit(X_train, y_train, sample_weight=weighter.get_sample_weights())
    """

    def __init__(
        self,
        weight: float = 2.0,
        k_neighbors: int = 5,
        n_forests: int = 5,
        n_trees_per_forest: int = 100,
        noise_tcp_threshold: float = 0.12,
        cat2_class_ratio_threshold: float = 0.4,
        random_state: int | None = None,
        noise_mode: str = "global",
    ):
        self.weight = weight
        self.k_neighbors = k_neighbors
        self.n_forests = n_forests
        self.n_trees_per_forest = n_trees_per_forest
        self.noise_tcp_threshold = noise_tcp_threshold
        self.cat2_class_ratio_threshold = cat2_class_ratio_threshold
        self.random_state = random_state
        self.noise_mode = noise_mode

    def fit(self, X, y):
        """Run triage and compute sample weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        from scipy.spatial import KDTree

        X, y = check_X_y(X, y)

        self.triage_ = ErrorTriage(
            n_forests=self.n_forests,
            n_trees_per_forest=self.n_trees_per_forest,
            noise_tcp_threshold=self.noise_tcp_threshold,
            cat2_class_ratio_threshold=self.cat2_class_ratio_threshold,
            random_state=self.random_state,
            noise_mode=self.noise_mode,
        )
        self.triage_.fit(X, y)

        cat2_mask = self.triage_.get_category_mask("data_limited")
        error_mask = self.triage_.error_mask_
        correct_mask = ~error_mask

        self.sample_weights_ = np.ones(len(X))

        for cls in np.unique(y):
            cls_cat2 = cat2_mask & (y == cls)
            cls_correct = correct_mask & (y == cls)
            if cls_cat2.sum() == 0 or cls_correct.sum() < 2:
                continue

            correct_idx = np.where(cls_correct)[0]
            tree = KDTree(X[correct_idx])
            k = min(self.k_neighbors, len(correct_idx))
            _, nn_idx = tree.query(X[cls_cat2], k=k)
            neighbor_indices = correct_idx[nn_idx.flatten()]
            self.sample_weights_[neighbor_indices] = self.weight

        return self

    def get_sample_weights(self):
        """Return the computed sample weights.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "sample_weights_")
        return self.sample_weights_
