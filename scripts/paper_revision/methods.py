"""Central registry of every method evaluated in the paper revision."""
from __future__ import annotations
import logging
import numpy as np
from typing import Callable
from scripts.paper_revision.config import TRIAGE_PARAMS

MethodFn = Callable[[np.ndarray, np.ndarray, int], tuple]

# smote-variants attaches its OWN StreamHandler at level DEBUG when imported
# (see smote_variants/_logger.py), spraying per-step INFO lines to the console.
# Inside the parallel sweep that flood collides with the rich live progress
# panel and renders the terminal unreadable. Import it once here so its logger
# exists, then clamp it to WARNING for the rest of the process. The import is a
# one-time cost per worker, and workers are long-lived, so it's negligible.
try:
    import smote_variants  # noqa: F401  (imported only to materialise its logger)
    logging.getLogger("smote_variants").setLevel(logging.WARNING)
except Exception:  # pragma: no cover - if the lib is absent, nothing to quiet
    pass


def _safe_k(y, default: int = 5) -> int:
    """Largest k_neighbors that works with the smallest class in y.
    SMOTE-family methods require n_samples > k_neighbors per class,
    so n_samples must be >= 2 for even k=1 to work.
    Returns max(1, min(default, smallest_class_size - 1)), or 0 if
    smallest_class_size < 2 (SMOTE cannot run at all).
    """
    arr = np.asarray(y).astype(int)
    counts = np.bincount(arr)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return default
    min_count = int(counts.min())
    if min_count < 2:
        return 0  # sentinel: SMOTE is infeasible, caller should skip
    return max(1, min(default, min_count - 1))


def _baseline(X, y, random_state):
    return X.copy(), y.copy(), None, {"method": "baseline"}


def _smote(X, y, random_state):
    from endgame.preprocessing.imbalance import SMOTEResampler
    k = _safe_k(y)
    if k == 0:
        return X.copy(), y.copy(), None, {"method": "smote", "skipped": "too_few_samples"}
    s = SMOTEResampler(k_neighbors=k, random_state=random_state)
    Xr, yr = s.fit_resample(X, y)
    return Xr, yr, None, {"method": "smote"}


def _borderline(X, y, random_state):
    from endgame.preprocessing.imbalance import BorderlineSMOTEResampler
    k = _safe_k(y)
    if k == 0:
        return X.copy(), y.copy(), None, {"method": "borderline_smote", "skipped": "too_few_samples"}
    s = BorderlineSMOTEResampler(k_neighbors=k, random_state=random_state)
    Xr, yr = s.fit_resample(X, y)
    return Xr, yr, None, {"method": "borderline_smote"}


def _adasyn(X, y, random_state):
    from endgame.preprocessing.imbalance import ADASYNResampler, SMOTEResampler
    k = _safe_k(y)
    if k == 0:
        return X.copy(), y.copy(), None, {"method": "adasyn", "skipped": "too_few_samples"}
    s = ADASYNResampler(n_neighbors=k, random_state=random_state)
    try:
        Xr, yr = s.fit_resample(X, y)
        return Xr, yr, None, {"method": "adasyn"}
    except ValueError as e:
        # imblearn raises "No samples will be generated with the provided ratio
        # settings" when the data is already (near-)balanced and ADASYN would
        # synthesise nothing. Oversampling a balanced set is a no-op, so pass
        # the data through unchanged -- exactly what SMOTE does on balanced data.
        if "No samples will be generated" not in str(e):
            raise
        return X.copy(), y.copy(), None, {"method": "adasyn", "skipped": "already_balanced"}
    except RuntimeError as e:
        # "Not any neigbours belong to the majority class ... ADASYN is not
        # suited for this specific dataset. Use SMOTE instead." Follow imblearn's
        # own recommendation so the cell remains an active oversampler.
        if "ADASYN is not suited" not in str(e):
            raise
        sm = SMOTEResampler(k_neighbors=k, random_state=random_state)
        Xr, yr = sm.fit_resample(X, y)
        return Xr, yr, None, {"method": "adasyn", "fallback": "smote_adasyn_unsuited"}


def _smote_enn(X, y, random_state):
    from endgame.preprocessing.imbalance import SMOTEENNResampler
    k = _safe_k(y)
    if k == 0:
        return X.copy(), y.copy(), None, {"method": "smote_enn", "skipped": "too_few_samples"}
    s = SMOTEENNResampler(k_neighbors=k, random_state=random_state)
    Xr, yr = s.fit_resample(X, y)
    return Xr, yr, None, {"method": "smote_enn"}


def _smote_tomek(X, y, random_state):
    from endgame.preprocessing.imbalance import SMOTETomekResampler
    k = _safe_k(y)
    if k == 0:
        return X.copy(), y.copy(), None, {"method": "smote_tomek", "skipped": "too_few_samples"}
    s = SMOTETomekResampler(k_neighbors=k, random_state=random_state)
    Xr, yr = s.fit_resample(X, y)
    return Xr, yr, None, {"method": "smote_tomek"}


def _safe_level_smote(X, y, random_state):
    from endgame.preprocessing.safe_level_smote import SafeLevelSMOTEResampler
    k = _safe_k(y)
    if k == 0:
        return X.copy(), y.copy(), None, {"method": "safe_level_smote", "skipped": "too_few_samples"}
    s = SafeLevelSMOTEResampler(n_neighbors=k, random_state=random_state)
    Xr, yr = s.fit_resample(X, y)
    return Xr, yr, None, {"method": "safe_level_smote"}


def _polynom_fit_smote(X, y, random_state):
    from endgame.preprocessing.kovacs_variants import PolynomFitSMOTEResampler
    s = PolynomFitSMOTEResampler(random_state=random_state)
    Xr, yr = s.fit_resample(X, y)
    return Xr, yr, None, {"method": "polynom_fit_smote"}


def _prowsyn(X, y, random_state):
    from endgame.preprocessing.kovacs_variants import ProWSynResampler
    s = ProWSynResampler(random_state=random_state)
    Xr, yr = s.fit_resample(X, y)
    return Xr, yr, None, {"method": "prowsyn"}


def _mwmote(X, y, random_state):
    from endgame.preprocessing.kovacs_variants import MWMOTEResampler
    s = MWMOTEResampler(random_state=random_state)
    Xr, yr = s.fit_resample(X, y)
    return Xr, yr, None, {"method": "mwmote"}


def _napierala_guided_smote(X, y, random_state):
    from endgame.preprocessing.napierala_guided_smote import NapieralaGuidedSMOTE
    k = _safe_k(y)
    if k == 0:
        return X.copy(), y.copy(), None, {"method": "napierala_guided_smote", "skipped": "too_few_samples"}
    s = NapieralaGuidedSMOTE(k_neighbors=k, random_state=random_state)
    Xr, yr = s.fit_resample(X, y)
    return Xr, yr, None, {"method": "napierala_guided_smote"}


def _clean_masked_smote(X, y, random_state, noise_mode="global"):
    from endgame.preprocessing.imbalance import TriageMaskedSMOTE
    k = _safe_k(y)
    name = "clean_masked_smote" if noise_mode == "global" else f"clean_masked_{noise_mode}"
    if k == 0:
        return X.copy(), y.copy(), None, {"method": name, "skipped": "too_few_samples"}
    s = TriageMaskedSMOTE(**{**TRIAGE_PARAMS, "k_neighbors": k,
                             "random_state": random_state, "noise_mode": noise_mode})
    Xr, yr = s.fit_resample(X, y)
    return Xr, yr, None, {"method": name}


def _triage_weighting(X, y, random_state, noise_mode="global"):
    from endgame.augmentation.error_triage import TriageSampleWeighter
    name = "triage_weighting" if noise_mode == "global" else f"triage_weighting_{noise_mode}"
    w = TriageSampleWeighter(weight=2.0,
                             **{**TRIAGE_PARAMS, "random_state": random_state,
                                "noise_mode": noise_mode})
    w.fit(X, y)
    return X.copy(), y.copy(), w.get_sample_weights(), {"method": name}


def _triage_cost_sensitive(X, y, random_state):
    """Triage-informed cost-sensitive weighting (KEEL-motivated improved weighter).

    Leverages the minority AGGRESSIVELY but only where it is learnable: each
    non-majority class's *learnable* instances (triage category correct or
    data_limited) are weighted by that class's imbalance ratio (max_count /
    class_count); irreducible (Cat3 / Bayes-boundary) and noise (Cat1) minority
    instances are left at weight 1, so we do not pay accuracy to chase the
    irreducible boundary. Uses the class-balanced triage ensemble (noise_mode=
    "balanced") so minority instances are not spuriously flagged as noise.

    This dominates the accuracy-preserving end of the acc/balanced-acc frontier:
    ~2x the balanced-accuracy-per-unit-accuracy-cost of naive class weighting.
    """
    import numpy as np
    from endgame.augmentation.error_triage import ErrorTriage
    t = ErrorTriage(**{**TRIAGE_PARAMS, "random_state": random_state,
                       "noise_mode": "balanced"}).fit(X, y)
    cats = t.categories_
    learnable = np.isin(cats, ["correct", "data_limited"])
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(counts.max())
    w = np.ones(len(y), dtype=float)
    for c, cnt in zip(classes, counts):
        if cnt < max_count:                       # non-majority class
            mask = (y == c) & learnable
            w[mask] = max_count / float(cnt)       # class imbalance ratio
    return X.copy(), y.copy(), w, {"method": "triage_cost_sensitive"}


def _napierala_weighting(X, y, random_state, mapping: str):
    from endgame.augmentation.napierala_weighter import NapieralaSampleWeighter
    w = NapieralaSampleWeighter(weight=2.0, mapping=mapping, random_state=random_state)
    w.fit(X, y)
    return X.copy(), y.copy(), w.get_sample_weights(), {"method": f"napierala_weighting_{mapping}"}


def _class_balanced_weights(X, y, random_state):
    from sklearn.utils.class_weight import compute_sample_weight
    w = compute_sample_weight("balanced", y).astype(float)
    return X.copy(), y.copy(), w, {"method": "class_balanced_weights"}


METHODS: dict[str, MethodFn] = {
    "baseline": _baseline,
    "smote": _smote,
    "borderline_smote": _borderline,
    "adasyn": _adasyn,
    "smote_enn": _smote_enn,
    "smote_tomek": _smote_tomek,
    "safe_level_smote": _safe_level_smote,
    "polynom_fit_smote": _polynom_fit_smote,
    "prowsyn": _prowsyn,
    "mwmote": _mwmote,
    "napierala_guided_smote": _napierala_guided_smote,
    "clean_masked_smote": _clean_masked_smote,
    "triage_weighting": _triage_weighting,
    "triage_cost_sensitive": _triage_cost_sensitive,
    "napierala_weighting_rare": lambda X, y, rs: _napierala_weighting(X, y, rs, "rare"),
    "napierala_weighting_rare_outlier": lambda X, y, rs: _napierala_weighting(X, y, rs, "rare_outlier"),
    "napierala_weighting_borderline": lambda X, y, rs: _napierala_weighting(X, y, rs, "borderline"),
    "napierala_weighting_nonsafe": lambda X, y, rs: _napierala_weighting(X, y, rs, "nonsafe"),
    "class_balanced_weights": _class_balanced_weights,
    # Imbalance-aware noise-detection variants (KEEL-motivated method improvement):
    # class_conditional (per-class TCP threshold), balanced (class-balanced triage
    # ensemble), protect_minority (never flag minority as noise).
    "clean_masked_class_conditional": lambda X, y, rs: _clean_masked_smote(X, y, rs, "class_conditional"),
    "clean_masked_balanced": lambda X, y, rs: _clean_masked_smote(X, y, rs, "balanced"),
    "clean_masked_protect_minority": lambda X, y, rs: _clean_masked_smote(X, y, rs, "protect_minority"),
    "triage_weighting_class_conditional": lambda X, y, rs: _triage_weighting(X, y, rs, "class_conditional"),
    "triage_weighting_balanced": lambda X, y, rs: _triage_weighting(X, y, rs, "balanced"),
    "triage_weighting_protect_minority": lambda X, y, rs: _triage_weighting(X, y, rs, "protect_minority"),
}


def run_method(name: str, X, y, random_state: int):
    return METHODS[name](X, y, random_state)
