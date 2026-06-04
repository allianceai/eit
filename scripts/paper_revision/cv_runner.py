"""5×5 RepeatedStratifiedKFold harness for one (method, dataset, classifier) cell."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from imblearn.metrics import geometric_mean_score
from scripts.paper_revision.config import (
    N_REPEATS, N_FOLDS, RANDOM_STATE, MAX_INSTANCES,
    XGB_PARAMS, RF_PARAMS, LGBM_PARAMS, LR_PARAMS, SVM_PARAMS, SVM_MAX_INSTANCES,
)
from scripts.paper_revision.methods import run_method


def _make_classifier(name: str):
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**XGB_PARAMS)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**RF_PARAMS)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**LGBM_PARAMS)
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**LR_PARAMS)
    if name == "svm":
        # RBF SVM needs scaled features; wrap in a pipeline. sample_weight is
        # forwarded to the SVC step by name (svc__sample_weight) in evaluate().
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        return Pipeline([("scaler", StandardScaler()), ("svc", SVC(**SVM_PARAMS))])
    raise ValueError(name)


def _stratified_subsample(X, y, cap, random_state):
    if len(X) <= cap:
        return X, y
    rng = np.random.default_rng(random_state)
    idx_keep = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        frac = cap / len(X)
        # Never request more than the class actually has (rare/singleton classes
        # otherwise hit "Cannot take a larger sample than population"): cap n_keep
        # at the class size.
        n_keep = min(len(cls_idx), max(2, int(len(cls_idx) * frac)))
        idx_keep.append(rng.choice(cls_idx, size=n_keep, replace=False))
    idx = np.concatenate(idx_keep)
    return X[idx], y[idx]


def evaluate_method_on_dataset(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    dataset_name: str,
    classifier: str = "xgboost",
    n_repeats: int = N_REPEATS,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
    subsample: bool = True,
) -> pd.DataFrame:
    """Returns a long-form dataframe: one row per (repeat, fold)."""
    if subsample:
        cap = SVM_MAX_INSTANCES if classifier == "svm" else MAX_INSTANCES
        X, y = _stratified_subsample(X, y, cap, random_state)

    # Adaptive splits (matches the original harness): a class with fewer members than
    # n_folds cannot be stratified into n_folds, so cap at the smallest class count.
    counts = np.bincount(np.asarray(y).astype(int))
    min_class_count = int(counts[counts > 0].min())
    actual_splits = max(2, min(n_folds, min_class_count))

    rskf = RepeatedStratifiedKFold(n_splits=actual_splits, n_repeats=n_repeats,
                                   random_state=random_state)
    rows = []
    for split_i, (tr, te) in enumerate(rskf.split(X, y)):
        repeat = split_i // actual_splits
        fold = split_i % actual_splits
        fold_seed = random_state + split_i

        Xtr, ytr = X[tr], y[tr]
        t0 = time.perf_counter()
        Xtr_r, ytr_r, w, info = run_method(method, Xtr, ytr, fold_seed)
        triage_time = time.perf_counter() - t0

        # Encode on the TRAIN classes only so XGBoost sees a contiguous [0..k-1].
        # (Encoding on the train/test union breaks when a rare class lands only in the
        # test fold, leaving the training labels non-contiguous -- which newer XGBoost
        # rejects.) A test instance whose true class is absent from the resampled train
        # set can never be predicted, so it is counted as an error -- the correct
        # outcome. Metrics are computed in the ORIGINAL label space, so test-only classes
        # are scored honestly; for the common case (train and test share all classes)
        # this is identical to the union encoding, and these metrics are encoding-invariant.
        le = LabelEncoder().fit(ytr_r)
        ytr_r_enc = le.transform(ytr_r)

        clf = _make_classifier(classifier)
        if w is not None:
            # Pipelines (svm) need the weight routed to the final estimator by name.
            fit_kw = {"svc__sample_weight": w} if classifier == "svm" else {"sample_weight": w}
            clf.fit(Xtr_r, ytr_r_enc, **fit_kw)
        else:
            clf.fit(Xtr_r, ytr_r_enc)
        yp = le.inverse_transform(clf.predict(X[te]))  # decode to original labels
        yte = y[te]
        rows.append({
            "dataset": dataset_name,
            "method": method,
            "classifier": classifier,
            "repeat": repeat,
            "fold": fold,
            "accuracy": accuracy_score(yte, yp),
            "balanced_accuracy": balanced_accuracy_score(yte, yp),
            "f1_macro": f1_score(yte, yp, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(yte, yp),
            "g_mean": geometric_mean_score(yte, yp, average="macro", correction=0.001),
            "triage_time_s": triage_time,
            "n_train_in": len(ytr),
            "n_train_out": len(ytr_r),
        })
    return pd.DataFrame(rows)
