#!/usr/bin/env python
"""Cheap, high-credibility additions:
 (A) Friedman-Nemenyi critical-difference diagram over the full 16-strategy menu
     (balanced accuracy) -> paper_v2/figures/fig_cd.pdf
 (B) Mediation: dataset overlap/boundary structure vs SMOTE's accuracy cost
     (links the controlled mechanism to the real-data outcome).
 (C) Separability: baseline out-of-fold AUC on the hardest (low balanced-accuracy)
     binary datasets, to substantiate the 'decision-threshold artifact' claim.

Usage:  python -m scripts.paper_revision.build_credibility
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata, spearmanr

from scripts.paper_revision.config import RESULTS_DIR
from scripts.paper_revision.meta_selection import build_matrices, FULL_MENU, metric_for

FIG = Path("paper_v2/figures")
NONGEN = {"baseline", "threshold_moved", "cost_sensitive", "triage_weighting",
          "balanced_rf", "easy_ensemble"}
LABEL = {"baseline": "baseline", "threshold_moved": "threshold-move",
         "cost_sensitive": "cost-sensitive", "balanced_rf": "balanced RF",
         "easy_ensemble": "EasyEnsemble", "triage_weighting": "triage-weight",
         "smote": "SMOTE", "borderline_smote": "Borderline", "adasyn": "ADASYN",
         "safe_level_smote": "Safe-Level", "prowsyn": "ProWSyn", "mwmote": "MWMOTE",
         "polynom_fit_smote": "polynom-fit", "clean_masked_smote": "clean-masked",
         "clean_masked_balanced": "clean-mask (bal)", "napierala_guided_smote": "Napierala"}


def _nemenyi_cd(k, N, alpha=0.05):
    try:
        from scipy.stats import studentized_range
        q = studentized_range.ppf(1 - alpha, k, np.inf)
    except Exception:  # fallback q_0.05 (Demsar table, interpolated) if SciPy lacks it
        q = 3.43
    return (q / np.sqrt(2)) * np.sqrt(k * (k + 1) / (6.0 * N))


def cd_diagram():
    feats, B, fc, menu = build_matrices(FULL_MENU, "balanced_accuracy")
    full = ~np.isnan(B).any(axis=1)             # datasets where all 16 strategies apply (binary)
    Bf = B[full]
    N, k = Bf.shape
    ranks = np.vstack([rankdata(-Bf[i]) for i in range(N)])   # rank 1 = best bacc
    avg = ranks.mean(0)
    chi, p = friedmanchisquare(*[Bf[:, j] for j in range(k)])
    CD = _nemenyi_cd(k, N)
    order = np.argsort(avg)
    print(f"\n=== CD diagram | full 16-strategy menu | balanced accuracy | N={N} binary datasets ===")
    print(f"Friedman chi2={chi:.1f}, p={p:.2e}; Nemenyi CD={CD:.2f}")
    for j in order:
        tag = "non-gen" if menu[j] in NONGEN else "gen"
        print(f"  [{tag:7s}] {menu[j]:24s} avg_rank={avg[j]:.2f}")
    _plot_cd(avg, menu, CD, N, chi, p)
    return avg, menu, CD


def _plot_cd(avg, menu, CD, N, chi, p):
    plt.rcParams.update({"font.family": "serif", "font.size": 12, "savefig.dpi": 300})
    k = len(menu)
    order = np.argsort(avg)
    lo, hi = 1, k
    sorted_avg = avg[order]
    # ---- maximal cliques only (a "not significantly different" group not contained
    # in a wider one) so we draw a handful of clean bars, not one per method ----
    raw = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and sorted_avg[j + 1] - sorted_avg[i] <= CD:
            j += 1
        if j > i:
            raw.append((i, j))
        i += 1
    cliques = [(a, b) for (a, b) in raw
               if not any(c <= a and b <= d and (c, d) != (a, b) for (c, d) in raw)]
    # dedicated band BELOW the method rows so bars never collide with labels
    band_gap, band_step = 0.9, 0.55
    y_band_top = -band_gap
    y_bottom = y_band_top - band_step * max(len(cliques) - 1, 0) - 0.6

    # taller canvas + more vertical room per method so the 16 labels do not collide
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    ax.set_xlim(lo - 0.5, hi + 0.5); ax.set_ylim(y_bottom, k + 2.6)
    ax.invert_xaxis()                       # best (low rank) on the right
    ax.hlines(k + 1, lo, hi, color="k", lw=1.2)
    for r in range(lo, hi + 1):
        ax.vlines(r, k + 0.9, k + 1.1, color="k", lw=1.2)
        ax.text(r, k + 1.45, str(r), ha="center", fontsize=11)
    ax.text((lo + hi) / 2, k + 2.25, "average rank (1 = best balanced accuracy)",
            ha="center", fontsize=11)
    # CD ruler
    ax.hlines(k + 1.9, lo, lo + CD, color="k", lw=2.4)
    ax.vlines([lo, lo + CD], k + 1.75, k + 2.05, color="k", lw=2.4)
    ax.text(lo + CD / 2, k + 2.18, f"CD = {CD:.2f}", ha="center", fontsize=11)
    for rank_i, j in enumerate(order):
        y = k - rank_i
        right = avg[j] < np.median(avg)
        xtext = lo if right else hi
        col = "C0" if menu[j] in NONGEN else "C1"
        ax.plot([avg[j], avg[j]], [k + 0.9, y], color=col, lw=1.3)
        ax.plot([avg[j], xtext], [y, y], color=col, lw=1.3)
        ax.text(xtext + (0.18 if right else -0.18), y,
                f"{LABEL.get(menu[j], menu[j])} ({avg[j]:.2f})",
                ha="left" if right else "right", va="center", fontsize=11.5,
                color=col, fontweight=("bold" if menu[j] in NONGEN else "normal"))
    # ---- draw the maximal cliques in the dedicated bottom band ----
    for ci, (a, b) in enumerate(cliques):
        yb = y_band_top - band_step * ci
        ax.hlines(yb, sorted_avg[a], sorted_avg[b], color="0.30", lw=5)
    if cliques:
        ax.text(hi + 0.45, y_band_top + band_step * 0.55,
                "groups joined by a bar are not\nsignificantly different (Nemenyi)",
                ha="left", va="top", fontsize=9.5, style="italic", color="0.30")
    ax.axis("off")
    ax.set_title("Critical-difference diagram, balanced accuracy "
                 f"($N={N}$ binary datasets, Friedman $p={p:.0e}$). "
                 "Bold/blue = non-generative; orange = generative (SMOTE family).",
                 fontsize=10.5)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(FIG / "fig_cd.pdf", bbox_inches="tight"); plt.close()
    print(f"wrote {FIG / 'fig_cd.pdf'}")


def mediation():
    mf = pd.read_parquet(RESULTS_DIR / "meta_features.parquet").drop_duplicates(["dataset", "benchmark"])
    rows = []
    for _, r in mf.iterrows():
        ba = metric_for(r.dataset, r.benchmark, "baseline", "accuracy")
        sa = metric_for(r.dataset, r.benchmark, "smote", "accuracy")
        if np.isnan(ba) or np.isnan(sa):
            continue
        rows.append({"dacc_smote": sa - ba, "N1": r.N1, "cat3_mass": r.cat3_mass,
                     "minority_error_rate": r.minority_error_rate, "F1": r.F1, "ir": r.ir})
    d = pd.DataFrame(rows)
    print(f"\n=== Mediation: overlap/boundary structure vs SMOTE accuracy cost (n={len(d)}) ===")
    print("(negative rho = more overlap/boundary -> larger SMOTE accuracy loss)")
    for f in ["N1", "cat3_mass", "minority_error_rate", "F1", "ir"]:
        rho, p = spearmanr(d[f], d["dacc_smote"])
        print(f"  rho({f:20s}, SMOTE dAcc) = {rho:+.2f}  (p={p:.4f})")


def separability():
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier
    from scripts.paper_revision.config import XGB_PARAMS, MAX_INSTANCES, RANDOM_STATE
    from scripts.paper_revision.cv_runner import _stratified_subsample
    from scripts.paper_revision.keel_datasets import KEEL_DATASETS, load_keel
    from scripts.paper_revision.datasets import DATASETS, load_dataset
    mf = pd.read_parquet(RESULTS_DIR / "meta_features.parquet").drop_duplicates(["dataset", "benchmark"])
    hard = mf[(mf.baseline_bacc < 0.85) & (mf.n_classes == 2)]
    print(f"\n=== Separability: baseline OOF AUC on hard binary datasets "
          f"(baseline bacc < 0.85), n={len(hard)} ===")
    load = {"keel": load_keel}
    specs = {d.name: d for d in DATASETS}
    aucs = []
    for _, r in hard.iterrows():
        try:
            if r.benchmark == "keel":
                X, y = load_keel(r.dataset)
            else:
                X, y = load_dataset(specs[r.dataset])
            X, y = _stratified_subsample(X, y, MAX_INSTANCES, RANDOM_STATE)
            ye = LabelEncoder().fit_transform(y)
            if len(np.unique(ye)) != 2:
                continue
            cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
            oof = cross_val_predict(XGBClassifier(**{**XGB_PARAMS, "n_jobs": 1}), X, ye,
                                    cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(ye, oof)
            aucs.append({"dataset": r.dataset, "baseline_bacc": r.baseline_bacc, "auc": auc})
            print(f"  {r.dataset:24s} bacc={r.baseline_bacc:.3f}  AUC={auc:.3f}", flush=True)
        except Exception as e:
            print(f"  ERR {r.dataset}: {e}", flush=True)
    a = pd.DataFrame(aucs)
    if len(a):
        print(f"\n  AUC on hard datasets: min={a.auc.min():.2f} median={a.auc.median():.2f} "
              f"max={a.auc.max():.2f}; {(a.auc>0.7).mean()*100:.0f}% have AUC>0.70 "
              f"despite balanced accuracy <0.85")
        a.to_parquet(RESULTS_DIR / "separability_auc.parquet")


if __name__ == "__main__":
    cd_diagram()
    mediation()
    separability()
