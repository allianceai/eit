#!/usr/bin/env python
"""Build all paper figures from results parquets.

fig1 (interventional) is TIER-1 (results/original_study/interventional_results).
fig2 (Pareto plane) and fig_cd (Friedman--Nemenyi critical-difference diagram)
are TIER-2 (the additive xgboost broad-method sweep), filtered to the recovered
54-dataset roster. The dropped Bayes-boundary demo figures are no longer built.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.paper_revision.config import RESULTS_DIR
from scripts.paper_revision.datasets import DATASETS

FIG_DIR = Path("paper_v2/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
MAIN_DIR = RESULTS_DIR / "main_benchmark"
ORIG_DIR = Path("results/original_study")
ROSTER = {d.name for d in DATASETS}


def _load_xgb() -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(f) for f in MAIN_DIR.glob("xgboost__*.parquet")],
                   ignore_index=True)
    return df[df["dataset"].isin(ROSTER)].reset_index(drop=True)


def fig1_interventional():
    """Mean Δ vs no-augmentation baseline by augmentation strategy (TIER-1)."""
    path = ORIG_DIR / "interventional_results.parquet"
    if not path.exists():
        print("WARN: interventional results missing; skipping fig1")
        return
    df = pd.read_parquet(path)
    metrics = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced acc."}
    pm = df.groupby(["dataset", "method"])[list(metrics)].mean().reset_index()
    base = pm[pm["method"] == "baseline"].set_index("dataset")[list(metrics)]
    strategies = ["augment_cat3", "augment_cat2", "augment_all_errors",
                  "augment_random", "remove_noise", "remove_random"]
    rows = {}
    for m in strategies:
        sub = pm[pm["method"] == m].set_index("dataset")[list(metrics)]
        common = base.index.intersection(sub.index)
        delta = (sub.loc[common] - base.loc[common]).mean()
        rows[m] = delta
    mat = pd.DataFrame(rows).T[list(metrics)] * 100.0  # percentage points
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    vmax = float(np.nanmax(np.abs(mat.values))) or 1.0
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(metrics.values())
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([s.replace("_", " ") for s in mat.index])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.values[i, j]:+.2f}", ha="center", va="center",
                    fontsize=8)
    plt.colorbar(im, label=r"Mean $\Delta$ vs.\ baseline (percentage points)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_interventional.pdf", bbox_inches="tight")
    plt.close()


def fig2_pareto():
    """Accuracy vs balanced-accuracy plane across all methods (TIER-2, 54 roster)."""
    df = _load_xgb()
    means = df.groupby("method")[["accuracy", "balanced_accuracy"]].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    categories = {
        "Triage weighting (ours)": ["triage_weighting"],
        "Napierala weighting": [m for m in means.index if m.startswith("napierala_weighting_")],
        "Clean-masked SMOTE (ours)": ["clean_masked_smote"],
        "Napierala-guided SMOTE": ["napierala_guided_smote"],
        "SMOTE family": ["smote", "borderline_smote", "adasyn", "safe_level_smote",
                         "polynom_fit_smote", "prowsyn", "mwmote"],
        "Baseline": ["baseline"],
    }
    colors = {"Triage weighting (ours)": "C0", "Napierala weighting": "C3",
              "Clean-masked SMOTE (ours)": "C2", "Napierala-guided SMOTE": "C4",
              "SMOTE family": "C1", "Baseline": "gray"}
    markers = {"Triage weighting (ours)": "*", "Napierala weighting": "D",
               "Clean-masked SMOTE (ours)": "s", "Napierala-guided SMOTE": "p",
               "SMOTE family": "^", "Baseline": "o"}
    sizes = {"Triage weighting (ours)": 240, "Clean-masked SMOTE (ours)": 130}
    for grp, methods in categories.items():
        sub = means[means.index.isin(methods)]
        if sub.empty:
            continue
        ax.scatter(sub["accuracy"], sub["balanced_accuracy"], label=grp,
                   c=colors[grp], marker=markers[grp], s=sizes.get(grp, 90),
                   edgecolors="k", zorder=3)
    # guide lines through the baseline
    if "baseline" in means.index:
        ax.axvline(means.loc["baseline", "accuracy"], color="gray", ls=":", lw=0.8, zorder=1)
        ax.axhline(means.loc["baseline", "balanced_accuracy"], color="gray", ls=":", lw=0.8, zorder=1)
    ax.set_xlabel("Mean accuracy"); ax.set_ylabel("Mean balanced accuracy")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_pareto.pdf", bbox_inches="tight")
    plt.close()


# Nemenyi critical values q_alpha (alpha=0.05), already divided by sqrt(2)
# (Demsar 2006, Table 5). Index by number of methods k.
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
        9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}


def fig_cd(metric="accuracy"):
    """Friedman--Nemenyi critical-difference diagram over the broad-method sweep.

    Ranks the oversamplers + baseline + triage methods per dataset (1 = best on
    `metric`), averages ranks, runs the Friedman omnibus test, and draws the
    Demsar CD diagram (methods within one critical difference are connected).
    """
    from scipy.stats import friedmanchisquare
    df = _load_xgb()
    methods = ["baseline", "smote", "borderline_smote", "adasyn", "safe_level_smote",
               "polynom_fit_smote", "prowsyn", "mwmote", "napierala_guided_smote",
               "clean_masked_smote", "triage_weighting"]
    methods = [m for m in methods if m in set(df["method"].unique())]
    pm = (df[df["method"].isin(methods)]
          .groupby(["dataset", "method"])[metric].mean().unstack())
    pm = pm.dropna(axis=0, how="any")[methods]  # complete cases, fixed col order
    N, k = pm.shape
    # ranks per dataset: higher metric = better = rank 1
    ranks = pm.rank(axis=1, ascending=False, method="average")
    avg = ranks.mean(axis=0).sort_values()
    chi, p = friedmanchisquare(*[pm[m].values for m in methods])
    cd = _Q05.get(k, 3.4) * np.sqrt(k * (k + 1) / (6.0 * N))

    # ---- draw ----
    names = list(avg.index); rk = avg.values
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * k + 1.4))
    ax.set_xlim(lo - 0.5, hi + 0.5); ax.set_ylim(0, k + 2)
    ax.axis("off")
    # top axis
    ax.plot([lo, hi], [k + 1, k + 1], "k-", lw=1)
    for x in range(lo, hi + 1):
        ax.plot([x, x], [k + 1, k + 1.15], "k-", lw=1)
        ax.text(x, k + 1.35, str(x), ha="center", va="bottom", fontsize=8)
    ax.text((lo + hi) / 2, k + 1.7, f"average rank ({metric.replace('_',' ')})",
            ha="center", fontsize=9)
    # CD bar
    ax.plot([lo, lo + cd], [k + 0.4, k + 0.4], "k-", lw=2)
    ax.plot([lo, lo], [k + 0.32, k + 0.48], "k-", lw=1)
    ax.plot([lo + cd, lo + cd], [k + 0.32, k + 0.48], "k-", lw=1)
    ax.text(lo + cd / 2, k + 0.55, f"CD = {cd:.2f}", ha="center", fontsize=8)
    # method labels (lower rank on the left)
    for i, (nm, r) in enumerate(zip(names, rk)):
        y = k - i
        side_x = lo - 0.4 if r < (lo + hi) / 2 else hi + 0.4
        ha = "right" if r < (lo + hi) / 2 else "left"
        ax.plot([r, r], [k + 1, y], "k-", lw=0.8)
        ax.plot([r, side_x], [y, y], "k-", lw=0.8)
        star = " *" if nm in ("triage_weighting", "clean_masked_smote") else ""
        ax.text(side_x + (-0.05 if ha == "right" else 0.05), y,
                nm.replace("_", " ") + star, ha=ha, va="center", fontsize=8)
    # cliques: connect consecutive methods within CD
    yb = 0.6
    i = 0
    while i < k:
        j = i
        while j + 1 < k and (rk[j + 1] - rk[i]) <= cd:
            j += 1
        if j > i:
            ax.plot([rk[i] - 0.03, rk[j] + 0.03], [yb, yb], "r-", lw=3)
            yb += 0.35
        i += 1
    ax.text((lo + hi) / 2, 0.1,
            f"Friedman $\\chi^2$={chi:.1f}, $p$={p:.1e}, $N$={N}, $k$={k}",
            ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_cd.pdf", bbox_inches="tight")
    plt.close()
    print(f"  fig_cd: Friedman p={p:.2e}, CD={cd:.2f}, "
          f"best={names[0]}({rk[0]:.2f}), triage_weighting rank="
          f"{avg.get('triage_weighting', float('nan')):.2f}")


def fig_overlap(scatter_sep=0.6):
    """Mechanism figure: SMOTE generates synthetic minority points ACROSS the Bayes
    boundary (x=0) into majority territory; clean-masked does not. Left/middle panels
    show one overlap level; right panel the wrong-side fraction vs separation.
    Built only when run_overlap_synthetic.py has produced the full sweep."""
    path = RESULTS_DIR / "overlap_synthetic.parquet"
    if not path.exists():
        print("  skip fig_overlap: overlap_synthetic.parquet missing")
        return
    d = pd.read_parquet(path)
    seps = sorted(d["separation"].unique())
    from scripts.paper_revision.methods import run_method
    rng = np.random.default_rng(0)
    s = scatter_sep
    Xmaj = rng.normal([-s, 0.0], 1.0, (900, 2)); Xmin = rng.normal([+s, 0.0], 1.0, (100, 2))
    X = np.vstack([Xmaj, Xmin]); y = np.array([0]*900 + [1]*100)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, method, title in [(axes[0], "smote", "Standard SMOTE"),
                              (axes[1], "clean_masked_smote", "Clean-masked SMOTE")]:
        Xr, yr, _, _ = run_method(method, X, y, 0)
        res_min = Xr[yr == 1]; syn = res_min[100:] if len(res_min) > 100 else res_min[:0]
        ax.scatter(Xmaj[:, 0], Xmaj[:, 1], s=5, c="0.7", label="majority")
        ax.scatter(Xmin[:, 0], Xmin[:, 1], s=14, c="C0", label="minority (real)")
        if len(syn):
            wrong = syn[:, 0] < 0
            ax.scatter(syn[wrong, 0], syn[wrong, 1], s=18, c="C3", marker="x",
                       label="synthetic, wrong side")
            ax.scatter(syn[~wrong, 0], syn[~wrong, 1], s=10, c="C1", marker="+",
                       label="synthetic, correct side")
        ax.axvline(0, c="g", lw=2, label="Bayes boundary")
        ax.set_title(title, fontsize=10); ax.set_xlim(-4, 4); ax.set_ylim(-3.5, 3.5)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=6, loc="upper left")

    ax = axes[2]
    g = d.groupby("separation")[["wrong_side_smote", "wrong_side_clean_masked"]].mean()
    ax.plot(g.index, 100*g["wrong_side_smote"], "o-", c="C3", label="Standard SMOTE")
    ax.plot(g.index, 100*g["wrong_side_clean_masked"], "s-", c="C2", label="Clean-masked")
    ax.axvline(scatter_sep, c="0.8", ls=":", lw=1)
    ax.set_xlabel("class separation $s$ (more overlap $\\leftarrow$)")
    ax.set_ylabel("synthetic points across\nBayes boundary (\\%)")
    ax.legend(fontsize=8); ax.set_title("Boundary-crossing generation", fontsize=10)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_overlap.pdf", bbox_inches="tight")
    plt.close()
    print(f"  fig_overlap: seps={seps}; "
          f"wrong-side SMOTE {100*g['wrong_side_smote'].mean():.1f}% "
          f"vs clean {100*g['wrong_side_clean_masked'].mean():.1f}%")


def main():
    for fn in (fig1_interventional, fig2_pareto, fig_cd, fig_overlap):
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {e}")
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
