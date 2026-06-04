"""Aggregation: per-dataset means, Wilcoxon vs baseline, Holm–Bonferroni."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def per_dataset_means(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse fold rows to one row per (dataset, method)."""
    return (df.groupby(["dataset", "method"])
              [["accuracy", "balanced_accuracy", "f1_macro", "mcc", "g_mean"]].mean()
              .reset_index())


def pairwise_vs_baseline(df: pd.DataFrame, *, baseline: str,
                         metric: str = "accuracy") -> pd.DataFrame:
    means = per_dataset_means(df)
    pivot = means.pivot(index="dataset", columns="method", values=metric)
    base = pivot[baseline]
    rows = []
    for m in pivot.columns:
        if m == baseline:
            continue
        other = pivot[m]
        common = base.dropna().index.intersection(other.dropna().index)
        b = base.loc[common].to_numpy()
        o = other.loc[common].to_numpy()
        diff = o - b
        # Wilcoxon (drop zero diffs per default zero_method="wilcox" handles)
        try:
            stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat, p = np.nan, 1.0
        win = float(np.mean(o > b))
        rows.append({
            "method": m,
            "metric": metric,
            "n_datasets": len(common),
            "mean_baseline": float(b.mean()),
            "mean_method": float(o.mean()),
            "mean_delta": float(diff.mean()),
            "win_rate": win,
            "p_wilcoxon": float(p),
        })
    return pd.DataFrame(rows)


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Holm–Bonferroni step-down adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, p[idx] * (n - rank))
        adj[idx] = min(running, 1.0)
    return adj
