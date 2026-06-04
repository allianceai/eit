#!/usr/bin/env python
"""Tier-stratified analysis: does the accuracy / balanced-accuracy tradeoff (and our
fix) hold on genuinely IMBALANCED data?  (Reviewer #3: "some datasets seem balanced?
why not only imbalanced benchmarks?"; Reviewer #1: "in balanced problems balanced
accuracy acts similarly to accuracy".)

Stratifies the broad-method sweep (results/paper_revision/main_benchmark, XGBoost) by
imbalance-ratio tier and reports, per tier, the headline comparisons:
  - smote vs baseline          (the SMOTE tradeoff: accuracy down / balanced-acc up)
  - triage_weighting vs baseline (our Pareto weighting)
  - clean_masked_smote vs smote  (our masked SMOTE)

Writes paper_v2/tables/tier_analysis.tex and results/paper_revision/tier_analysis.parquet.
Run after the sweep:  python -m scripts.paper_revision.build_tier_analysis
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.paper_revision.config import RESULTS_DIR
from scripts.paper_revision.datasets import DATASETS, load_dataset

warnings.filterwarnings("ignore")
MAIN = RESULTS_DIR / "main_benchmark"
TABLES = Path("paper_v2/tables")
TABLES.mkdir(parents=True, exist_ok=True)

# Imbalance-ratio tiers (majority/minority count on the full dataset).
TIERS = [("balanced", 1.0, 1.5), ("mild", 1.5, 3.0),
         ("moderate", 3.0, 9.0), ("high", 9.0, np.inf)]


def _imbalance_ratios() -> dict[str, float]:
    out = {}
    for spec in DATASETS:
        try:
            _, y = load_dataset(spec)
            _, c = np.unique(y, return_counts=True)
            out[spec.name] = float(c.max() / c.min())
        except Exception:
            continue
    return out


def _tier(ir: float) -> str:
    for name, lo, hi in TIERS:
        if lo <= ir < hi:
            return name
    return "high"


def _paired(piv: pd.DataFrame, method: str, ref: str, datasets) -> dict:
    sub = piv.loc[piv.index.intersection(datasets), [ref, method]].dropna()
    d = sub[method] - sub[ref]
    if len(d) < 3:
        return {"n": len(d), "delta_pp": np.nan, "win_rate": np.nan, "p": np.nan}
    p = wilcoxon(d).pvalue if (d != 0).any() else np.nan
    return {"n": int(len(d)), "delta_pp": float(d.mean() * 100),
            "win_rate": float((d > 0).mean() * 100), "p": float(p)}


def main():
    files = list(MAIN.glob("xgboost__*.parquet"))
    if not files:
        raise FileNotFoundError(f"No xgboost cells in {MAIN}; run the sweep first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ir = _imbalance_ratios()
    pm = df.groupby(["method", "dataset"])[["accuracy", "balanced_accuracy"]].mean().reset_index()

    comparisons = [("smote", "baseline", "SMOTE tradeoff"),
                   ("triage_weighting", "baseline", "triage weighting"),
                   ("clean_masked_smote", "smote", "clean-masked vs SMOTE")]

    rows = []
    for tier_name, lo, hi in TIERS:
        ds_tier = [d for d, r in ir.items() if lo <= r < hi]
        for metric in ["accuracy", "balanced_accuracy"]:
            piv = pm.pivot(index="dataset", columns="method", values=metric)
            for m, ref, label in comparisons:
                if m not in piv.columns or ref not in piv.columns:
                    continue
                res = _paired(piv, m, ref, ds_tier)
                rows.append({"tier": tier_name, "metric": metric, "comparison": label,
                             "n_datasets": res["n"], "delta_pp": res["delta_pp"],
                             "win_rate": res["win_rate"], "wilcoxon_p": res["p"]})

    out = pd.DataFrame(rows)
    out.to_parquet(RESULTS_DIR / "tier_analysis.parquet")

    # Compact LaTeX: balanced-accuracy panel (the metric that differs from accuracy on
    # imbalanced data) for the three comparisons across tiers.
    lines = [r"\begin{tabular}{llrrrr}\toprule",
             r"Tier & Comparison & $n$ & $\Delta$BAcc(pp) & WR(\%) & $p$ \\\midrule"]
    for tier_name, _, _ in TIERS:
        for _, _, label in comparisons:
            r = out[(out.tier == tier_name) & (out.metric == "balanced_accuracy")
                    & (out.comparison == label)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"{tier_name} & {label} & {int(r.n_datasets)} & "
                         f"{r.delta_pp:+.3f} & {r.win_rate:.0f} & {r.wilcoxon_p:.3f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (TABLES / "tier_analysis.tex").write_text("\n".join(lines))

    n_imb = sum(1 for r in ir.values() if r >= 1.5)
    print(f"tier_analysis: {len(ir)} datasets, {n_imb} imbalanced (IR>=1.5). "
          f"Wrote {RESULTS_DIR/'tier_analysis.parquet'} and {TABLES/'tier_analysis.tex'}")
    print(out[out.metric == "balanced_accuracy"].to_string(index=False))


if __name__ == "__main__":
    main()
