#!/usr/bin/env python
"""Triage-design ablation (reviewer #9): does Cat3 predict SMOTE harm better than
simpler hardness / overlap measures?

A reviewer will ask whether the aleatoric--epistemic triage is necessary or whether
a simple k-NN hardness / overlap score gives the same thing. We compare, across
datasets, how well each candidate per-dataset measure predicts SMOTE's accuracy
harm (and its balanced-accuracy gain), all from EXISTING cached artifacts
(meta_features.parquet + the benchmark cells -- no re-run):

  triage          : cat3_mass, frac_cat3
  Napierala       : nap_rare+nap_outlier (unsafe minority), nap_borderline
  instance-hard.  : max_fdr, N3 (LOO 1-NN error)
  overlap/geom.   : N1, N2, F1 (Fisher)
  trivial         : imbalance ratio (ir), minority_error_rate

Reports Spearman rho of each measure with SMOTE dAcc / dBAcc and a single-feature
leave-one-out R^2 for predicting dAcc, so we can state honestly whether Cat3 is a
uniquely good predictor (it is not -- the triage's value is explanatory, the
mechanism is carried by the intervention, not by cross-dataset prediction).

    python -m scripts.paper_revision.build_triage_ablation
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.paper_revision.config import RESULTS_DIR
from scripts.paper_revision.build_frontier import _collect

TAB_DIR = Path("paper_v2/tables")
META = RESULTS_DIR / "meta_features.parquet"

CANDIDATES = {
    "cat3_mass (triage)": "cat3_mass",
    "frac_cat3 (triage)": "frac_cat3",
    "Napierala rare+outlier": "_nap_unsafe",
    "Napierala borderline": "nap_borderline",
    "instance hardness (max_fdr)": "max_fdr",
    "LOO 1-NN error (N3)": "N3",
    "boundary frac (N1)": "N1",
    "intra/extra ratio (N2)": "N2",
    "Fisher F1": "F1",
    "imbalance ratio": "ir",
    "minority error rate": "minority_error_rate",
}


def _smote_deltas():
    df = pd.concat([_collect("keel"), _collect("original")], ignore_index=True)
    piv = df[df.method.isin(["baseline", "smote"])].pivot_table(
        index="dataset", columns="method", values=["accuracy", "balanced_accuracy"])
    piv = piv.dropna()
    out = pd.DataFrame({
        "dataset": piv.index,
        "dacc_smote": (piv[("accuracy", "smote")] - piv[("accuracy", "baseline")]).values,
        "dbacc_smote": (piv[("balanced_accuracy", "smote")] - piv[("balanced_accuracy", "baseline")]).values,
    })
    return out


def _loo_r2(x, y):
    """Leave-one-out R^2 of a single-feature OLS (honest, tiny-n safe)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 5 or np.std(x) < 1e-12:
        return float("nan")
    preds = np.empty(n)
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        b1, b0 = np.polyfit(x[tr], y[tr], 1)
        preds[i] = b0 + b1 * x[i]
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    meta = pd.read_parquet(META).drop_duplicates("dataset").copy()
    meta["_nap_unsafe"] = meta.get("nap_rare", 0).fillna(0) + meta.get("nap_outlier", 0).fillna(0)
    deltas = _smote_deltas()
    df = meta.merge(deltas, on="dataset", how="inner")
    print(f"n datasets with SMOTE outcome + meta-features: {len(df)}")

    rows = []
    for label, col in CANDIDATES.items():
        if col not in df.columns:
            continue
        rho_a, p_a = spearmanr(df[col], df["dacc_smote"], nan_policy="omit")
        rho_b, p_b = spearmanr(df[col], df["dbacc_smote"], nan_policy="omit")
        r2 = _loo_r2(df[col], df["dacc_smote"])
        rows.append(dict(measure=label, rho_dacc=rho_a, p_dacc=p_a,
                         rho_dbacc=rho_b, loo_r2_dacc=r2))
    res = pd.DataFrame(rows).reindex(
        columns=["measure", "rho_dacc", "p_dacc", "rho_dbacc", "loo_r2_dacc"])
    res["abs_rho_dacc"] = res.rho_dacc.abs()
    res = res.sort_values("abs_rho_dacc", ascending=False)
    res.to_parquet(RESULTS_DIR / "triage_ablation.parquet")

    print("\n=== Predicting SMOTE accuracy harm (dAcc, more negative = more harm) ===")
    print(f"{'measure':32s} {'rho(dAcc)':>10s} {'p':>9s} {'rho(dBAcc)':>11s} {'LOO R2(dAcc)':>13s}")
    for _, r in res.iterrows():
        print(f"{r.measure:32s} {r.rho_dacc:+10.3f} {r.p_dacc:9.2g} "
              f"{r.rho_dbacc:+11.3f} {r.loo_r2_dacc:13.3f}")

    # LaTeX table
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Per-dataset measure & $\rho$(SMOTE $\Delta$acc) & $\rho$($\Delta$bacc) & LOO $R^2$($\Delta$acc) \\",
             r"\midrule"]
    for _, r in res.iterrows():
        star = "$^{*}$" if r.p_dacc < 0.05 else ""
        r2s = "--" if not np.isfinite(r.loo_r2_dacc) else f"{r.loo_r2_dacc:.2f}"
        meas = str(r.measure).replace("_", r"\_")
        lines.append(f"{meas} & {r.rho_dacc:+.2f}{star} & {r.rho_dbacc:+.2f} & {r2s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / "table_triage_ablation.tex").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {TAB_DIR/'table_triage_ablation.tex'} and {RESULTS_DIR/'triage_ablation.parquet'}")


if __name__ == "__main__":
    main()
