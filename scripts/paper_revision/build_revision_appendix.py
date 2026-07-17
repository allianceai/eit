#!/usr/bin/env python
"""Appendix tables for the Neurocomputing revision experiments.

  table_nontree.tex : remedy decomposition by instrument (RF reference vs
                      bagged MLP vs bagged logistic regression) — R5/E1.
  table_msens.tex   : ensemble-size sensitivity of the decomposition — R6/E3b.

    python -m scripts.paper_revision.build_revision_appendix
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.paper_revision.config import RESULTS_DIR

TAB = Path("paper_v2/tables")
LABELS = {"rf": "Random forests (paper instrument)",
          "mlp_bag": "Bagged MLPs",
          "logreg_bag": "Bagged logistic regression"}


def table_nontree():
    # v2 sources (post external review): argmax error set + cross-fitted tau,
    # unweighted RF instrument (the headline instrument, reducibility_v2_global)
    from scripts.paper_revision.build_reducibility_v2 import _paper_roster
    rf = pd.read_parquet(RESULTS_DIR / "reducibility_v2_global.parquet")
    rf = _paper_roster(rf)
    rf = rf[rf.n_minority_err_argmax >= 5].copy()
    rf["instrument"] = "rf"
    nt = pd.read_parquet(RESULTS_DIR / "reducibility_nontree_v2.parquet")
    nt = _paper_roster(nt)
    nt = nt[nt.n_minority_err_default >= 5]
    rf_tr = rf.set_index(["dataset", "benchmark"])["frac_threshold_recoverable"]

    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Instrument & thr.-rec. & (median) & data-red. & irred. & noise & $r$ vs RF \\",
             r"\midrule"]
    for inst in ["rf", "mlp_bag", "logreg_bag"]:
        s = rf if inst == "rf" else nt[nt.instrument == inst]
        if inst == "rf":
            corr = "---"
        else:
            j = s.set_index(["dataset", "benchmark"])["frac_threshold_recoverable"].to_frame("tr").join(
                rf_tr.to_frame("tr_rf"), how="inner").dropna()
            corr = f"{np.corrcoef(j.tr, j.tr_rf)[0, 1]:.2f}"
        lines.append(
            f"{LABELS[inst]} & {100*s.frac_threshold_recoverable.mean():.0f}\\% & "
            f"({100*s.frac_threshold_recoverable.median():.0f}\\%) & "
            f"{100*s.frac_data_reducible.mean():.0f}\\% & "
            f"{100*s.frac_irreducible.mean():.0f}\\% & "
            f"{100*s.frac_noise.mean():.0f}\\% & {corr} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "table_nontree.tex").write_text("\n".join(lines) + "\n")
    print("wrote", TAB / "table_nontree.tex")


def table_msens():
    df = pd.read_parquet(RESULTS_DIR / "m_sensitivity_v2.parquet")
    ref = df[df.n_forests == 5].set_index("dataset")
    cat_cols = ["frac_errors_cat1", "frac_errors_cat2", "frac_errors_cat3"]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"$M$ & mean thr.-rec. & mean $|\Delta$thr.-rec.$|$ & "
             r"max $|\Delta$thr.-rec.$|$ & max $|\Delta$cat.\ fraction$|$ \\",
             r"\midrule"]
    for m in sorted(df.n_forests.unique()):
        cur = df[df.n_forests == m].set_index("dataset")
        common = ref.index.intersection(cur.index)
        d_tr = (cur.loc[common, "frac_threshold_recoverable"]
                - ref.loc[common, "frac_threshold_recoverable"]).abs() * 100
        d_cat = max(float((cur.loc[common, c] - ref.loc[common, c]).abs().max()) * 100
                    for c in cat_cols)
        mark = r" (ref.)" if m == 5 else ""
        lines.append(f"{m}{mark} & {100*cur.frac_threshold_recoverable.mean():.0f}\\% & "
                     f"{d_tr.mean():.1f}\\,pp & {d_tr.max():.1f}\\,pp & {d_cat:.1f}\\,pp \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "table_msens.tex").write_text("\n".join(lines) + "\n")
    print("wrote", TAB / "table_msens.tex")


if __name__ == "__main__":
    table_nontree()
    table_msens()
