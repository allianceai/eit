#!/usr/bin/env python
"""IR-stratified + binary/multiclass + high-overlap re-analysis (reviewer #4).

Reviewer #4: the 79-dataset headline mixes in many near-balanced datasets
(IR ~ 1.0-1.5) that are not meaningfully imbalanced-learning problems, which can
dilute the conclusion. This re-analyses the EXISTING benchmark cells (no new
runs) within imbalance strata and by task type, and shows the non-generative
dominance + the SMOTE accuracy/balanced-accuracy tradeoff in each subset.

Strata: IR>1.5, IR>3, IR>10 ; binary-only ; multiclass-only ;
        high-overlap (top tercile of Lorena N1 = boundary-point fraction).

Outputs a console report and a LaTeX table (paper_v2/tables/table_ir_strata.tex).

    python -m scripts.paper_revision.build_ir_strata
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.paper_revision.config import RESULTS_DIR
from scripts.paper_revision.build_frontier import _collect, GENERATIVE, NONGEN

TAB_DIR = Path("paper_v2/tables")
META = RESULTS_DIR / "meta_features.parquet"


def _load():
    df = pd.concat([_collect("keel"), _collect("original")], ignore_index=True)
    meta = pd.read_parquet(META)[["dataset", "ir", "n_classes", "N1"]].drop_duplicates("dataset")
    df = df.merge(meta, on="dataset", how="left")
    return df, meta


def _dominance(sub):
    """Per dataset: best non-gen vs best gen balanced accuracy."""
    wins = tot = 0
    adv = []
    for ds, g in sub.groupby("dataset"):
        gen = g[g.method.isin(GENERATIVE)]
        ng = g[g.method.isin(NONGEN)]
        if gen.empty or ng.empty:
            continue
        tot += 1
        d = ng.balanced_accuracy.max() - gen.balanced_accuracy.max()
        adv.append(d)
        if d >= 0:
            wins += 1
    p = float("nan")
    if len(adv) >= 6:
        try:
            p = wilcoxon(adv).pvalue
        except ValueError:
            p = float("nan")
    return tot, wins, 100 * np.mean(adv) if adv else float("nan"), p


def _smote_tradeoff(sub):
    """SMOTE vs baseline mean dAcc/dBAcc (paired over datasets) within the subset."""
    piv = sub[sub.method.isin(["baseline", "smote"])].pivot_table(
        index="dataset", columns="method", values=["accuracy", "balanced_accuracy"])
    if ("accuracy", "smote") not in piv or ("accuracy", "baseline") not in piv:
        return float("nan"), float("nan"), 0
    piv = piv.dropna()
    da = 100 * (piv[("accuracy", "smote")] - piv[("accuracy", "baseline")]).mean()
    db = 100 * (piv[("balanced_accuracy", "smote")] - piv[("balanced_accuracy", "baseline")]).mean()
    return da, db, len(piv)


def main():
    df, meta = _load()
    n1_hi = meta["N1"].quantile(2 / 3)
    datasets_meta = df.drop_duplicates("dataset").set_index("dataset")

    strata = {
        "all": df,
        "IR>1.5": df[df.ir > 1.5],
        "IR>3": df[df.ir > 3],
        "IR>10": df[df.ir > 10],
        "binary": df[df.n_classes == 2],
        "multiclass": df[df.n_classes > 2],
        "high-overlap (N1 top 1/3)": df[df.N1 >= n1_hi],
    }

    print("=== Non-generative dominance + SMOTE tradeoff by stratum (existing cells) ===")
    print(f"{'stratum':28s} {'nDS':>4s} {'NG>=G WR':>9s} {'bacc adv':>9s} {'p':>8s} "
          f"{'SMOTE dAcc':>11s} {'SMOTE dBAcc':>12s}")
    rows = []
    for name, sub in strata.items():
        tot, wins, adv, p = _dominance(sub)
        da, db, n_sm = _smote_tradeoff(sub)
        wr = 100 * wins / tot if tot else float("nan")
        print(f"{name:28s} {tot:4d} {wr:8.0f}% {adv:+8.2f}pp {p:8.2g} "
              f"{da:+10.2f}pp {db:+11.2f}pp")
        rows.append(dict(stratum=name, n=tot, ng_win_rate=wr, bacc_adv=adv, p=p,
                         smote_dacc=da, smote_dbacc=db, n_smote=n_sm))

    res = pd.DataFrame(rows)
    res.to_parquet(RESULTS_DIR / "ir_strata.parquet")

    # LaTeX table (the imbalanced strata are the headline)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    show = ["IR>1.5", "IR>3", "IR>10", "binary", "multiclass", "high-overlap (N1 top 1/3)"]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Subset & $n$ & NG$\ge$G & bacc adv. & SMOTE $\Delta$acc & SMOTE $\Delta$bacc \\",
             r"\midrule"]
    label = {"IR>1.5": r"$\mathrm{IR}>1.5$", "IR>3": r"$\mathrm{IR}>3$",
             "IR>10": r"$\mathrm{IR}>10$", "binary": "binary only",
             "multiclass": "multiclass only", "high-overlap (N1 top 1/3)": "high-overlap"}
    for _, r in res[res.stratum.isin(show)].iterrows():
        lines.append(f"{label[r.stratum]} & {int(r.n)} & {r.ng_win_rate:.0f}\\% & "
                     f"{r.bacc_adv:+.2f} & {r.smote_dacc:+.2f} & {r.smote_dbacc:+.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / "table_ir_strata.tex").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {TAB_DIR / 'table_ir_strata.tex'} and {RESULTS_DIR / 'ir_strata.parquet'}")


if __name__ == "__main__":
    main()
