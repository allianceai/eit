#!/usr/bin/env python
"""Generate every LaTeX table referenced by the paper from results parquets.

Two result tiers (see ANALYSIS_NOTES.md / run_all.sh):
  - TIER 1  results/original_study/  -- the original paper's authoritative headline
            results (cost/timing here back Table~\\ref{tab:cost}).
  - TIER 2  results/paper_revision/main_benchmark/  -- the additive broad-method /
            classifier / Napierala sweep.

ALL main_benchmark loads are FILTERED to the recovered 54-dataset roster
(`scripts.paper_revision.datasets.DATASETS`) so that drift-dataset cells left on
disk from the earlier reconstruction sweep (arrhythmia, dna, libras, ...) never
enter the numbers. Each table is emitted as a complete float (caption + label) so
the prose `\\ref{tab:...}` resolves; the section .tex just `\\input`s it.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.paper_revision.config import RESULTS_DIR
from scripts.paper_revision.aggregate import pairwise_vs_baseline
from scripts.paper_revision.datasets import DATASETS

TABLES_DIR = Path("paper_v2/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
MAIN_DIR = RESULTS_DIR / "main_benchmark"
ORIG_DIR = Path("results/original_study")
ROSTER = {d.name for d in DATASETS}  # the recovered 54


def _load_method_classifier(classifier="xgboost") -> pd.DataFrame:
    files = list(MAIN_DIR.glob(f"{classifier}__*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found for classifier '{classifier}' in {MAIN_DIR}. "
            "Run the sweep first (run_parallel.py)."
        )
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # Restrict to the recovered 54 roster (drop drift cells from the old sweep).
    return df[df["dataset"].isin(ROSTER)].reset_index(drop=True)


def _stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _tex(s: str) -> str:
    """Escape underscores for LaTeX text."""
    return s.replace("_", r"\_")


def _emit(name, label, caption, colspec, header, body_rows, placement="t",
          fontsize=None, colsep=None):
    """Write a complete table float carrying its own caption + label."""
    lines = [
        f"\\begin{{table}}[{placement}]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        *([f"\\{fontsize}"] if fontsize else []),
        *([f"\\setlength{{\\tabcolsep}}{{{colsep}}}"] if colsep else []),
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES_DIR / name).write_text("\n".join(lines) + "\n")


def table2_weighting_comparison():
    df = _load_method_classifier("xgboost")
    weighting_methods = [
        "baseline", "triage_weighting",
        "napierala_weighting_rare", "napierala_weighting_rare_outlier",
        "napierala_weighting_borderline", "napierala_weighting_nonsafe",
    ]
    sub = df[df["method"].isin(weighting_methods)]
    acc = pairwise_vs_baseline(sub, baseline="baseline", metric="accuracy")
    bacc = pairwise_vs_baseline(sub, baseline="baseline", metric="balanced_accuracy")
    merged = acc.merge(bacc, on="method", suffixes=("_acc", "_bacc"))
    order = {m: i for i, m in enumerate(weighting_methods)}
    merged = merged.sort_values("method", key=lambda s: s.map(order))
    body = []
    for _, r in merged.iterrows():
        body.append(
            f"{_tex(r['method'])} & "
            f"{r['mean_method_acc']:.4f} & {100*r['win_rate_acc']:.1f}\\% & "
            f"{r['p_wilcoxon_acc']:.3f}{_stars(r['p_wilcoxon_acc'])} & "
            f"{r['mean_method_bacc']:.4f} & {100*r['win_rate_bacc']:.1f}\\% & "
            f"{r['p_wilcoxon_bacc']:.3f}{_stars(r['p_wilcoxon_bacc'])} \\\\"
        )
    _emit(
        "table2_weighting.tex", "tab:weighting",
        "Triage-informed weighting vs.\\ four Napierala-weighting mappings "
        "(XGBoost, vs.\\ no-weighting baseline, "
        f"{sub['dataset'].nunique()} datasets). Win rate = fraction of datasets "
        "improved; $p$ from two-sided Wilcoxon over per-dataset means.",
        "lrrrrrr",
        r"Method & Mean Acc & WR Acc & $p_{\text{Acc}}$ & Mean BAcc & WR BAcc & $p_{\text{BAcc}}$",
        body,
    )


def table3_smote_comparison():
    df = _load_method_classifier("xgboost")
    smote_methods = [
        "smote", "borderline_smote", "adasyn", "safe_level_smote",
        "polynom_fit_smote", "prowsyn", "mwmote", "gsmote",
        "napierala_guided_smote", "clean_masked_smote",
    ]
    sub = df[df["method"].isin(["baseline"] + smote_methods)]
    acc = pairwise_vs_baseline(sub, baseline="smote", metric="accuracy")
    bacc = pairwise_vs_baseline(sub, baseline="smote", metric="balanced_accuracy")
    merged = acc.merge(bacc, on="method", suffixes=("_acc", "_bacc"))
    order = {m: i for i, m in enumerate(["baseline"] + smote_methods)}
    merged = merged.sort_values("method", key=lambda s: s.map(order))
    body = []
    for _, r in merged.iterrows():
        body.append(
            f"{_tex(r['method'])} & "
            f"{r['mean_method_acc']:.4f} & {100*r['win_rate_acc']:.1f}\\% & "
            f"{r['p_wilcoxon_acc']:.3f}{_stars(r['p_wilcoxon_acc'])} & "
            f"{r['mean_method_bacc']:.4f} & {100*r['win_rate_bacc']:.1f}\\% & "
            f"{r['p_wilcoxon_bacc']:.3f}{_stars(r['p_wilcoxon_bacc'])} \\\\"
        )
    _emit(
        "table3_smote.tex", "tab:smote",
        "Oversampler family vs.\\ standard SMOTE (XGBoost, "
        f"{sub['dataset'].nunique()} OpenML-roster datasets): mean metric, win rate "
        "over SMOTE, and Wilcoxon $p$. The geometric variant G-SMOTE (added at "
        "review) buys significantly \\emph{less} balanced accuracy than SMOTE at a "
        "comparable accuracy level; only polynom-fit (a known weak-tradeoff variant) "
        "and the triage clean-masked variant significantly recover accuracy.",
        "lrrrrrr",
        r"Method (vs.\ SMOTE) & Mean Acc & WR Acc & $p_{\text{Acc}}$ & Mean BAcc & WR BAcc & $p_{\text{BAcc}}$",
        body,
        fontsize="footnotesize",
        colsep="2.5pt",
    )


def table5_classifier_ablation():
    """Cross-classifier robustness (R3/R4). Per downstream classifier:
    the SMOTE tradeoff (vs baseline), triage-weighting (vs baseline, Pareto-neutral),
    and the masking-necessity contrast clean_masked vs SMOTE (recovers accuracy)
    against napierala_guided vs SMOTE (does not).
    """
    rows = []
    for clf in ["xgboost", "rf", "lgbm", "logreg", "svm"]:
        try:
            df = _load_method_classifier(clf)
        except FileNotFoundError:
            continue
        present = set(df["method"].unique())
        def delta(method, base, metric):
            if method not in present or base not in present:
                return (np.nan, np.nan)
            sub = df[df["method"].isin([base, method])]
            res = pairwise_vs_baseline(sub, baseline=base, metric=metric)
            res = res[res["method"] == method]
            if res.empty:
                return (np.nan, np.nan)
            return (float(res["mean_delta"].iloc[0]), float(res["p_wilcoxon"].iloc[0]))
        sm_acc = delta("smote", "baseline", "accuracy")
        sm_bacc = delta("smote", "baseline", "balanced_accuracy")
        tw_acc = delta("triage_weighting", "baseline", "accuracy")
        cm_acc = delta("clean_masked_smote", "smote", "accuracy")
        ng_acc = delta("napierala_guided_smote", "smote", "accuracy")
        rows.append(dict(clf=clf, sm_acc=sm_acc, sm_bacc=sm_bacc,
                         tw_acc=tw_acc, cm_acc=cm_acc, ng_acc=ng_acc))

    def cell(v):
        d, p = v
        if d != d:  # NaN
            return "--"
        return f"{100*d:+.2f}{_stars(p)}"

    body = []
    for r in rows:
        body.append(
            f"{r['clf']} & {cell(r['sm_acc'])} & {cell(r['sm_bacc'])} & "
            f"{cell(r['tw_acc'])} & {cell(r['cm_acc'])} & {cell(r['ng_acc'])} \\\\"
        )
    _emit(
        "table5_classifier_ablation.tex", "tab:classifier_ablation",
        "Cross-classifier robustness (mean $\\Delta$, percentage points; "
        "stars = Wilcoxon significance). SMOTE columns vs.\\ no-resampling "
        "baseline; clean-masked and Napierala-guided vs.\\ SMOTE. The "
        "accuracy/balanced-accuracy tradeoff (SMOTE) holds across every "
        "classifier family; triage weighting stays Pareto-neutral; triage "
        "masking (clean-masked) recovers accuracy over SMOTE where "
        "Napierala-guided does not.",
        "lrrrrr",
        r"Classifier & SMOTE $\Delta$Acc & SMOTE $\Delta$BAcc & "
        r"TriageWt $\Delta$Acc & CleanMask $\Delta$Acc & NapGuided $\Delta$Acc",
        body,
    )


def table7_cost():
    """Computational cost (Table~\\ref{tab:cost}) from TIER-1 timing_results."""
    path = ORIG_DIR / "timing_results.parquet"
    if not path.exists():
        raise FileNotFoundError(f"timing results not found at {path}")
    df = pd.read_parquet(path)
    agg = (df.groupby("dataset")
             .agg(n=("n_samples", "first"), d=("n_features", "first"),
                  triage_s=("triage_fit_time", "mean"),
                  xgb_s=("xgb_train_time", "mean"),
                  overhead=("triage_overhead_pct", "mean"))
             .reset_index()
             .sort_values("n"))
    body = []
    for _, r in agg.iterrows():
        body.append(
            f"\\texttt{{{_tex(r['dataset'])}}} & {int(r['n'])} & {int(r['d'])} & "
            f"{r['triage_s']:.2f} & {r['xgb_s']:.2f} & {r['overhead']:.1f} \\\\")
    _emit(
        "table7_cost.tex", "tab:cost",
        "Triage wall-clock cost vs.\\ a single XGBoost fit (mean over repeats). "
        "Overhead = triage time as a percentage of total pipeline time. "
        "The triage is a one-time, classifier-agnostic preprocessing cost.",
        "lrrrrr",
        r"Dataset & $n$ & $d$ & Triage (s) & XGB (s) & Overhead (\%)",
        body, placement="t")


def table_agreement():
    agreement_dir = RESULTS_DIR / "napierala_agreement"
    files = list(agreement_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No agreement parquets found in {agreement_dir}.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df["dataset"].isin(ROSTER)]  # 54 roster only
    pivot = (df.groupby(["napierala", "triage"])["count"].sum()
              .unstack(fill_value=0))
    # stable column order
    col_order = [c for c in ["correct", "data_limited", "irreducible", "noise"]
                 if c in pivot.columns]
    pivot = pivot[col_order]
    row_order = [r for r in ["safe", "borderline", "rare", "outlier", "majority"]
                 if r in pivot.index]
    pivot = pivot.loc[row_order]
    body = []
    for nap, row in pivot.iterrows():
        body.append(f"{_tex(nap)} & " + " & ".join(f"{int(v):,}".replace(",", r"\,")
                    for v in row.values) + r" \\")
    header = r"Napierala $\backslash$ Triage & " + " & ".join(_tex(c) for c in pivot.columns)
    _emit(
        "table_agreement.tex", "tab:agreement",
        "Cross-tabulation of Napierala\\,$\\times$\\,triage categories over the "
        f"{df['dataset'].nunique()}-dataset roster (instance counts). Napierala's "
        "``rare'' splits across triage's correct / data-limited / irreducible "
        "rather than mapping onto data-limited.",
        "l" + "r" * len(pivot.columns),
        header, body)


def build_dataset_registry(force: bool = False) -> pd.DataFrame:
    registry_path = RESULTS_DIR / "dataset_registry.parquet"
    if registry_path.exists() and not force:
        return pd.read_parquet(registry_path)
    from scripts.paper_revision.datasets import load_dataset
    rows = []
    for spec in DATASETS:
        try:
            X, y = load_dataset(spec)
        except Exception:
            # e.g. webpage (sparse ARFF) — fall back to the spec's approximate sizes.
            rows.append({"name": spec.name, "source": spec.source,
                         "openml_id": spec.identifier, "n": spec.n_samples,
                         "d": spec.n_features, "n_classes": -1, "ir": float("nan"),
                         "task": spec.task})
            continue
        _, counts = np.unique(y, return_counts=True)
        rows.append({
            "name": spec.name, "source": spec.source, "openml_id": spec.identifier,
            "n": int(X.shape[0]), "d": int(X.shape[1]),
            "n_classes": int(len(counts)), "ir": float(counts.max() / counts.min()),
            "task": spec.task,
        })
    df = pd.DataFrame(rows)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(registry_path)
    return df


def table_datasets(force_registry: bool = False):
    """Full 54-dataset listing for Appendix~\\ref{app:datasets} (longtable)."""
    df = build_dataset_registry(force=force_registry)
    header = r"Dataset & OpenML ID & $n$ & $d$ & $C$ & IR \\"
    lines = [
        r"\begingroup\footnotesize",
        r"\begin{longtable}{lrrrrc}",
        rf"\caption{{Full dataset listing ({len(df)} datasets; the recovered "
        r"original-paper roster). $n$ = instances; $d$ = features; $C$ = classes; "
        r"IR = imbalance ratio (majority / minority count). ``sklearn'' marks "
        r"datasets loaded from scikit-learn rather than OpenML; \texttt{webpage}'s "
        r"class statistics are not reported because its sparse high-dimensional "
        r"format excludes it from the menu evaluation (\S\ref{sec:design})."
        r"}\label{tab:datasets}\\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\multicolumn{6}{l}{\tablename~\thetable\ -- continued}\\",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule \multicolumn{6}{r}{\textit{continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule", r"\endlastfoot",
    ]
    # display-name corrections: internal keys are kept (caches/results are keyed
    # on them) but the paper table must show the dataset each OpenML id actually is
    display = {"oil_spill": "ipums_la_99-small", "thyroid": "thyroid-allbp"}
    for _, r in df.iterrows():
        oid = r["openml_id"]
        oid_s = "sklearn" if float(oid) < 0 else str(int(oid))
        ir_s = "--" if r["ir"] != r["ir"] else f"{r['ir']:.1f}"
        c_s = "--" if int(r["n_classes"]) < 0 else str(int(r["n_classes"]))
        lines.append(
            f"\\texttt{{{_tex(display.get(r['name'], r['name']))}}} & {oid_s} & {int(r['n'])} & "
            f"{int(r['d'])} & {c_s} & {ir_s} \\\\")
    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    (TABLES_DIR / "table_datasets.tex").write_text("\n".join(lines) + "\n")


def main():
    builders = [table2_weighting_comparison, table3_smote_comparison,
                table5_classifier_ablation, table7_cost, table_agreement,
                table_datasets]
    for b in builders:
        try:
            b()
            print(f"  ok  {b.__name__}")
        except Exception as e:
            print(f"  FAIL {b.__name__}: {e}")
    print("tables written to", TABLES_DIR)


if __name__ == "__main__":
    main()
