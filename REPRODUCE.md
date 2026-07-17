# Reproduction guide

Reproduces every figure, table, and headline number in
*"Don't Oversample the Boundary."* All commands are run from the repository
root with the project virtualenv:

```bash
PY=.venv/bin/python      # or: python -m venv .venv && pip install -r requirements.txt
```

## Data

- **KEEL / imbalanced-learn suite (27 datasets, IR 8–130):** fetched via
  `imbalanced-learn` `fetch_datasets()` (`scripts/paper_revision/keel_datasets.py`).
- **OpenML / scikit-learn roster (54 listed; 52 evaluated, USPS+webpage excluded):**
  `scripts/paper_revision/datasets.py` (downloaded via OpenML on first use).

All experiments subsample to ≤10,000 instances (stratified) and use
5×5 Repeated Stratified K-Fold; the triage and any moved threshold are fit on the
training split only.

## Reproduction map

| Paper element | Script | Output |
|---|---|---|
| Triage features (Cat1/2/3 etc.) | `build_meta_features.py` (reads cached `triage_features.parquet`) | `meta_features.parquet` |
| §5.1 Triage validation (synthetic + semi-synthetic) | `run_synthetic_validation.py` | `synthetic_validation.parquet` |
| §5.2 Mechanism, **Fig. overlap** | `run_overlap_synthetic.py` | `overlap_synthetic.parquet`, `figures/fig_overlap.pdf` |
| §5.3 Interventional causal (p<0.001), **Fig. interventional** | `run_interventional.py` | `interventional.parquet`, `figures/fig1_interventional.pdf` |
| §5.4 Full-menu benchmark (16 strategies) | SMOTE family/baseline reused from `{keel,main}_benchmark/`; non-generative via `run_frontier.py --roster {keel,original}` | `frontier_benchmark/` |
| §5.4 Dominance + **Fig. frontier**, **Table frontier** | `build_frontier.py` | `figures/fig_frontier.pdf` |
| §5.4 **CD diagram** (Friedman–Nemenyi), mediation, separability AUC | `build_credibility.py` | `figures/fig_cd.pdf`, `separability_auc.parquet` |
| §5.4 Base-learner robustness (RF, LogReg) | `run_frontier_clf.py --classifier {rf,logreg}` then `build_robustness_artifacts.py` | `frontier_benchmark/{rf,logreg}/`, `baselearner_dominance_*.parquet` |
| §5.5 **Regime map**, **Fig. regime** | `build_regime_map.py` | `figures/fig_regime_map.pdf`, `regime_map_*.parquet` |
| §5.6 Selection + **Table selection** | `meta_selection.py --menu full --metric {balanced_accuracy,g_mean,mcc} [--features {all,core,triage}]` | `meta_selection_full_*.parquet` |

## Revision experiments (Neurocomputing review, 2026-07)

Extra dependency: `pip install imbalanced-learn-extra` (== 0.2.10; provides
`imblearn_extra.gsmote.GeometricSMOTE`, the Douzas & Bacao authors' maintained
implementation).

| Reviewer point | Script | Output |
|---|---|---|
| R7 G-SMOTE benchmark cells (KEEL / OpenML) | `run_keel.py` / `run_parallel.py --only-method gsmote` | `keel_benchmark/`, `main_benchmark/` |
| R7 G-SMOTE boundary-crossing (controlled) | `run_overlap_gsmote.py` | `overlap_synthetic_gsmote.parquet` |
| R11/R5 parity v2: balanced Brier/ECE, gsmote, mlp/nb/svm learners | `run_threshold_parity.py --roster keel` | `threshold_parity_v2/` |
| R5 non-tree remedy decomposition | `run_reducibility_nontree.py` | `reducibility_nontree.parquet` |
| R6 ensemble-size (M) sensitivity | `run_m_sensitivity.py` | `m_sensitivity.parquet` |

## Headline numbers (verified)

- Interventional: boundary (Cat3) augmentation reproduces the SMOTE signature,
  balanced accuracy `p=0.001`; random augmentation null.
- Mechanism on real data: SMOTE accuracy cost vs. minority error rate
  Spearman `ρ=−0.50` (`p<10⁻⁴`), vs. imbalance ratio `ρ=−0.41`.
- Separability: 30 hardest binary datasets have median baseline OOF AUC `0.91`
  (93% > 0.70).
- Dominance (balanced accuracy): best non-generative ≥ best generative in
  **25/27 KEEL** (+5.76 pp) and **43/52 OpenML** (+2.02 pp) = **86%**.
  Friedman over the full menu `p≈5×10⁻⁵²`; threshold-moving best average rank.
- Base-learner robustness: RF +2.07 pp (`p=1.6×10⁻⁴`), LogReg +6.44 pp
  (`p=2×10⁻¹⁰`).
- Selection: learned selector beats always-SMOTE by ~3.5 pp; a direct classifier
  beats the best fixed strategy by +0.64 pp on balanced accuracy / G-mean
  (8/8 seeds, `p≤0.03`), null on MCC; the IR-only rule matches the full selector.

## Build the PDF

```bash
cd paper_v2 && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
