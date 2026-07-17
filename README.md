# Don't Oversample the Boundary

Code and data accompanying the paper **"Don't Oversample the Boundary: A Remedy
Decomposition of the Minority Deficit in Imbalanced Learning"** (Cameron Hamilton,
Alliance AI).

**TL;DR.** We introduce a *remedy decomposition* of the misclassified minority class:
each minority error is assigned to the fix that would resolve it —
*threshold-recoverable* (its score already clears a cross-fitted, tuned decision
threshold; only the default cutoff is wrong), *data-reducible* (more minority data
would help), or *irreducible* (genuine class overlap). Across 79 real datasets the
minority deficit is predominantly
**threshold-recoverable** (mean 68%, 80% at imbalance ratio >3; ~6% data-reducible,
~17% irreducible). This
explains why oversampling raises balanced accuracy *without* improving ranking (AUC
unchanged) and why a threshold move reproduces its benefit without synthetic data —
oversampling enacts a class-prior shift in probability space, an implicit
operating-point move. On controlled mixtures with a known Bayes boundary, standard
SMOTE places an overlap-dependent fraction of synthetic points across it — and so
does Geometric SMOTE, whose majority-bounded hypersphere cannot rescue seeds that
already sit in the overlap: the harm comes from *where the seeds are*, not the
interpolation geometry. Once every strategy is given the same out-of-fold threshold
tuning, the default-threshold gap between non-generative and generative families
collapses to within ±1 pp across XGBoost, random forests, and logistic regression
(naive Bayes and RBF-SVM behave alike); the exception is a single undertrained MLP,
whose *ranking* imbalance corrupts and resampling partially repairs. Practically:
diagnose the deficit, and for most of it, move the decision rather than oversample.

## Layout

```
endgame/                  Error Instance Triage + the resampler implementations used by the paper
scripts/paper_revision/   experiment runners + analysis/figure/table scripts
results/                  precomputed result files (parquet) backing every table and figure
paper_v2/                 the manuscript (LaTeX), figures, and tables
REPRODUCE.md              maps every figure/table/number to the script that produces it
requirements.txt          Python dependencies
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# headline: remedy decomposition of the minority deficit (mean 68% threshold-recoverable)
python -m scripts.paper_revision.build_reducibility_v2 --combine-only --noise-mode global --write-table

# threshold parity: the generative vs non-generative gap collapses at equal tuning
python -m scripts.paper_revision.build_threshold_parity

# critical-difference diagram, mediation, separability
python -m scripts.paper_revision.build_credibility

# rebuild the manuscript figures and tables from the provided results
python -m scripts.paper_revision.build_figures
python -m scripts.paper_revision.build_tables
```

See **[REPRODUCE.md](REPRODUCE.md)** for the full reproduction map and the end-to-end
commands (including the sweeps that regenerate `results/` from scratch).

## Revision experiments (2026-07)

The Neurocomputing revision added, under the identical protocol: Geometric SMOTE
(the authors' `imbalanced-learn-extra` implementation) on both the known-boundary
testbed (`run_overlap_gsmote.py`) and the full benchmark; the remedy decomposition
re-derived with bagged non-tree instruments (`run_reducibility_nontree.py`); an
ensemble-size sensitivity sweep (`run_m_sensitivity.py`); and a six-learner
threshold-parity analysis with class-balanced calibration metrics
(`run_threshold_parity.py`, results in `results/threshold_parity_v2/`). See the
"Revision experiments" section of REPRODUCE.md for the mapping.

## Post-review corrections (2026-07)

Verifying the revised manuscript against this implementation surfaced two
methodological gaps in the original headline computation, both corrected in
`build_reducibility_v2.py` (the old `build_reducibility.py` and its results are
retained for the audit trail): the default error set is now the ensemble's actual
out-of-bag argmax mistakes (the previous `P(minority) < 0.5` rule over-counted
errors on multiclass datasets), and the tuned threshold is now cross-fitted
(previously selected on the same out-of-bag scores used to score recovery). The
decomposition is measured on the unweighted ensemble — the off-the-shelf default
model the paper analyzes — with class-weighted (`--noise-mode`) and
standardized-geometry (`--standardize`) sensitivity passes included
(`results/reducibility_v2*.parquet`). The corrected headline: threshold-recoverable
68% mean (80% at IR>3) vs the previously reported 69%/74%. Two roster labels were
also corrected: OpenML id 1018 is `ipums_la_99-small` (not the classic oil-spill
set) and id 40474 is `thyroid-allbp` (not ann-thyroid); internal result keys keep
the old names, the paper's dataset table shows the true ones (see REPRODUCE.md).

All datasets are public (the `imbalanced-learn`/KEEL suite and OpenML) and are
downloaded automatically on first use.

## Citation

Hamilton, C. *Don't Oversample the Boundary: A Remedy Decomposition of the Minority
Deficit in Imbalanced Learning.* 2026.

## License

MIT (see `LICENSE`).
