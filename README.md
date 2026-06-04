# Don't Oversample the Boundary

Code and data accompanying the paper **"Don't Oversample the Boundary: A Remedy
Decomposition of the Minority Deficit in Imbalanced Learning"** (Cameron Hamilton,
Alliance AI).

**TL;DR.** We introduce a *remedy decomposition* of the misclassified minority class:
each minority error is assigned to the fix that would resolve it —
*threshold-recoverable* (the model already ranks it correctly; only the default cutoff
is wrong), *data-reducible* (more minority data would help), or *irreducible* (genuine
class overlap). Across 79 real datasets the minority deficit is overwhelmingly
**threshold-recoverable** (mean 69%; ~7% data-reducible, ~16% irreducible). This
explains why oversampling raises balanced accuracy *without* improving ranking (AUC
unchanged) and why a threshold move reproduces its benefit without synthetic data and
with better calibration — oversampling is, at best, an implicit operating-point shift.
On controlled mixtures with a known Bayes boundary, standard SMOTE places an
overlap-dependent fraction of synthetic points across it; and once every strategy is
given the same out-of-fold threshold tuning, the default-threshold gap between
non-generative and generative families collapses to within ±1 pp across three base
learners. Practically: diagnose the deficit, and for most of it, move the decision
rather than oversample.

## Layout

```
endgame/                  Error Instance Triage + the resampler implementations used by the paper
scripts/paper_revision/   experiment runners + analysis/figure/table scripts
results/paper_revision/   precomputed result files (parquet) backing every table and figure
paper_v2/                 the manuscript (LaTeX), figures, and tables
REPRODUCE.md              maps every figure/table/number to the script that produces it
requirements.txt          Python dependencies
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# headline: remedy decomposition of the minority deficit (mean 69% threshold-recoverable)
python -m scripts.paper_revision.build_reducibility

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

All datasets are public (the `imbalanced-learn`/KEEL suite and OpenML) and are
downloaded automatically on first use.

## Citation

Hamilton, C. *Don't Oversample the Boundary: A Remedy Decomposition of the Minority
Deficit in Imbalanced Learning.* 2026.

## License

MIT (see `LICENSE`).
