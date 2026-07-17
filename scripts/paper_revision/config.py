"""Central configuration for the paper revision experiments."""
from pathlib import Path

RESULTS_DIR = Path("results")  # public-release layout: results/ holds what the research repo keeps in results/paper_revision/
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CV protocol matches v1
N_REPEATS = 5
N_FOLDS = 5
RANDOM_STATE = 42

# Downstream classifier — XGBoost is primary, others used in N4 ablation
XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                  tree_method="hist", random_state=RANDOM_STATE,
                  n_jobs=4, verbosity=0)
RF_PARAMS = dict(n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=4)
LGBM_PARAMS = dict(n_estimators=300, max_depth=-1, learning_rate=0.1,
                   random_state=RANDOM_STATE, n_jobs=4, verbosity=-1)
LR_PARAMS = dict(max_iter=2000, random_state=RANDOM_STATE, n_jobs=4)
# SVM (5th classifier family for the R3/R4 robustness ablation). SVC(rbf) is
# O(n^2)-O(n^3), so SVM cells use a tighter subsample cap (SVM_MAX_INSTANCES) and
# are wrapped in a StandardScaler pipeline (cv_runner) since RBF needs scaled inputs.
SVM_PARAMS = dict(kernel="rbf", C=1.0, gamma="scale", cache_size=512,
                  random_state=RANDOM_STATE)
SVM_MAX_INSTANCES = 4_000  # per-cell cap for SVM only (tractability)
# MLP (Neurocomputing R5/E4: non-tree learners in the threshold-parity analysis).
# Modest fixed architecture, early stopping; wrapped in a StandardScaler pipeline.
MLP_PARAMS = dict(hidden_layer_sizes=(64, 32), max_iter=300,
                  early_stopping=True, n_iter_no_change=15,
                  random_state=RANDOM_STATE)

# Triage hyperparameters (locked to v1)
TRIAGE_PARAMS = dict(n_forests=5, n_trees_per_forest=100,
                     noise_tcp_threshold=0.12,
                     cat2_class_ratio_threshold=0.4,
                     random_state=RANDOM_STATE)

# Subsampling cap (matches v1 §4)
MAX_INSTANCES = 10_000

# Multiple-testing correction
ALPHA_FAMILY = 0.05  # Holm–Bonferroni applied at aggregation time
