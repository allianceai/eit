#!/usr/bin/env python
"""Parallel driver for the paper-revision main benchmark sweep.

Replaces the sequential loop in run_all.sh with a ProcessPoolExecutor
that processes (classifier, method, dataset) cells concurrently.

Cell budget (additional experiments only; everything else is REUSED from
results/original_study/ — see run_all.sh header):
  - xgboost × 15 methods × 58 datasets       = 870 cells  (broad-method + Napierala)
  - {rf,lgbm,logreg} × 3 methods × 58 datasets = 522 cells  (classifier robustness)
  - total = 1392 cells
The 10 methods are the broad reference set the ECML reviews asked for (R1: add
Safe-Level-SMOTE + Kovács variants) plus an in-run baseline/smote anchor and the
two triage methods, so the R1 comparison table is self-contained AND we confirm the
triage methods reproduce on the original roster. The SMOTE-family cleaning methods
(smote_enn/tomek), the weighting variants, and the interventional results are reused
from the original study, not regenerated here.

Resource model: each cell runs SINGLE-THREADED (native thread pools pinned to 1
at import; see below) and peaks at <1 GB RSS. Parallelism comes from running
~one worker per core (default: cpu_count - 2). On a 20-core / 64 GB box that is
18 workers using ~11 GB total. Do NOT raise per-worker threads — that reintro-
duces the oversubscription (~75 threads/worker) that thrashed the CPU and RAM.

Usage:
    python -m scripts.paper_revision.run_parallel [--workers 18]
        [--only-classifier xgboost] [--only-method triage_weighting]
        [--only-dataset iris] [--dry-run]
"""
from __future__ import annotations

# Pin every native thread pool to a single thread BEFORE numpy / BLAS / OpenMP
# get imported anywhere in this process. Each benchmark cell runs single-threaded;
# parallelism comes from running many worker processes (~one per core). These env
# vars MUST be set at import time: libgomp / OpenBLAS read them once when the
# library first initialises, so setting them later (e.g. inside the worker body,
# as this script used to) has NO effect and lets every worker spawn ~one thread
# per core -> ~75 threads/worker -> CPU thrash + glibc per-thread malloc arenas.
import os
for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse
import signal
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# ---------------------------------------------------------------------------
# Cell definition
# ---------------------------------------------------------------------------

# Broad-method comparison set (XGBoost only). The 6 new oversamplers the reviews
# requested (Borderline, ADASYN, Safe-Level, and the Kovács variants polynom_fit /
# ProWSyn / MWMOTE), plus baseline + smote as an in-run reference anchor, plus the two
# triage methods so the R1 table is self-contained. Everything else (smote_enn/tomek,
# weighting variants, interventional, etc.) is REUSED from results/original_study/.
XGBOOST_METHODS = [
    "baseline", "smote",
    "borderline_smote", "adasyn", "safe_level_smote",
    "polynom_fit_smote", "prowsyn", "mwmote",
    "clean_masked_smote", "triage_weighting",
    # Napierala-guided head-to-head (R1/R3: compare against the existing categorization).
    # napierala_guided_smote is the Napierala analogue of clean_masked; the four
    # napierala_weighting mappings are the Napierala analogue of triage_weighting.
    "napierala_guided_smote",
    "napierala_weighting_rare", "napierala_weighting_rare_outlier",
    "napierala_weighting_borderline", "napierala_weighting_nonsafe",
    # Imbalance-aware noise-detection variants (KEEL-motivated; the 15 above are
    # already on disk and skipped on resume). Lets us confirm on the original roster
    # that `balanced` is >= `global` and that protect_minority/class_conditional behave
    # consistently with the KEEL guardrail (they over-suppress noise detection).
    "clean_masked_class_conditional", "clean_masked_balanced", "clean_masked_protect_minority",
    "triage_weighting_class_conditional", "triage_weighting_balanced", "triage_weighting_protect_minority",
    # The formalized improved weighter: aggressive on learnable minority, not the boundary.
    "triage_cost_sensitive",
    # Neurocomputing R7: modern geometric variant (Douzas & Bacao 2019), authors'
    # maintained implementation. All earlier cells skip on resume; this adds 52 cells.
    "gsmote",
]

# Classifier-robustness ablation (R3: is the accuracy/balanced-accuracy tradeoff a
# guaranteed effect of SMOTE or classifier-dependent? R4: only common classifiers
# tested). Across RF / LightGBM / LogReg / SVM (4 distinct families: bagging, boosting,
# linear, kernel) on the 54-dataset roster we run: baseline (reference), smote (the
# tradeoff), triage_weighting (our weighting), AND the two masking methods
# clean_masked_smote + napierala_guided_smote so the MASKING-necessity contrast (triage
# masking beats SMOTE, Napierala-guided does not) is shown beyond XGBoost -- the v2
# necessity argument rests on masking, not weighting. SVM is new in v2 (the original
# robustness phase used LightGBM); SVC(rbf) cells use the tighter SVM cap in config.
ABLATION_METHODS = ["baseline", "smote", "triage_weighting",
                    "clean_masked_smote", "napierala_guided_smote"]
ABLATION_CLASSIFIERS = ["rf", "lgbm", "logreg", "svm"]

# Datasets in the 54-roster that are EXCLUDED from the extended (additive) sweep:
#   USPS    -- 256-dim / 10-class; the original authors removed it for exactly this
#              reason ("high dimensionality causes failures": the oversamplers, e.g.
#              MWMOTE/ProWSyn, stall on 256-dim multiclass). It remains part of the
#              54-dataset benchmark via the REUSED original headline results
#              (sample_weighting / masked_rebalancing), so the headline is unaffected.
#   webpage -- sparse ARFF the installed OpenML parser cannot read (fails fast on load);
#              also covered by the reused original results.
# Net: the extended broad-method / classifier / Napierala sweep spans 52 of the 54.
SWEEP_EXCLUDE = {"USPS", "webpage"}


class Cell(NamedTuple):
    classifier: str
    method: str
    dataset: str


def build_cell_list(
    only_classifier: str | None = None,
    only_method: str | None = None,
    only_dataset: str | None = None,
) -> list[Cell]:
    """Build the full Cartesian product of (classifier, method, dataset) cells."""
    from scripts.paper_revision.datasets import DATASETS

    dataset_names = [d.name for d in DATASETS if d.name not in SWEEP_EXCLUDE]

    cells: list[Cell] = []

    # XGBoost: all 15 broad-method cells
    for m in XGBOOST_METHODS:
        for d in dataset_names:
            cells.append(Cell("xgboost", m, d))

    # Ablation classifiers (rf/lgbm/logreg/svm): the 5 ablation methods
    for clf in ABLATION_CLASSIFIERS:
        for m in ABLATION_METHODS:
            for d in dataset_names:
                cells.append(Cell(clf, m, d))

    # Apply filters
    if only_classifier:
        cells = [c for c in cells if c.classifier == only_classifier]
    if only_method:
        cells = [c for c in cells if c.method == only_method]
    if only_dataset:
        cells = [c for c in cells if c.dataset == only_dataset]

    return cells


def _out_path(cell: Cell) -> Path:
    from scripts.paper_revision.config import RESULTS_DIR
    return RESULTS_DIR / "main_benchmark" / f"{cell.classifier}__{cell.method}__{cell.dataset}.parquet"


# ---------------------------------------------------------------------------
# Worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _init_worker():
    """Pool initializer: re-assert single-thread native pools in each worker.

    The module-level env vars already propagate through ``fork``; this is
    defense-in-depth (and the only thing that works under the ``spawn`` start
    method) so a worker can never silently fall back to one-thread-per-core.
    """
    import os
    for v in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[v] = "1"


def _run_cell(cell_tuple: tuple) -> dict:
    """Top-level worker function executed in a subprocess.

    Each cell runs single-threaded: native thread pools are pinned to 1 at module
    import (see top of file) and re-asserted in the pool initializer, and the
    estimators' own ``n_jobs`` is set to 1 here. Parallelism comes from running
    ~one worker per core, not from threads inside a cell. Measured: this keeps a
    worker at ~8 threads instead of ~75, so N workers ≈ N×8 threads total.
    """
    import time
    import traceback

    # Pin estimator-level parallelism to match the single-thread native pools.
    from scripts.paper_revision.config import XGB_PARAMS, RF_PARAMS, LGBM_PARAMS, LR_PARAMS
    XGB_PARAMS["n_jobs"] = 1
    RF_PARAMS["n_jobs"] = 1
    LGBM_PARAMS["n_jobs"] = 1
    LR_PARAMS["n_jobs"] = 1

    from scripts.paper_revision.datasets import DATASETS, load_dataset
    from scripts.paper_revision.cv_runner import evaluate_method_on_dataset
    from scripts.paper_revision.config import RESULTS_DIR

    cell = Cell(*cell_tuple)
    out_dir = RESULTS_DIR / "main_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cell.classifier}__{cell.method}__{cell.dataset}.parquet"

    if out_path.exists():
        return {
            "classifier": cell.classifier,
            "method": cell.method,
            "dataset": cell.dataset,
            "status": "skip",
            "mean_acc": float("nan"),
            "mean_bacc": float("nan"),
            "mean_f1m": float("nan"),
            "mean_mcc": float("nan"),
            "mean_g_mean": float("nan"),
            "error_message": "",
            "elapsed_s": 0.0,
        }

    spec_map = {d.name: d for d in DATASETS}
    spec = spec_map.get(cell.dataset)
    if spec is None:
        return {
            "classifier": cell.classifier,
            "method": cell.method,
            "dataset": cell.dataset,
            "status": "error",
            "mean_acc": float("nan"),
            "mean_bacc": float("nan"),
            "mean_f1m": float("nan"),
            "mean_mcc": float("nan"),
            "mean_g_mean": float("nan"),
            "error_message": f"Unknown dataset: {cell.dataset}",
            "elapsed_s": 0.0,
        }

    t0 = time.perf_counter()
    try:
        X, y = load_dataset(spec)
    except Exception as e:
        return {
            "classifier": cell.classifier,
            "method": cell.method,
            "dataset": cell.dataset,
            "status": "error",
            "mean_acc": float("nan"),
            "mean_bacc": float("nan"),
            "mean_f1m": float("nan"),
            "mean_mcc": float("nan"),
            "mean_g_mean": float("nan"),
            "error_message": f"load error: {e}",
            "elapsed_s": time.perf_counter() - t0,
        }

    # Per-cell hard cap (SIGALRM): a cell stuck in a Python-level loop self-aborts and
    # is logged as an error instead of stalling the pool. 900s is far above any normal cell.
    import signal
    class _CellTimeout(Exception):
        pass
    def _alarm(signum, frame):
        raise _CellTimeout()
    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(900)
    except (ValueError, AttributeError):
        pass

    try:
        # threadpool_limits clamps any pool that a library (smote_variants,
        # sklearn internals) creates at call time despite the env vars.
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            df = evaluate_method_on_dataset(
                cell.method, X, y,
                dataset_name=cell.dataset,
                classifier=cell.classifier,
                n_repeats=5,
                n_folds=5,
            )
        df.to_parquet(out_path)
        signal.alarm(0)
        elapsed = time.perf_counter() - t0
        return {
            "classifier": cell.classifier,
            "method": cell.method,
            "dataset": cell.dataset,
            "status": "ok",
            "mean_acc": float(df.accuracy.mean()),
            "mean_bacc": float(df.balanced_accuracy.mean()),
            "mean_f1m": float(df.f1_macro.mean()),
            "mean_mcc": float(df.mcc.mean()),
            "mean_g_mean": float(df.g_mean.mean()),
            "error_message": "",
            "elapsed_s": elapsed,
        }
    except Exception:
        signal.alarm(0)
        elapsed = time.perf_counter() - t0
        tb = traceback.format_exc()
        return {
            "classifier": cell.classifier,
            "method": cell.method,
            "dataset": cell.dataset,
            "status": "error",
            "mean_acc": float("nan"),
            "mean_bacc": float("nan"),
            "mean_f1m": float("nan"),
            "mean_mcc": float("nan"),
            "mean_g_mean": float("nan"),
            "error_message": tb,
            "elapsed_s": elapsed,
        }


# ---------------------------------------------------------------------------
# Progress driver
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser(
        description="Parallel main-benchmark driver with rich progress."
    )
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 2),
                    help="Parallel worker processes; each runs single-threaded "
                         "(default: cpu_count - 2)")
    ap.add_argument("--only-classifier", default=None,
                    choices=["xgboost", "rf", "lgbm", "logreg"])
    ap.add_argument("--only-method", default=None)
    ap.add_argument("--only-dataset", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print cells that would run without executing them.")
    args = ap.parse_args()

    # Build cell list
    all_cells = build_cell_list(
        only_classifier=args.only_classifier,
        only_method=args.only_method,
        only_dataset=args.only_dataset,
    )

    # Filter already-done cells
    pending = [c for c in all_cells if not _out_path(c).exists()]
    skipped_count = len(all_cells) - len(pending)

    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint

    console = Console(stderr=False)

    if args.dry_run:
        table = Table(title=f"Dry run: {len(pending)} cells to run "
                            f"({skipped_count} already done / {len(all_cells)} total)",
                      show_header=True, header_style="bold cyan")
        table.add_column("classifier", style="cyan")
        table.add_column("method", style="green")
        table.add_column("dataset", style="yellow")
        for c in pending[:50]:
            table.add_row(c.classifier, c.method, c.dataset)
        if len(pending) > 50:
            table.add_row("...", f"... (+{len(pending) - 50} more)", "...")
        console.print(table)
        console.print(f"[bold]Workers:[/bold] {args.workers}  "
                      f"[bold]Total:[/bold] {len(all_cells)}  "
                      f"[bold]Pending:[/bold] {len(pending)}  "
                      f"[bold]Already done:[/bold] {skipped_count}")
        return

    if not pending:
        console.print("[green]All cells already done — nothing to run.[/green]")
        return

    # Ensure output dirs exist
    from scripts.paper_revision.config import RESULTS_DIR
    out_dir = RESULTS_DIR / "main_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    error_log = RESULTS_DIR / "main_benchmark_errors.log"
    progress_path = RESULTS_DIR / "main_benchmark_progress.parquet"

    console.print(
        f"[bold cyan]run_parallel[/bold cyan]  "
        f"workers={args.workers}  pending={len(pending)}  skipped={skipped_count}"
    )

    # Progress tracking state
    completed_rows: list[dict] = []
    recent: deque[dict] = deque(maxlen=10)
    shutdown_requested = False

    # Signal handler for clean Ctrl-C
    original_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(sig, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        print(f"\n[{_ts()}] SIGINT received — cancelling pending futures...",
              file=sys.stderr)
        signal.signal(signal.SIGINT, original_sigint)

    signal.signal(signal.SIGINT, _handle_sigint)

    # Rich progress
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task_id = progress.add_task("main sweep", total=len(pending))

    def _recent_table() -> Table:
        t = Table(show_header=True, header_style="bold", expand=True,
                  title=f"Recent completions ({len(completed_rows)} done / {len(pending)} total)")
        t.add_column("status", width=6)
        t.add_column("classifier", width=10)
        t.add_column("method", width=32)
        t.add_column("dataset", width=22)
        t.add_column("elapsed", width=8)
        t.add_column("acc", width=7)
        t.add_column("bacc", width=7)
        t.add_column("f1m", width=7)
        t.add_column("mcc", width=7)
        t.add_column("gmean", width=7)
        for r in list(recent):
            status_style = "green" if r["status"] == "ok" else (
                "yellow" if r["status"] == "skip" else "red"
            )
            t.add_row(
                f"[{status_style}]{r['status']}[/{status_style}]",
                r["classifier"],
                r["method"],
                r["dataset"],
                f"{r['elapsed_s']:.1f}s",
                f"{r['mean_acc']:.4f}" if r["mean_acc"] == r["mean_acc"] else "—",
                f"{r['mean_bacc']:.4f}" if r["mean_bacc"] == r["mean_bacc"] else "—",
                f"{r['mean_f1m']:.4f}" if r["mean_f1m"] == r["mean_f1m"] else "—",
                f"{r['mean_mcc']:.4f}" if r["mean_mcc"] == r["mean_mcc"] else "—",
                f"{r['mean_g_mean']:.4f}" if r["mean_g_mean"] == r["mean_g_mean"] else "—",
            )
        return t

    with Live(console=console, refresh_per_second=4) as live:
        live.update(Panel(progress))

        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            max_tasks_per_child=50,
        ) as executor:
            future_to_cell = {
                executor.submit(_run_cell, tuple(c)): c
                for c in pending
            }

            for future in as_completed(future_to_cell):
                if shutdown_requested:
                    # Cancel remaining pending futures
                    for f in future_to_cell:
                        f.cancel()
                    break

                cell = future_to_cell[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "classifier": cell.classifier,
                        "method": cell.method,
                        "dataset": cell.dataset,
                        "status": "error",
                        "mean_acc": float("nan"),
                        "mean_bacc": float("nan"),
                        "mean_f1m": float("nan"),
                        "mean_mcc": float("nan"),
                        "mean_g_mean": float("nan"),
                        "error_message": traceback.format_exc(),
                        "elapsed_s": 0.0,
                    }

                # Timestamped stderr log line
                cell_label = f"{result['classifier']}__{result['method']}__{result['dataset']}"
                if result["status"] == "ok":
                    print(
                        f"[{_ts()}] ok  {cell_label}  "
                        f"{result['elapsed_s']:.1f}s  "
                        f"acc={result['mean_acc']:.4f}  "
                        f"bacc={result['mean_bacc']:.4f}  "
                        f"f1m={result['mean_f1m']:.4f}  "
                        f"mcc={result['mean_mcc']:.4f}  "
                        f"gmean={result['mean_g_mean']:.4f}",
                        file=sys.stderr,
                    )
                elif result["status"] == "skip":
                    print(f"[{_ts()}] skip {cell_label}", file=sys.stderr)
                else:
                    short_err = result["error_message"][:200].replace("\n", " ")
                    print(
                        f"[{_ts()}] err {cell_label}  {short_err}",
                        file=sys.stderr,
                    )
                    # Append to error log
                    with open(error_log, "a") as ef:
                        ef.write(f"[{_ts()}] {cell_label}\n{result['error_message']}\n{'='*60}\n")
                    console.print(
                        f"[bold red][ERROR][/bold red] {cell_label}: "
                        f"{result['error_message'][:200]}"
                    )

                completed_rows.append(result)
                recent.append(result)
                progress.advance(task_id)

                # Update live display
                from rich.columns import Columns
                live.update(Panel(Columns([progress, _recent_table()])))

                # Incremental progress parquet save
                pd.DataFrame(completed_rows).to_parquet(progress_path)

    # Final summary
    n_ok = sum(1 for r in completed_rows if r["status"] == "ok")
    n_err = sum(1 for r in completed_rows if r["status"] == "error")
    n_skip = sum(1 for r in completed_rows if r["status"] == "skip")
    console.print(
        f"\n[bold]Done.[/bold]  ok={n_ok}  errors={n_err}  skipped={n_skip}  "
        f"total_processed={len(completed_rows)}"
    )
    if n_err:
        console.print(f"[yellow]Error details in {error_log}[/yellow]")
    if shutdown_requested:
        console.print("[yellow]Run was interrupted by SIGINT. Re-run to resume.[/yellow]")


if __name__ == "__main__":
    main()
