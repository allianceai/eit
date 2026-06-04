#!/usr/bin/env python
"""Graphical abstract for the Neurocomputing submission.

Landscape panel (~2.5:1, Elsevier spec >= 531 x 1328 px) summarizing the headline
remedy decomposition + the practical takeaway. Pure schematic from reported numbers
(Table: remedy decomposition, mean over datasets).

Usage:  python -m scripts.paper_revision.build_graphical_abstract
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

FIG = Path("paper_v2/figures")

# remedy decomposition of the misclassified minority (mean over datasets, %)
SEGS = [
    ("Threshold-recoverable", 69, "#2c7fb8", "move the threshold\n(no new data)"),
    ("Data-reducible",         7, "#f4a936", "add minority\ndata"),
    ("Irreducible",           16, "#9e3b3b", "genuine\noverlap"),
    ("Noise",                  7, "#b8b8b8", "mislabeled"),
]


def main():
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(12.8, 5.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 10); ax.axis("off")

    ax.text(50, 9.4,
            "Don't oversample the boundary: diagnose the minority deficit, then move the decision",
            ha="center", va="center", fontsize=16.5, fontweight="bold")

    ax.text(2, 8.4, "Misclassified minority errors  (mean over 79 datasets)",
            ha="left", va="center", fontsize=12.5)

    # stacked bar
    total = sum(v for _, v, _, _ in SEGS)
    x, barw, y0, h = 2.0, 96.0, 5.4, 1.5
    for name, val, col, remedy in SEGS:
        w = val / total * barw
        ax.add_patch(plt.Rectangle((x, y0), w, h, color=col, ec="white", lw=1.6))
        ax.text(x + w / 2, y0 + h / 2, f"{val}%", ha="center", va="center",
                color="white", fontsize=13.5, fontweight="bold")
        ax.text(x + w / 2, y0 + h + 0.30, name, ha="center", va="bottom",
                fontsize=10 if w > 8 else 8.5)
        ax.text(x + w / 2, y0 - 0.30, remedy, ha="center", va="top",
                fontsize=8.8, style="italic", color="0.25")
        x += w

    # takeaways
    ax.text(2, 3.15,
            "Most minority errors are threshold-recoverable: the classifier already ranks them correctly.",
            fontsize=12.5, va="center", fontweight="bold")
    ax.text(2, 2.05,
            "A threshold move reproduces oversampling's benefit — no synthetic data, better calibration.",
            fontsize=11, va="center", color="0.2")
    ax.text(2, 1.05,
            "Oversampling adds no ranking gain; at equal threshold tuning the generative vs. "
            "non-generative gap is within ±1 pp.",
            fontsize=11, va="center", color="0.2")

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "graphical_abstract.pdf", bbox_inches="tight")
    fig.savefig(FIG / "graphical_abstract.png", dpi=220, bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG/'graphical_abstract.pdf'} and .png")


if __name__ == "__main__":
    main()
