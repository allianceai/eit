#!/usr/bin/env python
"""Assemble the Neurocomputing submission source bundle.

(1) Flattens manuscript_neurocomputing.tex by recursively inlining \\input{...}
    (sections + tables) into a single self-contained submission/source/manuscript.tex
    (figures stay external; \\bibliography{references} kept with references.bib).
(2) Copies the exact figures/bib/class/bst the manuscript needs into source/.
(3) Zips source/ -> submission/neurocomputing_submission.zip.

Usage:  python -m scripts.paper_revision.build_submission_bundle
"""
from __future__ import annotations
import re, shutil, zipfile
from pathlib import Path

ROOT = Path("paper_v2")
MAIN = ROOT / "manuscript_neurocomputing.tex"
SRC = ROOT / "submission" / "source"
ZIP = ROOT / "submission" / "neurocomputing_submission.zip"

FIGURES = ["fig1_interventional", "fig_cd", "fig_frontier", "fig_overlap", "fig_regime_map"]
EXTRA = ["references.bib", "elsarticle.cls", "elsarticle-num.bst"]

INPUT_RE = re.compile(r'^[ \t]*\\input\{([^}]+)\}[ \t]*$', re.M)


def resolve(name: str) -> Path:
    p = ROOT / name
    if p.suffix != ".tex" and not p.exists():
        p = ROOT / (name + ".tex")
    return p


def flatten(path: Path) -> str:
    text = path.read_text()

    def repl(m):
        inc = m.group(1)
        ip = resolve(inc)
        if ip.exists() and ("sections/" in inc or "tables/" in inc):
            return f"% --- begin {inc} ---\n{flatten(ip)}\n% --- end {inc} ---"
        return m.group(0)  # leave non-local inputs alone

    return INPUT_RE.sub(repl, text)


def main():
    if SRC.exists():
        shutil.rmtree(SRC)
    (SRC / "figures").mkdir(parents=True)

    flat = flatten(MAIN)
    (SRC / "manuscript.tex").write_text(flat)

    for f in FIGURES:
        shutil.copy(ROOT / "figures" / f"{f}.pdf", SRC / "figures" / f"{f}.pdf")
    for f in EXTRA:
        shutil.copy(ROOT / f, SRC / f)

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SRC.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(SRC))

    n_input = len(INPUT_RE.findall(MAIN.read_text()))
    remaining = len(INPUT_RE.findall(flat))
    print(f"flattened manuscript.tex: {n_input} top-level \\input -> {remaining} remaining")
    print(f"bundle: {ZIP}  ({ZIP.stat().st_size//1024} KB)")
    print("contents:", ", ".join(sorted(p.name for p in SRC.rglob("*") if p.is_file())))


if __name__ == "__main__":
    main()
