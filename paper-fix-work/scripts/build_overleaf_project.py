#!/usr/bin/env python3
"""Build a minimal, deterministic, self-contained Overleaf project."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEX = ROOT / "amber_aamas.tex"
SOURCE_BIB = ROOT / "amber_references.bib"
PROJECT = ROOT / "AAMAS-Overleaf-project"
OUTPUT_ZIP = ROOT / "AAMAS-Overleaf-project.zip"
FIXED_TIME = (2026, 7, 18, 0, 0, 0)

README = """# AMBER AAMAS Overleaf project

## Import

1. In Overleaf, choose **New Project -> Upload Project**.
2. Upload `AAMAS-Overleaf-project.zip`.
3. Confirm that the main document is `main.tex`.
4. Select **XeLaTeX** as the compiler if Overleaf does not compile it
   automatically with the project default.

The project contains the complete manuscript source, bibliography, and every
referenced vector figure. It intentionally omits generated logs, auxiliary
files, raw experiment data, and local review notes because none is needed to
compile the paper.

Before submission, replace `TBD` in `\\newcommand{\\submissionid}{TBD}` with
the assigned AAMAS tracking number.

The source compiles the complete manuscript with its appendix. The separate
submission and supplementary PDFs should be produced from the reviewed local
build after the tracking number is inserted.
"""


def referenced_figures(tex: str) -> list[Path]:
    names = sorted(set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)))
    figures = [ROOT / name for name in names]
    missing = [path for path in figures if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing referenced figures: {missing}")
    return figures


def project_files() -> list[Path]:
    return sorted(
        (path for path in PROJECT.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(PROJECT).as_posix(),
    )


def write_archive(files: list[Path]) -> None:
    with ZipFile(OUTPUT_ZIP, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(PROJECT).as_posix()
            info = ZipInfo(relative, FIXED_TIME)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def main() -> None:
    tex = SOURCE_TEX.read_text(encoding="utf-8")
    figures = referenced_figures(tex)
    PROJECT.mkdir(parents=True, exist_ok=True)
    (PROJECT / "figs").mkdir(parents=True, exist_ok=True)

    shutil.copyfile(SOURCE_TEX, PROJECT / "main.tex")
    shutil.copyfile(SOURCE_BIB, PROJECT / SOURCE_BIB.name)
    for figure in figures:
        shutil.copyfile(figure, PROJECT / figure.relative_to(ROOT))
    (PROJECT / "README.md").write_text(README, encoding="utf-8")

    files = project_files()
    write_archive(files)
    print(
        f"wrote {OUTPUT_ZIP.name}: {len(files)} files, "
        f"{OUTPUT_ZIP.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
