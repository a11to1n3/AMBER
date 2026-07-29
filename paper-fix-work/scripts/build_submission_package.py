#!/usr/bin/env python3
"""Build the anonymous, deterministic AMBER paper submission archive."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "AMBER-AAMAS-submission-package.zip"
FIXED_TIME = (2026, 7, 18, 0, 0, 0)

TOP_LEVEL = [
    "amber_aamas.pdf",
    "amber_aamas_submission.pdf",
    "amber_aamas_supplement.pdf",
    "amber_aamas.tex",
    "amber_references.bib",
    "AI_ASSISTANCE_DISCLOSURE.md",
    "FIGURE_MANIFEST.md",
    "REPRODUCIBILITY.md",
    "REVIEW_REPORT.md",
]

PLOTS = ["plot01", "plot02", "plot03", "plot07", "plot11", "plot12",
         "plot13", "plot14", "plot15", "plot17", "plot18", "plot21"]

SCRIPTS = [
    "build_aamas_upload_folder.py",
    "build_overleaf_project.py",
    "build_submission_package.py",
    "emergence_threshold_controlled.py",
    "mf_granularity.py",
    "monitor_cost_current.py",
    "publication_figure_style.py",
    "reconcile_all_framework_benchmark.py",
    "render_publication_figures.py",
    "theorem_referee.py",
    "topological_staging_experiment.py",
]

PRIVATE_MARKERS = [
    b"a11" + b"to1n3",
    b"citation." + b"needed",
    b"/users/" + b"duy" + b"pham",
    b"duy" + b"pham",
    b"cdb4f135551df0c9d072" + b"aefa53c1f8510c682c6d",
    b"192.168." + b"178.86",
    b"103.116." + b"53.27",
    b"213.181." + b"111.2",
]


def selected_files() -> list[Path]:
    files = [ROOT / name for name in TOP_LEVEL]
    files += [ROOT / "figs" / f"{plot}.{suffix}" for plot in PLOTS
              for suffix in ("pdf", "png")]
    files += [ROOT / "scripts" / name for name in SCRIPTS]
    files += sorted(path for path in (ROOT / "artifacts").rglob("*") if path.is_file())
    files += sorted(
        path
        for path in (ROOT / "reproducibility" / "code_snapshot").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and "ambr.egg-info" not in path.parts
        and path.suffix != ".pyc"
        and path.name != ".DS_Store"
    )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package inputs: {missing}")
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def assert_anonymous(files: list[Path]) -> None:
    for path in files:
        content = path.read_bytes().lower()
        for marker in PRIVATE_MARKERS:
            if marker.lower() in content:
                raise RuntimeError(
                    f"private marker {marker!r} found in {path.relative_to(ROOT)}"
                )


def write_archive(files: list[Path]) -> None:
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = ZipInfo(relative, FIXED_TIME)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def main() -> None:
    files = selected_files()
    assert_anonymous(files)
    write_archive(files)
    paper_source = (ROOT / "amber_aamas.tex").read_text()
    status = "WARNING: replace Paper ID TBD" if "{TBD}" in paper_source else "paper ID set"
    print(f"wrote {OUTPUT.name}: {len(files)} files, {OUTPUT.stat().st_size} bytes; {status}")


if __name__ == "__main__":
    main()
