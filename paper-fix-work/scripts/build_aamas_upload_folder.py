#!/usr/bin/env python3
"""Build an unambiguous AAMAS upload folder and supplementary archive."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "AAMAS-submission-ready"
MAIN_DIR = OUTPUT / "01-main-paper"
SUPPLEMENT_DIR = OUTPUT / "02-supplementary-material"
LOCAL_DIR = OUTPUT / "99-local-reference-do-not-upload"
FIXED_TIME = (2026, 7, 18, 0, 0, 0)

MAIN_SOURCE = ROOT / "amber_aamas_submission.pdf"
SUPPLEMENT_SOURCE = ROOT / "amber_aamas_supplement.pdf"
COMBINED_SOURCE = ROOT / "amber_aamas.pdf"
LOCAL_ARCHIVE_SOURCE = ROOT / "AMBER-AAMAS-submission-package.zip"

MAIN_OUTPUT = MAIN_DIR / "AMBER_AAMAS_main-paper.pdf"
SUPPLEMENT_OUTPUT = SUPPLEMENT_DIR / "AMBER_AAMAS_supplementary-material.zip"
COMBINED_OUTPUT = LOCAL_DIR / "AMBER_AAMAS_combined-with-appendix.pdf"
LOCAL_ARCHIVE_OUTPUT = LOCAL_DIR / "AMBER_AAMAS_source-and-reproducibility-archive.zip"

TOP_LEVEL_SUPPLEMENT = [
    "REPRODUCIBILITY.md",
    "FIGURE_MANIFEST.md",
]

SUPPLEMENT_SCRIPTS = [
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

SUPPLEMENT_README = """# AMBER supplementary material

This anonymous archive accompanies the AAMAS review submission.

- `AMBER_AAMAS_supplement.pdf` contains the technical appendix, additional
  experimental details, supporting figures, and implementation listings.
- `REPRODUCIBILITY.md` maps the paper's claims to raw artifacts and executable
  checks.
- `FIGURE_MANIFEST.md` maps each figure to its source artifact and renderer.
- `artifacts/` contains the released JSON and tabular evidence.
- `scripts/` contains the analysis and deterministic figure renderers.
- `reproducibility/code_snapshot/` contains the anonymous AMBER source,
  benchmark harnesses, and tests used for this revision.

The primary paper is self-contained. Reviewers may consult this material at
their discretion. No external mutable links are required to inspect it.
"""

UPLOAD_INSTRUCTIONS = """# AAMAS upload set

Upload exactly these two files through their corresponding submission fields:

1. **Primary paper:**
   `01-main-paper/AMBER_AAMAS_main-paper.pdf`
2. **Supplementary material:**
   `02-supplementary-material/AMBER_AAMAS_supplementary-material.zip`

The primary PDF contains eight content pages followed by one references-only
page. The supplementary ZIP contains the separate appendix PDF, anonymous code
snapshot, raw artifacts, and reproduction scripts.

Do **not** upload anything under `99-local-reference-do-not-upload/`. Those
files are retained only for local inspection and archival convenience.

Before submission, replace `Paper ID: TBD` in the manuscript source with the
assigned tracking number, rebuild the PDFs, and rerun
`scripts/build_aamas_upload_folder.py`.
"""


def supplement_files() -> list[tuple[Path, str]]:
    files = [(SUPPLEMENT_SOURCE, "AMBER_AAMAS_supplement.pdf")]
    files += [(ROOT / name, name) for name in TOP_LEVEL_SUPPLEMENT]
    files += [
        (ROOT / "scripts" / name, f"scripts/{name}")
        for name in SUPPLEMENT_SCRIPTS
    ]
    files += [
        (path, path.relative_to(ROOT).as_posix())
        for path in sorted((ROOT / "artifacts").rglob("*"))
        if path.is_file() and path.name != ".DS_Store"
    ]
    files += [
        (path, path.relative_to(ROOT).as_posix())
        for path in sorted((ROOT / "reproducibility" / "code_snapshot").rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and "ambr.egg-info" not in path.parts
        and path.suffix != ".pyc"
        and path.name != ".DS_Store"
    ]
    missing = [path for path, _ in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing supplementary inputs: {missing}")
    return sorted(files, key=lambda item: item[1])


def assert_anonymous(files: list[tuple[Path, str]]) -> None:
    for path, archive_name in files:
        content = path.read_bytes().lower()
        for marker in PRIVATE_MARKERS:
            if marker.lower() in content:
                raise RuntimeError(
                    f"private marker {marker!r} found in {archive_name}"
                )


def write_member(archive: ZipFile, name: str, content: bytes) -> None:
    info = ZipInfo(name, FIXED_TIME)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content, compresslevel=9)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    required = [MAIN_SOURCE, SUPPLEMENT_SOURCE, COMBINED_SOURCE, LOCAL_ARCHIVE_SOURCE]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing upload-folder inputs: {missing}")

    for directory in (MAIN_DIR, SUPPLEMENT_DIR, LOCAL_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(MAIN_SOURCE, MAIN_OUTPUT)
    shutil.copyfile(COMBINED_SOURCE, COMBINED_OUTPUT)
    shutil.copyfile(LOCAL_ARCHIVE_SOURCE, LOCAL_ARCHIVE_OUTPUT)

    files = supplement_files()
    assert_anonymous(files)
    with ZipFile(SUPPLEMENT_OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        write_member(archive, "README.md", SUPPLEMENT_README.encode("utf-8"))
        for path, archive_name in files:
            write_member(archive, archive_name, path.read_bytes())

    instructions = (
        UPLOAD_INSTRUCTIONS
        + "\n## Integrity record\n\n"
        + f"- Primary PDF SHA-256: `{sha256(MAIN_OUTPUT)}`\n"
        + f"- Supplementary ZIP SHA-256: `{sha256(SUPPLEMENT_OUTPUT)}`\n"
    )
    (OUTPUT / "UPLOAD-INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")

    print(f"wrote {OUTPUT.relative_to(ROOT)}/")
    print(f"primary: {MAIN_OUTPUT.stat().st_size} bytes")
    print(
        f"supplement: {SUPPLEMENT_OUTPUT.stat().st_size} bytes, "
        f"{len(files) + 1} files"
    )
    print("WARNING: replace Paper ID TBD before submission")


if __name__ == "__main__":
    main()
