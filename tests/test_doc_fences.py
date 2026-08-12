"""CI smoke: execute **self-contained** Python fences from selected docs.

This suite is a **regression smoke**, not a proof that every tutorial or
example script is end-to-end runnable:

* Only the paths in ``DOC_PATHS`` are scanned (not all of ``docs/``).
* Fences on ``FRAGMENT_ALLOWLIST`` are syntax-checked only (multi-cell
  continuations, incomplete recipes).
* ``HEAVY_ALLOWLIST`` fences are syntax-only unless ``AMBER_DOC_FENCE_FULL=1``.
* Full calibration scripts under ``examples/smac_*.py`` and multi-cell
  tutorial programs are **not** executed here — run those scripts (or
  ``scripts/run_gpu_claims.py`` for GPU) separately.

Intentional API fragments (method bodies, incomplete context) are allowlisted.
Large-N samples are scaled down in CI so the matrix stays fast.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Paths scanned for fenced Python (relative to repo root).
DOC_PATHS = [
    "README.md",
    "docs/quickstart.rst",
    "docs/tutorial.rst",
    "docs/installation.rst",
    "docs/going_faster.rst",
    "docs/index.rst",
    "docs/api/agent.rst",
    "docs/api/base.rst",
    "docs/api/contract.rst",
    "docs/api/environments.rst",
    "docs/api/experiment.rst",
    "docs/api/gpu.rst",
    "docs/api/gpu_ensemble.rst",
    "docs/api/model.rst",
    "docs/api/optimization.rst",
    "docs/api/performance.rst",
    "docs/api/results.rst",
]

# Fence ids that are intentional fragments or multi-cell continuations.
# Format: "relative/path:index" (0-based fence index within the file).
FRAGMENT_ALLOWLIST = {
    # View-API recipe body (not a full script).
    "README.md:3",
    # Tutorial-style continuation / SMAC stub comments only.
    "docs/api/optimization.rst:2",  # random_search needs prior MyModel+space
    "docs/api/optimization.rst:3",  # bayesian needs advanced extra + prior
    # Method-only / predicate demos without a Model shell.
    "docs/quickstart.rst:2",
    "docs/quickstart.rst:3",
    "docs/going_faster.rst:0",  # step body fragment in RST
    "docs/api/agent.rst:2",
    "docs/api/base.rst:0",  # class sketch without run()
    "docs/api/base.rst:1",  # Agent sketch without model shell
    "docs/api/sequences.rst:0",
    "docs/api/sequences.rst:1",
    # Tutorial multi-cell continuations (need prior class definitions).
    "docs/tutorial.rst:1",  # runs WealthModel from prior fence
    "docs/tutorial.rst:3",  # continues SpatialWealthModel
    "docs/tutorial.rst:5",  # plots AnalyticalWealthModel + plt
    "docs/tutorial.rst:7",  # random_search needs prior model+space
    "docs/tutorial.rst:9",  # experiment_results from prior fence
    # ParallelRunner fence is spawn-safe as a *file*, but executing the fence
    # body via a temp script still defines MyModel in __main__ → spawn fail.
    "docs/api/performance.rst:0",
}

# Fences that are complete but too heavy for every CI matrix cell.
# Still syntax-checked; execution is opt-in via AMBER_DOC_FENCE_FULL=1.
HEAVY_ALLOWLIST = {
    "README.md:5",  # 1M agents sample
    "README.md:2",  # 100k + Drift (ok but slower)
}


def _extract_md(text: str) -> list[str]:
    return [
        m.group(1)
        for m in re.finditer(r"```(?:python|py)\n(.*?)```", text, re.S | re.I)
    ]


def _extract_rst(text: str) -> list[str]:
    pattern = re.compile(
        r"\.\.\s*code-block::\s*python\s*\n((?:\n|(?:[ \t].*\n)|(?:[ \t]*\n))*)",
        re.I,
    )
    items: list[str] = []
    for m in pattern.finditer(text):
        block = m.group(1)
        lines = block.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            continue
        indents = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
        ind = min(indents) if indents else 0
        code = "\n".join(l[ind:] if len(l) >= ind else l for l in lines).rstrip() + "\n"
        items.append(code)
    return items


def _collect_fences() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for rel in DOC_PATHS:
        path = ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        codes = _extract_md(text) if path.suffix == ".md" else _extract_rst(text)
        for i, code in enumerate(codes):
            out.append((f"{rel}:{i}", code))
    return out


def _is_self_contained(code: str) -> bool:
    """Heuristic: has imports and either defines a runnable model or status/print."""
    # Explicit multi-cell / continuation markers must never run alone.
    if re.search(
        r"(?i)continues? (the )?(previous|prior|part)|paste both blocks|"
        r"must be defined in the previous",
        code,
    ):
        return False
    has_import = bool(re.search(r"^\s*(import|from)\s", code, re.M))
    if not has_import:
        return False
    if "class " in code and (".run(" in code or "grid_search(" in code or "Experiment(" in code):
        return True
    if "print_status" in code or "recommend(" in code:
        return True
    if "GPUEnsembleRunner" in code or "grid_search(" in code:
        return True
    if "print(" in code and "ambr" in code and "class " not in code:
        # version / status snippets — only if they don't reference undefined models
        if re.search(r"\b[A-Z][A-Za-z0-9_]*Model\b", code):
            return False
        return True
    return False


def _scale_for_ci(code: str) -> str:
    """Shrink large agent counts / step counts for CI wall-clock."""
    if os.environ.get("AMBER_DOC_FENCE_FULL", "").strip() in ("1", "true", "yes"):
        return code
    scaled = code
    # Common large-N literals in docs
    scaled = re.sub(r"\b1_000_000\b", "2_000", scaled)
    scaled = re.sub(r"\b100_000\b", "2_000", scaled)
    scaled = re.sub(r"\b50_000\b", "2_000", scaled)
    scaled = re.sub(r"\b10_000\b", "1_000", scaled)
    # Cap steps in dict literals when very large
    scaled = re.sub(
        r"(['\"]steps['\"]\s*:\s*)(\d{3,})",
        lambda m: m.group(1) + str(min(int(m.group(2)), 20)),
        scaled,
    )
    return scaled


def _run_code(code: str, timeout: float = 90.0) -> None:
    # Windows CI uses a legacy console code page; force UTF-8 for source + I/O
    # so doc fences with en-dashes / box-drawing prints do not crash.
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "AMBER_SUPPRESS_DEPRECATIONS": "1",
        "MPLBACKEND": "Agg",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    # Write as real UTF-8 bytes (NamedTemporaryFile text mode uses locale encoding
    # on Windows and can mangle non-ASCII into cp1252, then SyntaxError on \x97).
    body = "# -*- coding: utf-8 -*-\n" + code
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(body.encode("utf-8"))
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            cwd=str(ROOT),
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if proc.returncode != 0:
        raise AssertionError(
            f"exit={proc.returncode}\n"
            f"stdout:\n{(proc.stdout or '')[-1500:]}\n"
            f"stderr:\n{(proc.stderr or '')[-2000:]}"
        )


FENCES = _collect_fences()


@pytest.mark.unit
@pytest.mark.parametrize("fence_id,code", FENCES, ids=[f[0] for f in FENCES])
def test_doc_fence_syntax(fence_id: str, code: str):
    """Every fence must be valid Python syntax."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"{fence_id}: SyntaxError: {e}")


@pytest.mark.unit
@pytest.mark.parametrize(
    "fence_id,code",
    [
        (fid, code)
        for fid, code in FENCES
        if fid not in FRAGMENT_ALLOWLIST and _is_self_contained(code)
    ],
    ids=[
        fid
        for fid, code in FENCES
        if fid not in FRAGMENT_ALLOWLIST and _is_self_contained(code)
    ],
)
def test_doc_fence_runs(fence_id: str, code: str):
    """Self-contained fences must execute (scaled down unless FULL=1)."""
    if fence_id in HEAVY_ALLOWLIST and os.environ.get("AMBER_DOC_FENCE_FULL", "").strip() not in (
        "1",
        "true",
        "yes",
    ):
        # Still run scaled version for heavy fences
        pass
    scaled = _scale_for_ci(code)
    # Optional advanced deps: bayesian/smac may be missing on minimal installs
    if "bayesian_optimization" in scaled or "smac_batch_calibrate" in scaled:
        pytest.importorskip("smac")
    if "networkx" in scaled or "nx." in scaled:
        pytest.importorskip("networkx")
    _run_code(scaled)


@pytest.mark.unit
def test_fragment_allowlist_is_current():
    """Allowlist entries must refer to fences that still exist."""
    known = {fid for fid, _ in FENCES}
    stale = sorted(FRAGMENT_ALLOWLIST - known)
    # Allow paths we don't scan (sequences) without failing hard — only check scanned
    stale_scanned = [s for s in stale if s.split(":")[0] in DOC_PATHS]
    assert not stale_scanned, f"Stale FRAGMENT_ALLOWLIST entries: {stale_scanned}"
