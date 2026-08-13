"""Execute example notebooks (and guard the agents_df-concat regression).

Full cell execution is gated by ``AMBER_EXECUTE_NOTEBOOKS=1`` so the default
matrix stays cheap. CI enables that flag in the dedicated notebooks job.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
NOTEBOOKS = (
    EXAMPLES / "button_network_simulation.ipynb",
    EXAMPLES / "flocking_simulation.ipynb",
    EXAMPLES / "forest_fire_simulation.ipynb",
)
_CONCAT_MARKERS = (
    "pl.concat([self.agents_df",
    "self.agents_df = pl.concat",
)


def _notebook_source(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        chunks.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(chunks)


@pytest.mark.unit
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_example_notebooks_exist_and_avoid_agents_df_concat(path: Path):
    assert path.is_file(), f"missing notebook {path}"
    source = _notebook_source(path)
    for marker in _CONCAT_MARKERS:
        assert marker not in source, (
            f"{path.name} still concatenates history into agents_df "
            f"(marker {marker!r}). Regenerate from the working .py example."
        )
    # Must actually define / run a model, not be an empty stub.
    assert "class " in source and ".run()" in source


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_example_notebooks_execute(path: Path):
    if os.environ.get("AMBER_EXECUTE_NOTEBOOKS") != "1":
        pytest.skip("Set AMBER_EXECUTE_NOTEBOOKS=1 to execute example notebooks")

    os.environ.setdefault("MPLBACKEND", "Agg")
    nb = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src)
        if not text.strip():
            continue
        try:
            exec(compile(text, f"{path.name}:cell{index}", "exec"), namespace)
        except Exception as exc:
            raise AssertionError(
                f"{path.name} cell {index} failed:\n{text}\n{type(exc).__name__}: {exc}"
            ) from exc
    assert "results" in namespace
