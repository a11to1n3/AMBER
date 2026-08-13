"""Contracts for SMAC example install docs and optional plotting."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SMAC_SCRIPTS = (
    EXAMPLES / "smac_calibration_simple.py",
    EXAMPLES / "smac_calibration_basic.py",
    EXAMPLES / "smac_calibration_advanced.py",
)


@pytest.mark.unit
@pytest.mark.parametrize("path", SMAC_SCRIPTS, ids=lambda p: p.name)
def test_smac_examples_document_viz_extra_and_optional_plots(path: Path):
    text = path.read_text(encoding="utf-8")
    assert "ambr[advanced,viz]" in text, (
        f"{path.name} must document pip install 'ambr[advanced,viz]' "
        "(SMAC + matplotlib). A clean ambr[advanced] install has no matplotlib."
    )
    # Plotting must be skippable — never a hard ImportError on the success path.
    assert "_try_matplotlib" in text
    assert "raise ImportError" not in text
    # Windows cp1252 cannot encode emoji / box-drawing on success.
    text.encode("cp1252")
    if path.name == "smac_calibration_advanced.py":
        assert "per objective" in text.lower() or "per-objective" in text.lower()
        assert "strategy=\"pareto\"" not in text
        assert "fixed_params" in text
        assert "steps" in text
        assert '"seed"' in text or "'seed'" in text
    if path.name == "smac_calibration_basic.py":
        assert "--full" in text
        assert "n_trials=10" in text or "n_trials: int = 10" in text
        assert "SMAC incumbent cost" in text
        # --help must parse argv before the smoke run / SMAC import.
        main = text.split('if __name__ == "__main__":', 1)[1]
        assert main.find("parse_args(") < main.find("WealthTransferModel({")
    if path.name == "smac_calibration_simple.py":
        assert "--full" in text
        assert "n_trials=10" in text or "n_trials: int = 10" in text
        assert "SMAC incumbent cost" in text
        main = text.split('if __name__ == "__main__":', 1)[1]
        assert main.find("parse_args(") < main.find("SimpleWealthModel({")


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    (
        "gpu_quickstart.py",
        "flocking_tensor.py",
        "button_network_simulation.py",
        "flocking_simulation.py",
        "forest_fire_simulation.py",
        "schelling_vectorized.py",
        "segregation_model.py",
    ),
)
def test_console_example_scripts_are_cp1252_safe(name: str):
    """Windows default consoles use cp1252; stdout snippets must encode."""
    (EXAMPLES / name).read_text(encoding="utf-8").encode("cp1252")
