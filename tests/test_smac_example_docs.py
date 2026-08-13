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
