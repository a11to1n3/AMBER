"""Headless smoke for examples/virus_spread_simulation.py."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "virus_spread_simulation.py"


@pytest.mark.unit
def test_virus_run_headless_three_steps():
    # Import as a module path via runpy without executing __main__
    ns = runpy.run_path(str(EXAMPLE), run_name="not_main")
    run_headless = ns["run_headless"]
    model = run_headless(steps=3, n=25, seed=1)
    assert model.t == 3
    assert len(model.infected_history) >= 1
    assert model.susceptible_history[-1] + model.infected_history[-1] + model.recovered_history[-1] == 25


@pytest.mark.unit
def test_virus_cli_headless_subprocess():
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE), "--headless", "--steps", "3", "--n", "20"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "headless ok" in proc.stdout


@pytest.mark.unit
def test_virus_uses_run_step_not_manual_step_update():
    """Guard against regression to direct step()/update() driving."""
    text = EXAMPLE.read_text(encoding="utf-8")
    assert "run_step()" in text
    # The interactive loop must not pair bare step+update (lifecycle bug).
    assert "self.model.step()" not in text
    assert "self.model.update()" not in text
    # No module-global random; use model.rng only
    assert "import random" not in text
    assert "model.rng" in text
