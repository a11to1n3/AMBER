"""Native view-API SIR uses a cell-list infection (scales past all-pairs OOM)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from models.amber_models import AMBERVectorizedSIRModel  # noqa: E402


def _run_sir(n: int, steps: int = 10, seed: int = 0, **extra):
    cfg = {
        "n": n,
        "steps": steps,
        "show_progress": False,
        "seed": seed,
        "world_size": 100,
        "movement_speed": 2.0,
        "infection_radius": 5.0,
        "transmission_rate": 0.1,
        "recovery_time": 14,
        "initial_infected": 5,
        "max_per_cell": 64,
    }
    cfg.update(extra)
    return AMBERVectorizedSIRModel(cfg).cpu(mode="vectorized").run()


def test_sir_cell_list_conserves_population():
    n = 5_000
    res = _run_sir(n, steps=20, seed=1)
    last = res.model.to_dicts()[-1]
    assert last["susceptible"] + last["infected"] + last["recovered"] == n


def test_sir_cell_list_epidemic_spreads():
    res = _run_sir(2_000, steps=30, seed=2)
    last = res.model.to_dicts()[-1]
    assert last["infected"] + last["recovered"] > 5


def test_sir_cell_list_scales_past_allpairs_oom_n():
    """All-pairs S×I OOMs around 1e5; cell-list must finish at 1e5."""
    n = 100_000
    res = _run_sir(n, steps=3, seed=3)
    last = res.model.to_dicts()[-1]
    assert last["susceptible"] + last["infected"] + last["recovered"] == n
    assert last["infected"] > 5
