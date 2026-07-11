"""CPU / NumPy tests for the batched GPU ensemble runner.

These exercise the public ensemble API without CUDA: ``get_array_module()``
falls back to NumPy when CuPy is unavailable (the common CI case).
"""

from __future__ import annotations

import numpy as np
import pytest

from ambr.gpu import GPU_AVAILABLE, get_array_module, to_host
from ambr.gpu_ensemble import (
    BatchedWellMixedSIR,
    GPUEnsembleRunner,
    smac_batch_calibrate,
)


def test_get_array_module_numpy_fallback():
    xp = get_array_module(prefer_gpu=False)
    assert xp is np


def test_ensemble_runner_shapes_and_invariants():
    """B independent SIR runs → trajectories of shape (B, steps)."""
    model = BatchedWellMixedSIR()
    runner = GPUEnsembleRunner(model)
    B, N, steps = 4, 100, 12
    params = {
        "beta": np.linspace(0.1, 0.4, B),
        "gamma": np.full(B, 0.05),
        "i0_frac": np.full(B, 0.05),
    }
    traj = runner.run(n_agents=N, steps=steps, params=params, seed=1)

    assert set(traj) == {"I_frac"}
    i_frac = to_host(traj["I_frac"])
    assert i_frac.shape == (B, steps)
    assert np.all(np.isfinite(i_frac))
    assert np.all(i_frac >= 0.0) and np.all(i_frac <= 1.0)
    # With a seed and i0_frac > 0, prevalence should move for at least one run.
    assert i_frac.max() > 0.0


def test_ensemble_runner_reproducible_with_seed():
    model = BatchedWellMixedSIR()
    runner = GPUEnsembleRunner(model)
    params = {
        "beta": np.array([0.25, 0.35]),
        "gamma": np.array([0.08, 0.08]),
        "i0_frac": np.array([0.02, 0.02]),
    }
    a = to_host(runner.run(80, 8, params, seed=42)["I_frac"])
    b = to_host(runner.run(80, 8, params, seed=42)["I_frac"])
    c = to_host(runner.run(80, 8, params, seed=43)["I_frac"])
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_batched_sir_setup_seeds_initial_infected():
    """setup uses i0_frac to mark the first floor(i0_frac*N) agents infected."""
    model = BatchedWellMixedSIR()
    xp = get_array_module()
    B, N = 2, 50
    params = {
        "beta": xp.asarray([[0.2], [0.2]], dtype=xp.float32),
        "gamma": xp.asarray([[0.1], [0.1]], dtype=xp.float32),
        "i0_frac": xp.asarray([[0.1], [0.2]], dtype=xp.float32),
    }
    rng = xp.random.default_rng(0)
    state = model.setup(B, N, params, rng)
    status = to_host(state["status"])
    assert status.shape == (B, N)
    # First 5 agents of run 0, first 10 of run 1
    assert (status[0, :5] == model.INFECTED).all()
    assert (status[0, 5:] == model.S).all()
    assert (status[1, :10] == model.INFECTED).all()
    assert (status[1, 10:] == model.S).all()


def test_to_host_roundtrip_numpy():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = to_host(arr)
    np.testing.assert_array_equal(out, arr)
    assert isinstance(out, np.ndarray)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("smac"),
    reason="smac not installed",
)
def test_smac_batch_calibrate_tiny_budget():
    """Smoke: ask/tell loop returns a config and non-empty history."""
    model = BatchedWellMixedSIR()
    target = 0.15

    def loss_fn(traj):
        # Prefer mid-range peak prevalence.
        peaks = to_host(traj["I_frac"]).max(axis=1)
        return (peaks - target) ** 2

    best, history = smac_batch_calibrate(
        model,
        param_bounds={"beta": (0.05, 0.5)},
        loss_fn=loss_fn,
        n_agents=40,
        steps=8,
        rounds=2,
        batch_size=4,
        fixed_params={"gamma": 0.1, "i0_frac": 0.05},
        seed=0,
        quiet=True,
    )
    assert isinstance(history, list) and len(history) == 2
    assert all(isinstance(x, float) for x in history)
    # Incumbent may be None on some SMAC builds with tiny budgets; history is enough.
    if best is not None:
        assert "beta" in best
        assert 0.05 <= float(best["beta"]) <= 0.5


def test_gpu_flag_is_bool():
    assert isinstance(GPU_AVAILABLE, bool)
