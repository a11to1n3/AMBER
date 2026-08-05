"""Smoke for ensemble + optional smac_batch_calibrate."""

from __future__ import annotations

import numpy as np
import pytest

import ambr as am
from ambr.gpu_ensemble import BatchedWellMixedSIR, GPUEnsembleRunner


@pytest.mark.unit
def test_ensemble_sir_numpy_or_cupy():
    B = 3
    traj = GPUEnsembleRunner(BatchedWellMixedSIR()).run(
        n_agents=200,
        steps=8,
        params={
            "beta": np.linspace(0.1, 0.3, B),
            "gamma": np.full(B, 0.1),
            "i0_frac": np.full(B, 0.05),
        },
        seed=0,
    )
    assert "I_frac" in traj
    shape = tuple(traj["I_frac"].shape)
    assert shape == (B, 8)


@pytest.mark.unit
def test_smac_batch_calibrate_optional():
    pytest.importorskip("smac")
    pytest.importorskip("ConfigSpace")
    from ambr.gpu_ensemble import smac_batch_calibrate

    def loss_fn(traj):
        arr = np.asarray(am.to_host(traj["I_frac"]))
        return (arr[:, -1] - 0.05) ** 2

    best, history = smac_batch_calibrate(
        BatchedWellMixedSIR(),
        param_bounds={"beta": (0.05, 0.4), "gamma": (0.02, 0.2)},
        loss_fn=loss_fn,
        n_agents=150,
        steps=8,
        rounds=1,
        batch_size=3,
        fixed_params={"i0_frac": 0.05},
        seed=0,
        quiet=True,
    )
    assert isinstance(history, list)
    assert len(history) == 1
