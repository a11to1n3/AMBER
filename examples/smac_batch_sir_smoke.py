#!/usr/bin/env python3
"""Minimal ensemble + optional SMAC batch calibration smoke.

* Ensemble path always runs (NumPy if no CuPy).
* SMAC path requires ``pip install 'ambr[advanced]'`` and is skipped honestly
  when SMAC/ConfigSpace are missing.

Not a research-grade calibration. See ``docs/reproducibility.rst``.

Run::

    python examples/smac_batch_sir_smoke.py
"""

from __future__ import annotations

import sys

import numpy as np

import ambr as am
from ambr.gpu_ensemble import BatchedWellMixedSIR, GPUEnsembleRunner, smac_batch_calibrate


def run_ensemble_smoke() -> dict:
    B = 4
    runner = GPUEnsembleRunner(BatchedWellMixedSIR())
    traj = runner.run(
        n_agents=500,
        steps=15,
        params={
            "beta": np.linspace(0.15, 0.35, B),
            "gamma": np.full(B, 0.08),
            "i0_frac": np.full(B, 0.02),
        },
        seed=0,
    )
    shapes = {k: tuple(getattr(v, "shape", ())) for k, v in traj.items()}
    print("ensemble OK", shapes, "GPU_AVAILABLE=", am.GPU_AVAILABLE)
    return traj


def run_smac_smoke() -> None:
    try:
        import smac  # noqa: F401
        import ConfigSpace  # noqa: F401
    except ImportError:
        print(
            "SMAC path skipped — install with: pip install 'ambr[advanced]'\n"
            "(ensemble smoke above still ran)"
        )
        return

    def loss_fn(traj: dict):
        # traj["I_frac"] is (B, steps); return per-run MSE to a toy target
        arr = np.asarray(am.to_host(traj["I_frac"]))
        return (arr[:, -1] - 0.1) ** 2

    best, history = smac_batch_calibrate(
        BatchedWellMixedSIR(),
        param_bounds={"beta": (0.05, 0.5), "gamma": (0.01, 0.2)},
        loss_fn=loss_fn,
        n_agents=400,
        steps=12,
        rounds=2,
        batch_size=4,
        fixed_params={"i0_frac": 0.02},
        seed=0,
        quiet=True,
    )
    print("smac_batch OK best=", best, "history_len=", len(history))


def main() -> int:
    am.print_status()
    run_ensemble_smoke()
    run_smac_smoke()
    return 0


if __name__ == "__main__":
    sys.exit(main())
