"""Canonical random-walk reference (identity-keyed displacements, clip)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rng.counter_rng import EVT_DISPLACE_X, EVT_DISPLACE_Y, u01  # noqa: E402


def _disp(seed: int, t: int, event: int, agent: int, speed: float) -> float:
    return (2.0 * u01(seed, t, event, agent) - 1.0) * speed


def step_walk(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    t: int,
    world_size: float = 100.0,
    speed: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    nx = x.copy()
    ny = y.copy()
    for i in range(n):
        nx[i] = np.clip(x[i] + _disp(seed, t, EVT_DISPLACE_X, i, speed), 0.0, world_size)
        ny[i] = np.clip(y[i] + _disp(seed, t, EVT_DISPLACE_Y, i, speed), 0.0, world_size)
    return nx, ny


def run_walk(
    n: int,
    steps: int,
    seed: int = 0,
    world_size: float = 100.0,
    speed: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    # Deterministic identity-keyed init (not native stream).
    from rng.counter_rng import EVT_INIT, u01 as u

    x = np.array([u(seed, 0, EVT_INIT, i, 0, 0) * world_size for i in range(n)], dtype=np.float64)
    y = np.array([u(seed, 0, EVT_INIT, i, 0, 1) * world_size for i in range(n)], dtype=np.float64)
    for t in range(steps):
        x, y = step_walk(x, y, seed, t, world_size=world_size, speed=speed)
    return x, y
