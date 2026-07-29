"""Canonical wealth-transfer reference (snapshot delta semantics)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rng.counter_rng import EVT_RECIPIENT, int_range  # noqa: E402


def step_wealth(wealth: np.ndarray, seed: int, t: int, allow_self: bool = True) -> np.ndarray:
    """Apply one snapshot wealth-transfer step.

    Donor eligibility from entry state; one recipient per eligible donor;
    accumulate deltas; apply once.
    """
    n = wealth.shape[0]
    delta = np.zeros(n, dtype=np.int64)
    donors = np.flatnonzero(wealth > 0)
    for d in donors:
        r = int_range(0, n, seed, t, EVT_RECIPIENT, int(d), 0, 0)
        if not allow_self and r == int(d):
            # Resample with draw_index bump until different (deterministic).
            k = 1
            while r == int(d):
                r = int_range(0, n, seed, t, EVT_RECIPIENT, int(d), 0, k)
                k += 1
        delta[d] -= 1
        delta[r] += 1
    return wealth + delta


def run_wealth(
    n: int,
    steps: int,
    seed: int = 0,
    initial_wealth: int = 1,
    allow_self: bool = True,
) -> np.ndarray:
    w = np.full(n, initial_wealth, dtype=np.int64)
    for t in range(steps):
        w = step_wealth(w, seed, t, allow_self=allow_self)
    return w


def wealth_trajectory(
    n: int,
    steps: int,
    seed: int = 0,
    initial_wealth: int = 1,
) -> list[np.ndarray]:
    w = np.full(n, initial_wealth, dtype=np.int64)
    out = [w.copy()]
    for t in range(steps):
        w = step_wealth(w, seed, t)
        out.append(w.copy())
    return out
