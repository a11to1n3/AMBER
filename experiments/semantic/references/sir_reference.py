"""Canonical SIR-on-ring reference (snapshot infection, identity-keyed RVs)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rng.counter_rng import EVT_INFECTION, EVT_RECOVERY, u01  # noqa: E402

S, I, R = 0, 1, 2


def step_sir(
    status: np.ndarray,
    seed: int,
    t: int,
    radius: int = 3,
    transmission: float = 0.1,
    recovery_prob: float = 0.1,
) -> np.ndarray:
    """Snapshot SIR: newly infected cannot transmit in the same step."""
    n = status.shape[0]
    entry = status.copy()
    out = status.copy()
    infected = np.flatnonzero(entry == I)
    susceptible = np.flatnonzero(entry == S)

    # Infection: ring neighbors only (O(N * radius)); draws keyed by sorted
    # identities so candidate ordering cannot change the assigned RV.
    for i in susceptible:
        i = int(i)
        for d in range(-radius, radius + 1):
            if d == 0:
                continue
            j = (i + d) % n
            if entry[j] != I:
                continue
            a, b = (i, j) if i < j else (j, i)
            if u01(seed, t, EVT_INFECTION, a, b, 0) < transmission:
                out[i] = I
                break

    # Recovery from entry-infected only
    for j in infected:
        if u01(seed, t, EVT_RECOVERY, int(j), 0, 0) < recovery_prob:
            out[j] = R
    return out


def run_sir(
    n: int,
    steps: int,
    seed: int = 0,
    radius: int = 3,
    transmission: float = 0.1,
    recovery_prob: float = 0.1,
    initial_infected: int = 1,
) -> np.ndarray:
    status = np.zeros(n, dtype=np.int8)
    status[:initial_infected] = I
    for t in range(steps):
        status = step_sir(
            status, seed, t,
            radius=radius,
            transmission=transmission,
            recovery_prob=recovery_prob,
        )
    return status


def run_sir_inplace_sequential(
    n: int,
    steps: int,
    seed: int = 0,
    radius: int = 3,
    transmission: float = 0.1,
    recovery_prob: float = 0.1,
    initial_infected: int = 1,
    order: np.ndarray | None = None,
) -> np.ndarray:
    """Negative-control-style sequential: newly infected can transmit same step
    when activated later in the order. Used as activation contrast, not theorem
    instance.
    """
    status = np.zeros(n, dtype=np.int8)
    status[:initial_infected] = I
    if order is None:
        order = np.arange(n)
    for t in range(steps):
        # recovery first from current infected (same keys as snapshot for fairness
        # only when order is fixed identity; for activation study we reshuffle).
        entry = status.copy()
        for j in np.flatnonzero(entry == I):
            if u01(seed, t, EVT_RECOVERY, int(j), 0, 0) < recovery_prob:
                status[j] = R
        for i in order:
            if status[i] != S:
                continue
            for d in range(-radius, radius + 1):
                if d == 0:
                    continue
                j = (int(i) + d) % n
                if status[j] != I:
                    continue
                a, b = (int(i), j) if int(i) < j else (j, int(i))
                if u01(seed, t, EVT_INFECTION, a, b, 0) < transmission:
                    status[i] = I
                    break
    return status
