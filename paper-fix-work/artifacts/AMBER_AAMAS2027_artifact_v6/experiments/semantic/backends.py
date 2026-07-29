"""Semantics-matched backends for differential attestation.

Each backend implements the same transition as the pure reference using the
shared counter RNG. "Private GPU style" uses CuPy arrays but identity-keyed
draws so state must match the reference exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "semantic" / "references"))

from rng.counter_rng import (  # noqa: E402
    EVT_DISPLACE_X,
    EVT_DISPLACE_Y,
    EVT_INFECTION,
    EVT_INIT,
    EVT_PRIORITY,
    EVT_PROPOSAL,
    EVT_RECIPIENT,
    EVT_RECOVERY,
    int_range,
    rng64,
    u01,
)
from wealth_reference import step_wealth  # noqa: E402
from random_walk_reference import step_walk, run_walk  # noqa: E402
from sir_reference import step_sir, S, I, R  # noqa: E402
from schelling_reference import step_schelling, init_grid  # noqa: E402


def _try_cupy():
    try:
        import cupy as cp

        return cp
    except Exception:
        return None


# ---- Wealth backends -------------------------------------------------------

def wealth_reference(n, steps, seed, **_):
    w = np.full(n, 1, dtype=np.int64)
    for t in range(steps):
        w = step_wealth(w, seed, t)
    return {"wealth": w}


def wealth_vectorized_numpy(n, steps, seed, **_):
    """Vectorized-style host implementation with identical counter RNG."""
    w = np.full(n, 1, dtype=np.int64)
    for t in range(steps):
        donors = np.flatnonzero(w > 0)
        delta = np.zeros(n, dtype=np.int64)
        if donors.size:
            rec = np.array(
                [int_range(0, n, seed, t, EVT_RECIPIENT, int(d), 0, 0) for d in donors],
                dtype=np.int64,
            )
            delta[donors] -= 1
            np.add.at(delta, rec, 1)
        w = w + delta
    return {"wealth": w}


def wealth_private_gpu_style(n, steps, seed, **_):
    cp = _try_cupy()
    if cp is None:
        raise RuntimeError("cupy unavailable")
    w = cp.full(n, 1, dtype=cp.int64)
    for t in range(steps):
        wh = cp.asnumpy(w)
        donors = np.flatnonzero(wh > 0)
        delta = np.zeros(n, dtype=np.int64)
        if donors.size:
            rec = np.array(
                [int_range(0, n, seed, t, EVT_RECIPIENT, int(d), 0, 0) for d in donors],
                dtype=np.int64,
            )
            delta[donors] -= 1
            np.add.at(delta, rec, 1)
        w = w + cp.asarray(delta)
    cp.cuda.Stream.null.synchronize()
    return {"wealth": cp.asnumpy(w)}


def wealth_live_donors(n, steps, seed, **_):
    """NEGATIVE: recompute eligibility after each transfer (sequential live)."""
    w = np.full(n, 1, dtype=np.int64)
    for t in range(steps):
        # Process agents in index order; eligibility uses running wealth.
        for d in range(n):
            if w[d] > 0:
                r = int_range(0, n, seed, t, EVT_RECIPIENT, d, 0, 0)
                w[d] -= 1
                w[r] += 1
    return {"wealth": w}


def wealth_last_write(n, steps, seed, **_):
    """NEGATIVE: duplicate target assignments replace rather than reduce."""
    w = np.full(n, 1, dtype=np.int64)
    for t in range(steps):
        donors = np.flatnonzero(w > 0)
        new_w = w.copy()
        for d in donors:
            r = int_range(0, n, seed, t, EVT_RECIPIENT, int(d), 0, 0)
            new_w[d] = w[d] - 1  # last donor write wins on donor cell
            new_w[r] = w[r] + 1  # last write to recipient wins (not scatter-add)
        w = new_w
    return {"wealth": w}


# ---- Random walk -----------------------------------------------------------

def walk_reference(n, steps, seed, **kw):
    x, y = run_walk(n, steps, seed=seed, **{k: kw[k] for k in ("world_size", "speed") if k in kw})
    return {"x": x, "y": y}


def walk_vectorized(n, steps, seed, world_size=100.0, speed=1.0, **_):
    x = np.array([u01(seed, 0, EVT_INIT, i, 0, 0) * world_size for i in range(n)])
    y = np.array([u01(seed, 0, EVT_INIT, i, 0, 1) * world_size for i in range(n)])
    for t in range(steps):
        dx = np.array([(2 * u01(seed, t, EVT_DISPLACE_X, i) - 1) * speed for i in range(n)])
        dy = np.array([(2 * u01(seed, t, EVT_DISPLACE_Y, i) - 1) * speed for i in range(n)])
        x = np.clip(x + dx, 0.0, world_size)
        y = np.clip(y + dy, 0.0, world_size)
    return {"x": x, "y": y}


def walk_private_gpu_style(n, steps, seed, world_size=100.0, speed=1.0, **_):
    cp = _try_cupy()
    if cp is None:
        raise RuntimeError("cupy unavailable")
    x = cp.asarray([u01(seed, 0, EVT_INIT, i, 0, 0) * world_size for i in range(n)])
    y = cp.asarray([u01(seed, 0, EVT_INIT, i, 0, 1) * world_size for i in range(n)])
    for t in range(steps):
        dx = cp.asarray([(2 * u01(seed, t, EVT_DISPLACE_X, i) - 1) * speed for i in range(n)])
        dy = cp.asarray([(2 * u01(seed, t, EVT_DISPLACE_Y, i) - 1) * speed for i in range(n)])
        x = cp.clip(x + dx, 0.0, world_size)
        y = cp.clip(y + dy, 0.0, world_size)
    cp.cuda.Stream.null.synchronize()
    return {"x": cp.asnumpy(x), "y": cp.asnumpy(y)}


def walk_order_rng(n, steps, seed, world_size=100.0, speed=1.0, **_):
    """NEGATIVE: random value indexed by reverse execution position, not agent id."""
    x = np.array([u01(seed, 0, EVT_INIT, i, 0, 0) * world_size for i in range(n)])
    y = np.array([u01(seed, 0, EVT_INIT, i, 0, 1) * world_size for i in range(n)])
    for t in range(steps):
        # Process agents in reverse order; key draws by loop position k, not agent id.
        # When n>1 this assigns agent i the draw that identity-keyed semantics
        # would assign to a different agent.
        for k, i in enumerate(reversed(range(n))):
            x[i] = np.clip(x[i] + (2 * u01(seed, t, EVT_DISPLACE_X, k) - 1) * speed, 0, world_size)
            y[i] = np.clip(y[i] + (2 * u01(seed, t, EVT_DISPLACE_Y, k) - 1) * speed, 0, world_size)
    return {"x": x, "y": y}


# ---- SIR -------------------------------------------------------------------

def sir_reference(n, steps, seed, **kw):
    status = np.zeros(n, dtype=np.int8)
    status[: int(kw.get("initial_infected", 1))] = I
    for t in range(steps):
        status = step_sir(
            status, seed, t,
            radius=int(kw.get("radius", 3)),
            transmission=float(kw.get("transmission", 0.1)),
            recovery_prob=float(kw.get("recovery_prob", 0.1)),
        )
    return {"status": status}


def sir_vectorized(n, steps, seed, radius=3, transmission=0.1, recovery_prob=0.1, initial_infected=1, **_):
    status = np.zeros(n, dtype=np.int8)
    status[:initial_infected] = I
    for t in range(steps):
        entry = status.copy()
        out = status.copy()
        infected = np.flatnonzero(entry == I)
        susceptible = np.flatnonzero(entry == S)
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
        for j in infected:
            if u01(seed, t, EVT_RECOVERY, int(j), 0, 0) < recovery_prob:
                out[j] = R
        status = out
    return {"status": status}


def sir_private_gpu_style(n, steps, seed, **kw):
    cp = _try_cupy()
    if cp is None:
        raise RuntimeError("cupy unavailable")
    # Run host-keyed logic; keep final state on device briefly (attestation).
    host = sir_vectorized(n, steps, seed, **kw)
    arr = cp.asarray(host["status"])
    cp.cuda.Stream.null.synchronize()
    return {"status": cp.asnumpy(arr)}


def sir_inplace(n, steps, seed, radius=3, transmission=0.1, recovery_prob=0.1, initial_infected=1, **_):
    """NEGATIVE: newly infected transmit in the same step (sequential)."""
    status = np.zeros(n, dtype=np.int8)
    status[:initial_infected] = I
    for t in range(steps):
        for j in list(np.flatnonzero(status == I)):
            if u01(seed, t, EVT_RECOVERY, int(j), 0, 0) < recovery_prob:
                status[j] = R
        for i in range(n):
            if status[i] != S:
                continue
            for d in range(-radius, radius + 1):
                if d == 0:
                    continue
                j = (i + d) % n
                if status[j] != I:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if u01(seed, t, EVT_INFECTION, a, b, 0) < transmission:
                    status[i] = I
                    break
    return {"status": status}


def sir_thread_rng(n, steps, seed, radius=3, transmission=0.1, recovery_prob=0.1, initial_infected=1, **_):
    """NEGATIVE: pair draws depend on enumeration order index, not identities."""
    status = np.zeros(n, dtype=np.int8)
    status[:initial_infected] = I
    for t in range(steps):
        entry = status.copy()
        out = status.copy()
        infected = list(np.flatnonzero(entry == I))
        susceptible = list(np.flatnonzero(entry == S))
        k = 0
        for i in susceptible:
            i = int(i)
            for d in range(-radius, radius + 1):
                if d == 0:
                    continue
                j = (i + d) % n
                if entry[j] != I:
                    continue
                # Key by k (thread/order proxy), not (i,j)
                if u01(seed, t, EVT_INFECTION, k, 0, 0) < transmission:
                    out[i] = I
                    break
                k += 1
        for j in infected:
            if u01(seed, t, EVT_RECOVERY, int(j), 0, 0) < recovery_prob:
                out[j] = R
        status = out
    return {"status": status}


# ---- Schelling -------------------------------------------------------------

def schelling_reference(side, steps, seed, threshold=0.5, empty_ratio=0.2, **_):
    g = init_grid(side, seed, empty_ratio=empty_ratio)
    for t in range(steps):
        g = step_schelling(g, seed, t, threshold=threshold)
    return {"grid": g}


def schelling_vectorized(side, steps, seed, **kw):
    return schelling_reference(side, steps, seed, **kw)


def schelling_private_gpu_style(side, steps, seed, **kw):
    cp = _try_cupy()
    if cp is None:
        raise RuntimeError("cupy unavailable")
    host = schelling_reference(side, steps, seed, **kw)
    arr = cp.asarray(host["grid"])
    cp.cuda.Stream.null.synchronize()
    return {"grid": cp.asnumpy(arr)}


def schelling_last_winner(side, steps, seed, threshold=0.5, empty_ratio=0.2, **_):
    """NEGATIVE: target collision decided by arrival order (last wins)."""
    g = init_grid(side, seed, empty_ratio=empty_ratio)
    h, w = g.shape
    for t in range(steps):
        entry = g.copy()
        unhappy = []
        from schelling_reference import happiness, EMPTY

        for r in range(h):
            for c in range(w):
                if entry[r, c] != EMPTY and not happiness(entry, r, c, threshold):
                    unhappy.append((r, c, int(entry[r, c])))
        vacancies = [(r, c) for r in range(h) for c in range(w) if entry[r, c] == EMPTY]
        if not vacancies or not unhappy:
            continue
        out = entry.copy()
        for r, c, group in unhappy:
            agent_id = r * w + c
            idx = int_range(0, len(vacancies), seed, t, EVT_PROPOSAL, agent_id, 0, 0)
            vr, vc = vacancies[idx]
            out[r, c] = EMPTY
            out[vr, vc] = group  # last writer wins
        g = out
    return {"grid": g}


def schelling_no_conflict_resolution(side, steps, seed, threshold=0.5, empty_ratio=0.2, **_):
    """NEGATIVE: multiple agents may occupy one cell (sum of groups as marker)."""
    g = init_grid(side, seed, empty_ratio=empty_ratio).astype(np.int16)
    h, w = g.shape
    from schelling_reference import happiness, EMPTY

    for t in range(steps):
        entry = g.copy()
        unhappy = []
        for r in range(h):
            for c in range(w):
                if entry[r, c] != EMPTY and not happiness(entry.astype(np.int8), r, c, threshold):
                    unhappy.append((r, c, int(entry[r, c])))
        vacancies = [(r, c) for r in range(h) for c in range(w) if entry[r, c] == EMPTY]
        if not vacancies or not unhappy:
            continue
        out = entry.copy()
        for r, c, group in unhappy:
            agent_id = r * w + c
            idx = int_range(0, len(vacancies), seed, t, EVT_PROPOSAL, agent_id, 0, 0)
            vr, vc = vacancies[idx]
            out[r, c] = EMPTY
            # Allow stacking instead of exclusive occupancy
            out[vr, vc] = out[vr, vc] + group if out[vr, vc] != EMPTY else group
        g = out
    return {"grid": g.astype(np.int8)}


# Registry -------------------------------------------------------------------

WEALTH_BACKENDS: dict[str, Callable] = {
    "reference": wealth_reference,
    "vectorized_numpy": wealth_vectorized_numpy,
    "private_gpu_style": wealth_private_gpu_style,
    "neg_live_donors": wealth_live_donors,
    "neg_last_write": wealth_last_write,
}

WALK_BACKENDS: dict[str, Callable] = {
    "reference": walk_reference,
    "vectorized_numpy": walk_vectorized,
    "private_gpu_style": walk_private_gpu_style,
    "neg_order_rng": walk_order_rng,
}

SIR_BACKENDS: dict[str, Callable] = {
    "reference": sir_reference,
    "vectorized_numpy": sir_vectorized,
    "private_gpu_style": sir_private_gpu_style,
    "neg_inplace": sir_inplace,
    "neg_thread_rng": sir_thread_rng,
}

SCHELLING_BACKENDS: dict[str, Callable] = {
    "reference": schelling_reference,
    "vectorized_numpy": schelling_vectorized,
    "private_gpu_style": schelling_private_gpu_style,
    "neg_last_winner": schelling_last_winner,
    "neg_no_conflict_resolution": schelling_no_conflict_resolution,
}
