"""Canonical Schelling reference — three-stage synchronous conflict resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rng.counter_rng import EVT_PRIORITY, EVT_PROPOSAL, int_range, rng64  # noqa: E402

EMPTY = 0


def _neighbors(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def happiness(grid: np.ndarray, r: int, c: int, threshold: float) -> bool:
    g = grid[r, c]
    if g == EMPTY:
        return True
    neigh = _neighbors(grid, r, c)
    if not neigh:
        return True
    same = sum(1 for rr, cc in neigh if grid[rr, cc] == g)
    occupied = sum(1 for rr, cc in neigh if grid[rr, cc] != EMPTY)
    if occupied == 0:
        return True
    return (same / occupied) >= threshold


def step_schelling(
    grid: np.ndarray,
    seed: int,
    t: int,
    threshold: float = 0.5,
) -> np.ndarray:
    h, w = grid.shape
    entry = grid.copy()
    # Stage 1: happiness from entry
    unhappy = []
    for r in range(h):
        for c in range(w):
            if entry[r, c] != EMPTY and not happiness(entry, r, c, threshold):
                unhappy.append((r, c, int(entry[r, c])))

    vacancies = [(r, c) for r in range(h) for c in range(w) if entry[r, c] == EMPTY]
    if not vacancies or not unhappy:
        return entry

    # Stage 2: each unhappy agent proposes one vacancy (identity-indexed)
    proposals: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for r, c, group in unhappy:
        agent_id = r * w + c
        idx = int_range(0, len(vacancies), seed, t, EVT_PROPOSAL, agent_id, 0, 0)
        v = vacancies[idx]
        proposals.setdefault(v, []).append((r, c, group))

    # Stage 3: resolve conflicts with deterministic priority
    out = entry.copy()
    for v, claimants in proposals.items():
        if len(claimants) == 1:
            winner = claimants[0]
        else:
            vr, vc = v
            scored = []
            for r, c, group in claimants:
                agent_id = r * w + c
                prio = rng64(seed, t, EVT_PRIORITY, agent_id, vr * w + vc, 0)
                scored.append((prio, r, c, group))
            scored.sort()  # argmin priority
            winner = (scored[0][1], scored[0][2], scored[0][3])
        wr, wc, g = winner
        out[wr, wc] = EMPTY
        out[v] = g
    return out


def init_grid(
    side: int,
    seed: int,
    empty_ratio: float = 0.2,
    groups: int = 2,
) -> np.ndarray:
    n = side * side
    n_empty = int(round(n * empty_ratio))
    cells = np.ones(n, dtype=np.int8)
    # Alternate groups for non-empty; empty placed by counter draws
    for i in range(n):
        cells[i] = 1 + (i % groups)
    # Mark empties using deterministic permutation via priorities
    prios = [(rng64(seed, 0, EVT_PROPOSAL, i, 0, 0), i) for i in range(n)]
    prios.sort()
    for _, i in prios[:n_empty]:
        cells[i] = EMPTY
    return cells.reshape(side, side)


def run_schelling(
    side: int,
    steps: int,
    seed: int = 0,
    threshold: float = 0.5,
    empty_ratio: float = 0.2,
) -> np.ndarray:
    g = init_grid(side, seed, empty_ratio=empty_ratio)
    for t in range(steps):
        g = step_schelling(g, seed, t, threshold=threshold)
    return g


def step_schelling_sequential(
    grid: np.ndarray,
    seed: int,
    t: int,
    threshold: float = 0.5,
    order: np.ndarray | None = None,
) -> np.ndarray:
    """Sequential activation: each agent evaluates happiness on the *running*
    board and moves immediately, so later agents see earlier writes.

    Proposal draws remain identity-keyed; only the visibility of state differs
    from the three-stage snapshot rule.
    """
    h, w = grid.shape
    out = grid.copy()
    if order is None:
        order = np.arange(h * w)
    for agent_id in order:
        agent_id = int(agent_id)
        r, c = divmod(agent_id, w)
        if out[r, c] == EMPTY:
            continue
        if happiness(out, r, c, threshold):
            continue
        vacancies = [(rr, cc) for rr in range(h) for cc in range(w) if out[rr, cc] == EMPTY]
        if not vacancies:
            continue
        idx = int_range(0, len(vacancies), seed, t, EVT_PROPOSAL, agent_id, 0, 0)
        vr, vc = vacancies[idx]
        group = int(out[r, c])
        out[r, c] = EMPTY
        out[vr, vc] = group
    return out


def run_schelling_sequential(
    side: int,
    steps: int,
    seed: int = 0,
    threshold: float = 0.5,
    empty_ratio: float = 0.2,
    reshuffle: bool = True,
) -> np.ndarray:
    g = init_grid(side, seed, empty_ratio=empty_ratio)
    rng = np.random.default_rng(seed + 17_389)
    n = side * side
    order = np.arange(n)
    for t in range(steps):
        if reshuffle:
            order = rng.permutation(n)
        g = step_schelling_sequential(g, seed, t, threshold=threshold, order=order)
    return g


def segregation_index(grid: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of occupied agents that are happy at the given threshold."""
    h, w = grid.shape
    occ = 0
    happy = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] == EMPTY:
                continue
            occ += 1
            if happiness(grid, r, c, threshold):
                happy += 1
    return happy / occ if occ else 1.0


def mean_same_neighbor_frac(grid: np.ndarray) -> float:
    """Mean fraction of same-group neighbors among occupied agents with ≥1 neighbor."""
    h, w = grid.shape
    vals = []
    for r in range(h):
        for c in range(w):
            g = grid[r, c]
            if g == EMPTY:
                continue
            neigh = _neighbors(grid, r, c)
            occupied = [(rr, cc) for rr, cc in neigh if grid[rr, cc] != EMPTY]
            if not occupied:
                continue
            same = sum(1 for rr, cc in occupied if grid[rr, cc] == g)
            vals.append(same / len(occupied))
    return float(np.mean(vals)) if vals else 0.0


def cell_disagreement_frac(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of cells that differ (including empty)."""
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    return float(np.mean(a != b))
