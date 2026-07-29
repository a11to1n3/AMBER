"""Shared grid-vectorized Schelling segregation step (NumPy or CuPy).

Agents live on a G x G toroidal grid; a cell holds the agent's type (1 or 2),
0 = empty. Each step:
  * count same-type and total occupied neighbours (8-cell Moore, via rolls),
  * an agent is happy if it has no neighbours or its same-type fraction is
    >= tolerance,
  * unhappy agents relocate to random *distinct* empty cells.

The same code runs on the CPU (xp = numpy) or GPU (xp = cupy) so the backends
are compared on an identical algorithm.
"""

import math


def _perm(rng, m, xp):
    """A random permutation of range(m), backend-agnostic (CuPy's Generator has
    no .permutation, so we argsort random keys -- works for NumPy and CuPy)."""
    return xp.argsort(rng.random(m))


def schelling_setup(n, density, fraction_a, rng, xp):
    """Place n agents (types 1/2) on distinct random cells of a G x G grid."""
    G = int(math.ceil(math.sqrt(n / density)))
    perm = _perm(rng, G * G, xp)[:n]
    x = (perm % G).astype(xp.int32)
    y = (perm // G).astype(xp.int32)
    n_a = int(n * fraction_a)
    t = xp.where(xp.arange(n) < n_a, xp.int8(1), xp.int8(2))
    t = t[_perm(rng, n, xp)]
    return x, y, t, G


def _neighbour_counts(mask, xp):
    c = xp.zeros(mask.shape, dtype=xp.int32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            c += xp.roll(xp.roll(mask, dy, axis=0), dx, axis=1).astype(xp.int32)
    return c


def schelling_step(x, y, t, G, tolerance, rng, xp):
    """One Schelling step. Returns new (x, y) arrays (inputs are not mutated)."""
    x = x.copy()
    y = y.copy()
    gt = xp.zeros((G, G), dtype=xp.int8)
    gt[y, x] = t
    ca = _neighbour_counts(gt == 1, xp)
    cb = _neighbour_counts(gt == 2, xp)
    same = xp.where(t == 1, ca[y, x], cb[y, x])
    total = ca[y, x] + cb[y, x]
    happy = (total == 0) | (same.astype(xp.float32) >= tolerance * total)

    unhappy = xp.nonzero(~happy)[0]
    n_unhappy = int(unhappy.size)
    if n_unhappy:
        empty = xp.nonzero(gt.reshape(-1) == 0)[0]
        n_empty = int(empty.size)
        k = min(n_unhappy, n_empty)
        if k:
            dest = empty[_perm(rng, n_empty, xp)[:k]]
            mv = unhappy[_perm(rng, n_unhappy, xp)[:k]]
            x[mv] = (dest % G).astype(xp.int32)
            y[mv] = (dest // G).astype(xp.int32)
    return x, y


def happy_fraction(x, y, t, G, tolerance, xp):
    """Fraction of agents that are happy (for correctness checks)."""
    gt = xp.zeros((G, G), dtype=xp.int8)
    gt[y, x] = t
    ca = _neighbour_counts(gt == 1, xp)
    cb = _neighbour_counts(gt == 2, xp)
    same = xp.where(t == 1, ca[y, x], cb[y, x])
    total = ca[y, x] + cb[y, x]
    happy = (total == 0) | (same.astype(xp.float32) >= tolerance * total)
    return float(happy.mean())
