"""Numba-accelerated scatter / subset write paths (skip if numba missing).

Also covers the high-level :func:`apply_scatter_*` wrappers and the shared
id→row index used by both the view API and the OOP flush path.
"""

import numpy as np
import pytest

import ambr as am
from ambr._id_index import ids_are_arange, resolve_positions
from ambr.performance import (
    HAS_NUMBA,
    apply_scatter_add,
    apply_scatter_write,
    scatter_add_1d,
    scatter_write_1d,
)


pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


def test_scatter_add_1d_matches_numpy_add_at():
    base = np.zeros(10, dtype=np.float64)
    pos = np.array([0, 0, 3, 3, 3], dtype=np.int64)
    delta = np.ones(5, dtype=np.float64)
    out = scatter_add_1d(base.copy(), pos, delta)
    ref = np.zeros(10, dtype=np.float64)
    np.add.at(ref, pos, delta)
    np.testing.assert_allclose(out, ref)


def test_scatter_write_1d():
    base = np.arange(5, dtype=np.float64)
    pos = np.array([1, 3], dtype=np.int64)
    vals = np.array([10.0, 30.0])
    out = scatter_write_1d(base.copy(), pos, vals)
    assert out[1] == 10.0 and out[3] == 30.0
    assert out[0] == 0.0


def test_apply_scatter_add_matches_add_at():
    """High-level wrapper (dtype upcast + Numba/fallback) matches np.add.at."""
    base = np.zeros(8, dtype=np.int64)
    pos = np.array([1, 1, 4], dtype=np.int64)
    delta = np.array([0.5, 0.5, 2.0])  # forces float upcast
    out = apply_scatter_add(base.copy(), pos, delta)
    ref = np.zeros(8, dtype=np.float64)
    np.add.at(ref, pos, delta)
    np.testing.assert_allclose(out, ref)


def test_apply_scatter_write_last_wins():
    base = np.arange(5, dtype=np.float64)
    pos = np.array([2, 2], dtype=np.int64)
    vals = np.array([7.0, 9.0])
    out = apply_scatter_write(base.copy(), pos, vals)
    assert out[2] == 9.0  # last write wins


def test_resolve_positions_arange_fast_path():
    class M(am.Model):
        def setup(self):
            self.add_agents(5, x=0)

        def step(self):
            pass

    m = M({"steps": 0, "show_progress": False})
    m.setup()
    df = m.agents_df
    ids = df["id"].to_numpy()
    assert ids_are_arange(m, ids)
    pos = resolve_positions(m, df, np.array([0, 3, 4]))
    np.testing.assert_array_equal(pos, [0, 3, 4])


def test_agents_scatter_add_uses_numba_path_correctly():
    class M(am.Model):
        def setup(self):
            self.add_agents(20, wealth=np.zeros(20, dtype=np.float64))

        def step(self):
            # Duplicates: agent 0 gets +3, agent 5 gets +1
            self.agents.at[[0, 0, 0, 5]].scatter_add(wealth=1.0)

    res = M({"steps": 1, "show_progress": False}).run()
    w = res.agents["wealth"].to_list()
    assert w[0] == 3.0
    assert w[5] == 1.0
    assert sum(w) == 4.0
