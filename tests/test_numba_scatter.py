"""Numba-accelerated scatter / subset write paths (skip if numba missing)."""

import numpy as np
import pytest

import ambr as am
from ambr.performance import HAS_NUMBA, scatter_add_1d, scatter_write_1d


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
