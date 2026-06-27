"""Phase 1: one canonical RNG, and seeded code never touches global numpy RNG.

These pin the Tier-C leak fixes: GridEnvironment/SpaceEnvironment.random_position
and NetworkEnvironment.random_node must route through the model's ``rng`` rather
than the process-global ``np.random``. (ParameterSpace.sample's explicit-rng
leak test lives with the optimization changes in test_optimization.py.)
"""

import numpy as np
import pytest

import ambr as am
from ambr import (
    GridEnvironment,
    SpaceEnvironment,
    NetworkEnvironment,
)


def _assert_no_global_leak(fn):
    before = np.random.get_state()
    fn()
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]), "global np.random state was modified"


def test_model_rng_is_canonical_generator():
    m = am.Model({"seed": 7})
    assert isinstance(m.rng, np.random.Generator)
    assert isinstance(m.random.random(), float)
    # legacy nprandom alias still works, incl. the old-style randint shim
    assert 0 <= int(m.nprandom.randint(0, 5)) < 5


def test_grid_random_position_no_global_leak():
    m = am.Model({"seed": 1})
    grid = GridEnvironment(m, size=(10, 10))
    _assert_no_global_leak(lambda: [grid.random_position() for _ in range(50)])


def test_space_random_position_no_global_leak():
    m = am.Model({"seed": 1})
    space = SpaceEnvironment(m, bounds=[(0, 10), (0, 10)])
    _assert_no_global_leak(lambda: [space.random_position() for _ in range(50)])


def test_network_random_node_no_global_leak():
    nx = pytest.importorskip("networkx")
    m = am.Model({"seed": 1})
    net = NetworkEnvironment(m, nx.cycle_graph(8))
    _assert_no_global_leak(lambda: [net.random_node() for _ in range(50)])
