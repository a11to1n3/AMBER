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


def test_agent_draw_keys_are_stable_and_leave_shared_streams_untouched():
    model = am.Model({"seed": 7})
    model.add_agents(2, agent_class=am.Agent)
    a, b = model.agents
    numpy_state = model.rng.bit_generator.state
    python_state = model.random.getstate()
    first = a.draw("move")
    assert first.hex() == "0x1.3b35ccc19dcdap-1"
    assert a.draw("move", index=np.int64(1)).hex() == "0x1.185f8a4709f5fp-1"
    assert b.draw("move").hex() == "0x1.edc0e883b4b9cp-2"
    assert a.draw("κατεύθυνση").hex() == "0x1.9acad44b1ca9dp-1"
    assert a.draw("move") == first
    model.t = 1
    assert a.draw("move").hex() == "0x1.77df80fac593cp-2"
    assert model.rng.bit_generator.state == numpy_state
    assert model.random.getstate() == python_state

    with pytest.raises(ValueError, match="non-empty string"):
        a.draw("")
    with pytest.raises(ValueError, match="non-negative"):
        a.draw("move", index=-1)


def test_existing_draw_column_remains_accessible_on_agents():
    model = am.Model({"seed": 7})
    model.add_agents(2, agent_class=am.Agent, draw=[11, 7])
    a, b = model.agents
    assert (a.draw, b.draw) == (11, 7)
    a.draw += 2
    assert model.agents.draw.to_list() == [13, 7]


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
