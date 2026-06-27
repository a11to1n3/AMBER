"""Phase 6: vectorized SpaceEnvironment neighbours, position accessors, wrap->torus."""

import warnings

import numpy as np

import ambr as am
from ambr import GridEnvironment, SpaceEnvironment


def _space(coords, bounds=((0, 100), (0, 100)), torus=False):
    m = am.Model({"seed": 1, "show_progress": False})
    m.add_agents(len(coords))
    env = SpaceEnvironment(m, bounds=list(bounds), torus=torus)
    env.set_position(np.arange(len(coords)), np.asarray(coords, dtype=float))
    return m, env


def test_get_neighbors_within_radius_includes_self():
    m, env = _space([(0, 0), (1, 0), (5, 0), (50, 50)])
    assert set(env.get_neighbors(0, radius=2.0)) == {0, 1}
    assert set(env.get_neighbors((0.0, 0.0), radius=6.0)) == {0, 1, 2}


def test_positions_array_roundtrip():
    m, env = _space([(1.0, 2.0), (3.0, 4.0)])
    ids, pos = env.positions_array()
    assert ids.tolist() == [0, 1]
    assert pos.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_torus_distance_wraps():
    coords = [(1.0, 1.0), (99.0, 1.0)]  # gap 2 across the wrap, 98 direct
    _, env_t = _space(coords, torus=True)
    assert set(env_t.get_neighbors(0, radius=3.0)) == {0, 1}
    _, env_f = _space(coords, torus=False)
    assert set(env_f.get_neighbors(0, radius=3.0)) == {0}


def test_kdtree_path_matches_bruteforce_large_n():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(5000, 2))  # >= _KDTREE_MIN_AGENTS
    _, env = _space(coords)
    got = set(env.get_neighbors(0, radius=5.0))
    d = np.sqrt(((coords - coords[0]) ** 2).sum(1))
    assert got == set(np.where(d <= 5.0)[0].tolist())


def test_set_position_bulk_then_move_one():
    m, env = _space([(0, 0), (0, 0), (0, 0)])
    env.set_position([1, 2], [(10.0, 0.0), (20.0, 0.0)])
    ids, pos = env.positions_array()
    assert pos[ids.tolist().index(1)].tolist() == [10.0, 0.0]
    assert pos[ids.tolist().index(2)].tolist() == [20.0, 0.0]


def test_wrap_param_and_property_deprecated():
    m = am.Model({"seed": 1, "show_progress": False})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        env = GridEnvironment(m, size=(10, 10), wrap=True)
        assert env.torus is True
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert env.wrap is True  # property read returns torus...
        assert any(issubclass(x.category, DeprecationWarning) for x in w)  # ...and warns
