"""Coverage-focused tests for the environment topologies (Grid/Space/Network).

Exercises the node-id and agent-id branches of NetworkEnvironment, the
distance/move/space-position paths of SpaceEnvironment, and the position-tuple
neighbour/distance paths of GridEnvironment. The graph uses non-overlapping node
labels (10-14) so agent ids (0-4) take the agent-id branch.
"""

import math

import numpy as np
import pytest

import ambr as am
from ambr.environments import (
    GridEnvironment,
    NetworkEnvironment,
    Position,
    SpaceEnvironment,
)


def _model(n=5):
    m = am.Model({"seed": 1, "show_progress": False})
    m.add_agents(n)
    return m


# --------------------------------------------------------------------------- #
# NetworkEnvironment
# --------------------------------------------------------------------------- #

def _network():
    nx = pytest.importorskip("networkx")
    m = _model(5)
    g = nx.relabel_nodes(nx.cycle_graph(5), {i: i + 10 for i in range(5)})  # nodes 10..14
    net = NetworkEnvironment(m, g)
    for agent_id in range(5):
        net.move_agent(agent_id, Position((agent_id + 10,), "network"))  # agent i -> node i+10
    return m, net


def test_network_nodes_edges_add_remove():
    nx = pytest.importorskip("networkx")
    net = NetworkEnvironment(_model(0), nx.Graph())
    net.add_node(1)
    net.add_node(2)
    net.add_node(3)
    assert set(net.nodes) == {1, 2, 3}
    net.add_edge(1, 2)
    assert (1, 2) in net.edges or (2, 1) in net.edges
    net.remove_edge(1, 2)
    net.remove_node(3)
    assert 3 not in net.nodes


def test_network_node_id_branches():
    _, net = _network()
    # node-id branch (10..14 are graph nodes)
    assert set(net.get_neighbors(10)) == {11, 14}
    assert net.get_distance(10, 12) == 2
    assert net.get_degree(10) == 2
    assert net.get_clustering(10) == 0.0           # cycle has zero clustering
    assert net.get_clustering() == 0.0             # overall
    net.add_edge(10, 12)
    assert net.get_degree(10) == 3
    net.remove_edge(10, 12)
    assert net.get_degree(10) == 2


def test_network_agent_id_branches():
    _, net = _network()
    # agent-id branch (0..4 are not graph nodes)
    assert set(net.get_neighbors(0)) == {1, 4}     # agent 0 -> node 10 -> nodes 11,14 -> agents 1,4
    assert net.get_distance(0, 2) == 2
    assert net.get_degree(0) == 2
    assert net.get_clustering(0) == 0.0
    net.add_edge(0, 2)                              # edge between agents' nodes
    assert net.get_degree(0) == 3
    net.remove_edge(0, 2)
    assert net.get_degree(0) == 2


def test_network_distance_no_path_is_inf():
    nx = pytest.importorskip("networkx")
    m = _model(2)
    g = nx.Graph()
    g.add_node(10)
    g.add_node(11)  # disconnected
    net = NetworkEnvironment(m, g)
    net.move_agent(0, Position((10,), "network"))
    net.move_agent(1, Position((11,), "network"))
    assert net.get_distance(0, 1) == float("inf")
    assert net.get_distance(10, 11) == float("inf")


def test_network_random_node_and_empty():
    _, net = _network()
    assert net.random_node() in net.nodes
    nx = pytest.importorskip("networkx")
    empty = NetworkEnvironment(_model(0), nx.Graph())
    assert empty.random_node() is None


def test_network_move_agent_errors():
    _, net = _network()
    with pytest.raises(ValueError):
        net.move_agent(0, Position((10,), "grid"))        # wrong topology
    with pytest.raises(ValueError):
        net.move_agent(0, Position((999,), "network"))    # node not in graph


def test_network_edge_errors_on_missing_agents():
    _, net = _network()
    with pytest.raises(ValueError):
        net.add_edge(0, 99)                                # agent 99 doesn't exist
    with pytest.raises(ValueError):
        net.remove_edge(0, 99)


# --------------------------------------------------------------------------- #
# SpaceEnvironment
# --------------------------------------------------------------------------- #

def test_space_get_distance_positions_and_agents():
    m = _model(3)
    space = SpaceEnvironment(m, bounds=[(0, 100), (0, 100)])
    space.set_position([0, 1], [(0.0, 0.0), (3.0, 4.0)])
    assert math.isclose(space.get_distance((0.0, 0.0), (3.0, 4.0)), 5.0)   # position form
    assert math.isclose(space.get_distance(0, 1), 5.0)                     # agent form
    assert space.get_distance(0, 99) == float("inf")                       # missing agent


def test_space_move_agent_and_errors():
    m = _model(2)
    space = SpaceEnvironment(m, bounds=[(0, 10), (0, 10)])
    space.move_agent(0, Position((5.0, 5.0), "space"))
    ids, pos = space.positions_array()
    assert pos[ids.tolist().index(0)].tolist() == [5.0, 5.0]
    with pytest.raises(ValueError):
        space.move_agent(0, Position((5.0, 5.0), "grid"))        # wrong topology
    with pytest.raises(ValueError):
        space.move_agent(0, Position((5.0,), "space"))           # wrong dimensions
    with pytest.raises(ValueError):
        space.move_agent(0, Position((50.0, 50.0), "space"))     # out of bounds (non-torus)


def test_space_torus_distance_and_move():
    m = _model(2)
    space = SpaceEnvironment(m, bounds=[(0, 10), (0, 10)], torus=True)
    # torus distance wraps: |1-9|=8 but 10-8=2
    assert math.isclose(space._calculate_distance((1.0, 1.0), (9.0, 1.0)), 2.0)
    space.move_agent(0, Position((12.0, 3.0), "space"))          # wraps to (2, 3)
    ids, pos = space.positions_array()
    assert math.isclose(pos[ids.tolist().index(0)][0], 2.0)


def test_space_is_valid_position_wrong_dims():
    space = SpaceEnvironment(_model(1), bounds=[(0, 10), (0, 10)])
    assert space.is_valid_position((5.0,)) is False
    assert space.is_valid_position((5.0, 5.0)) is True


# --------------------------------------------------------------------------- #
# GridEnvironment (position-tuple branches)
# --------------------------------------------------------------------------- #

def test_grid_positions_2d_and_nd():
    grid2 = GridEnvironment(_model(1), size=(2, 3))
    assert len(grid2.positions) == 6
    grid3 = GridEnvironment(_model(1), size=(2, 2, 2))
    assert len(grid3.positions) == 8


def test_grid_neighbors_orthogonal_diagonal_distance():
    grid = GridEnvironment(_model(1), size=(10, 10))
    orth = grid.get_neighbors((5, 5))
    assert set(orth) == {(4, 5), (6, 5), (5, 4), (5, 6)}
    diag = grid.get_neighbors((5, 5), include_diagonal=True)
    assert len(diag) == 8
    far = grid.get_neighbors((5, 5), include_diagonal=True, distance=2)
    assert len(far) == 24
    # boundary clamps (no wrap)
    corner = grid.get_neighbors((0, 0))
    assert set(corner) == {(1, 0), (0, 1)}


def test_grid_neighbors_torus_wrap():
    grid = GridEnvironment(_model(1), size=(5, 5), torus=True)
    nb = grid.get_neighbors((0, 0))
    assert (4, 0) in nb and (0, 4) in nb           # wrapped neighbours


def test_grid_neighbors_3d():
    grid = GridEnvironment(_model(1), size=(5, 5, 5))
    nb = grid.get_neighbors((2, 2, 2), include_diagonal=True)
    assert len(nb) == 26


def test_grid_distance_manhattan_and_torus():
    grid = GridEnvironment(_model(1), size=(10, 10))
    assert grid.get_distance((0, 0), (3, 4)) == 7
    tgrid = GridEnvironment(_model(1), size=(10, 10), torus=True)
    assert tgrid.get_distance((0, 0), (9, 0)) == 1          # wrap
    assert grid.get_distance(0, 99) == float("inf")         # missing agent / no positions


def test_grid_is_valid_and_empty_positions():
    grid = GridEnvironment(_model(0), size=(3, 3))
    assert grid.is_valid_position((1, 1)) is True
    assert grid.is_valid_position((3, 0)) is False          # out of bounds
    assert grid.is_valid_position((1,)) is False            # wrong dims
    assert len(grid.empty_positions()) == 9                 # nothing placed
