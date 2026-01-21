"""
Tests for ambr.environments module.
"""

import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch
import ambr as am
from ambr.environments import GridEnvironment, SpaceEnvironment, NetworkEnvironment, Position
import polars as pl


class TestGridEnvironment:
    """Test cases for GridEnvironment class."""
    
    def test_grid_initialization_square(self):
        """Test GridEnvironment initialization with square grid."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()  # Use real DataFrame
        
        grid = GridEnvironment(mock_model, size=10)
        
        assert grid.model == mock_model
        assert grid.size == (10, 10)
        assert grid.width == 10
        assert grid.height == 10
        assert grid.torus is False
    
    def test_grid_initialization_rectangular(self):
        """Test GridEnvironment initialization with rectangular grid."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        grid = GridEnvironment(mock_model, size=(15, 20))
        
        assert grid.size == (15, 20)
        assert grid.width == 15
        assert grid.height == 20
    
    def test_grid_initialization_with_torus(self):
        """Test GridEnvironment initialization with torus topology."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        grid = GridEnvironment(mock_model, size=5, torus=True)
        
        assert grid.torus is True
        assert grid.size == (5, 5)
    
    def test_grid_positions_property(self):
        """Test accessing grid positions."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        grid = GridEnvironment(mock_model, size=(3, 4))
        
        # Grid should have 3*4 = 12 positions
        positions = grid.positions
        assert len(positions) == 12
        
        # Check some specific positions
        assert (0, 0) in positions
        assert (2, 3) in positions
        assert (3, 4) not in positions  # Out of bounds
    
    def test_grid_get_neighbors_no_torus(self):
        """Test getting neighbors without torus topology."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        grid = GridEnvironment(mock_model, size=(5, 5), torus=False)
        
        # Test corner position
        neighbors = grid.get_neighbors((0, 0))
        expected = [(0, 1), (1, 0)]
        assert set(neighbors) == set(expected)
        
        # Test middle position
        neighbors = grid.get_neighbors((2, 2))
        expected = [(1, 2), (3, 2), (2, 1), (2, 3)]
        assert set(neighbors) == set(expected)
        
        # Test edge position
        neighbors = grid.get_neighbors((0, 2))
        expected = [(0, 1), (0, 3), (1, 2)]
        assert set(neighbors) == set(expected)
    
    def test_grid_get_neighbors_with_torus(self):
        """Test getting neighbors with torus topology."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        grid = GridEnvironment(mock_model, size=(3, 3), torus=True)
        
        # Test corner position - should wrap around
        neighbors = grid.get_neighbors((0, 0))
        expected = [(0, 1), (0, 2), (1, 0), (2, 0)]
        assert set(neighbors) == set(expected)
        
        # Test opposite corner
        neighbors = grid.get_neighbors((2, 2))
        expected = [(2, 1), (2, 0), (1, 2), (0, 2)]
        assert set(neighbors) == set(expected)
    
    def test_grid_get_neighbors_diagonal(self):
        """Test getting neighbors including diagonals."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        grid = GridEnvironment(mock_model, size=(5, 5), torus=False)
        
        # Test with diagonal neighbors
        neighbors = grid.get_neighbors((2, 2), include_diagonal=True)
        expected = [
            (1, 1), (1, 2), (1, 3),
            (2, 1),         (2, 3),
            (3, 1), (3, 2), (3, 3)
        ]
        assert set(neighbors) == set(expected)
        
        # Test corner with diagonals
        neighbors = grid.get_neighbors((0, 0), include_diagonal=True)
        expected = [(0, 1), (1, 0), (1, 1)]
        assert set(neighbors) == set(expected)
    
    def test_grid_get_neighbors_distance(self):
        """Test getting neighbors at different distances."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        grid = GridEnvironment(mock_model, size=(5, 5), torus=False)
        
        # Test distance 2
        neighbors = grid.get_neighbors((2, 2), distance=2)
        # Should include positions at Manhattan distance 1 and 2
        assert (1, 2) in neighbors  # Distance 1
        assert (0, 2) in neighbors  # Distance 2
        assert (2, 0) in neighbors  # Distance 2
        
        # Test that it doesn't include positions too far
        assert len([n for n in neighbors if abs(n[0] - 2) + abs(n[1] - 2) <= 2]) == len(neighbors)
    
    def test_grid_get_distance(self):
        """Test calculating distance between positions."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        grid = GridEnvironment(mock_model, size=(10, 10), torus=False)
        
        # Test Manhattan distance
        distance = grid.get_distance((0, 0), (3, 4))
        assert distance == 7  # |3-0| + |4-0|
        
        # Test same position
        distance = grid.get_distance((2, 2), (2, 2))
        assert distance == 0
    
    def test_grid_get_distance_torus(self):
        """Test calculating distance with torus topology."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        grid = GridEnvironment(mock_model, size=(5, 5), torus=True)
        
        # Test wrapping distance
        distance = grid.get_distance((0, 0), (4, 4))
        # With torus: min(4, 5-4) + min(4, 5-4) = 1 + 1 = 2
        assert distance == 2
        
        # Test normal distance
        distance = grid.get_distance((1, 1), (2, 2))
        assert distance == 2
    
    def test_grid_is_valid_position(self):
        """Test checking if positions are valid."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        grid = GridEnvironment(mock_model, size=(3, 4))
        
        # Valid positions
        assert grid.is_valid_position((0, 0)) is True
        assert grid.is_valid_position((2, 3)) is True
        
        # Invalid positions
        assert grid.is_valid_position((3, 0)) is False  # x out of bounds
        assert grid.is_valid_position((0, 4)) is False  # y out of bounds
        assert grid.is_valid_position((-1, 0)) is False  # negative
    
    def test_grid_random_position(self):
        """Test getting random position."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        mock_model.nprandom = np.random.RandomState(42)
        
        grid = GridEnvironment(mock_model, size=(3, 3))
        
        # Get multiple random positions
        positions = [grid.random_position() for _ in range(10)]
        
        # All should be valid
        for pos in positions:
            assert grid.is_valid_position(pos)
        
        # Should have some variety (not all the same)
        assert len(set(positions)) > 1
    
    def test_grid_empty_positions(self):
        """Test getting empty positions."""
        mock_model = Mock()
        # Use real DataFrame with some occupied positions and explicit Object type for tuples
        mock_model.agents_df = pl.DataFrame({
            'id': [1, 2],
            'grid_position': pl.Series([(0, 0), (1, 1)], dtype=pl.Object)
        })
        
        grid = GridEnvironment(mock_model, size=(3, 3))
        
        empty = grid.empty_positions()
        
        # Should not include occupied positions
        assert (0, 0) not in empty
        assert (1, 1) not in empty
        
        # Should include other positions
        assert (0, 1) in empty
        assert (2, 2) in empty
    
    def test_grid_move_agent(self):
        """Test moving an agent in the grid."""
        mock_model = Mock()
        # Use real DataFrame
        mock_model.agents_df = pl.DataFrame({
            'id': [1],
            'grid_position': pl.Series([(0, 0)], dtype=pl.Object)
        })
        
        grid = GridEnvironment(mock_model, size=(5, 5))
        
        # Verify initial position
        
        # Move agent
        new_pos = Position((2, 3), 'grid')
        grid.move_agent(1, new_pos)
        
        # Verify new position
        # Use simple equality check
        current_pos = grid.df.filter(pl.col('id') == 1)['grid_position'].to_list()[0]
        # Handle case where Polars converts tuple to list
        if isinstance(current_pos, list):
            current_pos = tuple(current_pos)
        assert current_pos == (2, 3)
        
        # Test moving with wrapping
        grid.wrap = True
        wrap_pos = Position((5, 5), 'grid') # Should wrap to (0, 0)
        grid.move_agent(1, wrap_pos)
        current_pos_wrap = grid.df.filter(pl.col('id') == 1)['grid_position'].to_list()[0]
        if isinstance(current_pos_wrap, list):
            current_pos_wrap = tuple(current_pos_wrap)
        assert current_pos_wrap == (0, 0)



class TestSpaceEnvironment:
    """Test cases for SpaceEnvironment class."""
    
    def test_space_initialization_2d(self):
        """Test SpaceEnvironment initialization in 2D."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        space = SpaceEnvironment(mock_model, bounds=[(0, 10), (0, 20)])
        
        assert space.model == mock_model
        assert space.dimensions == 2
        assert space.bounds == [(0, 10), (0, 20)]
        assert space.torus is False
    
    def test_space_initialization_3d(self):
        """Test SpaceEnvironment initialization in 3D."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        space = SpaceEnvironment(mock_model, bounds=[(0, 5), (0, 10), (0, 15)], torus=True)
        
        assert space.dimensions == 3
        assert space.bounds == [(0, 5), (0, 10), (0, 15)]
        assert space.torus is True
    
    def test_space_get_neighbors(self):
        """Test getting neighbors within radius."""
        mock_model = Mock()
        # Create a proper DataFrame with agent positions
        mock_model.agents_df = pl.DataFrame({
            'id': [1, 2, 3],
            'step': [0, 0, 0],
            'wealth': [0, 0, 0]
        })
        
        space = SpaceEnvironment(mock_model, bounds=[(0, 20), (0, 20)])
        
        # Manually set the space positions in the environment's DataFrame
        # Manually set the space positions in the environment's DataFrame with Object type
        space.df = space.df.with_columns([
            pl.when(pl.col('id') == 1).then(pl.lit((1.0, 1.0), dtype=pl.Object))
            .when(pl.col('id') == 2).then(pl.lit((3.0, 3.0), dtype=pl.Object))
            .when(pl.col('id') == 3).then(pl.lit((10.0, 10.0), dtype=pl.Object))
            .otherwise(pl.col('space_position'))
            .alias('space_position')
        ])
        
        neighbors = space.get_neighbors([2.0, 2.0], radius=2.0)
        
        # Should include agents within radius
        # Distance from (2,2) to (1,1) = sqrt(2) ≈ 1.41 < 2.0 ✓
        # Distance from (2,2) to (3,3) = sqrt(2) ≈ 1.41 < 2.0 ✓  
        # Distance from (2,2) to (10,10) = sqrt(128) ≈ 11.31 > 2.0 ✗
        
        assert len(neighbors) == 2
        assert 1 in neighbors
        assert 2 in neighbors
        assert 3 not in neighbors
    
    def test_space_get_distance(self):
        """Test calculating Euclidean distance."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        space = SpaceEnvironment(mock_model, bounds=[(0, 10), (0, 10)])
        
        # Test 2D distance
        distance = space.get_distance([0, 0], [3, 4])
        assert distance == 5.0  # 3-4-5 triangle
        
        # Test same position
        distance = space.get_distance([1, 1], [1, 1])
        assert distance == 0.0
    
    def test_space_get_distance_3d(self):
        """Test calculating distance in 3D."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        space = SpaceEnvironment(mock_model, bounds=[(0, 10), (0, 10), (0, 10)])
        
        distance = space.get_distance([0, 0, 0], [2, 3, 6])
        expected = np.sqrt(2**2 + 3**2 + 6**2)  # sqrt(4 + 9 + 36) = sqrt(49) = 7
        assert abs(distance - expected) < 1e-10
    
    def test_space_get_distance_torus(self):
        """Test calculating distance with torus topology."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        space = SpaceEnvironment(mock_model, bounds=[(0, 10), (0, 10)], torus=True)
        
        # Test wrapping distance
        distance = space.get_distance([1, 1], [9, 9])
        # Regular distance would be sqrt((9-1)^2 + (9-1)^2) = sqrt(128) ≈ 11.31
        # Torus distance should be sqrt(2^2 + 2^2) = sqrt(8) ≈ 2.83 (wrapping)
        expected = np.sqrt(2**2 + 2**2)
        assert abs(distance - expected) < 1e-10
    
    def test_space_is_valid_position(self):
        """Test checking if positions are within bounds."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        space = SpaceEnvironment(mock_model, bounds=[(0, 10), (5, 15)])
        
        # Valid positions
        assert space.is_valid_position([5, 10]) is True
        assert space.is_valid_position([0, 5]) is True
        assert space.is_valid_position([10, 15]) is True
        
        # Invalid positions
        assert space.is_valid_position([-1, 10]) is False  # x out of bounds
        assert space.is_valid_position([5, 4]) is False   # y out of bounds
        assert space.is_valid_position([11, 10]) is False # x out of bounds
    
    def test_space_random_position(self):
        """Test getting random position."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        mock_model.nprandom = np.random.RandomState(42)
        
        space = SpaceEnvironment(mock_model, bounds=[(0, 10), (5, 15)])
        
        # Get multiple random positions
        positions = [space.random_position() for _ in range(10)]
        
        # All should be valid
        for pos in positions:
            assert space.is_valid_position(pos)
            assert 0 <= pos[0] <= 10
            assert 5 <= pos[1] <= 15
        
        # Should have some variety
        assert len(set(tuple(p) for p in positions)) > 1

    def test_space_move_agent(self):
        """Test moving an agent in continuous space."""
        mock_model = Mock()
        # Use real DataFrame
        mock_model.agents_df = pl.DataFrame({
            'id': [1],
            'space_position': [(1.0, 1.0)],
            'space_distance': [0.0]
        })
        
        space = SpaceEnvironment(mock_model, bounds=[(0, 10), (0, 10)])
        
        # Verify initial position
        # Verify initial position
        current_pos = space.df.filter(pl.col('id') == 1)['space_position'].to_list()[0]
        if isinstance(current_pos, list):
            current_pos = tuple(current_pos)
        assert current_pos == (1.0, 1.0)
        
        # Move agent
        new_pos = Position((5.5, 6.5), 'space')
        space.move_agent(1, new_pos)
        
        # Verify new position
        current_pos = space.df.filter(pl.col('id') == 1)['space_position'].to_list()[0]
        if isinstance(current_pos, list):
            current_pos = tuple(current_pos)
        assert current_pos == (5.5, 6.5)
        
        # Test moving with wrapping (torus)
        space.torus = True
        wrap_pos = Position((11.0, 12.0), 'space') # Should wrap to (1.0, 2.0)
        space.move_agent(1, wrap_pos)
        
        pos = space.df.filter(pl.col('id') == 1)['space_position'].to_list()[0]
        if isinstance(pos, list):
            pos = tuple(pos)
        assert abs(pos[0] - 1.0) < 1e-10
        assert abs(pos[1] - 2.0) < 1e-10


class TestNetworkEnvironment:
    """Test cases for NetworkEnvironment class."""
    
    def test_network_initialization_empty(self):
        """Test NetworkEnvironment initialization with empty graph."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        G = nx.Graph()
        network = NetworkEnvironment(mock_model, G)
        
        assert network.model == mock_model
        assert network.graph == G
        assert network.nodes == []
        assert network.edges == []
    
    def test_network_initialization_with_graph(self, mock_networkx_graph):
        """Test NetworkEnvironment initialization with existing graph."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        network = NetworkEnvironment(mock_model, mock_networkx_graph)
        
        assert network.graph == mock_networkx_graph
        assert len(network.nodes) == 5
        assert len(network.edges) == 3
        assert 0 in network.nodes
        assert (0, 1) in network.edges or (1, 0) in network.edges
    
    def test_network_get_neighbors(self, mock_networkx_graph):
        """Test getting neighbors of a node."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        network = NetworkEnvironment(mock_model, mock_networkx_graph)
        
        # Node 1 should be connected to nodes 0 and 2
        neighbors = network.get_neighbors(1)
        assert set(neighbors) == {0, 2}
        
        # Node 4 should have no neighbors (isolated)
        neighbors = network.get_neighbors(4)
        assert neighbors == []
    
    def test_network_get_distance(self, mock_networkx_graph):
        """Test calculating shortest path distance."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        network = NetworkEnvironment(mock_model, mock_networkx_graph)
        
        # Direct connection
        distance = network.get_distance(0, 1)
        assert distance == 1
        
        # Two-step path
        distance = network.get_distance(0, 2)
        assert distance == 2  # 0 -> 1 -> 2
        
        # Path to connected component  
        distance = network.get_distance(1, 3)
        assert distance == 2  # 1 -> 2 -> 3
        
        # No path (isolated node)
        distance = network.get_distance(0, 4)
        assert distance == float('inf')
    
    def test_network_add_node(self):
        """Test adding nodes to network."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        G = nx.Graph()
        network = NetworkEnvironment(mock_model, G)
        
        # Add node
        network.add_node(1, color='red')
        
        assert 1 in network.nodes
        assert network.graph.nodes[1]['color'] == 'red'
    
    def test_network_add_edge(self):
        """Test adding edges to network."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        network = NetworkEnvironment(mock_model, G)
        
        # Add edge
        network.add_edge(1, 2, weight=0.5)
        
        assert (1, 2) in network.edges or (2, 1) in network.edges
        assert network.graph[1][2]['weight'] == 0.5
    
    def test_network_remove_node(self, mock_networkx_graph):
        """Test removing nodes from network."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        network = NetworkEnvironment(mock_model, mock_networkx_graph)
        
        initial_nodes = len(network.nodes)
        network.remove_node(1)
        
        assert 1 not in network.nodes
        assert len(network.nodes) == initial_nodes - 1
        
        # Edges involving removed node should also be gone
        neighbors = network.get_neighbors(0)
        assert 1 not in neighbors
    
    def test_network_remove_edge(self, mock_networkx_graph):
        """Test removing edges from network."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        network = NetworkEnvironment(mock_model, mock_networkx_graph)
        
        # Remove edge
        network.remove_edge(0, 1)
        
        # Nodes should still exist
        assert 0 in network.nodes
        assert 1 in network.nodes
        
        # But edge should be gone
        neighbors_0 = network.get_neighbors(0)
        neighbors_1 = network.get_neighbors(1)
        assert 1 not in neighbors_0
        assert 0 not in neighbors_1
    
    def test_network_get_degree(self, mock_networkx_graph):
        """Test getting node degree."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        network = NetworkEnvironment(mock_model, mock_networkx_graph)
        
        # Node 1 has connections to 0 and 2
        degree = network.get_degree(1)
        assert degree == 2
        
        # Node 4 is isolated
        degree = network.get_degree(4)
        assert degree == 0
    
    def test_network_get_clustering(self, mock_networkx_graph):
        """Test getting clustering coefficient."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        
        network = NetworkEnvironment(mock_model, mock_networkx_graph)
        
        # Test clustering coefficient
        clustering = network.get_clustering()
        assert isinstance(clustering, float)
        assert 0 <= clustering <= 1
    
    def test_network_random_node(self, mock_networkx_graph):
        """Test getting random node."""
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        mock_model.nprandom = np.random.RandomState(42)
        
        network = NetworkEnvironment(mock_model, mock_networkx_graph)
        
        # Get multiple random nodes
        nodes = [network.random_node() for _ in range(10)]
        
        # All should be valid nodes
        for node in nodes:
            assert node in network.nodes
        
        # Should have some variety
        assert len(set(nodes)) > 1

    def test_network_move_agent(self, mock_networkx_graph):
        """Test moving an agent in the network."""
        mock_model = Mock()
        # Use real DataFrame
        mock_model.agents_df = pl.DataFrame({
            'id': [1],
            'node_id': [0],
            'network_distance': [0.0]
        })
        
        network = NetworkEnvironment(mock_model, mock_networkx_graph)
        
        # Verify initial position
        assert network.df.filter(pl.col('id') == 1)['node_id'].item() == 0
        
        # Move agent
        new_pos = Position((2,), 'network')
        network.move_agent(1, new_pos)
        
        # Verify new position
        current_node = network.df.filter(pl.col('id') == 1)['node_id'].to_list()[0]
        assert current_node == 2
        
        # Test invalid move (node not in graph)
        invalid_pos = Position((99,), 'network')
        with pytest.raises(ValueError):
            network.move_agent(1, invalid_pos)


class TestEnvironmentIntegration:
    """Integration tests for environment classes."""
    
    def test_grid_with_real_model(self, basic_model):
        """Test GridEnvironment with real model."""
        # Mock the agents_df attribute
        basic_model.agents_df = pl.DataFrame({'id': [], 'grid_position': []})
        
        grid = GridEnvironment(basic_model, size=(5, 5))
        
        assert grid.model == basic_model
        assert len(grid.positions) == 25
        
        # Test integration with model's random number generator
        pos = grid.random_position()
        assert grid.is_valid_position(pos)
    
    def test_space_with_real_model(self, basic_model):
        """Test SpaceEnvironment with real model."""
        basic_model.agents_df = pl.DataFrame({'id': [], 'space_position': []})
        
        space = SpaceEnvironment(basic_model, bounds=[(0, 10), (0, 10)])
        
        assert space.model == basic_model
        
        # Test integration with model's random number generator
        pos = space.random_position()
        assert space.is_valid_position(pos)
    
    def test_network_with_real_model(self, basic_model, mock_networkx_graph):
        """Test NetworkEnvironment with real model."""
        basic_model.agents_df = pl.DataFrame({'id': [], 'node_id': []})
        
        network = NetworkEnvironment(basic_model, mock_networkx_graph)
        
        assert network.model == basic_model
        assert len(network.nodes) > 0
        
        # Test network operations
        if len(network.nodes) > 0:
            node = network.random_node()
            neighbors = network.get_neighbors(node)
            assert isinstance(neighbors, list) 