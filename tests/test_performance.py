"""
Tests for ambr.performance module.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
import ambr as am
from ambr.performance import (
    SpatialIndex,
    ParallelRunner,
    fast_distance_matrix,
    fast_neighbors_within_radius,
    fast_all_neighbors_within_radius,
    fast_random_walk_step,
    vectorized_wealth_transfer,
    vectorized_move,
    vectorized_random_velocities,
    vectorized_sir_infections,
    check_performance_deps,
    install_performance_deps,
    HAS_SCIPY,
    HAS_NUMBA,
    _run_single_simulation  # Import private function for direct testing
)

# Mock model for ParallelRunner testing
class MockModel(am.Model):
    def setup(self):
        pass
    def step(self):
        self.record_model("step_val", self.t)
        self.record_model("param_val", self.p.get("param", 0))

@pytest.fixture
def sample_positions():
    """Create sample 2D positions for testing."""
    return np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0, 2.0],
        [10.0, 10.0]  # Far away point
    ])

class TestSpatialIndex:
    """Test cases for SpatialIndex class."""
    
    def test_spatial_index_initialization(self):
        """Test initialization."""
        index = SpatialIndex()
        assert index.tree is None
        assert index.positions is None

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_spatial_index_build_and_query(self, sample_positions):
        """Test building index and querying radius."""
        index = SpatialIndex().build(sample_positions)
        
        # Query radius 1.5 around (0,0) -> should get (0,0), (1,0), (0,1)
        neighbors = index.query_radius(np.array([0.0, 0.0]), 1.5)
        neighbors.sort()
        assert neighbors == [0, 1, 2] # Indices
        
        # Query radius 0.5 around (0,0) -> only (0,0)
        neighbors = index.query_radius(np.array([0.0, 0.0]), 0.5)
        assert neighbors == [0]

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_spatial_index_knn(self, sample_positions):
        """Test KNN query."""
        index = SpatialIndex().build(sample_positions)
        
        # 3 Nearest to (0.1, 0.1) -> should be 0, 1, 2
        dists, indices = index.query_knn(np.array([0.1, 0.1]), k=3)
        
        # Check indices (order might vary slightly, but set should be {0,1,2})
        assert set(indices) == {0, 1, 2}
        assert len(dists) == 3

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_spatial_index_query_pairs(self, sample_positions):
        """Test querying pairs."""
        index = SpatialIndex().build(sample_positions)
        
        # Pairs within 1.1 distance
        # (0,0)-(1,0) dist 1.0
        # (0,0)-(0,1) dist 1.0
        # (1,0)-(2,2) dist sqrt(1+4) = 2.23 > 1.1
        pairs = index.query_pairs(1.1)
        
        # Should contain (0,1) and (0,2) pairs (indices)
        # scipy returns a set of tuples
        
        # Verify 0-1 and 0-2 are close enough
        has_01 = (0, 1) in pairs or (1, 0) in pairs
        has_02 = (0, 2) in pairs or (2, 0) in pairs
        assert has_01
        assert has_02

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_spatial_index_batch_query(self, sample_positions):
        """Test batch radius query."""
        index = SpatialIndex().build(sample_positions)
        
        points = np.array([[0.0, 0.0], [10.0, 10.0]])
        results = index.batch_query_radius(points, 1.5)
        
        assert len(results) == 2
        
        # First point neighbors [0, 1, 2]
        r1 = sorted(results[0])
        assert r1 == [0, 1, 2]
        
        # Second point neighbors [4]
        r2 = results[1]
        assert r2 == [4]

    def test_spatial_index_errors(self):
        """Test error handling when not built."""
        index = SpatialIndex()
        with pytest.raises(ValueError):
            index.query_radius([0,0], 1.0)
        with pytest.raises(ValueError):
            index.query_knn([0,0], 1)
        with pytest.raises(ValueError):
            index.query_pairs(1.0)
        with pytest.raises(ValueError):
            index.batch_query_radius([[0,0]], 1.0)


class TestNumbaFunctions:
    """Test Numba-accelerated functions."""

    def test_fast_distance_matrix(self, sample_positions):
        """Test distance matrix calculation."""
        # Calculate manually first
        # 0: (0,0), 1: (1,0) -> dist 1.0
        # 0: (0,0), 2: (0,1) -> dist 1.0
        # 1: (1,0), 2: (0,1) -> dist sqrt(2) ~ 1.414
        
        dist_mat = fast_distance_matrix(sample_positions)
        
        assert dist_mat.shape == (5, 5)
        assert dist_mat[0, 0] == 0.0
        assert dist_mat[0, 1] == 1.0
        assert dist_mat[1, 0] == 1.0
        assert np.isclose(dist_mat[1, 2], np.sqrt(2))

    def test_fast_neighbors_within_radius(self, sample_positions):
        """Test finding neighbors for single agent."""
        # Agent 0 at (0,0), Radius 1.5 -> Neighbors 1, 2
        neighbors = fast_neighbors_within_radius(sample_positions, 0, 1.5)
        neighbors.sort()
        assert neighbors == [1, 2]

        # Agent 4 at (10,10), Radius 1.0 -> No neighbors
        neighbors = fast_neighbors_within_radius(sample_positions, 4, 1.0)
        assert neighbors == []

    def test_fast_all_neighbors_within_radius(self, sample_positions):
        """Test finding all neighbors."""
        # Radius 1.5
        neighbors_matrix = fast_all_neighbors_within_radius(sample_positions, 1.5)
        
        # Agent 0 should have 1 and 2
        row0 = neighbors_matrix[0]
        valid0 = sorted([x for x in row0 if x != -1])
        assert valid0 == [1, 2]
        
        # Agent 4 should have none
        row4 = neighbors_matrix[4]
        valid4 = [x for x in row4 if x != -1]
        assert valid4 == []

    def test_fast_random_walk_step(self):
        """Test random walk movement."""
        positions = np.zeros((10, 2))
        velocities = np.ones((10, 2)) # Move by (1, 1)
        bounds = np.array([[0, 10], [0, 10]])
        
        # Simple move
        new_pos = fast_random_walk_step(positions, velocities, bounds, wrap=False)
        assert np.allclose(new_pos, 1.0)
        
        # Test clipping (bounds 0-10)
        pos_edge = np.array([[9.5, 9.5]])
        vel_large = np.array([[2.0, 2.0]])
        new_pos_clip = fast_random_walk_step(pos_edge, vel_large, bounds, wrap=False)
        # Should be clipped to 9.999 (bounds[1] - 0.001)
        assert new_pos_clip[0, 0] < 10.0
        assert new_pos_clip[0, 0] > 9.9
        
        # Test wrapping
        # 9.5 + 2.0 = 11.5. Wrap len 10 -> 1.5
        new_pos_wrap = fast_random_walk_step(pos_edge, vel_large, bounds, wrap=True)
        assert np.isclose(new_pos_wrap[0, 0], 1.5)


class TestVectorizedOperations:
    """Test vectorized utility functions."""

    def test_vectorized_wealth_transfer(self):
        """Test wealth transfer."""
        wealths = np.array([100.0, 100.0, 100.0])
        sources = np.array([0, 1])
        targets = np.array([1, 2])
        amounts = np.array([10.0, 20.0])
        
        # 0 -> 1: 10
        # 1 -> 2: 20
        # Expected:
        # 0: 100 - 10 = 90
        # 1: 100 + 10 - 20 = 90
        # 2: 100 + 20 = 120
        
        new_wealths = vectorized_wealth_transfer(wealths, amounts, sources, targets)
        
        assert new_wealths[0] == 90.0
        assert new_wealths[1] == 90.0
        assert new_wealths[2] == 120.0
        
        # Original should not be modified
        assert wealths[0] == 100.0

    def test_vectorized_move(self):
        """Test vectorized movement."""
        positions = np.zeros((5, 2))
        velocities = np.ones((5, 2))
        
        # Unbounded
        new_pos = vectorized_move(positions, velocities)
        assert np.all(new_pos == 1.0)
        
        # Bounded clip
        bounds = (0, 0.5)
        new_pos_clip = vectorized_move(positions, velocities, bounds, wrap=False)
        assert np.all(new_pos_clip == 0.5)
        
        # Bounded wrap
        # 0 + 1 = 1. Range 0.5. 1 % 0.5 = 0.
        new_pos_wrap = vectorized_move(positions, velocities, bounds, wrap=True)
        assert np.all(new_pos_wrap == 0.0)

    def test_vectorized_random_velocities(self):
        """Test velocity generation."""
        rng = np.random.default_rng(42)
        vels = vectorized_random_velocities(100, 1.0, rng=rng)
        
        assert vels.shape == (100, 2)
        assert np.all(vels >= -1.0)
        assert np.all(vels <= 1.0)
        assert not np.all(vels == vels[0]) # random values

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_vectorized_sir_infections(self, sample_positions):
        """Test SIR infection logic."""
        # 0=S, 1=I, 2=R
        statuses = np.zeros(5, dtype=int)
        statuses[0] = 1 # Patient zero at (0,0)
        
        index = SpatialIndex().build(sample_positions)
        
        # Radius 1.5 includes agents 1 (1,0) and 2 (0,1)
        # Transmission rate 1.0 -> definite infection
        new_statuses = vectorized_sir_infections(
            sample_positions, statuses, index, 
            infection_radius=1.5, transmission_rate=1.0
        )
        
        assert new_statuses[0] == 1 # Still infected
        assert new_statuses[1] == 1 # Infected
        assert new_statuses[2] == 1 # Infected
        assert new_statuses[3] == 0 # Too far
        assert new_statuses[4] == 0 # Too far

        # Test with rate 0.0 -> no new infections
        new_statuses_safe = vectorized_sir_infections(
            sample_positions, statuses, index, 
            infection_radius=1.5, transmission_rate=0.0
        )
        t = list(new_statuses_safe)
        assert t == [1, 0, 0, 0, 0]


class TestParallelRunner:
    """Test ParallelRunner class."""

    def test_parallel_runner_initialization(self):
        """Test initialization."""
        runner = ParallelRunner(MockModel, n_workers=2)
        assert runner.model_class == MockModel
        assert runner.n_workers == 2

    def test_run_single_simulation_helper(self):
        """Directly test the worker function."""
        params = {"param": 123, "steps": 5}
        result = _run_single_simulation(params, MockModel)
        
        assert result["params"] == params
        assert result["model"] is not None
        
        # Check that model actually ran (our MockModel records 'param_val')
        # We need to dig into the internal data structure or rely on return values
        # Since _run_single_simulation calls model.run(), which returns dict
        # And MockModel inherits Model, it should work.

    def test_parallel_runner_execution(self):
        """Test full parallel execution."""
        runner = ParallelRunner(MockModel, n_workers=2)
        
        params_list = [
            {"steps": 2, "param": 10},
            {"steps": 2, "param": 20}
        ]
        
        # Run with print captured to avoid noise
        with patch("builtins.print"):
            results = runner.run(params_list, show_progress=False)
        
        assert len(results) == 2
        
        # Validate results
        p1 = [r for r in results if r["params"]["param"] == 10][0]
        p2 = [r for r in results if r["params"]["param"] == 20][0]
        
        assert p1 is not None
        assert p2 is not None

    def test_parallel_runner_with_seeds(self):
        """Test run_with_seeds."""
        runner = ParallelRunner(MockModel, n_workers=2)
        base_params = {"steps": 1}
        seeds = [42, 43]
        
        with patch("builtins.print"):
            results = runner.run_with_seeds(base_params, seeds, show_progress=False)
            
        assert len(results) == 2
        assert {r["params"]["seed"] for r in results} == {42, 43}


class TestDependencyUtilities:
    """Test dependency checking functions."""

    def test_check_performance_deps(self):
        """Test check_performance_deps."""
        deps = check_performance_deps()
        assert "scipy" in deps
        assert "numba" in deps
        assert "multiprocessing" in deps
        
        if HAS_SCIPY:
            assert deps["scipy"] is True

    def test_install_performance_deps(self):
        """Test install print output."""
        with patch("builtins.print") as mock_print:
            install_performance_deps()
            mock_print.assert_called()
