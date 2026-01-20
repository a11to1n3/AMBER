import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import benchmarks
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmarks.models.vectorized_models import (
    VectorizedWealthTransfer,
    VectorizedSIRModel,
    VectorizedRandomWalk
)

class TestScientificCorrectness:
    """
    Rigorously verify the scientific correctness of the vectorized models.
    """

    def test_wealth_conservation(self):
        """
        Wealth Transfer Model: Total wealth must be conserved.
        Closed system: Sum(wealth_t) == Sum(wealth_0).
        """
        n_agents = 1000
        initial_wealth = 5
        
        model = VectorizedWealthTransfer({
            'n': n_agents,
            'steps': 50,
            'initial_wealth': initial_wealth,
            'seed': 42
        })
        
        # Initial check
        model.setup() # Initialize state
        assert np.isclose(model.wealths.sum(), n_agents * initial_wealth), \
            "Initial total wealth incorrect"
            
        model.run()
        
        # Final check
        final_total = model.wealths.sum()
        expected = n_agents * initial_wealth
        
        assert np.isclose(final_total, expected), \
            f"Wealth conservation violated! Expected {expected}, got {final_total}"
            
    def test_wealth_inequality_growth(self):
        """
        Wealth Transfer Model: Gini coefficient should increase from 0 (equality).
        """
        model = VectorizedWealthTransfer({
            'n': 100,
            'steps': 100,
            'initial_wealth': 1,
            'seed': 42
        })
        
        model.setup()
        initial_gini = model._calculate_gini()
        model.run()
        final_gini = model._calculate_gini()
        
        assert initial_gini == 0.0, "Initial Gini should be 0 for equal start"
        assert final_gini > 0.1, "Gini should increase over time (entropy)"

    def test_sir_conservation(self):
        """
        SIR Model: Total population (S + I + R) must be constant.
        """
        n_agents = 500
        model = VectorizedSIRModel({
            'n': n_agents,
            'steps': 50,
            'initial_infected': 10,
            'seed': 42
        })
        
        results = model.run()
        
        # Get history from recorded data
        s = np.array(results['model']['susceptible'])
        i = np.array(results['model']['infected'])
        r = np.array(results['model']['recovered'])
        
        total = s + i + r
        
        # Check every step
        assert np.all(total == n_agents), \
            f"Population conservation violated. Min: {total.min()}, Max: {total.max()}"

    def test_sir_monotonic_recovery(self):
        """
        SIR Model: Recovered count should be non-decreasing.
        Dead/Recovered agents do not become susceptible again.
        """
        model = VectorizedSIRModel({
            'n': 500,
            'steps': 50,
            'initial_infected': 50,
            'seed': 42
        })
        
        results = model.run()
        r = np.array(results['model']['recovered'])
        
        # Check if sorted (non-decreasing)
        assert np.all(np.diff(r) >= 0), "Recovered population decreased (impossible)"

    def test_random_walk_boundedness(self):
        """
        Random Walk: Agents should stay within world bounds (if hard bounded or wrapped).
        The current implementation is HARD BOUNDED (0, world_size).
        """
        world_size = 100
        model = VectorizedRandomWalk({
            'n': 100,
            'steps': 50,
            'world_size': world_size,
            'speed': 5.0, # Fast movement to hit walls
            'seed': 42
        })
        
        model.run()
        
        x = model.positions[:, 0]
        y = model.positions[:, 1]
        
        assert np.all((x >= 0) & (x <= world_size)), "Agents escaped X bounds"
        assert np.all((y >= 0) & (y <= world_size)), "Agents escaped Y bounds"

if __name__ == "__main__":
    # Allow running directly
    sys.exit(pytest.main([__file__]))
