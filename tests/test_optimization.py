"""
Tests for ambr.optimization module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import ambr as am
from ambr.optimization import (
    ParameterSpace, 
    objective_function,
    grid_search,
    random_search,
    bayesian_optimization
)


class TestParameterSpace:
    """Test cases for ParameterSpace class."""
    
    def test_parameter_space_initialization(self):
        """Test ParameterSpace initialization."""
        parameters = {
            'param1': [1, 2, 3],
            'param2': am.IntRange(10, 20),
            'param3': 'fixed_value'
        }
        
        space = ParameterSpace(parameters)
        
        assert space.parameters == parameters
        assert 'param1' in space.parameters
        assert 'param2' in space.parameters
        assert 'param3' in space.parameters
    
    def test_parameter_space_sample(self):
        """Test sampling from parameter space."""
        parameters = {
            'list_param': [0.1, 0.2, 0.3],
            'range_param': am.IntRange(5, 15),
            'fixed_param': 'constant'
        }
        
        space = ParameterSpace(parameters)
        
        # Sample multiple parameter combinations
        samples = [space.sample() for _ in range(10)]
        
        # Check that all samples are valid
        for sample in samples:
            assert 'list_param' in sample
            assert 'range_param' in sample
            assert 'fixed_param' in sample
            
            assert sample['list_param'] in [0.1, 0.2, 0.3]
            assert 5 <= sample['range_param'] <= 15
            assert sample['fixed_param'] == 'constant'
    
    def test_parameter_space_sample_deterministic(self):
        """Test that sampling is deterministic with seed."""
        parameters = {
            'range_param': am.IntRange(1, 100),
            'list_param': [1, 2, 3, 4, 5]
        }
        
        # Create two spaces with same seed
        space1 = ParameterSpace(parameters)
        space2 = ParameterSpace(parameters)
        
        # Set same random seed
        np.random.seed(42)
        sample1 = space1.sample()
        
        np.random.seed(42)
        sample2 = space2.sample()
        
        assert sample1 == sample2
    
    def test_parameter_space_grid_sample(self):
        """Test grid sampling from parameter space."""
        parameters = {
            'param1': [1, 2],
            'param2': [0.1, 0.2],
            'fixed': 'value'
        }
        
        space = ParameterSpace(parameters)
        grid_samples = space.grid_sample()
        
        # Should have 2 * 2 = 4 combinations
        assert len(grid_samples) == 4
        
        # Check all combinations exist
        expected = [
            {'param1': 1, 'param2': 0.1, 'fixed': 'value'},
            {'param1': 1, 'param2': 0.2, 'fixed': 'value'},
            {'param1': 2, 'param2': 0.1, 'fixed': 'value'},
            {'param1': 2, 'param2': 0.2, 'fixed': 'value'}
        ]
        
        for exp in expected:
            assert exp in grid_samples
    
    def test_parameter_space_with_intrange_grid(self):
        """Test grid sampling with IntRange parameters."""
        parameters = {
            'range_param': am.IntRange(10, 12),  # Will sample 10, 11, 12
            'list_param': ['a', 'b']
        }
        
        space = ParameterSpace(parameters)
        grid_samples = space.grid_sample()
        
        # Should have 3 * 2 = 6 combinations  
        assert len(grid_samples) == 6
        
        # Check that IntRange values are properly expanded
        range_values = [s['range_param'] for s in grid_samples]
        assert 10 in range_values
        assert 11 in range_values
        assert 12 in range_values
    
    def test_parameter_space_empty(self):
        """Test ParameterSpace with empty parameters."""
        space = ParameterSpace({})
        
        sample = space.sample()
        assert sample == {}
        
        grid_samples = space.grid_sample()
        assert grid_samples == [{}]
    
    def test_parameter_space_only_fixed(self):
        """Test ParameterSpace with only fixed parameters."""
        parameters = {
            'param1': 'value1',
            'param2': 42
        }
        
        space = ParameterSpace(parameters)
        
        # Sample should always return the same values
        sample1 = space.sample()
        sample2 = space.sample()
        
        assert sample1 == sample2 == parameters
        
        # Grid sample should return single combination
        grid_samples = space.grid_sample()
        assert len(grid_samples) == 1
        assert grid_samples[0] == parameters


class TestObjectiveFunction:
    """Test cases for objective_function."""
    
    def test_objective_function_basic(self):
        """Test basic objective function usage."""
        class TestModel(am.Model):
            def setup(self):
                self.counter = 0
            
            def step(self):
                self.counter += 1
            
            def update(self):
                super().update()
                self.record_model('counter', self.counter)
            
            def end(self):
                # Record final counter in the last step data
                if self._model_data:
                    self._model_data[-1]['final_counter'] = self.counter
        
        parameters = {'steps': 5}
        
        with patch('builtins.print'):
            result = objective_function(TestModel, parameters, 'final_counter')
        
        # Should return the final counter value (4 steps means counter goes from 0 to 4)
        assert result == 4
    
    def test_objective_function_with_iterations(self):
        """Test objective function with multiple iterations."""
        class TestModel(am.Model):
            def setup(self):
                self.value = 0
            
            def step(self):
                # Return random value for testing averaging
                import random
                self.value = random.randint(1, 10)
            
            def update(self):
                super().update()
                # Only record after the first step (t > 0)
                if self.t > 0 and hasattr(self, 'value'):
                    self.record_model('test_metric', self.value)
        
        parameters = {'steps': 2}
        
        with patch('builtins.print'):
            result = objective_function(TestModel, parameters, 'test_metric', iterations=5)
        
        # Should return average of multiple runs
        assert isinstance(result, (int, float))
        assert result >= 1  # Minimum possible value
        assert result <= 10  # Maximum possible value
    
    def test_objective_function_minimize(self):
        """Test objective function with minimize=True."""
        class TestModel(am.Model):
            def setup(self):
                self.value = self.p.get('value', 10)
            
            def step(self):
                pass
            
            def update(self):
                super().update()
                self.record_model('result', self.value)
        
        parameters = {'steps': 1, 'value': 5}
        
        with patch('builtins.print'):
            result = objective_function(TestModel, parameters, 'result', minimize=True)
        
        # Should return the raw value (sorting is handled by optimization methods)
        assert result == 5
    
    def test_objective_function_missing_metric(self):
        """Test objective function with missing metric."""
        class TestModel(am.Model):
            def step(self):
                pass  # Don't record anything
        
        parameters = {'steps': 1}
        
        with patch('builtins.print'):
            result = objective_function(TestModel, parameters, 'nonexistent_metric')
        
        # Should return 0 when metric doesn't exist
        assert result == 0


class TestGridSearch:
    """Test cases for grid_search function."""
    
    def test_grid_search_basic(self):
        """Test basic grid search functionality."""
        class TestModel(am.Model):
            def setup(self):
                self.result = 0
            
            def step(self):
                # Metric is sum of parameters
                self.result = self.p['param1'] + self.p['param2']
            
            def update(self):
                super().update()
                if self.t > 0:  # Only record after step execution
                    self.record_model('sum', self.result)
        
        parameter_space = ParameterSpace({
            'param1': [1, 2],
            'param2': [10, 20],
            'steps': 2
        })
        
        with patch('builtins.print'):
            results = grid_search(TestModel, parameter_space, 'sum')
        
        # Should test all 4 combinations
        assert len(results) == 4
        
        # Check that results are sorted by objective value
        objectives = [r['objective'] for r in results]
        assert objectives == sorted(objectives, reverse=True)  # Descending order
        
        # Best result should be param1=2, param2=20 (sum=22)
        best = results[0]
        assert best['parameters']['param1'] == 2
        assert best['parameters']['param2'] == 20
        assert best['objective'] == 22
    
    def test_grid_search_minimize(self):
        """Test grid search with minimization."""
        class TestModel(am.Model):
            def setup(self):
                self.result = 0
            
            def step(self):
                self.result = self.p['x'] ** 2  # Quadratic function
            
            def update(self):
                super().update()
                if self.t > 0:  # Only record after step execution
                    self.record_model('squared', self.result)
        
        parameter_space = ParameterSpace({
            'x': [-2, -1, 0, 1, 2],
            'steps': 2
        })
        
        with patch('builtins.print'):
            results = grid_search(TestModel, parameter_space, 'squared', minimize=True)
        
        # Best result should be x=0 (squared=0)
        best = results[0]
        assert best['parameters']['x'] == 0
        assert best['objective'] == 0
    
    def test_grid_search_with_iterations(self):
        """Test grid search with multiple iterations."""
        class TestModel(am.Model):
            def setup(self):
                self.value = 0
            
            def step(self):
                # Add some randomness
                import random
                base_value = self.p['base']
                noise = random.uniform(-0.1, 0.1)
                self.value = base_value + noise
                
            def update(self):
                super().update()
                if self.t > 0:
                    self.record_model('noisy_value', self.value)
        
        parameter_space = ParameterSpace({
            'base': [1, 2],
            'steps': 2
        })
        
        with patch('builtins.print'):
            results = grid_search(TestModel, parameter_space, 'noisy_value', iterations=3)
        
        # Should still have 2 parameter combinations
        assert len(results) == 2
        
        # Results should be averaged over iterations
        for result in results:
            assert isinstance(result['objective'], (int, float))


class TestRandomSearch:
    """Test cases for random_search function."""
    
    def test_random_search_basic(self):
        """Test basic random search functionality."""
        class TestModel(am.Model):
            def setup(self):
                self.result = 0
                
            def step(self):
                self.result = self.p['param'] * 2
                
            def update(self):
                super().update()
                if self.t > 0:
                    self.record_model('doubled', self.result)
        
        parameter_space = ParameterSpace({
            'param': am.IntRange(1, 10),
            'steps': 2
        })
        
        with patch('builtins.print'):
            results = random_search(TestModel, parameter_space, 'doubled', n_samples=5)
        
        # Should test 5 random combinations
        assert len(results) == 5
        
        # All parameters should be within range
        for result in results:
            assert 1 <= result['parameters']['param'] <= 10
            assert result['objective'] == result['parameters']['param'] * 2
        
        # Results should be sorted
        objectives = [r['objective'] for r in results]
        assert objectives == sorted(objectives, reverse=True)
    
    def test_random_search_with_seed(self):
        """Test random search reproducibility with seed."""
        class TestModel(am.Model):
            def setup(self):
                self.value = 0
                
            def step(self):
                self.value = self.p['x']
                
            def update(self):
                super().update()
                if self.t > 0:
                    self.record_model('value', self.value)
        
        parameter_space = ParameterSpace({
            'x': am.IntRange(1, 100),
            'steps': 2
        })
        
        # Run twice with same seed
        with patch('builtins.print'):
            results1 = random_search(TestModel, parameter_space, 'value', n_samples=3, seed=42)
            results2 = random_search(TestModel, parameter_space, 'value', n_samples=3, seed=42)
        
        # Should get same parameter combinations
        params1 = [r['parameters'] for r in results1]
        params2 = [r['parameters'] for r in results2]
        assert params1 == params2


class TestBayesianOptimization:
    """Test cases for bayesian_optimization function."""
    
    @pytest.mark.slow
    def test_bayesian_optimization_basic(self):
        """Test basic Bayesian optimization functionality."""
        class TestModel(am.Model):
            def setup(self):
                self.result = 0
                
            def step(self):
                # Simple quadratic function with optimum at x=5
                x = self.p['x']
                self.result = -(x - 5) ** 2 + 25  # Maximum at x=5
                
            def update(self):
                super().update()
                if self.t > 0:
                    self.record_model('objective', self.result)
        
        parameter_space = ParameterSpace({
            'x': am.IntRange(0, 10),
            'steps': 2
        })
        
        with patch('builtins.print'):
            results = bayesian_optimization(
                TestModel, 
                parameter_space, 
                'objective', 
                n_calls=10,
                random_state=42
            )
        
        # Should find parameters close to optimum
        best = results[0]
        best_x = best['parameters']['x']
        
        # Should be close to optimal value of 5
        assert abs(best_x - 5) <= 2  # Allow some tolerance
        assert best['objective'] > 20  # Should be close to maximum of 25
    
    @pytest.mark.slow
    def test_bayesian_optimization_minimize(self):
        """Test Bayesian optimization with minimization."""
        class TestModel(am.Model):
            def setup(self):
                self.result = 0
                
            def step(self):
                x = self.p['x']
                self.result = (x - 3) ** 2  # Minimum at x=3
                
            def update(self):
                super().update()
                if self.t > 0:
                    self.record_model('cost', self.result)
        
        parameter_space = ParameterSpace({
            'x': am.IntRange(0, 6),
            'steps': 2
        })
        
        with patch('builtins.print'):
            results = bayesian_optimization(
                TestModel, 
                parameter_space, 
                'cost', 
                minimize=True,
                n_calls=8,
                random_state=42
            )
        
        # Should find minimum
        best = results[0]
        best_x = best['parameters']['x']
        
        # Should be close to optimal value of 3
        assert abs(best_x - 3) <= 1
        assert best['objective'] <= 2  # Should be close to minimum of 0
    
    def test_bayesian_optimization_with_noise(self):
        """Test Bayesian optimization with noisy objective."""
        class NoisyModel(am.Model):
            def setup(self):
                self.result = 0
                
            def step(self):
                x = self.p['x']
                noise = np.random.normal(0, 0.1)  # Small noise
                self.result = -(x - 2) ** 2 + 4 + noise
                
            def update(self):
                super().update()
                if self.t > 0:
                    self.record_model('noisy_obj', self.result)
        
        parameter_space = ParameterSpace({
            'x': am.IntRange(0, 4),
            'steps': 2
        })
        
        with patch('builtins.print'):
            results = bayesian_optimization(
                NoisyModel, 
                parameter_space, 
                'noisy_obj', 
                n_calls=6,
                random_state=42
            )
        
        # Should still find reasonable solution despite noise
        best = results[0]
        assert best['objective'] > 2  # Should be reasonably high


class TestOptimizationIntegration:
    """Integration tests for optimization functions."""
    
    def test_optimization_workflow(self):
        """Test complete optimization workflow."""
        class OptimizationModel(am.Model):
            def setup(self):
                self.agents = []
                for i in range(self.p['n_agents']):
                    agent = am.Agent(self, i)
                    self.agents.append(agent)
                self.efficiency = 0
            
            def step(self):
                # Simple metric based on parameters
                self.efficiency = self.p['n_agents'] * self.p['multiplier']
                
            def update(self):
                super().update()
                if self.t > 0:
                    self.record_model('efficiency', self.efficiency)
        
        parameter_space = ParameterSpace({
            'n_agents': [5, 10, 15],
            'multiplier': [1, 2, 3],
            'steps': 2
        })
        
        # Test grid search
        with patch('builtins.print'):
            grid_results = grid_search(OptimizationModel, parameter_space, 'efficiency')
        
        assert len(grid_results) == 9  # 3 * 3 combinations
        
        # Best should be n_agents=15, multiplier=3
        best = grid_results[0]
        assert best['parameters']['n_agents'] == 15
        assert best['parameters']['multiplier'] == 3
        assert best['objective'] == 45
        
        # Test random search
        with patch('builtins.print'):
            random_results = random_search(OptimizationModel, parameter_space, 'efficiency', n_samples=5)
        
        assert len(random_results) == 5
        
        # All results should be valid
        for result in random_results:
            assert result['parameters']['n_agents'] in [5, 10, 15]
            assert result['parameters']['multiplier'] in [1, 2, 3]
            expected = result['parameters']['n_agents'] * result['parameters']['multiplier']
            assert result['objective'] == expected 