"""
Tests for ambr.experiment module.
"""

import pytest
from unittest.mock import Mock, patch
import polars as pl
import ambr as am
from ambr.experiment import IntRange, Sample, Experiment


class TestIntRange:
    """Test cases for IntRange class."""
    
    def test_intrange_initialization(self):
        """Test IntRange initialization."""
        range_obj = IntRange(10, 50)
        
        assert range_obj.start == 10
        assert range_obj.end == 50
    
    def test_intrange_with_equal_start_end(self):
        """Test IntRange with equal start and end values."""
        range_obj = IntRange(25, 25)
        
        assert range_obj.start == 25
        assert range_obj.end == 25
    
    def test_intrange_with_negative_values(self):
        """Test IntRange with negative values."""
        range_obj = IntRange(-10, 10)
        
        assert range_obj.start == -10
        assert range_obj.end == 10
    
    def test_intrange_invalid_order(self):
        """Test IntRange with start > end (should still work)."""
        range_obj = IntRange(50, 10)
        
        assert range_obj.start == 50
        assert range_obj.end == 10
    
    def test_intrange_repr(self):
        """Test string representation of IntRange."""
        range_obj = IntRange(5, 15)
        repr_str = repr(range_obj)
        
        assert 'IntRange' in repr_str
        assert '5' in repr_str
        assert '15' in repr_str


class TestSample:
    """Test cases for Sample class."""
    
    def test_sample_initialization_simple(self):
        """Test Sample initialization with simple parameters."""
        parameters = {
            'fixed_param': 'value',
            'list_param': [1, 2, 3]
        }
        sample = Sample(parameters, n=5)
        
        assert sample.parameters == parameters
        assert sample.n == 5
        assert len(sample.combinations) == 5
    
    def test_sample_with_intrange(self):
        """Test Sample with IntRange parameters."""
        parameters = {
            'range_param': IntRange(10, 50),
            'fixed_param': 'constant'
        }
        sample = Sample(parameters, n=10)
        
        assert len(sample.combinations) == 10
        
        # Check that all combinations have the fixed parameter
        for combo in sample.combinations:
            assert combo['fixed_param'] == 'constant'
            assert 'range_param' in combo
            assert isinstance(combo['range_param'], int)
            assert 10 <= combo['range_param'] <= 50
    
    def test_sample_with_multiple_ranges(self):
        """Test Sample with multiple IntRange parameters."""
        parameters = {
            'range1': IntRange(1, 10),
            'range2': IntRange(100, 200),
            'fixed': 'value'
        }
        sample = Sample(parameters, n=5)
        
        assert len(sample.combinations) == 5
        
        for combo in sample.combinations:
            assert combo['fixed'] == 'value'
            assert 1 <= combo['range1'] <= 10
            assert 100 <= combo['range2'] <= 200
    
    def test_sample_with_list_parameters(self):
        """Test Sample with list parameters."""
        parameters = {
            'list_param': [0.1, 0.2, 0.3],
            'fixed_param': 42
        }
        sample = Sample(parameters, n=6)
        
        # Should cycle through list values
        assert len(sample.combinations) == 6
        
        list_values = [combo['list_param'] for combo in sample.combinations]
        assert all(val in [0.1, 0.2, 0.3] for val in list_values)
    
    def test_sample_only_fixed_parameters(self):
        """Test Sample with only fixed parameters."""
        parameters = {
            'param1': 'value1',
            'param2': 42
        }
        sample = Sample(parameters, n=3)
        
        # Should generate n identical combinations
        assert len(sample.combinations) == 3
        
        for combo in sample.combinations:
            assert combo == parameters
    
    def test_sample_empty_parameters(self):
        """Test Sample with empty parameters."""
        parameters = {}
        sample = Sample(parameters, n=3)
        
        assert len(sample.combinations) == 3
        assert all(combo == {} for combo in sample.combinations)
    
    def test_sample_n_zero(self):
        """Test Sample with n=0."""
        parameters = {'param': 'value'}
        sample = Sample(parameters, n=0)
        
        assert len(sample.combinations) == 0
    
    def test_sample_generate_combinations_deterministic(self):
        """Test that Sample generates deterministic combinations."""
        parameters = {
            'range_param': IntRange(1, 100),
            'fixed_param': 'test'
        }
        
        sample1 = Sample(parameters, n=10)
        sample2 = Sample(parameters, n=10)
        
        # Should generate the same combinations
        assert sample1.combinations == sample2.combinations


class TestExperiment:
    """Test cases for Experiment class."""
    
    def test_experiment_initialization(self):
        """Test Experiment initialization."""
        # Create a simple model for testing
        class TestModel(am.Model):
            def setup(self):
                pass
            def step(self):
                self.record_model('test_value', self.t)
        
        sample = Sample({'param': 'value'}, n=1)
        experiment = Experiment(TestModel, sample, iterations=2, record=True)
        
        assert experiment.model_type == TestModel
        assert experiment.sample == sample
        assert experiment.iterations == 2
        assert experiment.record is True
    
    def test_experiment_run_single_combination(self):
        """Test running experiment with single parameter combination."""
        class TestModel(am.Model):
            def setup(self):
                self.setup_called = True
            
            def step(self):
                self.record_model('step_value', self.t * 10)
            
            def end(self):
                self.record_model('final_value', 999)
        
        parameters = {'test_param': 'test_value', 'steps': 3}
        sample = Sample(parameters, n=1)
        experiment = Experiment(TestModel, sample, iterations=2)
        
        with patch('time.time', side_effect=[0, 0.1] * 10), patch('builtins.print'):
            results = experiment.run()
        
        # Check results structure
        assert 'info' in results
        assert 'parameters' in results
        assert 'agents' in results
        assert 'model' in results
        
        # Check info
        assert results['info']['model_type'] == 'TestModel'
        assert results['info']['sample_size'] == 1
        assert results['info']['iterations'] == 2
        
        # Check parameters
        assert len(results['parameters']) == 1
        assert results['parameters']['test_param'].item() == 'test_value'
        
        # Check that we have data from multiple iterations
        model_data = results['model']
        iterations = model_data['iteration'].unique().to_list()
        assert len(iterations) == 2
        assert 0 in iterations and 1 in iterations
    
    def test_experiment_run_multiple_combinations(self):
        """Test running experiment with multiple parameter combinations."""
        class TestModel(am.Model):
            def step(self):
                self.record_model('param_value', self.p['variable_param'])
        
        parameters = {
            'variable_param': IntRange(10, 30),
            'fixed_param': 'constant',
            'steps': 2
        }
        sample = Sample(parameters, n=3)
        experiment = Experiment(TestModel, sample, iterations=2)
        
        with patch('time.time', side_effect=[0, 0.1] * 20), patch('builtins.print'):
            results = experiment.run()
        
        # Should have 3 parameter combinations * 2 iterations = 6 total runs
        model_data = results['model']
        
        # Check we have the right number of parameter combinations
        param_combinations = results['parameters']
        assert len(param_combinations) == 3
        
        # Check we have data from all runs
        unique_combinations = model_data.select(['variable_param', 'iteration']).unique()
        assert len(unique_combinations) == 6  # 3 params × 2 iterations
    
    def test_experiment_with_agent_data(self):
        """Test experiment that generates agent data."""
        class TestModelWithAgents(am.Model):
            def setup(self):
                # Add some mock agents
                for i in range(3):
                    mock_agent = Mock()
                    mock_agent.id = i
                    self.add_agent(mock_agent)
                
                # key fix: ensure wealth column exists
                self.population.data = self.population.data.with_columns(
                    pl.lit(0).alias('wealth')
                )
            
            def step(self):
                # Update agent data
                self.batch_update_agents([0, 1, 2], {'wealth': self.t * 5, 'step': self.t})
        
        parameters = {'n_agents': 3, 'steps': 2}
        sample = Sample(parameters, n=1)
        experiment = Experiment(TestModelWithAgents, sample, iterations=1)
        
        with patch('time.time', side_effect=[0, 0.1] * 5), patch('builtins.print'):
            results = experiment.run()
        
        # Check that we have agent data
        agents_data = results['agents']
        assert len(agents_data) > 0
        assert 'id' in agents_data.columns
        assert 'step' in agents_data.columns
        assert 'wealth' in agents_data.columns
        
        # Check that parameter information was added to agent data
        assert 'n_agents' in agents_data.columns
        assert agents_data['n_agents'].unique().to_list() == [3]
    
    def test_experiment_no_iterations_label(self):
        """Test experiment with only 1 iteration (no iteration column)."""
        class SimpleModel(am.Model):
            def step(self):
                self.record_model('counter', 1)
        
        parameters = {'param': 'value', 'steps': 1}
        sample = Sample(parameters, n=1)
        experiment = Experiment(SimpleModel, sample, iterations=1)
        
        with patch('time.time', side_effect=[0, 0.1] * 5), patch('builtins.print'):
            results = experiment.run()
        
        # With only 1 iteration, should not have iteration column
        model_data = results['model']
        assert 'iteration' not in model_data.columns
    
    def test_experiment_empty_sample(self):
        """Test experiment with empty sample."""
        class EmptyModel(am.Model):
            def step(self):
                pass
        
        sample = Sample({}, n=0)
        experiment = Experiment(EmptyModel, sample, iterations=1)
        
        with patch('time.time', side_effect=[0, 0.1] * 5), patch('builtins.print'):
            results = experiment.run()
        
        # Should have empty results
        assert len(results['parameters']) == 0
        assert len(results['model']) == 0
        assert len(results['agents']) == 0
    
    def test_experiment_record_parameter(self):
        """Test that record parameter is properly handled."""
        class RecordModel(am.Model):
            def step(self):
                self.record_model('test', 1)
        
        sample = Sample({'param': 'value'}, n=1)
        
        # Test with record=True
        experiment_record = Experiment(RecordModel, sample, record=True)
        assert experiment_record.record is True
        
        # Test with record=False
        experiment_no_record = Experiment(RecordModel, sample, record=False)
        assert experiment_no_record.record is False


class TestExperimentIntegration:
    """Integration tests for Experiment class."""
    
    def test_experiment_with_real_model(self, basic_model):
        """Test experiment with a real model from fixtures."""
        # Use the basic_model fixture but need to recreate for experiment
        class TestModel(am.Model):
            def setup(self):
                self.test_agents = {}
                for i in range(3):
                    agent = am.Agent(self, i)
                    self.test_agents[i] = agent
            
            def step(self):
                self.record_model('step_count', self.t)
        
        parameters = {'n': 3, 'steps': 2}
        sample = Sample(parameters, n=1)
        experiment = Experiment(TestModel, sample, iterations=1)
        
        with patch('time.time', side_effect=[0, 0.1] * 5), patch('builtins.print'):
            results = experiment.run()
        
        assert len(results['model']) > 0
        assert results['info']['model_type'] == 'TestModel'
    
    @pytest.mark.slow
    def test_experiment_performance(self):
        """Test experiment performance with multiple runs."""
        class FastModel(am.Model):
            def setup(self):
                for i in range(10):
                    self.add_agent(Mock(id=i))
            
            def step(self):
                self.record_model('counter', self.t)
        
        parameters = {
            'param1': IntRange(1, 100),
            'param2': [0.1, 0.2],
            'steps': 5
        }
        sample = Sample(parameters, n=10)
        experiment = Experiment(FastModel, sample, iterations=3)
        
        import time
        start_time = time.time()
        
        with patch('builtins.print'):
            results = experiment.run()
        
        end_time = time.time()
        
        # Should complete in reasonable time (30 runs total)
        assert end_time - start_time < 60  # 60 seconds max
        
        # Should have correct number of runs
        expected_runs = 10 * 3  # sample size * iterations
        unique_runs = len(results['model'].select(['param1', 'param2', 'iteration']).unique())
        assert unique_runs == expected_runs 