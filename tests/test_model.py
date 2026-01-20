"""
Tests for amber.model module.
"""

import pytest
import polars as pl
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time
from datetime import datetime
import amber as am
from amber.model import Model


class TestModel:
    """Test cases for Model class."""
    
    def test_model_initialization(self, sample_parameters):
        """Test Model initialization."""
        model = Model(sample_parameters)
        
        assert model.p == sample_parameters
        assert model.t == 0
        assert hasattr(model, 'agents_df')
        assert hasattr(model, '_model_data')
        assert hasattr(model, 'random')
        assert hasattr(model, 'nprandom')
    
    def test_model_initialization_with_seed(self):
        """Test Model initialization with seed."""
        params = {'seed': 42, 'steps': 10}
        model = Model(params)
        
        # Test reproducibility
        val1 = model.random.random()
        np_val1 = model.nprandom.random()
        
        # Create new model with same seed
        model2 = Model(params)
        val2 = model2.random.random()
        np_val2 = model2.nprandom.random()
        
        assert val1 == val2
        assert np_val1 == np_val2
    
    def test_model_initialization_without_seed(self):
        """Test Model initialization without seed."""
        params = {'steps': 10}
        model = Model(params)
        
        # Should still have random generators
        assert hasattr(model, 'random')
        assert hasattr(model, 'nprandom')
        
        # Should be able to generate random numbers
        val = model.random.random()
        assert 0 <= val <= 1
    
    def test_model_dataframe_initialization(self):
        """Test that model DataFrame is properly initialized."""
        params = {'steps': 10}
        model = Model(params)
        
        assert isinstance(model.agents_df, pl.DataFrame)
        expected_columns = ['id', 'step', 'wealth']
        assert model.agents_df.columns == expected_columns
        assert len(model.agents_df) == 0  # Should start empty
    
    def test_setup_method_default(self):
        """Test default setup method."""
        params = {'steps': 10}
        model = Model(params)
        
        # Default setup should not raise errors
        result = model.setup()
        assert result is None
    
    def test_step_method_default(self):
        """Test default step method."""
        params = {'steps': 10}
        model = Model(params)
        
        # Default step should not raise errors
        result = model.step()
        assert result is None
    
    def test_update_method(self):
        """Test update method."""
        params = {'steps': 10}
        model = Model(params)
        
        initial_t = model.t
        model.update()
        
        assert model.t == initial_t + 1
        assert hasattr(model, '_current_step_data')
        assert model._current_step_data['t'] == model.t
    
    def test_end_method_default(self):
        """Test default end method."""
        params = {'steps': 10}
        model = Model(params)
        
        # Default end should not raise errors
        result = model.end()
        assert result is None
    
    def test_record_model(self):
        """Test recording model-level data."""
        params = {'steps': 10}
        model = Model(params)
        model.update()  # Initialize current step data
        
        model.record_model('test_metric', 42)
        model.record_model('another_metric', 'test_value')
        
        assert model._current_step_data['test_metric'] == 42
        assert model._current_step_data['another_metric'] == 'test_value'
    
    def test_record_model_without_update(self):
        """Test recording model data without calling update first."""
        params = {'steps': 10}
        model = Model(params)
        
        # Should not raise error, but won't record anything
        model.record_model('test_metric', 42)
        # Should not have current step data
        assert not hasattr(model, '_current_step_data')
    
    def test_finalize_step_data(self):
        """Test finalizing step data."""
        params = {'steps': 10}
        model = Model(params)
        model.update()
        model.record_model('test_value', 123)
        
        initial_length = len(model._model_data)
        model._finalize_step_data()
        
        assert len(model._model_data) == initial_length + 1
        assert model._model_data[-1]['test_value'] == 123
        assert model._model_data[-1]['t'] == model.t
    
    @patch('time.time')
    def test_print_progress(self, mock_time):
        """Test progress printing functionality."""
        mock_time.side_effect = [0, 1, 2, 3]  # Simulate time progression
        
        params = {'steps': 10}
        model = Model(params)
        model._start_time = 0
        model._last_progress_time = None
        
        # Test progress printing (should work without errors)
        with patch('builtins.print') as mock_print:
            model._print_progress(5, 10, force=True)
            mock_print.assert_called()
    
    @patch('time.time')
    @patch('builtins.print')
    def test_run_method_basic(self, mock_print, mock_time):
        """Test basic run method functionality."""
        mock_time.side_effect = [0, 0.1]  # Only need start and end time for show_progress=False
        
        class TestModel(Model):
            def setup(self):
                self.setup_called = True
            
            def step(self):
                self.record_model('step_count', self.t)
            
            def end(self):
                self.end_called = True
        
        params = {'steps': 3, 'show_progress': False}  # Disable progress reporting
        model = TestModel(params)
        
        results = model.run()
        
        # Check that methods were called
        assert hasattr(model, 'setup_called')
        assert hasattr(model, 'end_called')
        
        # Check results structure
        assert 'info' in results
        assert 'agents' in results
        assert 'model' in results
        
        # Check info
        assert results['info']['model_type'] == 'TestModel'
        assert results['info']['steps'] == 3
        
        # Check model data
        assert len(model._model_data) > 0
    
    def test_run_method_with_custom_steps(self):
        """Test run method with custom step count."""
        class TestModel(Model):
            def step(self):
                self.record_model('counter', self.t)
        
        params = {'steps': 10, 'show_progress': False}  # Default steps, no progress
        model = TestModel(params)
        
        # Run with fewer steps
        with patch('time.time', side_effect=[0, 0.1]), patch('builtins.print'):
            results = model.run(steps=5)
        
        assert model.t == 5
        assert results['info']['steps'] == 5
    
    def test_get_agent_data(self):
        """Test getting agent data."""
        params = {'steps': 10}
        model = Model(params)
        
        # Add some test data
        test_data = pl.DataFrame({
            'id': [1, 1, 2, 2],
            'step': [0, 1, 0, 1],
            'wealth': [10, 12, 15, 13]
        })
        model.agents_df = test_data
        
        agent_1_data = model.get_agent_data(1)
        
        assert len(agent_1_data) == 2
        assert agent_1_data['id'].to_list() == [1, 1]
        assert agent_1_data['wealth'].to_list() == [10, 12]
    
    def test_get_agents_by_condition(self):
        """Test getting agents by condition."""
        params = {'steps': 10}
        model = Model(params)
        
        # Add test data
        test_data = pl.DataFrame({
            'id': [1, 2, 3, 4],
            'step': [1, 1, 1, 1],
            'wealth': [10, 25, 15, 30]
        })
        model.agents_df = test_data
        
        # Get wealthy agents
        wealthy = model.get_agents_by_condition(pl.col('wealth') > 20)
        
        assert len(wealthy) == 2
        assert wealthy['id'].to_list() == [2, 4]
        assert wealthy['wealth'].to_list() == [25, 30]
    
    def test_update_agent_data(self):
        """Test updating agent data."""
        params = {'steps': 10}
        model = Model(params)
        
        # Initialize with some data
        initial_data = pl.DataFrame({
            'id': [1, 2],
            'step': [0, 0],
            'wealth': [10, 15]
        })
        model.agents_df = initial_data
        
        # Update agent data
        model.update_agent_data(1, {'wealth': 20, 'step': 1})
        
        # Check that data was updated
        agent_data = model.get_agent_data(1)
        latest = agent_data.filter(pl.col('step') == 1)
        assert len(latest) == 1
        assert latest['wealth'].item() == 20
    
    def test_add_agent(self):
        """Test adding a new agent."""
        params = {'steps': 10}
        model = Model(params)
        
        # Create mock agent
        mock_agent = Mock()
        mock_agent.id = 99
        
        model.add_agent(mock_agent)
        
        # Check that agent data was added
        agent_data = model.get_agent_data(99)
        assert len(agent_data) == 1
        assert agent_data['id'].item() == 99
        assert agent_data['step'].item() == 0  # Current step
        assert agent_data['wealth'].item() == 0  # Default wealth
    
    def test_batch_update_agents(self):
        """Test batch updating multiple agents."""
        params = {'steps': 10}
        model = Model(params)
        model.t = 1  # Set to step 1
        
        # Initialize some agents
        for i in range(3):
            mock_agent = Mock()
            mock_agent.id = i
            model.add_agent(mock_agent)
        
        # Batch update
        agent_ids = [0, 1, 2]
        data = {'wealth': 100, 'step': 1}
        model.batch_update_agents(agent_ids, data)
        
        # Check all agents were updated
        for agent_id in agent_ids:
            agent_data = model.get_agent_data(agent_id)
            latest = agent_data.filter(pl.col('step') == 1)
            assert len(latest) == 1
            assert latest['wealth'].item() == 100
    
    def test_batch_record_agents(self):
        """Test batch recording agent data."""
        params = {'steps': 10}
        model = Model(params)
        
        # Prepare batch data
        agent_data = [
            {'id': 1, 'step': 0, 'wealth': 50},
            {'id': 2, 'step': 0, 'wealth': 75},
            {'id': 3, 'step': 0, 'wealth': 25}
        ]
        
        model.batch_record_agents(agent_data)
        
        # Check that all data was recorded
        assert len(model.agents_df) == 3
        assert model.agents_df['id'].to_list() == [1, 2, 3]
        assert model.agents_df['wealth'].to_list() == [50, 75, 25]


class TestModelSubclassing:
    """Test Model subclassing functionality."""
    
    def test_model_inheritance(self):
        """Test that Model can be properly subclassed."""
        class CustomModel(Model):
            def setup(self):
                self.custom_setup_called = True
                self.agents = []
            
            def step(self):
                self.record_model('custom_step', self.t)
            
            def end(self):
                self.custom_end_called = True
        
        params = {'steps': 5}
        model = CustomModel(params)
        
        # Test that it's a Model
        assert isinstance(model, Model)
        assert isinstance(model, CustomModel)
        
        # Test custom setup
        model.setup()
        assert model.custom_setup_called is True
        
        # Test custom step
        model.update()
        model.step()
        assert model._current_step_data.get('custom_step') == 1
        
        # Test custom end
        model.end()
        assert model.custom_end_called is True


class TestModelIntegration:
    """Integration tests for Model class."""
    
    def test_full_simulation_workflow(self):
        """Test a complete simulation workflow."""
        class TestSimulation(Model):
            def setup(self):
                # Create some agents
                self.agent_count = self.p['n']
                for i in range(self.agent_count):
                    self.add_agent(Mock(id=i))
            
            def update(self):
                super().update()  # Call parent update first
                
                # Simple simulation step - record in update() where data recording works
                self.record_model('active_agents', self.agent_count)
                
                # Update some agent data
                if self.t > 1:  # t > 1 because update() increments t before we get here
                    self.batch_update_agents(
                        list(range(self.agent_count)),
                        {'wealth': (self.t - 1) * 10, 'step': self.t - 1}
                    )
                
                # Record final wealth on last step
                if self.t == self.p['steps']:
                    final_wealth = self.agents_df.filter(
                        pl.col('step') == pl.col('step').max()
                    )['wealth'].sum()
                    self.record_model('final_total_wealth', final_wealth)
            
            def end(self):
                pass  # Data recording doesn't work in end() method
        
        params = {'n': 10, 'steps': 5, 'show_progress': False}
        model = TestSimulation(params)
        
        with patch('time.time', side_effect=[0, 0.1]), patch('builtins.print'):
            results = model.run()
        
        # Check results
        assert results['info']['steps'] == 5
        assert len(results['agents']) > 0
        assert len(results['model']) > 0
        
        # Check that simulation ran correctly
        final_step_data = model._model_data[-1]
        assert 'final_total_wealth' in final_step_data
        assert final_step_data['final_total_wealth'] > 0
    
    @pytest.mark.slow
    def test_model_performance(self):
        """Test model performance with larger datasets."""
        class PerformanceModel(Model):
            def setup(self):
                # Create many agents
                for i in range(1000):
                    self.add_agent(Mock(id=i))
            
            def step(self):
                # Batch operations
                agent_data = [
                    {'id': i, 'step': self.t, 'wealth': np.random.randint(1, 100)}
                    for i in range(1000)
                ]
                self.batch_record_agents(agent_data)
        
        params = {'steps': 10, 'show_progress': False}
        model = PerformanceModel(params)
        
        start_time = time.time()
        with patch('builtins.print'):
            results = model.run()
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 30  # 30 seconds max
        assert len(results['agents']) > 0 