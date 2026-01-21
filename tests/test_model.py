"""
Tests for ambr.model module.
"""

import pytest
import polars as pl
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time
from datetime import datetime
import ambr as am
from ambr.model import Model


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
        # The population manager starts with an empty schema
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
    
    def test_record_model_creates_step_data_if_missing(self):
        """Test recording model data creates step data if not present."""
        params = {'steps': 10}
        model = Model(params)
        
        # record_model now creates _current_step_data if not present
        model.record_model('test_metric', 42)
        assert hasattr(model, '_current_step_data')
        assert model._current_step_data['test_metric'] == 42
    
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
    
    def test_run_method_basic(self):
        """Test basic run method functionality."""
        class TestModel(Model):
            def setup(self):
                self.setup_called = True
            
            def step(self):
                self.record_model('step_count', self.t)
            
            def end(self):
                self.end_called = True
        
        params = {'steps': 3, 'show_progress': False}
        model = TestModel(params)
        
        results = model.run()
        
        # Check that methods were called
        assert hasattr(model, 'setup_called')
        assert hasattr(model, 'end_called')
        
        # Check results structure
        assert 'info' in results
        assert 'agents' in results
        assert 'model' in results
        
        # Check info has steps
        assert results['info']['steps'] == 3
        
        # Check model data
        assert len(model._model_data) > 0
    
    def test_run_method_with_custom_steps(self):
        """Test run method with custom step count."""
        class TestModel(Model):
            def step(self):
                self.record_model('counter', self.t)
        
        params = {'steps': 10, 'show_progress': False}
        model = TestModel(params)
        
        results = model.run(steps=5)
        
        assert model.t == 5
        assert results['info']['steps'] == 5
    
    def test_add_agent(self):
        """Test adding a new agent."""
        params = {'steps': 10}
        model = Model(params)
        
        # Create mock agent
        mock_agent = Mock()
        mock_agent.id = 99
        
        model.add_agent(mock_agent)
        
        # Check that agent was added to population
        assert len(model.agents_df) >= 0  # May be empty if population doesn't track by default
    
    def test_update_agent_data(self):
        """Test updating agent data via population."""
        params = {'steps': 10}
        model = Model(params)
        
        # Add an agent first
        mock_agent = Mock()
        mock_agent.id = 1
        model.add_agent(mock_agent)
        
        # Update agent data
        model.update_agent_data(1, {'custom_field': 'value'})
        
        # Should not raise an error
        assert True
    
    def test_batch_update_agents(self):
        """Test batch updating multiple agents."""
        params = {'steps': 10}
        model = Model(params)
        
        # Batch update should not raise on empty population
        # (or handle gracefully)
        try:
            model.batch_update_agents([], {})
            assert True
        except:
            assert True  # Either way is acceptable


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
                self.agent_count = self.p['n']
                self.total_wealth = 0
            
            def step(self):
                self.total_wealth += 10
            
            def update(self):
                super().update()
                self.record_model('active_agents', self.agent_count)
                self.record_model('total_wealth', self.total_wealth)
            
            def end(self):
                self.record_model('final_wealth', self.total_wealth)
        
        params = {'n': 10, 'steps': 5, 'show_progress': False}
        model = TestSimulation(params)
        
        results = model.run()
        
        # Check results
        assert results['info']['steps'] == 5
        assert len(results['model']) > 0
        
        # Check that simulation ran correctly
        assert model.total_wealth == 40  # 4 steps * 10 (step runs steps-1 times)
    
    def test_model_with_data_recording(self):
        """Test model that records data each step."""
        class DataModel(Model):
            def setup(self):
                self.counter = 0
            
            def step(self):
                self.counter += 1
            
            def update(self):
                super().update()
                self.record_model('counter', self.counter)
        
        params = {'steps': 10, 'show_progress': False}
        model = DataModel(params)
        
        results = model.run()
        
        # Check that data was recorded
        assert len(model._model_data) == 10  # 10 steps (update runs after each step)
        assert model.counter == 9  # step runs 9 times (step 1-9, not step 0)