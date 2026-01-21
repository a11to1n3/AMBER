"""
Tests for ambr.__init__ module.
"""

import pytest
import ambr as am


class TestPackageInitialization:
    """Test cases for package initialization and exports."""
    
    def test_package_import(self):
        """Test that amber package can be imported."""
        # This test passes if the import above works
        assert am is not None
    
    def test_model_export(self):
        """Test that Model class is properly exported."""
        assert hasattr(am, 'Model')
        assert am.Model is not None
        
        # Test that we can create a Model instance
        model = am.Model({'test': 'param'})
        assert model is not None
        assert model.p['test'] == 'param'
    
    def test_agent_export(self):
        """Test that Agent class is properly exported."""
        assert hasattr(am, 'Agent')
        assert am.Agent is not None
        
        # Test that we can create an Agent instance
        from unittest.mock import Mock
        mock_model = Mock()
        agent = am.Agent(mock_model, 1)
        assert agent is not None
        assert agent.id == 1
    
    def test_base_model_export(self):
        """Test that BaseModel class is properly exported."""
        assert hasattr(am, 'BaseModel')
        assert am.BaseModel is not None
        
        # Test that we can create a BaseModel instance
        base = am.BaseModel({'param': 'value'})
        assert base is not None
        assert base.p['param'] == 'value'
    
    def test_experiment_classes_export(self):
        """Test that experiment classes are properly exported."""
        # Test IntRange
        assert hasattr(am, 'IntRange')
        assert am.IntRange is not None
        
        int_range = am.IntRange(1, 10)
        assert int_range.start == 1
        assert int_range.end == 10
        
        # Test Sample
        assert hasattr(am, 'Sample')
        assert am.Sample is not None
        
        sample = am.Sample({'param': 'value'}, n=1)
        assert sample.n == 1
        assert len(sample.combinations) == 1
        
        # Test Experiment
        assert hasattr(am, 'Experiment')
        assert am.Experiment is not None
    
    def test_sequences_export(self):
        """Test that sequences classes are properly exported."""
        assert hasattr(am, 'AgentList')
        assert am.AgentList is not None
        
        # Test that we can create an AgentList
        from unittest.mock import Mock
        mock_model = Mock()
        agent_list = am.AgentList(mock_model, [])
        assert agent_list is not None
        assert len(agent_list) == 0
    
    def test_environments_export(self):
        """Test that environment classes are properly exported."""
        # Test GridEnvironment
        assert hasattr(am, 'GridEnvironment')
        assert am.GridEnvironment is not None
        
        # Test SpaceEnvironment
        assert hasattr(am, 'SpaceEnvironment')
        assert am.SpaceEnvironment is not None
        
        # Test NetworkEnvironment
        assert hasattr(am, 'NetworkEnvironment')
        assert am.NetworkEnvironment is not None
    
    def test_optimization_export(self):
        """Test that optimization classes are properly exported."""
        # Test ParameterSpace
        assert hasattr(am, 'ParameterSpace')
        assert am.ParameterSpace is not None
        
        space = am.ParameterSpace({'param': [1, 2, 3]})
        assert space is not None
        
        # Test optimization functions
        assert hasattr(am, 'objective_function')
        assert hasattr(am, 'grid_search')
        assert hasattr(am, 'random_search')
        assert hasattr(am, 'bayesian_optimization')
    
    def test_all_exports_exist(self):
        """Test that all expected exports exist."""
        expected_exports = [
            # Core classes
            'Model', 'Agent', 'BaseModel',
            
            # Experiment classes
            'IntRange', 'Sample', 'Experiment',
            
            # Sequences
            'AgentList',
            
            # Environments
            'GridEnvironment', 'SpaceEnvironment', 'NetworkEnvironment',
            
            # Optimization
            'ParameterSpace', 'objective_function', 'grid_search', 
            'random_search', 'bayesian_optimization'
        ]
        
        for export in expected_exports:
            assert hasattr(am, export), f"Expected export '{export}' not found"
    
    def test_no_unwanted_exports(self):
        """Test that private/internal modules are not exported."""
        # Get all public attributes (not starting with _)
        public_attrs = [name for name in dir(am) if not name.startswith('_')]
        
        # Define expected public exports
        expected_public = {
            'Model', 'Agent', 'BaseModel', 'BaseAgent',
            'IntRange', 'Sample', 'Experiment',
            'AgentList',
            'GridEnvironment', 'SpaceEnvironment', 'NetworkEnvironment',
            'ParameterSpace', 'objective_function', 'grid_search', 
            'random_search', 'bayesian_optimization',
            'SMACOptimizer', 'MultiObjectiveSMAC', 'SMACParameterSpace',
            # New exports
            'Population', 'BatchUpdateContext', 'SpatialIndex', 'ParallelRunner',
            'check_performance_deps', 'vectorized_move', 'vectorized_wealth_transfer',
            'vectorized_random_velocities', 'HAS_SCIPY', 'HAS_NUMBA',
            'HAS_SMAC'
        }
        
        # Check that we don't have unexpected exports
        unexpected = set(public_attrs) - expected_public
        
        # Filter out standard module attributes that are okay
        allowed_extras = {'__version__', '__author__', '__email__', '__url__'}
        unexpected = unexpected - allowed_extras
        
        assert len(unexpected) == 0, f"Unexpected public exports found: {unexpected}"
    
    def test_class_inheritance(self):
        """Test that classes have proper inheritance."""
        # Model should inherit from BaseModel
        assert issubclass(am.Model, am.BaseModel)
        
        # Test that instances work correctly
        model = am.Model({'test': 'value'})
        assert isinstance(model, am.BaseModel)
        assert isinstance(model, am.Model)
    
    def test_module_docstring(self):
        """Test that the module has a docstring."""
        assert am.__doc__ is not None
        assert len(am.__doc__.strip()) > 0
    
    def test_version_info(self):
        """Test that version information is available if defined."""
        # This is optional - not all packages define version in __init__
        if hasattr(am, '__version__'):
            assert isinstance(am.__version__, str)
            assert len(am.__version__) > 0


class TestPackageIntegration:
    """Integration tests for package-level functionality."""
    
    def test_complete_workflow_through_package(self):
        """Test a complete workflow using package-level imports."""
        # Create a simple model using package imports
        class TestModel(am.Model):
            def setup(self):
                self.agents = []
                for i in range(3):
                    agent = am.Agent(self, i)
                    self.agents.append(agent)
                
                # Store in AgentList
                self.agent_list = am.AgentList(self, self.agents)
            
            def step(self):
                self.record_model('agent_count', len(self.agent_list))
                
                # Test AgentList functionality
                for agent in self.agent_list:
                    # Each agent exists
                    assert agent.id in [0, 1, 2]
        
        # Create and run model
        parameters = {'steps': 2}
        model = TestModel(parameters)
        
        # Test that setup works
        model.setup()
        assert len(model.agent_list) == 3
        
        # Test step
        model.update()
        model.step()
        
        # Verify model state
        assert model.t == 1
    
    def test_experiment_workflow_through_package(self):
        """Test experiment workflow using package imports."""
        class SimpleModel(am.Model):
            def step(self):
                result = self.p['multiplier'] * 2
                self.record_model('result', result)
        
        # Create parameter sample
        sample = am.Sample({
            'multiplier': [1, 2, 3],
            'steps': 1
        }, n=3)
        
        # Create experiment
        experiment = am.Experiment(SimpleModel, sample, iterations=1)
        
        # Run experiment
        from unittest.mock import patch
        with patch('time.time', side_effect=[0, 0.1] * 10), patch('builtins.print'):
            results = experiment.run()
        
        # Verify results
        assert 'model' in results
        assert 'agents' in results
        assert 'parameters' in results
        assert len(results['parameters']) == 3
    
    def test_optimization_workflow_through_package(self):
        """Test optimization workflow using package imports."""
        class OptimizationModel(am.Model):
            def update(self):
                super().update()  # Call parent update first
                if self.t > 0:  # Only record after actual steps
                    # Simple objective function
                    x = self.p['x']
                    objective = -(x - 3) ** 2 + 9  # Maximum at x=3
                    self.record_model('objective', objective)
        
        # Create parameter space
        space = am.ParameterSpace({
            'x': am.IntRange(0, 6),
            'steps': 2  # Need at least 2 steps for data recording to work
        })
        
        # Test grid search
        from unittest.mock import patch
        with patch('time.time', side_effect=[0, 0.1] * 10), patch('builtins.print'):
            results = am.grid_search(OptimizationModel, space, 'objective')
        
        # Should find optimum at x=3 - check that the best result has highest objective
        best = results[0]
        assert best['objective'] == 9, f"Expected objective 9, got {best['objective']}"
        assert best['parameters']['x'] == 3
    
    def test_environment_creation_through_package(self):
        """Test environment creation using package imports."""
        from unittest.mock import Mock
        import polars as pl
        
        # Mock model
        mock_model = Mock()
        mock_model.agents_df = pl.DataFrame()
        mock_model.nprandom = Mock()
        
        # Test GridEnvironment
        grid = am.GridEnvironment(mock_model, size=(5, 5))
        assert grid.width == 5
        assert grid.height == 5
        
        # Test SpaceEnvironment
        space = am.SpaceEnvironment(mock_model, bounds=[(0, 10), (0, 10)])
        assert space.dimensions == 2
        
        # Test NetworkEnvironment
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        network = am.NetworkEnvironment(mock_model, G)
        assert len(network.nodes) == 3 