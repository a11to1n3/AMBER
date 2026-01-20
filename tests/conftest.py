"""
Pytest configuration and common fixtures for AMBER tests.
"""

import pytest
import polars as pl
import numpy as np
from unittest.mock import Mock, MagicMock
import sys
import os

# Add src to path so we can import amber modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import amber as am


@pytest.fixture
def sample_parameters():
    """Basic parameters for testing models."""
    return {
        'steps': 10,
        'n': 100,
        'seed': 42
    }


@pytest.fixture
def basic_model(sample_parameters):
    """Create a basic model instance for testing."""
    
    class TestModel(am.Model):
        def setup(self):
            self.test_agents = {}
            for i in range(5):
                agent = am.Agent(self, i)
                self.test_agents[i] = agent
        
        def step(self):
            self.record_model('step_count', self.t)
    
    return TestModel(sample_parameters)


@pytest.fixture
def basic_agent():
    """Create a basic agent instance for testing."""
    mock_model = Mock()
    mock_model.p = {'test_param': 'test_value'}
    agent = am.Agent(mock_model, 42)
    return agent


@pytest.fixture
def sample_dataframe():
    """Create a sample polars DataFrame for testing."""
    return pl.DataFrame({
        'id': [0, 1, 2, 3, 4],
        'step': [0, 0, 0, 0, 0],
        'value': [1.0, 2.0, 3.0, 4.0, 5.0]
    })


@pytest.fixture
def int_range():
    """Create an IntRange for testing."""
    return am.IntRange(10, 50)


@pytest.fixture
def parameter_space():
    """Create a ParameterSpace for testing."""
    return am.ParameterSpace({
        'param1': am.IntRange(1, 10),
        'param2': [0.1, 0.2, 0.3],
        'param3': 'fixed_value'
    })


@pytest.fixture
def sample_with_ranges():
    """Create a Sample with parameter ranges."""
    return am.Sample({
        'n': am.IntRange(10, 100),
        'rate': [0.1, 0.2],
        'steps': 5
    }, n=10)


@pytest.fixture
def mock_networkx_graph():
    """Create a mock NetworkX graph for testing."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4])
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    return G


@pytest.fixture(scope="session")
def temp_output_dir(tmp_path_factory):
    """Create a temporary directory for test outputs."""
    return tmp_path_factory.mktemp("amber_test_outputs")


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location."""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit) 