# AMBER Test Suite

This directory contains comprehensive tests for the AMBER agent-based modeling package.

## Test Structure

The test suite is organized into several categories:

### Unit Tests
- `test_base.py` - Tests for the BaseModel class
- `test_agent.py` - Tests for the Agent class  
- `test_model.py` - Tests for the Model class
- `test_experiment.py` - Tests for experiment framework (IntRange, Sample, Experiment)
- `test_sequences.py` - Tests for AgentList class
- `test_environments.py` - Tests for environment classes (Grid, Space, Network)
- `test_optimization.py` - Tests for optimization functions and ParameterSpace
- `test_init.py` - Tests for package initialization and exports

### Integration Tests  
- `test_integration.py` - Tests combining multiple components and real-world workflows

### Configuration
- `conftest.py` - Pytest configuration and shared fixtures
- `pytest.ini` - Pytest settings and markers

## Running Tests

### Prerequisites
Install test dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start
```bash
# Run all tests
make test

# Run fast tests only (exclude slow tests)
make test-fast

# Run with coverage report
make test-coverage
```

### Using pytest directly
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run specific test class
pytest tests/test_model.py::TestModel

# Run specific test method
pytest tests/test_model.py::TestModel::test_model_initialization

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src/amber --cov-report=html

# Run only unit tests
pytest -m unit

# Run only integration tests  
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Categories

### Unit Tests (marked with `@pytest.mark.unit`)
Fast tests that test individual components in isolation. These should run in under 1 second each.

### Integration Tests (marked with `@pytest.mark.integration`)
Tests that verify multiple components working together. May take longer to run.

### Slow Tests (marked with `@pytest.mark.slow`)
Performance tests, large-scale simulations, or optimization tests. These may take several seconds or minutes.

## Test Coverage

The test suite aims for high coverage across all modules:

- **BaseModel**: Parameter handling, inheritance
- **Agent**: Initialization, attributes, model integration
- **Model**: Core simulation loop, data recording, agent management
- **Experiment Framework**: Parameter sampling, experiment execution
- **Environments**: Grid, continuous space, and network environments
- **Optimization**: Parameter spaces, grid search, random search, Bayesian optimization
- **Sequences**: AgentList functionality and integration
- **Package Integration**: Import structure, workflow testing

## Writing New Tests

### Guidelines
1. **Naming**: Test files should start with `test_`, classes with `Test`, methods with `test_`
2. **Structure**: Group related tests in classes, use descriptive method names
3. **Isolation**: Each test should be independent and not rely on other tests
4. **Mocking**: Use mocks for external dependencies (time, random, etc.)
5. **Fixtures**: Use pytest fixtures for common setup/teardown
6. **Markers**: Add appropriate markers (@pytest.mark.unit, @pytest.mark.slow, etc.)

### Example Test Structure
```python
import pytest
from unittest.mock import Mock, patch
import ambr as am

class TestMyComponent:
    """Test cases for MyComponent class."""
    
    def test_basic_functionality(self):
        """Test basic component functionality."""
        # Arrange
        component = am.MyComponent()
        
        # Act  
        result = component.do_something()
        
        # Assert
        assert result is not None
    
    @pytest.mark.slow
    def test_performance(self):
        """Test component performance with large inputs."""
        # Performance test code here
        pass
```

### Common Fixtures Available
- `sample_parameters`: Basic parameter dictionary
- `basic_model`: Simple model instance for testing
- `basic_agent`: Simple agent instance for testing
- `sample_dataframe`: Polars DataFrame for testing
- `mock_networkx_graph`: NetworkX graph for testing
- `temp_output_dir`: Temporary directory for test outputs

## Continuous Integration

Tests are designed to run in CI environments:
- All external dependencies are mocked or made optional
- Random seeds are set for reproducibility
- Timeouts prevent hanging tests
- Coverage reports are generated

## Troubleshooting

### Common Issues

**Import Errors**: Make sure AMBER is installed in development mode:
```bash
pip install -e .
```

**Random Test Failures**: Tests use fixed seeds, but some randomness may leak through. Check for unmocked random number generation.

**Slow Performance**: Use `pytest -m "not slow"` to skip performance tests during development.

**Coverage Issues**: Some modules may have coverage issues due to:
- Platform-specific code paths
- Error handling for rare edge cases
- Optional dependencies

### Debugging Tests
```bash
# Run with debug output
pytest -vvv -s

# Stop on first failure
pytest -x

# Run specific failing test with debug
pytest tests/test_model.py::TestModel::test_failing_method -vvv -s
```

## Contributing

When adding new features to AMBER:

1. **Write tests first** (TDD approach recommended)
2. **Ensure high coverage** of new code
3. **Add integration tests** for new workflows
4. **Update fixtures** if needed for new test patterns
5. **Mark tests appropriately** (unit/integration/slow)
6. **Document complex test scenarios** in docstrings

For questions about testing, see the main AMBER documentation or raise an issue in the repository. 