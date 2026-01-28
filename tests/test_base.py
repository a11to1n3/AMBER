"""
Tests for ambr.base module.
"""

import pytest
from unittest.mock import Mock, patch
import ambr as am
from ambr.base import BaseModel


class TestBaseModel:
    """Test cases for BaseModel class."""
    
    def test_init_with_parameters(self):
        """Test BaseModel initialization with parameters."""
        params = {'param1': 'value1', 'param2': 42}
        model = BaseModel(params)
        
        assert model.p == params
        assert model.p['param1'] == 'value1'
        assert model.p['param2'] == 42
    
    def test_init_empty_parameters(self):
        """Test BaseModel initialization with empty parameters."""
        model = BaseModel({})
        assert model.p == {}
    
    def test_parameter_access(self):
        """Test parameter access methods."""
        params = {'test_key': 'test_value', 'number': 123}
        model = BaseModel(params)
        
        # Test dictionary-style access
        assert model.p['test_key'] == 'test_value'
        assert model.p['number'] == 123
        
        # Test get method with default
        assert model.p.get('nonexistent', 'default') == 'default'
        assert model.p.get('test_key', 'default') == 'test_value'
    
    def test_parameter_modification(self):
        """Test that parameters can be modified after initialization."""
        params = {'initial': 'value'}
        model = BaseModel(params)
        
        # Modify existing parameter
        model.p['initial'] = 'modified'
        assert model.p['initial'] == 'modified'
        
        # Add new parameter
        model.p['new_param'] = 'new_value'
        assert model.p['new_param'] == 'new_value'
    
    def test_parameter_types(self):
        """Test that various parameter types are handled correctly."""
        params = {
            'string': 'text',
            'integer': 42,
            'float': 3.14,
            'boolean': True,
            'list': [1, 2, 3],
            'dict': {'nested': 'value'},
            'none': None
        }
        model = BaseModel(params)
        
        assert model.p['string'] == 'text'
        assert model.p['integer'] == 42
        assert model.p['float'] == 3.14
        assert model.p['boolean'] is True
        assert model.p['list'] == [1, 2, 3]
        assert model.p['dict'] == {'nested': 'value'}
        assert model.p['none'] is None
    
    def test_parameters_reference_behavior(self):
        """Test parameter reference behavior with AttrDict wrapper."""
        original_params = {'key': 'original'}
        model = BaseModel(original_params)
        
        # BaseModel now wraps parameters in AttrDict for attribute-style access
        # The wrapper contains the same values but is a new object
        assert model.p == original_params  # Values are equal
        assert model.p['key'] == 'original'  # Dict-style access works
        assert model.p.key == 'original'  # Attribute-style access works (new feature)
        
        # Modifying through model updates the AttrDict
        model.p['key'] = 'modified'
        assert model.p['key'] == 'modified'
        assert model.p.key == 'modified'  # Also works via attribute access
    
    def test_repr_string(self):
        """Test string representation of BaseModel."""
        params = {'param1': 'value1', 'param2': 42}
        model = BaseModel(params)
        
        repr_str = repr(model)
        # BaseModel uses default object repr - just check it exists and contains class name
        assert 'BaseModel' in repr_str
        assert 'object at' in repr_str 