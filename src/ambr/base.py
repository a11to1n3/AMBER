from typing import Any, Dict, Optional, List, Type
import polars as pl
import numpy as np
import random


class AttrDict(dict):
    """
    Dictionary that allows attribute-style access to its keys.
    
    This provides AgentPy compatibility where parameters are accessed as:
        self.p.param_name instead of self.p['param_name']
    """
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")


class BaseModel:
    """Base class for all simulation models, using DataFrames for data storage."""
    def __init__(self, parameters: Dict[str, Any]):
        self.p = AttrDict(parameters)  # Wrap parameters for attribute-style access
        self.t = 0
        self.agents_df = pl.DataFrame({'id': [], 't': []})
        self.model_df = pl.DataFrame({'t': [0], **{k: [v] for k, v in parameters.items()}})
        seed = parameters.get('seed', None)
        self.random = random.Random(seed)
        self.nprandom = np.random.default_rng(seed)

class BaseAgent:
    """Base class for all agents in the simulation, using DataFrames for data storage."""
    def __init__(self, model: 'BaseModel', agent_id: int):
        self.model = model
        self.id = agent_id
        self.p = model.p if model is not None else AttrDict({})