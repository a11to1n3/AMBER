"""
AMBER: Agent-Based Modeling Environment and Research Framework

A comprehensive Python framework for building, running, and analyzing agent-based models.
AMBER provides tools for creating complex simulations with agents, environments, and 
sophisticated experimental workflows.

Key features:
- Flexible agent and model architecture
- Built-in environments (grid, space, network)
- Experiment management and parameter sweeping
- Optimization tools for model calibration
- High-performance data handling with Polars

Example:
    >>> import ambr as am
    >>> 
    >>> class SimpleModel(am.Model):
    ...     def setup(self):
    ...         for i in range(10):
    ...             agent = am.Agent(self, i)
    ...             self.add_agent(agent)
    ...     
    ...     def step(self):
    ...         self.record_model('agent_count', len(self.agents))
    >>> 
    >>> model = SimpleModel({'steps': 5})
    >>> results = model.run()
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _metadata_version

from .agent import Agent
from .model import Model
from .population import Population, BatchUpdateContext
from .contract import (
    ContractCertificate,
    ContractViolation,
    ContractViolationError,
    ContractMonitor,
    CONTRACT_MODES,
)
from .base import BaseModel, BaseAgent
from .sequences import AgentList
from .environments import GridEnvironment, SpaceEnvironment, NetworkEnvironment
from .experiment import Experiment, Sample, IntRange
from .optimization import (
    ParameterSpace, 
    objective_function,
    grid_search,
    random_search,
    bayesian_optimization,
    SMACOptimizer,
    MultiObjectiveSMAC,
    SMACParameterSpace
)
from .performance import (
    SpatialIndex,
    ParallelRunner,
    vectorized_move,
    vectorized_wealth_transfer,
    vectorized_random_velocities,
    check_performance_deps,
    HAS_SCIPY,
    HAS_NUMBA,
    jit as numba_jit,
)
from .tensor_lane import TensorLane, borrow_numeric, commit_columns
from .gpu import GPU_AVAILABLE, get_array_module, to_device, to_host, require_gpu, synchronize
from .results import RunResults
from .lanes import ArrayKernelModel, status, print_status, recommend

try:
    __version__ = _metadata_version('ambr')
except _PackageNotFoundError:
    __version__ = '0.4.1'

__author__ = 'a11to1n3'
__email__ = 'citation.needed@example.com'
__url__ = 'https://github.com/a11to1n3/AMBER'

__all__ = [
    'Agent',
    'Model',
    'Population',
    'BatchUpdateContext',
    'ContractCertificate',
    'ContractViolation',
    'ContractViolationError',
    'ContractMonitor',
    'CONTRACT_MODES',
    'BaseModel',
    'BaseAgent',
    'AgentList',
    'GridEnvironment',
    'SpaceEnvironment',
    'NetworkEnvironment',
    'Experiment',
    'Sample',
    'IntRange',
    'ParameterSpace',
    'objective_function',
    'grid_search',
    'random_search',
    'bayesian_optimization',
    'SMACOptimizer',
    'MultiObjectiveSMAC',
    'SMACParameterSpace',
    # Performance utilities
    'SpatialIndex',
    'ParallelRunner',
    'vectorized_move',
    'vectorized_wealth_transfer',
    'vectorized_random_velocities',
    'check_performance_deps',
    'HAS_SCIPY',
    'HAS_NUMBA',
    'numba_jit',
    # Tensor lane / GPU (0.4)
    'TensorLane',
    'borrow_numeric',
    'commit_columns',
    'GPU_AVAILABLE',
    'get_array_module',
    'to_device',
    'to_host',
    'require_gpu',
    'synchronize',
    'RunResults',
    # Progressive speed lanes
    'ArrayKernelModel',
    'status',
    'print_status',
    'recommend',
]
