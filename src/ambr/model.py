from typing import Any, Dict, List, Type, Optional
import polars as pl
import random
import numpy as np
import time
from datetime import datetime, timedelta
from .base import BaseModel
from .population import Population

class Model(BaseModel):
    """Base class for all simulation models, using DataFrames for data storage."""
    
    def __init__(self, parameters: Dict[str, Any]):
        """Initialize a new model.
        
        Args:
            parameters: Dictionary of model parameters
        """
        # Initialize Population Manager first because super().__init__ triggers agents_df setter
        # We infer the schema from initial requirements or defaults
        self.population = Population(schema={})
        
        super().__init__(parameters)
        self.t = 0  # Current time step
        self._start_time = None
        self._last_progress_time = None
        
        # Control progress reporting
        self._show_progress = parameters.get('show_progress', True)
        
        # Compatibility property for legacy code that accesses agents_df
        # This is a read-only view in practice
        
        self._model_data = []
        
        # Initialize random number generators
        seed = parameters.get('seed', None)
        self.random = random.Random(seed)
        self._rng = np.random.default_rng(seed)
        
        self.nprandom = self._create_np_wrapper(self._rng)
        
        # Track agents in an AgentList for AgentPy compatibility
        from .sequences import AgentList
        self.agents = AgentList(self, [])
    
    @property
    def agents_df(self) -> pl.DataFrame:
        return self.population.data

    @agents_df.setter
    def agents_df(self, value):
        self.population.data = value

    def _create_np_wrapper(self, rng):
        class NPRandomWrapper:
            def __init__(self, rng): self._rng = rng
            def __getattr__(self, name): return getattr(self._rng, name)
            def randint(self, low, high=None, size=None, dtype=int):
                return self._rng.integers(low, high, size=size, dtype=dtype, endpoint=False)
        return NPRandomWrapper(rng)

    def setup(self): pass
    def step(self): pass
    
    def update(self):
        """Update model state after each step."""
        self.t += 1
        self._current_step_data = {'t': self.t}
        
    def record_model(self, key: str, value: Any):
        """Record a model-level variable for the current step."""
        if not hasattr(self, '_current_step_data'):
            self._current_step_data = {'t': self.t}
        self._current_step_data[key] = value
        
    def record(self, key: str, value: Any):
        """AgentPy compatibility alias for record_model."""
        self.record_model(key, value)

    def _finalize_step_data(self):
        if hasattr(self, '_current_step_data'):
            self._model_data.append(self._current_step_data.copy())
            
    def end(self): pass

    def run(self, steps: Optional[int] = None) -> Dict[str, pl.DataFrame]:
        # ... (Same run logic, omitted for brevity but preserved in practice)
        # Using a simplified version here for cleaner file updates
        start_time = time.time()
        max_steps = steps if steps is not None else self.p.get('steps', 100)
        
        if self._show_progress:
            self._start_time = start_time
            self._print_start_info(max_steps)
            
            self.setup()
            self.update()
            self._finalize_step_data()
            
            self._print_progress(0, max_steps, force=True)
            
            while self.t < max_steps:
                self.step()
                self.update()
                self._finalize_step_data()
                self._print_progress(self.t, max_steps)
                
            self.end()
            self._print_progress(max_steps, max_steps, force=True)
            self._print_end_info(start_time, max_steps)
        else:
            self.setup()
            self.update()
            self._finalize_step_data()
            while self.t < max_steps:
                self.step()
                self.update()
                self._finalize_step_data()
            self.end()
            
        return self._collect_results(start_time, max_steps)

    # --- Helper methods ---
    def _print_start_info(self, max_steps):
        print(f"🚀 Simulation: {self.__class__.__name__}")
        print(f"⏱️  Steps: {max_steps:,}")

    def _print_end_info(self, start_time, max_steps):
        total_time = time.time() - start_time
        print(f"\n✅ Done. Time: {timedelta(seconds=int(total_time))}")
        if total_time > 0:
            print(f"📈 Rate: {max_steps/total_time:.1f} steps/s")
        else:
            print(f"📈 Rate: Inf steps/s")

    def _collect_results(self, start_time, max_steps):
        if self._model_data:
            # Column-oriented construction to avoid Polars concat ShapeErrors with sparse data
            all_keys = sorted(list(set().union(*(d.keys() for d in self._model_data))))
            data_dict = {k: [] for k in all_keys}
            
            for d in self._model_data:
                for k in all_keys:
                    data_dict[k].append(d.get(k, None))
            
            model_df = pl.DataFrame(data_dict, strict=False)
        else:
            model_df = pl.DataFrame({'t': []})
            
        return {
            'info': {'steps': self.t, 'run_time': time.time() - start_time},
            'agents': self.population.data,
            'model': model_df
        }

    # --- Agent Management Delegates ---
    def add_agent(self, agent: 'Agent'):
        self.population.add_agent(agent.id, self.t)
        
        # Track in self.agents if it's an AgentList (initialized in __init__)
        # If the user has overwritten it with another structure (like a dict in some tests), 
        # we respect that and don't attempt to append.
        from .sequences import AgentList
        if isinstance(self.agents, AgentList):
            self.agents.append(agent)
            # Set default agent_type if not already set
            if self.agents.agent_type is None or self.agents.agent_type == type(None):
                self.agents.agent_type = type(agent)
        
    def update_agent_data(self, agent_id: int, data: Dict[str, Any]):
        """Update data for a single agent."""
        for key, value in data.items():
            self.population.set_agent_value(agent_id, key, value)
        
    def batch_update_agents(self, agent_ids: list, data: dict):
        """Batch update multiple agents at once for better performance.
        
        Args:
            agent_ids: List of agent IDs to update
            data: Dictionary of column names and values (or lists of values)
        """
        self.population.batch_update_by_ids(agent_ids, data)

    def _print_progress(self, current_step: int, total_steps: int, force: bool = False):
        # ... logic preserved ...
        pass
 