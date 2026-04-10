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
        # Population must exist before super().__init__ (which triggers the
        # agents_df setter).
        self.population = Population(schema={})

        # {column: {agent_id: value}} buffer flushed on the next agents_df read.
        self._pending_writes: Dict[str, Dict[Any, Any]] = {}
        # Monotonic counter bumped whenever the id column changes. Views use
        # it to invalidate cached id→row-position lookups.
        self._id_version: int = 0

        super().__init__(parameters)
        self.t = 0
        self._start_time = None
        self._last_progress_time = None
        self._show_progress = parameters.get('show_progress', True)
        self._model_data = []

        seed = parameters.get('seed', None)
        self.random = random.Random(seed)
        self._rng = np.random.default_rng(seed)
        self.nprandom = self._create_np_wrapper(self._rng)

        from .sequences import AgentList
        self.agents = AgentList(self, [])
    
    @property
    def agents_df(self) -> pl.DataFrame:
        self._flush_pending_writes()
        return self.population.data

    @agents_df.setter
    def agents_df(self, value):
        self._pending_writes = {}
        self.population.data = value
        self._bump_id_version()

    def _queue_write(self, column: str, agent_id: Any, value: Any) -> None:
        self._pending_writes.setdefault(column, {})[agent_id] = value

    def _bump_id_version(self) -> None:
        self._id_version += 1

    def _flush_pending_writes(self) -> None:
        """Apply all queued ``Agent.record()`` / ``update_data()`` writes
        as a single ``df.update(on='id')`` hash join."""
        if not self._pending_writes:
            return
        # Clear before the write so an exception can't leave a stale buffer
        # that would double-apply on the next flush.
        pending = self._pending_writes
        self._pending_writes = {}

        df = self.population.data
        missing = [c for c in pending if c not in df.columns]
        if missing:
            df = df.with_columns(
                [pl.Series(c, [None] * df.height, strict=False) for c in missing]
            )

        # df.update(on='id') skips null cells, so untouched (col, id) pairs
        # retain their current value.
        touched_ids = list({aid for col_map in pending.values() for aid in col_map})
        update_cols: Dict[str, list] = {'id': touched_ids}
        for col, id_to_val in pending.items():
            update_cols[col] = [id_to_val.get(aid, None) for aid in touched_ids]
        update_df = pl.DataFrame(
            [pl.Series(k, v, strict=False) for k, v in update_cols.items()]
        )
        self.population.data = df.update(update_df, on='id', how='left')

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

    def run_step(self) -> None:
        """Execute a single simulation step. The first call also runs ``setup``."""
        if not getattr(self, "_step_loop_started", False):
            self.setup()
            self.update()
            self._finalize_step_data()
            self._step_loop_started = True
            return
        self.step()
        self.update()
        self._finalize_step_data()

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
            
            series_list = []
            for k, v in data_dict.items():
                try:
                    s = pl.Series(k, v, strict=False)
                except (TypeError, ValueError):
                    # Fallback to Object type for columns with mixed None/Arrays which confuses Polars inference
                    try:
                        s = pl.Series(k, v, dtype=pl.Object, strict=False)
                    except Exception:
                        # thorough fallback
                        s = pl.Series(k, v, dtype=pl.Object)
                series_list.append(s)
            
            model_df = pl.DataFrame(series_list)
        else:
            model_df = pl.DataFrame({'t': []})
            
        return {
            'info': {'steps': self.t, 'run_time': time.time() - start_time},
            'agents': self.population.data,
            'model': model_df
        }

    # --- Agent Management Delegates ---
    def add_agent(self, agent: 'Agent'):
        """Add a single agent. Prefer :meth:`add_agents` for bulk creation."""
        # Forward Python attributes set on the instance (e.g. ``agent.wealth = 5``
        # before this call) into the population row.
        attributes = {
            k: v
            for k, v in vars(agent).items()
            if k not in {"model", "id", "p"} and not k.startswith("_")
        }
        self.population.add_agent(agent.id, self.t, **attributes)
        self._bump_id_version()

        from .sequences import AgentList
        if isinstance(self.agents, AgentList):
            self.agents.append(agent)
            if self.agents.agent_type is None:
                self.agents.agent_type = type(agent)

    def add_agents(
        self,
        n: int,
        *,
        agent_class: Optional[Type] = None,
        **columns: Any,
    ):
        """Bulk-create ``n`` agents with columnar initial state::

            self.add_agents(100, wealth=self.nprandom.integers(1, 10, 100),
                                 status='S')

        Scalar kwargs broadcast; list / ``np.ndarray`` / ``pl.Series`` values
        must have length ``n``. Pass ``agent_class=`` to also spin up Python
        instances so ``AgentList.call`` and per-agent iteration work.
        """
        from .sequences import AgentList

        start_id = (
            self.population.data['id'].max() + 1
            if not self.population.data.is_empty() and 'id' in self.population.data.columns
            else 0
        )
        self.population.batch_add_agents(n, step=self.t, **columns)
        self._bump_id_version()

        if not isinstance(self.agents, AgentList):
            return self.agents  # type: ignore[return-value]

        if agent_class is not None:
            for aid in range(start_id, start_id + n):
                self.agents.append(agent_class(self, aid))
            if self.agents.agent_type is None:
                self.agents.agent_type = agent_class
        return self.agents

    def get_agent_data(self, agent_id: Any) -> pl.DataFrame:
        """Return a 1-row DataFrame with the current state of ``agent_id``."""
        return self.population.data.filter(pl.col('id') == agent_id)

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
        pass
 