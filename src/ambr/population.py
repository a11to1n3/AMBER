from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING, Type
import warnings
import polars as pl
import numpy as np

from ._deprecation import warn_deprecated

if TYPE_CHECKING:
    from .agent import Agent

class Population:
    """
    Manages the columnar state of all agents using Polars DataFrames.
    Acts as the single point of truth for agent data.

    Prefer writing through the model/view API (``model.agents.col = ...``,
    ``agents.set(...)``, ``agents.commit(...)``, or ``Model._set_frame``) so
    the snapshot-view contract can observe commits. Assigning
    ``population.data = ...`` still works but is deprecated and invisible
    to ``Model.run(contract=...)``; use the view API instead.
    """
    
    def __init__(self, schema: Dict[str, Type] = None):
        if schema is None:
            schema = {}
            
        # Core columns that always exist
        self.schema = {
            'id': pl.Int64,
            'step': pl.Int64,
            **schema
        }
        
        # Backing store. Prefer Model._set_frame / view API over assigning
        # ``.data`` (the public setter warns; use ``replace_frame`` internally).
        self._data = pl.DataFrame(schema=self.schema)

        # Buffer for batch operations
        self._pending_updates: Dict[int, Dict[str, Any]] = {}

    @property
    def data(self) -> pl.DataFrame:
        """Read the agent table (single system of record)."""
        return self._data

    @data.setter
    def data(self, value: pl.DataFrame) -> None:
        """Direct assign is deprecated; prefer the view / ``_set_frame`` seam."""
        warn_deprecated(
            "assigning Population.data",
            "agents.set(**cols) / agents.col = values (or Model agents_df)",
            stacklevel=3,
        )
        self._data = value

    def replace_frame(self, df: pl.DataFrame) -> None:
        """Replace the table without deprecation (Model write path only)."""
        self._data = df

    @property
    def size(self) -> int:
        return len(self._data)
    
    def _align_and_concat(self, new_df: pl.DataFrame) -> pl.DataFrame:
        """
        Robustly concatenate new_df to self._data, handling type mismatches.
        This is the core fix for Polars Null vs Int64 conflicts.
        """
        if self._data.is_empty():
            return new_df
        
        # Get union of all columns
        all_cols = set(self._data.columns) | set(new_df.columns)
        
        # Align self._data: add missing columns from new_df
        for col in all_cols:
            if col not in self._data.columns:
                # Column exists in new_df but not self._data
                dtype = new_df.schema[col]
                self._data = self._data.with_columns(pl.lit(None).cast(dtype).alias(col))
            if col not in new_df.columns:
                # Column exists in self._data but not new_df
                dtype = self._data.schema[col]
                new_df = new_df.with_columns(pl.lit(None).cast(dtype).alias(col))
        
        # Handle type mismatches between matching columns
        for col in all_cols:
            left_type = self._data.schema[col]
            right_type = new_df.schema[col]
            
            if left_type != right_type:
                # Promote Null to concrete type
                if left_type == pl.Null and right_type != pl.Null:
                    self._data = self._data.with_columns(pl.col(col).cast(right_type))
                elif right_type == pl.Null and left_type != pl.Null:
                    new_df = new_df.with_columns(pl.col(col).cast(left_type))
                # For other mismatches, try to find supertype or cast to Object
                else:
                    try:
                        # Try casting new_df to self._data's type
                        new_df = new_df.with_columns(pl.col(col).cast(left_type))
                    except (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError, TypeError, ValueError):
                        # Last resort: cast both to String. This is
                        # destructive — log a warning so the user can
                        # pre-cast their input columns explicitly.
                        warnings.warn(
                            f"Column {col!r} has mismatched types "
                            f"({left_type} vs {right_type}); falling back to "
                            f"String. Pre-cast your columns to avoid this.",
                            UserWarning,
                            stacklevel=2,
                        )
                        self._data = self._data.with_columns(pl.col(col).cast(pl.Utf8))
                        new_df = new_df.with_columns(pl.col(col).cast(pl.Utf8))
        
        # Ensure column order matches
        new_df = new_df.select(self._data.columns)
        
        return pl.concat([self._data, new_df], how="vertical")
        
    def add_agent(self, agent_id: int, step: int = 0, **attributes):
        """Adds a single agent to the population."""
        row = {'id': agent_id, 'step': step, **attributes}
        
        # Ensure all schema columns are present
        for col, dtype in self.schema.items():
            if col not in row:
                row[col] = None
        
        new_row = pl.DataFrame([row])
        self._data = self._align_and_concat(new_row)

    def batch_add_agents(self, count: int, step: int = 0, **attributes):
        """Adds multiple agents efficiently."""
        start_id = self._data['id'].max() + 1 if not self._data.is_empty() else 0
        ids = range(start_id, start_id + count)
        
        new_data = {
            'id': list(ids),
            'step': [step] * count
        }
        
        for k, v in attributes.items():
            if isinstance(v, pl.Series):
                if v.len() != count:
                    raise ValueError(f"Attribute {k} length mismatch")
                new_data[k] = v.to_list()
            elif isinstance(v, (list, np.ndarray)):
                if len(v) != count:
                    raise ValueError(f"Attribute {k} length mismatch")
                new_data[k] = list(v) if isinstance(v, np.ndarray) else v
            else:
                new_data[k] = [v] * count
                
        # Fill missing schema columns
        for col in self.schema:
            if col not in new_data:
                new_data[col] = [None] * count
                
        new_df = pl.DataFrame(new_data)
        self._data = self._align_and_concat(new_df)
        
    def get_agent_value(self, agent_id: int, column: str) -> Any:
        res = self._data.filter(pl.col("id") == agent_id).select(column)
        if res.is_empty():
            raise KeyError(f"Agent {agent_id} not found")
        return res.item(0, 0)

    def set_agent_value(self, agent_id: int, column: str, value: Any):
        """Deprecated: use ``agent.<col> = value`` or ``agents.at[id].col = value``."""
        warn_deprecated(
            "Population.set_agent_value(...)",
            "agent.<col> = value or agents.at[id].set(**cols)",
        )
        self._set_agent_value(agent_id, column, value)

    def _set_agent_value(self, agent_id: int, column: str, value: Any) -> None:
        """Internal single-cell write (no deprecation warning)."""
        if hasattr(value, 'dtype'):  # Handle numpy scalars
            if np.issubdtype(value.dtype, np.integer):
                pl_type = pl.Int64
            elif np.issubdtype(value.dtype, np.floating):
                pl_type = pl.Float64
            else:
                pl_type = pl.Object
        else:
            pl_type = pl.Int64 if isinstance(value, int) else pl.Float64 if isinstance(value, float) else pl.Utf8 if isinstance(value, str) else pl.Object

        if column not in self._data.columns:
            self._data = self._data.with_columns(pl.lit(None).cast(pl_type).alias(column))

        self._data = self._data.with_columns(
            pl.when(pl.col("id") == agent_id)
            .then(pl.lit(value))
            .otherwise(pl.col(column))
            .alias(column)
        )

    def batch_update(self, updates: Dict[str, Union[np.ndarray, list]], selector: Optional[pl.Expr] = None):
        """Deprecated: use ``agents.set(**cols)`` or ``agents.where(...).set(...)``."""
        warn_deprecated(
            "Population.batch_update(...)",
            "agents.set(**cols) or agents.where(expr).set(**cols)",
        )
        self._batch_update(updates, selector)

    def _batch_update(
        self,
        updates: Dict[str, Union[np.ndarray, list]],
        selector: Optional[pl.Expr] = None,
    ) -> None:
        if selector is None:
            self._data = self._data.with_columns([
                pl.Series(k, v) for k, v in updates.items()
            ])
        else:
            cols = []
            for col, val in updates.items():
                cols.append(
                    pl.when(selector)
                    .then(val)
                    .otherwise(pl.col(col))
                    .alias(col)
                )
            self._data = self._data.with_columns(cols)

    def batch_update_by_ids(self, ids: Union[list, np.ndarray], data: Dict[str, Union[list, np.ndarray, Any]]):
        """Deprecated: use ``agents.at[ids].set(**data)``."""
        warn_deprecated(
            "Population.batch_update_by_ids(...)",
            "agents.at[ids].set(**cols)",
        )
        self._batch_update_by_ids(ids, data)

    def _batch_update_by_ids(
        self,
        ids: Union[list, np.ndarray],
        data: Dict[str, Union[list, np.ndarray, Any]],
    ) -> None:
        id_series = pl.Series("id", ids)
        count = len(ids)

        update_data = {"id": id_series}

        for col, val in data.items():
            if isinstance(val, (list, np.ndarray)):
                if len(val) != count:
                    raise ValueError(f"Value length mismatch for {col}")
                update_data[f"{col}_new"] = val
            else:
                update_data[f"{col}_new"] = [val] * count

        update_df = pl.DataFrame(update_data)

        self._data = self._data.join(update_df, on="id", how="left")

        cols = []
        for col in data.keys():
            new_col = f"{col}_new"
            cols.append(
                pl.when(pl.col(new_col).is_not_null())
                .then(pl.col(new_col))
                .otherwise(pl.col(col))
                .alias(col)
            )

        self._data = self._data.with_columns(cols).drop([f"{col}_new" for col in data.keys()])

    def create_batch_context(self):
        """Legacy batched-update context manager.

        .. deprecated::
            Prefer the vectorized view API:
            ``model.agents.at[ids].col = values`` or
            ``model.agents.at[ids].scatter_add(col=delta)``. The view API
            flushes through the same hash-join path but is discoverable
            via attribute access rather than a context manager.
        """
        import warnings

        warnings.warn(
            "Population.create_batch_context() is deprecated; use "
            "model.agents.at[ids].col = values (or scatter_add) instead. "
            "See the AMBER quickstart for examples.",
            DeprecationWarning,
            stacklevel=2,
        )
        return BatchUpdateContext(self)

class BatchUpdateContext:
    """Context manager for buffering updates to minimize DataFrame copies."""
    def __init__(self, population: Population):
        self.population = population
        self.updates = {}

    def __enter__(self):
        return self

    def add_update(self, agent_id: int, col: str, val: Any):
        if agent_id not in self.updates:
            self.updates[agent_id] = {}
        self.updates[agent_id][col] = val

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.updates:
            return
            
        ids = list(self.updates.keys())
        cols = set()
        for u in self.updates.values():
            cols.update(u.keys())
            
        final_data = {}
        for col in cols:
            vals = []
            for aid in ids:
                vals.append(self.updates[aid].get(col, None))
            final_data[col] = vals
            
        self.population._batch_update_by_ids(ids, final_data)
