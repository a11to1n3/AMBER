from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING, Type
import polars as pl
import numpy as np

if TYPE_CHECKING:
    from .agent import Agent

class Population:
    """
    Manages the columnar state of all agents using Polars DataFrames.
    Acts as the single point of truth for agent data.
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
        
        # Initialize empty DataFrame
        self.data = pl.DataFrame(schema=self.schema)
        
        # Buffer for batch operations
        self._pending_updates: Dict[int, Dict[str, Any]] = {}
        
    @property
    def size(self) -> int:
        return len(self.data)
    
    def _align_and_concat(self, new_df: pl.DataFrame) -> pl.DataFrame:
        """
        Robustly concatenate new_df to self.data, handling type mismatches.
        This is the core fix for Polars Null vs Int64 conflicts.
        """
        if self.data.is_empty():
            return new_df
        
        # Get union of all columns
        all_cols = set(self.data.columns) | set(new_df.columns)
        
        # Align self.data: add missing columns from new_df
        for col in all_cols:
            if col not in self.data.columns:
                # Column exists in new_df but not self.data
                dtype = new_df.schema[col]
                self.data = self.data.with_columns(pl.lit(None).cast(dtype).alias(col))
            if col not in new_df.columns:
                # Column exists in self.data but not new_df
                dtype = self.data.schema[col]
                new_df = new_df.with_columns(pl.lit(None).cast(dtype).alias(col))
        
        # Handle type mismatches between matching columns
        for col in all_cols:
            left_type = self.data.schema[col]
            right_type = new_df.schema[col]
            
            if left_type != right_type:
                # Promote Null to concrete type
                if left_type == pl.Null and right_type != pl.Null:
                    self.data = self.data.with_columns(pl.col(col).cast(right_type))
                elif right_type == pl.Null and left_type != pl.Null:
                    new_df = new_df.with_columns(pl.col(col).cast(left_type))
                # For other mismatches, try to find supertype or cast to Object
                else:
                    try:
                        # Try casting new_df to self.data's type
                        new_df = new_df.with_columns(pl.col(col).cast(left_type))
                    except:
                        # Last resort: cast both to String
                        self.data = self.data.with_columns(pl.col(col).cast(pl.Utf8))
                        new_df = new_df.with_columns(pl.col(col).cast(pl.Utf8))
        
        # Ensure column order matches
        new_df = new_df.select(self.data.columns)
        
        return pl.concat([self.data, new_df], how="vertical")
        
    def add_agent(self, agent_id: int, step: int = 0, **attributes):
        """Adds a single agent to the population."""
        row = {'id': agent_id, 'step': step, **attributes}
        
        # Ensure all schema columns are present
        for col, dtype in self.schema.items():
            if col not in row:
                row[col] = None
        
        new_row = pl.DataFrame([row])
        self.data = self._align_and_concat(new_row)

    def batch_add_agents(self, count: int, step: int = 0, **attributes):
        """Adds multiple agents efficiently."""
        start_id = self.data['id'].max() + 1 if not self.data.is_empty() else 0
        ids = range(start_id, start_id + count)
        
        new_data = {
            'id': list(ids),
            'step': [step] * count
        }
        
        for k, v in attributes.items():
            if isinstance(v, (list, np.ndarray)):
                if len(v) != count:
                    raise ValueError(f"Attribute {k} length mismatch")
                new_data[k] = v
            else:
                new_data[k] = [v] * count
                
        # Fill missing schema columns
        for col in self.schema:
            if col not in new_data:
                new_data[col] = [None] * count
                
        new_df = pl.DataFrame(new_data)
        self.data = self._align_and_concat(new_df)
        
    def get_agent_value(self, agent_id: int, column: str) -> Any:
        res = self.data.filter(pl.col("id") == agent_id).select(column)
        if res.is_empty():
            raise KeyError(f"Agent {agent_id} not found")
        return res.item(0, 0)

    def set_agent_value(self, agent_id: int, column: str, value: Any):
        """Sets a value for a single agent. Very slow if used in loops."""
        # Determine Polars type from value
        if hasattr(value, 'dtype'):  # Handle numpy scalars
            if np.issubdtype(value.dtype, np.integer):
                pl_type = pl.Int64
            elif np.issubdtype(value.dtype, np.floating):
                pl_type = pl.Float64
            else:
                pl_type = pl.Object
        else:
            pl_type = pl.Int64 if isinstance(value, int) else pl.Float64 if isinstance(value, float) else pl.Utf8 if isinstance(value, str) else pl.Object
        
        # Check if column exists, if not create it with correct type
        if column not in self.data.columns:
            self.data = self.data.with_columns(pl.lit(None).cast(pl_type).alias(column))

        # Polars explicit update
        self.data = self.data.with_columns(
            pl.when(pl.col("id") == agent_id)
            .then(pl.lit(value))
            .otherwise(pl.col(column))
            .alias(column)
        )
        
    def batch_update(self, updates: Dict[str, Union[np.ndarray, list]], selector: Optional[pl.Expr] = None):
        """Updates columns for all agents (or a filtered subset)."""
        if selector is None:
            self.data = self.data.with_columns([
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
            self.data = self.data.with_columns(cols)

    def batch_update_by_ids(self, ids: Union[list, np.ndarray], data: Dict[str, Union[list, np.ndarray, Any]]):
        """Updates specific agents identified by IDs."""
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
        
        self.data = self.data.join(update_df, on="id", how="left")
        
        cols = []
        for col in data.keys():
            new_col = f"{col}_new"
            cols.append(
                pl.when(pl.col(new_col).is_not_null())
                .then(pl.col(new_col))
                .otherwise(pl.col(col))
                .alias(col)
            )
        
        self.data = self.data.with_columns(cols).drop([f"{col}_new" for col in data.keys()])

    def create_batch_context(self):
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
            
        self.population.batch_update_by_ids(ids, final_data)
