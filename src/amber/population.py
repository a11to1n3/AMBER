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
        
    def add_agent(self, agent_id: int, step: int = 0, **attributes):
        """Adds a single agent to the population."""
        row = {'id': agent_id, 'step': step, **attributes}
        
        # Ensure all schema columns are present
        for col, dtype in self.schema.items():
            if col not in row:
                row[col] = None
        
        # Efficiently append using specialized Polars patterns or buffering
        # For single adds, concat is okay-ish but batch adds are better.
        # Here we just concat for safety.
        new_row = pl.DataFrame([row], schema=self.schema)
        self.data = pl.concat([self.data, new_row], how="vertical")

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
                
        self.data = pl.concat([self.data, pl.DataFrame(new_data, schema=self.schema)], how="vertical")
        
    def get_agent_value(self, agent_id: int, column: str) -> Any:
        # Warning: This is O(N) because of the filter. 
        # For high perf, avoid this in loops.
        # But we implement it for the Proxy pattern.
        res = self.data.filter(pl.col("id") == agent_id).select(column)
        if res.is_empty():
            raise KeyError(f"Agent {agent_id} not found")
        return res.item(0, 0)

    def set_agent_value(self, agent_id: int, column: str, value: Any):
        """Sets a value for a single agent. Very slow if used in loops."""
        # Check if column exists, if not create it
        if column not in self.data.columns:
            self.data = self.data.with_columns(pl.lit(None).alias(column))
            self.schema[column] = type(value)

        # Polars explicit update
        self.data = self.data.with_columns(
            pl.when(pl.col("id") == agent_id)
            .then(pl.lit(value))
            .otherwise(pl.col(column))
            .alias(column)
        )
        
    def batch_update(self, updates: Dict[str, Union[np.ndarray, list]], selector: Optional[pl.Expr] = None):
        """
        Updates columns for all agents (or a filtered subset).
        
        Args:
            updates: Dict of {column: values}
            selector: Optional Polars expression to filter which rows to update
        """
        # If no selector, it's a full column replacement (fastest)
        if selector is None:
            self.data = self.data.with_columns([
                pl.Series(k, v) for k, v in updates.items()
            ])
        else:
            # Conditional update
            cols = []
            for col, val in updates.items():
                cols.append(
                    pl.when(selector)
                    .then(val) # 'val' can be scalar or expression/series
                    .otherwise(pl.col(col))
                    .alias(col)
                )
            self.data = self.data.with_columns(cols)

    def batch_update_by_ids(self, ids: Union[list, np.ndarray], data: Dict[str, Union[list, np.ndarray, Any]]):
        """
        Updates specific agents identified by IDs.
        """
        # Convert IDs to Polars Series for joining
        id_series = pl.Series("id", ids)
        
        # Check consistency
        count = len(ids)
        
        # Prepare update DataFrame
        update_data = {"id": id_series}
        
        for col, val in data.items():
            if isinstance(val, (list, np.ndarray)):
                if len(val) != count:
                    raise ValueError(f"Value length mismatch for {col}")
                update_data[f"{col}_new"] = val
            else:
                update_data[f"{col}_new"] = [val] * count
                
        update_df = pl.DataFrame(update_data)
        
        # Efficient join-update pattern
        # Left join original data with updates
        self.data = self.data.join(update_df, on="id", how="left")
        
        # Coalesce columns
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
        self.updates = {} # {id: {col: val}}

    def __enter__(self):
        return self

    def add_update(self, agent_id: int, col: str, val: Any):
        if agent_id not in self.updates:
            self.updates[agent_id] = {}
        self.updates[agent_id][col] = val

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.updates:
            return
            
        # Convert buffer to batch update format
        ids = list(self.updates.keys())
        cols = set()
        for u in self.updates.values():
            cols.update(u.keys())
            
        data = {c: [] for c in cols}
        
        # Fill data (handle missing updates for some agents with None or keep separate?)
        # For efficiency, we group by column set or just fill. 
        # Sparse updates in batch are tricky.
        # Simple approach: one batch update per column set? 
        # Or just one big update with Nones? 
        # Population.batch_update_by_ids handles Nones if we pass them.
        
        # Actually, let's just do a columnar transform
        count = len(ids)
        final_data = {}
        
        for col in cols:
            vals = []
            for ais in ids:
                vals.append(self.updates[ais].get(col, None)) 
            final_data[col] = vals
            
        self.population.batch_update_by_ids(ids, final_data)
