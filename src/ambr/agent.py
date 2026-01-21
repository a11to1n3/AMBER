from typing import Any, Dict, Optional
import polars as pl
from .base import BaseAgent

class Agent(BaseAgent):
    """Base class for all agents in the simulation, using DataFrames for data storage."""
    
    def __repr__(self):
        """String representation of the agent."""
        return f"Agent(id={self.id})"
    
    def setup(self):
        """Initialize agent attributes. Override this method in subclasses."""
        pass
        
    def record(self, name: str, value: Any):
        """Record a variable value for this agent in the model's DataFrame.
        
        Args:
            name: Name of the variable
            value: Value to record
        """
        # Add new column to model's DataFrame
        self.model.agents_df = self.model.agents_df.with_columns([
            pl.when(pl.col('id') == self.id)
            .then(pl.lit(value))
            .otherwise(pl.col(name))
            .alias(name)
        ])
        
    def get_data(self) -> pl.DataFrame:
        """Get this agent's data from the model's DataFrame.
        
        Returns:
            DataFrame containing this agent's data
        """
        return self.model.agents_df.filter(pl.col('id') == self.id)
        
    def update_data(self, data: Dict[str, Any]):
        """Update this agent's data in the model's DataFrame.
        
        Args:
            data: Dictionary of column names and values to update
        """
        for name, value in data.items():
            self.model.agents_df = self.model.agents_df.with_columns([
                pl.when(pl.col('id') == self.id)
                .then(pl.lit(value))
                .otherwise(pl.col(name))
                .alias(name)
            ])
            
    def get_neighbors(self, condition: Optional[pl.Expr] = None) -> pl.DataFrame:
        """Get neighboring agents' data based on a condition.
        
        Args:
            condition: Optional Polars expression to filter neighbors
            
        Returns:
            DataFrame containing neighboring agents' data
        """
        neighbors = self.model.agents_df.filter(pl.col('id') != self.id)
        if condition is not None:
            neighbors = neighbors.filter(condition)
        return neighbors 