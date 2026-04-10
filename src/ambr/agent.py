from typing import Any, Dict, Optional

import polars as pl

from .base import BaseAgent


class Agent(BaseAgent):
    """Base class for all agents in the simulation."""

    def __repr__(self):
        return f"Agent(id={self.id})"

    def setup(self):
        """Override in subclasses to initialize agent attributes."""
        pass

    def record(self, name: str, value: Any):
        """Record a variable value for this agent."""
        self.model._queue_write(name, self.id, value)

    def get_data(self) -> pl.DataFrame:
        """Return this agent's row as a 1-row DataFrame."""
        return self.model.agents_df.filter(pl.col('id') == self.id)

    def update_data(self, data: Dict[str, Any]):
        """Update this agent's columns from a dict."""
        for name, value in data.items():
            self.model._queue_write(name, self.id, value)

    def get_neighbors(self, condition: Optional[pl.Expr] = None) -> pl.DataFrame:
        """Return all other agents, optionally filtered by ``condition``."""
        neighbors = self.model.agents_df.filter(pl.col('id') != self.id)
        if condition is not None:
            neighbors = neighbors.filter(condition)
        return neighbors
