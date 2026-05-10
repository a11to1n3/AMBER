from typing import Any, Dict, Optional

import polars as pl

from .base import BaseAgent

# Names that are set during __init__ or are internal and should NOT be
# routed to the DataFrame when assigned on a Python Agent instance.
_AGENT_PROTECTED_ATTRS = frozenset({"model", "id", "p"})


class Agent(BaseAgent):
    """Base class for all agents in the simulation.

    .. note::

        Setting an attribute on a Python ``Agent`` instance (e.g.
        ``agent.wealth = 5``) automatically queues a write to the
        underlying columnar DataFrame. This keeps the Python objects
        and the columnar store in sync — you can freely mix OOP-style
        attribute access with the view API (``model.agents.wealth``).
        Internal attributes (``model``, ``id``, ``p``, anything
        starting with ``_``) are stored on the instance only.
    """

    def __repr__(self):
        return f"Agent(id={self.id})"

    def __setattr__(self, name: str, value: Any) -> None:
        """Route non-internal attribute writes to the DataFrame.

        * ``model``, ``id``, ``p`` — stored on the Python instance only.
        * Private names (``_`` prefix) — stored on the Python instance only.
        * Everything else — queued to the DataFrame via
          ``model._queue_write`` **and** stored on the instance for
          fast local access during the same step.
        """
        if name in _AGENT_PROTECTED_ATTRS or name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        # Store locally so ``agent.wealth`` returns the latest value
        # for the remainder of this step.
        object.__setattr__(self, name, value)
        # Queue the write so the columnar view stays in sync.
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "_queue_write"):
            model._queue_write(name, self.id, value)

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
