from operator import index as integer_index
from typing import TYPE_CHECKING, Any, Dict, Optional

import polars as pl

from .base import BaseAgent
from ._deprecation import warn_deprecated
from ._id_index import resolve_position
from ._keyed_rng import keyed_uniform
from .execution import sync_agent_writes, sync_device_column_to_frame

if TYPE_CHECKING:
    from .model import Model

# Names that are set during __init__ or are internal and should NOT be
# routed to the DataFrame when assigned on a Python Agent instance.
_AGENT_PROTECTED_ATTRS = frozenset({"model", "id", "p"})


def _read_container_cell(attrs, name, column, position, value):
    """Reuse a Python container while its committed cell is unchanged."""
    is_list = isinstance(value, pl.Series)
    cell = column.slice(position, 1)
    cells = attrs.setdefault("_ambr_container_cells", {})
    previous = cells.get(name)
    original = attrs.get(name)
    if isinstance(original, list if is_list else dict):
        if previous is None:
            try:
                previous = pl.Series(name, [original], dtype=cell.dtype, strict=False)
            except (TypeError, ValueError, pl.exceptions.PolarsError):
                previous = None
        if previous is not None and previous.dtype == cell.dtype and previous.equals(cell):
            cells[name] = cell
            return original
    if is_list:
        value = value.to_list()
    cells[name] = cell
    attrs[name] = value
    return value


class Agent(BaseAgent):
    """Base class for all agents in the simulation.

    .. note::

        Setting an attribute on a Python ``Agent`` instance (e.g.
        ``agent.wealth = 5``) automatically queues a write to the
        underlying columnar DataFrame. This keeps the Python objects
        and the columnar store in sync — you can freely mix OOP-style
        attribute access with the view API (``model.agents.wealth``).
        Assign a list or dict after mutating it in place to publish the
        new value to the columnar store.
        Internal attributes (``model``, ``id``, ``p``, anything
        starting with ``_``) are stored on the instance only.
    """

    if TYPE_CHECKING:
        model: Model

    def __repr__(self):
        return f"Agent(id={self.id})"

    def __getattr__(self, name: str) -> Any:
        if name in ("random", "rng", "nprandom"):
            return getattr(self.model, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __getattribute__(self, name: str) -> Any:
        """Read pending writes before committed cells for tracked agents."""
        if name in _AGENT_PROTECTED_ATTRS or name.startswith("_"):
            value = object.__getattribute__(self, name)
            if name == "__dict__":
                model = value.get("model")
                if model is not None:
                    sync_agent_writes(model)
            return value
        attrs = object.__getattribute__(self, "__dict__")
        descriptor = getattr(type(self), name, None)
        # A tracked column named "draw" predates the keyed draw API.
        if isinstance(descriptor, property) or (
            name not in attrs and hasattr(descriptor, "__get__")
            and (name != "draw" or descriptor is not Agent.draw)
        ) or callable(attrs.get(name)):
            return object.__getattribute__(self, name)

        model = attrs.get("model")
        agent_id = attrs.get("id")
        model_attrs = getattr(model, "__dict__", {})
        root = model_attrs.get("agents")
        tracked = getattr(root, "__dict__", {}).get("_agents_by_id", {})
        if tracked.get(agent_id) is self:
            pending = model_attrs["_pending_writes"].get(name)
            if pending is not None and agent_id in pending:
                value = pending[agent_id]
            else:
                execution = model_attrs.get("_execution")
                if execution is not None and execution.tracked_agents and name in execution.dirty_columns:
                    sync_device_column_to_frame(model, name)
                frame = model_attrs["population"].data
                columns = frame.columns
                if frame.is_empty() or "id" not in columns:
                    return object.__getattribute__(self, name)
                if name not in columns:
                    if name in attrs:
                        raise AttributeError(f"Tracked column {name!r} does not exist")
                    return object.__getattribute__(self, name)
                try:
                    position = resolve_position(model, frame, agent_id)
                    value = frame.item(position, name)
                except (KeyError, IndexError):
                    return object.__getattribute__(self, name)
                # Polars materializes nested cells as new objects. Keep mutable
                # Python state created on the instance usable in ordinary methods.
                if isinstance(value, (pl.Series, dict)):
                    column = frame[name]
                    if column.dtype.base_type() in (pl.List, pl.Array, pl.Struct):
                        value = _read_container_cell(attrs, name, column, position, value)
        else:
            value = object.__getattribute__(self, name)

        monitor = model_attrs.get("_contract")
        if monitor is not None and monitor.active:
            monitor.record_activation_read(name, agent_id)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        """Route non-internal attribute writes to the DataFrame.

        * ``model``, ``id``, ``p`` — stored on the Python instance only.
        * Private names (``_`` prefix) — stored on the Python instance only.
        * Everything else is queued to the DataFrame. The instance copy is
          available during setup before the agent has a row.
        """
        if name in _AGENT_PROTECTED_ATTRS or name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        object.__setattr__(self, name, value)
        cells = object.__getattribute__(self, "__dict__").get("_ambr_container_cells")
        if cells is not None:
            cells.pop(name, None)
        model = getattr(self, "model", None)
        queue_write = getattr(model, "_queue_write", None)
        if queue_write is not None:
            agent_id = self.id
            queue_write(name, agent_id, value)
            monitor = getattr(model, "_contract", None)
            if monitor is not None and monitor.active:
                monitor.record_activation_write(name, agent_id)

    def setup(self):
        """Override in subclasses to initialize agent attributes."""
        pass

    def draw(self, event: str, *, index: int = 0) -> float:
        """Return a uniform draw keyed by model seed, step, agent ID, event and index.

        Reusing a key returns the same value. Use distinct event names for
        distinct decisions, or different indices for multiple draws of one event.
        """
        if type(event) is not str or not event:
            raise ValueError("event must be a non-empty string")
        index = integer_index(index)
        if index < 0:
            raise ValueError("draw index must be non-negative")
        model = self.model
        aid = self.id
        seed, step = model._keyed_seed, model.t
        value = keyed_uniform(seed, step, aid, event, index)
        monitor = model._contract
        if monitor.active:
            monitor.record_keyed_draw(seed, step, aid, event, index, value)
        return value

    def scatter_add(self, **increments: Any) -> None:
        """Accumulate increments into this agent's columns."""
        self.model.agents.at[[self.id]].scatter_add(**increments)

    def scatter_max(self, **values: Any) -> None:
        """Take the maximum of each value and this agent's current column."""
        self.model.agents.at[[self.id]].scatter_max(**values)

    def record(self, name: str, value: Any):
        """Deprecated: assign the attribute directly (``agent.<name> = value``)."""
        warn_deprecated("Agent.record(name, value)", "agent.<name> = value")
        # Route through __setattr__ so instance cache and the write queue stay
        # consistent with direct assignment.
        setattr(self, name, value)

    def get_data(self) -> pl.DataFrame:
        """Return this agent's row as a 1-row DataFrame."""
        return self.model.agents_df.filter(pl.col('id') == self.id)

    def update_data(self, data: Dict[str, Any]):
        """Deprecated: assign attributes directly (``agent.<name> = value``)."""
        warn_deprecated("Agent.update_data(data)", "direct attribute assignment")
        for name, value in data.items():
            setattr(self, name, value)

    def get_neighbors(self, condition: Optional[pl.Expr] = None) -> pl.DataFrame:
        """Return all other agents, optionally filtered by ``condition``."""
        neighbors = self.model.agents_df.filter(pl.col('id') != self.id)
        if condition is not None:
            neighbors = neighbors.filter(condition)
        return neighbors
