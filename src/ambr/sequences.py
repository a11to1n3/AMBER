"""AgentList and view-based subset types.

The full population view lives at ``model.agents``. Filtered and scatter
views are produced by ``agents.where(...)`` / ``agents[mask]`` /
``agents.at[ids]``. All views route column reads/writes through
``model.agents_df`` so the DataFrame is the single source of truth.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import polars as pl

from ._deprecation import warn_deprecated

from .agent import Agent
from .model import Model


# Names that must live on the Python instance, not as DataFrame columns.
_INTERNAL_ATTRS = frozenset({
    "model",
    "agent_type",
    "_agent_objects",
    "_agents_by_id",
    "_ids",
    "_parent",
})


def _normalize_delta(name: str, value: Any, n: int) -> np.ndarray:
    """Coerce a scatter_add / _write_column value into a length-``n`` ndarray."""
    if isinstance(value, pl.Series):
        if value.len() != n:
            raise ValueError(
                f"length {value.len()} for {name!r} does not match view length {n}"
            )
        return value.to_numpy()
    if isinstance(value, np.ndarray):
        if len(value) != n:
            raise ValueError(
                f"length {len(value)} for {name!r} does not match view length {n}"
            )
        return value
    if isinstance(value, list):
        if len(value) != n:
            raise ValueError(
                f"length {len(value)} for {name!r} does not match view length {n}"
            )
        return np.asarray(value)
    return np.full(n, value)


class _BaseView:
    """Attribute/assignment protocol shared by every view type."""

    # Subclasses override.
    def _ids_series(self) -> pl.Series:
        raise NotImplementedError

    # --- attribute protocol -------------------------------------------------

    def __getattr__(self, name: str):
        if name.startswith("_") or name in _INTERNAL_ATTRS:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        model = self.__dict__.get("model")
        if model is None:
            raise AttributeError(name)
        df = model.agents_df
        if name not in df.columns:
            # Backward-compat fallback: if the name is a callable method
            # on the underlying agents, return a wrapper that dispatches
            # to each agent and collects results into a numpy array.
            root = self._root()
            agents = getattr(root, '_agent_objects', None)
            if agents:
                first = agents[0] if agents else None
                method = getattr(first, name, None) if first is not None else None
                if callable(method):
                    import numpy as np
                    def _dispatch(*args, **kwargs):
                        results = [getattr(a, name)(*args, **kwargs) for a in self]
                        try:
                            return np.array(results)
                        except (ValueError, TypeError):
                            return results
                    return _dispatch
            raise AttributeError(
                f"{type(self).__name__!r} has no column {name!r}; "
                f"available columns: {df.columns}"
            )
        ids = self._ids_series()
        if ids.len() == df.height and ids.equals(df["id"]):
            return df[name]
        # Align the returned Series with the view's id order so that
        # duplicate-id scatter views still return a length-matched column.
        # When name is "id", skip the join to avoid duplicate column error.
        if name == "id":
            return ids
        ids_df = pl.DataFrame([ids.rename("id")])
        return ids_df.join(df.select("id", name), on="id", how="left")[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in _INTERNAL_ATTRS:
            object.__setattr__(self, name, value)
            return
        self._write_column(name, value)

    # --- columnar writes ----------------------------------------------------

    def _write_column(self, name: str, value: Any) -> None:
        """Assign ``value`` to column ``name`` over this view's agents.

        Accepts scalars, ``pl.Series`` / ``np.ndarray`` / list matching
        ``len(view)``, and ``pl.Expr`` evaluated over the view's rows.
        """
        model = self.__dict__["model"]
        model._flush_pending_writes()

        ids = self._ids_series()
        n = ids.len()
        df = model.agents_df

        if isinstance(value, pl.Expr):
            sub = df.filter(pl.col("id").is_in(ids.to_list())).select(
                pl.col("id"), value.alias("__new__")
            )
            if name not in df.columns:
                df = df.with_columns(pl.Series(name, [None] * df.height, strict=False))
            joined = df.join(sub, on="id", how="left").with_columns(
                pl.when(pl.col("__new__").is_not_null())
                .then(pl.col("__new__"))
                .otherwise(pl.col(name))
                .alias(name)
            ).drop("__new__")
            model._set_frame(joined, written_columns=[name])
            return

        values = value if isinstance(value, pl.Series) else pl.Series(
            name,
            [value] * n if not isinstance(value, (list, np.ndarray)) else value,
            strict=False,
        )
        if values.len() != n:
            raise ValueError(
                f"Cannot assign Series of length {values.len()} to view of length {n}"
            )

        # Whole-population fast path: one with_columns, no join.
        if n == df.height and ids.equals(df["id"]):
            model._set_frame(
                df.with_columns(values.alias(name)), written_columns=[name]
            )
            return

        if name not in df.columns:
            df = df.with_columns(pl.Series(name, [None] * df.height, strict=False))
        update_df = pl.DataFrame([ids.rename("id"), values.rename(name)])
        model._set_frame(
            df.update(update_df, on="id", how="left"), written_columns=[name]
        )

    @property
    def ids(self) -> pl.Series:
        return self._ids_series()

    # --- filtering ----------------------------------------------------------

    def where(self, predicate) -> "FilteredAgentList":
        """Return a view of agents matching a boolean Series or Polars expression."""
        model = self.__dict__["model"]
        base_ids = self._ids_series()
        if isinstance(predicate, pl.Series):
            if predicate.dtype != pl.Boolean:
                raise TypeError("Series predicate must be boolean")
            if predicate.len() != base_ids.len():
                raise ValueError(
                    f"Boolean mask length {predicate.len()} does not match "
                    f"view length {base_ids.len()}"
                )
            new_ids = base_ids.filter(predicate)
        elif isinstance(predicate, pl.Expr):
            df = model.agents_df
            sub = df.filter(pl.col("id").is_in(base_ids.to_list())).filter(predicate)
            new_ids = sub["id"]
        else:
            raise TypeError("predicate must be a polars Series (boolean) or Expr")
        return FilteredAgentList(model, new_ids, parent=self._root())

    def select(self, selection) -> "_BaseView":
        """Deprecated AgentPy filter; use ``where(expr)`` / ``at[ids]`` / ``[mask]``."""
        warn_deprecated(
            "AgentList.select(...)",
            "agents.where(expr) / agents.at[ids] / agents[mask]",
        )
        return self._select_impl(selection)

    def _select_impl(self, selection) -> "_BaseView":
        """AgentPy-compatible filter.

        Accepts bool masks (list/ndarray/Series), Polars expressions,
        and id lists. Returns a FilteredAgentList or ScatterAgentList.
        """
        model = self.__dict__["model"]
        root = self._root()
        if isinstance(selection, pl.Expr):
            return self.where(selection)
        if isinstance(selection, pl.Series):
            if selection.dtype == pl.Boolean:
                return self.where(selection)
            return FilteredAgentList(model, selection.rename("id"), parent=root)
        if isinstance(selection, (list, np.ndarray)):
            arr = np.asarray(selection)
            if arr.dtype == bool:
                ids = self._ids_series()
                if len(arr) != ids.len():
                    raise ValueError(
                        f"Boolean mask length ({len(arr)}) does not match "
                        f"view length ({ids.len()})"
                    )
                picked = ids.filter(pl.Series("mask", arr))
                return FilteredAgentList(model, picked, parent=root)
            # List of ids
            return FilteredAgentList(model, pl.Series("id", arr.tolist()), parent=root)
        raise TypeError(f"select() unsupported type: {type(selection)}")

    def _root(self) -> "AgentList":
        return self.__dict__.get("_parent") or self  # type: ignore[return-value]

    # --- length / iteration -------------------------------------------------

    def __len__(self) -> int:
        return self._ids_series().len()

    def __iter__(self):
        lookup = getattr(self._root(), "_agents_by_id", None) or {}
        for aid in self._ids_series().to_list():
            agent = lookup.get(aid)
            if agent is not None:
                yield agent

    @property
    def at(self) -> "_AtIndexer":
        return _AtIndexer(self)

    # --- method dispatch ----------------------------------------------------

    def call(self, method_name: str, *args, **kwargs):
        """Invoke ``method_name`` on each Python Agent in this view."""
        results = []
        for agent in self:
            method = getattr(agent, method_name, None)
            if callable(method):
                results.append(method(*args, **kwargs))
        try:
            return np.array(results)
        except ValueError:
            return results

    def apply(self, func: Callable[[Agent], Any]) -> pl.Series:
        return pl.Series([func(a) for a in self])

    # --- legacy aliases -----------------------------------------------------

    def record(self, name: str, value: Any) -> None:
        """Deprecated alias for ``view.<name> = value`` (or ``view.set(...)``)."""
        warn_deprecated("AgentList.record(name, value)", "view.<name> = value (or view.set(...))")
        self._write_column(name, value)

    def update_data(self, data: Dict[str, Any]) -> None:
        """Deprecated alias for a multi-column write (``view.set(**cols)``)."""
        warn_deprecated("AgentList.update_data(data)", "view.set(**cols)")
        self.set(**data)

    # --- ergonomic read / write --------------------------------------------

    def numpy(self, *columns: str):
        """Return the named columns as numpy arrays, aligned to this view.

        ``x = agents.numpy('x')`` returns one array; ``x, y = agents.numpy('x',
        'y')`` returns a tuple -- a one-call replacement for the repeated
        ``agents.x.to_numpy()`` idiom.
        """
        arrays = tuple(getattr(self, c).to_numpy() for c in columns)
        return arrays[0] if len(columns) == 1 else arrays

    def set(self, **columns: Any) -> None:
        """Write one or more whole columns over this view's agents atomically.

        ``agents.set(x=nx, y=ny)`` is a single ``with_columns`` / update (one
        contract commit per column), not a loop of independent assignments.
        Expression values (``pl.Expr``) fall back to per-column writes.
        """
        if not columns:
            return
        if any(isinstance(v, pl.Expr) for v in columns.values()):
            for name, value in columns.items():
                self._write_column(name, value)
            return

        model = self.__dict__["model"]
        model._flush_pending_writes()
        ids = self._ids_series()
        n = ids.len()
        df = model.agents_df

        series_by_name: Dict[str, pl.Series] = {}
        for name, value in columns.items():
            values = value if isinstance(value, pl.Series) else pl.Series(
                name,
                [value] * n if not isinstance(value, (list, np.ndarray)) else value,
                strict=False,
            )
            if values.len() != n:
                raise ValueError(
                    f"Cannot assign Series of length {values.len()} "
                    f"to view of length {n} for column {name!r}"
                )
            series_by_name[name] = values

        names = list(series_by_name)
        # Whole-population fast path: one with_columns for all columns.
        if n == df.height and ids.equals(df["id"]):
            model._set_frame(
                df.with_columns(
                    [s.alias(name) for name, s in series_by_name.items()]
                ),
                written_columns=names,
            )
            return

        for name in names:
            if name not in df.columns:
                df = df.with_columns(
                    pl.Series(name, [None] * df.height, strict=False)
                )
        update_df = pl.DataFrame(
            [ids.rename("id")]
            + [s.rename(name) for name, s in series_by_name.items()]
        )
        model._set_frame(
            df.update(update_df, on="id", how="left"),
            written_columns=names,
        )

    # --- scatter-add --------------------------------------------------------

    def scatter_add(self, **increments: Any) -> None:
        """Accumulate per-id deltas into columns, summing across duplicate ids.

        ``view.at[[1, 1, 3]].scatter_add(wealth=1)`` gives agent ``1`` a +2
        and agent ``3`` a +1. Accepts the same value shapes as column
        assignment.
        """
        if not increments:
            return
        model = self.__dict__["model"]
        model._flush_pending_writes()

        ids = self._ids_series()
        n = ids.len()
        df = model.agents_df
        if n == 0:
            missing = [c for c in increments if c not in df.columns]
            if missing:
                model._set_frame(df.with_columns(
                    [pl.Series(c, [None] * df.height, strict=False) for c in missing]
                ))
            return

        delta_np: Dict[str, np.ndarray] = {
            col: _normalize_delta(col, val, n) for col, val in increments.items()
        }

        positions = _resolve_positions(model, df, ids.to_numpy())

        new_columns: List[pl.Series] = []
        for col_name, delta in delta_np.items():
            if col_name in df.columns:
                base = df[col_name].to_numpy()
                # np.add.at won't upcast the output dtype, so expand first
                # when we're adding e.g. a float delta to an int column.
                result_dtype = np.result_type(base.dtype, delta.dtype)
                base = base.astype(result_dtype, copy=True) if base.dtype != result_dtype else base.copy()
            else:
                base = np.zeros(df.height, dtype=delta.dtype)
            np.add.at(base, positions, delta)
            new_columns.append(pl.Series(col_name, base, strict=False))

        model._set_frame(df.with_columns(new_columns))


def _resolve_positions(model: Model, df: pl.DataFrame, ids_np: np.ndarray) -> np.ndarray:
    """Map view ids to row positions in ``df``, caching the lookup by id version."""
    df_ids_np = df["id"].to_numpy()
    # Contiguous [0, N) fast path (the common case after add_agents).
    if (
        df_ids_np.dtype.kind in ("i", "u")
        and df_ids_np.size == df.height
        and df_ids_np.size > 0
        and df_ids_np[0] == 0
        and df_ids_np[-1] == df_ids_np.size - 1
        and np.array_equal(df_ids_np, np.arange(df_ids_np.size))
    ):
        return np.asarray(ids_np, dtype=np.int64)

    # General case: reuse the id→position hash table across scatter_add calls
    # within the same step, invalidating whenever the id column changes.
    cached: Optional[Tuple[int, Dict[int, int]]] = getattr(model, "_id_pos_cache", None)
    version = getattr(model, "_id_version", 0)
    if cached is None or cached[0] != version:
        id_to_pos = {int(v): i for i, v in enumerate(df_ids_np)}
        model._id_pos_cache = (version, id_to_pos)
    else:
        id_to_pos = cached[1]
    return np.fromiter(
        (id_to_pos[int(v)] for v in ids_np),
        dtype=np.int64,
        count=int(ids_np.size),
    )


class _AtIndexer:
    """``view.at[ids]`` -> ScatterAgentList keyed by those ids."""

    def __init__(self, view: "_BaseView"):
        object.__setattr__(self, "_view", view)

    def __getitem__(self, key) -> "ScatterAgentList":
        view: _BaseView = self._view
        model = view.__dict__["model"]
        if isinstance(key, (int, np.integer)):
            ids = pl.Series("id", [int(key)])
        elif isinstance(key, pl.Series):
            ids = key.rename("id")
        elif isinstance(key, (list, np.ndarray)):
            ids = pl.Series("id", list(key))
        else:
            raise TypeError(
                f"at[...] accepts int, list, ndarray, or Series (got {type(key).__name__})"
            )
        return ScatterAgentList(model, ids, parent=view._root())


class AgentList(_BaseView):
    """Full view over a model's population. Lives at ``model.agents``."""

    def __init__(
        self,
        model: Model,
        agents_or_n: Union[List[Agent], int] = None,
        agent_type: Optional[Type[Agent]] = None,
    ):
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "_agent_objects", [])
        object.__setattr__(self, "_agents_by_id", {})

        if agents_or_n is None:
            agents_or_n = []

        resolved_type: Optional[Type[Agent]] = agent_type
        if isinstance(agents_or_n, list):
            objs = list(agents_or_n)
            if resolved_type is None and objs:
                resolved_type = type(objs[0])
            for a in objs:
                self._track_agent(a)
        else:
            if agent_type is None:
                raise ValueError("agent_type is required when creating new agents")
            resolved_type = agent_type
            for i in range(agents_or_n):
                a = agent_type(model, i)
                a.setup()
                self._track_agent(a)

        object.__setattr__(self, "agent_type", resolved_type)

    # --- internal tracking ---------------------------------------------------

    def _track_agent(self, agent: Agent) -> None:
        self._agent_objects.append(agent)
        aid = getattr(agent, "id", None)
        if aid is not None:
            self._agents_by_id[aid] = agent

    def _untrack_agent(self, agent: Agent) -> None:
        aid = getattr(agent, "id", None)
        if aid is not None:
            self._agents_by_id.pop(aid, None)

    # --- view hooks ----------------------------------------------------------

    def _ids_series(self) -> pl.Series:
        # When Agent objects are tracked, return only their IDs — not
        # the entire population. This keeps each AgentList isolated.
        if self._agent_objects:
            ids = [a.id for a in self._agent_objects]
            return pl.Series("id", ids, dtype=pl.Int64)
        # Fall back to the full DataFrame for view-only or vectorized usage.
        df = self.model.agents_df
        return df["id"] if "id" in df.columns else pl.Series("id", [], dtype=pl.Int64)

    @property
    def frame(self) -> pl.DataFrame:
        """Read-only snapshot of the full agent table (alias for ``model.agents_df``)."""
        return self.model.agents_df

    @property
    def agents(self) -> List[Agent]:
        """Deprecated: iterate ``model.agents`` directly, or use ``by_id`` / ``ids``."""
        warn_deprecated("AgentList.agents", "iterating model.agents (or agents.by_id / agents.ids)")
        return self._agent_objects

    def __iter__(self):
        return iter(self._agent_objects)

    def __len__(self) -> int:
        # Prefer the Python-side tracking list (OOP-style models) and fall
        # back to agents_df.height for fully vectorized models that never
        # materialise Agent instances.
        if self._agent_objects:
            return len(self._agent_objects)
        model = self.__dict__.get("model")
        if model is None:
            return 0
        try:
            df = model.agents_df
        except Exception:
            return 0
        if not isinstance(df, pl.DataFrame) or "id" not in df.columns:
            return 0
        return df.height

    def __contains__(self, agent) -> bool:
        return agent in self._agent_objects

    def __repr__(self) -> str:
        return f"AgentList({len(self)} agents)"

    def __getitem__(self, idx):
        """Index by position (int/slice), id list, boolean mask, or ``pl.Expr``."""
        if isinstance(idx, (int, np.integer)):
            return self._agent_objects[int(idx)]
        if isinstance(idx, slice):
            return self._agent_objects[idx]
        if isinstance(idx, pl.Expr):
            return self.where(idx)
        if isinstance(idx, pl.Series):
            if idx.dtype == pl.Boolean:
                return self.where(idx)
            return FilteredAgentList(self.model, idx.rename("id"), parent=self)
        if isinstance(idx, (list, np.ndarray)):
            arr = np.asarray(idx)
            if arr.dtype == bool:
                if len(arr) != len(self._agent_objects):
                    raise ValueError(
                        f"Boolean mask length ({len(arr)}) does not match "
                        f"AgentList length ({len(self._agent_objects)})"
                    )
                picked = [
                    getattr(a, "id", None)
                    for a, keep in zip(self._agent_objects, arr)
                    if keep
                ]
                return FilteredAgentList(
                    self.model, pl.Series("id", picked), parent=self
                )
            # list of positions → pick those agents by index
            picked_ids = [
                getattr(self._agent_objects[int(i)], "id", None) for i in arr
            ]
            return FilteredAgentList(
                self.model, pl.Series("id", picked_ids), parent=self
            )
        raise TypeError(f"Invalid index type: {type(idx)}")

    def __setitem__(self, idx, agent) -> None:
        old = self._agent_objects[idx]
        self._untrack_agent(old)
        self._agent_objects[idx] = agent
        aid = getattr(agent, "id", None)
        if aid is not None:
            self._agents_by_id[aid] = agent

    def __add__(self, other):
        if isinstance(other, AgentList):
            combined = self._agent_objects + other._agent_objects
        elif isinstance(other, list):
            combined = self._agent_objects + other
        else:
            raise TypeError(f"Cannot add {type(other)} to AgentList")
        return AgentList(self.model, combined, agent_type=self.agent_type)

    # --- list-like mutation --------------------------------------------------

    def append(self, agent: Agent) -> None:
        self._track_agent(agent)

    def extend(self, agents: List[Agent]) -> None:
        for a in agents:
            self._track_agent(a)

    def remove(self, agent: Agent) -> None:
        self._agent_objects.remove(agent)
        self._untrack_agent(agent)

    def clear(self) -> None:
        self._agent_objects.clear()
        self._agents_by_id.clear()

    def copy(self) -> "AgentList":
        new_list = AgentList(self.model, list(self._agent_objects))
        new_list.agent_type = self.agent_type
        return new_list

    def index(self, agent: Agent) -> int:
        return self._agent_objects.index(agent)

    def count(self, agent: Agent) -> int:
        return self._agent_objects.count(agent)

    def pop(self, idx: int = -1) -> Agent:
        a = self._agent_objects.pop(idx)
        self._untrack_agent(a)
        return a

    def insert(self, idx: int, agent: Agent) -> None:
        self._agent_objects.insert(idx, agent)
        aid = getattr(agent, "id", None)
        if aid is not None:
            self._agents_by_id[aid] = agent

    def reverse(self) -> None:
        self._agent_objects.reverse()

    def sort(self, key=None, reverse: bool = False) -> None:
        self._agent_objects.sort(key=key, reverse=reverse)

    # --- legacy property ----------------------------------------------------

    @property
    def agent_ids(self):
        """Deprecated alias for :attr:`ids`."""
        warn_deprecated("AgentList.agent_ids", "agents.ids")
        return [getattr(agent, "id", i) for i, agent in enumerate(self._agent_objects)]

    # --- legacy APIs (kept as thin wrappers around the column protocol) ----

    def get_data(self) -> pl.DataFrame:
        if hasattr(self.model, "agents_df"):
            return self.model.agents_df
        return pl.DataFrame()

    def by_id(self, agent_id) -> Agent:
        """Return the tracked Agent object with this id (the per-agent / OOP lane).

        Lets per-agent code reach another agent without a hand-rolled id->object
        dict (``add_agents(n, agent_class=...)`` tracks the objects for you).
        """
        agent = self._agents_by_id.get(agent_id)
        if agent is None:
            raise KeyError(f"no tracked agent with id {agent_id!r}")
        return agent

    def borrow(self, column: str):
        """Zero-copy, read-only borrow of a numeric column for tensor kernels.

        Returns ``(array, is_view)``; pair with :meth:`commit`. See
        ``ambr.tensor_lane`` for the snapshot-view contract on borrow/commit.
        """
        from .tensor_lane import borrow_numeric
        return borrow_numeric(self.model, column)

    def commit(self, **columns: Any) -> None:
        """Atomically write back derived columns (the tensor-lane commit path).

        ``agents.commit(x=nx, y=ny)``. Routes through ``commit_columns`` so the
        snapshot-view contract observes the writes.
        """
        from .tensor_lane import commit_columns
        commit_columns(self.model, **columns)

    def group_by(self, by: str) -> Dict[Any, "FilteredAgentList"]:
        groups: Dict[Any, FilteredAgentList] = {}
        if not hasattr(self.model, "agents_df"):
            return groups
        df = self.model.agents_df
        if by not in df.columns:
            return groups
        for group_value, group_df in df.group_by(by):
            groups[group_value[0] if isinstance(group_value, tuple) else group_value] = (
                FilteredAgentList(self.model, group_df["id"], parent=self)
            )
        return groups


class _SubView(_BaseView):
    """Base for views backed by an explicit id list."""

    def __init__(self, model: Model, ids: pl.Series, parent: AgentList):
        object.__setattr__(self, "model", model)
        if ids.name != "id":
            ids = ids.rename("id")
        object.__setattr__(self, "_ids", ids)
        object.__setattr__(self, "_parent", parent)

    def _ids_series(self) -> pl.Series:
        return self._ids

    def _root(self) -> AgentList:
        return self._parent

    def __getitem__(self, idx):
        """Index by position (int/slice/list/ndarray) within this view."""
        if isinstance(idx, (int, np.integer)):
            id_list = self._ids.to_list()
            aid = id_list[int(idx)]
            lookup = getattr(self._root(), "_agents_by_id", None) or {}
            agent = lookup.get(aid)
            if agent is not None:
                return agent
            raise IndexError(f"Agent id={aid} not found in AgentList")
        if isinstance(idx, slice):
            id_list = self._ids.to_list()[idx]
            lookup = getattr(self._root(), "_agents_by_id", None) or {}
            return [lookup.get(aid) for aid in id_list]
        if isinstance(idx, (list, np.ndarray)):
            arr = np.asarray(idx)
            id_list = self._ids.to_list()
            if arr.dtype == bool:
                return self._select_impl(arr)
            # List of positions → return a FilteredAgentList view
            picked_ids = [id_list[int(i)] for i in arr]
            return FilteredAgentList(
                self.__dict__["model"],
                pl.Series("id", picked_ids),
                parent=self._root(),
            )
        if isinstance(idx, pl.Expr):
            return self.where(idx)
        if isinstance(idx, pl.Series):
            if idx.dtype == pl.Boolean:
                return self.where(idx)
            return FilteredAgentList(self.__dict__["model"], idx.rename("id"), parent=self._root())
        raise TypeError(f"{type(self).__name__} indices must be int, slice, list, or ndarray")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._ids.len()} agents)"


class FilteredAgentList(_SubView):
    """Subset view produced by ``agents.where(...)`` / ``agents[mask]``."""


class ScatterAgentList(_SubView):
    """Id-indexed view produced by ``agents.at[ids]`` (ids may repeat)."""
