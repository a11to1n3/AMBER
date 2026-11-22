"""AgentList and view-based subset types (vectorized / OOP agent API).

Architecture
------------
* ``model.agents`` is an :class:`AgentList` — the full population view.
* Filtered / scatter views come from ``agents.where(...)``, ``agents[mask]``,
  and ``agents.at[ids]``.
* All column reads/writes go through ``model.agents_df`` / ``model._set_frame``
  so Polars remains the single source of truth.

Write path (DRY)
----------------
Columnar writes share helpers in this module plus:

* :mod:`ambr._id_index` — id → row position (cached per id-version)
* :mod:`ambr.performance` — ``apply_scatter_write`` / ``apply_scatter_add``
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import polars as pl

from ._deprecation import warn_deprecated
from ._id_index import resolve_positions
from .agent import Agent
from .model import Model
from ._auto_batch import try_batch
from .performance import apply_scatter_add, apply_scatter_write
from .gpu import host_value


# Names that must live on the Python instance, not as DataFrame columns.
_INTERNAL_ATTRS = frozenset({
    "model",
    "agent_type",
    "_agent_objects",
    "_agents_by_id",
    "_ids",
    "_parent",
})


# =============================================================================
# Shared value / frame helpers (used by _write_column, set, scatter_add)
# =============================================================================


def _require_length(name: str, length: int, n: int) -> None:
    """Raise if a value's length does not match the view length ``n``."""
    if length != n:
        raise ValueError(
            f"length {length} for {name!r} does not match view length {n}"
        )


def _normalize_delta(name: str, value: Any, n: int, xp=np) -> Any:
    """Coerce a scatter_add value into a length-``n`` array (NumPy or CuPy)."""
    if isinstance(value, pl.Series):
        _require_length(name, value.len(), n)
        data = value.to_numpy()
        return xp.asarray(data) if xp is not np else data
    if isinstance(value, list):
        _require_length(name, len(value), n)
        data = np.asarray(value)
        return xp.asarray(data) if xp is not np else data
    if hasattr(value, "shape") or (
        hasattr(value, "__len__") and not isinstance(value, (str, bytes))
    ):
        if getattr(type(value), "__module__", "").split(".")[0] == xp.__name__:
            _require_length(name, len(value), n)
            return value
        from .gpu import to_host

        host = np.asarray(to_host(value))
        _require_length(name, len(host), n)
        return xp.asarray(host) if xp is not np else host
    # Scalar broadcast.
    return xp.full(n, value) if xp is not np else np.full(n, value)


def _value_to_series(name: str, value: Any, n: int) -> pl.Series:
    """Coerce assignment value to a Polars Series of length ``n``."""
    value = host_value(value)
    if isinstance(value, pl.Series):
        series = value
    else:
        data = (
            value
            if isinstance(value, (list, np.ndarray))
            else [value] * n
        )
        series = pl.Series(name, data, strict=False)
    if series.len() != n:
        raise ValueError(
            f"Cannot assign Series of length {series.len()} to view of length {n}"
            + (f" for column {name!r}" if name else "")
        )
    return series


def _ensure_columns(df: pl.DataFrame, names: List[str]) -> pl.DataFrame:
    """Add missing columns as null-filled so updates can target them."""
    missing = [c for c in names if c not in df.columns]
    if not missing:
        return df
    return df.with_columns(
        [pl.Series(c, [None] * df.height, strict=False) for c in missing]
    )


def _is_full_population(ids: pl.Series, df: pl.DataFrame) -> bool:
    """True when ``ids`` is exactly the full agent table order (fast path)."""
    return ids.len() == df.height and ids.equals(df["id"])


def _commit_with_columns(
    model: Model,
    df: pl.DataFrame,
    series_list: List[pl.Series],
    *,
    written_columns: Optional[List[str]] = None,
    target_ids=None,
) -> None:
    """Single seam: ``with_columns`` + optional contract commit."""
    model._set_frame(
        df.with_columns(series_list),
        written_columns=written_columns,
        target_ids=target_ids,
    )


def _ordinary_targets(model: Model, ids: pl.Series, df: pl.DataFrame):
    """Validate ordinary assignment targets before writing any columns."""
    full = _is_full_population(ids, df)
    if len(ids) < 2:
        unique = True
    elif full:
        # Most populations have increasing integer IDs. Avoid a hash pass for
        # them and check actual values even after a direct frame replacement.
        values = ids.to_numpy()
        unique = (values.dtype.kind in "iu" and np.all(values[:-1] < values[1:]))
    else:
        unique = False
    if not unique and ids.n_unique() != ids.len():
        raise ValueError(
            "Ordinary writes require unique target IDs. Use a scatter reduction for repeated targets."
        )
    positions = resolve_positions(model, df, ids.to_numpy()) if len(ids) and not full else None
    return full, positions


def _record_expression_reads(model: Model, df: pl.DataFrame, expression: pl.Expr) -> None:
    if model._contract.active:
        columns = df.columns if expression.meta.has_multiple_outputs() else expression.meta.root_names()
        for name in columns:
            model._contract_record_borrow(name)


class _BaseView:
    """Attribute/assignment protocol shared by every view type.

    Subclasses implement :meth:`_ids_series` (the agents this view covers).
    Reads join against ``model.agents_df``; writes go through
    :meth:`_write_column` / :meth:`set` / :meth:`scatter_add`.
    """

    # Subclasses override.
    def _ids_series(self) -> pl.Series:
        raise NotImplementedError

    # --- attribute protocol -------------------------------------------------

    def __getattr__(self, name: str):
        return self._get_view_attribute(name)

    def _get_view_attribute(self, name: str, *, broadcast: bool = True):
        if name.startswith("_") or name in _INTERNAL_ATTRS:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        model = self.__dict__.get("model")
        if model is None:
            raise AttributeError(name)
        df = model._frame_for_columns()
        has_column = name in df.columns
        # Tracked agents may define step() alongside the creation-step column.
        if broadcast and (name == "step" or not has_column):
            root = self._root()
            agents = getattr(root, "_agent_objects", None)
            if agents:
                method = getattr(agents[0], name, None)
                if callable(method):
                    def _dispatch(*args, **kwargs):
                        if not args and not kwargs and (count := try_batch(self, name)):
                            return np.full(count, None, dtype=object)
                        results = self._call_agents(name, args, kwargs, optional=False)
                        try:
                            return np.array(results)
                        except (ValueError, TypeError):
                            return results
                    return _dispatch
        if not has_column:
            raise AttributeError(
                f"{type(self).__name__!r} has no column {name!r}; "
                f"available columns: {df.columns}"
            )
        model._contract_record_borrow(name)
        ids = self._ids_series()
        from .device_columns import DeviceColumn
        from .execution import device_column_names, get_device_column

        resident = name in device_column_names(model)
        # Full population: return the column without a join.
        if _is_full_population(ids, df):
            if type(self) is AgentList:
                return DeviceColumn(model, name) if resident else df[name]
        # Align Series with the view's id order (scatter views may repeat ids).
        # When name is "id", skip the join to avoid a duplicate-column error.
        if name == "id":
            return ids
        if resident:
            from .gpu import to_host

            positions = resolve_positions(model, df, ids.to_numpy())
            host = to_host(get_device_column(model, name)[positions])
            return pl.Series(name, host)
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

        Path selection (fast → slow):

        1. Polars expression → filter + join
        2. Full population → single ``with_columns``
        3. Unique subset + numeric column → Numba/NumPy scatter-write
        4. Fallback → ``df.update(on='id')``
        """
        model = self.__dict__["model"]
        ids = self._ids_series()
        n = ids.len()
        df = model._frame_for_columns()
        full, positions = _ordinary_targets(model, ids, df)

        # --- expression path ------------------------------------------------
        if isinstance(value, pl.Expr):
            df = model.agents_df
            _record_expression_reads(model, df, value)
            sub = df.filter(pl.col("id").is_in(ids.to_list())).select(
                pl.col("id"), value.alias("__new__")
            )
            df = _ensure_columns(df, [name])
            joined = df.join(sub, on="id", how="left").with_columns(
                pl.when(pl.col("__new__").is_not_null())
                .then(pl.col("__new__"))
                .otherwise(pl.col(name))
                .alias(name)
            ).drop("__new__")
            model._set_frame(joined, written_columns=[name], target_ids=ids)
            return

        # --- GPU whole-population: accept device/host arrays without Polars ---
        if full and type(self) is AgentList:
            from .device_columns import device_scatter_write, model_uses_device_columns
            from .execution import active_xp, device_column_names

            if (
                model_uses_device_columns(model)
                and name in device_column_names(model)
            ):
                xp = active_xp(model)
                pos = xp.arange(n, dtype=xp.int64)
                if isinstance(value, pl.Series):
                    vals = xp.asarray(value.to_numpy())
                elif hasattr(value, "shape"):
                    vals = xp.asarray(value)
                elif isinstance(value, list):
                    vals = xp.asarray(value)
                else:
                    vals = xp.full(n, value)
                if getattr(vals, "shape", ()) and int(vals.shape[0]) != n:
                    raise ValueError(
                        f"Cannot assign array of length {vals.shape[0]} "
                        f"to view of length {n} for column {name!r}"
                    )
                device_scatter_write(
                    model, name, pos, vals, positions_on_device=True, target_ids=ids
                )
                return

        values = _value_to_series(name, value, n)

        # --- whole-population fast path (root AgentList only) ---------------
        if full and type(self) is AgentList:
            _commit_with_columns(
                model, df, [values.alias(name)], written_columns=[name], target_ids=ids
            )
            return

        # --- subset scatter-write (unique ids, existing numeric column) -----
        if n > 0 and name in df.columns:
            try:
                if positions is None:  # a subview in full table order
                    positions = np.arange(n, dtype=np.int64)
                vals = values.to_numpy()
                from .device_columns import (
                    device_scatter_write,
                    model_uses_device_columns,
                )
                from .execution import active_xp, device_column_names

                if (
                    model_uses_device_columns(model)
                    and name in device_column_names(model)
                    and vals.dtype != object
                ):
                    xp = active_xp(model)
                    device_scatter_write(
                        model,
                        name,
                        xp.asarray(positions, dtype=xp.int64),
                        xp.asarray(vals),
                        positions_on_device=True,
                        target_ids=ids,
                    )
                    return
                df = model.agents_df
                base = df[name].to_numpy()
                # Object / mixed columns fall through to the join path.
                if base.dtype != object and vals.dtype != object:
                    out = apply_scatter_write(base.copy(), positions, vals)
                    _commit_with_columns(
                        model,
                        df,
                        [pl.Series(name, out, strict=False)],
                        written_columns=[name],
                        target_ids=ids,
                    )
                    return
            except (TypeError, ValueError):
                pass

        # --- general update join --------------------------------------------
        df = model.agents_df
        df = _ensure_columns(df, [name])
        update_df = pl.DataFrame([ids.rename("id"), values.rename(name)])
        model._set_frame(
            df.update(update_df, on="id", how="left"), written_columns=[name], target_ids=ids
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
            _record_expression_reads(model, df, predicate)
            new_ids = sub["id"]
        elif hasattr(predicate, "dtype") and getattr(predicate.dtype, "kind", "") in (
            "b",
            "?",
        ):
            from .gpu import to_host

            mask = np.asarray(to_host(predicate), dtype=bool).ravel()
            if mask.size != base_ids.len():
                raise ValueError(
                    f"Boolean mask length {mask.size} does not match "
                    f"view length {base_ids.len()}"
                )
            new_ids = base_ids.filter(pl.Series("mask", mask))
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
        if not args and not kwargs and (count := try_batch(self, method_name)):
            return np.full(count, None, dtype=object)
        results = self._call_agents(method_name, args, kwargs, optional=True)
        try:
            return np.array(results)
        except ValueError:
            return results

    def _call_agents(self, method_name, args, kwargs, *, optional):
        model = self.__dict__["model"]
        monitor = model._contract
        ordered = self
        if monitor.active:
            monitor.begin_sweep(self, method_name, single_phase=not args and not kwargs)
        if monitor._in_replay:
            from .scheduling import _replay_order

            override = model._activation_override_order
            if override is None:
                raise ValueError("replay requires one activation sweep")
            ordered = _replay_order(list(self), model, override)
        traced = monitor.active or monitor._in_replay
        results = []
        with model._watch_sweep_rng():
            for agent in ordered:
                aid = getattr(agent, "id", None) if traced else None
                if aid is not None:
                    if monitor._in_replay:
                        monitor._current_activation = aid
                    else:
                        monitor.begin_activation(aid)
                try:
                    if optional:
                        method = getattr(agent, method_name, None)
                        if callable(method):
                            results.append(method(*args, **kwargs))
                    else:
                        results.append(getattr(agent, method_name)(*args, **kwargs))
                finally:
                    if aid is not None:
                        if monitor._in_replay:
                            monitor._current_activation = None
                        else:
                            monitor.end_activation()
        return results

    def random(self, n: int = 1, replace: bool = False):
        """Sample agent id(s) from this view (AgentPy-style helper).

        Returns a single id when ``n == 1``, otherwise a list of ids.
        Uses ``model.rng`` when available.
        """
        ids = self._ids_series().to_list()
        if not ids:
            raise ValueError("cannot sample from an empty agent view")
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        model = self.__dict__["model"]
        rng = getattr(model, "rng", None)
        if rng is None:
            rng = np.random.default_rng()
        if n == 1 and not replace:
            return int(rng.choice(ids))
        if not replace and n > len(ids):
            raise ValueError(
                f"cannot sample {n} unique agents from a view of size {len(ids)}"
            )
        picked = rng.choice(ids, size=n, replace=replace)
        return [int(x) for x in np.atleast_1d(picked)]

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

    def _named_column(self, name: str):
        if name == "step":
            return self._get_view_attribute(name, broadcast=False)
        return getattr(self, name)

    def _column_array(self, name: str):
        """Column as ndarray — zero-copy on GPU via :class:`~ambr.device_columns.DeviceColumn`."""
        col = self._named_column(name)
        if hasattr(col, "array"):
            return col.array
        self.__dict__["model"]._contract_record_mutable_borrow(name)
        return col.to_numpy()

    def array(self, *columns: str):
        """Return columns as NumPy/CuPy arrays aligned to this view.

        On ``model.gpu().run()``, numeric columns are device-resident and
        returned without a host round-trip. Use :meth:`numpy` when you need
        host ``ndarray`` outputs.
        """
        arrays = tuple(self._column_array(c) for c in columns)
        return arrays[0] if len(columns) == 1 else arrays

    def numpy(self, *columns: str):
        """Return the named columns as host numpy arrays, aligned to this view.

        ``x = agents.numpy('x')`` returns one array; ``x, y = agents.numpy('x',
        'y')`` returns a tuple -- a one-call replacement for the repeated
        ``agents.x.to_numpy()`` idiom.
        """
        arrays = tuple(self._named_column(c).to_numpy() for c in columns)
        return arrays[0] if len(columns) == 1 else arrays

    def set(self, **columns: Any) -> None:
        """Write one or more whole columns over this view's agents atomically.

        ``agents.set(x=nx, y=ny)`` is a single ``with_columns`` / update (one
        contract commit per column), not a loop of independent assignments.
        Expression values (``pl.Expr``) fall back to per-column writes.
        """
        if not columns:
            return
        # Expressions need the per-column path (filter + eval).
        if any(isinstance(v, pl.Expr) for v in columns.values()):
            for name, value in columns.items():
                self._write_column(name, value)
            return

        model = self.__dict__["model"]
        model._flush_pending_writes()
        ids = self._ids_series()
        n = ids.len()
        df = model.agents_df
        full, _ = _ordinary_targets(model, ids, df)

        series_by_name: Dict[str, pl.Series] = {
            name: _value_to_series(name, value, n)
            for name, value in columns.items()
        }
        names = list(series_by_name)

        # Whole-population: one with_columns for all columns.
        if full:
            _commit_with_columns(
                model,
                df,
                [s.alias(name) for name, s in series_by_name.items()],
                written_columns=names,
                target_ids=ids,
            )
            return

        # Subset: ensure columns exist, then hash-join update.
        df = _ensure_columns(df, names)
        update_df = pl.DataFrame(
            [ids.rename("id")]
            + [s.rename(name) for name, s in series_by_name.items()]
        )
        model._set_frame(
            df.update(update_df, on="id", how="left"),
            written_columns=names,
            target_ids=ids,
        )

    def update_where(self, predicate, **columns: Any) -> None:
        """Filter then write — one-liner sugar for the vectorized lane.

        Equivalent to ``self.where(predicate).set(**columns)``::

            agents.update_where(agents.wealth > 0, wealth=agents.wealth - 1)
        """
        self.where(predicate).set(**columns)

    # --- scatter-add --------------------------------------------------------

    def _device_ids_array(self) -> Any | None:
        """Device-resident id list for a scatter view (GPU fast path), if any."""
        return getattr(self, "_device_ids", None)

    def scatter_add(self, **increments: Any) -> None:
        """Accumulate per-id deltas into columns, summing across duplicate ids.

        ``view.at[[1, 1, 3]].scatter_add(wealth=1)`` gives agent ``1`` a +2
        and agent ``3`` a +1. Accepts the same value shapes as column
        assignment.

        Uses :func:`ambr._id_index.resolve_positions` and
        :func:`ambr.performance.apply_scatter_add` (Numba on CPU when installed).
        Deliberately omits ``written_columns`` so the contract treats scatter
        as the sanctioned multi-write reducer (not a partial-map conflict).
        """
        self._scatter_reduce(increments, "+")

    def scatter_max(self, **values: Any) -> None:
        """Reduce each target by maximum, starting new columns at zero."""
        self._scatter_reduce(values, "max")

    def _record_scatter_targets(self, columns, operator: str) -> None:
        monitor = self.__dict__["model"]._contract
        if not monitor.active or (monitor._current_activation is None and self.model.mode != "oop"):
            return
        device_ids = self._device_ids_array()
        if device_ids is not None:
            from .gpu import to_host

            ids = to_host(device_ids).tolist()
        else:
            ids = self._ids_series().to_list()
        monitor.record_activation_reductions(columns, ids, operator)

    def _scatter_reduce(self, values, operator: str) -> None:
        if not values:
            return
        model = self.__dict__["model"]
        model._flush_pending_writes()

        from .device_columns import (
            device_resolve_positions,
            device_scatter_reduce,
            model_uses_device_columns,
        )

        device_ids = self._device_ids_array()
        from .execution import active_xp, device_column_names

        if (
            model_uses_device_columns(model)
            and device_ids is not None
            and all(col in device_column_names(model) for col in values)
        ):
            n = int(device_ids.size)
            if n == 0:
                return
            positions = device_resolve_positions(model, device_ids)
            xp = active_xp(model)
            for col_name, val in values.items():
                delta = _normalize_delta(col_name, val, n, xp=xp)
                device_scatter_reduce(
                    model,
                    col_name,
                    positions,
                    delta,
                    operator,
                    positions_on_device=True,
                )
            self._record_scatter_targets(values, operator)
            return

        if device_ids is None:
            ids = self._ids_series()
        else:
            from .gpu import to_host

            ids = pl.Series("id", to_host(device_ids))
        n = ids.len()
        df = model._frame_for_columns()
        if n == 0:
            missing = [name for name in values if name not in df.columns]
            if missing:
                from .execution import sync_all_device_columns

                sync_all_device_columns(model, dirty_only=True)
                model._set_frame(_ensure_columns(model.population.data, missing))
            return

        positions = resolve_positions(model, df, ids.to_numpy())

        if model_uses_device_columns(model) and all(
            col in device_column_names(model) for col in values
        ):
            xp = active_xp(model)
            for col_name, val in values.items():
                delta = _normalize_delta(col_name, val, n, xp=xp)
                device_scatter_reduce(model, col_name, positions, delta, operator)
            self._record_scatter_targets(values, operator)
            return

        df = model.agents_df
        # A mixed reduction can include columns that lack a device buffer.
        # The host fold must see current values for its resident inputs.
        from .execution import active_execution

        execution = active_execution(model)
        resident = set()
        if execution is not None and not execution.tracked_agents:
            resident = values.keys() & execution.device_columns.keys()

        delta_np: Dict[str, np.ndarray] = {
            col: _normalize_delta(col, val, n) for col, val in values.items()
        }

        new_columns: List[pl.Series] = []
        domains: Dict[str, str] = {}
        for col_name, delta in delta_np.items():
            if col_name in df.columns:
                # Copy so we never mutate Polars' backing buffer in place.
                base = df[col_name].to_numpy().copy()
            else:
                base = np.zeros(df.height, dtype=delta.dtype)
            if operator == "+":
                out = apply_scatter_add(base, positions, delta)
            else:
                np.maximum.at(base, positions, delta)
                out = base
            domains[col_name] = str(getattr(out, "dtype", delta.dtype))
            new_columns.append(pl.Series(col_name, out, strict=False))

        # Reductions may commit repeatedly within a step.
        model._set_frame(df.with_columns(new_columns))
        for column in new_columns:
            if column.name in resident:
                current = execution.device_columns[column.name]
                uploaded = execution.xp.asarray(column.to_numpy())
                if current.dtype == uploaded.dtype:
                    current[:] = uploaded
                else:
                    execution.device_columns[column.name] = uploaded
        model._contract_record_reduction(values, domains=domains, operator=operator)
        self._record_scatter_targets(values, operator)


def _scatter_view_from_key(model: Model, view: "_BaseView", key) -> "ScatterAgentList":
    """Build a :class:`ScatterAgentList`, keeping device id buffers on GPU when active."""
    from .device_columns import model_uses_device_columns

    from .execution import active_xp

    if model_uses_device_columns(model):
        xp = active_xp(model)
        ids = None
        if isinstance(key, pl.Series):
            key = key.to_numpy()
        if isinstance(key, (int, np.integer, list, np.ndarray)):
            key = np.asarray(key, dtype=np.int64).ravel()
            ids = pl.Series("id", key)
        device_ids = xp.asarray(key, dtype=xp.int64).ravel()
        scatter = ScatterAgentList(model, ids, parent=view._root())
        object.__setattr__(scatter, "_device_ids", device_ids)
        return scatter

    if isinstance(key, (int, np.integer)):
        ids = pl.Series("id", [int(key)])
    elif isinstance(key, pl.Series):
        ids = key.rename("id")
    elif isinstance(key, list):
        ids = pl.Series("id", list(key))
    elif isinstance(key, np.ndarray):
        ids = pl.Series("id", key.tolist())
    else:
        from .gpu import to_host

        try:
            arr = np.asarray(to_host(key)).ravel()
        except (TypeError, ValueError):
            raise TypeError(
                f"at[...] accepts int, list, ndarray, or Series "
                f"(got {type(key).__name__})"
            ) from None
        ids = pl.Series("id", arr.tolist())
    return ScatterAgentList(model, ids, parent=view._root())


class _AtIndexer:
    """``view.at[ids]`` -> ScatterAgentList keyed by those ids."""

    def __init__(self, view: "_BaseView"):
        object.__setattr__(self, "_view", view)

    def __getitem__(self, key) -> "ScatterAgentList":
        view: _BaseView = self._view
        model = view.__dict__["model"]
        return _scatter_view_from_key(model, view, key)


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
        df = self.model._frame_for_columns()
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
            df = model._frame_for_columns()
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

    def __init__(self, model: Model, ids: Optional[pl.Series], parent: AgentList):
        object.__setattr__(self, "model", model)
        if ids is not None and ids.name != "id":
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
            id_list = self._ids_series().to_list()
            aid = id_list[int(idx)]
            lookup = getattr(self._root(), "_agents_by_id", None) or {}
            agent = lookup.get(aid)
            if agent is not None:
                return agent
            raise IndexError(f"Agent id={aid} not found in AgentList")
        if isinstance(idx, slice):
            id_list = self._ids_series().to_list()[idx]
            lookup = getattr(self._root(), "_agents_by_id", None) or {}
            return [lookup.get(aid) for aid in id_list]
        if isinstance(idx, (list, np.ndarray)):
            arr = np.asarray(idx)
            id_list = self._ids_series().to_list()
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
        return f"{type(self).__name__}({self._ids_series().len()} agents)"


class FilteredAgentList(_SubView):
    """Subset view produced by ``agents.where(...)`` / ``agents[mask]``."""


class ScatterAgentList(_SubView):
    """Id-indexed view produced by ``agents.at[ids]`` (ids may repeat)."""

    def _ids_series(self) -> pl.Series:
        if self._ids is None:
            from .gpu import to_host

            object.__setattr__(self, "_ids", pl.Series("id", to_host(self._device_ids)))
        return self._ids


# Save the implementations before users can replace a reducer or its indexer.
_SCATTER_SEAMS = (
    (AgentList, "at", AgentList.at),
    (_AtIndexer, "__getitem__", _AtIndexer.__getitem__),
    (ScatterAgentList, "scatter_add", ScatterAgentList.scatter_add),
    (ScatterAgentList, "scatter_max", ScatterAgentList.scatter_max),
    (ScatterAgentList, "_scatter_reduce", ScatterAgentList._scatter_reduce),
)
