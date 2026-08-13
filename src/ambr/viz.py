"""Lightweight plotting helpers (optional; install ``ambr[viz]``).

These helpers are **not** a Solara/dashboard product — they export common
charts from :class:`~ambr.results.RunResults` / agent tables so tutorials
do not need boilerplate.

This module is loaded lazily via :func:`ambr.__getattr__` so a plain
``import ambr`` never touches matplotlib. Callers (or CI / docs builds)
should set ``MPLBACKEND=Agg`` when a non-interactive backend is required;
this module does **not** call ``matplotlib.use(...)``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

__all__ = ["plot_timeseries", "plot_grid", "HAS_MATPLOTLIB"]

try:
    import matplotlib.pyplot as plt

    # Probe a real backend symbol — some envs import half-broken matplotlib.
    _ = plt.subplots
    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover - missing or NumPy-ABI-broken installs
    plt = None  # type: ignore
    HAS_MATPLOTLIB = False


def _require_mpl():
    if not HAS_MATPLOTLIB or plt is None:
        raise ImportError(
            "matplotlib is required for ambr.viz helpers. "
            "Install with: pip install matplotlib  (or pip install 'ambr[viz]')"
        )


def _as_frame(obj: Any):
    """Accept RunResults, mapping with 'model'/'agents', or a Polars/pandas frame."""
    if obj is None:
        raise ValueError("results/frame is None")
    if hasattr(obj, "model") and not hasattr(obj, "columns"):
        # RunResults-like
        return obj
    return obj


def plot_timeseries(
    results: Any,
    columns: Optional[Sequence[str]] = None,
    *,
    x: str = "t",
    ax: Any = None,
    title: Optional[str] = None,
    **plot_kwargs: Any,
) -> Any:
    """Plot model-level time series from ``results.model`` / ``results['model']``.

    Args:
        results: :class:`~ambr.results.RunResults` or a frame with a time column.
        columns: Metric columns to plot (default: all numeric except ``x``).
        x: Time column name (default ``'t'``).
        ax: Optional matplotlib Axes.
        title: Optional title.
        **plot_kwargs: Forwarded to ``Axes.plot``.

    Returns:
        The matplotlib Axes used.
    """
    _require_mpl()
    frame = results
    if hasattr(results, "__getitem__") and not hasattr(results, "columns"):
        try:
            frame = results["model"]
        except Exception:
            frame = getattr(results, "model", results)

    if not hasattr(frame, "columns"):
        raise TypeError(
            "plot_timeseries expects RunResults or a Polars/pandas model frame"
        )

    col_names = list(frame.columns)
    if columns is None:
        num_cols: List[str] = []
        for c in col_names:
            if c == x:
                continue
            try:
                series = frame[c]
                dtype = str(getattr(series, "dtype", ""))
                if any(t in dtype for t in ("float", "int", "UInt", "Int", "Float")):
                    num_cols.append(c)
            except Exception:
                continue
        columns = num_cols or [c for c in col_names if c != x]

    if ax is None:
        _, ax = plt.subplots()
    if x in col_names:
        xs = _series_to_list(frame[x])
    else:
        xs = list(range(_frame_height(frame)))
    for c in columns:
        if c in col_names:
            ax.plot(xs, _series_to_list(frame[c]), label=str(c), **plot_kwargs)
    ax.set_xlabel(x)
    if title:
        ax.set_title(title)
    if columns:
        ax.legend()
    return ax


def plot_grid(
    agents: Any,
    *,
    x: str = "x",
    y: str = "y",
    color: Optional[str] = None,
    ax: Any = None,
    title: Optional[str] = None,
    s: float = 20.0,
    **scatter_kwargs: Any,
) -> Any:
    """Scatter-plot agent positions from an agents DataFrame or RunResults.

    Args:
        agents: ``results.agents`` / ``results['agents']`` or a frame with
            ``x``/``y`` columns.
        x, y: Coordinate column names.
        color: Optional column for point colors.
        ax: Optional matplotlib Axes.
        title: Optional title.
        s: Marker size.
        **scatter_kwargs: Forwarded to ``Axes.scatter``.

    Returns:
        The matplotlib Axes used.
    """
    _require_mpl()
    frame = agents
    if hasattr(agents, "__getitem__") and not hasattr(agents, "columns"):
        try:
            frame = agents["agents"]
        except Exception:
            frame = getattr(agents, "agents", agents)

    if not hasattr(frame, "columns"):
        raise TypeError("plot_grid expects RunResults or a frame with columns")

    col_names = list(frame.columns)
    for col in (x, y):
        if col not in col_names:
            raise KeyError(
                f"plot_grid requires column {col!r}; available: {col_names}"
            )

    if ax is None:
        _, ax = plt.subplots()
    xs = _series_to_list(frame[x])
    ys = _series_to_list(frame[y])
    c = _series_to_list(frame[color]) if color and color in col_names else None
    sc = ax.scatter(xs, ys, c=c, s=s, **scatter_kwargs)
    if c is not None:
        try:
            plt.colorbar(sc, ax=ax, label=color)
        except Exception:
            pass
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def _series_to_list(series: Any) -> list:
    if hasattr(series, "to_list"):
        return series.to_list()
    if hasattr(series, "to_numpy"):
        return list(series.to_numpy())
    return list(series)


def _frame_height(frame: Any) -> int:
    if hasattr(frame, "height"):
        return int(frame.height)
    if hasattr(frame, "__len__"):
        return len(frame)
    return 0
