"""Lightweight plotting helpers (optional use; matplotlib already a core dep).

Install is covered by the base ``ambr`` dependency on matplotlib. The
``ambr[viz]`` extra is an alias for documentation ("I want plotting helpers").

These helpers are **not** a Solara/dashboard product — they export common
charts from :class:`~ambr.results.RunResults` / agent tables so tutorials
do not need boilerplate.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Union

__all__ = ["plot_timeseries", "plot_grid", "HAS_MATPLOTLIB"]

try:
    import matplotlib

    matplotlib.use("Agg", force=False)
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

    # Polars → columns via to_pandas when available for simple plotting
    if hasattr(frame, "to_pandas"):
        df = frame.to_pandas()
    elif hasattr(frame, "columns") and hasattr(frame, "__getitem__"):
        # already pandas-like
        df = frame
    else:
        raise TypeError(
            "plot_timeseries expects RunResults or a Polars/pandas model frame"
        )

    if columns is None:
        columns = [
            c
            for c in df.columns
            if c != x and getattr(df[c], "dtype", None) is not None
        ]
        # prefer numeric
        num_cols: List[str] = []
        for c in columns:
            try:
                if str(df[c].dtype).startswith(("float", "int", "UInt", "Int", "Float")):
                    num_cols.append(c)
                elif hasattr(df[c], "dtype") and "float" in str(df[c].dtype).lower():
                    num_cols.append(c)
            except Exception:
                continue
        columns = num_cols or [c for c in df.columns if c != x]

    if ax is None:
        _, ax = plt.subplots()
    xs = df[x] if x in df.columns else range(len(df))
    for c in columns:
        if c in df.columns:
            ax.plot(xs, df[c], label=str(c), **plot_kwargs)
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

    if hasattr(frame, "to_pandas"):
        df = frame.to_pandas()
    else:
        df = frame

    for col in (x, y):
        if col not in df.columns:
            raise KeyError(
                f"plot_grid requires column {col!r}; available: {list(df.columns)}"
            )

    if ax is None:
        _, ax = plt.subplots()
    c = df[color] if color and color in df.columns else None
    sc = ax.scatter(df[x], df[y], c=c, s=s, **scatter_kwargs)
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
