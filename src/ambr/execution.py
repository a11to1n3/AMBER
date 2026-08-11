"""Run-time execution placement for :class:`~ambr.model.Model`.

Keras-style API on the model surface::

    model.cpu().run()
    model.cpu(mode="vectorized").run()
    model.cpu(mode="oop").run()
    model.gpu().run()
    model.gpu(mode="vectorized").run()

Mode may also be passed to :meth:`~ambr.model.Model.run` (overrides fluent)::

    model.cpu().run(mode="vectorized")

This module owns the *active* execution state for a run (device-resident
columns, array module, device RNG). User intent lives on the model as
``_device`` / ``_execution_mode``; :func:`begin_execution` materialises that
into ``model._execution`` for the step loop and tears it down in
:func:`end_execution`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import polars as pl

from ._deprecation import warn_deprecated
from .gpu import get_array_module, make_device_rng, require_gpu, synchronize, to_device, to_host

if TYPE_CHECKING:
    from .model import Model

EXECUTION_DEVICES = ("cpu", "gpu")
EXECUTION_MODES = ("vectorized", "oop")


@dataclass
class ExecutionConfig:
    """Resolved device + lane for one :meth:`~ambr.model.Model.run`."""

    device: str = "cpu"
    mode: str = "vectorized"

    def __post_init__(self) -> None:
        self.device = self.device.lower()
        self.mode = self.mode.lower()
        if self.device not in EXECUTION_DEVICES:
            raise ValueError(
                f"device must be one of {EXECUTION_DEVICES}, got {self.device!r}"
            )
        if self.mode not in EXECUTION_MODES:
            raise ValueError(
                f"mode must be one of {EXECUTION_MODES}, got {self.mode!r}"
            )


@dataclass
class ActiveExecution:
    """Live execution state for the current run (attached to ``model._execution``)."""

    config: ExecutionConfig
    xp: Any = np
    device_columns: Dict[str, Any] = field(default_factory=dict)
    # Columns mutated during the active run.  Keeping this separate from the
    # resident-column set lets GPU teardown flush only changed state; static
    # columns such as ``id`` do not need a PCIe round-trip every run.
    dirty_columns: set[str] = field(default_factory=set)
    device_rng: Any = None
    ids_are_arange: bool = False


def resolve_config(
    model: "Model",
    *,
    device: Optional[str] = None,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
) -> ExecutionConfig:
    """Merge fluent placement, ``run()`` kwargs, and ``model.p`` defaults."""
    if backend is not None and device is None:
        warn_deprecated(
            "run(backend=...)",
            "model.cpu()/model.gpu() or run(device=...)",
        )
    resolved_device = (
        device
        or backend
        or getattr(model, "_device", None)
        or model.p.get("device")
        or model.p.get("backend")
        or "cpu"
    ).lower()
    resolved_mode = (
        mode
        or getattr(model, "_execution_mode", None)
        or model.p.get("mode")
        or "vectorized"
    ).lower()
    return ExecutionConfig(device=resolved_device, mode=resolved_mode)


def active_execution(model: "Model") -> Optional[ActiveExecution]:
    return getattr(model, "_execution", None)


def is_gpu_active(model: "Model") -> bool:
    ex = active_execution(model)
    return ex is not None and ex.config.device == "gpu" and bool(ex.device_columns)


def device_column_names(model: "Model") -> frozenset[str]:
    ex = active_execution(model)
    if ex is None:
        return frozenset()
    return frozenset(ex.device_columns)


def get_device_column(model: "Model", name: str) -> Any:
    ex = active_execution(model)
    if ex is None:
        raise KeyError(name)
    return ex.device_columns[name]


def active_xp(model: "Model"):
    ex = active_execution(model)
    return ex.xp if ex is not None else np


def active_rng(model: "Model"):
    ex = active_execution(model)
    if ex is not None and ex.config.device == "gpu" and ex.device_rng is not None:
        return ex.device_rng
    return model._host_rng


def _numeric_host_array(series: pl.Series) -> np.ndarray | None:
    try:
        arr = series.to_numpy()
    except Exception:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return arr


def _validate_mode(model: "Model", config: ExecutionConfig) -> None:
    if config.device == "gpu" and config.mode == "oop":
        raise ValueError(
            "GPU execution supports mode='vectorized' only; "
            "use cpu(mode='oop') for Python Agent objects."
        )
    if config.mode != "oop":
        return
    agents = getattr(model, "agents", None)
    tracked = getattr(agents, "_agent_objects", None) if agents is not None else None
    if not tracked:
        import warnings

        warnings.warn(
            "run(mode='oop') but no Agent objects are tracked; "
            "use add_agents(..., agent_class=...) or the vectorized view API.",
            UserWarning,
            stacklevel=3,
        )


def begin_execution(model: "Model", config: ExecutionConfig) -> None:
    """Start a run under ``config`` (upload GPU columns when ``device='gpu'``)."""
    if active_execution(model) is not None:
        raise RuntimeError("begin_execution called while a run is already active")
    _validate_mode(model, config)
    ex = ActiveExecution(config=config, xp=np)
    if config.device == "gpu":
        require_gpu()
        ex.xp = get_array_module(True)
        df = model.agents_df
        for name in df.columns:
            if name == "step":
                continue
            arr = _numeric_host_array(df[name])
            if arr is not None:
                ex.device_columns[name] = to_device(arr)
        from ._id_index import ids_are_arange

        if "id" in df.columns and df.height:
            ex.ids_are_arange = ids_are_arange(model, df["id"].to_numpy())
        ex.device_rng = make_device_rng(model.p.get("seed"))
    model._execution = ex


def begin_fast_execution(
    model: "Model",
    config: ExecutionConfig,
    device_columns: Dict[str, Any],
    *,
    device_rng: Any = None,
) -> None:
    """Start a device-first run supplied by a private model fast path.

    Unlike :func:`begin_execution`, this does not upload a pre-existing
    Polars frame.  It is used only by built-in optimized models that create
    their initial numeric columns directly on the selected device; the public
    ``model.gpu().run()`` contract and normal teardown remain the same.
    """
    if active_execution(model) is not None:
        raise RuntimeError("begin_fast_execution called while a run is already active")
    _validate_mode(model, config)
    require_gpu()
    ex = ActiveExecution(
        config=config,
        xp=get_array_module(True),
        device_columns=device_columns,
        # Fast setup starts from an id-only placeholder frame, so every
        # device-created column must be materialized at teardown even for a
        # zero-step run.
        dirty_columns=set(device_columns),
        device_rng=device_rng or make_device_rng(model.p.get("seed")),
        ids_are_arange=("id" in device_columns),
    )
    model._execution = ex


def sync_device_column_to_frame(model: "Model", name: str) -> None:
    ex = active_execution(model)
    if ex is None or name not in ex.device_columns:
        return
    host = to_host(ex.device_columns[name])
    df = model.population.data
    model.population.replace_frame(df.with_columns(pl.Series(name, host)))


def sync_all_device_columns(model: "Model", *, dirty_only: bool = False) -> None:
    ex = active_execution(model)
    if ex is None or ex.config.device != "gpu":
        return
    names = ex.dirty_columns if dirty_only else ex.device_columns
    for name in list(names):
        sync_device_column_to_frame(model, name)
    if dirty_only:
        ex.dirty_columns.clear()


def end_execution(model: "Model") -> None:
    """Flush device columns (if any) and clear active execution state.

    ``model._execution`` is always cleared, even when host sync / device
    synchronize fails, so a failed teardown cannot leave a stale active
    execution behind. Sync errors are re-raised after the clear.
    """
    ex = active_execution(model)
    if ex is None:
        return
    sync_error: Optional[BaseException] = None
    try:
        if ex.config.device == "gpu":
            # Device columns are the canonical state during the run.  Only flush
            # columns that the model actually wrote; unchanged columns remain in
            # the original Polars frame and need no host transfer.
            try:
                sync_all_device_columns(model, dirty_only=True)
                synchronize()
            except Exception as exc:
                sync_error = exc
    finally:
        # Always clear — never leave _execution active after teardown.
        model._execution = None
    if sync_error is not None:
        raise sync_error
