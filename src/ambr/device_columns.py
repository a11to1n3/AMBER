"""Device-resident column views and GPU scatter kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .execution import (
    active_execution,
    is_gpu_active,
    sync_all_device_columns,
)
from .gpu import scatter_add, to_host

if TYPE_CHECKING:
    from .model import Model


class DeviceColumn:
    """Stand-in for a Polars column backed by a device-resident array."""

    def __init__(self, model: "Model", name: str):
        self._model = model
        self._name = name

    def _array(self):
        ex = active_execution(self._model)
        if ex is None or self._name not in ex.device_columns:
            raise KeyError(self._name)
        self._model._contract_record_borrow(self._name)
        return ex.device_columns[self._name]

    @property
    def array(self):
        """Zero-copy access to the backing NumPy or CuPy column."""
        array = self._array()
        # ``agents.array(...)`` is a mutable zero-copy borrow.  Marking the
        # column dirty on borrow is conservative but preserves correctness for
        # callers that mutate the returned CuPy array in place, which cannot be
        # observed by the assignment/scatter hooks below.
        ex = active_execution(self._model)
        if ex is not None and ex.config.device == "gpu":
            ex.dirty_columns.add(self._name)
        self._model._contract_record_mutable_borrow(self._name)
        return array

    def to_numpy(self) -> np.ndarray:
        """Explicit host export (PCIe round-trip on GPU)."""
        return to_host(self._array())

    def sum(self) -> Any:
        return to_host(self._array().sum()).item()

    def mean(self) -> Any:
        return to_host(self._array().mean()).item()

    def __len__(self) -> int:
        return int(self._array().size)

    def _compare(self, op, other):
        return op(self._array(), other)

    def __gt__(self, other):
        return self._compare(lambda a, b: a > b, other)

    def __ge__(self, other):
        return self._compare(lambda a, b: a >= b, other)

    def __lt__(self, other):
        return self._compare(lambda a, b: a < b, other)

    def __le__(self, other):
        return self._compare(lambda a, b: a <= b, other)

    def __eq__(self, other):
        return self._compare(lambda a, b: a == b, other)

    def __ne__(self, other):
        return self._compare(lambda a, b: a != b, other)


def model_uses_device_columns(model: "Model") -> bool:
    return is_gpu_active(model)


def device_resolve_positions(model: "Model", ids_on_device: Any) -> Any:
    """Map agent ids to row positions without leaving the device when ids are 0..N-1."""
    ex = active_execution(model)
    if ex is None:
        raise RuntimeError("device_resolve_positions requires an active GPU run")
    xp = ex.xp
    ids_on_device = xp.asarray(ids_on_device, dtype=xp.int64).ravel()
    if ex.ids_are_arange:
        return ids_on_device
    from ._id_index import resolve_positions

    ids_host = np.asarray(to_host(ids_on_device), dtype=np.int64)
    positions = resolve_positions(model, model.population.data, ids_host)
    return xp.asarray(positions, dtype=xp.int64)


def device_scatter_write(
    model: "Model",
    col_name: str,
    positions: Any,
    values: Any,
    *,
    positions_on_device: bool = False,
) -> None:
    """Scatter-write into a device-resident column (last write wins per index)."""
    ex = active_execution(model)
    if ex is None:
        raise RuntimeError("device_scatter_write requires an active GPU run")
    xp = ex.xp
    base = ex.device_columns[col_name]
    pos = (
        positions
        if positions_on_device
        else xp.asarray(positions, dtype=xp.int64)
    )
    vals = (
        values
        if getattr(type(values), "__module__", "").split(".")[0] == xp.__name__
        else xp.asarray(values)
    )
    base[pos] = vals
    ex.device_columns[col_name] = base
    ex.dirty_columns.add(col_name)
    model._contract.record_commit([col_name])


def device_scatter_add(
    model: "Model",
    col_name: str,
    positions: Any,
    delta: Any,
    *,
    positions_on_device: bool = False,
) -> None:
    """Scatter-add on a device-resident column; Polars stays stale until sync."""
    ex = active_execution(model)
    if ex is None:
        raise RuntimeError("device_scatter_add requires an active GPU run")
    xp = ex.xp
    base = ex.device_columns[col_name]
    pos = (
        positions
        if positions_on_device
        else xp.asarray(positions, dtype=xp.int64)
    )
    d = (
        delta
        if getattr(type(delta), "__module__", "").split(".")[0] == xp.__name__
        else xp.asarray(delta)
    )
    scatter_add(base, pos, d)
    ex.device_columns[col_name] = base
    ex.dirty_columns.add(col_name)
    model._contract_record_reduction([col_name])


# Re-export for contract snapshots / model hooks.
__all__ = [
    "DeviceColumn",
    "device_resolve_positions",
    "device_scatter_add",
    "device_scatter_write",
    "model_uses_device_columns",
    "sync_all_device_columns",
]
