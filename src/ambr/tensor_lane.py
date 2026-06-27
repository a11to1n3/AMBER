"""Polars-backed numeric tensor lane.

A thin, **version-invariant** bridge that exposes a contiguous ``(N, F)`` float
view over a chosen set of homogeneous numeric agent columns, so that fused
elementwise updates and (especially) dense interaction kernels can run as NumPy
tensor / ``einsum`` operations and write their results back into the model's
Polars DataFrame -- which remains the single system of record.

Design goals
------------
* **One backing store.** The hot numeric attributes live *inside* Polars
  (either as one ``Array(F)`` column or as F plain columns); the tensor view
  borrows that memory rather than maintaining a parallel buffer. A borrow is a
  *point-in-time, read-only* snapshot -- committing a column rebinds its buffer,
  so outstanding borrows of that column go stale (borrow inputs, then commit).
* **Faster only where it helps.** Borrowing is most valuable for *read-heavy*
  kernels (interactions / reductions) where Polars is forced into an O(N^2)
  self-join; for trivial elementwise maps, plain Polars columns are already
  fine and this lane offers no benefit (see the project benchmarks).

Version invariance
------------------
This module **never branches on** ``polars.__version__`` and **never touches
Polars/Arrow private internals**. Instead it *probes behaviour once* at import
time:

  * does ``pl.Series(name, <2-D ndarray>)`` yield something whose ``to_numpy()``
    is a 2-D, zero-copy view? -> :data:`ARRAY_BACKING_AVAILABLE`
  * is a single numeric column's ``to_numpy()`` zero-copy? -> :data:`SINGLE_COL_ZERO_COPY`

Correctness does **not** depend on either probe succeeding: when the fast path
is unavailable the lane falls back to a copy-based path that is correct on every
Polars version. Zero-copy is an optimisation that is *measured at runtime* (via
``numpy.shares_memory``), never assumed. The only public-API calls used --
``Series.to_numpy()`` (no kwargs), ``pl.Series(name, ndarray)``,
``DataFrame.select/with_columns/drop/get_column``, ``Series.n_chunks/rechunk``
-- have been stable across the Polars 0.20 -> 1.x line.
"""

from typing import List, Optional, Tuple

import numpy as np
import polars as pl


# --- behavioural capability probes (run once, cached) ----------------------

def _probe_array_backing() -> bool:
    """True iff a 2-D ndarray round-trips through Polars as a zero-copy view."""
    try:
        a = np.arange(8, dtype=np.float64).reshape(4, 2)
        s = pl.Series("__probe__", a)
        out = s.to_numpy()
        return (
            getattr(out, "ndim", 0) == 2
            and tuple(out.shape) == (4, 2)
            and bool(np.shares_memory(out, s.to_numpy()))
        )
    except Exception:
        return False


def _probe_single_col_zero_copy() -> bool:
    """True iff a single numeric column's ``to_numpy()`` shares Polars memory."""
    try:
        s = pl.Series("__probe__", np.arange(8, dtype=np.float64))
        out = s.to_numpy()
        return bool(np.shares_memory(out, s.to_numpy()))
    except Exception:
        return False


ARRAY_BACKING_AVAILABLE: bool = _probe_array_backing()
SINGLE_COL_ZERO_COPY: bool = _probe_single_col_zero_copy()


def borrow_numeric(model, column: str) -> Tuple[np.ndarray, bool]:
    """Borrow a single numeric column as a 1-D array (read-only).

    Returns ``(array, is_view)`` where ``is_view`` reports whether the array
    shares memory with the Polars buffer (zero-copy) on this build. The array
    is **always marked read-only**, on every build, so the rule that a borrow
    must not be mutated in place holds regardless of whether the zero-copy fast
    path was taken (otherwise a stray ``x += ...`` would raise on a zero-copy
    build but silently corrupt a throwaway copy on others).

    The array is a *point-in-time snapshot*: a later :func:`commit_columns` (or
    any write) on this column rebinds the Polars buffer and leaves this array
    reading the old one. Borrow all inputs first, compute, then commit.
    """
    if getattr(model, "_contract_active", False):
        model._contract_record_borrow(column)
    df = model.agents_df  # flushes pending writes; single source of truth
    col = df.get_column(column)
    if col.n_chunks() > 1:
        # A chunked column has no single contiguous view; rechunk into a local
        # (the returned array pins this buffer). We deliberately do NOT persist
        # the rechunk -- a "borrow" must not mutate the model.
        col = col.rechunk()
    arr = col.to_numpy()
    is_view = bool(np.shares_memory(arr, col.to_numpy()))
    arr.flags.writeable = False
    return arr, is_view


class TensorLane:
    """A packed ``(N, F)`` numeric view over a model's homogeneous columns.

    Use :meth:`borrow` to read the packed tensor (zero-copy where the build
    supports it), :meth:`commit` to write a full ``(N, F)`` result back, and
    :meth:`write_result` to attach a derived 1-D column (e.g. an interaction
    output) without disturbing the pack.

    The backing strategy is chosen by the capability probe, not by version:

    * ``"array"``  -- the F columns are folded into one ``Array(F)`` column;
      :meth:`borrow` returns a zero-copy ``(N, F)`` view.
    * ``"stacked"`` -- the F columns are kept as-is; :meth:`borrow` returns a
      ``(N, F)`` copy via ``DataFrame.to_numpy`` (correct on every version).
    """

    PACKED_SUFFIX = "__packed__"

    def __init__(
        self,
        model,
        columns: List[str],
        dtype=np.float64,
        prefer_array: bool = True,
    ):
        if not columns:
            raise ValueError("TensorLane requires at least one column")
        self.model = model
        self.columns = list(columns)
        self.F = len(self.columns)
        self.dtype = dtype
        self._packed_name = "_".join(self.columns)[:48] + self.PACKED_SUFFIX
        self._mode = "array" if (prefer_array and ARRAY_BACKING_AVAILABLE) else "stacked"
        self._packed = False
        if self._mode == "array":
            self._install_array_column()

    # --- mode / introspection ----------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_zero_copy(self) -> bool:
        """Whether :meth:`borrow` returns a view (vs a copy) on this build."""
        return self._mode == "array" and ARRAY_BACKING_AVAILABLE

    def _df(self) -> pl.DataFrame:
        return self.model.agents_df

    # --- packing / unpacking ------------------------------------------------

    def _install_array_column(self) -> None:
        df = self._df()
        mat = np.ascontiguousarray(df.select(self.columns).to_numpy(), dtype=self.dtype)
        new = df.drop(self.columns).with_columns(pl.Series(self._packed_name, mat))
        self.model._set_frame(new)
        self._packed = True

    def unpack(self) -> None:
        """Restore the F named columns (for relational / named-attribute access)."""
        if self._mode != "array" or not self._packed:
            return
        df = self._df()
        if self._packed_name not in df.columns:
            return
        mat = df.get_column(self._packed_name).to_numpy()
        cols = [pl.Series(name, mat[:, i]) for i, name in enumerate(self.columns)]
        new = df.drop(self._packed_name).with_columns(cols)
        self.model._set_frame(new)
        self._packed = False

    # --- borrow / commit ----------------------------------------------------

    def borrow(self) -> Tuple[np.ndarray, bool]:
        """Return ``(arr, is_view)`` for the packed ``(N, F)`` tensor (read-only)."""
        if getattr(self.model, "_contract_active", False):
            for c in self.columns:
                self.model._contract_record_borrow(c)
        df = self._df()
        if self._mode == "array":
            col = df.get_column(self._packed_name)
            if col.n_chunks() > 1:
                col = col.rechunk()  # local; do not persist (a borrow must not mutate)
            arr = col.to_numpy()
            is_view = bool(np.shares_memory(arr, col.to_numpy()))
            arr.flags.writeable = False
            return arr, is_view
        # stacked fallback: always a correct (N, F) copy, marked read-only
        arr = np.ascontiguousarray(df.select(self.columns).to_numpy(), dtype=self.dtype)
        arr.flags.writeable = False
        return arr, False

    def commit(self, arr: np.ndarray) -> None:
        """Write a full ``(N, F)`` result back into the model's DataFrame."""
        arr = np.asarray(arr)
        if arr.shape[1] != self.F:
            raise ValueError(f"expected (N, {self.F}) array, got {arr.shape}")
        df = self._df()
        if self._mode == "array":
            self.model._set_frame(df.with_columns(
                pl.Series(self._packed_name, np.ascontiguousarray(arr, dtype=self.dtype))
            ))
        else:
            cols = [
                _cast_to_existing(df, pl.Series(name, arr[:, i]), name)
                for i, name in enumerate(self.columns)
            ]
            self.model._set_frame(df.with_columns(cols))
        if getattr(self.model, "_contract_active", False):
            self.model._contract_record_commit(self.columns)

    def write_result(self, name: str, values) -> None:
        """Attach a derived 1-D column (e.g. an interaction output)."""
        df = self._df()
        series = _cast_to_existing(df, pl.Series(name, np.asarray(values)), name)
        self.model._set_frame(df.with_columns(series))
        if getattr(self.model, "_contract_active", False):
            self.model._contract_record_commit([name])


def _cast_to_existing(df: pl.DataFrame, series: pl.Series, name: str) -> pl.Series:
    """Cast ``series`` back to ``name``'s existing dtype so a commit never
    silently changes a column's type (e.g. float math into an Int64 column)."""
    if name in df.columns:
        return series.cast(df.get_column(name).dtype)
    return series


def commit_columns(model, **columns) -> None:
    """Write several 1-D result columns back in a single, atomic ``with_columns``.

    Updating *existing* columns (rather than adding new ones) keeps the agent
    schema stable across a step -- which is what ``Model.run(contract='check')``
    verifies. Each committed column's existing dtype is preserved, so a column
    is never silently re-typed. Pre-declare these columns in ``setup`` so the
    contract certificate stays clean.

    Note: this writes via ``population.data = df.with_columns(...)`` (the
    vectorized lane path). When the contract is active the commit is reported
    to the model so duplicate same-step commits of a column are flagged.
    """
    df = model.agents_df
    series = [
        _cast_to_existing(df, pl.Series(name, np.asarray(v)), name)
        for name, v in columns.items()
    ]
    model._set_frame(df.with_columns(series))
    if getattr(model, "_contract_active", False):
        model._contract_record_commit(columns.keys())
