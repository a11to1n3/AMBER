"""AMBER performance utilities — spatial index, Numba scatters, parallel runs.

This module is the **shared home** for CPU hot-path kernels used by the
vectorized write lane (``sequences``) and optional spatial helpers:

* **Scatter kernels** (``scatter_*_1d`` / ``apply_scatter_*``) — Numba when
  installed, NumPy fallbacks otherwise. Used by subset column writes and
  ``scatter_add``.
* **SpatialIndex** — SciPy KD-Tree neighbor queries (optional).
* **ParallelRunner** — multi-process experiment fan-out.
* **Vectorized helpers** — batch move / transfer / SIR utilities.

Numba is the recommended CPU accelerator on Mac (no CUDA). Import
:data:`HAS_NUMBA` to branch; never hard-require numba at import time.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import secrets
import stat
import time
import traceback as _traceback_mod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
import multiprocessing as mp

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies (soft imports — never fail module load)
# ---------------------------------------------------------------------------

try:
    from scipy.spatial import cKDTree

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from numba import jit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # No-op stand-in so @jit still decorates cleanly without numba installed.
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args or callable(args[0]) else decorator

    prange = range


# =============================================================================
# KD-Tree Spatial Indexing
# =============================================================================


class SpatialIndex:
    """
    Fast spatial indexing using KD-Tree for O(log n) neighbor queries.

    Usage:
        index = SpatialIndex()
        index.build(positions)  # positions is Nx2 or Nx3 array
        neighbors = index.query_radius(point, radius)
        k_nearest = index.query_knn(point, k=5)
    """

    def __init__(self):
        self.tree = None
        self.positions = None

    def build(self, positions: np.ndarray) -> "SpatialIndex":
        """
        Build the spatial index from positions.

        Args:
            positions: Nx2 or NxD array of coordinates

        Returns:
            self for chaining
        """
        if not HAS_SCIPY:
            raise ImportError(
                "scipy required for SpatialIndex. Install with: pip install scipy"
            )

        self.positions = np.asarray(positions)
        self.tree = cKDTree(self.positions)
        return self

    def query_radius(self, point: np.ndarray, radius: float) -> List[int]:
        """
        Find all points within radius of query point.

        Args:
            point: Query point coordinates
            radius: Search radius

        Returns:
            List of indices of points within radius
        """
        if self.tree is None:
            raise ValueError("Index not built. Call build() first.")
        return self.tree.query_ball_point(point, radius)

    def query_knn(self, point: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors to query point.

        Args:
            point: Query point coordinates
            k: Number of neighbors to find

        Returns:
            Tuple of (distances, indices)
        """
        if self.tree is None:
            raise ValueError("Index not built. Call build() first.")
        distances, indices = self.tree.query(point, k=k)
        return distances, indices

    def query_pairs(self, radius: float) -> set:
        """
        Find all pairs of points within radius of each other.

        Args:
            radius: Maximum distance between pairs

        Returns:
            Set of (i, j) index pairs
        """
        if self.tree is None:
            raise ValueError("Index not built. Call build() first.")
        return self.tree.query_pairs(radius)

    def batch_query_radius(self, points: np.ndarray, radius: float) -> List[List[int]]:
        """
        Find neighbors for multiple query points.

        Args:
            points: MxD array of query points
            radius: Search radius

        Returns:
            List of neighbor lists for each query point
        """
        if self.tree is None:
            raise ValueError("Index not built. Call build() first.")
        return self.tree.query_ball_point(points, radius)


# =============================================================================
# Scatter kernels (Numba when available, NumPy fallbacks)
# =============================================================================
#
# Used by the vectorized write path:
#   * subset column assign  → apply_scatter_write
#   * agents.at[ids].scatter_add(...) → apply_scatter_add
#
# Low-level ``*_1d`` functions are pure loops (Numba-jitted when possible).
# High-level ``apply_*`` wrappers own contiguity, dtype casts, and fallbacks
# so call sites stay one-liners (DRY).


@jit(nopython=True, cache=True)
def scatter_add_1d(
    base: np.ndarray, positions: np.ndarray, delta: np.ndarray
) -> np.ndarray:
    """Accumulate ``delta`` into ``base`` at ``positions`` (duplicate-safe).

    Same semantics as ``np.add.at(base, positions, delta)`` but often faster
    for irregular ABM scatter patterns on CPU (including Apple Silicon).
    Mutates and returns ``base``. Prefer :func:`apply_scatter_add` at call sites.
    """
    n = positions.shape[0]
    for i in range(n):
        base[positions[i]] += delta[i]
    return base


@jit(nopython=True, cache=True)
def scatter_write_1d(
    base: np.ndarray, positions: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Write ``values`` into ``base`` at ``positions`` (last write wins).

    Mutates and returns ``base``. Prefer :func:`apply_scatter_write` at call sites.
    """
    n = positions.shape[0]
    for i in range(n):
        base[positions[i]] = values[i]
    return base


def _as_contiguous_int64_positions(positions: np.ndarray) -> np.ndarray:
    """Normalize row indices for Numba nopython kernels."""
    return np.ascontiguousarray(positions, dtype=np.int64)


def apply_scatter_add(
    base: np.ndarray,
    positions: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """Scatter-add with Numba acceleration when available.

    Falls back to ``np.add.at`` for object dtypes or when Numba is missing.
    Always returns the array holding the result (may be a new buffer if a
    dtype upcast or contiguity copy was required). Callers **must** use the
    return value::

        out = apply_scatter_add(column_copy, positions, delta)
    """
    # Object / mixed columns cannot go through nopython kernels.
    delta_dtype = getattr(delta, "dtype", None)
    if base.dtype == np.dtype(object) or delta_dtype == np.dtype(object):
        np.add.at(base, positions, delta)
        return base

    # np.add.at will not upcast the destination; expand dtype first.
    result_dtype = np.result_type(base.dtype, delta.dtype)
    if base.dtype != result_dtype:
        base = np.asarray(base, dtype=result_dtype)

    if HAS_NUMBA:
        out = np.ascontiguousarray(base)
        pos_i = _as_contiguous_int64_positions(positions)
        delta_c = np.ascontiguousarray(delta, dtype=out.dtype)
        scatter_add_1d(out, pos_i, delta_c)
        return out

    np.add.at(base, positions, delta)
    return base


def apply_scatter_write(
    base: np.ndarray,
    positions: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Scatter-write (last write wins) with Numba when available.

    Falls back to advanced indexing when Numba is missing or dtypes are object.
    Returns the array holding the result (use the return value).
    """
    values_dtype = getattr(values, "dtype", None)
    if base.dtype == np.dtype(object) or values_dtype == np.dtype(object):
        base[positions] = values
        return base

    if HAS_NUMBA:
        out = np.ascontiguousarray(base)
        pos_i = _as_contiguous_int64_positions(positions)
        vals_c = np.ascontiguousarray(values, dtype=out.dtype)
        scatter_write_1d(out, pos_i, vals_c)
        return out

    base[positions] = values
    return base


# =============================================================================
# Spatial / distance Numba helpers
# =============================================================================


@jit(nopython=True, cache=True)
def fast_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance matrix using Numba.

    Args:
        positions: Nx2 array of coordinates

    Returns:
        NxN distance matrix
    """
    n = positions.shape[0]
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            distances[i, j] = d
            distances[j, i] = d

    return distances


@jit(nopython=True, cache=True)
def fast_neighbors_within_radius(
    positions: np.ndarray, query_idx: int, radius: float
) -> List[int]:
    """
    Find all neighbors within radius of a specific agent (Numba-accelerated).

    Args:
        positions: Nx2 array of coordinates
        query_idx: Index of query agent
        radius: Search radius

    Returns:
        List of neighbor indices
    """
    n = positions.shape[0]
    radius_sq = radius * radius
    neighbors = []

    qx = positions[query_idx, 0]
    qy = positions[query_idx, 1]

    for i in range(n):
        if i == query_idx:
            continue
        dx = positions[i, 0] - qx
        dy = positions[i, 1] - qy
        dist_sq = dx * dx + dy * dy
        if dist_sq <= radius_sq:
            neighbors.append(i)

    return neighbors


@jit(nopython=True, parallel=True, cache=True)
def fast_all_neighbors_within_radius(
    positions: np.ndarray, radius: float
) -> np.ndarray:
    """
    Find all neighbor pairs within radius (Numba-parallel).

    Args:
        positions: Nx2 array of coordinates
        radius: Search radius

    Returns:
        Nx(max_neighbors) array of neighbor indices (-1 for empty slots)
    """
    n = positions.shape[0]
    radius_sq = radius * radius
    max_neighbors = min(100, n)  # Reasonable upper bound

    # Output array: each row contains neighbor indices for that agent
    neighbors = np.full((n, max_neighbors), -1, dtype=np.int64)

    for i in prange(n):
        count = 0
        for j in range(n):
            if i == j:
                continue
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist_sq = dx * dx + dy * dy
            if dist_sq <= radius_sq and count < max_neighbors:
                neighbors[i, count] = j
                count += 1

    return neighbors


@jit(nopython=True, cache=True)
def fast_random_walk_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    bounds: np.ndarray,
    wrap: bool = False,
) -> np.ndarray:
    """
    Update positions with velocities (Numba-accelerated).

    Args:
        positions: Nx2 array of positions
        velocities: Nx2 array of velocities
        bounds: 2x2 array [[x_min, x_max], [y_min, y_max]]
        wrap: Whether to wrap at boundaries

    Returns:
        Updated positions
    """
    n = positions.shape[0]
    new_positions = positions + velocities

    for i in range(n):
        for d in range(2):
            if wrap:
                range_size = bounds[d, 1] - bounds[d, 0]
                while new_positions[i, d] < bounds[d, 0]:
                    new_positions[i, d] += range_size
                while new_positions[i, d] >= bounds[d, 1]:
                    new_positions[i, d] -= range_size
            else:
                if new_positions[i, d] < bounds[d, 0]:
                    new_positions[i, d] = bounds[d, 0]
                elif new_positions[i, d] >= bounds[d, 1]:
                    new_positions[i, d] = bounds[d, 1] - 0.001

    return new_positions


# =============================================================================
# Multiprocessing Utilities
# =============================================================================


RunStatus = Literal["success", "failed", "timeout"]


@dataclass
class RunOutcome:
    """Structured per-run outcome from :class:`ParallelRunner` (input order).

    Attributes
    ----------
    index:
        Position in the original ``param_list``.
    status:
        ``success``, ``failed``, or ``timeout``.
    params:
        Parameter dict used for this run.
    result:
        On success: mapping with ``model`` / ``agents`` / ``info`` (and
        ``params``). ``None`` on failure/timeout.
    error_type / error_message / traceback:
        Populated when ``status`` is not ``success``.
    attempts:
        Number of attempts used (1 + retries).
    """

    index: int
    status: RunStatus
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    attempts: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunOutcome":
        return cls(
            index=int(data["index"]),
            status=data["status"],  # type: ignore[arg-type]
            params=dict(data.get("params") or {}),
            result=data.get("result"),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            traceback=data.get("traceback"),
            attempts=int(data.get("attempts", 1)),
        )


# Current writer schema (Arrow IPC frames + workload fingerprint).
# Schema 1 was lossy record JSON; schema 2 added IPC frames.
_CHECKPOINT_SCHEMA = 3
_LEGACY_CHECKPOINT_SCHEMAS = frozenset({1, 2})


class CheckpointSerializationError(ValueError):
    """Raised when a result frame cannot be encoded losslessly for a checkpoint."""


def _serialize_frame(df: Any) -> Optional[Dict[str, Any]]:
    """Type-preserving encoding of a Polars frame (Arrow IPC + base64).

    Record-based JSON (schema 1) loses dtypes. Schema 2 uses Arrow IPC so
    UInt8 / Datetime / Categorical round-trip. Encoding failures raise
    :class:`CheckpointSerializationError` instead of silently storing
    ``repr(df)`` (which reloads as a string and destroys frame structure).
    """
    if df is None:
        return None
    import polars as pl

    if not isinstance(df, pl.DataFrame):
        raise CheckpointSerializationError(
            "Checkpoint frames must be Polars DataFrames; "
            f"got {type(df).__name__}"
        )
    try:
        buf = io.BytesIO()
        df.write_ipc(buf)
    except Exception as exc:
        dtypes = [f"{c}:{dt}" for c, dt in zip(df.columns, df.dtypes)]
        raise CheckpointSerializationError(
            "Failed to encode DataFrame as Arrow IPC for checkpoint "
            f"(dtypes=[{', '.join(dtypes)}]). "
            "Object / unsupported columns cannot be checkpointed losslessly. "
            f"Underlying error: {exc}"
        ) from exc
    return {
        "_kind": "polars_ipc_b64",
        "columns": list(df.columns),
        "dtypes": [str(dt) for dt in df.dtypes],
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
    }


def _deserialize_frame(payload: Any, *, schema_version: int) -> Any:
    """Decode a frame payload for a known checkpoint schema version."""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(
            f"Checkpoint frame payload must be an object, got {type(payload).__name__}"
        )
    kind = payload.get("_kind")
    if kind == "polars_ipc_b64":
        # Preferred for schema 2; also accepted if found under legacy files.
        import polars as pl

        try:
            raw = base64.b64decode(payload["data"])
            return pl.read_ipc(io.BytesIO(raw))
        except Exception as exc:
            raise ValueError(
                f"Failed to decode polars_ipc_b64 checkpoint frame: {exc}"
            ) from exc
    if kind == "polars_records":
        # Only schema 1 used lossy record JSON; schema 2+ is IPC-only.
        if schema_version != 1:
            raise ValueError(
                f"Frame kind 'polars_records' is only valid for "
                f"schema_version=1; got schema_version={schema_version}"
            )
        # Schema 1: lossy record JSON (dtypes not preserved).
        import polars as pl

        rows = payload.get("rows") or []
        if not rows:
            cols = payload.get("columns") or []
            return pl.DataFrame({c: [] for c in cols}) if cols else pl.DataFrame()
        return pl.DataFrame(rows)
    if kind == "repr":
        raise ValueError(
            "Checkpoint frame kind 'repr' is no longer supported "
            "(destroyed DataFrame structure). Re-run the experiment."
        )
    raise ValueError(
        f"Unknown checkpoint frame kind {kind!r} "
        f"(schema_version={schema_version})"
    )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Exclusive random-temp write + fsync + replace (symlink-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    tmp = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    fd = None
    try:
        fd = os.open(str(tmp), flags, 0o644)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError(f"Refusing non-regular checkpoint temp: {tmp}")
        written = 0
        while written < len(data):
            n = os.write(fd, data[written:])
            if n <= 0:
                raise OSError(f"Short write to {tmp}")
            written += n
        os.fsync(fd)
        os.close(fd)
        fd = None
        st = os.lstat(tmp)
        if not stat.S_ISREG(st.st_mode):
            raise OSError(f"Checkpoint temp is not a regular file: {tmp}")
        os.replace(str(tmp), str(path))
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            if tmp.exists() or tmp.is_symlink():
                tmp.unlink()
        except OSError:
            pass


def _serialize_result(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    return {
        "params": result.get("params"),
        "info": result.get("info"),
        "model": _serialize_frame(result.get("model")),
        "agents": _serialize_frame(result.get("agents")),
    }


def _deserialize_result(
    data: Optional[Dict[str, Any]], *, schema_version: int
) -> Optional[Dict[str, Any]]:
    if data is None:
        return None
    return {
        "params": data.get("params"),
        "info": data.get("info"),
        "model": _deserialize_frame(data.get("model"), schema_version=schema_version),
        "agents": _deserialize_frame(
            data.get("agents"), schema_version=schema_version
        ),
    }


def _qualified_model_name(model_class: Type) -> str:
    return f"{model_class.__module__}.{model_class.__qualname__}"


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _workload_fingerprint(
    model_class: Type, param_list: List[Dict[str, Any]]
) -> str:
    """Hash of model identity + full parameter list (order-sensitive)."""
    payload = {
        "model_class": _qualified_model_name(model_class),
        "n": len(param_list),
        "params": param_list,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _params_match(saved: Any, requested: Dict[str, Any]) -> bool:
    if not isinstance(saved, dict):
        return False
    return _stable_json(saved) == _stable_json(requested)


def _is_cancelled_outcome(outcome: RunOutcome) -> bool:
    """Never-run / fail_fast-cancelled slots must not block resume."""
    return outcome.error_type == "Cancelled"


def _run_single_simulation(
    index: int,
    params: Dict[str, Any],
    model_class: Type,
) -> Dict[str, Any]:
    """Worker: run one simulation and return a picklable outcome dict."""
    try:
        model = model_class(params)
        results = model.run()
        return {
            "index": index,
            "status": "success",
            "params": params,
            "result": {
                "params": params,
                "model": results.get("model"),
                "agents": results.get("agents"),
                "info": results.get("info"),
            },
            "error_type": None,
            "error_message": None,
            "traceback": None,
            "attempts": 1,
        }
    except Exception as exc:
        return {
            "index": index,
            "status": "failed",
            "params": params,
            "result": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": "".join(
                _traceback_mod.format_exception(type(exc), exc, exc.__traceback__)
            ),
            "attempts": 1,
        }


def _process_worker(
    conn: Any,
    index: int,
    params: Dict[str, Any],
    model_class: Type,
) -> None:
    """Child process entry: run one simulation and send the outcome dict."""
    try:
        conn.send(_run_single_simulation(index, params, model_class))
    except Exception as exc:  # pragma: no cover - extremely defensive
        conn.send(
            {
                "index": index,
                "status": "failed",
                "params": params,
                "result": None,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": "".join(
                    _traceback_mod.format_exception(
                        type(exc), exc, exc.__traceback__
                    )
                ),
                "attempts": 1,
            }
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _terminate_process(proc: mp.Process, grace: float = 1.0) -> None:
    """Hard-kill a worker process (terminate → kill)."""
    if not proc.is_alive():
        proc.join(timeout=grace)
        return
    proc.terminate()
    proc.join(timeout=grace)
    if proc.is_alive():
        try:
            proc.kill()  # Python 3.7+
        except AttributeError:  # pragma: no cover
            pass
        proc.join(timeout=grace)


class ParallelRunner:
    """Run multiple independent simulations in **CPU processes**.

    Important
    ---------
    * **Not automatic:** ``model.run()`` is always a **single** simulation.
      Use this class (or :class:`~ambr.experiment.Experiment` for sequential
      sweeps, or :class:`~ambr.gpu_ensemble.GPUEnsembleRunner` for GPU batches)
      when you want many runs.
    * Uses ``multiprocessing`` with the ``spawn`` context — models and params
      must be picklable. Each task runs in its **own process** so ``timeout``
      can hard-terminate hung workers (``Process.terminate`` / ``kill``).
    * Does **not** use the GPU. For many short GPU replicates, prefer
      :class:`~ambr.gpu_ensemble.GPUEnsembleRunner`.
    * Returns a list of :class:`RunOutcome` **in input order** (not completion
      order). Failures and timeouts are structured, never silent.
    * Checkpoints are **JSON** (no ``pickle``). Resume requires
      ``trust_checkpoint=True`` and a schema-validated file.

    Usage::

        runner = ParallelRunner(MyModel, n_workers=4)
        outcomes = runner.run(
            [
                {"n": 100, "steps": 20, "seed": 0, "show_progress": False},
                {"n": 100, "steps": 20, "seed": 1, "show_progress": False},
            ],
            fail_fast=False,
            timeout=120,
            retry=1,
            max_in_flight=8,
        )
        for o in outcomes:
            if o.status == "success":
                print(o.index, o.result["info"]["run_uuid"])
    """

    def __init__(self, model_class: Type, n_workers: int = None):
        """
        Initialize parallel runner.

        Args:
            model_class: Model class to instantiate
            n_workers: Maximum concurrent worker processes (default: CPU count).
                Must be ``>= 1``.
        """
        self.model_class = model_class
        n = n_workers if n_workers is not None else mp.cpu_count()
        if int(n) < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers!r}")
        self.n_workers = int(n)

    def run(
        self,
        param_list: List[Dict[str, Any]],
        show_progress: bool = True,
        *,
        fail_fast: bool = False,
        timeout: Optional[float] = None,
        retry: int = 0,
        max_in_flight: Optional[int] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        resume: bool = False,
        trust_checkpoint: bool = False,
    ) -> List[RunOutcome]:
        """Run simulations in parallel; return :class:`RunOutcome` in input order.

        Args:
            param_list: List of parameter dictionaries (order preserved).
            show_progress: Print a completion counter.
            fail_fast: Stop submitting new work after the first failure/timeout
                and **terminate every remaining live worker**.
            timeout: Per-run wall-clock seconds (``None`` = no limit). Timed-out
                workers are **terminated** (not cooperatively cancelled).
            retry: Extra attempts after a failed attempt (not applied to
                timeouts by default).
            max_in_flight: Optional additional cap on concurrent processes.
                Effective concurrency is ``min(n_workers, max_in_flight)`` when
                set; never exceeds ``n_workers``. Must be ``>= 1`` if provided.
            checkpoint_path: Optional JSON path; completed outcomes are
                written after each finish for crash-safe resume.
            resume: If True and ``checkpoint_path`` exists, skip indices
                already present **only when** the checkpoint workload
                fingerprint and per-entry params match this run. Requires
                ``trust_checkpoint=True``. Cancelled/never-run slots are not
                treated as finished (they remain pending).
            trust_checkpoint: Explicit opt-in to read a checkpoint file.
                Checkpoints are JSON (not pickle); still only load files you
                control.

        Returns:
            ``len(param_list)`` outcomes in the same order as ``param_list``.
        """
        total = len(param_list)
        outcomes: List[Optional[RunOutcome]] = [None] * total
        max_attempts = 1 + max(0, int(retry))
        if max_in_flight is not None and int(max_in_flight) < 1:
            raise ValueError(
                f"max_in_flight must be >= 1 when set, got {max_in_flight!r}"
            )
        # Never exceed n_workers; max_in_flight is an optional tighter cap.
        in_flight_cap = self.n_workers
        if max_in_flight is not None:
            in_flight_cap = min(self.n_workers, int(max_in_flight))
        ckpt = Path(checkpoint_path) if checkpoint_path else None
        workload_fp = _workload_fingerprint(self.model_class, param_list)
        model_name = _qualified_model_name(self.model_class)

        if resume:
            if not trust_checkpoint:
                raise ValueError(
                    "resume=True requires trust_checkpoint=True. "
                    "Only resume from checkpoint files you control "
                    "(JSON schema; pickle is not used)."
                )
            if ckpt is not None and ckpt.is_file():
                meta, loaded = self._load_checkpoint(ckpt)
                saved_fp = meta.get("workload_fingerprint")
                if saved_fp is not None and saved_fp != workload_fp:
                    raise ValueError(
                        "Checkpoint workload fingerprint mismatch: the saved "
                        "run is for a different model and/or parameter list. "
                        f"checkpoint={saved_fp!r} requested={workload_fp!r}. "
                        "Use a matching param_list/model_class or a new checkpoint."
                    )
                saved_model = meta.get("model_class")
                if saved_model is not None and saved_model != model_name:
                    raise ValueError(
                        f"Checkpoint model_class mismatch: "
                        f"checkpoint={saved_model!r} requested={model_name!r}"
                    )
                for idx, outcome in loaded.items():
                    if not (0 <= idx < total):
                        continue
                    # Cancelled / never-run slots stay pending for resume.
                    if _is_cancelled_outcome(outcome):
                        continue
                    if not _params_match(outcome.params, param_list[idx]):
                        raise ValueError(
                            f"Checkpoint entry {idx} params do not match this "
                            f"run (saved={outcome.params!r}, "
                            f"requested={param_list[idx]!r})"
                        )
                    # Legacy checkpoints without a fingerprint still require
                    # per-index param equality (checked above).
                    outcomes[idx] = outcome

        pending_indices = [i for i, o in enumerate(outcomes) if o is None]
        stop_submitting = False
        completed_count = sum(1 for o in outcomes if o is not None)

        ctx = mp.get_context("spawn")
        # Single authoritative live-process registry. Updated immediately on
        # every process start (initial submit *and* retries). fail_fast /
        # finally always terminate this set — never a dual still_active list
        # that can drop retries from cleanup.
        # Entry: (process, parent_conn, index, attempt, started)
        active: List[Tuple[mp.Process, Any, int, int, float]] = []
        idx_iter = iter(pending_indices)

        def _start_worker(i: int, attempt: int) -> None:
            """Spawn one worker and register it in ``active`` immediately."""
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_process_worker,
                args=(child_conn, i, param_list[i], self.model_class),
            )
            proc.start()
            child_conn.close()  # only child writes
            active.append((proc, parent_conn, i, attempt, time.monotonic()))

        def _submit_one() -> bool:
            if stop_submitting:
                return False
            try:
                i = next(idx_iter)
            except StopIteration:
                return False
            _start_worker(i, 1)
            return True

        def _record(outcome: RunOutcome) -> None:
            nonlocal completed_count, stop_submitting
            outcomes[outcome.index] = outcome
            completed_count += 1
            if show_progress:
                print(
                    f"\rCompleted {completed_count}/{total} simulations",
                    end="",
                )
            if ckpt is not None:
                self._save_checkpoint(
                    ckpt,
                    outcomes,
                    model_class=self.model_class,
                    param_list=param_list,
                    workload_fingerprint=workload_fp,
                )
            if outcome.status != "success" and fail_fast:
                stop_submitting = True

        def _kill_all_live(*, cancel_unfinished: bool) -> None:
            """Terminate every process currently in the live registry."""
            for proc, conn, i, attempt, _started in list(active):
                try:
                    conn.close()
                except Exception:
                    pass
                _terminate_process(proc)
                if cancel_unfinished and outcomes[i] is None:
                    outcomes[i] = RunOutcome(
                        index=i,
                        status="failed",
                        params=dict(param_list[i]),
                        error_type="Cancelled",
                        error_message="Cancelled (fail_fast)",
                        attempts=attempt,
                    )
            active.clear()

        try:
            while len(active) < in_flight_cap and _submit_one():
                pass

            while active:
                now = time.monotonic()
                progressed = False
                # Snapshot for safe iteration; mutations (remove finished /
                # append retries) go to the authoritative ``active`` list.
                snapshot = list(active)

                for entry in snapshot:
                    if entry not in active:
                        # Already removed earlier in this scan.
                        continue
                    proc, conn, i, attempt, started = entry

                    # Hard wall-clock timeout → terminate the process.
                    if timeout is not None and (now - started) >= float(timeout):
                        try:
                            conn.close()
                        except Exception:
                            pass
                        _terminate_process(proc)
                        try:
                            active.remove(entry)
                        except ValueError:
                            pass
                        _record(
                            RunOutcome(
                                index=i,
                                status="timeout",
                                params=dict(param_list[i]),
                                error_type="TimeoutError",
                                error_message=f"Exceeded timeout={timeout}s",
                                attempts=attempt,
                            )
                        )
                        progressed = True
                        continue

                    # Non-blocking poll for a finished worker message.
                    try:
                        ready = conn.poll(0)
                    except Exception:
                        ready = False

                    if ready:
                        try:
                            raw = conn.recv()
                        except EOFError:
                            raw = {
                                "index": i,
                                "status": "failed",
                                "params": dict(param_list[i]),
                                "result": None,
                                "error_type": "EOFError",
                                "error_message": "Worker closed pipe without result",
                                "traceback": None,
                                "attempts": attempt,
                            }
                        try:
                            conn.close()
                        except Exception:
                            pass
                        proc.join(timeout=1.0)
                        if proc.is_alive():
                            _terminate_process(proc)
                        try:
                            active.remove(entry)
                        except ValueError:
                            pass

                        outcome = RunOutcome.from_dict(raw)
                        outcome.attempts = attempt

                        if (
                            outcome.status == "failed"
                            and attempt < max_attempts
                            and not stop_submitting
                        ):
                            # Retry: register in ``active`` immediately so a
                            # later fail_fast in this same scan (or finally)
                            # can terminate it.
                            _start_worker(i, attempt + 1)
                            progressed = True
                            continue

                        _record(outcome)
                        progressed = True
                        continue

                    if not proc.is_alive():
                        # Died without a message
                        try:
                            conn.close()
                        except Exception:
                            pass
                        proc.join(timeout=0.5)
                        try:
                            active.remove(entry)
                        except ValueError:
                            pass
                        _record(
                            RunOutcome(
                                index=i,
                                status="failed",
                                params=dict(param_list[i]),
                                error_type="WorkerDied",
                                error_message=(
                                    f"Worker exited with code {proc.exitcode}"
                                ),
                                attempts=attempt,
                            )
                        )
                        progressed = True
                        continue

                    # Still running — leave in active.

                if stop_submitting:
                    # Kill the entire live registry, including retries started
                    # mid-scan that never existed on the pre-scan snapshot.
                    _kill_all_live(cancel_unfinished=True)
                    break

                while len(active) < in_flight_cap and _submit_one():
                    progressed = True

                if not progressed and active:
                    # Avoid busy-spin; short sleep while workers run.
                    time.sleep(0.05)
        finally:
            # Ensure no orphaned workers on unexpected exit.
            _kill_all_live(cancel_unfinished=False)

        if show_progress:
            print()

        for i, o in enumerate(outcomes):
            if o is None:
                outcomes[i] = RunOutcome(
                    index=i,
                    status="failed",
                    params=dict(param_list[i]),
                    error_type="Cancelled",
                    error_message="Not run (fail_fast or cancelled)",
                )

        if ckpt is not None:
            self._save_checkpoint(
                ckpt,
                outcomes,
                model_class=self.model_class,
                param_list=param_list,
                workload_fingerprint=workload_fp,
            )

        return [o for o in outcomes if o is not None]

    def run_with_seeds(
        self,
        base_params: Dict[str, Any],
        seeds: List[int],
        show_progress: bool = True,
        **run_kwargs: Any,
    ) -> List[RunOutcome]:
        """Run the same parameters with different random seeds."""
        param_list = [{**base_params, "seed": seed} for seed in seeds]
        return self.run(param_list, show_progress=show_progress, **run_kwargs)

    @staticmethod
    def _save_checkpoint(
        path: Path,
        outcomes: List[Optional[RunOutcome]],
        *,
        model_class: Type,
        param_list: List[Dict[str, Any]],
        workload_fingerprint: str,
    ) -> None:
        """Write a JSON checkpoint (never pickle — see SECURITY.md).

        Uses a random exclusive temp name (not a predictable ``*.tmp`` path)
        so a pre-planted symlink cannot redirect the write outside the
        destination directory.

        Never-run ``Cancelled`` outcomes are **omitted** so a later resume
        treats those indices as pending work.
        """
        payload_outcomes: Dict[str, Any] = {}
        for i, o in enumerate(outcomes):
            if o is None:
                continue
            if _is_cancelled_outcome(o):
                # Leave pending for resume — do not mark as finished work.
                continue
            d = o.to_dict()
            d["result"] = _serialize_result(o.result)
            payload_outcomes[str(i)] = d
        payload = {
            "schema_version": _CHECKPOINT_SCHEMA,
            "format": "ambr.ParallelRunner.checkpoint+json",
            "workload_fingerprint": workload_fingerprint,
            "model_class": _qualified_model_name(model_class),
            "n_params": len(param_list),
            "outcomes": payload_outcomes,
        }
        text = (json.dumps(payload, indent=2, default=str) + "\n").encode("utf-8")
        _atomic_write_bytes(Path(path), text)

    @staticmethod
    def _load_checkpoint(
        path: Path,
    ) -> Tuple[Dict[str, Any], Dict[int, RunOutcome]]:
        """Load a JSON checkpoint. Raises on unknown / unsafe formats.

        Returns ``(metadata, outcomes_by_index)``. Cancelled entries are not
        restored (callers treat those indices as pending).
        """
        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Checkpoint {path} is not UTF-8 JSON. "
                "Pickle checkpoints are not supported (RCE risk)."
            ) from exc
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Checkpoint {path} is not valid JSON. "
                "Pickle checkpoints are not supported (RCE risk)."
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint root must be a JSON object")
        version = payload.get("schema_version")
        supported = {_CHECKPOINT_SCHEMA} | _LEGACY_CHECKPOINT_SCHEMAS
        if version not in supported:
            raise ValueError(
                f"Unsupported checkpoint schema_version={version!r}; "
                f"supported={sorted(supported)}"
            )
        if payload.get("format") != "ambr.ParallelRunner.checkpoint+json":
            raise ValueError(
                f"Unsupported checkpoint format={payload.get('format')!r}"
            )
        outcomes_raw = payload.get("outcomes")
        if not isinstance(outcomes_raw, dict):
            raise ValueError("Checkpoint 'outcomes' must be an object")
        out: Dict[int, RunOutcome] = {}
        for k, v in outcomes_raw.items():
            if not isinstance(v, dict):
                continue
            data = dict(v)
            data["result"] = _deserialize_result(
                data.get("result"), schema_version=int(version)
            )
            outcome = RunOutcome.from_dict(data)
            if _is_cancelled_outcome(outcome):
                continue
            out[int(k)] = outcome
        meta = {
            "schema_version": version,
            "workload_fingerprint": payload.get("workload_fingerprint"),
            "model_class": payload.get("model_class"),
            "n_params": payload.get("n_params"),
        }
        return meta, out


# =============================================================================
# Vectorized Operations
# =============================================================================


def vectorized_wealth_transfer(
    wealths: np.ndarray,
    transfer_amounts: np.ndarray,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
) -> np.ndarray:
    """
    Perform batch wealth transfers using vectorized operations.

    Args:
        wealths: Array of agent wealths
        transfer_amounts: Array of transfer amounts
        source_ids: Indices of source agents
        target_ids: Indices of target agents

    Returns:
        Updated wealth array
    """
    new_wealths = wealths.copy()

    # Use np.add.at for efficient in-place accumulation
    np.subtract.at(new_wealths, source_ids, transfer_amounts)
    np.add.at(new_wealths, target_ids, transfer_amounts)

    return new_wealths


def vectorized_move(
    positions: np.ndarray,
    velocities: np.ndarray,
    bounds: Optional[Tuple[float, float]] = None,
    wrap: bool = False,
) -> np.ndarray:
    """
    Move all agents in one vectorized operation.

    Args:
        positions: Nx2 array of positions
        velocities: Nx2 array of velocities (or scalar for uniform)
        bounds: Optional (min, max) bounds
        wrap: Whether to wrap at boundaries

    Returns:
        Updated positions
    """
    new_positions = positions + velocities

    if bounds is not None:
        min_val, max_val = bounds
        if wrap:
            range_size = max_val - min_val
            new_positions = min_val + np.mod(new_positions - min_val, range_size)
        else:
            new_positions = np.clip(new_positions, min_val, max_val)

    return new_positions


def vectorized_random_velocities(
    n: int, speed: float, rng: np.random.Generator = None, dimensions: int = 2
) -> np.ndarray:
    """
    Generate random velocity vectors.

    Args:
        n: Number of agents
        speed: Maximum speed
        rng: Random number generator
        dimensions: Number of dimensions (default 2)

    Returns:
        Nx(dimensions) array of velocities
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(-speed, speed, (n, dimensions))


def vectorized_sir_infections(
    positions: np.ndarray,
    statuses: np.ndarray,
    spatial_index: "SpatialIndex",
    infection_radius: float,
    transmission_rate: float,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Process all SIR infections in vectorized manner using spatial index.

    Args:
        positions: Nx2 array of agent positions
        statuses: Array of health statuses (0=S, 1=I, 2=R)
        spatial_index: Pre-built spatial index
        infection_radius: Infection radius
        transmission_rate: Probability of infection per contact
        rng: Random number generator

    Returns:
        Updated statuses array
    """
    if rng is None:
        rng = np.random.default_rng()

    new_statuses = statuses.copy()

    # Find all infected agents
    infected_mask = statuses == 1
    infected_indices = np.where(infected_mask)[0]

    # For each infected agent, find susceptible neighbors
    for inf_idx in infected_indices:
        neighbors = spatial_index.query_radius(positions[inf_idx], infection_radius)

        for neighbor_idx in neighbors:
            if statuses[neighbor_idx] == 0:  # Susceptible
                if rng.random() < transmission_rate:
                    new_statuses[neighbor_idx] = 1  # Infected

    return new_statuses


# =============================================================================
# Convenience Functions
# =============================================================================


def check_performance_deps() -> Dict[str, bool]:
    """Check which performance dependencies are available."""
    return {
        "scipy": HAS_SCIPY,
        "numba": HAS_NUMBA,
        "multiprocessing": True,  # Always available
    }


def install_performance_deps():
    """Print instructions for installing performance dependencies."""
    deps = check_performance_deps()
    print("AMBER Performance Dependencies Status:")
    print("-" * 40)
    for dep, available in deps.items():
        status = "✅ Available" if available else "❌ Not installed"
        print(f"  {dep}: {status}")

    if not all(deps.values()):
        print("\nTo install missing dependencies:")
        if not deps["scipy"]:
            print("  pip install scipy")
        if not deps["numba"]:
            print("  pip install numba")


# Export all public functions
__all__ = [
    "SpatialIndex",
    "ParallelRunner",
    "RunOutcome",
    # Scatter (vectorized write path)
    "scatter_add_1d",
    "scatter_write_1d",
    "apply_scatter_add",
    "apply_scatter_write",
    # Spatial Numba helpers
    "fast_distance_matrix",
    "fast_neighbors_within_radius",
    "fast_all_neighbors_within_radius",
    "fast_random_walk_step",
    # Vectorized utilities
    "vectorized_wealth_transfer",
    "vectorized_move",
    "vectorized_random_velocities",
    "vectorized_sir_infections",
    "check_performance_deps",
    "install_performance_deps",
    "HAS_SCIPY",
    "HAS_NUMBA",
]
