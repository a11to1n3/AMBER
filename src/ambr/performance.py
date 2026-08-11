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

import pickle
import time
import traceback as _traceback_mod
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    wait,
)
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


class ParallelRunner:
    """Run multiple independent simulations in **CPU process pools**.

    Important
    ---------
    * **Not automatic:** ``model.run()`` is always a **single** simulation.
      Use this class (or :class:`~ambr.experiment.Experiment` for sequential
      sweeps, or :class:`~ambr.gpu_ensemble.GPUEnsembleRunner` for GPU batches)
      when you want many runs.
    * Uses ``multiprocessing`` with the ``spawn`` context — models and params
      must be picklable.
    * Does **not** use the GPU. For many short GPU replicates, prefer
      :class:`~ambr.gpu_ensemble.GPUEnsembleRunner`.
    * Returns a list of :class:`RunOutcome` **in input order** (not completion
      order). Failures and timeouts are structured, never silent.

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
            n_workers: Number of parallel workers (default: CPU count)
        """
        self.model_class = model_class
        self.n_workers = n_workers or mp.cpu_count()

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
    ) -> List[RunOutcome]:
        """Run simulations in parallel; return :class:`RunOutcome` in input order.

        Args:
            param_list: List of parameter dictionaries (order preserved).
            show_progress: Print a completion counter.
            fail_fast: Stop submitting new work after the first failure/timeout.
            timeout: Per-run wall-clock seconds (``None`` = no limit). Timed-out
                runs are recorded as ``status='timeout'``.
            retry: Extra attempts after a failed attempt (not applied to
                timeouts by default).
            max_in_flight: Bound concurrent submissions (default:
                ``n_workers * 2``). Prevents unbounded queue growth.
            checkpoint_path: Optional pickle path; completed outcomes are
                written after each finish for crash-safe resume.
            resume: If True and ``checkpoint_path`` exists, skip indices
                already present in the checkpoint.

        Returns:
            ``len(param_list)`` outcomes in the same order as ``param_list``.
        """
        total = len(param_list)
        outcomes: List[Optional[RunOutcome]] = [None] * total
        max_attempts = 1 + max(0, int(retry))
        in_flight_cap = max_in_flight if max_in_flight is not None else max(
            self.n_workers * 2, 1
        )
        ckpt = Path(checkpoint_path) if checkpoint_path else None

        if resume and ckpt is not None and ckpt.is_file():
            for idx, outcome in self._load_checkpoint(ckpt).items():
                if 0 <= idx < total:
                    outcomes[idx] = outcome

        # Indices still needing work
        pending_indices = [i for i, o in enumerate(outcomes) if o is None]
        stop_submitting = False
        completed_count = sum(1 for o in outcomes if o is not None)

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=self.n_workers, mp_context=ctx
        ) as executor:
            # future -> (index, attempt, submit_monotonic)
            active: Dict[Any, Tuple[int, int, float]] = {}
            idx_iter = iter(pending_indices)

            def _submit_one() -> bool:
                nonlocal stop_submitting
                if stop_submitting:
                    return False
                try:
                    i = next(idx_iter)
                except StopIteration:
                    return False
                fut = executor.submit(
                    _run_single_simulation, i, param_list[i], self.model_class
                )
                active[fut] = (i, 1, time.monotonic())
                return True

            while len(active) < in_flight_cap and _submit_one():
                pass

            while active:
                done_set, _ = wait(
                    list(active.keys()),
                    timeout=0.25,
                    return_when=FIRST_COMPLETED,
                )
                now = time.monotonic()

                # Per-task wall-clock timeout for still-running futures
                if timeout is not None:
                    for fut, (i, attempt, started) in list(active.items()):
                        if fut in done_set:
                            continue
                        if now - started >= float(timeout):
                            fut.cancel()
                            del active[fut]
                            outcomes[i] = RunOutcome(
                                index=i,
                                status="timeout",
                                params=dict(param_list[i]),
                                error_type="TimeoutError",
                                error_message=f"Exceeded timeout={timeout}s",
                                attempts=attempt,
                            )
                            completed_count += 1
                            if show_progress:
                                print(
                                    f"\rCompleted {completed_count}/{total} simulations",
                                    end="",
                                )
                            if ckpt is not None:
                                self._save_checkpoint(ckpt, outcomes)
                            if fail_fast:
                                stop_submitting = True
                                for other in list(active.keys()):
                                    other.cancel()
                                active.clear()
                                break

                if stop_submitting and not active:
                    break

                for fut in done_set:
                    if fut not in active:
                        continue
                    i, attempt, _started = active.pop(fut)
                    try:
                        raw = fut.result(timeout=0)
                        outcome = RunOutcome.from_dict(raw)
                        outcome.attempts = attempt
                    except FuturesTimeoutError:
                        outcome = RunOutcome(
                            index=i,
                            status="timeout",
                            params=dict(param_list[i]),
                            error_type="TimeoutError",
                            error_message=f"Exceeded timeout={timeout}s",
                            attempts=attempt,
                        )
                    except Exception as exc:
                        outcome = RunOutcome(
                            index=i,
                            status="failed",
                            params=dict(param_list[i]),
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            traceback="".join(
                                _traceback_mod.format_exception(
                                    type(exc), exc, exc.__traceback__
                                )
                            ),
                            attempts=attempt,
                        )

                    # Retry failed (not timeout) attempts
                    if (
                        outcome.status == "failed"
                        and attempt < max_attempts
                        and not stop_submitting
                    ):
                        new_fut = executor.submit(
                            _run_single_simulation,
                            i,
                            param_list[i],
                            self.model_class,
                        )
                        active[new_fut] = (i, attempt + 1, time.monotonic())
                        continue

                    outcomes[i] = outcome
                    completed_count += 1
                    if show_progress:
                        print(
                            f"\rCompleted {completed_count}/{total} simulations",
                            end="",
                        )
                    if ckpt is not None:
                        self._save_checkpoint(ckpt, outcomes)

                    if outcome.status != "success" and fail_fast:
                        stop_submitting = True
                        for other in list(active.keys()):
                            other.cancel()
                        active.clear()
                        break

                    while len(active) < in_flight_cap and _submit_one():
                        pass

        if show_progress:
            print()

        # Any index never finished (fail_fast cancel) → explicit failed records
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
            self._save_checkpoint(ckpt, outcomes)

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
        path: Path, outcomes: List[Optional[RunOutcome]]
    ) -> None:
        payload = {
            i: o.to_dict()
            for i, o in enumerate(outcomes)
            if o is not None
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @staticmethod
    def _load_checkpoint(path: Path) -> Dict[int, RunOutcome]:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        out: Dict[int, RunOutcome] = {}
        if isinstance(payload, dict):
            for k, v in payload.items():
                out[int(k)] = RunOutcome.from_dict(v)
        return out


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
