"""
AMBER Performance Utilities

High-performance implementations for common ABM operations:
- KD-Tree for O(log n) spatial neighbor queries
- Numba JIT for hot loop acceleration
- Multiprocessing for parallel experiment execution
- Vectorized operations for batch updates
"""

from typing import List, Tuple, Optional, Dict, Any, Type
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Optional imports with fallbacks
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

    # Create a no-op decorator
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
# Numba-Accelerated Functions
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


def _run_single_simulation(params: Dict[str, Any], model_class: Type) -> Dict[str, Any]:
    """Worker function for parallel simulation."""
    model = model_class(params)
    results = model.run()
    return {
        "params": params,
        "model": results.get("model"),
        "agents": results.get("agents"),
        "info": results.get("info"),
    }


class ParallelRunner:
    """
    Run multiple simulations in parallel across CPU cores.

    Usage:
        runner = ParallelRunner(MyModel, n_workers=8)
        results = runner.run(param_list)
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
        self, param_list: List[Dict[str, Any]], show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run simulations in parallel.

        Args:
            param_list: List of parameter dictionaries
            show_progress: Whether to show progress

        Returns:
            List of result dictionaries
        """
        results = []
        total = len(param_list)

        # Use spawn context for better compatibility
        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=self.n_workers, mp_context=ctx
        ) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_run_single_simulation, params, self.model_class): i
                for i, params in enumerate(param_list)
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if show_progress:
                        print(f"\rCompleted {completed}/{total} simulations", end="")
                except Exception as e:
                    print(f"\nSimulation failed: {e}")
                    results.append(
                        {"error": str(e), "params": param_list[futures[future]]}
                    )
                    completed += 1

        if show_progress:
            print()  # New line after progress

        return results

    def run_with_seeds(
        self, base_params: Dict[str, Any], seeds: List[int], show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run same parameters with different random seeds.

        Args:
            base_params: Base parameter dictionary
            seeds: List of random seeds
            show_progress: Whether to show progress

        Returns:
            List of result dictionaries
        """
        param_list = [{**base_params, "seed": seed} for seed in seeds]
        return self.run(param_list, show_progress)


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
    "fast_distance_matrix",
    "fast_neighbors_within_radius",
    "fast_all_neighbors_within_radius",
    "fast_random_walk_step",
    "vectorized_wealth_transfer",
    "vectorized_move",
    "vectorized_random_velocities",
    "vectorized_sir_infections",
    "check_performance_deps",
    "install_performance_deps",
    "HAS_SCIPY",
    "HAS_NUMBA",
]
