"""
Environment implementations for AMBER framework.
Supports different types of spatial and network topologies.
"""

from typing import Dict, List, Optional, Tuple, Union
from types import MethodType
import polars as pl
import numpy as np
import networkx as nx
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ._deprecation import warn_deprecated

try:  # optional fast path for large-N neighbour queries
    from scipy.spatial import cKDTree as _cKDTree
    _HAS_KDTREE = True
except Exception:  # pragma: no cover - scipy is an optional accelerator
    _cKDTree = None
    _HAS_KDTREE = False

_KDTREE_MIN_AGENTS = 2000  # below this, vectorized numpy brute force is faster

@dataclass
class Position:
    """Represents a position in any topology."""
    coordinates: Tuple[float, ...]
    topology_type: str

class Environment(ABC):
    """Base class for all environments."""
    
    def __init__(self, model):
        """Initialize environment with reference to model."""
        self.model = model
        # Fallback private store used when model doesn't expose population.
        self._df: Optional[pl.DataFrame] = None

    @property
    def df(self) -> pl.DataFrame:
        """The model's current agent DataFrame (always fresh)."""
        if self.model is not None and hasattr(self.model, 'agents_df'):
            return self.model.agents_df
        if self._df is not None:
            return self._df
        return pl.DataFrame()

    @df.setter
    def df(self, value: pl.DataFrame) -> None:
        """Replace the model's agent DataFrame.

        Prefers a real ``model._set_frame`` bound method (the single write
        seam) so the contract monitor and pending-write buffer stay consistent.
        Falls back to a plain attribute write (mock models) or a private store.
        ``unittest.mock.Mock`` auto-creates callables for missing attrs, so we
        require a true bound method rather than ``callable(...)``.
        """
        model = self.model
        set_frame = getattr(model, '_set_frame', None) if model is not None else None
        if isinstance(set_frame, MethodType):
            # Clear any pending OOP buffer -- environment writes are authoritative.
            if hasattr(model, '_pending_writes'):
                model._pending_writes = {}
            set_frame(value)
            bump = getattr(model, '_bump_id_version', None)
            if isinstance(bump, MethodType):
                bump()
            return
        if model is not None and hasattr(model, 'agents_df'):
            # Plain attribute (e.g. Mock model) — write directly.
            object.__setattr__(model, 'agents_df', value)
            self._df = value
        else:
            self._df = value

    @abstractmethod
    def get_neighbors(self, agent_id: int) -> List[int]:
        """Get neighboring agents for a given agent."""
        pass
    
    @abstractmethod
    def get_distance(self, agent1_id: int, agent2_id: int) -> float:
        """Calculate distance between two agents."""
        pass
    
    @abstractmethod
    def move_agent(self, agent_id: int, new_position: Position) -> None:
        """Move an agent to a new position."""
        pass

class GridEnvironment(Environment):
    """N-dimensional grid environment with discrete positions."""
    
    def __init__(self, model, size: Union[int, Tuple[int, ...]], torus: bool = False,
                 wrap: Optional[bool] = None):
        """
        Initialize grid environment.

        Args:
            model: Reference to the model
            size: Grid size - int for square grid or tuple for rectangular grid
            torus: Whether to wrap around grid boundaries
            wrap: Deprecated alias for ``torus``.
        """
        super().__init__(model)

        # Handle both int and tuple size formats
        if isinstance(size, int):
            self.dimensions = (size, size)
            self.size = (size, size)
        else:
            self.dimensions = size
            self.size = size

        if wrap is not None:
            warn_deprecated("GridEnvironment(wrap=...)", "torus=", stacklevel=3)
            torus = wrap
        self.torus = torus

    @property
    def wrap(self):
        """Deprecated alias for :attr:`torus`."""
        warn_deprecated("GridEnvironment.wrap", "GridEnvironment.torus", stacklevel=2)
        return self.torus

    @wrap.setter
    def wrap(self, value):
        warn_deprecated("GridEnvironment.wrap", "GridEnvironment.torus", stacklevel=2)
        self.torus = value
    
    @property
    def width(self):
        """Get grid width (first dimension)."""
        return self.dimensions[0]
    
    @property
    def height(self):
        """Get grid height (second dimension if it exists)."""
        return self.dimensions[1] if len(self.dimensions) > 1 else 1
    
    @property
    def positions(self):
        """Get all possible positions in the grid."""
        positions = []
        if len(self.dimensions) == 2:
            for x in range(self.dimensions[0]):
                for y in range(self.dimensions[1]):
                    positions.append((x, y))
        else:
            # For N-dimensional grids
            import itertools
            ranges = [range(dim) for dim in self.dimensions]
            positions = list(itertools.product(*ranges))
        return positions
    
    def get_neighbors(self, position_or_agent_id, include_diagonal=False, distance=1):
        """Get neighboring positions or agents.
        
        Args:
            position_or_agent_id: Either a position tuple or agent ID
            include_diagonal: Whether to include diagonal neighbors
            distance: Maximum distance for neighbors
        """
        if isinstance(position_or_agent_id, (tuple, list)):
            # Position-based neighbor search
            position = position_or_agent_id
            neighbors = []
            
            if include_diagonal:
                # Include all 8 neighbors in 2D (or more in higher dimensions)
                offsets = []
                if len(self.dimensions) == 2:
                    for dx in range(-distance, distance + 1):
                        for dy in range(-distance, distance + 1):
                            if dx == 0 and dy == 0:
                                continue
                            offsets.append((dx, dy))
                else:
                    import itertools
                    ranges = [range(-distance, distance + 1) for _ in self.dimensions]
                    offsets = [offset for offset in itertools.product(*ranges) 
                              if not all(o == 0 for o in offset)]
            else:
                # Only orthogonal neighbors
                offsets = []
                for dim in range(len(self.dimensions)):
                    for offset_val in range(-distance, distance + 1):
                        if offset_val == 0:
                            continue
                        offset = [0] * len(self.dimensions)
                        offset[dim] = offset_val
                        offsets.append(tuple(offset))
            
            seen = set()
            origin = tuple(position)
            for offset in offsets:
                new_pos = []
                valid = True
                for i, (coord, off) in enumerate(zip(position, offset)):
                    new_coord = coord + off
                    if self.torus:
                        new_coord = new_coord % self.dimensions[i]
                    elif not (0 <= new_coord < self.dimensions[i]):
                        valid = False
                        break
                    new_pos.append(new_coord)

                if not valid:
                    continue
                new_tuple = tuple(new_pos)
                # Dedup wrap-around collisions and exclude origin.
                if new_tuple == origin or new_tuple in seen:
                    continue
                seen.add(new_tuple)
                neighbors.append(new_tuple)

            return neighbors
        else:
            # Agent-based neighbor search
            agent_id = position_or_agent_id
            if self.df.is_empty():
                return []
                
            agent_pos_rows = self.df.filter(pl.col('id') == agent_id)
            if agent_pos_rows.is_empty():
                return []
                
            agent_pos = agent_pos_rows['grid_position'].item()
            if agent_pos is None:
                return []
            
            neighbor_positions = self.get_neighbors(agent_pos, include_diagonal, distance)
            
            # Find agents at neighbor positions
            neighbors = []
            for pos in neighbor_positions:
                pos_agents = self.df.filter(
                    pl.col('grid_position').map_elements(lambda x: x == pos if x is not None else False)
                )['id'].to_list()
                neighbors.extend(pos_agents)
                
            return neighbors
    
    def get_distance(self, pos1_or_agent1, pos2_or_agent2) -> float:
        """Calculate Manhattan distance between two positions or agents."""
        # Handle different input types
        if isinstance(pos1_or_agent1, (tuple, list)):
            pos1, pos2 = pos1_or_agent1, pos2_or_agent2
        else:
            # Agent IDs
            if self.df.is_empty():
                return float('inf')
                
            agent1_rows = self.df.filter(pl.col('id') == pos1_or_agent1)
            agent2_rows = self.df.filter(pl.col('id') == pos2_or_agent2)
            
            if agent1_rows.is_empty() or agent2_rows.is_empty():
                return float('inf')
                
            pos1 = agent1_rows['grid_position'].item()
            pos2 = agent2_rows['grid_position'].item()
            
            if pos1 is None or pos2 is None:
                return float('inf')
        
        # Calculate Manhattan distance
        if self.torus:
            # Handle torus wrapping
            distance = 0
            for p1, p2, dim in zip(pos1, pos2, self.dimensions):
                diff = abs(p1 - p2)
                wrap_diff = dim - diff
                distance += min(diff, wrap_diff)
            return distance
        else:
            return sum(abs(p1 - p2) for p1, p2 in zip(pos1, pos2))
    
    def is_valid_position(self, position):
        """Check if a position is valid in the grid."""
        if len(position) != len(self.dimensions):
            return False
        
        for coord, dim in zip(position, self.dimensions):
            if not (0 <= coord < dim):
                return False
        
        return True
    
    def random_position(self):
        """Get a random position in the grid."""
        rng = getattr(self.model, 'rng', None) or np.random.default_rng()
        return tuple(int(rng.integers(0, dim)) for dim in self.dimensions)
    
    def empty_positions(self) -> List[Tuple[int, int]]:
        """Return a list of empty positions."""
        occupied = set()
        if hasattr(self, 'df') and not self.df.is_empty() and 'grid_position' in self.df.columns:
            # Handle both list and tuple types from Polars
            if self.df['grid_position'].dtype == pl.Object:
                occupied = set(self.df['grid_position'].to_list())
            else:
                # Likely list type, convert to tuples
                occupied = set(tuple(p) if isinstance(p, list) else p for p in self.df['grid_position'].to_list())
            
        return [pos for pos in self.positions if pos not in occupied]
    
    def move_agent(self, agent_id: int, new_position: Position) -> None:
        """Move an agent to a new grid position."""
        if new_position.topology_type != 'grid':
            raise ValueError("Position must be of type 'grid'")
            
        # Validate position
        if len(new_position.coordinates) != len(self.dimensions):
            raise ValueError("Position dimensions don't match grid dimensions")
            
        coords = list(new_position.coordinates)
        for i, (coord, dim) in enumerate(zip(coords, self.dimensions)):
            if not (0 <= coord < dim):
                if not self.torus:
                    raise ValueError("Position out of bounds")
                coords[i] = coord % dim
                
        # Update agent position
        if hasattr(self, 'df') and not self.df.is_empty():
            # Ensure tuple is treated as object
            coords_val = tuple(coords)
            self.df = self.df.with_columns([
                pl.when(pl.col('id') == agent_id)
                .then(pl.lit(coords_val, dtype=pl.Object))
                .otherwise(pl.col('grid_position'))
                .alias('grid_position')
            ])

class SpaceEnvironment(Environment):
    """N-dimensional continuous space environment."""
    
    def __init__(self, model, bounds: List[Tuple[float, float]], torus: bool = False):
        """
        Initialize continuous space environment.
        
        Args:
            model: Reference to the model
            bounds: List of (min, max) tuples for each dimension
            torus: Whether space wraps around boundaries
        """
        super().__init__(model)
        self.bounds = bounds
        self.dimensions = len(bounds)
        self.torus = torus
        
        # Initialise space-specific columns on the environment's frame. Chain
        # both onto self.df so neither overwrites the other (the previous code
        # re-read model.agents_df for the second column, dropping the first).
        if model is not None and hasattr(model, 'agents_df'):
            self.df = model.agents_df
            if 'space_position' not in self.df.columns:
                self.df = self.df.with_columns(
                    pl.lit(None, dtype=pl.Object).alias('space_position')
                )
            if 'space_distance' not in self.df.columns:
                self.df = self.df.with_columns(
                    pl.lit(0.0).alias('space_distance')
                )
    
    def positions_array(self):
        """Return ``(ids, positions)`` over agents that have a position set.

        ``ids`` is shape ``(M,)``, ``positions`` is ``(M, d)``. The single place
        the tuple-valued ``space_position`` column is unpacked into a contiguous
        matrix for vectorized distance queries.
        """
        if not hasattr(self, 'df') or self.df.is_empty():
            return np.empty(0, dtype=np.int64), np.empty((0, self.dimensions))
        sub = self.df.filter(pl.col('space_position').is_not_null())
        if sub.is_empty():
            return np.empty(0, dtype=np.int64), np.empty((0, self.dimensions))
        ids = sub['id'].to_numpy()
        pos = np.asarray(sub['space_position'].to_list(), dtype=np.float64)
        return ids, pos

    def _distances_to(self, query, pos):
        """Vectorized Euclidean distances from ``query`` (d,) to ``pos`` (M, d).

        Torus-aware: under ``self.torus`` each per-axis gap is
        ``min(|delta|, range - |delta|)``.
        """
        diff = np.abs(pos - np.asarray(query, dtype=np.float64))
        if self.torus:
            ranges = np.array([mx - mn for mn, mx in self.bounds], dtype=np.float64)
            diff = np.minimum(diff, ranges - diff)
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))

    def get_neighbors(self, pos_or_agent_id, radius: float) -> List[int]:
        """Get agents within ``radius`` of a position or an agent (vectorized).

        Replaces the former per-row Python loop / ``map_elements`` scan with a
        single numpy pass, plus an optional scipy KD-tree for large populations.
        Semantics are preserved: the queried agent is included (distance
        ``0 <= radius``) and agents without a position are skipped.
        """
        ids, pos = self.positions_array()
        if ids.size == 0:
            return []

        if isinstance(pos_or_agent_id, (list, tuple)):
            query = pos_or_agent_id
        else:
            match = self.df.filter(pl.col('id') == pos_or_agent_id)
            if match.is_empty():
                return []
            query = match['space_position'].item()
            if query is None:
                return []

        # Large-N, non-torus: KD-tree. Otherwise: vectorized brute force.
        if _HAS_KDTREE and not self.torus and ids.size >= _KDTREE_MIN_AGENTS:
            tree = _cKDTree(pos)
            idx = np.sort(np.asarray(
                tree.query_ball_point(np.asarray(query, dtype=np.float64), radius),
                dtype=np.int64,
            ))
            return ids[idx].tolist()
        dists = self._distances_to(query, pos)
        return ids[dists <= radius].tolist()

    def set_position(self, agent_ids, coords) -> None:
        """Vectorized position write for many agents at once.

        ``agent_ids`` is shape ``(M,)``; ``coords`` is ``(M, d)`` (or ``(d,)``
        for a single agent). A bulk alternative to repeated :meth:`move_agent`.
        """
        if not hasattr(self, 'df') or self.df.is_empty():
            return
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim == 1:
            coords = coords[None, :]
        agent_ids = np.asarray(agent_ids).ravel()
        mapping = {int(i): tuple(float(c) for c in row)
                   for i, row in zip(agent_ids, coords)}
        new = [mapping.get(int(i), p) for i, p in
               zip(self.df['id'].to_list(), self.df['space_position'].to_list())]
        self.df = self.df.with_columns(pl.Series('space_position', new, dtype=pl.Object))
    
    def get_distance(self, pos1_or_agent1, pos2_or_agent2) -> float:
        """Calculate Euclidean distance between two positions or agents."""
        # Handle different input types
        if isinstance(pos1_or_agent1, (tuple, list)):
            pos1, pos2 = pos1_or_agent1, pos2_or_agent2
            return self._calculate_distance(pos1, pos2)
        else:
            # Agent IDs
            if self.df.is_empty():
                return float('inf')
                
            pos1_rows = self.df.filter(pl.col('id') == pos1_or_agent1)
            pos2_rows = self.df.filter(pl.col('id') == pos2_or_agent2)
            
            if pos1_rows.is_empty() or pos2_rows.is_empty():
                return float('inf')
                
            pos1 = pos1_rows['space_position'].item()
            pos2 = pos2_rows['space_position'].item()
            
            if pos1 is None or pos2 is None:
                return float('inf')
                
            return self._calculate_distance(pos1, pos2)
    
    def is_valid_position(self, position):
        """Check if a position is within bounds."""
        if len(position) != self.dimensions:
            return False
        
        for coord, (min_val, max_val) in zip(position, self.bounds):
            if not (min_val <= coord <= max_val):
                return False
        
        return True
    
    def random_position(self):
        """Get a random position within bounds."""
        rng = getattr(self.model, 'rng', None) or np.random.default_rng()

        position = []
        for min_val, max_val in self.bounds:
            coord = rng.uniform(min_val, max_val)
            position.append(coord)
        
        return position
    

    
    def move_agent(self, agent_id: int, new_position: Position) -> None:
        """Move an agent to a new continuous position."""
        if new_position.topology_type != 'space':
            raise ValueError("Position must be of type 'space'")
            
        # Validate position
        if len(new_position.coordinates) != self.dimensions:
            raise ValueError("Position dimensions don't match space dimensions")
            
        coords = list(new_position.coordinates)
        for i, (coord, (min_val, max_val)) in enumerate(zip(coords, self.bounds)):
            if self.torus:
                # Wrap coordinates for torus topology
                range_size = max_val - min_val
                coords[i] = min_val + ((coord - min_val) % range_size)
            elif not (min_val <= coord <= max_val):
                raise ValueError("Position out of bounds")
                
        # Update agent position. set_position normalises to an Object column,
        # avoiding dtype conflicts between list- and object-typed positions.
        self.set_position([agent_id], [tuple(coords)])
    
    def _calculate_distance(self, pos1: Tuple[float, ...], pos2: Tuple[float, ...]) -> float:
        """Calculate Euclidean distance between two positions."""
        if self.torus:
            # Handle torus wrapping
            distance_squared = 0
            for p1, p2, (min_val, max_val) in zip(pos1, pos2, self.bounds):
                range_size = max_val - min_val
                diff = abs(p1 - p2)
                wrap_diff = range_size - diff
                min_diff = min(diff, wrap_diff)
                distance_squared += min_diff ** 2
            return np.sqrt(distance_squared)
        else:
            return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)))

class NetworkEnvironment(Environment):
    """Graph-based network environment."""
    
    def __init__(self, model, graph: Optional[nx.Graph] = None):
        """
        Initialize network environment.
        
        Args:
            model: Reference to the model
            graph: Optional initial network graph
        """
        super().__init__(model)
        if graph is not None:
            self.graph = graph
        else:
            self.graph = nx.Graph()
        
        # Initialise network-specific columns on the model's DataFrame if needed.
        # Chain both onto df so the second column doesn't drop the first (the
        # previous code re-read model.agents_df for each, losing node_id).
        if model is not None and hasattr(model, 'agents_df'):
            df = model.agents_df
            if 'node_id' not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias('node_id'))
            if 'network_distance' not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias('network_distance'))
            self.df = df
    
    @property
    def nodes(self):
        """Get all nodes in the network."""
        return list(self.graph.nodes())
    
    @property
    def edges(self):
        """Get all edges in the network."""
        return list(self.graph.edges())
    
    def add_node(self, node_id, **attr):
        """Add a node to the network."""
        self.graph.add_node(node_id, **attr)
    
    def remove_node(self, node_id):
        """Remove a node from the network."""
        self.graph.remove_node(node_id)
    
    def get_neighbors(self, node_or_agent_id) -> List[int]:
        """Get neighboring nodes or agents in the network."""
        if isinstance(node_or_agent_id, int) and self.graph.has_node(node_or_agent_id):
            # Direct node ID
            return list(self.graph.neighbors(node_or_agent_id))
        else:
            # Agent ID
            if self.df.is_empty():
                return []
                
            agent_rows = self.df.filter(pl.col('id') == node_or_agent_id)
            if agent_rows.is_empty():
                return []
                
            node_id = agent_rows['node_id'].item()
            if node_id is None:
                return []
                
            # Get neighbors from graph
            neighbor_nodes = list(self.graph.neighbors(node_id))
            
            # Convert node IDs to agent IDs
            return self.df.filter(pl.col('node_id').is_in(neighbor_nodes))['id'].to_list()
    
    def get_distance(self, node1_or_agent1, node2_or_agent2) -> float:
        """Calculate shortest path distance between two nodes or agents."""
        # Handle different input types
        if isinstance(node1_or_agent1, int) and self.graph.has_node(node1_or_agent1):
            node1, node2 = node1_or_agent1, node2_or_agent2
        else:
            # Agent IDs
            if self.df.is_empty():
                return float('inf')
                
            agent1_rows = self.df.filter(pl.col('id') == node1_or_agent1)
            agent2_rows = self.df.filter(pl.col('id') == node2_or_agent2)
            
            if agent1_rows.is_empty() or agent2_rows.is_empty():
                return float('inf')
                
            node1 = agent1_rows['node_id'].item()
            node2 = agent2_rows['node_id'].item()
            
            if node1 is None or node2 is None:
                return float('inf')
        
        try:
            return nx.shortest_path_length(self.graph, node1, node2)
        except nx.NetworkXNoPath:
            return float('inf')
    
    def get_degree(self, node_or_agent_id):
        """Get the degree of a node or agent."""
        if isinstance(node_or_agent_id, int) and self.graph.has_node(node_or_agent_id):
            return self.graph.degree(node_or_agent_id)
        else:
            # Agent ID
            if self.df.is_empty():
                return 0
                
            agent_rows = self.df.filter(pl.col('id') == node_or_agent_id)
            if agent_rows.is_empty():
                return 0
                
            node_id = agent_rows['node_id'].item()
            if node_id is None or not self.graph.has_node(node_id):
                return 0
                
            return self.graph.degree(node_id)
    
    def get_clustering(self, node_or_agent_id=None):
        """Get clustering coefficient for a node, agent, or the entire network."""
        if node_or_agent_id is None:
            # Return overall clustering
            return nx.average_clustering(self.graph)
        elif isinstance(node_or_agent_id, int) and self.graph.has_node(node_or_agent_id):
            # Direct node ID
            return nx.clustering(self.graph, node_or_agent_id)
        else:
            # Agent ID
            if self.df.is_empty():
                return 0.0
                
            agent_rows = self.df.filter(pl.col('id') == node_or_agent_id)
            if agent_rows.is_empty():
                return 0.0
                
            node_id = agent_rows['node_id'].item()
            if node_id is None or not self.graph.has_node(node_id):
                return 0.0
                
            return nx.clustering(self.graph, node_id)
    
    def random_node(self):
        """Get a random node from the network."""
        if not self.graph.nodes():
            return None
        rng = getattr(self.model, 'rng', None) or np.random.default_rng()
        nodes = self.nodes
        return nodes[int(rng.integers(0, len(nodes)))]
    
    def move_agent(self, agent_id: int, new_position: Position) -> None:
        """Move an agent to a new node in the network."""
        if new_position.topology_type != 'network':
            raise ValueError("Position must be of type 'network'")
            
        # Validate node exists
        if not self.graph.has_node(new_position.coordinates[0]):
            raise ValueError("Node does not exist in network")
            
        # Update agent position
        if hasattr(self, 'df') and not self.df.is_empty():
            self.df = self.df.with_columns([
                pl.when(pl.col('id') == agent_id)
                .then(pl.lit(new_position.coordinates[0]))
                .otherwise(pl.col('node_id'))
                .alias('node_id')
            ])
    
    def add_edge(self, node1_or_agent1, node2_or_agent2, **attr) -> None:
        """Add an edge between two nodes or agents."""
        if isinstance(node1_or_agent1, int) and self.graph.has_node(node1_or_agent1):
            # Direct node IDs
            node1, node2 = node1_or_agent1, node2_or_agent2
        else:
            # Agent IDs
            if self.df.is_empty():
                raise ValueError("No agents in environment")
                
            agent1_rows = self.df.filter(pl.col('id') == node1_or_agent1)
            agent2_rows = self.df.filter(pl.col('id') == node2_or_agent2)
            
            if agent1_rows.is_empty() or agent2_rows.is_empty():
                raise ValueError("One or both agents not found")
                
            node1 = agent1_rows['node_id'].item()
            node2 = agent2_rows['node_id'].item()
            
            if node1 is None or node2 is None:
                raise ValueError("Both agents must be assigned to nodes")
        
        self.graph.add_edge(node1, node2, **attr)
    
    def remove_edge(self, node1_or_agent1, node2_or_agent2) -> None:
        """Remove an edge between two nodes or agents."""
        if isinstance(node1_or_agent1, int) and self.graph.has_node(node1_or_agent1):
            # Direct node IDs
            node1, node2 = node1_or_agent1, node2_or_agent2
        else:
            # Agent IDs
            if self.df.is_empty():
                raise ValueError("No agents in environment")
                
            agent1_rows = self.df.filter(pl.col('id') == node1_or_agent1)
            agent2_rows = self.df.filter(pl.col('id') == node2_or_agent2)
            
            if agent1_rows.is_empty() or agent2_rows.is_empty():
                raise ValueError("One or both agents not found")
                
            node1 = agent1_rows['node_id'].item()
            node2 = agent2_rows['node_id'].item()
            
            if node1 is None or node2 is None:
                raise ValueError("Both agents must be assigned to nodes")
        
        self.graph.remove_edge(node1, node2) 