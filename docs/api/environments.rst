Environments
============

.. automodule:: ambr.environments
   :members:
   :undoc-members:
   :show-inheritance:

AMBER provides several built-in environment types for different spatial and network topologies.

Grid Environment
----------------

.. autoclass:: ambr.GridEnvironment
   :members:
   :undoc-members:

The GridEnvironment provides a 2D grid-based space where agents can be positioned and move around.

**Usage:**

.. code-block:: python

   # Create a 10x10 grid
   grid = am.GridEnvironment(model, size=(10, 10))
   
   # Place an agent
   position = grid.random_position()
   agent.position = position
   
   # Get neighbors
   neighbors = grid.get_neighbors(position)

Space Environment
-----------------

.. autoclass:: ambr.SpaceEnvironment
   :members:
   :undoc-members:

The SpaceEnvironment provides continuous 2D space with configurable boundaries.

**Usage:**

.. code-block:: python

   # Create continuous space
   space = am.SpaceEnvironment(model, bounds=[(0, 100), (0, 100)])
   
   # Place an agent
   position = (25.5, 37.2)
   agent.position = position
   
   # Get neighbors within radius
   neighbors = space.get_neighbors(position, radius=5.0)

Network Environment
-------------------

.. autoclass:: ambr.NetworkEnvironment
   :members:
   :undoc-members:

The NetworkEnvironment provides graph-based topology for agent interactions.

**Usage:**

.. code-block:: python

   import networkx as nx
   
   # Create network from NetworkX graph
   G = nx.erdos_renyi_graph(100, 0.1)
   network = am.NetworkEnvironment(model, G)
   
   # Place agent on node
   agent.node = 42
   
   # Get connected neighbors
   neighbors = network.get_neighbors(42) 