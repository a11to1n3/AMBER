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

**Usage:** Placement uses the ``grid_position`` column via
:meth:`~ambr.GridEnvironment.add_agent_from_id` (not bare ``x`` / ``y``).
See also :doc:`../environments_schelling`.

.. code-block:: python

   import ambr as am

   class GridDemo(am.Model):
       def setup(self):
           self.grid = am.GridEnvironment(self, size=(10, 10))
           # Create population first, then place on the grid.
           self.add_agents(5)
           empty = self.grid.empty_positions()
           for i, aid in enumerate(self.agents.ids.to_list()):
               self.grid.add_agent_from_id(aid, empty[i])
           pos = empty[0]
           print("neighbors of", pos, ":", self.grid.get_neighbors(pos))

   GridDemo({"steps": 1, "seed": 0, "show_progress": False}).run()

Space Environment
-----------------

.. autoclass:: ambr.SpaceEnvironment
   :members:
   :undoc-members:

The SpaceEnvironment provides continuous 2D space with configurable boundaries.

**Usage:**

.. code-block:: python

   import ambr as am

   class SpaceDemo(am.Model):
       def setup(self):
           # Create agents first so SpaceEnvironment can attach columns.
           self.add_agents(3)
           self.space = am.SpaceEnvironment(self, bounds=[(0, 100), (0, 100)])
           for i, aid in enumerate(self.agents.ids.to_list()):
               self.space.set_position(aid, (25.5 + i, 37.2))
           print(self.space.get_neighbors((25.5, 37.2), radius=5.0))

   SpaceDemo({"steps": 1, "seed": 0, "show_progress": False}).run()

Network Environment
-------------------

.. autoclass:: ambr.NetworkEnvironment
   :members:
   :undoc-members:

The NetworkEnvironment provides graph-based topology for agent interactions.

**Usage:** Agents store placement in ``node_id``. Create the population
**before** the network so the environment can attach that column, then assign
``node_id`` for each agent.

.. code-block:: python

   import ambr as am
   import networkx as nx

   class NetDemo(am.Model):
       def setup(self):
           # Population first so NetworkEnvironment can attach node_id.
           self.add_agents(5)
           G = nx.path_graph(5)  # connected so node neighbors are non-empty
           self.network = am.NetworkEnvironment(self, G)
           nodes = list(G.nodes())
           for i, aid in enumerate(self.agents.ids.to_list()):
               self.agents.at[aid].set(node_id=nodes[i % len(nodes)])
           # When agent ids and node ids overlap (0..n-1), agent wins by
           # default. Pass as_node=True for explicit graph-node queries.
           aid0 = self.agents.ids.to_list()[0]
           print("agent 0 neighbors (agent ids):", self.network.get_neighbors(aid0))
           print("node 0 neighbors (node ids):", self.network.get_neighbors(
               0, as_node=True
           ))

   NetDemo({"steps": 1, "seed": 0, "show_progress": False}).run()
