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

   import ambr as am

   class GridDemo(am.Model):
       def setup(self):
           self.grid = am.GridEnvironment(self, size=(10, 10))
           self.add_agents(5, x=0, y=0)
           # place first agent at a random cell
           pos = self.grid.random_position()
           self.agents.at[self.agents.ids.to_list()[0]].set(x=pos[0], y=pos[1])
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

**Usage:**

.. code-block:: python

   import ambr as am
   import networkx as nx

   class NetDemo(am.Model):
       def setup(self):
           G = nx.erdos_renyi_graph(20, 0.2, seed=0)
           self.network = am.NetworkEnvironment(self, G)
           self.add_agents(5, node=0)
           print("neighbors of 0:", self.network.get_neighbors(0))

   NetDemo({"steps": 1, "seed": 0, "show_progress": False}).run()
