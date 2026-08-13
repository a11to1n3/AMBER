Agent
=====

.. automodule:: ambr.agent
   :members:
   :undoc-members:
   :show-inheritance:

The Agent class represents individual entities in your simulation. Each agent has:

* A unique ID within the model
* Access to the model and its parameters
* Ability to store custom attributes
* Methods for interacting with other agents and the environment

Basic Usage
-----------

.. code-block:: python

   import ambr as am

   class MyAgent(am.Agent):
       def setup(self):
           self.wealth = 10
           self.age = 0

       def step(self):
           # Define agent behavior
           self.age += 1

   class MyModel(am.Model):
       def setup(self):
           # bulk-create tracked Python agents in one call
           self.add_agents(100, agent_class=MyAgent)

       def step(self):
           for agent in self.agents:
               agent.step()

   results = MyModel({"steps": 5, "seed": 0, "show_progress": False}).run()
   print(results.agents.head().to_dicts())

Custom Agent Classes
--------------------

User-facing custom agents should subclass :class:`~ambr.agent.Agent`
(not :class:`~ambr.base.BaseAgent`). AMBER always calls ``agent.setup()``
after construction; ``Agent`` provides a default no-op ``setup`` and
routes attribute writes into the columnar DataFrame. ``BaseAgent`` is a
low-level primitive without ``setup`` or DataFrame sync — inheriting from
it alone raises ``AttributeError: ... has no attribute 'setup'``.

.. code-block:: python

   import ambr as am

   class CustomAgent(am.Agent):
       def setup(self):
           # Called automatically when the agent is created
           self.custom_property = "value"

       def step(self):
           # Optional OOP step; vectorized models usually skip this
           pass

       def custom_method(self):
           return self.custom_property

See :doc:`base` for when ``BaseAgent`` is appropriate (library internals /
advanced embedding only).

Agent Properties
----------------

**Built-in Properties:**

* ``id`` - Unique identifier for the agent
* ``model`` - Reference to the parent model
* ``p`` - Shortcut to model parameters (``model.p``)

**Custom Properties:**

On ``am.Agent``, assignments sync to the population DataFrame (except
``model`` / ``id`` / ``p`` and private ``_`` names):

.. code-block:: python

   agent.wealth = 100
   agent.position = (5, 10)
   agent.state = "active"
