Agent
=====

.. automodule:: amber.agent
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

   import amber as am
   
   class MyAgent(am.Agent):
       def __init__(self, model, agent_id):
           super().__init__(model, agent_id)
           self.wealth = 10
           self.age = 0
       
       def step(self):
           # Define agent behavior
           self.age += 1
           # ... other behaviors
   
   # In your model's setup():
   agent = MyAgent(self, agent_id)
   self.add_agent(agent)

Custom Agent Classes
--------------------

You can create custom agent classes by inheriting from ``BaseAgent``:

.. code-block:: python

   from amber.base import BaseAgent
   
   class CustomAgent(BaseAgent):
       def __init__(self, model, agent_id):
           super().__init__(model, agent_id)
           self.custom_property = "value"
       
       def custom_method(self):
           # Your custom behavior
           pass

Agent Properties
----------------

**Built-in Properties:**

* ``id`` - Unique identifier for the agent
* ``model`` - Reference to the parent model
* ``p`` - Shortcut to model parameters (``model.p``)

**Custom Properties:**

You can add any custom properties to agents by setting them as attributes:

.. code-block:: python

   agent.wealth = 100
   agent.position = (5, 10)
   agent.state = "active" 