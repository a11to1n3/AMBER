Sequences
=========

.. automodule:: ambr.sequences
   :members:
   :undoc-members:
   :show-inheritance:

The sequences module provides specialized data structures for managing collections of agents.

AgentList
---------

.. autoclass:: ambr.AgentList
   :members:
   :undoc-members:

The AgentList class provides a list-like interface for managing collections of agents with additional functionality.

**Usage:**

.. code-block:: python

   # Create AgentList with agents
   agents = [am.Agent(model, i) for i in range(10)]
   agent_list = am.AgentList(model, agents)
   
   # Or create with count and type
   agent_list = am.AgentList(model, 10, am.Agent)
   
   # Use like a regular list
   agent_list.append(new_agent)
   agent_list.remove(old_agent)
   
   # Access agents
   first_agent = agent_list[0]
   for agent in agent_list:
       # Do something with agent
       pass

**Features:**

* List-like interface (append, remove, insert, etc.)
* Iteration support
* Indexing and slicing
* Length and containment checks
* Agent type validation
* Integration with AMBER models 