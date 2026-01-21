Model
=====

.. automodule:: ambr.model
   :members:
   :undoc-members:
   :show-inheritance:

The Model class is the foundation of any AMBER simulation. It provides the framework for:

* Managing simulation time and execution
* Storing and updating agent data
* Recording model-level metrics
* Coordinating agent behaviors

Basic Usage
-----------

.. code-block:: python

   import ambr as am
   
   class MyModel(am.Model):
       def setup(self):
           # Initialize agents and environment
           pass
       
       def step(self):
           # Define what happens each time step
           pass
   
   # Run the model
   model = MyModel({'steps': 100, 'seed': 42})
   results = model.run()

Key Methods
-----------

**Lifecycle Methods:**

* ``setup()`` - Called once at the beginning to initialize the model
* ``step()`` - Called each time step to update agent states
* ``update()`` - Called after step() to update model state
* ``end()`` - Called once at the end of the simulation

**Data Management:**

* ``add_agent(agent)`` - Add a new agent to the model
* ``update_agent_data(agent_id, data)`` - Update data for a specific agent
* ``get_agent_data(agent_id)`` - Retrieve data for a specific agent
* ``record_model(name, value)`` - Record a model-level metric

**Execution:**

* ``run()`` - Execute the full simulation and return results
* ``run_step()`` - Execute a single time step 