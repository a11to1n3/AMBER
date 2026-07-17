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

   # Run the model (fluent placement, 0.4.3)
   model = MyModel({'steps': 100, 'seed': 42})
   results = model.cpu(mode="vectorized").run()
   # results = model.gpu().run()   # same step body on GPU

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

**Execution / placement (0.4.3):**

* ``cpu(mode=None)`` - Place the next ``run`` on CPU (optional
  ``mode='vectorized'|'oop'``); returns ``self`` for chaining
* ``gpu(mode=None)`` - Place the next ``run`` on GPU with device-resident
  columns; same view-API ``step`` as CPU
* ``run(...)`` - Execute the full simulation and return results.
  Accepts ``device=``, ``mode=``, ``contract=``; legacy ``backend=`` still
  works but is deprecated
* ``run_step()`` - Execute a single time step

Mode defaults to ``vectorized``. Fluent placement and ``run(mode=...)`` /
``run(device=...)`` compose: kwargs to ``run`` override what
``cpu()`` / ``gpu()`` set when both are used.
