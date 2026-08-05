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

       def step_vectorized(self):
           # Columnar / array-native implementation
           pass

       def step_oop(self):
           # Optional tracked-Agent implementation
           pass

   # Run the model (fluent placement)
   model = MyModel({'steps': 100, 'seed': 42, 'show_progress': False})
   results = model.cpu(mode="vectorized").run()
   # results = model.gpu().run()   # vectorized lane; needs NVIDIA + CuPy

Key Methods
-----------

**Lifecycle Methods:**

* ``setup()`` - Called once at the beginning to initialize the model
* ``step_vectorized()`` - Called for vectorized CPU/GPU runs
* ``step_oop()`` - Called for CPU OOP runs with tracked Agent objects
* ``step()`` - Backwards-compatible fallback when a lane hook is not defined
* ``update()`` - Called after step() to update model state
* ``end()`` - Called once at the end of the simulation

**Data Management:**

* ``add_agent(agent)`` - Add a new agent to the model
* ``update_agent_data(agent_id, data)`` - Update data for a specific agent
* ``get_agent_data(agent_id)`` - Retrieve data for a specific agent
* ``record_model(name, value)`` - Record a model-level metric

**Execution / placement (0.4.4):**

* ``cpu(mode=None)`` - Place the next ``run`` on CPU (optional
  ``mode='vectorized'|'oop'``); returns ``self`` for chaining
* ``gpu(mode=None)`` - Place the next ``run`` on GPU with device-resident
  columns; GPU runs are vectorized-only
* ``approve_fast_path(evidence)`` - Explicitly allow a private optimized GPU
  loop on this model instance and retain the caller-supplied evidence label;
  AMBER does not verify that label (requires ``contract="off"``)
* ``revoke_fast_path_approval()`` - Return the instance to the general runner
* ``run(...)`` - Execute the full simulation and return results.
  Accepts ``device=``, ``mode=``, ``contract=``; legacy ``backend=`` still
  works but is deprecated
* ``run_step()`` - Execute a single time step

Mode defaults to ``vectorized``. Fluent placement and ``run(mode=...)`` /
``run(device=...)`` compose: kwargs to ``run`` override what
``cpu()`` / ``gpu()`` set when both are used.
