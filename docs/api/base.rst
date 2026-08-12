Base Classes
============

.. automodule:: ambr.base
   :members:
   :undoc-members:
   :show-inheritance:

Low-level primitives used by AMBER internals. **For application models and
agents, subclass** :class:`~ambr.model.Model` **and**
:class:`~ambr.agent.Agent` **instead** (see :doc:`model` and :doc:`agent`).
``BaseModel`` / ``BaseAgent`` do not implement the full simulation lifecycle
or DataFrame-backed attribute sync.

BaseModel
---------

.. autoclass:: ambr.BaseModel
   :members:
   :undoc-members:

Abstract foundation for models (parameters, RNG, bare DataFrames). User
models should inherit from :class:`~ambr.model.Model`, which adds
``run()``, reporters, environments, and agent management:

.. code-block:: python

   import ambr as am

   class CustomModel(am.Model):
       def setup(self):
           self.add_agents(self.p.get("n_agents", 10), wealth=1)

       def step(self):
           # Vectorized or OOP step logic
           pass

BaseAgent
---------

.. autoclass:: ambr.BaseAgent
   :members:
   :undoc-members:

Minimal agent shell (``model``, ``id``, ``p`` only). **Do not subclass this
for normal simulations.** Creation always calls ``agent.setup()``, which
``BaseAgent`` does not define, and attribute assignment is not synced to
the population table. Use :class:`~ambr.agent.Agent`:

.. code-block:: python

   import ambr as am

   class CustomAgent(am.Agent):
       def setup(self):
           self.custom_property = "value"

       def step(self):
           pass

**What BaseAgent provides (only):**

* Parameter access via ``self.p``
* Model reference via ``self.model``
* Unique ``id``

**What you get only from** :class:`~ambr.agent.Agent` **:**

* Default ``setup()`` hook (safe no-op; override as needed)
* Attribute writes queued into the columnar population DataFrame
