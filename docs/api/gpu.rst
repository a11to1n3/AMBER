GPU backend
===========

AMBER's GPU support has two layers:

1. **Native placement (0.4.3, preferred for a single large run)** —
   ``model.gpu().run()`` on the same view-API ``Model`` you use on CPU.
   Numeric columns stay device-resident for the run; the ``step`` body does
   not change. Switch back with ``model.cpu(mode="vectorized").run()``.
2. **Array-module helpers** — ``get_array_module``, ``to_device``,
   ``to_host``, ``scatter_add`` for code that writes against ``xp``
   (CuPy when available, else NumPy).

Requires **NVIDIA GPU + CuPy** (not Apple Metal/MPS). Install CuPy matching
your CUDA toolkit (see :doc:`../installation` and :doc:`../going_faster`).

Native placement
----------------

.. code-block:: python

   import ambr as am

   class WealthModel(am.Model):
       def setup(self):
           n = int(self.p.get("n", 10_000))
           self.add_agents(n, wealth=self.rng.integers(1, 10, size=n))

       def step(self):
           donors = self.agents.where(self.agents.wealth > 0)
           if len(donors) == 0:
               return
           donors.wealth -= 1
           ids = self.agents.ids.to_numpy()
           recipients = self.rng.choice(ids, size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

   # Same Model + step on CPU or GPU
   results = WealthModel({"n": 100_000, "steps": 50, "seed": 0}).gpu().run()
   # results = WealthModel(...).cpu(mode="vectorized").run()

Mode defaults to ``vectorized``. You can also write
``model.gpu(mode="vectorized").run()`` or pass ``mode=`` / ``device=`` to
:meth:`~ambr.model.Model.run` (``run`` overrides fluent placement when both
are given).

For many short replicate runs (calibration), use the ensemble path instead:
:doc:`gpu_ensemble`.

Array module
------------

An array-module abstraction over CuPy with a NumPy fallback, so device code is
portable: ``get_array_module``, ``to_device``, and ``to_host`` resolve to CuPy
when a GPU is present and fall back to NumPy when it is not.

.. automodule:: ambr.gpu
   :members:
   :undoc-members:
   :show-inheritance:
