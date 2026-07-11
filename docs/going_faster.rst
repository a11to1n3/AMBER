Going faster (lanes)
====================

AMBER does **not** hide speed behind a magic ``gpu=True`` switch on the view
API (that would silently move Polars data over PCIe every step). Instead there
are four **lanes**. You “turn one on” by writing that style — and helpers tell
you which to pick.

Check this machine
------------------

.. code-block:: python

   import ambr as am
   am.print_status()
   print(am.recommend(100_000))          # single large run
   print(am.recommend(1_000, ensemble=True))  # many short runs / calibration

Lane 1 — OOP (AgentPy-shaped)
-----------------------------

Always available. Best for small N or sequential logic. See :doc:`from_agentpy`.

Lane 2 — Vectorized (default fast path)
---------------------------------------

**There is no flag.** Using the view API *is* the vectorized lane::

   donors = self.agents.where(self.agents.wealth > 0)
   donors.wealth -= 1
   # or one-liner:
   self.agents.update_where(self.agents.wealth > 0, wealth=self.agents.wealth - 1)

   self.agents.at[ids].scatter_add(wealth=1)

Use this for almost all CPU models above a few thousand agents.

Lane 3 — Tensor (dense NumPy kernels)
-------------------------------------

When a step is interaction-heavy (O(N²) distances, etc.)::

   x, _ = self.agents.borrow("x")
   y, _ = self.agents.borrow("y")
   # ... pure NumPy ...
   self.agents.commit(x=new_x, y=new_y)

See :doc:`api/tensor_lane` and ``examples/flocking_tensor.py``.

Lane 4 — GPU
------------

Install CuPy (match your CUDA), then either:

**A. Single large run —** :class:`~ambr.lanes.ArrayKernelModel`::

   import ambr as am

   class Drift(am.ArrayKernelModel):
       def init_state(self, xp, n, rng, p):
           return {"x": rng.random(n, dtype=xp.float32)}

       def step_state(self, xp, state, rng, p):
           state["x"] = state["x"] + float(p.get("dx", 0.01))
           return state

       def metrics(self, xp, state):
           return {"mean_x": float(am.to_host(state["x"].mean()))}

   res = Drift({"n": 1_000_000, "steps": 100, "seed": 0}).run()
   print(res.info)   # shows array_module / device
   print(res.model.tail())

Same code runs on **NumPy** if no GPU is present (automatic fallback).

**B. Many short runs (calibration) —** :class:`~ambr.gpu_ensemble.GPUEnsembleRunner`::

   from ambr.gpu_ensemble import GPUEnsembleRunner, BatchedWellMixedSIR
   runner = GPUEnsembleRunner(BatchedWellMixedSIR())
   traj = runner.run(n_agents=10_000, steps=50, params={"beta": betas, ...})

Rule of thumb
-------------

* **N ≲ 2k** — OOP is fine  
* **2k–500k** — vectorized  
* **dense interactions** — tensor  
* **N ≳ 1M array math, or B≫1 calibration** — GPU helpers above  

``am.recommend(n)`` encodes that heuristic.
