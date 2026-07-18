Going faster (lanes)
====================

AMBER does **not** hide speed behind a silent ``gpu=True`` flag that rewrites
semantics. From **0.4.3** you place a run with Keras-style
``model.cpu(mode=...).run()`` / ``model.gpu().run()`` (mode defaults to
``vectorized``; ``run(mode=...)`` still overrides). Vectorized runs dispatch
``step_vectorized()`` and GPU keeps numeric columns device-resident. CPU OOP
runs dispatch ``step_oop()`` over tracked Agent objects. There are still four
**lanes** for *how* you write the step — helpers tell you which to pick.

Check this machine
------------------

.. code-block:: python

   import ambr as am
   am.print_status()
   print(am.recommend(100_000))          # single large run
   print(am.recommend(1_000, ensemble=True))  # many short runs / calibration

CPU acceleration with Numba (great on Mac)
------------------------------------------

Modern Macs have **no CUDA**, so AMBER cannot use the GPU lane there. For
CPU speed, install Numba (optional ``perf`` extra)::

   pip install 'ambr[perf]'
   # or: pip install numba

When ``HAS_NUMBA`` is true, AMBER JIT-compiles:

* ``scatter_add`` accumulations
* subset column writes (``view.col = ...`` / ``set`` on filtered views)

Your vectorized model code does **not** change — acceleration is automatic.
``am.print_status()`` reports whether Numba is active.

For custom array kernels, reuse the same optional stack::

   from ambr.performance import HAS_NUMBA, jit  # no-op decorator if missing

   @jit(nopython=True, cache=True)
   def my_kernel(x, y):
       ...

Lane 1 — OOP (AgentPy-shaped)
-----------------------------

Always available. Best for small N or sequential logic. See :doc:`from_agentpy`.

Lane 2 — Vectorized (default fast path)
---------------------------------------

**There is no flag.** Use ``step_vectorized()`` for the vectorized lane::

   xp = self.xp
   wealth = self.agents.array("wealth")
   donors = xp.nonzero(wealth > 0)[0]
   wealth[donors] -= 1

   recipients = self.rng.choice(self.agents.array("id"), size=int(donors.size))
   self.agents.at[recipients].scatter_add(wealth=1)

For object-oriented models, implement ``step_oop()`` and run with
``model.cpu(mode="oop")``. GPU runs use the vectorized lane only.

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

Requires an **NVIDIA GPU + CuPy** (not Apple Metal/MPS). Install CuPy matching
your CUDA, then either:

**A. Single large run — same view-API model (0.4.3, preferred)::

   results = MyVectorizedModel({"n": 1_000_000, "steps": 50, "seed": 0}).gpu().run()
   # CPU counterpart:
   # results = MyVectorizedModel(...).cpu(mode="vectorized").run()

**B. Array-kernel model —** :class:`~ambr.lanes.ArrayKernelModel`::

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

``ArrayKernelModel`` runs on **NumPy** if no GPU is present (automatic fallback).

**C. Many short runs (calibration) —** :class:`~ambr.gpu_ensemble.GPUEnsembleRunner`::

   from ambr.gpu_ensemble import GPUEnsembleRunner, BatchedWellMixedSIR
   runner = GPUEnsembleRunner(BatchedWellMixedSIR())
   traj = runner.run(n_agents=10_000, steps=50, params={"beta": betas, ...})

Rule of thumb
-------------

* **N ≲ 2k** — OOP is fine
* **2k–500k** — vectorized (``cpu(mode="vectorized")``)
* **dense interactions** — tensor
* **N ≳ 1M view-API or array math** — ``model.gpu().run()`` / ``ArrayKernelModel``
* **B≫1 calibration** — GPU ensemble helpers above

``am.recommend(n)`` encodes that heuristic.
