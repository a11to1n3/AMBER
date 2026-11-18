Going faster (lanes)
====================

Use the same model with ``model.run()``, ``model.vectorize().run()``, or
``model.gpu().run()`` on NVIDIA CUDA. Define ordinary ``setup()`` and
``step()`` methods and call ``self.agents.step()`` or
``self.activate_agents()`` to run the agents' rules.
The default CPU run already uses automatic batching. ``vectorize()``
selects the vectorized mode explicitly and preserves the selected device.
The default mode uses ``step_vectorized()`` when defined, then ``step()``,
then a standalone ``step_oop()``. When you set ``cpu(mode="oop")``, AMBER
uses ``step_oop()`` first. Both devices can batch supported Python agent rules.

Check this machine
------------------

.. code-block:: python

   import ambr as am
   am.print_status()
   print(am.recommend(100_000))          # single large run
   print(am.recommend(1_000, ensemble=True))  # many short runs / calibration

OOP activation (optional)
-------------------------

For tracked Python agents, use thin Mesa-inspired activation helpers
(:mod:`ambr.scheduling`) — they do **not** change vectorized semantics::

   def step_oop(self):
       self.activate_agents(mode="random")   # sequential | random | simultaneous

Vectorized models should keep order inside ``step_vectorized`` (or use
:func:`~ambr.scheduling.shuffled_ids`). Activation helpers are **not** a
schedule-proof; see :doc:`api/contract`.

CPU acceleration with Numba (great on Mac)
------------------------------------------

Modern Macs have **no CUDA** and AMBER does **not** use Apple Metal/MPS.
The GPU lane requires **NVIDIA + CuPy** (``pip install 'ambr[gpu]'`` or a
CUDA-matched wheel). Re-check GPU claim samples on a CUDA host with::

   python scripts/run_gpu_claims.py --quick

For CPU speed on Mac / no-CUDA machines, install Numba (optional ``perf`` extra)::

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

On CPU and GPU, AMBER automatically batches eligible own-field
integer additions and subtractions in sequential activation and method
broadcasts over a homogeneous tracked population. A rule such as
``if self.active: self.age += 1`` batches only writes to the selected agents.
The condition can check a Boolean or integer field, or compare an integer
field with a constant. The original Python loop
handles other methods, including transfers between agents, shared random draws,
and methods with custom attribute access. Monitoring and debuggers retain the
Python calls.
Inspect ``results.info["auto_batch"]`` for the number of batches and agent
calls replaced, or use ``run(auto_batch=False)`` to compare with the loop.
Use the same model with ``gpu().run()`` on NVIDIA with CuPy. AMBER keeps
batched columns on the device across steps until a host read, an agent-state
snapshot, or teardown needs the data.
See :doc:`from_agentpy` for the OOP API.

Lane 2 — Vectorized column operations
---------------------------------------

You can write column operations directly in ``step()`` or supply a dedicated
``step_vectorized()`` hook. The view API includes
``where``, column assignment, and ``scatter_add``. Do **not** mutate
``agents.array(...)`` in place — that returns a read-only snapshot on the CPU
Polars path::

   donors = self.agents.where(self.agents.wealth > 0)
   donors.wealth -= 1
   recipients = self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))
   self.agents.at[recipients].scatter_add(wealth=1)

For models with both hooks, use ``model.cpu(mode="oop")`` to select
``step_oop()`` explicitly.

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

Requires an **NVIDIA GPU + CuPy** (not Apple Metal/MPS). Install the GPU extra
(or a CUDA-matched CuPy wheel)::

   pip install 'ambr[gpu]'
   # if needed: pip install cupy-cuda12x   # match your toolkit

Then either:

**A. Single large run — vectorized view-API model (preferred)**::

   import ambr as am
   model = MyVectorizedModel(
       {"n": 1_000_000, "steps": 50, "seed": 0, "show_progress": False}
   )
   if am.GPU_AVAILABLE:
       results = model.gpu().run()
   else:
       results = model.cpu(mode="vectorized").run()
   # Optional private GPU loop (only if the model defines one; not monitored):
   # model.approve_fast_path("my-label").gpu().run(contract="off")
   #
   # ``approve_fast_path(evidence)`` is **caller-attested**: AMBER records the
   # label and requires it (plus ``contract="off"``) before private loops run.
   # It does **not** verify equivalence to a reference trajectory.
   # Re-check GPU claims on NVIDIA hardware:
   #   python scripts/run_gpu_claims.py --quick

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
   print(res.model.tail(3).to_dicts())

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
