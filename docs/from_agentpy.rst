Coming from AgentPy
===================

**AgentPy** is the most approachable Python ABM API for many users: subclass
``Model`` / ``Agent``, put people in an ``AgentList``, call methods on the
list, read parameters from ``self.p``, and ``model.run()``. AMBER deliberately
keeps that shape for the **OOP lane**, then layers a **vectorized lane** for
scale.

Judgement (why AgentPy as the UX target)
----------------------------------------

==================  ===========================================================
Framework           Intuition
==================  ===========================================================
**AgentPy**         Smallest conceptual model: agents + list + step + run.
Mesa                Familiar, but schedulers / spaces add ceremony for a first model.
Agents.jl           Excellent, but Julia + a different package culture.
AMBER (vectorized)  Fastest Python path, but Polars-first docs can feel foreign
                    if you only know AgentPy.
==================  ===========================================================

AMBER's product stance: **feel like AgentPy when you write agent methods;
feel like a dataframe when you need 10⁵–10⁷ agents.**

Side-by-side
------------

AgentPy::

   import agentpy as ap

   class WealthAgent(ap.Agent):
       def setup(self):
           self.wealth = 1
       def transfer(self):
           if self.wealth > 0:
               other = self.model.agents.random()
               other.wealth += 1
               self.wealth -= 1

   class WealthModel(ap.Model):
       def setup(self):
           self.agents = ap.AgentList(self, self.p.agents, WealthAgent)
       def step(self):
           self.agents.transfer()
       def update(self):
           self.record('total', sum(self.agents.wealth))

   results = WealthModel({'agents': 50, 'steps': 20, 'seed': 1}).run()

AMBER (same idea)::

   import ambr as am

   class WealthAgent(am.Agent):
       def setup(self):
           self.wealth = 1
       def transfer(self):
           if self.wealth > 0:
               other_id = self.model.agents.random()
               self.model.agents.by_id(other_id).wealth += 1
               self.wealth -= 1

   class WealthModel(am.Model):
       def setup(self):
           self.agents = am.AgentList(self, self.p.agents, WealthAgent)
       def step(self):
           self.agents.transfer()   # method broadcast
       def update(self):
           # prefer record_model (record is a deprecated alias)
           self.record_model('total', int(self.agents.wealth.sum()))

   results = WealthModel(
       {'agents': 50, 'steps': 20, 'seed': 1, 'show_progress': False}
   ).run()
   print(results.model)    # attribute access (also results['model'])
   print(results.agents)

Results: AgentPy ``DataDict`` vs AMBER ``RunResults``
-----------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - AgentPy
     - AMBER
   * - ``results.variables.Model``
     - ``results.model`` (Polars DataFrame)
   * - ``results.agents`` / variables
     - ``results.agents`` end-of-run table; history via ``agent_reporters`` → long ``agent_vars``
   * - ``results.info``
     - ``results.info`` provenance dict (device, mode, UUID, versions, …)
   * - save/load
     - ``results.save(path)`` / ``RunResults.load``
   * - arrange / Sobol helpers
     - Polars + external SALib / your notebook

::

   results = model.run()
   print(results.keys_overview())
   results.save("out/run0")
   restored = am.RunResults.load("out/run0")

Notes on the mapping
--------------------

* ``self.p`` / ``self.t`` / ``setup`` / ``step`` / ``update`` / ``run`` — same roles.
* ``AgentList(model, n, AgentType)`` — supported; attribute and method broadcast work.
* ``agents.random()`` — returns an **id** (use ``agents.by_id(id)`` for the object).
  AgentPy returns an agent instance; AMBER keeps ids cheap for the columnar store.
* ``self.record(...)`` — still works (deprecated alias of ``record_model``; fine for
  migration; prefer ``record_model`` or ``model_reporters`` in new code).
* Results — AgentPy ``DataDict`` vs AMBER :class:`~ambr.results.RunResults`
  (dict + attributes). Keys: ``info``, ``agents``, ``model``.
* Progress printing is **off** by default (``show_progress=True`` to restore demos).

When to switch to the vectorized lane
-------------------------------------

Keep AgentPy-style OOP while N is small or logic is inherently sequential.
Move to the view API when loops dominate runtime::

   def step(self):
       donors = self.agents.where(self.agents.wealth > 0)
       donors.wealth -= 1
       recv = self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))
       self.agents.at[recv].scatter_add(wealth=1)

See :doc:`quickstart` for the full canonical-verb table, and
:doc:`api/contract` if you enable ``contract='check'``.

Vectorized lane on GPU (implement ``step_vectorized`` or legacy ``step``;
requires **NVIDIA + CuPy**, not Apple Metal/MPS)::

   model = MyVectorizedModel({"n": 10_000, "steps": 20, "seed": 0, "show_progress": False})
   if am.GPU_AVAILABLE:
       results = model.gpu().run()
   else:
       results = model.cpu(mode="vectorized").run()
   # OOP agents: MyOOPModel(...).cpu(mode="oop").run()  # not on GPU

See :doc:`going_faster` for placement, Numba, and ensemble calibration;
:doc:`reproducibility` for CPU≠GPU bit-identical policy.
