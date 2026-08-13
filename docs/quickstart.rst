Quick Start Guide
=================

This guide will get you up and running with AMBER in just a few minutes.

Two lanes, one model
--------------------

AMBER keeps an **AgentPy-shaped OOP lane** (agents, lists, method calls) and a
**vectorized lane** (columnar views) on the same ``Model``. Start with the lane
that matches how you think; scale with the other.

Coming from AgentPy? See :doc:`from_agentpy`.

Want speed / GPU without guessing? See :doc:`going_faster` and run::

   import ambr as am
   am.print_status()
   print(am.recommend(100_000))

Place a run with Keras-style chaining (mode defaults to ``vectorized``).
Prefer ``step_vectorized()`` for the columnar lane and ``step_oop()`` for
tracked Agent objects; legacy ``step()`` is the fallback. GPU runs use the
vectorized lane only and require **NVIDIA + CuPy** (not Apple Metal/MPS)::

   results = model.cpu(mode="vectorized").run()
   if am.GPU_AVAILABLE:
       results = model.gpu().run()
   results = model.cpu(mode="oop").run()  # Agent objects; not available on GPU

For a CPU boost on Mac (no CUDA), install Numba::

   pip install 'ambr[perf]'

Lane A — AgentPy-shaped (intuitive first model)
-----------------------------------------------

.. code-block:: python

   import ambr as am

   class WealthAgent(am.Agent):
       def setup(self):
           self.wealth = 1

       def transfer(self):
           if self.wealth > 0:
               other = self.model.agents.by_id(self.model.agents.random())
               other.wealth += 1
               self.wealth -= 1

   class WealthModel(am.Model):
       def setup(self):
           self.agents = am.AgentList(self, self.p.n, WealthAgent)

       def step_oop(self):
           # Explicit OOP lane — called only under cpu(mode="oop")
           self.agents.transfer()  # call the method on every agent

       def update(self):
           self.record_model('total', int(self.agents.wealth.sum()))

   # mode="oop" so provenance reports execution_lane cpu/oop (not vectorized)
   results = WealthModel({'n': 50, 'steps': 20, 'seed': 1}).cpu(mode="oop").run()
   print(results.model)     # or results['model']
   print(results.agents.head())
   print(results.info.get("mode"), results.info.get("execution_lane"))

Lane B — vectorized (fast path at scale)
----------------------------------------

AMBER stores the population as a Polars DataFrame. The view API is three moves:

* ``self.agents`` — whole population
* ``self.agents.where(predicate)`` — filter
* ``self.agents.at[ids]`` — select by id

``view.col`` is a Series; ``view.col = value`` writes back through the model.

Canonical verbs (learn these)
-----------------------------

Prefer this small surface; everything else is a legacy alias or an internal
helper scheduled for removal in 1.0.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Role
     - Canonical
     - Avoid / deprecated
   * - Device / mode
     - ``model.cpu(mode=...).run()`` / ``model.gpu().run()``
     - ``run(backend=...)`` (use ``device=`` or fluent placement)
   * - Select
     - ``agents`` / ``.where`` / ``.at``
     - ``agents.select`` (use ``where`` / ``at`` / ``[mask]``)
   * - Write
     - ``view.col = …`` / ``view.set(…)``
     - ``update_agent_data``, ``batch_update_agents``, ``Population.batch_*``
   * - Accumulate
     - ``view.scatter_add(…)``
     - double-assigning the same cell/column in one step
   * - Array kernels
     - ``agents.borrow`` / ``agents.commit``
     - maintaining a parallel NumPy buffer by hand
   * - Create
     - ``add_agents(n, **cols)``
     - per-agent ``Agent`` + ``add_agent`` loops (unless you need OOP)
   * - Metrics
     - ``record_model`` / ``model_reporters``
     - ``model.record`` (alias of ``record_model``)

**Efficiency note.** Batch performance comes from using these verbs (one
columnar write per step), not from extra public ``batch_*`` methods. Subset
``set`` / ``scatter_add`` / tensor ``commit`` are the hot paths that stay
optimized; aliases only forward to the same seams.

Your first model
----------------

A wealth transfer model where agents randomly exchange money, written the
vectorized way:

.. code-block:: python

   import ambr as am

   class WealthModel(am.Model):
       def setup(self):
           # Bulk-create 100 agents with random initial wealth — no loop.
           self.add_agents(
               100,
               wealth=self.rng.integers(1, 10, size=100),
           )

       def step_vectorized(self):
           # Every agent with wealth > 0 gives $1 to a random other agent.
           donors = self.agents.where(self.agents.wealth > 0)
           donors.wealth -= 1

           # Pick a random recipient for each donor (with replacement),
           # then scatter the $1 credits — duplicate recipients correctly
           # receive multiple dollars via scatter_add.
           ids = self.agents.ids.to_numpy()
           recipients = self.rng.choice(ids, size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

       def update(self):
           # Prefer update() / model_reporters for post-step aggregates.
           # record_model inside step() is also retained (later stages win on
           # duplicate keys: step → model_reporters → update).
           self.record_model('total_wealth', int(self.agents.wealth.sum()))

   # Prefer GPU when available (NVIDIA + CuPy); else CPU vectorized.
   model = WealthModel({'steps': 100, 'seed': 42, 'show_progress': False})
   if am.GPU_AVAILABLE:
       results = model.gpu().run()
   else:
       results = model.cpu(mode="vectorized").run()
   print(results.info)

   # Inspect the results
   print("Final wealth distribution (first 10 agents):")
   print(results['agents'].select(['id', 'wealth']).head(10))

That's the whole idiom. No per-agent loops, no ``update_agent_data`` calls,
and no ``.item()`` ceremonies. ``step_vectorized()`` is a handful of view-API
calls regardless of whether you have 100 agents or 100 000 — and the
vectorized lane runs under ``.gpu()`` with device-resident columns when
NVIDIA + CuPy are available.

Understanding the results
-------------------------

The model returns a dictionary with three keys:

* ``agents`` — a Polars DataFrame of agent state at the end of the run
* ``model`` — a Polars DataFrame of the model-level metrics you reported
* ``info`` — a Python dict of run provenance (``steps``, ``run_time``,
  ``run_uuid``, ``model_class``, ``parameters``, ``seed``,
  ``execution_lane``, library versions, …). Not a DataFrame.

.. code-block:: python

   print("Agent data shape:", results['agents'].shape)
   print("Agent columns:", results['agents'].columns)
   print("Simulation info:", results['info'])

Filtering and conditional updates
---------------------------------

``where`` accepts either an attribute-predicate (``self.agents.wealth > 0``)
or a raw Polars expression (``pl.col('wealth') > 0``). Both lower to the
same filter:

.. code-block:: python

   import polars as pl

   # Attribute-predicate form — the most common
   wealthy = self.agents.where(self.agents.wealth > 100)

   # Polars expression form — useful when chaining multiple conditions
   rich_adults = self.agents.where((pl.col('wealth') > 100) & (pl.col('age') >= 18))

   # Mark the wealthy with a tag column — created on first assignment
   wealthy.tag = 'rich'

Adding spatial structure
------------------------

Let's enhance the model with a 20×20 grid:

.. code-block:: python

   import ambr as am

   class SpatialWealthModel(am.Model):
       def setup(self):
           self.grid = am.GridEnvironment(self, size=(20, 20))

           n = 200
           # Place agents randomly on the grid (sampled with replacement)
           xs = self.rng.integers(0, 20, size=n)
           ys = self.rng.integers(0, 20, size=n)
           self.add_agents(
               n,
               wealth=self.rng.integers(1, 10, size=n),
               x=xs,
               y=ys,
           )

       def step(self):
           # Same "donor gives $1" idiom as before.
           donors = self.agents.where(self.agents.wealth > 0)
           donors.wealth -= 1
           ids = self.agents.ids.to_numpy()
           recipients = self.rng.choice(ids, size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

   spatial_model = SpatialWealthModel({'steps': 50, 'seed': 42, 'show_progress': False})
   spatial_results = spatial_model.run()

Model-level analytics
---------------------

Aggregate metrics go through ``self.record_model`` (often inside ``update()``)
or a class-level ``model_reporters`` dict. Values written inside ``step()`` /
``step_vectorized()`` are **retained**; on duplicate keys the later stage
wins: ``step()`` → declarative ``model_reporters`` → ``update()``.

.. code-block:: python

   import ambr as am
   import numpy as np

   class AnalyticalWealthModel(am.Model):
       def setup(self):
           self.add_agents(100, wealth=self.rng.integers(1, 10, size=100))

       def step(self):
           donors = self.agents.where(self.agents.wealth > 0)
           donors.wealth -= 1
           ids = self.agents.ids.to_numpy()
           recipients = self.rng.choice(ids, size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

       def update(self):
           # Post-step aggregates (update wins over reporters / step on the
           # same key). Prefer model_reporters for simple declarative metrics.
           wealth = self.agents.wealth
           self.record_model('mean_wealth', float(wealth.mean()))
           self.record_model('wealth_std', float(wealth.std() or 0.0))
           self.record_model('gini', self._gini(wealth.to_numpy()))

       @staticmethod
       def _gini(values):
           if values.size == 0 or values.sum() == 0:
               return 0.0
           sorted_vals = np.sort(values)
           n = len(sorted_vals)
           cum = np.cumsum(sorted_vals)
           return (n + 1 - 2 * cum.sum() / cum[-1]) / n

   results = AnalyticalWealthModel(
       {'steps': 50, 'seed': 42, 'show_progress': False}
   ).run()
   print(results.model.tail())

When per-agent loops are OK
---------------------------

The view API isn't mandatory — you can still write OOP-style agents for
behaviours that genuinely don't vectorize (graph traversal, bespoke
scheduling). Assigning ``agent.col = value`` writes through the same batched
flush path, so you won't pay a per-call DataFrame clone:

.. code-block:: python

   import ambr as am

   class Walker(am.Agent):
       def step(self):
           # Per-agent behaviour — appropriate when the logic depends on
           # this specific agent's neighbourhood in a way that can't be
           # expressed as a single Polars expression.
           neighbours = self.get_neighbors()
           self.neighbour_count = neighbours.height

In general: reach for ``self.agents.where(...).col = ...`` first. Fall
back to per-agent style only when the logic is inherently sequential or
needs side effects on external state.

Next Steps
----------

* :doc:`tutorial` — the longer-form walkthrough
* :doc:`going_faster` — Numba, ``cpu()`` / ``gpu()``, array kernels, ensemble
* :doc:`api/sequences` — the full view API reference
* ``examples/`` — worked models you can copy from

Key concepts
------------

* **Views are always DataFrame-backed** on CPU. ``self.agents.wealth`` is a
  Polars Series read from ``self.agents_df``; under ``model.gpu().run()``
  numeric columns are device-resident for the step body.
* **Bulk create with** ``add_agents``. Avoid ``Agent(self, i); add_agent(agent)``
  loops unless you actually need a Python class per agent.
* **Use** ``scatter_add`` **for resource flow.** ``view.col = ...`` handles
  deterministic updates; use ``scatter_add`` when ids may repeat and you
  want the deltas to sum.
* **Place with** ``cpu()`` / ``gpu()`` **before** ``run()`` (or pass
  ``device=`` / ``mode=`` to ``run``). Prefer fluent placement over legacy
  ``backend=``.
* **Reproducibility** comes from ``self.rng`` and ``self.random``,
  both seeded from ``parameters['seed']``.
