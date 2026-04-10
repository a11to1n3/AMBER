Quick Start Guide
=================

This guide will get you up and running with AMBER in just a few minutes.

The vectorized mindset
----------------------

AMBER stores the entire population as a Polars DataFrame under the hood, so
the fast path is to update **many agents in one expression** instead of
looping over individuals. The view API is built around three moves:

* ``self.agents`` — a view of the whole population
* ``self.agents.where(predicate)`` — a view of agents matching a condition
* ``self.agents.at[ids]`` — a view of specific agents by id

On any view, ``view.col`` returns a Polars Series sourced from
``self.agents_df``, and ``view.col = value`` queues a columnar update that
lands in the DataFrame on the next flush.

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
               wealth=self.nprandom.integers(1, 10, size=100),
           )

       def step(self):
           # Every agent with wealth > 0 gives $1 to a random other agent.
           donors = self.agents.where(self.agents.wealth > 0)
           donors.wealth -= 1

           # Pick a random recipient for each donor (with replacement),
           # then scatter the $1 credits — duplicate recipients correctly
           # receive multiple dollars via scatter_add.
           ids = self.agents.ids.to_numpy()
           recipients = self.nprandom.choice(ids, size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

           # Track aggregate state at the model level.
           self.record('total_wealth', int(self.agents.wealth.sum()))

   # Run the model
   model = WealthModel({'steps': 100, 'seed': 42, 'show_progress': False})
   results = model.run()

   # Inspect the results
   print("Final wealth distribution (first 10 agents):")
   print(results['agents'].select(['id', 'wealth']).head(10))

That's the whole idiom. No per-agent loops, no ``update_agent_data`` calls,
and no ``.item()`` ceremonies. The ``step()`` body is four Polars calls
regardless of whether you have 100 agents or 100 000.

Understanding the results
-------------------------

The model returns a dictionary with three keys:

* ``agents`` — a Polars DataFrame of agent state at the end of the run
* ``model`` — a Polars DataFrame of the model-level metrics you ``record``-ed
* ``info`` — a small dict with ``steps`` and ``run_time``

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

   class SpatialWealthModel(am.Model):
       def setup(self):
           self.grid = am.GridEnvironment(self, size=(20, 20))

           n = 200
           # Place agents randomly on the grid (sampled with replacement)
           xs = self.nprandom.integers(0, 20, size=n)
           ys = self.nprandom.integers(0, 20, size=n)
           self.add_agents(
               n,
               wealth=self.nprandom.integers(1, 10, size=n),
               x=xs,
               y=ys,
           )

       def step(self):
           # Same "donor gives $1" idiom as before.
           donors = self.agents.where(self.agents.wealth > 0)
           donors.wealth -= 1
           ids = self.agents.ids.to_numpy()
           recipients = self.nprandom.choice(ids, size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

   spatial_model = SpatialWealthModel({'steps': 50, 'seed': 42, 'show_progress': False})
   spatial_results = spatial_model.run()

Model-level analytics
---------------------

Aggregate metrics go through ``self.record``, which takes any scalar you
can compute from the current DataFrame:

.. code-block:: python

   import numpy as np

   class AnalyticalWealthModel(am.Model):
       def setup(self):
           self.add_agents(100, wealth=self.nprandom.integers(1, 10, size=100))

       def step(self):
           donors = self.agents.where(self.agents.wealth > 0)
           donors.wealth -= 1
           ids = self.agents.ids.to_numpy()
           recipients = self.nprandom.choice(ids, size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

           # Polars Series expose the usual aggregate methods.
           wealth = self.agents.wealth
           self.record('mean_wealth', float(wealth.mean()))
           self.record('wealth_std', float(wealth.std() or 0.0))
           self.record('gini', self._gini(wealth.to_numpy()))

       @staticmethod
       def _gini(values):
           if values.size == 0 or values.sum() == 0:
               return 0.0
           sorted_vals = np.sort(values)
           n = len(sorted_vals)
           cum = np.cumsum(sorted_vals)
           return (n + 1 - 2 * cum.sum() / cum[-1]) / n

When per-agent loops are OK
---------------------------

The view API isn't mandatory — you can still write OOP-style agents for
behaviours that genuinely don't vectorize (graph traversal, bespoke
scheduling). ``Agent.record()`` and ``Agent.update_data()`` queue writes
through the same batched flush path, so you won't pay a per-call
DataFrame clone:

.. code-block:: python

   class Walker(am.Agent):
       def step(self):
           # Per-agent behaviour — appropriate when the logic depends on
           # this specific agent's neighbourhood in a way that can't be
           # expressed as a single Polars expression.
           neighbours = self.get_neighbors()
           self.record('neighbour_count', neighbours.height)

In general: reach for ``self.agents.where(...).col = ...`` first. Fall
back to per-agent style only when the logic is inherently sequential or
needs side effects on external state.

Next Steps
----------

* :doc:`tutorial` — the longer-form walkthrough
* :doc:`api/sequences` — the full view API reference
* ``examples/`` — worked models you can copy from

Key concepts
------------

* **Views are always DataFrame-backed.** ``self.agents.wealth`` is a Polars
  Series read from ``self.agents_df``; it stays in sync with every other
  update automatically.
* **Bulk create with** ``add_agents``. Avoid ``Agent(self, i); add_agent(agent)``
  loops unless you actually need a Python class per agent.
* **Use** ``scatter_add`` **for resource flow.** ``view.col = ...`` handles
  deterministic updates; use ``scatter_add`` when ids may repeat and you
  want the deltas to sum.
* **Reproducibility** comes from ``self.nprandom`` and ``self.random``,
  both seeded from ``parameters['seed']``.
