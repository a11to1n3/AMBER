Sequences
=========

.. module:: ambr.sequences

The ``sequences`` module defines AMBER's vectorized view API. The full
population lives at ``model.agents``; filtered and scatter views are
produced by ``where`` / indexing / ``at[...]``. All three view types share
the same attribute/assignment protocol — column reads return Polars Series
sourced from ``model.agents_df`` on CPU, and column writes go through
``Model._set_frame`` (contract-observed when enabled).

Under ``model.gpu().run()`` (0.4.3), numeric columns are **device-resident**
for the step body; the same ``where`` / column assign / ``scatter_add``
idiom applies. Host Polars is synced at step boundaries when the contract
or reporters need a CPU snapshot.

Canonical operations on a view:

* **Read** — ``view.col``, ``view.numpy('x', 'y')``, ``view.frame`` / ``view.ids``
* **Write** — ``view.col = values``, ``view.set(x=…, y=…)`` (atomic multi-column)
* **Reduce** — ``view.scatter_add(col=delta)`` (duplicate ids sum)
* **Tensor** — ``view.borrow(col)`` / ``view.commit(**cols)``

Prefer these over ``Model.update_agent_data`` / ``batch_update_agents`` and
``Population.set_agent_value`` / ``batch_update*`` (deprecated aliases).

AgentList
---------

.. autoclass:: ambr.AgentList
   :members:
   :undoc-members:
   :show-inheritance:

The full population view. Lives at ``model.agents`` and acts as both the
entry point for vectorized queries and a legacy list of ``Agent`` objects.

**Vectorized usage (preferred):**

.. code-block:: python

   # Filter by predicate and update columnar state
   rich = model.agents.where(model.agents.wealth > 100)
   rich.tag = 'rich'

   # Scatter-add deltas for random id draws (duplicates sum correctly)
   recipients = model.rng.choice(model.agents.ids.to_numpy(), size=50)
   model.agents.at[recipients].scatter_add(wealth=1)

**Legacy list usage (still supported):**

.. code-block:: python

   # Indexing, iteration, append/remove — works as before
   first = model.agents[0]
   for agent in model.agents:
       agent.step()
   model.agents.append(new_agent)

FilteredAgentList
-----------------

.. autoclass:: ambr.sequences.FilteredAgentList
   :members:
   :undoc-members:
   :show-inheritance:

Returned from ``model.agents.where(...)`` or ``model.agents[mask]``.
Operates on the subset of rows matching a predicate. Writing to a column
on this view touches only the filtered agents.

ScatterAgentList
----------------

.. autoclass:: ambr.sequences.ScatterAgentList
   :members:
   :undoc-members:
   :show-inheritance:

Returned from ``model.agents.at[ids]``. Unlike a filtered view, a scatter
view can contain duplicate ids — which is the whole point for "random
recipient" style updates. Use ``scatter_add`` to accumulate deltas when
ids repeat; plain assignment falls back to last-write-wins semantics.

Features
--------

* DataFrame-backed attribute reads and writes — no sync gotchas.
* Predicate filtering via ``where(...)`` with attribute predicates or raw
  Polars expressions.
* Scatter-add for flow-of-resources updates.
* Full back-compat with legacy list-style access (indexing, iteration,
  ``append``/``remove``, ``call``/``apply``).
