Deprecations (removed in 1.0)
=============================

AMBER **0.4** settled on one canonical verb per task. Legacy spellings still
work but emit :class:`DeprecationWarning` and are **scheduled for removal in
1.0**. Silence warnings in benchmark / reproducibility runs with::

   export AMBER_SUPPRESS_DEPRECATIONS=1

Canonical → legacy table
------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Use this (canonical)
     - Instead of (deprecated → 1.0)
   * - ``agent.<col> = value``
     - ``Agent.record(name, value)``, ``Agent.update_data({...})``
   * - ``view.<col> = value`` / ``view.set(**cols)``
     - ``AgentList.record``, ``AgentList.update_data``
   * - ``agents.at[id].set(...)`` or ``agent.<col> = ...``
     - ``Model.update_agent_data(id, {...})``
   * - ``agents.at[ids].set(...)`` / ``agents.where(...).set(...)``
     - ``Model.batch_update_agents(...)``
   * - view ``set`` / column assign
     - ``Population.set_agent_value``, ``batch_update``, ``batch_update_by_ids``
   * - view write path / ``Model._set_frame`` (internal)
     - ``population.data = ...`` (setter warns; invisible to contract)
   * - ``record_model(key, value)`` or ``model_reporters``
     - ``Model.record(key, value)``
   * - ``agents.ids`` / iterate ``model.agents``
     - ``AgentList.agent_ids``, ``AgentList.agents``
   * - ``agents.where(...)`` / ``agents.at[ids]`` / ``agents[mask]``
     - ``AgentList.select(...)``
   * - ``GridEnvironment(..., torus=...)`` / ``.torus``
     - ``wrap=`` / ``.wrap``
   * - Prefer not using ``Population.create_batch_context()``
     - batch context helper (use view ``set``)
   * - ``model.cpu(mode=...).run()`` / ``model.gpu().run()`` or ``run(device=...)``
     - ``run(backend=...)`` (alias of device placement)
   * - ``Experiment(model_type=..., sample=...)``
     - ``Experiment(model_class=..., parameters=...)``

Machine-readable inventory
--------------------------

Python source of truth (kept in sync by tests)::

   from ambr.deprecation_inventory import DEPRECATIONS_TO_REMOVE_IN_1_0

Also: :mod:`ambr.population.Population.create_batch_context` emits a plain
``DeprecationWarning`` (prefer view ``set`` / ``scatter_add``).

Rules of thumb
--------------

1. **One write path per column per step** under simultaneous activation
   (or use ``scatter_add`` as the sanctioned reducer). See
   ``model.run(contract="check")``.
2. **Do not assign** ``population.data`` from user code — use
   ``agents.col = ...`` / ``agents.set`` / ``agents.commit``.
3. **AgentPy-shaped OOP** (``AgentList``, methods, ``by_id``) remains fully
   supported; only the *parallel batch aliases* and record helpers go away.
4. **Place runs with** ``cpu()`` / ``gpu()`` (or ``run(device=...)``), not
   legacy ``backend=``.

How warnings look
-----------------

.. code-block:: text

   DeprecationWarning: Model.update_agent_data(...) is deprecated since
   AMBER 0.4 and will be removed in 1.0; use agent.<col> = value or
   agents.at[id].set(**cols) instead.

Tracking
--------

* Emit helper: ``ambr._deprecation.warn_deprecated``
* Inventory: ``ambr.deprecation_inventory`` + this page
* Behaviour tests: ``tests/test_deprecations.py``
* Sync test: ``tests/test_deprecation_inventory.py``
* Roadmap: :doc:`roadmap_1_0`
