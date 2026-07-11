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

Rules of thumb
--------------

1. **One write path per column per step** under simultaneous activation
   (or use ``scatter_add`` as the sanctioned reducer). See
   ``model.run(contract="check")``.
2. **Do not assign** ``population.data`` from user code — use
   ``agents.col = ...`` / ``agents.set`` / ``agents.commit``.
3. **AgentPy-shaped OOP** (``AgentList``, methods, ``by_id``) remains fully
   supported; only the *parallel batch aliases* and record helpers go away.

How warnings look
-----------------

.. code-block:: text

   DeprecationWarning: Model.update_agent_data(...) is deprecated since
   AMBER 0.4 and will be removed in 1.0; use agent.<col> = value or
   agents.at[id].set(**cols) instead.

Tracking
--------

Source of truth for emit sites: ``ambr._deprecation.warn_deprecated`` call
sites under ``src/ambr/``. Tests: ``tests/test_deprecations.py``.
