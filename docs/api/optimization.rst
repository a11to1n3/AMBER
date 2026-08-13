Optimization
============

.. automodule:: ambr.optimization
   :members:
   :undoc-members:
   :show-inheritance:

The optimization module provides tools for parameter tuning and model optimization.

Parameter Space
---------------

.. autoclass:: ambr.ParameterSpace
   :members:
   :undoc-members:

Define parameter ranges for optimization:

.. code-block:: python

   from ambr import ParameterSpace, IntRange

   space = ParameterSpace({
       'n_agents': IntRange(50, 200),
       'learning_rate': [0.01, 0.1, 0.5],
       'strategy': ['random', 'greedy', 'smart']
   })

Optimization Functions
----------------------

Grid Search
~~~~~~~~~~~

.. autofunction:: ambr.grid_search

Exhaustive search over all parameter combinations. Signature::

   grid_search(model_class, parameter_space, metric, iterations=1, minimize=False)
       -> List[Dict]  # each item: {'parameters': dict, 'objective': float}

.. code-block:: python

   import ambr as am
   from ambr import ParameterSpace, IntRange

   class MyModel(am.Model):
       def setup(self):
           n = int(self.p.get('n_agents', 50))
           self.add_agents(n, wealth=self.rng.integers(1, 10, size=n))

       def step(self):
           donors = self.agents.where(self.agents.wealth > 0)
           donors.wealth -= 1
           ids = self.agents.ids.to_numpy()
           recipients = self.rng.choice(ids, size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

       def update(self):
           self.record_model('final_wealth', int(self.agents.wealth.sum()))

   space = ParameterSpace({
       'n_agents': [20, 40],
       'steps': 10,
       'seed': 0,
       'show_progress': False,
   })
   results = am.grid_search(
       MyModel,
       space,
       metric='final_wealth',
       iterations=1,
       minimize=False,
   )
   best = results[0]
   print(best['parameters'], best['objective'])

Random Search
~~~~~~~~~~~~~

.. autofunction:: ambr.random_search

Random sampling from parameter space (same return shape as ``grid_search``):

.. code-block:: python

   # Same MyModel + space as the grid_search example above (paste that block first).
   import ambr as am
   results = am.random_search(
       MyModel,
       space,
       metric='final_wealth',
       n_samples=10,
       iterations=1,
       minimize=False,
       seed=0,
   )
   best = results[0]
   print(best['parameters'], best['objective'])

Bayesian Optimization
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ambr.bayesian_optimization

Intelligent parameter search using SMAC's **RandomForest** surrogate
(Gaussian-process surrogates are not supported). Requires
``pip install 'ambr[advanced]'`` / SMAC + ConfigSpace.
Same return shape as ``grid_search``:

.. code-block:: python

   # Same MyModel + space as the grid_search example above (paste that block first).
   # Requires: pip install 'ambr[advanced]'
   import ambr as am
   results = am.bayesian_optimization(
       MyModel,
       space,
       metric='final_wealth',
       n_calls=10,
       iterations=1,
       minimize=False,
       random_state=0,
   )
   best = results[0]
   print(best['parameters'], best['objective'])

SMACOptimizer (advanced)
------------------------

.. autoclass:: ambr.SMACOptimizer
   :members:
   :undoc-members:

Requires ``pip install 'ambr[advanced]'`` (SMAC + ConfigSpace).

**Supported options (0.5.x):**

* ``strategy``: ``bayesian`` (default), ``random`` (RandomFacade),
  ``algorithm_configuration``
* ``use_random_search=True`` — same as ``strategy='random'``
* ``acquisition_function``: ``ei``, ``lcb``, ``pi``, ``eips``, ``ts``
* ``surrogate_model``: ``random_forest`` only
* ``fixed_params``: merged into every trial (e.g. ``n_agents``, ``steps``)
* ``deterministic``: ``None`` (default) is ``True`` when
  ``fixed_params['seed']`` is not ``None`` (same config is not
  re-evaluated); ``False`` otherwise, including ``{"seed": None}``
* multi-fidelity: ``use_multi_fidelity=True`` plus a fidelity parameter
  (``is_fidelity=True`` with numeric bounds → SMAC min/max budget)

**Not supported:** ``gaussian_process``, ``random_forest_with_instances``,
``acquisition_function='log_ei'`` (raise ``ValueError``).

``optimize()`` returns ``best_config``, ``best_cost`` / ``best_objective``,
``n_evaluations``, and ``history`` (Polars: search columns + ``cost``,
``objective``, ``time``, ``trial``).

Full scripts: ``examples/smac_calibration_simple.py``,
``examples/smac_calibration_basic.py``.

MultiObjectiveSMAC
------------------

.. autoclass:: ambr.MultiObjectiveSMAC
   :members:
   :undoc-members:

Independent **per-objective** :class:`ambr.SMACOptimizer` searches, then a
post-hoc non-dominated set. This is **not** ParEGO / EHVI.

* ``n_trials`` is the budget **for each objective** (total SMAC evaluations
  ``≈ n_trials × len(objectives)``).
* ``strategy`` is forwarded to each scalar optimizer: ``bayesian``
  (default), ``random``, ``algorithm_configuration``. ``strategy='pareto'``
  raises ``ValueError`` — the Pareto front is always assembled afterwards.
* ``fixed_params`` are merged into every trial **and** into incumbent
  re-scoring. Pass ``steps`` here; otherwise :meth:`Model.run` defaults to
  100 steps per evaluation. If ``seed`` is omitted, the constructor
  ``seed`` is used for every model evaluation so Pareto values match the
  searched front. A non-``None`` model seed also sets SMAC
  ``deterministic=True``; ``{"seed": None}`` stays stochastic.

Demo script: ``examples/smac_calibration_advanced.py`` (3 trials × 4
objectives, ``steps=8``).
