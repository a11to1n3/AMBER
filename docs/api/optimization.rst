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

Intelligent parameter search using SMAC Bayesian optimization
(requires ``pip install 'ambr[advanced]'`` / SMAC + ConfigSpace).
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
