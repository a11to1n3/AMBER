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

Exhaustive search over all parameter combinations:

.. code-block:: python

   best_params, best_score = am.grid_search(
       model_class=MyModel,
       param_space=space,
       metric='final_wealth',
       minimize=False,
       n_runs=5
   )

Random Search
~~~~~~~~~~~~~

.. autofunction:: ambr.random_search

Random sampling from parameter space:

.. code-block:: python

   best_params, best_score = am.random_search(
       model_class=MyModel,
       param_space=space,
       metric='convergence_time',
       minimize=True,
       n_samples=50,
       n_runs=3
   )

Bayesian Optimization
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ambr.bayesian_optimization

Intelligent parameter search using Bayesian optimization:

.. code-block:: python

   best_params, best_score = am.bayesian_optimization(
       model_class=MyModel,
       param_space=space,
       metric='efficiency',
       minimize=False,
       n_calls=30,
       n_runs=5
   ) 