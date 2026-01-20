Experiment
==========

.. automodule:: amber.experiment
   :members:
   :undoc-members:
   :show-inheritance:

The experiment module provides tools for running multiple model configurations and parameter sweeps.

Experiment Class
----------------

.. autoclass:: amber.Experiment
   :members:
   :undoc-members:

Run multiple model configurations:

.. code-block:: python

   from amber import Experiment, Sample, IntRange
   
   # Define parameter variations
   params = Sample({
       'n_agents': IntRange(50, 200),
       'steps': 100,
       'seed': [1, 2, 3, 4, 5]  # Multiple seeds for robustness
   })
   
   # Create and run experiment
   experiment = Experiment(
       model_class=MyModel,
       parameters=params,
       iterations=20  # Number of parameter combinations
   )
   
   results = experiment.run()

Sample Class
------------

.. autoclass:: amber.Sample
   :members:
   :undoc-members:

Parameter sampling for experiments:

.. code-block:: python

   # Sample with ranges and fixed values
   sample = Sample({
       'population': IntRange(100, 1000),
       'mutation_rate': [0.01, 0.05, 0.1],
       'selection_pressure': 0.8,  # Fixed value
   })
   
   # Generate parameter combinations
   for params in sample.generate(n=50):
       model = MyModel(params)
       results = model.run()

IntRange Class
--------------

.. autoclass:: amber.IntRange
   :members:
   :undoc-members:

Integer range specification:

.. code-block:: python

   # Define integer ranges
   population_range = IntRange(50, 500)  # 50 to 500 inclusive
   
   # Use in parameter definitions
   params = {
       'n_agents': population_range,
       'max_steps': IntRange(100, 1000)
   } 