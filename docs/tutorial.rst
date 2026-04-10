Tutorial
========

This tutorial will guide you through building increasingly complex agent-based models with AMBER.

Part 1: Your First Model
-------------------------

Let's start with a simple wealth transfer model where agents randomly
exchange money. AMBER's view API expresses this in a handful of Polars
operations — no per-agent loop.

**Step 1: Define the Model**

.. code-block:: python

   import ambr as am

   class WealthModel(am.Model):
       def setup(self):
           # Bulk-create the population with columnar initial state.
           n = self.p['n_agents']
           self.add_agents(
               n,
               wealth=self.nprandom.integers(1, 10, size=n),
           )

       def step(self):
           # Every agent with wealth > 0 gives $1 to a random other agent.
           donors = self.agents.where(self.agents.wealth > 0)
           donors.wealth -= 1

           # Scatter the $1 credits. Using ``scatter_add`` (rather than a
           # plain ``view.wealth = ...``) is what makes the math right when
           # two donors happen to pick the same recipient.
           ids = self.agents.ids.to_numpy()
           recipients = self.nprandom.choice(ids, size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

**Step 2: Run the Model**

.. code-block:: python

   # Run the simulation
   model = WealthModel({
       'n_agents': 100,
       'steps': 100,
       'seed': 42
   })
   results = model.run()
   
   # Examine results
   print("Final wealth distribution:")
   final_wealth = results['agents'].filter(
       results['agents']['step'] == results['agents']['step'].max()
   )
   print(final_wealth.select(['id', 'wealth']).head(10))

Part 2: Adding Spatial Structure
---------------------------------

Now let's enhance our model with a grid environment where agents can only interact with neighbors.

**Step 1: Create Spatial Model**

.. code-block:: python

   class SpatialWealthModel(am.Model):
       def setup(self):
           # Create grid environment
           self.grid = am.GridEnvironment(self, size=(20, 20))

           n = self.p['n_agents']
           # Columnar creation: position + wealth together, no loop.
           self.add_agents(
               n,
               wealth=self.nprandom.integers(1, 10, size=n),
               x=self.nprandom.integers(0, 20, size=n),
               y=self.nprandom.integers(0, 20, size=n),
           )

       def step(self):
           # Same donor-pays-$1 idiom as Part 1 — the grid only affects
           # *who* the recipients are, not the vectorized shape of the
           # update. Here we keep the transfer global for simplicity; see
           # ``examples/segregation_model.py`` for a neighbourhood-scoped
           # variant built on ``GridEnvironment.get_neighbors``.
           donors = self.agents.where(self.agents.wealth > 0)
           donors.wealth -= 1
           ids = self.agents.ids.to_numpy()
           recipients = self.nprandom.choice(ids, size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

**Step 2: Visualize Results**

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Run spatial model
   spatial_model = SpatialWealthModel({
       'n_agents': 200,
       'steps': 50,
       'seed': 42
   })
   results = spatial_model.run()
   
   # Plot wealth distribution on grid
   final_data = results['agents'].filter(
       results['agents']['step'] == results['agents']['step'].max()
   )
   
   plt.figure(figsize=(10, 8))
   scatter = plt.scatter(
       final_data['x'], 
       final_data['y'], 
       c=final_data['wealth'], 
       cmap='viridis',
       s=50
   )
   plt.colorbar(scatter, label='Wealth')
   plt.title('Final Wealth Distribution on Grid')
   plt.xlabel('X Position')
   plt.ylabel('Y Position')
   plt.show()

Part 3: Data Collection and Analysis
-------------------------------------

Let's add comprehensive data collection to track model-level metrics.

**Step 1: Enhanced Model with Analytics**

.. code-block:: python

   import numpy as np

   class AnalyticalWealthModel(am.Model):
       def setup(self):
           n = self.p['n_agents']
           self.add_agents(n, wealth=self.nprandom.integers(1, 10, size=n))

       def step(self):
           donors = self.agents.where(self.agents.wealth > 0)
           donors.wealth -= 1
           ids = self.agents.ids.to_numpy()
           recipients = self.nprandom.choice(ids, size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

           # Polars Series aggregates are the idiomatic way to record metrics.
           wealth = self.agents.wealth
           self.record_model('total_wealth', int(wealth.sum()))
           self.record_model('mean_wealth', float(wealth.mean()))
           self.record_model('wealth_std', float(wealth.std() or 0.0))
           self.record_model('gini_coefficient', self.calculate_gini(wealth.to_numpy()))

       @staticmethod
       def calculate_gini(values):
           """Calculate Gini coefficient of wealth inequality."""
           if values.size == 0 or values.sum() == 0:
               return 0.0
           sorted_vals = np.sort(values)
           n = len(sorted_vals)
           cumsum = np.cumsum(sorted_vals)
           return (n + 1 - 2 * cumsum.sum() / cumsum[-1]) / n

**Step 2: Analyze Results**

.. code-block:: python

   # Run analytical model
   model = AnalyticalWealthModel({
       'n_agents': 100,
       'steps': 200,
       'seed': 42
   })
   results = model.run()
   
   # Create comprehensive analysis plots
   fig, axes = plt.subplots(2, 2, figsize=(15, 10))
   
   # Plot 1: Mean wealth over time
   axes[0,0].plot(results['model']['mean_wealth'])
   axes[0,0].set_title('Mean Wealth Over Time')
   axes[0,0].set_xlabel('Time Step')
   axes[0,0].set_ylabel('Mean Wealth')
   
   # Plot 2: Wealth inequality (Gini coefficient)
   axes[0,1].plot(results['model']['gini_coefficient'])
   axes[0,1].set_title('Wealth Inequality (Gini Coefficient)')
   axes[0,1].set_xlabel('Time Step')
   axes[0,1].set_ylabel('Gini Coefficient')
   
   # Plot 3: Wealth standard deviation
   axes[1,0].plot(results['model']['wealth_std'])
   axes[1,0].set_title('Wealth Standard Deviation')
   axes[1,0].set_xlabel('Time Step')
   axes[1,0].set_ylabel('Standard Deviation')
   
   # Plot 4: Final wealth distribution histogram
   final_wealth = results['agents'].filter(
       results['agents']['step'] == results['agents']['step'].max()
   )['wealth']
   axes[1,1].hist(final_wealth, bins=20, alpha=0.7)
   axes[1,1].set_title('Final Wealth Distribution')
   axes[1,1].set_xlabel('Wealth')
   axes[1,1].set_ylabel('Frequency')
   
   plt.tight_layout()
   plt.show()

Part 4: Parameter Optimization
-------------------------------

Let's use AMBER's optimization tools to find the best parameters for our model.

**Step 1: Define Optimization Target**

.. code-block:: python

   from ambr import ParameterSpace, IntRange, grid_search
   
   # Define parameter space to explore
   param_space = ParameterSpace({
       'n_agents': IntRange(50, 200),
       'steps': [50, 100, 150],
       'seed': IntRange(1, 10)
   })
   
   # Run grid search to minimize final Gini coefficient
   best_params, best_score = grid_search(
       model_class=AnalyticalWealthModel,
       param_space=param_space,
       metric='gini_coefficient',  # Minimize inequality
       minimize=True,
       n_runs=3  # Average over multiple runs
   )
   
   print(f"Best parameters: {best_params}")
   print(f"Best Gini coefficient: {best_score}")

**Step 2: Compare Optimization Methods**

.. code-block:: python

   from ambr import random_search, bayesian_optimization
   
   # Compare different optimization approaches
   methods = {
       'grid_search': grid_search,
       'random_search': random_search,
       'bayesian_optimization': bayesian_optimization
   }
   
   results = {}
   for name, method in methods.items():
       if name == 'random_search':
           best_params, best_score = method(
               model_class=AnalyticalWealthModel,
               param_space=param_space,
               metric='gini_coefficient',
               minimize=True,
               n_samples=20,
               n_runs=3
           )
       elif name == 'bayesian_optimization':
           best_params, best_score = method(
               model_class=AnalyticalWealthModel,
               param_space=param_space,
               metric='gini_coefficient',
               minimize=True,
               n_calls=20,
               n_runs=3
           )
       else:  # grid_search
           best_params, best_score = method(
               model_class=AnalyticalWealthModel,
               param_space=param_space,
               metric='gini_coefficient',
               minimize=True,
               n_runs=3
           )
       
       results[name] = {
           'params': best_params,
           'score': best_score
       }
   
   # Compare results
   for method, result in results.items():
       print(f"{method}: Gini = {result['score']:.4f}, Params = {result['params']}")

Part 5: Running Experiments
----------------------------

Finally, let's use the experiment framework to run systematic parameter sweeps.

**Step 1: Design Experiment**

.. code-block:: python

   from ambr import Experiment, Sample, IntRange
   
   # Define parameter variations
   experiment_params = Sample({
       'n_agents': IntRange(50, 300),
       'steps': 100,
       'seed': [1, 2, 3, 4, 5]  # Multiple seeds for robustness
   })
   
   # Create experiment
   experiment = Experiment(
       model_class=AnalyticalWealthModel,
       parameters=experiment_params,
       iterations=50  # Number of parameter combinations to try
   )
   
   # Run experiment
   experiment_results = experiment.run()

**Step 2: Analyze Experiment Results**

.. code-block:: python

   # Analyze relationship between population size and inequality
   import pandas as pd
   
   # Convert to pandas for easier analysis
   df = experiment_results.to_pandas()
   
   # Group by number of agents and calculate mean Gini coefficient
   gini_by_population = df.groupby('n_agents')['gini_coefficient'].mean()
   
   plt.figure(figsize=(10, 6))
   plt.plot(gini_by_population.index, gini_by_population.values, 'o-')
   plt.xlabel('Number of Agents')
   plt.ylabel('Mean Gini Coefficient')
   plt.title('Wealth Inequality vs Population Size')
   plt.grid(True, alpha=0.3)
   plt.show()
   
   # Statistical analysis
   correlation = df['n_agents'].corr(df['gini_coefficient'])
   print(f"Correlation between population size and inequality: {correlation:.3f}")

Next Steps
----------

You now have the foundation to build complex agent-based models with AMBER. Here are some directions to explore:

1. **Custom Agent Behaviors**: Create specialized agent classes with complex decision-making
2. **Network Models**: Use NetworkEnvironment for social network simulations
3. **Multi-Agent Interactions**: Implement group behaviors and collective decision-making
4. **Real-Time Visualization**: Add interactive plotting and animation
5. **Advanced Analytics**: Implement custom metrics and statistical analysis
6. **Performance Optimization**: Scale models to handle thousands of agents

For more examples, see the :doc:`examples/index` section and explore the ``examples/`` directory in the repository. 