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
               wealth=self.rng.integers(1, 10, size=n),
           )

       def step(self):
           # Every agent with wealth > 0 gives $1 to a random other agent.
           donors = self.agents.where(self.agents.wealth > 0)
           donors.wealth -= 1

           # Scatter the $1 credits. Using ``scatter_add`` (rather than a
           # plain ``view.wealth = ...``) is what makes the math right when
           # two donors happen to pick the same recipient.
           ids = self.agents.ids.to_numpy()
           recipients = self.rng.choice(ids, size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

**Step 2: Run the Model**

.. code-block:: python

   # Run the simulation (fluent placement; default mode is vectorized)
   model = WealthModel({
       'n_agents': 100,
       'steps': 100,
       'seed': 42
   })
   results = model.cpu(mode="vectorized").run()
   # Vectorized lane on GPU:  model.gpu().run()  (NVIDIA + CuPy)

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

   import ambr as am

   class SpatialWealthModel(am.Model):
       def setup(self):
           # Create grid environment
           self.grid = am.GridEnvironment(self, size=(20, 20))

           n = self.p['n_agents']
           # Columnar creation: position + wealth together, no loop.
           self.add_agents(
               n,
               wealth=self.rng.integers(1, 10, size=n),
               x=self.rng.integers(0, 20, size=n),
               y=self.rng.integers(0, 20, size=n),
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
           recipients = self.rng.choice(ids, size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

**Step 2: Run and inspect (plot optional)**

.. code-block:: python

   import ambr as am

   # Continues Part 2: SpatialWealthModel must be defined in the previous block
   # (or paste both blocks into one file).
   spatial_model = SpatialWealthModel({
       'n_agents': 200,
       'steps': 50,
       'seed': 42,
       'show_progress': False,
   })
   results = spatial_model.run()

   # End-of-run agent table (step column present when agent history is kept)
   agents = results['agents']
   if 'step' in agents.columns:
       final_data = agents.filter(agents['step'] == agents['step'].max())
   else:
       final_data = agents
   print(final_data.select(['id', 'x', 'y', 'wealth']).head())

   # Optional plot — requires a NumPy-compatible matplotlib:
   #   pip install -U 'matplotlib>=3.8'
   # import matplotlib.pyplot as plt
   # plt.figure(figsize=(10, 8))
   # scatter = plt.scatter(
   #     final_data['x'], final_data['y'], c=final_data['wealth'],
   #     cmap='viridis', s=50,
   # )
   # plt.colorbar(scatter, label='Wealth')
   # plt.title('Final Wealth Distribution on Grid')
   # plt.xlabel('X Position'); plt.ylabel('Y Position')
   # plt.show()

Part 3: Data Collection and Analysis
-------------------------------------

Let's add comprehensive data collection to track model-level metrics.

**Step 1: Enhanced Model with Analytics**

.. code-block:: python

   import ambr as am
   import numpy as np

   class AnalyticalWealthModel(am.Model):
       def setup(self):
           n = self.p['n_agents']
           self.add_agents(n, wealth=self.rng.integers(1, 10, size=n))

       def step(self):
           donors = self.agents.where(self.agents.wealth > 0)
           donors.wealth -= 1
           ids = self.agents.ids.to_numpy()
           recipients = self.rng.choice(ids, size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

       def update(self):
           # Post-step metrics (record_model in step() is also kept; update
           # wins on duplicate keys). Prefer model_reporters when declarative.
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

   import ambr as am
   import numpy as np
   from ambr import ParameterSpace, grid_search

   # Self-contained: same model as Part 3 (required metric recorded in update()).
   class AnalyticalWealthModel(am.Model):
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
           wealth = self.agents.wealth
           self.record_model('gini_coefficient', self.calculate_gini(wealth.to_numpy()))

       @staticmethod
       def calculate_gini(values):
           if values.size == 0 or values.sum() == 0:
               return 0.0
           sorted_vals = np.sort(values)
           n = len(sorted_vals)
           cumsum = np.cumsum(sorted_vals)
           return (n + 1 - 2 * cumsum.sum() / cumsum[-1]) / n

   parameter_space = ParameterSpace({
       'n_agents': [50, 100],
       'steps': [20, 40],
       'seed': 1,
       'show_progress': False,
   })

   # grid_search returns a list of dicts sorted best-first:
   #   [{'parameters': {...}, 'objective': float}, ...]
   results = grid_search(
       AnalyticalWealthModel,
       parameter_space,
       metric='gini_coefficient',  # last recorded value of this model metric
       iterations=1,               # average this many runs per combo
       minimize=True,
   )
   best = results[0]
   print(f"Best parameters: {best['parameters']}")
   print(f"Best Gini coefficient: {best['objective']}")

**Step 2: Compare Optimization Methods**

.. code-block:: python

   from ambr import random_search

   # Continues the previous block (AnalyticalWealthModel + parameter_space).
   # random_search uses the same return shape as grid_search (list of dicts).
   # bayesian_optimization requires SMAC (pip install 'ambr[advanced]').
   random_results = random_search(
       AnalyticalWealthModel,
       parameter_space,
       metric='gini_coefficient',
       n_samples=8,
       iterations=1,
       minimize=True,
       seed=0,
   )
   print(
       "random_search best:",
       random_results[0]['parameters'],
       random_results[0]['objective'],
   )

Part 5: Running Experiments
----------------------------

Finally, let's use the experiment framework to run systematic parameter sweeps.

**Step 1: Design Experiment**

.. code-block:: python

   import ambr as am
   import numpy as np
   from ambr import Experiment, Sample, IntRange

   # Sample(parameters, n) — n is required (number of combinations to draw).
   # Experiment(model_type, sample, iterations=...) matches the live API.
   class AnalyticalWealthModel(am.Model):
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
           wealth = self.agents.wealth
           vals = wealth.to_numpy()
           if vals.size == 0 or vals.sum() == 0:
               g = 0.0
           else:
               s = np.sort(vals.astype(float))
               n = len(s)
               # cumsum[-1] is total wealth (not s[-1], which is only the max)
               g = (n + 1 - 2 * np.cumsum(s).sum() / s.sum()) / n
           self.record_model('gini_coefficient', float(g))

   experiment_params = Sample(
       {
           'n_agents': IntRange(50, 150),  # inclusive start, exclusive end
           'steps': 20,
           'seed': [1, 2, 3],
           'show_progress': False,
       },
       n=6,  # draw 6 parameter combinations
   )

   experiment = Experiment(
       model_type=AnalyticalWealthModel,
       sample=experiment_params,
       iterations=1,  # repeats per combination
   )
   experiment_results = experiment.run()
   # run() returns a dict of Polars frames: info, parameters, agents, model
   print(experiment_results['info'])
   print(experiment_results['model'].head())

**Step 2: Analyze Experiment Results**

.. code-block:: python

   # Continues Step 1 — experiment_results is a dict of Polars frames, not pandas.
   import polars as pl

   model_df = experiment_results['model']
   final = model_df.group_by(['n_agents', 'seed']).agg(
       pl.col('gini_coefficient').last()
   )
   gini_by_population = (
       final.group_by('n_agents')
       .agg(pl.col('gini_coefficient').mean())
       .sort('n_agents')
   )
   print(gini_by_population)

   # Optional plot (matplotlib):
   # import matplotlib.pyplot as plt
   # plt.plot(gini_by_population['n_agents'], gini_by_population['gini_coefficient'], 'o-')
   # plt.xlabel('Number of Agents'); plt.ylabel('Mean Gini Coefficient')
   # plt.title('Wealth Inequality vs Population Size'); plt.grid(True, alpha=0.3); plt.show()


Next Steps
----------

You now have the foundation to build complex agent-based models with AMBER. Here are some directions to explore:

1. **Custom Agent Behaviors**: Create specialized agent classes with complex decision-making
2. **Network Models**: Use NetworkEnvironment for social network simulations
3. **Multi-Agent Interactions**: Implement group behaviors and collective decision-making
4. **Real-Time Visualization**: Add interactive plotting and animation
5. **Advanced Analytics**: Implement custom metrics and statistical analysis
6. **Performance Optimization**: Scale models with the view API,
   ``model.cpu(mode="vectorized")``, and ``model.gpu().run()``
   (see :doc:`going_faster`)

For more examples, see the :doc:`examples/index` section and explore the ``examples/`` directory in the repository.
