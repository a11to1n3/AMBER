Tutorial
========

This tutorial will guide you through building increasingly complex agent-based models with AMBER.

Part 1: Your First Model
-------------------------

Let's start with a simple wealth transfer model where agents randomly exchange money.

**Step 1: Define the Model**

.. code-block:: python

   import ambr as am
   import numpy as np
   
   class WealthModel(am.Model):
       def setup(self):
           # Create agents with random initial wealth
           for i in range(self.p['n_agents']):
               agent = am.Agent(self, i)
               agent.wealth = self.nprandom.randint(1, 10)
               self.add_agent(agent)
       
       def step(self):
           # Each agent gives $1 to a random other agent
           for agent_id in range(self.p['n_agents']):
               agent_data = self.get_agent_data(agent_id)
               if agent_data['wealth'].item() > 0:
                   # Choose random recipient
                   recipient_id = self.nprandom.randint(0, self.p['n_agents'])
                   if recipient_id != agent_id:
                       # Transfer $1
                       self.update_agent_data(agent_id, {
                           'wealth': agent_data['wealth'].item() - 1
                       })
                       recipient_data = self.get_agent_data(recipient_id)
                       self.update_agent_data(recipient_id, {
                           'wealth': recipient_data['wealth'].item() + 1
                       })

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
           
           # Create agents and place them on grid
           for i in range(self.p['n_agents']):
               agent = am.Agent(self, i)
               agent.wealth = self.nprandom.randint(1, 10)
               
               # Find random position
               position = self.grid.random_position()
               agent.position = position
               
               self.add_agent(agent)
               self.update_agent_data(i, {
                   'x': position[0],
                   'y': position[1],
                   'wealth': agent.wealth
               })
       
       def step(self):
           for agent_id in range(self.p['n_agents']):
               agent_data = self.get_agent_data(agent_id)
               if agent_data['wealth'].item() > 0:
                   # Get current position
                   pos = (agent_data['x'].item(), agent_data['y'].item())
                   
                   # Find neighbors
                   neighbors = self.grid.get_neighbors(pos)
                   if neighbors:
                       # Choose random neighbor
                       neighbor_pos = neighbors[self.nprandom.randint(0, len(neighbors))]
                       
                       # Find agent at that position
                       neighbor_data = self.agents_df.filter(
                           (self.agents_df['x'] == neighbor_pos[0]) & 
                           (self.agents_df['y'] == neighbor_pos[1])
                       )
                       
                       if not neighbor_data.is_empty():
                           neighbor_id = neighbor_data['id'].item()
                           
                           # Transfer wealth
                           self.update_agent_data(agent_id, {
                               'wealth': agent_data['wealth'].item() - 1
                           })
                           neighbor_wealth = self.get_agent_data(neighbor_id)['wealth'].item()
                           self.update_agent_data(neighbor_id, {
                               'wealth': neighbor_wealth + 1
                           })

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

   class AnalyticalWealthModel(am.Model):
       def setup(self):
           # Same setup as before
           for i in range(self.p['n_agents']):
               agent = am.Agent(self, i)
               agent.wealth = self.nprandom.randint(1, 10)
               self.add_agent(agent)
       
       def step(self):
           # Wealth transfer logic (same as before)
           for agent_id in range(self.p['n_agents']):
               agent_data = self.get_agent_data(agent_id)
               if agent_data['wealth'].item() > 0:
                   recipient_id = self.nprandom.randint(0, self.p['n_agents'])
                   if recipient_id != agent_id:
                       self.update_agent_data(agent_id, {
                           'wealth': agent_data['wealth'].item() - 1
                       })
                       recipient_data = self.get_agent_data(recipient_id)
                       self.update_agent_data(recipient_id, {
                           'wealth': recipient_data['wealth'].item() + 1
                       })
           
           # Record model-level statistics
           wealth_values = self.agents_df['wealth'].to_list()
           self.record_model('total_wealth', sum(wealth_values))
           self.record_model('mean_wealth', np.mean(wealth_values))
           self.record_model('wealth_std', np.std(wealth_values))
           self.record_model('gini_coefficient', self.calculate_gini(wealth_values))
       
       def calculate_gini(self, wealth_list):
           """Calculate Gini coefficient of wealth inequality."""
           sorted_wealth = sorted(wealth_list)
           n = len(sorted_wealth)
           if n == 0 or sum(sorted_wealth) == 0:
               return 0
           cumsum = np.cumsum(sorted_wealth)
           return (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n

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