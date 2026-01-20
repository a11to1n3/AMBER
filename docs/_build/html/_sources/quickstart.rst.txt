Quick Start Guide
=================

This guide will get you up and running with AMBER in just a few minutes.

Your First Model
----------------

Let's create a simple wealth transfer model where agents randomly exchange money:

.. code-block:: python

   import amber as am
   import numpy as np

   class WealthModel(am.Model):
       def setup(self):
           # Create 100 agents with random initial wealth
           for i in range(100):
               agent = am.Agent(self, i)
               agent.wealth = self.nprandom.randint(1, 10)
               self.add_agent(agent)

       def step(self):
           # Each agent gives $1 to a random other agent
           for agent_id in range(len(self.agents)):
               if self.get_agent_data(agent_id)['wealth'].item() > 0:
                   # Find a random recipient
                   recipient_id = self.nprandom.randint(0, 100)
                   if recipient_id != agent_id:
                       # Transfer $1
                       self.update_agent_data(agent_id, {
                           'wealth': self.get_agent_data(agent_id)['wealth'].item() - 1
                       })
                       self.update_agent_data(recipient_id, {
                           'wealth': self.get_agent_data(recipient_id)['wealth'].item() + 1
                       })

   # Run the model
   model = WealthModel({'steps': 100, 'seed': 42})
   results = model.run()

   # Analyze results
   print(f"Final wealth distribution:")
   print(results['agents'].select(['id', 'wealth']).tail(10))

Understanding the Results
-------------------------

The model returns a dictionary with three main components:

* **agents**: DataFrame containing all agent data over time
* **model**: DataFrame containing model-level metrics over time  
* **info**: Dictionary with simulation metadata

.. code-block:: python

   # Examine the structure
   print("Agent data shape:", results['agents'].shape)
   print("Agent columns:", results['agents'].columns)
   
   print("Model data shape:", results['model'].shape)
   print("Model columns:", results['model'].columns)
   
   print("Simulation info:", results['info'])

Adding Environments
-------------------

Let's enhance our model with a grid environment where agents can only interact with neighbors:

.. code-block:: python

   class SpatialWealthModel(am.Model):
       def setup(self):
           # Create a 10x10 grid
           self.grid = am.GridEnvironment(self, size=(10, 10))
           
           # Place agents randomly on the grid
           for i in range(100):
               agent = am.Agent(self, i)
               agent.wealth = self.nprandom.randint(1, 10)
               
               # Find random empty position
               position = self.grid.random_position()
               agent.position = position
               
               self.add_agent(agent)
               self.update_agent_data(i, {
                   'x': position[0],
                   'y': position[1],
                   'wealth': agent.wealth
               })

       def step(self):
           for agent_id in range(100):
               agent_data = self.get_agent_data(agent_id)
               if agent_data['wealth'].item() > 0:
                   # Get current position
                   pos = (agent_data['x'].item(), agent_data['y'].item())
                   
                   # Find neighbors
                   neighbors = self.grid.get_neighbors(pos)
                   if neighbors:
                       # Choose random neighbor for transaction
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

   # Run spatial model
   spatial_model = SpatialWealthModel({'steps': 50, 'seed': 42})
   spatial_results = spatial_model.run()

Data Collection and Analysis
----------------------------

AMBER automatically collects agent data at each time step. You can also record model-level metrics:

.. code-block:: python

   class AnalyticalWealthModel(am.Model):
       def setup(self):
           # Same setup as before
           for i in range(100):
               agent = am.Agent(self, i)
               agent.wealth = self.nprandom.randint(1, 10)
               self.add_agent(agent)

       def step(self):
           # Wealth transfer logic (same as before)
           # ...
           
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
           cumsum = np.cumsum(sorted_wealth)
           return (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n

   # Run and analyze
   model = AnalyticalWealthModel({'steps': 100, 'seed': 42})
   results = model.run()

   # Plot results
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(12, 4))
   
   plt.subplot(1, 3, 1)
   plt.plot(results['model']['mean_wealth'])
   plt.title('Mean Wealth Over Time')
   plt.xlabel('Time Step')
   plt.ylabel('Mean Wealth')
   
   plt.subplot(1, 3, 2)
   plt.plot(results['model']['wealth_std'])
   plt.title('Wealth Standard Deviation')
   plt.xlabel('Time Step')
   plt.ylabel('Std Dev')
   
   plt.subplot(1, 3, 3)
   plt.plot(results['model']['gini_coefficient'])
   plt.title('Wealth Inequality (Gini)')
   plt.xlabel('Time Step')
   plt.ylabel('Gini Coefficient')
   
   plt.tight_layout()
   plt.show()

Next Steps
----------

Now that you've created your first AMBER models, you can:

1. **Explore Examples**: Check out the :doc:`examples/index` and the ``examples/`` directory for more complex models
2. **Learn the API**: Read the full :doc:`api/index` documentation
3. **Follow Tutorials**: Work through detailed :doc:`tutorial` guides
4. **Experiment**: Try different environments, agent behaviors, and analysis techniques

Key Concepts to Remember
------------------------

* **Models** define the overall simulation structure and rules
* **Agents** are individual entities with behaviors and properties
* **Environments** provide spatial or network contexts for agent interactions
* **Data Collection** happens automatically, with options for custom metrics
* **Reproducibility** is ensured through random seeds and deterministic execution 