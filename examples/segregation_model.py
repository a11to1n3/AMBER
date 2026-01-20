#!/usr/bin/env python
# coding: utf-8
# Segregation Model

This notebook presents an agent-based model of segregation dynamics. It demonstrates how to use the AMBER package to work with a spatial grid and create animations.

## About the model

The model is based on the NetLogo Segregation model from Uri Wilensky, who describes it as follows:

*This project models the behavior of two types of agents in a neighborhood. The orange agents and blue agents get along with one another. But each agent wants to make sure that it lives near some of "its own." That is, each orange agent wants to live near at least some orange agents, and each blue agent wants to live near at least some blue agents. The simulation shows how these individual preferences ripple through the neighborhood, leading to large-scale patterns.*

# In[56]:


# Model design
import amber as am
import polars as pl
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import IPython

## Model definition

To start, we define our agents who initiate with a random group and have two methods to check whether they are happy and to move to a new location if they are not.

# In[57]:


class Person(am.Agent):
    
    def __init__(self, model, agent_id):
        super().__init__(model, agent_id)
        self.x = 0
        self.y = 0
        self.group = 0
        self.share_similar = 0.0
        self.happy = False
        
    def update_happiness(self):
        """ Be happy if rate of similar neighbors is high enough. """
        neighbors = self.model.get_neighbors(self.x, self.y)
        similar = sum(1 for n_id in neighbors if self.model.agent_objects[n_id].group == self.group)
        ln = len(neighbors)
        self.share_similar = similar / ln if ln > 0 else 0
        self.happy = self.share_similar >= self.model.p['want_similar']
        
    def find_new_home(self):
        """ Move to random free spot and update free spots efficiently. """
        if len(self.model.empty_spots) == 0:
            return  # No empty spots available
        
        # Get available spots (excluding current position)
        available_spots = [spot for spot in self.model.empty_spots if spot != (self.x, self.y)]
        
        if not available_spots:
            return  # No different spots available
        
        # Choose new position
        new_spot = self.model.random.choice(available_spots)
        
        # Add current position back to empty spots
        self.model.empty_spots.append((self.x, self.y))
        self.model.grid[self.x][self.y] = -1
        
        # Move to new position
        self.x, self.y = new_spot
        self.model.grid[self.x][self.y] = self.id
        self.model.empty_spots.remove(new_spot)

Next, we define our model, which consists of our agents and a spatial grid environment. At every step, unhappy people move to a new location. After every step (update), agents update their happiness. If all agents are happy, the simulation is stopped.

# In[58]:


class SegregationModel(am.Model):
    
    def setup(self):      
        # Parameters
        s = self.p['size'] 
        n = self.n = int(self.p['density'] * (s ** 2))
         
        # Create grid (2D array, -1 = empty, agent_id = occupied)
        self.grid = [[-1 for _ in range(s)] for _ in range(s)]
        
        # Pre-calculate all empty spots for efficient agent placement
        self.empty_spots = [(x, y) for x in range(s) for y in range(s)]
        
        # Create persistent agent objects with vectorized placement
        self.agent_objects = {}
        for i in range(n):
            agent = Person(self, i)
            # Place agent directly without calling get_empty_spots repeatedly
            if self.empty_spots:
                spot = self.random.choice(self.empty_spots)
                agent.x, agent.y = spot
                agent.group = self.random.choice(range(self.p['n_groups']))
                self.grid[agent.x][agent.y] = i
                self.empty_spots.remove(spot)
            self.agent_objects[i] = agent
        
        # Create AgentList for compatibility
        self.agents = am.AgentList(self, 0, Person)
        self.agents.agent_ids = list(range(n))
        
        # Track agents needing DataFrame updates
        self._agents_to_update = set()
        
        # Track simulation state
        self.simulation_finished = False
        
        # Initialize DataFrame with minimal data (only record final states)
        self.agents_df = pl.DataFrame({
            'id': pl.Series([], dtype=pl.Int64),
            'step': pl.Series([], dtype=pl.Int64),
            'x': pl.Series([], dtype=pl.Int64),
            'y': pl.Series([], dtype=pl.Int64),
            'group': pl.Series([], dtype=pl.Int64),
            'share_similar': pl.Series([], dtype=pl.Float64),
            'happy': pl.Series([], dtype=pl.Boolean)
        })

    def get_empty_spots(self):
        """Get list of empty spots on the grid (efficient)."""
        # Use the maintained list for efficiency
        return self.empty_spots
    
    def get_neighbors(self, x, y):
        """Get neighbor agent IDs for a given position."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[0]):
                    agent_id = self.grid[nx][ny]
                    if agent_id != -1:
                        neighbors.append(agent_id)
        return neighbors
    
    def update(self):
        # Call parent update to increment time step
        super().update()
        
        # Update list of unhappy people
        for agent in self.agent_objects.values():
            agent.update_happiness()
        
        self.unhappy = [agent for agent in self.agent_objects.values() if not agent.happy]
        
        # Check if simulation should finish (all agents happy)
        if len(self.unhappy) == 0: 
            self.simulation_finished = True
            
    def step(self):
        # Skip step if simulation is finished
        if self.simulation_finished:
            return
            
        # Move unhappy people to new location (with safety limit)
        moves_made = 0
        max_moves = min(len(self.unhappy), 50)  # Limit moves per step
        
        for i, agent in enumerate(self.unhappy):
            if i >= max_moves:
                break
            if len(self.empty_spots) > 0:  # Only move if there are empty spots
                agent.find_new_home()
                moves_made += 1
    
    def reset_simulation(self):
        """Reset simulation state for fresh runs."""
        self.t = 0
        self.simulation_finished = False
        self._model_data = []
    
    def run(self, steps = None):
        """Override run method to stop early when all agents are happy."""
        # Initialize model state
        self.t = 0
        self.simulation_finished = False
        if not hasattr(self, '_model_data'):
            self._model_data = []
        
        # Call setup if needed
        if not hasattr(self, 'agent_objects'):
            self.setup()
        
        max_steps = steps if steps is not None else self.p.get('steps', 100)
        
        # Initialize unhappy agents list (first update)
        for agent in self.agent_objects.values():
            agent.update_happiness()
        self.unhappy = [agent for agent in self.agent_objects.values() if not agent.happy]
        
        # Check if already finished at start
        if len(self.unhappy) == 0:
            self.simulation_finished = True
        
        # Run simulation steps
        while self.t < max_steps and not self.simulation_finished:
            self.step()
            self.update()
            
        # Record final state and return results
        self.end()
        
        return {
            'info': {
                'model_type': self.__class__.__name__,
                'parameters': self.p,
                'steps': self.t,
                'completed_early': self.simulation_finished
            },
            'agents': self.agents_df,
            'model': pl.DataFrame(self._model_data) if self._model_data else pl.DataFrame({'t': []})
        }
    
    def _record_final_state(self):
        """Record final state of all agents to DataFrame (called only at end)."""
        agent_data = []
        for agent_id, agent in self.agent_objects.items():
            agent_data.append({
                'id': agent_id,
                'step': self.t,
                'x': agent.x,
                'y': agent.y,
                'group': agent.group,
                'share_similar': agent.share_similar,
                'happy': agent.happy
            })
        
        if agent_data:
            self.agents_df = pl.DataFrame(agent_data)
        else:
            self.agents_df = pl.DataFrame({
                'id': [], 'step': [], 'x': [], 'y': [], 
                'group': [], 'share_similar': [], 'happy': []
            })
        
    def get_segregation(self):
        # Calculate average percentage of similar neighbors
        total_similar = sum(agent.share_similar for agent in self.agent_objects.values())
        return round(total_similar / self.n, 2)
            
    def end(self): 
        # Record final agent states (only done once at the end)
        self._record_final_state()
        # Measure segregation at the end of the simulation
        self.record_model('segregation', self.get_segregation())

## Single-run animation

Uri Wilensky explains the dynamic of the segregation model as follows:

*Agents are randomly distributed throughout the neighborhood. But many agents are "unhappy" since they don't have enough same-color neighbors. The unhappy agents move to new locations in the vicinity. But in the new locations, they might tip the balance of the local population, prompting other agents to leave. If a few agents move into an area, the local blue agents might leave. But when the blue agents move to a new area, they might prompt orange agents to leave that area.*

*Over time, the number of unhappy agents decreases. But the neighborhood becomes more segregated, with clusters of orange agents and clusters of blue agents.*

*In the case where each agent wants at least 30% same-color neighbors, the agents end up with (on average) 70% same-color neighbors. So relatively small individual preferences can lead to significant overall segregation.*

To observe this effect in our model, we can create an animation of a single run. To do so, we first define a set of parameters.

# In[59]:


# Use smaller parameters for faster testing
parameters = {
    'want_similar': 0.3, # For agents to be happy
    'n_groups': 2, # Number of groups
    'density': 0.8, # Reduced density for faster performance
    'size': 20, # Smaller grid: 20x20 instead of 50x50
    'steps': 50  # Maximum number of steps
}

print(f"Grid size: {parameters['size']}x{parameters['size']}")
print(f"Number of agents: {int(parameters['density'] * parameters['size']**2)}")
print(f"Empty spots: {parameters['size']**2 - int(parameters['density'] * parameters['size']**2)}")

We can now create an animation plot and display it directly in Jupyter as follows.

# In[60]:


# First, let's test just the model setup to see if that works
print("Testing model setup...")
model = SegregationModel(parameters)

# Call setup manually since AMBER only calls it during run()
print("Calling setup()...")
model.setup()

print(f"✅ Model created successfully!")
print(f"   - Agents: {len(model.agent_objects)}")
print(f"   - Empty spots: {len(model.empty_spots)}")
print(f"   - Grid size: {len(model.grid)}x{len(model.grid[0])}")

# Check initial happiness without updating time
for agent in model.agent_objects.values():
    agent.update_happiness()
model.unhappy = [agent for agent in model.agent_objects.values() if not agent.happy]
print(f"   - Initially unhappy agents: {len(model.unhappy)}")

# Now let's try running just a few steps manually (without time increment)
print("\nTesting manual steps...")
for i in range(5):
    if len(model.unhappy) == 0:
        print(f"   - All agents happy after {i} manual steps!")
        break
    
    print(f"   - Manual step {i}: {len(model.unhappy)} unhappy agents")
    
    # Move some unhappy agents
    moves_made = 0
    max_moves = min(len(model.unhappy), 10)  # Limit for testing
    
    for j, agent in enumerate(model.unhappy):
        if j >= max_moves:
            break
        if len(model.empty_spots) > 0:
            agent.find_new_home()
            moves_made += 1
    
    print(f"     → Moved {moves_made} agents")
    
    # Update happiness without time increment
    for agent in model.agent_objects.values():
        agent.update_happiness()
    model.unhappy = [agent for agent in model.agent_objects.values() if not agent.happy]

print("\n🎯 Manual test complete! Creating fresh model for full simulation...")

# Create a completely fresh model for the full simulation
model = SegregationModel(parameters)
results = model.run()

print(f"\n📊 Results:")
print(f"Final segregation level: {model.get_segregation()}")
print(f"Total steps taken: {model.t}")
print(f"Total agents: {model.n}")
print(f"Unhappy agents at end: {len(model.unhappy)}")

## 🚀 Launch Interactive Simulation

Run the cell below to launch the modern interactive simulation interface. The interface includes:

### 🎛️ **Control Panel**
- **Agents**: Number of agents in the simulation (10-500)
- **Transfer Rate**: Fraction of wealth transferred per interaction (0.01-0.5)
- **FPS**: Animation speed in frames per second (1-30)
- **Max Steps**: Maximum simulation steps (50-1000)

### 🎮 **Control Buttons**
- **▶️ Start**: Begin a new simulation
- **⏸️ Pause/Resume**: Pause or resume the current simulation
- **🔄 Reset**: Reset to initial conditions

### 📊 **Real-time Visualizations**
- **Gini Coefficient Plot**: Shows wealth inequality evolution over time
- **Wealth Distribution**: Live histogram of current wealth distribution

### ✨ **Modern Features**
- **Continuous wealth**: Uses float values for more realistic transfers
- **Real-time updates**: Plots update live during simulation
- **Responsive design**: Modern, clean interface
- **Thread-safe**: Non-blocking simulation execution
## 🚀 Launch Interactive Simulation

Run the cell below to launch the modern interactive simulation interface. The interface includes:

### 🎛️ **Control Panel**
- **Agents**: Number of agents in the simulation (10-500)
- **Transfer Rate**: Fraction of wealth transferred per interaction (0.01-0.5)
- **FPS**: Animation speed in frames per second (1-30)
- **Max Steps**: Maximum simulation steps (50-1000)

### 🎮 **Control Buttons**
- **▶️ Start**: Begin a new simulation
- **⏸️ Pause/Resume**: Pause or resume the current simulation
- **🔄 Reset**: Reset to initial conditions

### 📊 **Real-time Visualizations**
- **Gini Coefficient Plot**: Shows wealth inequality evolution over time
- **Wealth Distribution**: Live histogram of current wealth distribution

### ✨ **Modern Features**
- **Continuous wealth**: Uses float values for more realistic transfers
- **Real-time updates**: Plots update live during simulation
- **Responsive design**: Modern, clean interface
- **Thread-safe**: Non-blocking simulation execution

# In[61]:


# Add visualization function
def plot_segregation_grid(model, ax):
    """Plot the current state of the segregation model."""
    size = model.p['size']
    grid_display = np.zeros((size, size))
    
    # Fill grid with group values (-1 for empty, 0 and 1 for groups)
    for x in range(size):
        for y in range(size):
            agent_id = model.grid[x][y]
            if agent_id == -1:
                grid_display[x][y] = -1  # Empty
            else:
                grid_display[x][y] = model.agent_objects[agent_id].group
    
    # Create color map: white for empty, different colors for groups
    import matplotlib.colors as mcolors
    colors = ['white', 'orange', 'lightblue']
    cmap = mcolors.ListedColormap(colors[:model.p['n_groups']+1])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    ax.clear()
    im = ax.imshow(grid_display, cmap=cmap, norm=norm)
    ax.set_title(f"Segregation Model - Step: {model.t}, Segregation: {model.get_segregation()}")
    ax.set_xticks([])
    ax.set_yticks([])
    
    return im

# Plot final state if the simulation completed
if 'model' in locals() and hasattr(model, 'agent_objects'):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_segregation_grid(model, ax)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No model available for visualization")

## 🚀 Launch Interactive Simulation

Run the cell below to launch the modern interactive simulation interface. The interface includes:

### 🎛️ **Control Panel**
- **Agents**: Number of agents in the simulation (10-500)
- **Transfer Rate**: Fraction of wealth transferred per interaction (0.01-0.5)
- **FPS**: Animation speed in frames per second (1-30)
- **Max Steps**: Maximum simulation steps (50-1000)

### 🎮 **Control Buttons**
- **▶️ Start**: Begin a new simulation
- **⏸️ Pause/Resume**: Pause or resume the current simulation
- **🔄 Reset**: Reset to initial conditions

### 📊 **Real-time Visualizations**
- **Gini Coefficient Plot**: Shows wealth inequality evolution over time
- **Wealth Distribution**: Live histogram of current wealth distribution

### ✨ **Modern Features**
- **Continuous wealth**: Uses float values for more realistic transfers
- **Real-time updates**: Plots update live during simulation
- **Responsive design**: Modern, clean interface
- **Thread-safe**: Non-blocking simulation execution
## 🚀 Launch Interactive Simulation

Run the cell below to launch the modern interactive simulation interface. The interface includes:

### 🎛️ **Control Panel**
- **Agents**: Number of agents in the simulation (10-500)
- **Transfer Rate**: Fraction of wealth transferred per interaction (0.01-0.5)
- **FPS**: Animation speed in frames per second (1-30)
- **Max Steps**: Maximum simulation steps (50-1000)

### 🎮 **Control Buttons**
- **▶️ Start**: Begin a new simulation
- **⏸️ Pause/Resume**: Pause or resume the current simulation
- **🔄 Reset**: Reset to initial conditions

### 📊 **Real-time Visualizations**
- **Gini Coefficient Plot**: Shows wealth inequality evolution over time
- **Wealth Distribution**: Live histogram of current wealth distribution

### ✨ **Modern Features**
- **Continuous wealth**: Uses float values for more realistic transfers
- **Real-time updates**: Plots update live during simulation
- **Responsive design**: Modern, clean interface
- **Thread-safe**: Non-blocking simulation execution
## Multi-run experiment

To explore how different individual preferences lead to different average levels of segregation, we can conduct a multi-run experiment. To do so, we first prepare a parameter sample that includes different values for peoples' preferences and the population density.
Now let's create parameter combinations to test different scenarios:

# In[62]:


# Create different parameter combinations manually since AMBER's parameter sampling is simpler
parameter_combinations = []
want_similar_values = [0, 0.125, 0.25, 0.375, 0.5, 0.625]
density_values = [0.5, 0.7, 0.95]

for want_sim in want_similar_values:
    for density in density_values:
        combo = parameters.copy()
        combo['want_similar'] = want_sim
        combo['density'] = density
        parameter_combinations.append(combo)

print(f"Total parameter combinations: {len(parameter_combinations)}")
print("Sample combinations:")
for i, combo in enumerate(parameter_combinations[:3]):
    print(f"  {i+1}: want_similar={combo['want_similar']}, density={combo['density']}")

We now run an experiment where we simulate each parameter combination in our sample over 5 iterations.

# In[63]:


# Run experiments manually since we have custom parameter combinations
all_results = []
experiment_data = []

print("Running experiments...")
for i, params in enumerate(parameter_combinations):
    print(f"Running combination {i+1}/{len(parameter_combinations)}: want_similar={params['want_similar']}, density={params['density']}")
    
    # Run multiple iterations for each parameter combination
    for iteration in range(3):  # 3 iterations per combination
        # Create a copy of parameters with iteration info
        run_params = params.copy()
        run_params['iteration'] = iteration
        run_params['seed'] = 42 + i * 10 + iteration  # Different seed for each run
        
        try:
            # Run the model
            model = SegregationModel(run_params)
            result = model.run()
            
            # Extract key results into a standardized format
            experiment_row = {
                'want_similar': float(params['want_similar']),
                'density': float(params['density']),
                'iteration': int(iteration),
                'final_step': int(model.t),
                'n_agents': int(model.n),
                'segregation': float(model.get_segregation()) if hasattr(model, 'get_segregation') else 0.0,
                'run_id': i * 3 + iteration
            }
            
            experiment_data.append(experiment_row)
            print(f"  ✅ Iteration {iteration}: segregation = {experiment_row['segregation']}")
            
        except Exception as e:
            print(f"  ❌ Error in iteration {iteration}: {e}")

# Create combined results DataFrame with consistent schema
if experiment_data:
    combined_results = pl.DataFrame(experiment_data)
    print(f"\nExperiment completed! Total runs: {len(experiment_data)}")
    print(f"Data shape: {combined_results.shape}")
    print(f"Columns: {list(combined_results.columns)}")
else:
    print("No results collected!")
    combined_results = pl.DataFrame()

Finally, we can arrange the results from our experiment and use the seaborn library to visualize the different segregation levels over our parameter ranges.

# In[64]:


sns.set_theme()

# Extract model results and create visualization
if len(combined_results) > 0 and 'segregation' in combined_results.columns:
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=combined_results, 
        x='want_similar', 
        y='segregation', 
        hue='density'
    )
    plt.title('Segregation Levels by Individual Preferences and Population Density')
    plt.xlabel('Want Similar (Individual Preference)')
    plt.ylabel('Average Segregation Level')
    plt.show()
    
    # Show summary statistics
    print("\nSummary Statistics:")
    summary = combined_results.group_by(['want_similar', 'density']).agg([
        pl.col('segregation').mean().alias('mean_segregation'),
        pl.col('segregation').std().alias('std_segregation')
    ]).sort(['want_similar', 'density'])
    print(summary)
else:
    print("No segregation data available for visualization.")
    if len(combined_results) > 0:
        print("Available columns:", combined_results.columns)
    else:
        print("No experiment results found.")

## Key Insights

The segregation model demonstrates several important phenomena:

1. **Emergent Segregation**: Even with relatively small individual preferences (30% want similar), the collective result is much higher segregation (often 70%+).

2. **Non-linear Effects**: Small changes in individual preferences can lead to dramatic changes in overall segregation patterns.

3. **Density Effects**: Population density affects how quickly segregation emerges and how stable the patterns become.

4. **Tipping Points**: There are critical thresholds where segregation patterns can rapidly change.

This model provides valuable insights into how individual-level preferences and behaviors can create unexpected macro-level social patterns.
