#!/usr/bin/env python
# coding: utf-8
# Flocking Behavior Simulation

This notebook presents an agent-based model that simulates the flocking behavior of animals. It demonstrates how to use the AMBER package for models with continuous space in two or three dimensions.

## About the Model

The boids model was invented by Craig Reynolds, who describes it as follows:

*"In 1986 I made a computer model of coordinated animal motion such as bird flocks and fish schools. It was based on three dimensional computational geometry of the sort normally used in computer animation or computer aided design. I called the generic simulated flocking creatures boids. The basic flocking model consists of three simple steering behaviors which describe how an individual boid maneuvers based on the positions and velocities its nearby flockmates:*

- **Separation**: steer to avoid crowding local flockmates
- **Alignment**: steer towards the average heading of local flockmates  
- **Cohesion**: steer to move toward the average position of local flockmates

*The model presented here is a simplified implementation of this algorithm, following the Boids Pseudocode written by Conrad Parker.*

# In[1]:


# Import required libraries
import amber as am
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from mpl_toolkits.mplot3d import Axes3D
import IPython
from IPython.display import HTML


# In[2]:


# Utility functions

def normalize(v):
    """Normalize a vector to length 1."""
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

def distance(pos1, pos2):
    """Calculate Euclidean distance between two positions."""
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

## Model Definition

The Boids model is based on two classes, one for the agents (Boid), and one for the overall model (BoidsModel). Each agent starts with a random position and velocity, which are implemented as numpy arrays.

The methods `update_velocity()` and `update_position()` are separated so that all agents can update their velocity before the actual movement takes place.

# In[3]:


class Boid(am.Agent):
    """An agent with a position and velocity in a continuous space,
    who follows Craig Reynolds three rules of flocking behavior; 
    plus a fourth rule to avoid the edges of the simulation space."""
    
    def __init__(self, model, agent_id):
        super().__init__(model, agent_id)
        self.velocity = None
        self.position = None
    
    def setup(self): 
        # Initialize random velocity
        ndim = self.model.p['ndim']
        self.velocity = normalize(
            self.model.nprandom.random(ndim) - 0.5)
        
        # Initialize random position
        size = self.model.p['size']
        self.position = self.model.nprandom.random(ndim) * size
    
    def get_neighbors(self, radius):
        """Get neighbors within a certain radius."""
        neighbors = []
        for other_id, other_agent in self.model.boid_agents.items():
            if other_id != self.id:
                dist = distance(self.position, other_agent.position)
                if dist <= radius:
                    neighbors.append(other_agent)
        return neighbors
    
    def update_velocity(self):
        """Update velocity based on flocking rules."""
        pos = self.position
        ndim = self.model.p['ndim']
        
        # Rule 1 - Cohesion
        outer_neighbors = self.get_neighbors(self.model.p['outer_radius'])
        if len(outer_neighbors) > 0:
            # Calculate center of mass
            positions = np.array([nb.position for nb in outer_neighbors])
            center = np.mean(positions, axis=0)
            v1 = (center - pos) * self.model.p['cohesion_strength']
        else:
            v1 = np.zeros(ndim)
        
        # Rule 2 - Separation
        v2 = np.zeros(ndim)
        inner_neighbors = self.get_neighbors(self.model.p['inner_radius'])
        for nb in inner_neighbors:
            v2 -= (nb.position - pos)
        v2 *= self.model.p['separation_strength']
        
        # Rule 3 - Alignment
        if len(outer_neighbors) > 0:
            # Calculate average velocity
            velocities = np.array([nb.velocity for nb in outer_neighbors])
            average_v = np.mean(velocities, axis=0)
            v3 = (average_v - self.velocity) * self.model.p['alignment_strength']
        else:
            v3 = np.zeros(ndim)
        
        # Rule 4 - Borders (avoid edges)
        v4 = np.zeros(ndim)
        d = self.model.p['border_distance']
        s = self.model.p['border_strength']
        size = self.model.p['size']
        
        for i in range(ndim):
            if pos[i] < d:
                v4[i] += s
            elif pos[i] > size - d:
                v4[i] -= s
        
        # Update velocity
        self.velocity += v1 + v2 + v3 + v4
        self.velocity = normalize(self.velocity)
    
    def update_position(self):
        """Move the agent based on its velocity."""
        self.position += self.velocity


# In[4]:


class BoidsModel(am.Model):
    """
    An agent-based model of animals' flocking behavior,
    based on Craig Reynolds' Boids Model [1]
    and Conrad Parkers' Boids Pseudocode [2].
    
    [1] http://www.red3d.com/cwr/boids/
    [2] http://www.vergenet.net/~conrad/boids/pseudocode.html
    """
    
    def setup(self):
        """Initialize the agents and model."""
        
        # Initialize DataFrame with correct columns for boids model
        ndim = self.p['ndim']
        columns = {
            'id': pl.Series([], dtype=pl.Int64),
            'step': pl.Series([], dtype=pl.Int64),
        }
        
        # Add position columns for each dimension
        for i in range(ndim):
            columns[f'pos_{i}'] = pl.Series([], dtype=pl.Float64)
            columns[f'vel_{i}'] = pl.Series([], dtype=pl.Float64)
            
        self.agents_df = pl.DataFrame(columns)
        
        # Create boid agents
        self.boid_agents = {}
        for i in range(self.p['population']):
            agent = Boid(self, i)
            agent.setup()
            self.boid_agents[i] = agent
        
        # Record initial state
        self._record_all_agents()
    
    def _record_all_agents(self):
        """Record current state of all agents."""
        agent_data = []
        ndim = self.p['ndim']
        
        for agent_id, agent in self.boid_agents.items():
            data = {
                'id': agent_id,
                'step': self.t,
            }
            # Add position and velocity data
            for i in range(ndim):
                data[f'pos_{i}'] = agent.position[i]
                data[f'vel_{i}'] = agent.velocity[i]
            
            agent_data.append(data)
        
        if agent_data:
            new_data = pl.DataFrame(agent_data)
            self.agents_df = pl.concat([self.agents_df, new_data])

    def step(self):   
        """Execute one step of the simulation."""
        
        # Update velocities first (synchronous)
        for agent in self.boid_agents.values():
            agent.update_velocity()
        
        # Then update positions
        for agent in self.boid_agents.values():
            agent.update_position()
        
        # Record current state
        self._record_all_agents()
        
    def get_positions(self):
        """Get current positions of all agents."""
        positions = []
        for agent in self.boid_agents.values():
            positions.append(agent.position)
        return np.array(positions)

## Visualization Functions

Next, we define visualization functions that can create animated plots of the flocking behavior.

# In[5]:


from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def create_boids_animation(parameters, steps=100):
    """Create an animated visualization of the boids flocking simulation."""
    ndim = parameters['ndim']
    print(f"🚀 Creating {ndim}D Boids Animation...")
    print(f"📊 Population: {parameters['population']} boids")
    print(f"📐 Space size: {parameters['size']}^{ndim}")
    
    # Initialize model
    model = BoidsModel(parameters)
    model.setup()
    model.update()
    
    # Store simulation states
    states = []
    
    # Run simulation and collect states
    for step in range(steps):
        # Record current state
        positions = model.get_positions()
        states.append({
            'positions': positions.copy(),
            'step': model.t
        })
        
        if step > 0:
            model.step()
            model.update()
        
        if step % 20 == 0:
            print(f"Step {step}/{steps}")
    
    # Create animation
    projection = '3d' if ndim == 3 else None
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection=projection)
    
    def animate(frame):
        ax.clear()
        state = states[frame]
        positions = state['positions']
        
        ax.set_title(f"Boids Flocking Model {ndim}D - Step {state['step']}")
        
        if ndim == 2:
            ax.scatter(positions[:, 0], positions[:, 1], s=8, c='darkblue', alpha=0.7)
            ax.set_xlim(0, parameters['size'])
            ax.set_ylim(0, parameters['size'])
            ax.set_aspect('equal')
        elif ndim == 3:
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=8, c='darkblue', alpha=0.7)
            ax.set_xlim(0, parameters['size'])
            ax.set_ylim(0, parameters['size'])
            ax.set_zlim(0, parameters['size'])
        
        # Remove axes for cleaner look
        ax.grid(True, alpha=0.3)
        
        return ax.collections
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(states), interval=100, blit=False, repeat=True)
    
    print(f"✅ Animation created with {len(states)} frames")
    plt.close()  # Close the figure to prevent static display
    
    return HTML(anim.to_jshtml())

## Simulation Parameters

Let's define the parameters for our flocking simulations. These control the behavior of the boids and the environment.

# In[6]:


# Parameters for 2D simulation
parameters_2d = {  
    'size': 50,
    'seed': 123,
    'steps': 200,
    'ndim': 2,
    'population': 200,
    'inner_radius': 3,        # Separation radius
    'outer_radius': 10,       # Cohesion/alignment radius  
    'border_distance': 10,    # Distance from edge to start avoiding
    'cohesion_strength': 0.005,    # Strength of cohesion force
    'separation_strength': 0.1,    # Strength of separation force
    'alignment_strength': 0.3,     # Strength of alignment force
    'border_strength': 0.5         # Strength of border avoidance
} 

print("2D Simulation Parameters:")
for key, value in parameters_2d.items():
    print(f"  {key}: {value}")

## 2D Simulation

Let's run the flocking simulation in 2D space. We'll see how the boids start scattered and gradually form cohesive flocks.

# In[7]:


# Create and display 2D animation
boids_2d_animation = create_boids_animation(parameters_2d, steps=100)
boids_2d_animation

## 3D Simulation

Now let's try the same simulation in 3D space with more boids for a more complex flocking behavior.

# In[8]:


# Parameters for 3D simulation  
parameters_3d = parameters_2d.copy()
parameters_3d.update({
    'ndim': 3,
    'population': 300,  # Fewer boids for 3D to keep animation smooth
    'size': 60,         # Larger space for 3D
})

print("3D Simulation Parameters:")
print(f"  Dimensions: {parameters_3d['ndim']}D")
print(f"  Population: {parameters_3d['population']} boids")
print(f"  Space size: {parameters_3d['size']}^3")

# Create and display 3D animation
boids_3d_animation = create_boids_animation(parameters_3d, steps=80)
boids_3d_animation


