#!/usr/bin/env python
# coding: utf-8
"""
🦠 Interactive AMBER Virus Spread Simulation

This notebook demonstrates a **modern interactive epidemiological model** using AMBER with cutting-edge visualization:

- **🧬 SIR Model**: Susceptible → Infected → Recovered dynamics
- **🌐 Spatial Spread**: Agents move and interact in 2D space
- **📊 Real-time Analytics**: Live tracking of infection curves
- **🎛️ Interactive Controls**: Adjust transmission rates, recovery times, and movement
- **🎨 Dynamic Visualization**: Color-coded agents and epidemic curves
- **⚡ High Performance**: Designed for large populations

Experience realistic epidemic modeling with modern interactive tools!
"""

# In[6]:


# Required imports for virus spread simulation
import amber as am
import polars as pl
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
import threading
import time
import random
from typing import Optional, Callable
from enum import Enum

print("✅ All packages loaded successfully!")
print("🦠 Ready for virus spread simulation!")


# In[7]:


class HealthStatus(Enum):
    """Health status enumeration for SIR model."""
    SUSCEPTIBLE = "S"
    INFECTED = "I" 
    RECOVERED = "R"

class VirusAgent(am.Agent):
    """Agent that can be infected with a virus and spread it to others."""
    
    def __init__(self, model, agent_id):
        super().__init__(model, agent_id)
        self.status = HealthStatus.SUSCEPTIBLE
        self.infection_time = 0
        self.x = float(random.uniform(0, model.p.get('world_size', 100)))
        self.y = float(random.uniform(0, model.p.get('world_size', 100)))
        
    def setup(self):
        """Initialize agent."""
        # Patient zero - infect a few agents initially
        if self.id < self.model.p.get('initial_infected', 5):
            self.status = HealthStatus.INFECTED
            self.infection_time = 0
    
    def move(self):
        """Random movement within world boundaries."""
        if self.status == HealthStatus.INFECTED:
            # Infected agents move less (they're sick)
            movement_speed = self.model.p.get('movement_speed', 2.0) * 0.5
        else:
            movement_speed = self.model.p.get('movement_speed', 2.0)
            
        # Random walk with boundaries
        world_size = float(self.model.p.get('world_size', 100))
        dx = random.uniform(-movement_speed, movement_speed)
        dy = random.uniform(-movement_speed, movement_speed)
        
        self.x = float(max(0, min(world_size, self.x + dx)))
        self.y = float(max(0, min(world_size, self.y + dy)))
    
    def interact(self):
        """Check for infections with nearby agents."""
        if self.status != HealthStatus.INFECTED:
            return
            
        infection_radius = self.model.p.get('infection_radius', 5.0)
        transmission_rate = self.model.p.get('transmission_rate', 0.1)
        
        # Find nearby susceptible agents
        for other_id, other_agent in self.model.agent_objects.items():
            if other_id == self.id or other_agent.status != HealthStatus.SUSCEPTIBLE:
                continue
                
            # Calculate distance
            distance = np.sqrt((self.x - other_agent.x)**2 + (self.y - other_agent.y)**2)
            
            if distance <= infection_radius:
                # Probability of transmission
                if random.random() < transmission_rate:
                    other_agent.status = HealthStatus.INFECTED
                    other_agent.infection_time = 0
                    self.model._agents_to_update.add(other_id)
    
    def update_health(self):
        """Update health status based on infection time."""
        if self.status == HealthStatus.INFECTED:
            self.infection_time += 1
            recovery_time = self.model.p.get('recovery_time', 14)
            
            if self.infection_time >= recovery_time:
                self.status = HealthStatus.RECOVERED
                self.model._agents_to_update.add(self.id)


# In[8]:


class VirusSpreadModel(am.Model):
    """Interactive virus spread model with SIR dynamics."""

    def __init__(self, parameters=None, update_callback: Optional[Callable] = None):
        super().__init__(parameters)
        self.update_callback = update_callback
        self.running = False
        self.paused = False
        
        # Data for real-time plotting
        self.susceptible_history = []
        self.infected_history = []
        self.recovered_history = []
        self.step_history = []
        
        # Agent position data for spatial visualization
        self.agent_positions = []
        self.agent_statuses = []
        
    def setup(self):
        """Initialize model with agents."""
        # Create persistent agents for optimal performance
        self.agent_objects = {}
        for i in range(self.p['n']):
            agent = VirusAgent(self, i)
            agent.setup()
            self.agent_objects[i] = agent
        
        # Create AgentList for compatibility
        self.agents = am.AgentList(self, 0, VirusAgent)
        self.agents.agent_ids = list(range(self.p['n']))
        
        # Track agents needing DataFrame updates
        self._agents_to_update = set()
        
        # Record initial state
        self._record_initial_state()
        self._update_history()

    def _record_initial_state(self):
        """Record initial agent states."""
        agent_data = [{
            'id': int(agent_id),
            'step': int(self.t),
            'status': str(agent.status.value),
            'x': float(agent.x),
            'y': float(agent.y),
            'infection_time': int(agent.infection_time)
        } for agent_id, agent in self.agent_objects.items()]
        
        if agent_data:
            self.agents_df = pl.DataFrame(agent_data)

    def _update_history(self):
        """Update history for real-time plotting and trigger callback."""
        # Count agents by status
        susceptible = sum(1 for agent in self.agent_objects.values() 
                         if agent.status == HealthStatus.SUSCEPTIBLE)
        infected = sum(1 for agent in self.agent_objects.values() 
                      if agent.status == HealthStatus.INFECTED)
        recovered = sum(1 for agent in self.agent_objects.values() 
                       if agent.status == HealthStatus.RECOVERED)
        
        self.susceptible_history.append(susceptible)
        self.infected_history.append(infected)
        self.recovered_history.append(recovered)
        self.step_history.append(self.t)
        
        # Update spatial data
        positions = [(agent.x, agent.y) for agent in self.agent_objects.values()]
        statuses = [agent.status.value for agent in self.agent_objects.values()]
        
        self.agent_positions = positions
        self.agent_statuses = statuses
        
        # Trigger callback for real-time updates
        if self.update_callback:
            self.update_callback(self)

    def step(self):
        """Execute one simulation step."""
        if self.paused:
            return
            
        self._agents_to_update.clear()
        
        # All agents move, interact, and update health
        for agent in self.agent_objects.values():
            agent.move()
            agent.interact()
            agent.update_health()
        
        # Batch update changed agents
        if self._agents_to_update:
            self._batch_update_agents()

    def _batch_update_agents(self):
        """Efficiently batch update agent data."""
        agent_data = [{
            'id': int(agent_id),
            'step': int(self.t),
            'status': str(self.agent_objects[agent_id].status.value),
            'x': float(self.agent_objects[agent_id].x),
            'y': float(self.agent_objects[agent_id].y),
            'infection_time': int(self.agent_objects[agent_id].infection_time)
        } for agent_id in self._agents_to_update]
        
        if agent_data:
            new_data = pl.DataFrame(agent_data)
            self.agents_df = pl.concat([self.agents_df, new_data])

    def update(self):
        """Update model state and trigger callbacks."""
        super().update()
        self._update_history()
        
        # Control simulation speed
        fps = self.p.get('fps', 10)
        if fps > 0:
            time.sleep(1.0 / fps)

    def pause(self):
        """Pause the simulation."""
        self.paused = True
        
    def resume(self):
        """Resume the simulation."""
        self.paused = False
        
    def reset(self):
        """Reset simulation to initial state."""
        self.t = 0
        self.paused = False
        self.running = False
        
        # Reset agents
        for i, agent in enumerate(self.agent_objects.values()):
            agent.status = HealthStatus.SUSCEPTIBLE
            agent.infection_time = 0
            agent.x = float(random.uniform(0, self.p.get('world_size', 100)))
            agent.y = float(random.uniform(0, self.p.get('world_size', 100)))
            
            # Re-infect initial agents
            if i < self.p.get('initial_infected', 5):
                agent.status = HealthStatus.INFECTED
                agent.infection_time = 0
            
        # Clear history
        self.susceptible_history = []
        self.infected_history = []
        self.recovered_history = []
        self.step_history = []
        self.agent_positions = []
        self.agent_statuses = []
        
        # Reset DataFrames
        self.agents_df = pl.DataFrame()
        self.model_df = pl.DataFrame()
        
        # Record initial state
        self._record_initial_state()
        self._update_history()

    def end(self):
        """Finalize simulation."""
        self.running = False


# In[9]:


class VirusSpreadSimulation:
    """State-of-the-art interactive virus spread simulation interface."""
    
    def __init__(self):
        self.model = None
        self.simulation_thread = None
        self.running = False
        self._create_interface()
        
    def _create_interface(self):
        """Create the complete interactive interface."""
        # Parameter controls with modern styling
        style = {'description_width': '140px'}
        layout = widgets.Layout(width='300px')
        
        self.population_slider = widgets.IntSlider(
            value=500, min=50, max=2000, step=50,
            description='Population:', style=style, layout=layout
        )
        
        self.transmission_rate_slider = widgets.FloatSlider(
            value=0.05, min=0.01, max=0.3, step=0.01,
            description='Transmission Rate:', style=style, layout=layout
        )
        
        self.recovery_time_slider = widgets.IntSlider(
            value=14, min=5, max=30, step=1,
            description='Recovery Time:', style=style, layout=layout
        )
        
        self.movement_speed_slider = widgets.FloatSlider(
            value=2.0, min=0.5, max=5.0, step=0.1,
            description='Movement Speed:', style=style, layout=layout
        )
        
        self.infection_radius_slider = widgets.FloatSlider(
            value=5.0, min=1.0, max=15.0, step=0.5,
            description='Infection Radius:', style=style, layout=layout
        )
        
        self.initial_infected_slider = widgets.IntSlider(
            value=5, min=1, max=20, step=1,
            description='Initial Infected:', style=style, layout=layout
        )
        
        self.fps_slider = widgets.IntSlider(
            value=15, min=1, max=30, step=1,
            description='FPS:', style=style, layout=layout
        )
        
        self.max_steps_slider = widgets.IntSlider(
            value=300, min=100, max=1000, step=50,
            description='Max Steps:', style=style, layout=layout
        )
        
        # Control buttons with modern styling
        button_layout = widgets.Layout(width='90px', height='35px')
        
        self.start_button = widgets.Button(
            description='▶️ Start', 
            button_style='success',
            layout=button_layout,
            tooltip='Start simulation'
        )
        
        self.pause_button = widgets.Button(
            description='⏸️ Pause',
            button_style='warning', 
            layout=button_layout,
            tooltip='Pause/Resume simulation'
        )
        
        self.reset_button = widgets.Button(
            description='🔄 Reset',
            button_style='info',
            layout=button_layout,
            tooltip='Reset to initial state'
        )
        
        # Status displays with modern HTML styling
        self.status_display = widgets.HTML(
            value="<div style='font-size: 14px; font-weight: bold;'>Status: <span style='color: #666;'>Ready</span></div>"
        )
        
        self.step_display = widgets.HTML(
            value="<div style='font-size: 14px;'>Step: <span style='color: #007acc;'>0</span></div>"
        )
        
        self.infected_display = widgets.HTML(
            value="<div style='font-size: 14px;'>Infected: <span style='color: #d73027;'>0</span></div>"
        )
        
        self.recovered_display = widgets.HTML(
            value="<div style='font-size: 14px;'>Recovered: <span style='color: #1a9641;'>0</span></div>"
        )
        
        # Create interactive plots
        self._create_plots()
        
        # Bind event handlers
        self.start_button.on_click(self._start_simulation)
        self.pause_button.on_click(self._pause_simulation)
        self.reset_button.on_click(self._reset_simulation)
        
        # Create layout
        self._create_layout()
        
    def _create_plots(self):
        """Create modern interactive Plotly visualizations."""
        # SIR curves plot
        self.sir_figure = go.FigureWidget()
        
        # Add traces for S, I, R
        self.sir_figure.add_trace(go.Scatter(
            x=[], y=[], 
            mode='lines',
            name='Susceptible',
            line=dict(color='#1f77b4', width=3)
        ))
        
        self.sir_figure.add_trace(go.Scatter(
            x=[], y=[], 
            mode='lines',
            name='Infected',
            line=dict(color='#d62728', width=3)
        ))
        
        self.sir_figure.add_trace(go.Scatter(
            x=[], y=[], 
            mode='lines',
            name='Recovered',
            line=dict(color='#2ca02c', width=3)
        ))
        
        self.sir_figure.update_layout(
            title=dict(
                text='<b>SIR Epidemic Curves</b>',
                font=dict(size=16)
            ),
            xaxis_title='Time Step',
            yaxis_title='Number of Agents',
            height=350,
            margin=dict(l=60, r=30, t=60, b=50),
            plot_bgcolor='rgba(240,240,240,0.3)',
            legend=dict(x=0.7, y=0.95)
        )
        
        # Spatial visualization
        self.spatial_figure = go.FigureWidget()
        self.spatial_figure.add_trace(go.Scatter(
            x=[], y=[], 
            mode='markers',
            marker=dict(
                size=8,
                color=[],
                colorscale=[
                    [0, '#1f77b4'],    # Susceptible - Blue
                    [0.5, '#d62728'],  # Infected - Red  
                    [1, '#2ca02c']     # Recovered - Green
                ],
                cmin=0,
                cmax=2,
                showscale=False
            ),
            name='Agents'
        ))
        
        self.spatial_figure.update_layout(
            title=dict(
                text='<b>Spatial Distribution</b>',
                font=dict(size=16)
            ),
            xaxis_title='X Position',
            yaxis_title='Y Position',
            height=350,
            margin=dict(l=60, r=30, t=60, b=50),
            plot_bgcolor='rgba(240,240,240,0.3)',
            showlegend=False,
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 100])
        )
        
    def _create_layout(self):
        """Create the responsive layout."""
        # Control panel with modern styling
        control_panel = widgets.VBox([
            widgets.HTML(
                value="<h3 style='margin-bottom: 20px; color: #333;'>🦠 Epidemic Controls</h3>"
            ),
            self.population_slider,
            self.transmission_rate_slider,
            self.recovery_time_slider,
            self.movement_speed_slider,
            self.infection_radius_slider,
            self.initial_infected_slider,
            self.fps_slider,
            self.max_steps_slider,
            widgets.HTML("<div style='margin: 15px 0;'></div>"),  # Spacer
            widgets.HBox([
                self.start_button, 
                self.pause_button, 
                self.reset_button
            ], layout=widgets.Layout(justify_content='space-between')),
            widgets.HTML(
                value="<h4 style='margin: 20px 0 10px 0; color: #333;'>📊 Status</h4>"
            ),
            self.status_display,
            self.step_display,
            self.infected_display,
            self.recovered_display
        ], layout=widgets.Layout(
            width='340px', 
            padding='20px',
            border='1px solid #ddd',
            border_radius='8px',
            background_color='#fafafa'
        ))
        
        # Plots panel
        plots_panel = widgets.VBox([
            self.sir_figure,
            self.spatial_figure
        ], layout=widgets.Layout(padding='20px'))
        
        # Main interface
        self.interface = widgets.HBox([
            control_panel, 
            plots_panel
        ], layout=widgets.Layout(
            border='2px solid #d62728',
            border_radius='10px',
            padding='10px',
            background_color='white'
        ))
        
    def _get_parameters(self):
        """Get current parameter values from controls."""
        return {
            'n': self.population_slider.value,
            'transmission_rate': self.transmission_rate_slider.value,
            'recovery_time': self.recovery_time_slider.value,
            'movement_speed': self.movement_speed_slider.value,
            'infection_radius': self.infection_radius_slider.value,
            'initial_infected': self.initial_infected_slider.value,
            'fps': self.fps_slider.value,
            'steps': self.max_steps_slider.value,
            'world_size': 100
        }
        
    def _update_visualizations(self, model):
        """Update plots with current model data."""
        # Update SIR curves
        with self.sir_figure.batch_update():
            self.sir_figure.data[0].x = model.step_history
            self.sir_figure.data[0].y = model.susceptible_history
            self.sir_figure.data[1].x = model.step_history
            self.sir_figure.data[1].y = model.infected_history
            self.sir_figure.data[2].x = model.step_history
            self.sir_figure.data[2].y = model.recovered_history
            
        # Update spatial visualization
        if model.agent_positions:
            x_coords = [pos[0] for pos in model.agent_positions]
            y_coords = [pos[1] for pos in model.agent_positions]
            
            # Map status to colors: S=0 (blue), I=1 (red), R=2 (green)
            status_colors = []
            for status in model.agent_statuses:
                if status == 'S':
                    status_colors.append(0)
                elif status == 'I':
                    status_colors.append(1)
                else:  # 'R'
                    status_colors.append(2)
            
            with self.spatial_figure.batch_update():
                self.spatial_figure.data[0].x = x_coords
                self.spatial_figure.data[0].y = y_coords
                self.spatial_figure.data[0].marker.color = status_colors
                
        # Update status displays
        self.step_display.value = f"<div style='font-size: 14px;'>Step: <span style='color: #007acc;'>{model.t}</span></div>"
        
        if model.infected_history:
            infected_count = model.infected_history[-1]
            recovered_count = model.recovered_history[-1]
            
            self.infected_display.value = f"<div style='font-size: 14px;'>Infected: <span style='color: #d73027;'>{infected_count}</span></div>"
            self.recovered_display.value = f"<div style='font-size: 14px;'>Recovered: <span style='color: #1a9641;'>{recovered_count}</span></div>"
            
    def _model_update_callback(self, model):
        """Callback function for real-time model updates."""
        self._update_visualizations(model)
        
    def _start_simulation(self, button):
        """Start simulation in background thread."""
        if self.running:
            return
            
        # Create model with current parameters
        params = self._get_parameters()
        self.model = VirusSpreadModel(
            parameters=params, 
            update_callback=self._model_update_callback
        )
        
        # Update UI
        self.status_display.value = "<div style='font-size: 14px; font-weight: bold;'>Status: <span style='color: #28a745;'>Running</span></div>"
        self.running = True
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
    def _run_simulation(self):
        """Main simulation execution loop."""
        try:
            self.model.setup()
            
            while self.model.t < self.model.p['steps'] and self.running:
                if not self.model.paused:
                    self.model.step()
                    self.model.update()
                    
                    # Stop if no more infected agents
                    if self.model.infected_history and self.model.infected_history[-1] == 0:
                        break
                else:
                    time.sleep(0.1)  # Small delay when paused
                    
            self.model.end()
            
        except Exception as e:
            print(f"❌ Simulation error: {e}")
        finally:
            self.running = False
            self.status_display.value = "<div style='font-size: 14px; font-weight: bold;'>Status: <span style='color: #dc3545;'>Completed</span></div>"
            
    def _pause_simulation(self, button):
        """Pause or resume the simulation."""
        if not self.model:
            return
            
        if self.model.paused:
            self.model.resume()
            self.pause_button.description = '⏸️ Pause'
            self.status_display.value = "<div style='font-size: 14px; font-weight: bold;'>Status: <span style='color: #28a745;'>Running</span></div>"
        else:
            self.model.pause()
            self.pause_button.description = '▶️ Resume'
            self.status_display.value = "<div style='font-size: 14px; font-weight: bold;'>Status: <span style='color: #ffc107;'>Paused</span></div>"
            
    def _reset_simulation(self, button):
        """Reset simulation to initial conditions."""
        self.running = False
        
        if self.model:
            self.model.reset()
            self._update_visualizations(self.model)
            
        # Reset UI
        self.status_display.value = "<div style='font-size: 14px; font-weight: bold;'>Status: <span style='color: #666;'>Ready</span></div>"
        self.pause_button.description = '⏸️ Pause'
        
    def display(self):
        """Display the interactive interface."""
        return self.interface

"""
## 🚀 Launch Virus Spread Simulation

Run the cell below to launch the interactive epidemiological simulation. The interface includes:

### 🎛️ **Epidemic Parameters**
- **Population**: Total number of agents (50-2000)
- **Transmission Rate**: Probability of infection per contact (0.01-0.3)
- **Recovery Time**: Days until infected agents recover (5-30)
- **Movement Speed**: How fast agents move around (0.5-5.0)
- **Infection Radius**: Distance for potential transmission (1.0-15.0)
- **Initial Infected**: Number of initially infected agents (1-20)

### 🎮 **Control Buttons**
- **▶️ Start**: Begin epidemic simulation
- **⏸️ Pause/Resume**: Pause or resume the simulation
- **🔄 Reset**: Reset to initial conditions

### 📊 **Real-time Visualizations**
- **SIR Curves**: Classic epidemiological curves showing Susceptible, Infected, and Recovered populations over time
- **Spatial Distribution**: Live 2D visualization of agent positions colored by health status
  - 🔵 **Blue**: Susceptible agents
  - 🔴 **Red**: Infected agents  
  - 🟢 **Green**: Recovered agents

### ✨ **Advanced Features**
- **Spatial dynamics**: Agents move randomly and interact based on proximity
- **Realistic transmission**: Distance-based infection with probabilistic transmission
- **Automatic termination**: Simulation stops when no infected agents remain
- **Real-time analytics**: Live tracking of epidemic progression
"""

# In[10]:


# Create and display the interactive virus spread simulation
virus_simulation = VirusSpreadSimulation()
virus_simulation.display()

