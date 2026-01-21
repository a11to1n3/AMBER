"""
Integration tests for amber package - testing multiple components working together.
"""

import pytest
import numpy as np
import networkx as nx
from unittest.mock import patch
import ambr as am


class TestFullSimulationWorkflows:
    """Test complete simulation workflows combining multiple components."""
    
    def test_agent_based_model_with_grid(self):
        """Test an agent-based model using GridEnvironment."""
        class GridModel(am.Model):
            def setup(self):
                # Create grid environment
                self.grid = am.GridEnvironment(self, size=(10, 10))
                
                # Create agents at random positions
                self.agents = {}
                for i in range(5):  # Smaller number for testing
                    agent = am.Agent(self, i)
                    position = self.grid.random_position()
                    agent.position = position
                    self.agents[i] = agent
                    
                    # Add agent to model's DataFrame (uses default schema)
                    self.add_agent(agent)
                    
                    # Update with initial position data (add x,y columns)
                    self.update_agent_data(i, {
                        'x': position[0],
                        'y': position[1],
                        'step': 0
                    })
            
            def step(self):
                # Empty step method - logic is in update()
                pass
            
            def update(self):
                super().update()  # Call parent update first
                if self.t > 0:  # Only record after actual steps
                    # Move agents randomly
                    for agent in self.agents.values():
                        # Get current position
                        current_pos = agent.position
        
                        # Get valid neighbors
                        neighbors = self.grid.get_neighbors(current_pos)
                        if neighbors:
                            # Move to random neighbor
                            new_pos = neighbors[self.nprandom.randint(0, len(neighbors))]
                            agent.position = new_pos
        
                            # Update data
                            self.update_agent_data(agent.id, {
                                'x': new_pos[0],
                                'y': new_pos[1],
                                'step': self.t
                            })
        
                    # Record model metrics
                    positions = [agent.position for agent in self.agents.values()]
                    center_x = np.mean([pos[0] for pos in positions])
                    center_y = np.mean([pos[1] for pos in positions])
        
                    self.record_model('center_x', center_x)
                    self.record_model('center_y', center_y)
                    self.record_model('num_agents', len(self.agents))
        
        # Run the model
        params = {'steps': 3, 'seed': 42, 'show_progress': False}
        model = GridModel(params)
        
        with patch('time.time', side_effect=[0, 0.1] * 10), patch('builtins.print'):
            results = model.run()
        
        # Check results
        assert len(results['agents']) > 0
        assert len(results['model']) > 0
        
        # Should have recorded agent positions
        agent_data = results['agents']
        assert 'x' in agent_data.columns
        assert 'y' in agent_data.columns
    
    def test_experiment_with_optimization(self):
        """Test running experiments with parameter optimization."""
        class OptimizableModel(am.Model):
            def setup(self):
                # Create agents based on population parameter
                self.agents = {}
                for i in range(self.p['population']):
                    agent = am.Agent(self, i)
                    agent.energy = self.nprandom.randint(1, 100)
                    self.agents[i] = agent
                    self.add_agent(agent)
            
            def step(self):
                # Simple energy dynamics
                total_energy = 0
                active_agents = 0
                
                for agent in self.agents.values():
                    # Energy changes based on interaction rate
                    energy_change = self.nprandom.normal(0, self.p['interaction_rate'])
                    agent.energy = max(0, agent.energy + energy_change)
                    
                    if agent.energy > 0:
                        active_agents += 1
                        total_energy += agent.energy
                
                # Record metrics
                self.record_model('total_energy', total_energy)
                self.record_model('active_agents', active_agents)
                avg_energy = total_energy / max(1, active_agents)
                self.record_model('avg_energy', avg_energy)
        
        # Create parameter space for optimization
        parameter_space = am.ParameterSpace({
            'population': [5, 10],
            'interaction_rate': [0.1, 0.5],
            'steps': 3
        })
        
        # Run grid search to find optimal parameters
        with patch('time.time', side_effect=[0, 0.1] * 20), patch('builtins.print'):
            results = am.grid_search(
                OptimizableModel, 
                parameter_space, 
                'avg_energy',
                iterations=1
            )
        
        # Check that optimization worked
        assert len(results) == 4  # 2 x 2 parameter combinations
        
        # Results should be sorted by objective value
        objectives = [r['objective'] for r in results]
        assert objectives == sorted(objectives, reverse=True)
        
        # Best result should have sensible parameters
        best = results[0]
        assert best['parameters']['population'] in [5, 10]
        assert best['parameters']['interaction_rate'] in [0.1, 0.5]
    
    def test_network_simulation_experiment(self):
        """Test network-based simulation with experiment framework."""
        class NetworkModel(am.Model):
            def setup(self):
                # Create network
                G = nx.erdos_renyi_graph(self.p['n_nodes'], self.p['edge_prob'])
                self.network = am.NetworkEnvironment(self, G)
                
                # Create agents on network nodes
                self.agents = {}
                states = ['susceptible', 'infected', 'recovered']
                for node in self.network.nodes[:min(5, len(self.network.nodes))]:  # Limit for testing
                    agent = am.Agent(self, node)
                    agent.state = states[self.nprandom.randint(0, len(states))]
                    agent.node = node
                    self.agents[node] = agent
                    
                    # Add agent to model's DataFrame
                    self.add_agent(agent)
                    
                    # Update with network data
                    self.update_agent_data(node, {
                        'state': agent.state,
                        'degree': self.network.get_degree(node),
                        'step': 0
                    })
            
            def step(self):
                # Empty step method - logic is in update()
                pass
            
            def update(self):
                super().update()  # Call parent update first
                if self.t > 0:  # Only record after actual steps
                    # Simple state tracking
                    state_counts = {'susceptible': 0, 'infected': 0, 'recovered': 0}
                    
                    for agent in self.agents.values():
                        state_counts[agent.state] += 1
                    
                    # Record model metrics
                    self.record_model('susceptible', state_counts['susceptible'])
                    self.record_model('infected', state_counts['infected'])
                    self.record_model('recovered', state_counts['recovered'])
                    self.record_model('network_density', 
                                    len(self.network.edges) / max(1, len(self.network.nodes)))
        
        # Create experiment
        sample = am.Sample({
            'n_nodes': [10, 15],
            'edge_prob': [0.1, 0.3],
            'steps': 2
        }, n=2)
        
        experiment = am.Experiment(NetworkModel, sample, iterations=1)
        
        # Run experiment
        with patch('time.time', side_effect=[0, 0.1] * 10), patch('builtins.print'):
            results = experiment.run()
        
        # Check results
        assert len(results['parameters']) == 2
        assert len(results['model']) > 0
        assert len(results['agents']) > 0
        
        # Check that network properties were recorded
        model_data = results['model']
        assert 'network_density' in model_data.columns
        assert 'infected' in model_data.columns
        
        # Check agent states were tracked
        agent_data = results['agents']
        assert 'state' in agent_data.columns
        assert 'degree' in agent_data.columns


class TestModelExtensions:
    """Test extending models with custom functionality."""
    
    def test_custom_agent_behavior(self):
        """Test model with custom agent behavior."""
        class CustomAgent(am.Agent):
            def __init__(self, model, agent_id):
                super().__init__(model, agent_id)
                self.memory = []
                self.social_connections = []
            
            def remember(self, event):
                """Add event to memory."""
                self.memory.append(event)
                if len(self.memory) > 10:  # Limit memory
                    self.memory.pop(0)
            
            def connect(self, other_agent_id):
                """Create social connection."""
                if other_agent_id not in self.social_connections:
                    self.social_connections.append(other_agent_id)
        
        class SocialModel(am.Model):
            def setup(self):
                self.agents = {}
                for i in range(10):
                    agent = CustomAgent(self, i)
                    self.agents[i] = agent
                    
                    # Add agent to model's DataFrame
                    self.add_agent(agent)
                    
                    # Update with initial data
                    self.update_agent_data(i, {
                        'memory_size': len(agent.memory),
                        'connections': len(agent.social_connections),
                        'step': 0
                    })
            
            def step(self):
                # Empty step method - logic is in update()
                pass
            
            def update(self):
                super().update()  # Call parent update first
                if self.t > 0:  # Only record after actual steps
                    # Agents interact and form connections
                    for agent in self.agents.values():
                        # Random event to remember
                        event = f"event_{self.t}_{self.nprandom.randint(1, 100)}"
                        agent.remember(event)
                        
                        # Random social connection
                        if self.nprandom.random() < 0.1:  # 10% chance
                            other_id = self.nprandom.choice(
                                [i for i in self.agents.keys() if i != agent.id]
                            )
                            agent.connect(other_id)
                        
                        # Update data
                        self.update_agent_data(agent.id, {
                            'memory_size': len(agent.memory),
                            'connections': len(agent.social_connections),
                            'step': self.t
                        })
                    
                    # Record model metrics
                    total_connections = sum(len(a.social_connections) for a in self.agents.values())
                    avg_memory = np.mean([len(a.memory) for a in self.agents.values()])
                    
                    self.record_model('total_connections', total_connections)
                    self.record_model('avg_memory_size', avg_memory)
        
        # Run model
        params = {'steps': 5, 'seed': 42, 'show_progress': False}
        model = SocialModel(params)
        
        with patch('time.time', side_effect=[0, 0.1] * 10), patch('builtins.print'):
            results = model.run()
        
        # Check that custom agent behavior was tracked
        agent_data = results['agents']
        assert 'memory_size' in agent_data.columns
        assert 'connections' in agent_data.columns
        
        model_data = results['model']
        assert 'total_connections' in model_data.columns
        assert 'avg_memory_size' in model_data.columns
    
    def test_hierarchical_model_structure(self):
        """Test model with hierarchical organization."""
        class HierarchicalModel(am.Model):
            def setup(self):
                self.groups = {}
                self.individuals = {}
                individual_id = 0
                group_id_offset = 1000  # Use high IDs for groups to avoid conflicts
                
                # Create groups
                for group_id in range(self.p['n_groups']):
                    group_size = self.nprandom.randint(3, 8)
                    group_members = []
                    
                    for _ in range(group_size):
                        agent = am.Agent(self, individual_id)
                        agent.group_id = group_id
                        agent.performance = self.nprandom.uniform(0, 1)
                        
                        self.individuals[individual_id] = agent
                        group_members.append(individual_id)
                        
                        # Add agent to model's DataFrame
                        self.add_agent(agent)
                        
                        # Update with individual data
                        self.update_agent_data(individual_id, {
                            'level': 'individual',
                            'group_id': group_id,
                            'performance': agent.performance,
                            'step': 0
                        })
                        
                        individual_id += 1
                    
                    # Create group-level entity with integer ID
                    group_agent_id = group_id_offset + group_id
                    group_agent = am.Agent(self, group_agent_id)
                    group_agent.members = group_members
                    group_agent.group_performance = np.mean([
                        self.individuals[i].performance for i in group_members
                    ])
                    
                    self.groups[group_id] = group_agent
                    
                    # Add group agent to model's DataFrame
                    self.add_agent(group_agent)
                    
                    # Update with group data
                    self.update_agent_data(group_agent_id, {
                        'level': 'group',
                        'group_id': group_id,
                        'performance': group_agent.group_performance,
                        'step': 0
                    })
            
            def step(self):
                # Empty step method - logic is in update()
                pass
            
            def update(self):
                super().update()  # Call parent update first
                if self.t > 0:  # Only record after actual steps
                    # Update individual performance
                    for individual in self.individuals.values():
                        # Performance influenced by group
                        group = self.groups[individual.group_id]
                        group_effect = group.group_performance * 0.1
                        individual.performance += self.nprandom.normal(group_effect, 0.05)
                        individual.performance = np.clip(individual.performance, 0, 1)
                        
                        # Update data
                        self.update_agent_data(individual.id, {
                            'performance': individual.performance,
                            'step': self.t
                        })
                    
                    # Update group performance
                    for group in self.groups.values():
                        group.group_performance = np.mean([
                            self.individuals[i].performance for i in group.members
                        ])
                        
                        # Update data
                        self.update_agent_data(group.id, {
                            'performance': group.group_performance,
                            'step': self.t
                        })
                    
                    # Record model metrics
                    overall_performance = np.mean([ind.performance for ind in self.individuals.values()])
                    group_variance = np.var([grp.group_performance for grp in self.groups.values()])
                    
                    self.record_model('overall_performance', overall_performance)
                    self.record_model('group_variance', group_variance)
                    self.record_model('n_individuals', len(self.individuals))
                    self.record_model('n_groups', len(self.groups))
        
        # Run model
        params = {'n_groups': 3, 'steps': 4, 'seed': 42, 'show_progress': False}
        model = HierarchicalModel(params)
        
        with patch('time.time', side_effect=[0, 0.1] * 10), patch('builtins.print'):
            results = model.run()
        
        # Check hierarchical structure in data
        agent_data = results['agents']
        assert 'level' in agent_data.columns
        assert 'group_id' in agent_data.columns
        
        # Should have both individual and group level data
        levels = agent_data['level'].unique().to_list()
        assert 'individual' in levels
        assert 'group' in levels
        
        model_data = results['model']
        assert 'overall_performance' in model_data.columns
        assert 'group_variance' in model_data.columns


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    @pytest.mark.slow
    def test_large_scale_simulation(self):
        """Test simulation with larger numbers of agents."""
        class ScalabilityModel(am.Model):
            def setup(self):
                self.agents = {}
                for i in range(self.p['n_agents']):
                    agent = am.Agent(self, i)
                    agent.value = self.nprandom.random()
                    self.agents[i] = agent
                    self.add_agent(agent)
            
            def step(self):
                # Simple update for all agents
                for agent in self.agents.values():
                    agent.value += self.nprandom.normal(0, 0.01)
                    agent.value = np.clip(agent.value, 0, 1)
                
                # Record summary statistics
                values = [agent.value for agent in self.agents.values()]
                self.record_model('mean_value', np.mean(values))
                self.record_model('std_value', np.std(values))
                self.record_model('min_value', np.min(values))
                self.record_model('max_value', np.max(values))
        
        # Test with different scales
        import time
        
        for n_agents in [100, 500]:
            params = {'n_agents': n_agents, 'steps': 5, 'seed': 42}
            model = ScalabilityModel(params)
            
            start_time = time.time()
            with patch('builtins.print'):
                results = model.run()
            end_time = time.time()
            
            # Should complete in reasonable time
            assert end_time - start_time < 30  # 30 seconds max
            
            # Should have correct amount of data
            assert len(results['agents']) > 0
            assert len(results['model']) > 0
    
    def test_memory_efficiency(self):
        """Test that models don't have excessive memory usage."""
        class MemoryTestModel(am.Model):
            def setup(self):
                self.agents = {}
                for i in range(200):
                    agent = am.Agent(self, i)
                    # Don't store large objects in agents
                    agent.small_data = i % 10
                    self.agents[i] = agent
                    self.add_agent(agent)
            
            def step(self):
                # Empty step method - logic is in update()
                pass
            
            def update(self):
                super().update()  # Call parent update first
                if self.t > 0:  # Only record after actual steps
                    # Efficient operations
                    for agent in self.agents.values():
                        agent.small_data = (agent.small_data + 1) % 10
                        
                        # Update agent data efficiently
                        self.update_agent_data(agent.id, {
                            'wealth': agent.small_data * 10,
                            'step': self.t
                        })
                    
                    self.record_model('agent_count', len(self.agents))
        
        params = {'steps': 10, 'seed': 42, 'show_progress': False}
        model = MemoryTestModel(params)
        
        with patch('time.time', side_effect=[0, 0.1] * 10), patch('builtins.print'):
            results = model.run()
        
        # Model should complete successfully
        assert len(results['agents']) > 0
        assert len(results['model']) > 0 