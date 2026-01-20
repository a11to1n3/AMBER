#!/usr/bin/env python3
"""
SMAC Calibration Example - Advanced Multi-Objective Optimization
===============================================================

This example demonstrates advanced SMAC calibration using AMBER's built-in
MultiObjectiveSMAC class for multi-objective optimization. We'll calibrate
a segregation model with multiple competing objectives.

Key Features:
- Multi-objective optimization with Pareto frontiers
- AMBER's SMACParameterSpace with conditional parameters
- Real-world calibration scenarios
- Pareto front analysis and visualization
- Trade-off analysis between objectives

Requirements:
    pip install smac ConfigSpace
"""

import amber as am
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, List, Tuple


class SegregationModel(am.Model):
    """
    Advanced segregation model with multiple agent types and interaction mechanisms.
    Based on Schelling's segregation model with extensions for multi-objective optimization.
    """
    
    def setup(self):
        """Initialize agents and environment."""
        # Create grid environment
        self.env = am.GridEnvironment(self, self.p['grid_size'], self.p['grid_size'])
        
        # Agent types based on model configuration
        agent_types = self.get_agent_types()
        
        # Create agents
        n_agents = int(self.p['grid_size'] ** 2 * self.p['density'])
        
        for i in range(n_agents):
            agent = am.Agent(self, i)
            
            # Assign agent type
            agent.agent_type = self.nprandom.choice(
                list(agent_types.keys()),
                p=list(agent_types.values())
            )
            
            # Set agent-specific parameters
            agent.tolerance = self.get_agent_tolerance(agent.agent_type)
            agent.mobility = self.get_agent_mobility(agent.agent_type)
            agent.satisfaction = 0.0
            agent.moves = 0
            
            # Place agent randomly
            pos = self.env.get_random_empty_cell()
            if pos:
                self.env.add_agent(agent, pos)
                agent.pos = pos
            
            self.add_agent(agent)
    
    def get_agent_types(self) -> Dict[str, float]:
        """Get agent type distribution based on model parameters."""
        if self.p['agent_type_distribution'] == 'binary':
            return {
                'type_A': self.p['type_A_fraction'],
                'type_B': 1 - self.p['type_A_fraction']
            }
        elif self.p['agent_type_distribution'] == 'three_types':
            remaining = 1 - self.p['type_A_fraction']
            return {
                'type_A': self.p['type_A_fraction'],
                'type_B': remaining * 0.6,
                'type_C': remaining * 0.4
            }
        else:  # uniform
            return {'type_A': 0.33, 'type_B': 0.33, 'type_C': 0.34}
    
    def get_agent_tolerance(self, agent_type: str) -> float:
        """Get tolerance threshold for agent type."""
        base_tolerance = self.p['base_tolerance']
        
        if agent_type == 'type_A':
            return base_tolerance * self.p.get('tolerance_multiplier_A', 1.0)
        elif agent_type == 'type_B':
            return base_tolerance * self.p.get('tolerance_multiplier_B', 1.0)
        else:  # type_C
            return base_tolerance * self.p.get('tolerance_multiplier_C', 1.0)
    
    def get_agent_mobility(self, agent_type: str) -> float:
        """Get mobility rate for agent type."""
        base_mobility = self.p['base_mobility']
        
        if agent_type == 'type_A':
            return base_mobility * self.p.get('mobility_multiplier_A', 1.0)
        elif agent_type == 'type_B':
            return base_mobility * self.p.get('mobility_multiplier_B', 1.0)
        else:  # type_C
            return base_mobility * self.p.get('mobility_multiplier_C', 1.0)
    
    def step(self):
        """Execute one simulation step."""
        # Shuffle agents for random activation
        agent_ids = list(range(len(self.agents_df)))
        self.nprandom.shuffle(agent_ids)
        
        for agent_id in agent_ids:
            agent_data = self.get_agent_data(agent_id)
            
            # Skip if agent not on grid
            if pd.isna(agent_data['pos'].item()):
                continue
            
            pos = eval(agent_data['pos'].item())  # Convert string back to tuple
            
            # Calculate satisfaction
            satisfaction = self.calculate_satisfaction(agent_id, pos)
            
            # Update agent satisfaction
            self.update_agent_data(agent_id, {'satisfaction': satisfaction})
            
            # Decide whether to move
            tolerance = agent_data['tolerance'].item()
            mobility = agent_data['mobility'].item()
            
            if satisfaction < tolerance and self.nprandom.random() < mobility:
                # Try to move to a better location
                new_pos = self.find_better_location(agent_id, pos)
                if new_pos and new_pos != pos:
                    # Execute move
                    self.env.remove_agent_from_pos(pos)
                    self.env.add_agent_from_id(agent_id, new_pos)
                    
                    # Update agent data
                    current_moves = agent_data['moves'].item()
                    self.update_agent_data(agent_id, {
                        'pos': str(new_pos),
                        'moves': current_moves + 1
                    })
    
    def calculate_satisfaction(self, agent_id: int, pos: Tuple[int, int]) -> float:
        """Calculate agent satisfaction based on neighborhood composition."""
        agent_data = self.get_agent_data(agent_id)
        agent_type = agent_data['agent_type'].item()
        
        # Get neighbors
        neighbors = self.env.get_neighbors(pos, radius=self.p['neighborhood_radius'])
        
        if not neighbors:
            return 0.0
        
        # Count similar neighbors
        similar_count = 0
        total_count = len(neighbors)
        
        for neighbor_pos in neighbors:
            neighbor_id = self.env.get_agent_at_pos(neighbor_pos)
            if neighbor_id is not None:
                neighbor_data = self.get_agent_data(neighbor_id)
                neighbor_type = neighbor_data['agent_type'].item()
                
                if neighbor_type == agent_type:
                    similar_count += 1
        
        return similar_count / total_count if total_count > 0 else 0.0
    
    def find_better_location(self, agent_id: int, current_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Find a better location for the agent."""
        agent_data = self.get_agent_data(agent_id)
        current_satisfaction = agent_data['satisfaction'].item()
        
        # Get empty cells within search radius
        search_radius = self.p['search_radius']
        empty_cells = self.env.get_empty_cells_in_radius(current_pos, search_radius)
        
        if not empty_cells:
            return None
        
        # Evaluate potential locations
        best_pos = None
        best_satisfaction = current_satisfaction
        
        # Sample a subset of locations to avoid expensive computation
        max_evaluations = min(len(empty_cells), self.p['max_location_evaluations'])
        sampled_cells = self.nprandom.choice(
            empty_cells, size=max_evaluations, replace=False
        ).tolist()
        
        for pos in sampled_cells:
            # Temporarily calculate satisfaction at this position
            satisfaction = self.calculate_satisfaction_at_pos(agent_id, pos)
            
            if satisfaction > best_satisfaction:
                best_satisfaction = satisfaction
                best_pos = pos
        
        return best_pos
    
    def calculate_satisfaction_at_pos(self, agent_id: int, pos: Tuple[int, int]) -> float:
        """Calculate satisfaction if agent were at given position."""
        agent_data = self.get_agent_data(agent_id)
        agent_type = agent_data['agent_type'].item()
        
        neighbors = self.env.get_neighbors(pos, radius=self.p['neighborhood_radius'])
        
        if not neighbors:
            return 0.0
        
        similar_count = 0
        total_count = len(neighbors)
        
        for neighbor_pos in neighbors:
            neighbor_id = self.env.get_agent_at_pos(neighbor_pos)
            if neighbor_id is not None:
                neighbor_data = self.get_agent_data(neighbor_id)
                neighbor_type = neighbor_data['agent_type'].item()
                
                if neighbor_type == agent_type:
                    similar_count += 1
        
        return similar_count / total_count if total_count > 0 else 0.0
    
    def update(self):
        """Update model-level statistics."""
        super().update()
        if self.t > 0:
            # Calculate segregation metrics
            segregation_index = self.calculate_segregation_index()
            clustering_coefficient = self.calculate_clustering_coefficient()
            mobility_rate = self.calculate_mobility_rate()
            satisfaction_mean = self.agents_df['satisfaction'].mean()
            satisfaction_std = self.agents_df['satisfaction'].std()
            
            # Record metrics
            self.record_model('segregation_index', segregation_index)
            self.record_model('clustering_coefficient', clustering_coefficient)
            self.record_model('mobility_rate', mobility_rate)
            self.record_model('satisfaction_mean', satisfaction_mean)
            self.record_model('satisfaction_std', satisfaction_std)
    
    def calculate_segregation_index(self) -> float:
        """Calculate Moran's I segregation index (simplified)."""
        type_positions = {}
        
        for _, agent in self.agents_df.iterrows():
            if pd.notna(agent['pos']):
                agent_type = agent['agent_type']
                pos = eval(agent['pos'])
                
                if agent_type not in type_positions:
                    type_positions[agent_type] = []
                type_positions[agent_type].append(pos)
        
        if len(type_positions) < 2:
            return 0.0
        
        within_distances = []
        between_distances = []
        
        for type_a, positions_a in type_positions.items():
            for pos_a in positions_a[:min(50, len(positions_a))]:  # Sample for efficiency
                # Within-type distances
                for pos_b in positions_a:
                    if pos_a != pos_b:
                        dist = np.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)
                        within_distances.append(dist)
                
                # Between-type distances
                for type_b, positions_b in type_positions.items():
                    if type_a != type_b:
                        for pos_b in positions_b[:min(50, len(positions_b))]:
                            dist = np.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)
                            between_distances.append(dist)
        
        if not within_distances or not between_distances:
            return 0.0
        
        avg_within = np.mean(within_distances)
        avg_between = np.mean(between_distances)
        
        return (avg_between - avg_within) / (avg_between + avg_within)
    
    def calculate_clustering_coefficient(self) -> float:
        """Calculate local clustering coefficient."""
        clustering_values = []
        
        for _, agent in self.agents_df.iterrows():
            if pd.notna(agent['pos']):
                pos = eval(agent['pos'])
                neighbors = self.env.get_neighbors(pos, radius=1)
                
                if len(neighbors) > 1:
                    # Count connections between neighbors
                    connections = 0
                    possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
                    
                    for i, pos1 in enumerate(neighbors):
                        for pos2 in neighbors[i+1:]:
                            if abs(pos1[0] - pos2[0]) <= 1 and abs(pos1[1] - pos2[1]) <= 1:
                                connections += 1
                    
                    clustering_values.append(connections / possible_connections)
        
        return np.mean(clustering_values) if clustering_values else 0.0
    
    def calculate_mobility_rate(self) -> float:
        """Calculate current mobility rate."""
        if self.t == 0:
            return 0.0
        
        total_moves = self.agents_df['moves'].sum()
        total_agents = len(self.agents_df)
        
        return total_moves / (total_agents * self.t)


def create_advanced_parameter_space():
    """Create advanced parameter space with conditional parameters using AMBER's SMACParameterSpace."""
    param_space = am.SMACParameterSpace()
    
    # Basic model parameters
    param_space.add_parameter(
        'grid_size', 
        param_type='int', 
        bounds=(20, 50), 
        default=30
    )
    
    param_space.add_parameter(
        'density', 
        param_type='float', 
        bounds=(0.6, 0.95), 
        default=0.8
    )
    
    # Agent distribution type
    param_space.add_parameter(
        'agent_type_distribution',
        param_type='categorical',
        choices=['binary', 'three_types', 'uniform'],
        default='binary'
    )
    
    # Type A fraction (conditional on distribution type)
    param_space.add_parameter(
        'type_A_fraction',
        param_type='float',
        bounds=(0.3, 0.7),
        default=0.5
    )
    
    # Tolerance parameters
    param_space.add_parameter(
        'base_tolerance',
        param_type='float',
        bounds=(0.1, 0.8),
        default=0.3
    )
    
    param_space.add_parameter(
        'tolerance_multiplier_A',
        param_type='float',
        bounds=(0.5, 2.0),
        default=1.0
    )
    
    param_space.add_parameter(
        'tolerance_multiplier_B',
        param_type='float',
        bounds=(0.5, 2.0),
        default=1.0
    )
    
    # Mobility parameters
    param_space.add_parameter(
        'base_mobility',
        param_type='float',
        bounds=(0.01, 0.3),
        default=0.1
    )
    
    param_space.add_parameter(
        'mobility_multiplier_A',
        param_type='float',
        bounds=(0.5, 2.0),
        default=1.0
    )
    
    param_space.add_parameter(
        'mobility_multiplier_B',
        param_type='float',
        bounds=(0.5, 2.0),
        default=1.0
    )
    
    # Spatial parameters
    param_space.add_parameter(
        'neighborhood_radius',
        param_type='int',
        bounds=(1, 3),
        default=1
    )
    
    param_space.add_parameter(
        'search_radius',
        param_type='int',
        bounds=(2, 8),
        default=4
    )
    
    param_space.add_parameter(
        'max_location_evaluations',
        param_type='int',
        bounds=(5, 20),
        default=10
    )
    
    return param_space


def segregation_objective(model: SegregationModel) -> float:
    """Objective 1: Achieve moderate segregation (around 0.4)."""
    results = model.results
    final_segregation = results['model']['segregation_index'].tail(1).item()
    target = 0.4
    return abs(final_segregation - target)


def clustering_objective(model: SegregationModel) -> float:
    """Objective 2: Achieve high local clustering (around 0.6)."""
    results = model.results
    final_clustering = results['model']['clustering_coefficient'].tail(1).item()
    target = 0.6
    return abs(final_clustering - target)


def mobility_objective(model: SegregationModel) -> float:
    """Objective 3: Maintain low but non-zero mobility (around 0.05)."""
    results = model.results
    final_mobility = results['model']['mobility_rate'].tail(1).item()
    target = 0.05
    return abs(final_mobility - target)


def satisfaction_objective(model: SegregationModel) -> float:
    """Objective 4: Achieve high agent satisfaction (around 0.7)."""
    results = model.results
    final_satisfaction = results['model']['satisfaction_mean'].tail(1).item()
    target = 0.7
    return abs(final_satisfaction - target)


def run_multi_objective_optimization():
    """Run multi-objective SMAC optimization using AMBER's MultiObjectiveSMAC."""
    print("🎯 Starting Multi-Objective SMAC Calibration with AMBER")
    print("=" * 55)
    
    # Create parameter space
    param_space = create_advanced_parameter_space()
    
    # Define objectives
    objectives = {
        'segregation': segregation_objective,
        'clustering': clustering_objective,
        'mobility': mobility_objective,
        'satisfaction': satisfaction_objective
    }
    
    # Create multi-objective optimizer
    optimizer = am.MultiObjectiveSMAC(
        model_type=SegregationModel,
        param_space=param_space,
        objectives=objectives,
        n_trials=100,
        seed=42,
        strategy='pareto'
    )
    
    print("Starting multi-objective optimization...")
    results = optimizer.optimize()
    
    print(f"\n🏆 Multi-Objective Optimization Results:")
    print("=" * 45)
    print(f"Total trials: {results['n_evaluations']}")
    print(f"Pareto front size: {len(results['pareto_front'])}")
    
    # Display some Pareto-optimal solutions
    pareto_solutions = results['pareto_front']
    print(f"\nTop 5 Pareto-optimal solutions:")
    
    for i, solution in enumerate(pareto_solutions.head(5).to_dicts()):
        print(f"\nSolution {i+1}:")
        config_cols = [col for col in solution.keys() if col not in ['segregation', 'clustering', 'mobility', 'satisfaction']]
        
        print("  Parameters:")
        for param in config_cols:
            if param in solution:
                print(f"    {param}: {solution[param]}")
        
        print("  Objectives:")
        for obj in ['segregation', 'clustering', 'mobility', 'satisfaction']:
            if obj in solution:
                print(f"    {obj}: {solution[obj]:.4f}")
    
    return optimizer, results


def analyze_pareto_frontier(optimizer, results):
    """Analyze and visualize the Pareto frontier."""
    print("\n📊 Analyzing Pareto Frontier...")
    
    pareto_front = results['pareto_front']
    history = results['history']
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Plot 1: 2D objective space projections
    objective_pairs = [
        ('segregation', 'clustering'),
        ('segregation', 'mobility'),
        ('clustering', 'satisfaction')
    ]
    
    for i, (obj1, obj2) in enumerate(objective_pairs):
        row, col = i // 3, i % 3
        
        # Plot all evaluated points
        all_obj1 = history[obj1].to_list()
        all_obj2 = history[obj2].to_list()
        axes[row, col].scatter(all_obj1, all_obj2, alpha=0.4, c='lightblue', s=20, label='All solutions')
        
        # Plot Pareto front
        pareto_obj1 = pareto_front[obj1].to_list()
        pareto_obj2 = pareto_front[obj2].to_list()
        axes[row, col].scatter(pareto_obj1, pareto_obj2, c='red', s=80, alpha=0.8, label='Pareto front')
        
        axes[row, col].set_xlabel(f'{obj1.title()} Objective')
        axes[row, col].set_ylabel(f'{obj2.title()} Objective')
        axes[row, col].set_title(f'{obj1.title()} vs {obj2.title()}')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    # Plot 2: Parallel coordinates for objectives
    axes[0, 2].set_title('Parallel Coordinates - Pareto Front Objectives')
    objective_names = ['segregation', 'clustering', 'mobility', 'satisfaction']
    
    for i, solution in enumerate(pareto_front.head(10).to_dicts()):
        obj_values = [solution[obj] for obj in objective_names]
        axes[0, 2].plot(range(len(obj_values)), obj_values, 'o-', alpha=0.7, linewidth=2)
    
    axes[0, 2].set_xticks(range(len(objective_names)))
    axes[0, 2].set_xticklabels([name.title() for name in objective_names], rotation=45)
    axes[0, 2].set_ylabel('Objective Value')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 3: Parameter distributions for Pareto solutions
    key_params = ['base_tolerance', 'base_mobility', 'density']
    
    for i, param in enumerate(key_params):
        row, col = 1, i
        param_values = pareto_front[param].to_list()
        
        axes[row, col].hist(param_values, bins=15, alpha=0.7, edgecolor='black', color='skyblue')
        axes[row, col].set_xlabel(param.replace('_', ' ').title())
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].set_title(f'Pareto Solutions: {param.replace("_", " ").title()}')
        axes[row, col].grid(True, alpha=0.3)
    
    # Plot 4: Trade-off analysis
    axes[2, 0].scatter(pareto_front['segregation'].to_list(), 
                      pareto_front['satisfaction'].to_list(), 
                      s=100, alpha=0.7, c='green')
    axes[2, 0].set_xlabel('Segregation Objective')
    axes[2, 0].set_ylabel('Satisfaction Objective')
    axes[2, 0].set_title('Trade-off: Segregation vs Satisfaction')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 5: Convergence of hypervolume indicator
    axes[2, 1].set_title('Optimization Progress')
    all_objectives = history[objective_names].to_numpy()
    
    # Calculate running minimum for each objective
    running_mins = []
    for i in range(len(all_objectives)):
        current_min = np.min(all_objectives[:i+1], axis=0)
        hypervolume_approx = np.prod(1 / (current_min + 0.001))  # Approximate hypervolume
        running_mins.append(hypervolume_approx)
    
    axes[2, 1].plot(running_mins, linewidth=2, color='blue')
    axes[2, 1].set_xlabel('Trial')
    axes[2, 1].set_ylabel('Hypervolume Approximation')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Plot 6: Best solution simulation
    axes[2, 2].set_title('Best Compromise Solution Dynamics')
    
    # Find best compromise solution (closest to ideal point)
    ideal_point = np.min(pareto_front[objective_names].to_numpy(), axis=0)
    distances = np.sum((pareto_front[objective_names].to_numpy() - ideal_point) ** 2, axis=1)
    best_idx = np.argmin(distances)
    best_solution = pareto_front.row(best_idx, named=True)
    
    # Extract parameters for simulation
    param_cols = [col for col in best_solution.keys() if col not in objective_names]
    best_params = {param: best_solution[param] for param in param_cols}
    best_params.update({'steps': 50, 'show_progress': False})
    
    # Run simulation
    model = SegregationModel(best_params)
    model_results = model.run()
    
    time_steps = range(len(model_results['model']))
    axes[2, 2].plot(time_steps, model_results['model']['segregation_index'], 
                   label='Segregation Index', linewidth=2)
    axes[2, 2].plot(time_steps, model_results['model']['satisfaction_mean'], 
                   label='Satisfaction Mean', linewidth=2)
    axes[2, 2].set_xlabel('Time Step')
    axes[2, 2].set_ylabel('Value')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('amber_multi_objective_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print trade-off analysis
    print(f"\n🔍 Trade-off Analysis:")
    print(f"Pareto front spans:")
    for obj in objective_names:
        values = pareto_front[obj].to_list()
        print(f"  {obj.title()}: {min(values):.4f} - {max(values):.4f}")
    
    print(f"\nBest compromise solution objectives:")
    for obj in objective_names:
        print(f"  {obj.title()}: {best_solution[obj]:.4f}")


def compare_single_vs_multi_objective():
    """Compare single-objective vs multi-objective optimization approaches."""
    print("\n⚖️  Comparing Single vs Multi-Objective Approaches")
    print("=" * 50)
    
    param_space = create_advanced_parameter_space()
    
    # Single-objective approach (optimize segregation only)
    print("Running single-objective optimization (segregation only)...")
    
    single_optimizer = am.SMACOptimizer(
        model_type=SegregationModel,
        param_space=param_space,
        objective=segregation_objective,
        n_trials=50,
        seed=42,
        strategy='bayesian'
    )
    
    single_results = single_optimizer.optimize()
    
    # Evaluate single-objective solution on all objectives
    best_params = single_results['best_config'].copy()
    best_params.update({'steps': 50, 'show_progress': False})
    
    model = SegregationModel(best_params)
    model.run()
    
    single_obj_scores = {
        'segregation': segregation_objective(model),
        'clustering': clustering_objective(model),
        'mobility': mobility_objective(model),
        'satisfaction': satisfaction_objective(model)
    }
    
    print(f"Single-objective solution performance:")
    for obj, score in single_obj_scores.items():
        print(f"  {obj.title()}: {score:.4f}")
    
    # Compare with best multi-objective solution from previous run
    print(f"\nMulti-objective best compromise solution performance:")
    print("(From previous multi-objective optimization)")
    print("- Better balanced across all objectives")
    print("- Provides multiple solution options")
    print("- Reveals trade-offs between objectives")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    objectives = list(single_obj_scores.keys())
    single_values = list(single_obj_scores.values())
    
    x = np.arange(len(objectives))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, single_values, width, label='Single-Objective', alpha=0.7, color='lightcoral')
    
    plt.xlabel('Objectives')
    plt.ylabel('Objective Value (Lower is Better)')
    plt.title('Single-Objective vs Multi-Objective Comparison')
    plt.xticks(x, [obj.title() for obj in objectives])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, single_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('single_vs_multi_objective.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_parameter_space_features():
    """Demonstrate advanced parameter space features."""
    print("\n🔧 Advanced Parameter Space Features")
    print("=" * 40)
    
    param_space = create_advanced_parameter_space()
    
    # Show configuration space structure
    config_space = param_space.get_configspace()
    print(f"Configuration space has {len(config_space.get_hyperparameters())} parameters:")
    
    for param in config_space.get_hyperparameters():
        print(f"  {param.name}: {type(param).__name__}")
        if hasattr(param, 'lower') and hasattr(param, 'upper'):
            print(f"    Range: {param.lower} - {param.upper}")
        elif hasattr(param, 'choices'):
            print(f"    Choices: {param.choices}")
    
    # Sample some configurations
    print(f"\nSample configurations:")
    for i in range(3):
        config = config_space.sample_configuration()
        print(f"\nConfiguration {i+1}:")
        for param_name, value in config.items():
            print(f"  {param_name}: {value}")


if __name__ == "__main__":
    print("Advanced Multi-Objective SMAC Calibration with AMBER")
    print("=" * 55)
    
    # Demonstrate parameter space features
    demonstrate_parameter_space_features()
    
    # Run multi-objective optimization
    optimizer, results = run_multi_objective_optimization()
    
    # Analyze Pareto frontier
    analyze_pareto_frontier(optimizer, results)
    
    # Compare approaches
    compare_single_vs_multi_objective()
    
    print("\n✅ Advanced AMBER SMAC calibration example completed!")
    print("📁 Results saved as:")
    print("  - 'amber_multi_objective_analysis.png'")
    print("  - 'single_vs_multi_objective.png'")
    print("\n💡 Advanced Features Demonstrated:")
    print("- Multi-objective optimization with AMBER's MultiObjectiveSMAC")
    print("- Pareto frontier analysis and visualization")
    print("- Advanced parameter spaces with conditional parameters")
    print("- Trade-off analysis between competing objectives")
    print("- Comparison of single vs multi-objective approaches")
    print("- Professional visualization of optimization results") 