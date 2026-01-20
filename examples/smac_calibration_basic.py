#!/usr/bin/env python3
"""
SMAC Calibration Example - Using AMBER's Built-in Optimization
=============================================================

This example demonstrates how to use AMBER's built-in SMACOptimizer class
to calibrate parameters in agent-based models. AMBER provides a convenient
wrapper around SMAC with specialized features for agent-based modeling.

Key Features:
- SMACParameterSpace for defining parameter ranges
- SMACOptimizer with multiple strategies and acquisition functions
- Built-in objective function handling
- Automatic result analysis and visualization

Requirements:
    pip install smac ConfigSpace
"""

import amber as am
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


class WealthTransferModel(am.Model):
    """A wealth transfer model for SMAC optimization."""
    
    def setup(self):
        """Initialize the model with agents."""
        # Create agents with initial wealth
        for i in range(self.p['n_agents']):
            agent = am.Agent(self, i)
            agent.wealth = self.nprandom.randint(1, 10)
            self.add_agent(agent)
    
    def step(self):
        """Execute one simulation step."""
        # Each agent potentially transfers wealth based on model parameters
        for agent_id in range(self.p['n_agents']):
            agent_data = self.get_agent_data(agent_id)
            current_wealth = agent_data['wealth'].item()
            
            # Transfer probability based on wealth and model parameters
            transfer_prob = self.p['base_transfer_rate'] * (current_wealth / 10.0) ** self.p['wealth_exponent']
            
            if current_wealth > 0 and self.nprandom.random() < transfer_prob:
                # Choose recipient
                recipient_id = self.nprandom.randint(0, self.p['n_agents'])
                if recipient_id != agent_id:
                    # Transfer amount based on generosity parameter
                    transfer_amount = max(1, int(current_wealth * self.p['generosity_factor']))
                    transfer_amount = min(transfer_amount, current_wealth)
                    
                    # Execute transfer
                    self.update_agent_data(agent_id, {
                        'wealth': current_wealth - transfer_amount
                    })
                    recipient_data = self.get_agent_data(recipient_id)
                    self.update_agent_data(recipient_id, {
                        'wealth': recipient_data['wealth'].item() + transfer_amount
                    })
    
    def update(self):
        """Update model-level statistics."""
        super().update()
        if self.t > 0:  # Only record after actual steps
            # Calculate wealth statistics
            wealth_values = self.agents_df['wealth'].to_list()
            total_wealth = sum(wealth_values)
            mean_wealth = np.mean(wealth_values)
            wealth_std = np.std(wealth_values)
            
            # Calculate Gini coefficient (inequality measure)
            gini = self.calculate_gini(wealth_values)
            
            # Record model metrics
            self.record_model('total_wealth', total_wealth)
            self.record_model('mean_wealth', mean_wealth)
            self.record_model('wealth_std', wealth_std)
            self.record_model('gini_coefficient', gini)
            self.record_model('wealth_concentration', self.calculate_concentration(wealth_values))
    
    def calculate_gini(self, wealth_list):
        """Calculate Gini coefficient of wealth inequality."""
        if not wealth_list or sum(wealth_list) == 0:
            return 0.0
        
        sorted_wealth = sorted(wealth_list)
        n = len(sorted_wealth)
        cumsum = np.cumsum(sorted_wealth)
        return (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n
    
    def calculate_concentration(self, wealth_list):
        """Calculate wealth concentration (top 20% share)."""
        if not wealth_list:
            return 0.0
        
        sorted_wealth = sorted(wealth_list, reverse=True)
        top_20_percent = int(0.2 * len(sorted_wealth))
        if top_20_percent == 0:
            top_20_percent = 1
        
        top_wealth = sum(sorted_wealth[:top_20_percent])
        total_wealth = sum(sorted_wealth)
        
        return top_wealth / total_wealth if total_wealth > 0 else 0.0


def create_parameter_space():
    """Create parameter space using AMBER's SMACParameterSpace."""
    param_space = am.SMACParameterSpace()
    
    # Add continuous parameters
    param_space.add_parameter(
        'base_transfer_rate', 
        param_type='float', 
        bounds=(0.01, 0.5), 
        default=0.1
    )
    
    param_space.add_parameter(
        'wealth_exponent', 
        param_type='float', 
        bounds=(0.1, 2.0), 
        default=1.0
    )
    
    param_space.add_parameter(
        'generosity_factor', 
        param_type='float', 
        bounds=(0.01, 0.3), 
        default=0.1
    )
    
    # Add integer parameter
    param_space.add_parameter(
        'n_agents', 
        param_type='int', 
        bounds=(50, 200), 
        default=100
    )
    
    return param_space


def objective_function(model: WealthTransferModel) -> float:
    """
    Objective function for SMAC optimization.
    
    We want to find parameters that create a specific wealth distribution:
    - Moderate inequality (Gini coefficient around 0.35)
    - Realistic wealth concentration (around 0.6)
    
    Args:
        model: Completed model instance
    
    Returns:
        Objective value to minimize (lower is better)
    """
    results = model.results
    
    # Extract final metrics
    final_gini = results['model']['gini_coefficient'].tail(1).item()
    final_concentration = results['model']['wealth_concentration'].tail(1).item()
    
    # Define target values
    target_gini = 0.35
    target_concentration = 0.6
    
    # Calculate penalties for deviation from targets
    gini_penalty = (final_gini - target_gini) ** 2
    concentration_penalty = (final_concentration - target_concentration) ** 2
    
    # Combined objective (minimize total penalty)
    objective = gini_penalty + concentration_penalty
    
    print(f"Gini: {final_gini:.3f} (target: {target_gini}), "
          f"Concentration: {final_concentration:.3f} (target: {target_concentration})")
    print(f"Objective: {objective:.6f}")
    
    return objective


def run_smac_optimization():
    """Run SMAC optimization using AMBER's SMACOptimizer."""
    print("🚀 Starting SMAC Calibration with AMBER's SMACOptimizer")
    print("=" * 60)
    
    # Create parameter space
    param_space = create_parameter_space()
    
    # Create SMAC optimizer
    optimizer = am.SMACOptimizer(
        model_type=WealthTransferModel,
        param_space=param_space,
        objective=objective_function,
        n_trials=50,
        seed=42,
        strategy='bayesian',  # Use Bayesian optimization
        acquisition_function='ei',  # Expected Improvement
        initial_design='latin_hypercube',  # Latin Hypercube Sampling
        surrogate_model='random_forest'  # Random Forest surrogate
    )
    
    # Add fixed parameters that don't need optimization
    fixed_params = {
        'steps': 100,
        'show_progress': False
    }
    
    # Run optimization
    print("Starting optimization...")
    results = optimizer.optimize()
    
    # Display results
    print("\n🎯 Optimization Results:")
    print("=" * 40)
    print(f"Best configuration found:")
    best_config = results['best_config']
    for param, value in best_config.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest objective value: {results['best_objective']:.6f}")
    print(f"Total function evaluations: {results['n_evaluations']}")
    
    return optimizer, results


def analyze_optimization_results(optimizer, results):
    """Analyze and visualize optimization results."""
    print("\n📊 Analyzing Optimization Results...")
    
    # Get optimization history
    history = results['history']
    
    # Extract data for plotting
    objectives = history['objective'].to_list()
    configs = history.drop(['objective', 'trial']).to_dict('records')
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Objective value over time
    axes[0, 0].plot(objectives, 'b-', alpha=0.7, linewidth=2)
    axes[0, 0].axhline(y=min(objectives), color='r', linestyle='--', alpha=0.7, label='Best found')
    axes[0, 0].set_xlabel('Trial')
    axes[0, 0].set_ylabel('Objective Value')
    axes[0, 0].set_title('Optimization Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2-4: Parameter evolution
    param_names = ['base_transfer_rate', 'wealth_exponent', 'generosity_factor']
    
    for i, param in enumerate(param_names):
        row, col = (0, i + 1) if i < 2 else (1, i - 2)
        param_values = [config[param] for config in configs]
        
        scatter = axes[row, col].scatter(
            range(len(param_values)), param_values, 
            alpha=0.6, c=objectives, cmap='viridis', s=50
        )
        axes[row, col].set_xlabel('Trial')
        axes[row, col].set_ylabel(param.replace('_', ' ').title())
        axes[row, col].set_title(f'{param.replace("_", " ").title()} Evolution')
        plt.colorbar(scatter, ax=axes[row, col], label='Objective Value')
        axes[row, col].grid(True, alpha=0.3)
    
    # Plot 5: Best configuration simulation
    axes[1, 1].set_title('Best Configuration Dynamics')
    
    # Run best configuration for detailed analysis
    best_params = results['best_config'].copy()
    best_params.update({'steps': 100, 'show_progress': False})
    
    model = WealthTransferModel(best_params)
    model_results = model.run()
    
    time_steps = range(len(model_results['model']))
    axes[1, 1].plot(time_steps, model_results['model']['gini_coefficient'], 
                   label='Gini Coefficient', linewidth=2)
    axes[1, 1].plot(time_steps, model_results['model']['wealth_concentration'], 
                   label='Wealth Concentration', linewidth=2)
    axes[1, 1].axhline(y=0.35, color='r', linestyle='--', alpha=0.7, label='Target Gini')
    axes[1, 1].axhline(y=0.6, color='g', linestyle='--', alpha=0.7, label='Target Concentration')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Final wealth distribution
    axes[1, 2].set_title('Final Wealth Distribution')
    final_wealth = model_results['agents'].filter(
        model_results['agents']['step'] == model_results['agents']['step'].max()
    )['wealth'].to_list()
    
    axes[1, 2].hist(final_wealth, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    axes[1, 2].set_xlabel('Wealth')
    axes[1, 2].set_ylabel('Number of Agents')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('amber_smac_calibration_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"Total trials: {len(objectives)}")
    print(f"Best objective: {min(objectives):.6f}")
    print(f"Worst objective: {max(objectives):.6f}")
    print(f"Average objective: {np.mean(objectives):.6f}")
    print(f"Improvement: {(max(objectives) - min(objectives)):.6f}")
    print(f"Final Gini coefficient: {model_results['model']['gini_coefficient'].tail(1).item():.3f}")
    print(f"Final wealth concentration: {model_results['model']['wealth_concentration'].tail(1).item():.3f}")


def compare_optimization_strategies():
    """Compare different SMAC optimization strategies."""
    print("\n🔍 Comparing Optimization Strategies")
    print("=" * 40)
    
    param_space = create_parameter_space()
    strategies = ['bayesian', 'random']
    acquisition_functions = ['ei', 'lcb', 'pi']
    
    comparison_results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")
        
        if strategy == 'bayesian':
            # Test different acquisition functions for Bayesian optimization
            for acq_func in acquisition_functions:
                print(f"  Using {acq_func} acquisition function...")
                
                optimizer = am.SMACOptimizer(
                    model_type=WealthTransferModel,
                    param_space=param_space,
                    objective=objective_function,
                    n_trials=20,  # Fewer trials for comparison
                    seed=42,
                    strategy=strategy,
                    acquisition_function=acq_func,
                    surrogate_model='random_forest'
                )
                
                results = optimizer.optimize()
                comparison_results[f'{strategy}_{acq_func}'] = results['best_objective']
                print(f"    Best objective: {results['best_objective']:.6f}")
        
        else:  # random strategy
            optimizer = am.SMACOptimizer(
                model_type=WealthTransferModel,
                param_space=param_space,
                objective=objective_function,
                n_trials=20,
                seed=42,
                strategy=strategy
            )
            
            results = optimizer.optimize()
            comparison_results[strategy] = results['best_objective']
            print(f"  Best objective: {results['best_objective']:.6f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    methods = list(comparison_results.keys())
    values = list(comparison_results.values())
    
    bars = plt.bar(methods, values, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.xlabel('Optimization Method')
    plt.ylabel('Best Objective Value (Lower is Better)')
    plt.title('Comparison of SMAC Optimization Strategies')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('smac_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find best strategy
    best_method = min(comparison_results.items(), key=lambda x: x[1])
    print(f"\n🏆 Best performing method: {best_method[0]} (objective: {best_method[1]:.6f})")


def demonstrate_parameter_importance():
    """Demonstrate parameter importance analysis using SMAC results."""
    print("\n🔬 Parameter Importance Analysis")
    print("=" * 35)
    
    # Run optimization with more trials for better importance analysis
    param_space = create_parameter_space()
    
    optimizer = am.SMACOptimizer(
        model_type=WealthTransferModel,
        param_space=param_space,
        objective=objective_function,
        n_trials=100,
        seed=42,
        strategy='bayesian',
        acquisition_function='ei'
    )
    
    results = optimizer.optimize()
    history = results['history']
    
    # Calculate parameter importance based on correlation with objective
    param_names = ['base_transfer_rate', 'wealth_exponent', 'generosity_factor', 'n_agents']
    importances = {}
    
    for param in param_names:
        param_values = history[param].to_list()
        objectives = history['objective'].to_list()
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(param_values, objectives)[0, 1]
        importances[param] = abs(correlation)  # Use absolute value
    
    # Sort by importance
    sorted_importance = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    print("Parameter importance (based on correlation with objective):")
    for param, importance in sorted_importance:
        print(f"  {param.replace('_', ' ').title()}: {importance:.3f}")
    
    # Visualize parameter importance
    plt.figure(figsize=(10, 6))
    params, values = zip(*sorted_importance)
    
    bars = plt.barh(params, values, alpha=0.7, color='lightblue')
    plt.xlabel('Importance Score (Absolute Correlation)')
    plt.ylabel('Parameter')
    plt.title('Parameter Importance Analysis')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('parameter_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("SMAC Calibration with AMBER's Built-in Optimization")
    print("=" * 55)
    
    # Run main SMAC optimization
    optimizer, results = run_smac_optimization()
    
    # Analyze results
    analyze_optimization_results(optimizer, results)
    
    # Compare different strategies
    compare_optimization_strategies()
    
    # Demonstrate parameter importance
    demonstrate_parameter_importance()
    
    print("\n✅ AMBER SMAC calibration example completed!")
    print("📁 Results saved as:")
    print("  - 'amber_smac_calibration_results.png'")
    print("  - 'smac_strategy_comparison.png'")
    print("  - 'parameter_importance.png'")
    print("\n💡 Key Advantages of AMBER's SMACOptimizer:")
    print("- Seamless integration with AMBER models")
    print("- Built-in parameter space definition")
    print("- Automatic objective function handling")
    print("- Multiple optimization strategies")
    print("- Comprehensive result analysis")
    print("- Easy comparison of different approaches") 