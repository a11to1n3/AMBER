#!/usr/bin/env python3
"""
Simple SMAC Calibration Example
==============================

This example shows the simplest way to use AMBER's built-in SMAC optimization
to calibrate a basic agent-based model. Perfect for getting started!

Key Features:
- Simple parameter space definition
- Basic objective function
- Quick optimization run
- Easy result interpretation

Requirements:
    pip install smac ConfigSpace
"""

import ambr as am
import numpy as np
import matplotlib.pyplot as plt


class SimpleWealthModel(am.Model):
    """A very simple wealth transfer model for demonstration."""
    
    def setup(self):
        """Create agents with random initial wealth."""
        for i in range(self.p['n_agents']):
            agent = am.Agent(self, i)
            agent.wealth = self.nprandom.randint(1, 100)
            self.add_agent(agent)
    
    def step(self):
        """Simple wealth transfer step."""
        # Each agent may give money to another agent
        for agent_id in range(self.p['n_agents']):
            agent_data = self.get_agent_data(agent_id)
            wealth = agent_data['wealth'].item()
            
            # Transfer probability based on model parameter
            if wealth > 0 and self.nprandom.random() < self.p['transfer_rate']:
                # Give some money to a random other agent
                recipient_id = self.nprandom.randint(0, self.p['n_agents'])
                if recipient_id != agent_id:
                    amount = int(wealth * self.p['transfer_fraction'])
                    amount = max(1, min(amount, wealth))
                    
                    # Execute transfer
                    self.update_agent_data(agent_id, {'wealth': wealth - amount})
                    recipient_data = self.get_agent_data(recipient_id)
                    self.update_agent_data(recipient_id, {
                        'wealth': recipient_data['wealth'].item() + amount
                    })
    
    def update(self):
        """Track wealth inequality."""
        super().update()
        if self.t > 0:
            wealth_values = self.agents_df['wealth'].to_list()
            
            # Calculate Gini coefficient (inequality measure)
            gini = self.calculate_gini(wealth_values)
            self.record_model('gini_coefficient', gini)
            self.record_model('mean_wealth', np.mean(wealth_values))
            self.record_model('std_wealth', np.std(wealth_values))
    
    def calculate_gini(self, wealth_list):
        """Simple Gini coefficient calculation."""
        if not wealth_list or sum(wealth_list) == 0:
            return 0.0
        
        sorted_wealth = sorted(wealth_list)
        n = len(sorted_wealth)
        cumsum = np.cumsum(sorted_wealth)
        return (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n


def create_simple_parameter_space():
    """Create a simple parameter space with just 2 parameters."""
    param_space = am.SMACParameterSpace()
    
    # How often agents transfer money (0.0 to 1.0)
    param_space.add_parameter(
        'transfer_rate',
        param_type='float',
        bounds=(0.01, 0.5),
        default=0.1
    )
    
    # What fraction of wealth they transfer (0.0 to 1.0)
    param_space.add_parameter(
        'transfer_fraction',
        param_type='float',
        bounds=(0.01, 0.3),
        default=0.1
    )
    
    return param_space


def simple_objective(model: SimpleWealthModel) -> float:
    """
    Simple objective: try to achieve a Gini coefficient of 0.4
    (moderate inequality).
    """
    results = model.results
    final_gini = results['model']['gini_coefficient'].tail(1).item()
    target_gini = 0.4
    
    # Return absolute difference (SMAC will minimize this)
    return abs(final_gini - target_gini)


def run_simple_optimization():
    """Run a simple SMAC optimization."""
    print("🚀 Simple SMAC Calibration Example")
    print("=" * 35)
    
    # Step 1: Create parameter space
    param_space = create_simple_parameter_space()
    print("✓ Parameter space created")
    
    # Step 2: Create optimizer
    optimizer = am.SMACOptimizer(
        model_type=SimpleWealthModel,
        param_space=param_space,
        objective=simple_objective,
        n_trials=20,  # Small number for quick demo
        seed=42
    )
    print("✓ Optimizer created")
    
    # Step 3: Run optimization
    print("\n🔍 Running optimization...")
    results = optimizer.optimize()
    
    # Step 4: Show results
    print("\n🎯 Results:")
    print(f"Best objective value: {results['best_objective']:.4f}")
    print("\nBest parameters:")
    for param, value in results['best_config'].items():
        print(f"  {param}: {value:.4f}")
    
    return optimizer, results


def analyze_simple_results(optimizer, results):
    """Analyze and visualize the simple optimization results."""
    print("\n📊 Analysis:")
    
    # Get optimization history
    history = results['history']
    
    # Show improvement over time
    objectives = history['objective'].to_list()
    print(f"Started with objective: {objectives[0]:.4f}")
    print(f"Ended with objective: {min(objectives):.4f}")
    print(f"Improvement: {objectives[0] - min(objectives):.4f}")
    
    # Test the best configuration
    best_params = results['best_config'].copy()
    best_params.update({
        'n_agents': 100,  # Fixed parameter
        'steps': 50,      # Fixed parameter
        'show_progress': False
    })
    
    print(f"\n🧪 Testing best configuration...")
    model = SimpleWealthModel(best_params)
    model_results = model.run()
    
    final_gini = model_results['model']['gini_coefficient'].tail(1).item()
    print(f"Final Gini coefficient: {final_gini:.4f} (target: 0.4)")
    
    # Simple visualization
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Optimization progress
    plt.subplot(1, 3, 1)
    plt.plot(objectives, 'b-o', markersize=4)
    plt.axhline(y=min(objectives), color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Trial')
    plt.ylabel('Objective Value')
    plt.title('Optimization Progress')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Gini coefficient over time
    plt.subplot(1, 3, 2)
    time_steps = range(len(model_results['model']))
    gini_values = model_results['model']['gini_coefficient'].to_list()
    plt.plot(time_steps, gini_values, 'g-', linewidth=2)
    plt.axhline(y=0.4, color='r', linestyle='--', alpha=0.7, label='Target')
    plt.xlabel('Time Step')
    plt.ylabel('Gini Coefficient')
    plt.title('Best Configuration Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Final wealth distribution
    plt.subplot(1, 3, 3)
    final_wealth = model_results['agents'].filter(
        model_results['agents']['step'] == model_results['agents']['step'].max()
    )['wealth'].to_list()
    
    plt.hist(final_wealth, bins=15, alpha=0.7, edgecolor='black', color='lightblue')
    plt.xlabel('Wealth')
    plt.ylabel('Number of Agents')
    plt.title('Final Wealth Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_smac_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_with_random_search():
    """Compare SMAC with simple random search."""
    print("\n🔄 Comparing SMAC vs Random Search")
    print("=" * 35)
    
    param_space = create_simple_parameter_space()
    
    # SMAC optimization (already done above, but let's do a fresh one)
    smac_optimizer = am.SMACOptimizer(
        model_type=SimpleWealthModel,
        param_space=param_space,
        objective=simple_objective,
        n_trials=20,
        seed=42,
        strategy='bayesian'
    )
    
    smac_results = smac_optimizer.optimize()
    smac_best = smac_results['best_objective']
    
    # Random search for comparison
    random_optimizer = am.SMACOptimizer(
        model_type=SimpleWealthModel,
        param_space=param_space,
        objective=simple_objective,
        n_trials=20,
        seed=42,
        strategy='random'  # Use random search instead
    )
    
    random_results = random_optimizer.optimize()
    random_best = random_results['best_objective']
    
    print(f"SMAC best objective:   {smac_best:.4f}")
    print(f"Random best objective: {random_best:.4f}")
    
    if smac_best < random_best:
        improvement = ((random_best - smac_best) / random_best) * 100
        print(f"SMAC is {improvement:.1f}% better! 🎉")
    else:
        print("Random search performed similarly (this can happen with simple problems)")
    
    # Visualize comparison
    plt.figure(figsize=(10, 4))
    
    # Plot optimization curves
    plt.subplot(1, 2, 1)
    smac_objectives = smac_results['history']['objective'].to_list()
    random_objectives = random_results['history']['objective'].to_list()
    
    plt.plot(smac_objectives, 'b-o', label='SMAC (Bayesian)', markersize=4)
    plt.plot(random_objectives, 'r-s', label='Random Search', markersize=4)
    plt.xlabel('Trial')
    plt.ylabel('Objective Value')
    plt.title('SMAC vs Random Search')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot parameter exploration
    plt.subplot(1, 2, 2)
    smac_transfer_rates = smac_results['history']['transfer_rate'].to_list()
    smac_transfer_fractions = smac_results['history']['transfer_fraction'].to_list()
    
    random_transfer_rates = random_results['history']['transfer_rate'].to_list()
    random_transfer_fractions = random_results['history']['transfer_fraction'].to_list()
    
    plt.scatter(smac_transfer_rates, smac_transfer_fractions, 
               c=smac_objectives, cmap='Blues', alpha=0.7, label='SMAC', s=50)
    plt.scatter(random_transfer_rates, random_transfer_fractions,
               c=random_objectives, cmap='Reds', alpha=0.7, label='Random', s=50, marker='s')
    
    plt.xlabel('Transfer Rate')
    plt.ylabel('Transfer Fraction')
    plt.title('Parameter Space Exploration')
    plt.legend()
    plt.colorbar(label='Objective Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('smac_vs_random.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Simple SMAC Calibration with AMBER")
    print("=" * 35)
    print("This example shows the easiest way to get started with")
    print("SMAC optimization in AMBER. We'll optimize a simple")
    print("wealth transfer model to achieve moderate inequality.")
    print()
    
    # Run the optimization
    optimizer, results = run_simple_optimization()
    
    # Analyze results
    analyze_simple_results(optimizer, results)
    
    # Compare with random search
    compare_with_random_search()
    
    print("\n✅ Simple SMAC example completed!")
    print("📁 Results saved as:")
    print("  - 'simple_smac_results.png'")
    print("  - 'smac_vs_random.png'")
    print("\n💡 Key Takeaways:")
    print("- SMAC optimization is easy to set up with AMBER")
    print("- Just define parameters, objective, and run optimize()")
    print("- SMAC often outperforms random search")
    print("- Great for finding good model parameters automatically")
    print("\n🎓 Next Steps:")
    print("- Try smac_calibration_basic.py for more features")
    print("- Try smac_calibration_advanced.py for multi-objective optimization") 