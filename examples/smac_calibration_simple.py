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
    pip install 'ambr[advanced]'          # SMAC + ConfigSpace
    pip install 'ambr[advanced,viz]'      # plus matplotlib for plots

Default ``__main__`` is a short search (10 trials x 15 steps). Pass
``--full`` for SMAC vs random comparison.
"""

import argparse

import ambr as am
import numpy as np


_VIZ_WARNED = False


def _try_matplotlib():
    """Return pyplot, or None if matplotlib is missing / ABI-incompatible.

    Search completes without viz extras. Plots need ``ambr[advanced,viz]``.
    """
    global _VIZ_WARNED
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        if not _VIZ_WARNED:
            print(
                "Skipping plots — install visualization extras:\n"
                "  pip install 'ambr[advanced,viz]'\n"
                f"Underlying error: {exc!r}"
            )
            _VIZ_WARNED = True
        return None


class SimpleWealthModel(am.Model):
    """A very simple wealth transfer model, written with the vectorized view API."""

    def setup(self):
        """Create agents with random initial wealth — one columnar call."""
        n = int(self.p['n_agents'])
        self.add_agents(n, wealth=self.rng.integers(1, 100, size=n))

    def step(self):
        """Simple wealth transfer step, fully columnar."""
        n = int(self.p['n_agents'])
        wealth = self.agents.wealth.to_numpy()

        # Each agent transfers with probability p['transfer_rate'] and only
        # if it has something to give.
        active_mask = (wealth > 0) & (self.rng.random(size=n) < self.p['transfer_rate'])
        if not active_mask.any():
            return

        amount = np.maximum(1, (wealth * self.p['transfer_fraction']).astype(int))
        amount = np.minimum(amount, wealth) * active_mask

        donor_ids = self.agents.ids.to_numpy()[active_mask]
        donor_amounts = amount[active_mask]
        self.agents.at[donor_ids].scatter_add(wealth=-donor_amounts)

        recipient_ids = self.rng.choice(self.agents.ids.to_numpy(), size=int(active_mask.sum()))
        self.agents.at[recipient_ids].scatter_add(wealth=donor_amounts)

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


def run_simple_optimization(n_trials: int = 10, steps: int = 15, seed: int = 42):
    """Run a simple SMAC optimization."""
    print("Simple SMAC Calibration Example")
    print("=" * 35)
    print(f"Budget: n_trials={n_trials}, steps={steps} (pass --full for comparison)")

    # Step 1: Create parameter space
    param_space = create_simple_parameter_space()
    print("Parameter space created")

    # Step 2: Create optimizer (fixed_params: non-search model knobs)
    optimizer = am.SMACOptimizer(
        model_type=SimpleWealthModel,
        param_space=param_space,
        objective=simple_objective,
        n_trials=n_trials,
        seed=seed,
        fixed_params={
            "n_agents": 100,
            "steps": steps,
            "seed": seed,
            "show_progress": False,
        },
    )
    print("Optimizer created")

    # Step 3: Run optimization
    print("\nRunning optimization...")
    results = optimizer.optimize()

    # Step 4: Show results
    print("\nResults:")
    print(f"SMAC incumbent cost: {results['best_objective']:.4f}")
    print("\nBest parameters:")
    for param, value in results['best_config'].items():
        print(f"  {param}: {value:.4f}")

    return optimizer, results


def analyze_simple_results(optimizer, results):
    """Analyze and visualize the simple optimization results."""
    print("\nAnalysis:")

    # History columns: search-space knobs + cost/objective, time, trial
    history = results['history']
    objectives = history['cost'].to_list()  # same as history['objective']
    print(f"n_evaluations: {results['n_evaluations']}")
    print(f"SMAC incumbent cost: {results.get('best_objective'):.4f}")
    print(f"Minimum trial cost (history): {min(objectives):.4f}")
    print(f"Started with cost: {objectives[0]:.4f}")

    # Replay the incumbent with the same pinned model seed as search.
    best_params = {
        "n_agents": 100,
        "steps": int(optimizer.fixed_params.get("steps", 15)),
        "seed": int(optimizer.fixed_params.get("seed", 42)),
        "show_progress": False,
        **results["best_config"],
    }

    print("\nTesting best configuration (same seed as search)...")
    model = SimpleWealthModel(best_params)
    model_results = model.run()

    final_gini = model_results['model']['gini_coefficient'].tail(1).item()
    print(f"Final Gini coefficient: {final_gini:.4f} (target: 0.4)")

    # Simple visualization (optional: ambr[advanced,viz])
    plt = _try_matplotlib()
    if plt is None:
        return

    plt.figure(figsize=(12, 4))

    # Plot 1: Optimization progress
    plt.subplot(1, 3, 1)
    plt.plot(objectives, 'b-o', markersize=4)
    plt.axhline(y=min(objectives), color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Trial')
    plt.ylabel('Cost (objective)')
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

    # Plot 3: Final wealth distribution (results.agents is end-of-run state)
    plt.subplot(1, 3, 3)
    final_wealth = model_results.agents['wealth'].to_list()

    plt.hist(final_wealth, bins=15, alpha=0.7, edgecolor='black', color='lightblue')
    plt.xlabel('Wealth')
    plt.ylabel('Number of Agents')
    plt.title('Final Wealth Distribution')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simple_smac_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_with_random_search(n_trials: int = 20, steps: int = 50, seed: int = 42):
    """Compare SMAC with simple random search."""
    print("\nComparing SMAC vs Random Search")
    print("=" * 35)

    param_space = create_simple_parameter_space()

    fixed = {"n_agents": 100, "steps": steps, "seed": seed, "show_progress": False}

    # SMAC Bayesian vs RandomFacade (strategy='random')
    smac_optimizer = am.SMACOptimizer(
        model_type=SimpleWealthModel,
        param_space=param_space,
        objective=simple_objective,
        n_trials=n_trials,
        seed=seed,
        strategy="bayesian",
        fixed_params=fixed,
    )

    smac_results = smac_optimizer.optimize()
    smac_best = smac_results['best_objective']

    random_optimizer = am.SMACOptimizer(
        model_type=SimpleWealthModel,
        param_space=param_space,
        objective=simple_objective,
        n_trials=n_trials,
        seed=seed,
        strategy="random",  # RandomFacade (not Bayesian)
        fixed_params=fixed,
    )

    random_results = random_optimizer.optimize()
    random_best = random_results['best_objective']

    print(f"SMAC best objective:   {smac_best:.4f}")
    print(f"Random best objective: {random_best:.4f}")

    if smac_best < random_best:
        improvement = ((random_best - smac_best) / random_best) * 100
        print(f"SMAC is {improvement:.1f}% better.")
    else:
        print("Random search performed similarly (this can happen with simple problems)")

    # Visualize comparison (optional: ambr[advanced,viz])
    plt = _try_matplotlib()
    if plt is None:
        return

    plt.figure(figsize=(10, 4))

    # Plot optimization curves (history uses cost; objective is an alias)
    plt.subplot(1, 2, 1)
    smac_objectives = smac_results['history']['cost'].to_list()
    random_objectives = random_results['history']['cost'].to_list()

    plt.plot(smac_objectives, 'b-o', label='SMAC (Bayesian)', markersize=4)
    plt.plot(random_objectives, 'r-s', label='Random Search', markersize=4)
    plt.xlabel('Trial')
    plt.ylabel('Cost (objective)')
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
    plt.colorbar(label='Cost')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('smac_vs_random.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMBER simple SMAC calibration demo")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run SMAC vs random comparison (extra 2 x n_trials evaluations)",
    )
    args = parser.parse_args()

    print("Simple SMAC Calibration with AMBER")
    print("=" * 35)

    # Smoke: model alone (no SMAC / matplotlib) so the example is runnable without extras.
    smoke = SimpleWealthModel({
        'n_agents': 80,
        'transfer_rate': 0.2,
        'transfer_fraction': 0.1,
        'steps': 15,
        'seed': 0,
        'show_progress': False,
    })
    smoke_res = smoke.run()
    print(
        "Smoke run OK:",
        smoke_res['info'],
        "metrics=",
        smoke_res['model'].columns,
    )

    try:
        import ConfigSpace  # noqa: F401
        import smac  # noqa: F401
    except ImportError:
        print("smac/ConfigSpace not installed — skipping SMAC optimization section")
        print("Install with: pip install 'ambr[advanced]'  # or: pip install smac ConfigSpace")
        raise SystemExit(0)

    print("This example optimizes a simple wealth transfer model for moderate inequality.")
    if _try_matplotlib() is None:
        print("Preflight: matplotlib missing — optimization will run; plots skipped.")
        print("For plots: pip install 'ambr[advanced,viz]'")
    if args.full:
        optimizer, results = run_simple_optimization(n_trials=20, steps=50, seed=42)
        analyze_simple_results(optimizer, results)
        compare_with_random_search(n_trials=20, steps=50, seed=42)
    else:
        optimizer, results = run_simple_optimization(n_trials=10, steps=15, seed=42)
        analyze_simple_results(optimizer, results)
        print("Skipped SMAC vs random comparison (pass --full).")

    print("\nSimple SMAC example completed.")
    print("PNG outputs written when matplotlib is available.")
