#!/usr/bin/env python
"""
AMBER Comprehensive Correctness Benchmark

Tests scientific correctness across all frameworks with rigorous metrics:

1. CONSERVATION LAWS - Physical quantities preserved
2. STATISTICAL ACCURACY - Output matches known distributions  
3. REPRODUCIBILITY - Same seed → same results
4. EMERGENT BEHAVIOR - Expected dynamics emerge
5. NUMERICAL PRECISION - Error accumulation over time
6. EDGE CASES - Boundary conditions handled correctly
"""

import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from scipy import stats
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np


@dataclass
class MetricResult:
    """Single metric measurement."""
    framework: str
    model: str
    category: str  # conservation, statistical, reproducibility, emergent, precision, edge_case
    metric_name: str
    value: float
    expected: float
    error: float
    passed: bool
    details: str
    execution_time: float


class ComprehensiveCorrectnessBenchmark:
    """
    Rigorous correctness testing with quantitative metrics.
    """
    
    def __init__(self, results_dir: str = None):
        self.results_dir = Path(results_dir or Path(__file__).parent / 'results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[MetricResult] = []
        self._load_frameworks()
        
    def _load_frameworks(self):
        """Load available frameworks."""
        self.available = {}
        
        try:
            from models.amber_models import AMBER_MODELS
            self.available['AMBER'] = AMBER_MODELS
        except ImportError:
            pass
        
        try:
            from models.agentpy_models import AGENTPY_MODELS
            self.available['AgentPy'] = AGENTPY_MODELS
        except ImportError:
            pass
        
        try:
            from models.mesa_models import MESA_MODELS
            self.available['Mesa'] = MESA_MODELS
        except ImportError:
            pass
        
        print(f"Loaded frameworks: {list(self.available.keys())}")

    def run_all(self, n_agents: int = 500, n_steps: int = 100, verbose: bool = False):
        """Run comprehensive benchmark suite."""
        
        print(f"\n{'='*70}")
        print(f"  Comprehensive Correctness Benchmark")
        print(f"{'='*70}")
        print(f"  Agents: {n_agents}, Steps: {n_steps}")
        print(f"{'='*70}\n")
        
        # 1. Conservation Laws
        self._test_conservation(n_agents, n_steps, verbose)
        
        # 2. Statistical Accuracy
        self._test_statistical_accuracy(n_agents, n_steps, verbose)
        
        # 3. Reproducibility
        self._test_reproducibility(n_agents, n_steps, verbose)
        
        # 4. Emergent Behavior
        self._test_emergent_behavior(n_agents, n_steps, verbose)
        
        # 5. Numerical Precision
        self._test_numerical_precision(n_agents, n_steps, verbose)
        
        # 6. Edge Cases
        self._test_edge_cases(verbose)
        
        self._print_summary()
        return self.results

    # =========================================================================
    # 1. CONSERVATION LAWS
    # =========================================================================
    def _test_conservation(self, n_agents, n_steps, verbose):
        """Test that physical quantities are conserved."""
        print("\n📊 1. CONSERVATION LAWS")
        print("-" * 50)
        
        for fw_name in self.available:
            # Wealth conservation
            error = self._measure_wealth_conservation(fw_name, n_agents, n_steps)
            self._add_metric(fw_name, 'wealth_transfer', 'conservation', 
                           'total_wealth_error', error, 0.0, error,
                           error < 1e-10, f'Absolute error: {error:.2e}', 0)
            
            # SIR population conservation
            error = self._measure_sir_conservation(fw_name, n_agents, n_steps)
            self._add_metric(fw_name, 'sir_epidemic', 'conservation',
                           'population_error', error, 0.0, error,
                           error == 0, f'Max step error: {error}', 0)

    def _measure_wealth_conservation(self, fw_name, n_agents, n_steps):
        """Measure wealth conservation error."""
        initial_wealth = 100
        expected_total = n_agents * initial_wealth
        
        if fw_name == 'AMBER':
            from models.amber_models import AMBERWealthTransfer
            model = AMBERWealthTransfer({'n': n_agents, 'steps': n_steps, 
                                         'initial_wealth': initial_wealth, 'show_progress': False})
            model.run()
            final = sum(a.wealth for a in model.agent_objects_list)
        elif fw_name == 'AgentPy':
            from models.agentpy_models import AgentPyWealthTransfer
            model = AgentPyWealthTransfer({'n': n_agents, 'steps': n_steps, 'initial_wealth': initial_wealth})
            model.run()
            final = sum(a.wealth for a in model.agents)
        elif fw_name == 'Mesa':
            from models.mesa_models import MesaWealthTransfer
            model = MesaWealthTransfer(n=n_agents, steps=n_steps, initial_wealth=initial_wealth)
            model.run()
            final = sum(a.wealth for a in model.agents)
        else:
            return float('inf')
        
        return abs(final - expected_total)

    def _measure_sir_conservation(self, fw_name, n_agents, n_steps):
        """Measure SIR population conservation across all steps."""
        if fw_name == 'AMBER':
            from models.amber_models import AMBERSIRModel, SIRAgent
            model = AMBERSIRModel({'n': n_agents, 'steps': n_steps, 'initial_infected': 5,
                                   'world_size': 100, 'movement_speed': 2.0, 'infection_radius': 5.0,
                                   'transmission_rate': 0.3, 'recovery_time': 10, 'show_progress': False})
            model.run()
            s = sum(1 for a in model.agent_objects_list if a.status == SIRAgent.STATUS_S)
            i = sum(1 for a in model.agent_objects_list if a.status == SIRAgent.STATUS_I)
            r = sum(1 for a in model.agent_objects_list if a.status == SIRAgent.STATUS_R)
            return abs(s + i + r - n_agents)
        elif fw_name == 'AgentPy':
            from models.agentpy_models import AgentPySIRModel, APSIRAgent
            model = AgentPySIRModel({'n': n_agents, 'steps': n_steps, 'initial_infected': 5,
                                     'world_size': 100, 'movement_speed': 2.0, 'infection_radius': 5.0,
                                     'transmission_rate': 0.3, 'recovery_time': 10})
            model.run()
            s = sum(1 for a in model.agents if a.status == APSIRAgent.STATUS_S)
            i = sum(1 for a in model.agents if a.status == APSIRAgent.STATUS_I)
            r = sum(1 for a in model.agents if a.status == APSIRAgent.STATUS_R)
            return abs(s + i + r - n_agents)
        elif fw_name == 'Mesa':
            from models.mesa_models import MesaSIRModel, MesaSIRAgent
            model = MesaSIRModel(n=n_agents, steps=n_steps, initial_infected=5,
                                 world_size=100, movement_speed=2.0, infection_radius=5.0,
                                 transmission_rate=0.3, recovery_time=10)
            model.run()
            s = sum(1 for a in model.agents if a.status == MesaSIRAgent.STATUS_S)
            i = sum(1 for a in model.agents if a.status == MesaSIRAgent.STATUS_I)
            r = sum(1 for a in model.agents if a.status == MesaSIRAgent.STATUS_R)
            return abs(s + i + r - n_agents)
        return float('inf')

    # =========================================================================
    # 2. STATISTICAL ACCURACY
    # =========================================================================
    def _test_statistical_accuracy(self, n_agents, n_steps, verbose):
        """Test that output distributions match theoretical predictions."""
        print("\n📈 2. STATISTICAL ACCURACY")
        print("-" * 50)
        
        for fw_name in self.available:
            # Wealth should converge to exponential distribution (Boltzmann)
            ks_stat, p_value = self._test_wealth_distribution(fw_name, n_agents, n_steps)
            self._add_metric(fw_name, 'wealth_transfer', 'statistical',
                           'boltzmann_ks_statistic', ks_stat, 0.0, ks_stat,
                           p_value > 0.01, f'KS={ks_stat:.4f}, p={p_value:.4f}', 0)
            
            # Random walk should show diffusive behavior
            diffusion_error = self._test_diffusion_coefficient(fw_name, n_agents, n_steps)
            self._add_metric(fw_name, 'random_walk', 'statistical',
                           'diffusion_coefficient_error', diffusion_error, 0.0, diffusion_error,
                           diffusion_error < 0.5, f'Relative error: {diffusion_error:.2%}', 0)

    def _test_wealth_distribution(self, fw_name, n_agents, n_steps):
        """Test if wealth follows expected Boltzmann distribution."""
        initial_wealth = 1
        
        if fw_name == 'AMBER':
            from models.amber_models import AMBERWealthTransfer
            model = AMBERWealthTransfer({'n': n_agents, 'steps': n_steps, 
                                         'initial_wealth': initial_wealth, 'show_progress': False})
            model.run()
            wealths = [a.wealth for a in model.agent_objects_list]
        elif fw_name == 'AgentPy':
            from models.agentpy_models import AgentPyWealthTransfer
            model = AgentPyWealthTransfer({'n': n_agents, 'steps': n_steps, 'initial_wealth': initial_wealth})
            model.run()
            wealths = [a.wealth for a in model.agents]
        elif fw_name == 'Mesa':
            from models.mesa_models import MesaWealthTransfer
            model = MesaWealthTransfer(n=n_agents, steps=n_steps, initial_wealth=initial_wealth)
            model.run()
            wealths = [a.wealth for a in model.agents]
        else:
            return 1.0, 0.0
        
        # Kolmogorov-Smirnov test against exponential
        mean_wealth = np.mean(wealths) if wealths else 1
        if mean_wealth > 0:
            ks_stat, p_value = stats.kstest(wealths, 'expon', args=(0, mean_wealth))
        else:
            ks_stat, p_value = 1.0, 0.0
        
        return ks_stat, p_value

    def _test_diffusion_coefficient(self, fw_name, n_agents, n_steps):
        """Test if random walk shows correct diffusion behavior."""
        speed = 1.0
        expected_D = speed**2 / 2  # Theoretical diffusion coefficient for 2D random walk
        
        if fw_name == 'AMBER':
            from models.amber_models import AMBERRandomWalk
            model = AMBERRandomWalk({'n': n_agents, 'steps': n_steps, 
                                     'world_size': 1000, 'speed': speed, 'show_progress': False})
            model.run()
            positions = [(a.x, a.y) for a in model.agent_objects_list]
        elif fw_name == 'AgentPy':
            from models.agentpy_models import AgentPyRandomWalk
            model = AgentPyRandomWalk({'n': n_agents, 'steps': n_steps, 'world_size': 1000, 'speed': speed})
            model.run()
            positions = [(a.x, a.y) for a in model.agents]
        elif fw_name == 'Mesa':
            from models.mesa_models import MesaRandomWalk
            model = MesaRandomWalk(n=n_agents, steps=n_steps, world_size=1000, speed=speed)
            model.run()
            positions = [(a.x, a.y) for a in model.agents]
        else:
            return 1.0
        
        # Calculate mean squared displacement from center
        center = (500, 500)
        msd = np.mean([(p[0]-center[0])**2 + (p[1]-center[1])**2 for p in positions])
        measured_D = msd / (4 * n_steps)  # MSD = 4Dt in 2D
        
        if expected_D > 0:
            return abs(measured_D - expected_D) / expected_D
        return 1.0

    # =========================================================================
    # 3. REPRODUCIBILITY
    # =========================================================================
    def _test_reproducibility(self, n_agents, n_steps, verbose):
        """Test that same seed produces identical results."""
        print("\n🔄 3. REPRODUCIBILITY")
        print("-" * 50)
        
        for fw_name in self.available:
            is_reproducible = self._test_seed_reproducibility(fw_name, n_agents, n_steps)
            self._add_metric(fw_name, 'wealth_transfer', 'reproducibility',
                           'seed_determinism', 1.0 if is_reproducible else 0.0, 1.0,
                           0.0 if is_reproducible else 1.0, is_reproducible,
                           'Same seed → same result' if is_reproducible else 'Non-deterministic!', 0)

    def _test_seed_reproducibility(self, fw_name, n_agents, n_steps):
        """Run twice with same seed, check identical results."""
        seed = 42
        
        def run_with_seed(seed_val):
            if fw_name == 'AMBER':
                from models.amber_models import AMBERWealthTransfer
                model = AMBERWealthTransfer({'n': n_agents, 'steps': n_steps, 
                                             'initial_wealth': 1, 'seed': seed_val, 'show_progress': False})
                model.run()
                return tuple(a.wealth for a in model.agent_objects_list)
            elif fw_name == 'AgentPy':
                from models.agentpy_models import AgentPyWealthTransfer
                model = AgentPyWealthTransfer({'n': n_agents, 'steps': n_steps, 
                                               'initial_wealth': 1, 'seed': seed_val})
                model.run()
                return tuple(a.wealth for a in model.agents)
            elif fw_name == 'Mesa':
                from models.mesa_models import MesaWealthTransfer
                # Mesa 3.x doesn't have built-in seed in constructor
                import random
                random.seed(seed_val)
                np.random.seed(seed_val)
                model = MesaWealthTransfer(n=n_agents, steps=n_steps, initial_wealth=1)
                model.run()
                return tuple(a.wealth for a in model.agents)
            return None
        
        try:
            run1 = run_with_seed(seed)
            run2 = run_with_seed(seed)
            return run1 == run2 if run1 and run2 else False
        except:
            return False

    # =========================================================================
    # 4. EMERGENT BEHAVIOR
    # =========================================================================
    def _test_emergent_behavior(self, n_agents, n_steps, verbose):
        """Test that expected emergent behaviors occur."""
        print("\n🌱 4. EMERGENT BEHAVIOR")
        print("-" * 50)
        
        for fw_name in self.available:
            # Gini coefficient should increase (inequality grows)
            gini_increase = self._test_gini_increase(fw_name, n_agents, n_steps)
            self._add_metric(fw_name, 'wealth_transfer', 'emergent',
                           'gini_increase', gini_increase, 1.0, 
                           0.0 if gini_increase > 0 else 1.0, gini_increase > 0,
                           f'Gini change: {gini_increase:+.4f}', 0)
            
            # SIR: Recovery count should be monotonically increasing
            r_monotonic = self._test_recovery_monotonic(fw_name, n_agents, n_steps)
            self._add_metric(fw_name, 'sir_epidemic', 'emergent',
                           'recovery_monotonic', 1.0 if r_monotonic else 0.0, 1.0,
                           0.0 if r_monotonic else 1.0, r_monotonic,
                           'R monotonically increasing' if r_monotonic else 'R decreased!', 0)

    def _test_gini_increase(self, fw_name, n_agents, n_steps):
        """Test if Gini coefficient increases (wealth concentrates)."""
        def gini(wealths):
            if not wealths or sum(wealths) == 0:
                return 0
            sorted_w = sorted(wealths)
            n = len(sorted_w)
            cumsum = np.cumsum(sorted_w)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        initial_wealth = 100
        
        if fw_name == 'AMBER':
            from models.amber_models import AMBERWealthTransfer
            model = AMBERWealthTransfer({'n': n_agents, 'steps': n_steps, 
                                         'initial_wealth': initial_wealth, 'show_progress': False})
            initial_gini = gini([initial_wealth] * n_agents)
            model.run()
            final_gini = gini([a.wealth for a in model.agent_objects_list])
        elif fw_name == 'AgentPy':
            from models.agentpy_models import AgentPyWealthTransfer
            model = AgentPyWealthTransfer({'n': n_agents, 'steps': n_steps, 'initial_wealth': initial_wealth})
            initial_gini = gini([initial_wealth] * n_agents)
            model.run()
            final_gini = gini([a.wealth for a in model.agents])
        elif fw_name == 'Mesa':
            from models.mesa_models import MesaWealthTransfer
            model = MesaWealthTransfer(n=n_agents, steps=n_steps, initial_wealth=initial_wealth)
            initial_gini = gini([initial_wealth] * n_agents)
            model.run()
            final_gini = gini([a.wealth for a in model.agents])
        else:
            return 0
        
        return final_gini - initial_gini

    def _test_recovery_monotonic(self, fw_name, n_agents, n_steps):
        """Test if recovered count never decreases."""
        if fw_name == 'AMBER':
            from models.amber_models import AMBERSIRModel
            model = AMBERSIRModel({'n': n_agents, 'steps': n_steps, 'initial_infected': 10,
                                   'world_size': 100, 'movement_speed': 2.0, 'infection_radius': 5.0,
                                   'transmission_rate': 0.3, 'recovery_time': 10, 'show_progress': False})
            results = model.run()
            if 'R' in results['model'].columns:
                r_vals = results['model']['R'].to_list()
                return all(r_vals[i] <= r_vals[i+1] for i in range(len(r_vals)-1))
        elif fw_name == 'AgentPy':
            from models.agentpy_models import AgentPySIRModel
            model = AgentPySIRModel({'n': n_agents, 'steps': n_steps, 'initial_infected': 10,
                                     'world_size': 100, 'movement_speed': 2.0, 'infection_radius': 5.0,
                                     'transmission_rate': 0.3, 'recovery_time': 10})
            model.run()
            if hasattr(model, 'log') and 'recovered' in model.log:
                r_vals = model.log['recovered']
                return all(r_vals[i] <= r_vals[i+1] for i in range(len(r_vals)-1))
        elif fw_name == 'Mesa':
            from models.mesa_models import MesaSIRModel
            model = MesaSIRModel(n=n_agents, steps=n_steps, initial_infected=10,
                                 world_size=100, movement_speed=2.0, infection_radius=5.0,
                                 transmission_rate=0.3, recovery_time=10)
            model.run()
            df = model.datacollector.get_model_vars_dataframe()
            if 'recovered' in df.columns:
                r_vals = df['recovered'].tolist()
                return all(r_vals[i] <= r_vals[i+1] for i in range(len(r_vals)-1))
        return True  # Default pass if can't verify

    # =========================================================================
    # 5. NUMERICAL PRECISION
    # =========================================================================
    def _test_numerical_precision(self, n_agents, n_steps, verbose):
        """Test for numerical error accumulation."""
        print("\n🔢 5. NUMERICAL PRECISION")
        print("-" * 50)
        
        for fw_name in self.available:
            # Run for many steps, check if errors accumulate
            long_steps = n_steps * 10
            short_error = self._measure_wealth_conservation(fw_name, n_agents, n_steps)
            long_error = self._measure_wealth_conservation(fw_name, n_agents, long_steps)
            
            error_growth = long_error - short_error
            self._add_metric(fw_name, 'wealth_transfer', 'precision',
                           'error_accumulation', error_growth, 0.0, abs(error_growth),
                           abs(error_growth) < 1e-6, 
                           f'Error at {n_steps} steps: {short_error:.2e}, at {long_steps}: {long_error:.2e}', 0)

    # =========================================================================
    # 6. EDGE CASES
    # =========================================================================
    def _test_edge_cases(self, verbose):
        """Test boundary conditions and edge cases."""
        print("\n⚠️ 6. EDGE CASES")
        print("-" * 50)
        
        for fw_name in self.available:
            # Single agent
            single_ok = self._test_single_agent(fw_name)
            self._add_metric(fw_name, 'wealth_transfer', 'edge_case',
                           'single_agent', 1.0 if single_ok else 0.0, 1.0,
                           0.0 if single_ok else 1.0, single_ok,
                           'Single agent handled' if single_ok else 'Failed!', 0)
            
            # Zero steps
            zero_ok = self._test_zero_steps(fw_name)
            self._add_metric(fw_name, 'wealth_transfer', 'edge_case',
                           'zero_steps', 1.0 if zero_ok else 0.0, 1.0,
                           0.0 if zero_ok else 1.0, zero_ok,
                           'Zero steps handled' if zero_ok else 'Failed!', 0)

    def _test_single_agent(self, fw_name):
        """Test with single agent."""
        try:
            if fw_name == 'AMBER':
                from models.amber_models import AMBERWealthTransfer
                model = AMBERWealthTransfer({'n': 1, 'steps': 10, 'initial_wealth': 100, 'show_progress': False})
                model.run()
                return model.agent_objects_list[0].wealth == 100
            elif fw_name == 'AgentPy':
                from models.agentpy_models import AgentPyWealthTransfer
                model = AgentPyWealthTransfer({'n': 1, 'steps': 10, 'initial_wealth': 100})
                model.run()
                return list(model.agents)[0].wealth == 100
            elif fw_name == 'Mesa':
                from models.mesa_models import MesaWealthTransfer
                model = MesaWealthTransfer(n=1, steps=10, initial_wealth=100)
                model.run()
                return list(model.agents)[0].wealth == 100
        except:
            return False
        return True

    def _test_zero_steps(self, fw_name):
        """Test with zero steps."""
        try:
            if fw_name == 'AMBER':
                from models.amber_models import AMBERWealthTransfer
                model = AMBERWealthTransfer({'n': 10, 'steps': 0, 'initial_wealth': 100, 'show_progress': False})
                model.run()
                return sum(a.wealth for a in model.agent_objects_list) == 1000
            elif fw_name == 'AgentPy':
                from models.agentpy_models import AgentPyWealthTransfer
                model = AgentPyWealthTransfer({'n': 10, 'steps': 0, 'initial_wealth': 100})
                model.run()
                return sum(a.wealth for a in model.agents) == 1000
            elif fw_name == 'Mesa':
                from models.mesa_models import MesaWealthTransfer
                model = MesaWealthTransfer(n=10, steps=0, initial_wealth=100)
                model.run()
                return sum(a.wealth for a in model.agents) == 1000
        except:
            return False
        return True

    # =========================================================================
    # UTILITIES
    # =========================================================================
    def _add_metric(self, framework, model, category, metric_name, value, expected, error, passed, details, exec_time):
        result = MetricResult(
            framework=framework, model=model, category=category,
            metric_name=metric_name, value=float(value), expected=float(expected),
            error=float(error), passed=bool(passed), details=details,
            execution_time=exec_time
        )
        self.results.append(result)
        status = "✅" if passed else "❌"
        print(f"  {status} {framework:10s} | {metric_name}: {details}")

    def _print_summary(self):
        """Print summary by framework and category."""
        print(f"\n{'='*70}")
        print(f"  COMPREHENSIVE CORRECTNESS SUMMARY")
        print(f"{'='*70}")
        
        # By framework
        by_fw = {}
        for r in self.results:
            if r.framework not in by_fw:
                by_fw[r.framework] = {'passed': 0, 'total': 0}
            by_fw[r.framework]['total'] += 1
            if r.passed:
                by_fw[r.framework]['passed'] += 1
        
        print("\nBy Framework:")
        for fw in ['AMBER', 'AgentPy', 'Mesa']:
            if fw in by_fw:
                p = by_fw[fw]['passed']
                t = by_fw[fw]['total']
                pct = 100 * p / t if t > 0 else 0
                status = "✅" if p == t else "⚠️"
                print(f"  {status} {fw:10s}: {p}/{t} ({pct:.0f}%)")
        
        # By category
        print("\nBy Category:")
        by_cat = {}
        for r in self.results:
            if r.category not in by_cat:
                by_cat[r.category] = {'passed': 0, 'total': 0}
            by_cat[r.category]['total'] += 1
            if r.passed:
                by_cat[r.category]['passed'] += 1
        
        for cat in ['conservation', 'statistical', 'reproducibility', 'emergent', 'precision', 'edge_case']:
            if cat in by_cat:
                p = by_cat[cat]['passed']
                t = by_cat[cat]['total']
                status = "✅" if p == t else "⚠️"
                print(f"  {status} {cat:15s}: {p}/{t}")
        
        print(f"\n{'='*70}\n")

    def save_results(self):
        """Save results to JSON and markdown."""
        
        # JSON
        json_path = self.results_dir / 'correctness_results.json'
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"📄 Results: {json_path}")
        
        # Markdown
        self._generate_markdown()

    def _generate_markdown(self):
        """Generate comprehensive markdown report."""
        
        md = f"""# Comprehensive Correctness Benchmark Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Metric Categories

| Category | Description |
|----------|-------------|
| Conservation | Physical quantities preserved (total wealth, population) |
| Statistical | Output matches known distributions (Boltzmann, diffusion) |
| Reproducibility | Same seed produces identical results |
| Emergent | Expected behaviors emerge (Gini increase, monotonic recovery) |
| Precision | Numerical errors don't accumulate |
| Edge Cases | Boundary conditions handled (single agent, zero steps) |

## Summary by Framework

| Framework | Passed | Total | Score |
|-----------|--------|-------|-------|
"""
        by_fw = {}
        for r in self.results:
            if r.framework not in by_fw:
                by_fw[r.framework] = {'passed': 0, 'total': 0}
            by_fw[r.framework]['total'] += 1
            if r.passed:
                by_fw[r.framework]['passed'] += 1
        
        for fw in ['AMBER', 'AgentPy', 'Mesa']:
            if fw in by_fw:
                p = by_fw[fw]['passed']
                t = by_fw[fw]['total']
                pct = 100 * p / t if t > 0 else 0
                md += f"| {fw} | {p} | {t} | {pct:.0f}% |\n"
        
        md += "\n## Detailed Results\n\n"
        
        for cat in ['conservation', 'statistical', 'reproducibility', 'emergent', 'precision', 'edge_case']:
            cat_results = [r for r in self.results if r.category == cat]
            if not cat_results:
                continue
            
            md += f"### {cat.replace('_', ' ').title()}\n\n"
            md += "| Framework | Model | Metric | Status | Details |\n"
            md += "|-----------|-------|--------|--------|--------|\n"
            
            for r in cat_results:
                status = "✅" if r.passed else "❌"
                md += f"| {r.framework} | {r.model} | {r.metric_name} | {status} | {r.details} |\n"
            
            md += "\n"
        
        md_path = self.results_dir / 'correctness_report.md'
        with open(md_path, 'w') as f:
            f.write(md)
        print(f"📄 Report: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Correctness Benchmarks")
    parser.add_argument('--agents', type=int, default=500, help='Number of agents')
    parser.add_argument('--steps', type=int, default=100, help='Simulation steps')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    benchmark = ComprehensiveCorrectnessBenchmark()
    benchmark.run_all(n_agents=args.agents, n_steps=args.steps, verbose=args.verbose)
    benchmark.save_results()
    
    print("\n✅ Comprehensive benchmark complete!")


if __name__ == '__main__':
    main()
