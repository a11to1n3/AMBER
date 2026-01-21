#!/usr/bin/env python
"""
AMBER Multi-Framework Correctness Benchmark

Verifies scientific correctness of simulations across ALL frameworks:
- AMBER (ours)
- AgentPy
- Mesa
- Melodie
- SimPy
- Agents.jl (Julia)

Tests fundamental physical laws and invariants are preserved.
"""

import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple, Callable

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np


@dataclass
class CorrectnessResult:
    """Result of a correctness benchmark."""
    framework: str
    model: str
    test_name: str
    passed: bool
    execution_time: float
    n_agents: int
    n_steps: int
    metric_value: float
    expected: str
    details: str
    timestamp: str


class MultiFrameworkCorrectnessBenchmark:
    """
    Runs correctness benchmarks across all 6 ABM frameworks.
    Tests scientific invariants for each model type.
    """
    
    FRAMEWORKS = ['AMBER', 'AgentPy', 'Mesa', 'Melodie', 'SimPy']  # Agents.jl handled separately
    
    def __init__(self, results_dir: str = None):
        self.results_dir = Path(results_dir or Path(__file__).parent / 'results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[CorrectnessResult] = []
        self._load_frameworks()
        
    def _load_frameworks(self):
        """Load available frameworks."""
        self.available = {}
        
        # AMBER
        try:
            from models.amber_models import AMBER_MODELS
            self.available['AMBER'] = AMBER_MODELS
            print("✅ AMBER loaded")
        except ImportError as e:
            print(f"⚠️  AMBER not available: {e}")
        
        # AgentPy
        try:
            from models.agentpy_models import AGENTPY_MODELS
            self.available['AgentPy'] = AGENTPY_MODELS
            print("✅ AgentPy loaded")
        except ImportError as e:
            print(f"⚠️  AgentPy not available: {e}")
        
        # Mesa
        try:
            from models.mesa_models import MESA_MODELS
            self.available['Mesa'] = MESA_MODELS
            print("✅ Mesa loaded")
        except ImportError as e:
            print(f"⚠️  Mesa not available: {e}")
        
        # Melodie
        try:
            from models.melodie_models import WealthModel, WalkModel, SIRModel
            self.available['Melodie'] = {
                'wealth_transfer': (WealthModel, 'WealthScenario'),
                'random_walk': (WalkModel, 'WalkScenario'),
                'sir_epidemic': (SIRModel, 'SIRScenario'),
            }
            print("✅ Melodie loaded")
        except ImportError as e:
            print(f"⚠️  Melodie not available: {e}")
        
        # SimPy
        try:
            from models.simpy_models import run_wealth_benchmark, run_walk_benchmark, run_sir_benchmark
            self.available['SimPy'] = {
                'wealth_transfer': run_wealth_benchmark,
                'random_walk': run_walk_benchmark,
                'sir_epidemic': run_sir_benchmark,
            }
            print("✅ SimPy loaded")
        except ImportError as e:
            print(f"⚠️  SimPy not available: {e}")
        
        # Check for Julia/Agents.jl
        try:
            result = subprocess.run(['julia', '--version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                self.available['Agents.jl'] = True
                print("✅ Agents.jl (Julia) available")
        except Exception:
            print("⚠️  Agents.jl (Julia) not available")

    def run_all(self, n_agents: int = 100, n_steps: int = 50, verbose: bool = False):
        """Run correctness benchmarks for all frameworks."""
        
        print(f"\n{'='*70}")
        print(f"  Multi-Framework Correctness Benchmark Suite")
        print(f"{'='*70}")
        print(f"  Frameworks: {', '.join(self.available.keys())}")
        print(f"  Agents: {n_agents}, Steps: {n_steps}")
        print(f"{'='*70}\n")
        
        # Test each model type
        self._test_wealth_transfer(n_agents, n_steps, verbose)
        self._test_sir_epidemic(n_agents, n_steps, verbose)
        self._test_random_walk(n_agents, n_steps, verbose)
        
        # Print summary
        self._print_summary()
        
        return self.results

    def _test_wealth_transfer(self, n_agents: int, n_steps: int, verbose: bool):
        """Test wealth conservation across all frameworks."""
        print("\n📊 WEALTH TRANSFER - Conservation Law")
        print("-" * 50)
        
        initial_wealth = 100
        expected_total = n_agents * initial_wealth
        
        # AMBER
        if 'AMBER' in self.available:
            self._test_amber_wealth(n_agents, n_steps, initial_wealth, verbose)
        
        # AgentPy
        if 'AgentPy' in self.available:
            self._test_agentpy_wealth(n_agents, n_steps, initial_wealth, verbose)
        
        # Mesa
        if 'Mesa' in self.available:
            self._test_mesa_wealth(n_agents, n_steps, initial_wealth, verbose)
        
        # Melodie
        if 'Melodie' in self.available:
            self._test_melodie_wealth(n_agents, n_steps, initial_wealth, verbose)
        
        # SimPy
        if 'SimPy' in self.available:
            self._test_simpy_wealth(n_agents, n_steps, initial_wealth, verbose)

    def _test_amber_wealth(self, n_agents, n_steps, initial_wealth, verbose):
        from models.amber_models import AMBERWealthTransfer
        
        start = time.perf_counter()
        model = AMBERWealthTransfer({
            'n': n_agents, 'steps': n_steps, 
            'initial_wealth': initial_wealth, 'show_progress': False
        })
        model.run()
        elapsed = time.perf_counter() - start
        
        expected_total = n_agents * initial_wealth
        final_total = sum(a.wealth for a in model.agent_objects_list)
        error = abs(final_total - expected_total)
        passed = error < 1e-6
        
        self._add_result('AMBER', 'wealth_transfer', 'conservation', passed, elapsed,
                        n_agents, n_steps, error, 'Total wealth constant',
                        f'Expected: {expected_total}, Got: {final_total}', verbose)

    def _test_agentpy_wealth(self, n_agents, n_steps, initial_wealth, verbose):
        from models.agentpy_models import AgentPyWealthTransfer
        
        start = time.perf_counter()
        model = AgentPyWealthTransfer({'n': n_agents, 'steps': n_steps, 'initial_wealth': initial_wealth})
        model.run()
        elapsed = time.perf_counter() - start
        
        expected_total = n_agents * initial_wealth
        final_total = sum(a.wealth for a in model.agents)
        error = abs(final_total - expected_total)
        passed = error < 1e-6
        
        self._add_result('AgentPy', 'wealth_transfer', 'conservation', passed, elapsed,
                        n_agents, n_steps, error, 'Total wealth constant',
                        f'Expected: {expected_total}, Got: {final_total}', verbose)

    def _test_mesa_wealth(self, n_agents, n_steps, initial_wealth, verbose):
        from models.mesa_models import MesaWealthTransfer
        
        start = time.perf_counter()
        model = MesaWealthTransfer(n=n_agents, steps=n_steps, initial_wealth=initial_wealth)
        model.run()
        elapsed = time.perf_counter() - start
        
        expected_total = n_agents * initial_wealth
        final_total = sum(a.wealth for a in model.agents)
        error = abs(final_total - expected_total)
        passed = error < 1e-6
        
        self._add_result('Mesa', 'wealth_transfer', 'conservation', passed, elapsed,
                        n_agents, n_steps, error, 'Total wealth constant',
                        f'Expected: {expected_total}, Got: {final_total}', verbose)

    def _test_melodie_wealth(self, n_agents, n_steps, initial_wealth, verbose):
        try:
            import Melodie
            from models.melodie_models import WealthModel, WealthScenario, WealthAgent, WealthEnvironment
            import os
            
            config = Melodie.Config(
                project_name='CorrectnessBench', project_root='.', 
                sqlite_folder='.', output_folder='.', input_folder='.'
            )
            
            scenario = WealthScenario()
            scenario.periods = n_steps
            scenario.agent_num = n_agents
            scenario.id = 0
            
            start = time.perf_counter()
            model = WealthModel(config, scenario)
            model.setup()
            
            for i in range(n_agents):
                agent = model.agent_list.add()
                agent.id = i
                agent.setup()
                agent.wealth = initial_wealth
            
            model.run()
            elapsed = time.perf_counter() - start
            
            expected_total = n_agents * initial_wealth
            final_total = sum(a.wealth for a in model.agent_list)
            error = abs(final_total - expected_total)
            passed = error < 1e-6
            
            self._add_result('Melodie', 'wealth_transfer', 'conservation', passed, elapsed,
                            n_agents, n_steps, error, 'Total wealth constant',
                            f'Expected: {expected_total}, Got: {final_total}', verbose)
            
            if os.path.exists('CorrectnessBench.sqlite'):
                os.remove('CorrectnessBench.sqlite')
        except Exception as e:
            self._add_result('Melodie', 'wealth_transfer', 'conservation', False, 0,
                            n_agents, n_steps, -1, 'Total wealth constant',
                            f'Error: {str(e)[:50]}', verbose)

    def _test_simpy_wealth(self, n_agents, n_steps, initial_wealth, verbose):
        import simpy
        import random
        
        def wealth_agent_tracked(env, agent_id, agents_data):
            while True:
                if agents_data[agent_id]['wealth'] > 0:
                    partner_id = random.randrange(len(agents_data))
                    if partner_id != agent_id:
                        agents_data[agent_id]['wealth'] -= 1
                        agents_data[partner_id]['wealth'] += 1
                yield env.timeout(1)
        
        start = time.perf_counter()
        env = simpy.Environment()
        agents_data = [{'id': i, 'wealth': initial_wealth} for i in range(n_agents)]
        
        for i in range(n_agents):
            env.process(wealth_agent_tracked(env, i, agents_data))
        
        env.run(until=n_steps)
        elapsed = time.perf_counter() - start
        
        expected_total = n_agents * initial_wealth
        final_total = sum(a['wealth'] for a in agents_data)
        error = abs(final_total - expected_total)
        passed = error < 1e-6
        
        self._add_result('SimPy', 'wealth_transfer', 'conservation', passed, elapsed,
                        n_agents, n_steps, error, 'Total wealth constant',
                        f'Expected: {expected_total}, Got: {final_total}', verbose)

    def _test_sir_epidemic(self, n_agents: int, n_steps: int, verbose: bool):
        """Test SIR population conservation (S+I+R=N)."""
        print("\n🦠 SIR EPIDEMIC - Population Conservation")
        print("-" * 50)
        
        # AMBER
        if 'AMBER' in self.available:
            self._test_amber_sir(n_agents, n_steps, verbose)
        
        # AgentPy
        if 'AgentPy' in self.available:
            self._test_agentpy_sir(n_agents, n_steps, verbose)
        
        # Mesa
        if 'Mesa' in self.available:
            self._test_mesa_sir(n_agents, n_steps, verbose)

    def _test_amber_sir(self, n_agents, n_steps, verbose):
        from models.amber_models import AMBERSIRModel, SIRAgent
        
        start = time.perf_counter()
        model = AMBERSIRModel({
            'n': n_agents, 'steps': n_steps, 'initial_infected': 5,
            'world_size': 100, 'movement_speed': 2.0, 'infection_radius': 5.0,
            'transmission_rate': 0.3, 'recovery_time': 10, 'show_progress': False
        })
        model.run()
        elapsed = time.perf_counter() - start
        
        s = sum(1 for a in model.agent_objects_list if a.status == SIRAgent.STATUS_S)
        i = sum(1 for a in model.agent_objects_list if a.status == SIRAgent.STATUS_I)
        r = sum(1 for a in model.agent_objects_list if a.status == SIRAgent.STATUS_R)
        total = s + i + r
        error = abs(total - n_agents)
        passed = error == 0
        
        self._add_result('AMBER', 'sir_epidemic', 'population_conservation', passed, elapsed,
                        n_agents, n_steps, error, 'S+I+R=N',
                        f'S={s}, I={i}, R={r}, Total={total}', verbose)

    def _test_agentpy_sir(self, n_agents, n_steps, verbose):
        from models.agentpy_models import AgentPySIRModel, APSIRAgent
        
        start = time.perf_counter()
        model = AgentPySIRModel({
            'n': n_agents, 'steps': n_steps, 'initial_infected': 5,
            'world_size': 100, 'movement_speed': 2.0, 'infection_radius': 5.0,
            'transmission_rate': 0.3, 'recovery_time': 10
        })
        model.run()
        elapsed = time.perf_counter() - start
        
        s = sum(1 for a in model.agents if a.status == APSIRAgent.STATUS_S)
        i = sum(1 for a in model.agents if a.status == APSIRAgent.STATUS_I)
        r = sum(1 for a in model.agents if a.status == APSIRAgent.STATUS_R)
        total = s + i + r
        error = abs(total - n_agents)
        passed = error == 0
        
        self._add_result('AgentPy', 'sir_epidemic', 'population_conservation', passed, elapsed,
                        n_agents, n_steps, error, 'S+I+R=N',
                        f'S={s}, I={i}, R={r}, Total={total}', verbose)

    def _test_mesa_sir(self, n_agents, n_steps, verbose):
        from models.mesa_models import MesaSIRModel, MesaSIRAgent
        
        start = time.perf_counter()
        model = MesaSIRModel(
            n=n_agents, steps=n_steps, initial_infected=5,
            world_size=100, movement_speed=2.0, infection_radius=5.0,
            transmission_rate=0.3, recovery_time=10
        )
        model.run()
        elapsed = time.perf_counter() - start
        
        s = sum(1 for a in model.agents if a.status == MesaSIRAgent.STATUS_S)
        i = sum(1 for a in model.agents if a.status == MesaSIRAgent.STATUS_I)
        r = sum(1 for a in model.agents if a.status == MesaSIRAgent.STATUS_R)
        total = s + i + r
        error = abs(total - n_agents)
        passed = error == 0
        
        self._add_result('Mesa', 'sir_epidemic', 'population_conservation', passed, elapsed,
                        n_agents, n_steps, error, 'S+I+R=N',
                        f'S={s}, I={i}, R={r}, Total={total}', verbose)

    def _test_random_walk(self, n_agents: int, n_steps: int, verbose: bool):
        """Test random walk spatial bounds."""
        print("\n🚶 RANDOM WALK - Spatial Bounds")
        print("-" * 50)
        
        world_size = 100
        
        # AMBER
        if 'AMBER' in self.available:
            self._test_amber_walk(n_agents, n_steps, world_size, verbose)
        
        # AgentPy
        if 'AgentPy' in self.available:
            self._test_agentpy_walk(n_agents, n_steps, world_size, verbose)
        
        # Mesa
        if 'Mesa' in self.available:
            self._test_mesa_walk(n_agents, n_steps, world_size, verbose)

    def _test_amber_walk(self, n_agents, n_steps, world_size, verbose):
        from models.amber_models import AMBERRandomWalk
        
        start = time.perf_counter()
        model = AMBERRandomWalk({
            'n': n_agents, 'steps': n_steps, 
            'world_size': world_size, 'speed': 2.0, 'show_progress': False
        })
        model.run()
        elapsed = time.perf_counter() - start
        
        out_of_bounds = 0
        for a in model.agent_objects_list:
            if a.x < 0 or a.x > world_size or a.y < 0 or a.y > world_size:
                out_of_bounds += 1
        
        passed = out_of_bounds == 0
        
        self._add_result('AMBER', 'random_walk', 'spatial_bounds', passed, elapsed,
                        n_agents, n_steps, out_of_bounds, 'All agents in [0, world_size]',
                        f'Out of bounds: {out_of_bounds}', verbose)

    def _test_agentpy_walk(self, n_agents, n_steps, world_size, verbose):
        from models.agentpy_models import AgentPyRandomWalk
        
        start = time.perf_counter()
        model = AgentPyRandomWalk({
            'n': n_agents, 'steps': n_steps, 
            'world_size': world_size, 'speed': 2.0
        })
        model.run()
        elapsed = time.perf_counter() - start
        
        out_of_bounds = 0
        for a in model.agents:
            if a.x < 0 or a.x > world_size or a.y < 0 or a.y > world_size:
                out_of_bounds += 1
        
        passed = out_of_bounds == 0
        
        self._add_result('AgentPy', 'random_walk', 'spatial_bounds', passed, elapsed,
                        n_agents, n_steps, out_of_bounds, 'All agents in [0, world_size]',
                        f'Out of bounds: {out_of_bounds}', verbose)

    def _test_mesa_walk(self, n_agents, n_steps, world_size, verbose):
        from models.mesa_models import MesaRandomWalk
        
        start = time.perf_counter()
        model = MesaRandomWalk(n=n_agents, steps=n_steps, world_size=world_size, speed=2.0)
        model.run()
        elapsed = time.perf_counter() - start
        
        out_of_bounds = 0
        for a in model.agents:
            if a.x < 0 or a.x > world_size or a.y < 0 or a.y > world_size:
                out_of_bounds += 1
        
        passed = out_of_bounds == 0
        
        self._add_result('Mesa', 'random_walk', 'spatial_bounds', passed, elapsed,
                        n_agents, n_steps, out_of_bounds, 'All agents in [0, world_size]',
                        f'Out of bounds: {out_of_bounds}', verbose)

    def _add_result(self, framework, model, test_name, passed, elapsed, 
                   n_agents, n_steps, metric, expected, details, verbose):
        result = CorrectnessResult(
            framework=framework,
            model=model,
            test_name=test_name,
            passed=passed,
            execution_time=round(elapsed, 4),
            n_agents=n_agents,
            n_steps=n_steps,
            metric_value=float(metric) if not isinstance(metric, (bool, np.bool_)) else 0,
            expected=expected,
            details=details,
            timestamp=datetime.now().isoformat()
        )
        self.results.append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {framework:12s} | {test_name}")
        if verbose:
            print(f"       {details}")

    def _print_summary(self):
        """Print summary by framework."""
        print(f"\n{'='*70}")
        print(f"  CORRECTNESS SUMMARY")
        print(f"{'='*70}")
        
        # Group by framework
        by_fw = {}
        for r in self.results:
            if r.framework not in by_fw:
                by_fw[r.framework] = []
            by_fw[r.framework].append(r)
        
        for fw in ['AMBER', 'AgentPy', 'Mesa', 'Melodie', 'SimPy', 'Agents.jl']:
            if fw in by_fw:
                tests = by_fw[fw]
                passed = sum(1 for t in tests if t.passed)
                total = len(tests)
                status = "✅" if passed == total else "⚠️" if passed > 0 else "❌"
                print(f"  {status} {fw:12s}: {passed}/{total} tests passed")
        
        print(f"{'='*70}\n")

    def save_results(self):
        """Save results to JSON and markdown."""
        
        def to_json_safe(obj):
            d = asdict(obj)
            for k, v in d.items():
                if hasattr(v, 'item'):
                    d[k] = v.item()
                elif isinstance(v, (np.bool_,)):
                    d[k] = bool(v)
            return d
        
        json_path = self.results_dir / 'correctness_results.json'
        with open(json_path, 'w') as f:
            json.dump([to_json_safe(r) for r in self.results], f, indent=2)
        print(f"📄 Results saved to: {json_path}")
        
        self._generate_markdown_report()

    def _generate_markdown_report(self):
        """Generate comparison table."""
        
        md = f"""# Multi-Framework Correctness Benchmark

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary by Framework

| Framework | Tests Passed | Pass Rate |
|-----------|--------------|-----------|
"""
        by_fw = {}
        for r in self.results:
            if r.framework not in by_fw:
                by_fw[r.framework] = []
            by_fw[r.framework].append(r)
        
        for fw in ['AMBER', 'AgentPy', 'Mesa', 'Melodie', 'SimPy', 'Agents.jl']:
            if fw in by_fw:
                tests = by_fw[fw]
                passed = sum(1 for t in tests if t.passed)
                total = len(tests)
                rate = 100 * passed / total if total > 0 else 0
                status = "✅" if passed == total else "⚠️"
                md += f"| {status} {fw} | {passed}/{total} | {rate:.0f}% |\n"
        
        md += "\n## Detailed Results by Test\n\n"
        
        # Group by model
        for model in ['wealth_transfer', 'sir_epidemic', 'random_walk']:
            model_results = [r for r in self.results if r.model == model]
            if not model_results:
                continue
            
            md += f"### {model.replace('_', ' ').title()}\n\n"
            md += "| Framework | Test | Status | Time | Details |\n"
            md += "|-----------|------|--------|------|--------|\n"
            
            for r in model_results:
                status = "✅" if r.passed else "❌"
                md += f"| {r.framework} | {r.test_name} | {status} | {r.execution_time:.4f}s | {r.details} |\n"
            
            md += "\n"
        
        md += """## Scientific Invariants Tested

| Model | Invariant | Description |
|-------|-----------|-------------|
| Wealth Transfer | Conservation | Total wealth S = constant |
| SIR Epidemic | Conservation | S + I + R = N at all times |
| Random Walk | Boundedness | All agents stay within world limits |
"""
        
        md_path = self.results_dir / 'correctness_report.md'
        with open(md_path, 'w') as f:
            f.write(md)
        print(f"📄 Report saved to: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Framework Correctness Benchmarks")
    parser.add_argument('--agents', type=int, default=100, help='Number of agents')
    parser.add_argument('--steps', type=int, default=50, help='Simulation steps')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    benchmark = MultiFrameworkCorrectnessBenchmark()
    benchmark.run_all(n_agents=args.agents, n_steps=args.steps, verbose=args.verbose)
    benchmark.save_results()
    
    print("\n✅ Multi-framework correctness benchmark complete!")


if __name__ == '__main__':
    main()
