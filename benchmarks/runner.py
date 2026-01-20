#!/usr/bin/env python
"""
AMBER vs AgentPy vs Mesa Benchmark Runner

Comprehensive performance comparison between agent-based modeling frameworks.

Usage:
    python runner.py --quick      # Quick test with small scale
    python runner.py --full       # Full benchmark with all scales
    python runner.py --agents 1000 --steps 50  # Custom run
"""

import sys
import os
import json
import time
import tracemalloc
import argparse
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, asdict

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np

# Try to import tabulate for nice tables
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Try to import matplotlib for charts
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    framework: str
    model: str
    n_agents: int
    n_steps: int
    execution_time: float  # seconds
    peak_memory_mb: float
    time_per_step: float
    timestamp: str


class BenchmarkRunner:
    """Unified benchmark runner for comparing ABM frameworks."""
    
    AGENT_COUNTS = [100, 500, 1000, 5000, 10000]
    QUICK_AGENT_COUNTS = [100, 500]
    DEFAULT_STEPS = 100
    QUICK_STEPS = 20
    
    MODEL_CONFIGS = {
        'wealth_transfer': {
            'initial_wealth': 1,
        },
        'sir_epidemic': {
            'initial_infected': 5,
            'world_size': 100,
            'movement_speed': 2.0,
            'infection_radius': 5.0,
            'transmission_rate': 0.1,
            'recovery_time': 14,
        },
        'random_walk': {
            'world_size': 100,
            'speed': 1.0,
        }
    }
    
    def __init__(self, results_dir: str = None):
        self.results_dir = Path(results_dir or Path(__file__).parent / 'results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self._load_frameworks()
    
    def _load_frameworks(self):
        """Load model implementations from each framework."""
        self.frameworks = {}
        
        # Load AMBER models
        try:
            from models.amber_models import AMBER_MODELS
            self.frameworks['AMBER'] = AMBER_MODELS
            print("✅ AMBER models loaded")
        except ImportError as e:
            print(f"⚠️  AMBER models not available: {e}")
        
        # Load AgentPy models
        try:
            from models.agentpy_models import AGENTPY_MODELS
            self.frameworks['AgentPy'] = AGENTPY_MODELS
            print("✅ AgentPy models loaded")
        except ImportError as e:
            print(f"⚠️  AgentPy not available: {e}")
        
        # Load Mesa models
        try:
            from models.mesa_models import MESA_MODELS
            self.frameworks['Mesa'] = MESA_MODELS
            print("✅ Mesa models loaded")
        except ImportError as e:
            print(f"⚠️  Mesa not available: {e}")
    
    def _run_single_benchmark(
        self,
        framework: str,
        model_name: str,
        model_class: type,
        n_agents: int,
        n_steps: int
    ) -> BenchmarkResult:
        """Run a single benchmark and measure performance."""
        
        # Force garbage collection
        gc.collect()
        
        # Prepare parameters
        config = self.MODEL_CONFIGS.get(model_name, {}).copy()
        config['n'] = n_agents
        config['steps'] = n_steps
        
        # Start memory tracking
        tracemalloc.start()
        
        # Time the simulation
        start_time = time.perf_counter()
        
        try:
            # AMBER and AgentPy accept dict, Mesa accepts kwargs
            if framework == 'Mesa':
                model = model_class(**config)
            else:  # AMBER and AgentPy
                model = model_class(config)
            model.run()
        except Exception as e:
            tracemalloc.stop()
            print(f"  ❌ Error: {e}")
            return None
        
        end_time = time.perf_counter()
        
        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        execution_time = end_time - start_time
        peak_memory_mb = peak / 1024 / 1024
        time_per_step = execution_time / n_steps
        
        return BenchmarkResult(
            framework=framework,
            model=model_name,
            n_agents=n_agents,
            n_steps=n_steps,
            execution_time=round(execution_time, 4),
            peak_memory_mb=round(peak_memory_mb, 2),
            time_per_step=round(time_per_step, 6),
            timestamp=datetime.now().isoformat()
        )
    
    def run_benchmarks(
        self,
        agent_counts: List[int] = None,
        n_steps: int = None,
        models: List[str] = None,
        frameworks: List[str] = None,
        n_runs: int = 3
    ):
        """Run full benchmark suite."""
        
        agent_counts = agent_counts or self.AGENT_COUNTS
        n_steps = n_steps or self.DEFAULT_STEPS
        models = models or list(self.MODEL_CONFIGS.keys())
        frameworks = frameworks or list(self.frameworks.keys())
        
        total_runs = len(frameworks) * len(models) * len(agent_counts) * n_runs
        current_run = 0
        
        print(f"\n{'='*60}")
        print(f"  AMBER vs AgentPy vs Mesa Performance Benchmark")
        print(f"{'='*60}")
        print(f"  Frameworks: {', '.join(frameworks)}")
        print(f"  Models: {', '.join(models)}")
        print(f"  Agent counts: {agent_counts}")
        print(f"  Steps per run: {n_steps}")
        print(f"  Runs per config: {n_runs}")
        print(f"  Total benchmarks: {total_runs}")
        print(f"{'='*60}\n")
        
        for model_name in models:
            print(f"\n📊 Model: {model_name.upper()}")
            print("-" * 40)
            
            for n_agents in agent_counts:
                print(f"\n  Agents: {n_agents}")
                
                for framework in frameworks:
                    if framework not in self.frameworks:
                        continue
                    
                    model_registry = self.frameworks[framework]
                    if model_name not in model_registry:
                        continue
                    
                    model_class = model_registry[model_name]
                    
                    # Run multiple times and take average
                    run_results = []
                    for run_idx in range(n_runs):
                        current_run += 1
                        result = self._run_single_benchmark(
                            framework, model_name, model_class, n_agents, n_steps
                        )
                        if result:
                            run_results.append(result)
                    
                    if run_results:
                        # Average the results
                        avg_result = BenchmarkResult(
                            framework=framework,
                            model=model_name,
                            n_agents=n_agents,
                            n_steps=n_steps,
                            execution_time=round(np.mean([r.execution_time for r in run_results]), 4),
                            peak_memory_mb=round(np.mean([r.peak_memory_mb for r in run_results]), 2),
                            time_per_step=round(np.mean([r.time_per_step for r in run_results]), 6),
                            timestamp=datetime.now().isoformat()
                        )
                        self.results.append(avg_result)
                        
                        status = "✅"
                        print(f"    {status} {framework:10s}: {avg_result.execution_time:8.3f}s | {avg_result.peak_memory_mb:8.2f}MB")
        
        print(f"\n{'='*60}")
        print(f"  Benchmark Complete!")
        print(f"{'='*60}\n")
        
        return self.results
    
    def save_results(self):
        """Save results to JSON and markdown files."""
        
        # Save JSON
        json_path = self.results_dir / 'benchmark_results.json'
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"📄 Results saved to: {json_path}")
        
        # Generate summary table
        self._generate_summary_table()
        
        # Generate chart if matplotlib available
        if HAS_MATPLOTLIB:
            self._generate_charts()
    
    def _generate_summary_table(self):
        """Generate markdown summary table."""
        
        # Organize data by model and agent count
        summary = {}
        for r in self.results:
            key = (r.model, r.n_agents)
            if key not in summary:
                summary[key] = {}
            summary[key][r.framework] = r
        
        # Build table for each model
        md_content = "# Benchmark Results\n\n"
        md_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for model_name in self.MODEL_CONFIGS.keys():
            md_content += f"## {model_name.replace('_', ' ').title()}\n\n"
            
            # Execution time table
            md_content += "### Execution Time (seconds)\n\n"
            headers = ["Agents"] + list(self.frameworks.keys())
            rows = []
            
            for n_agents in self.AGENT_COUNTS:
                key = (model_name, n_agents)
                if key not in summary:
                    continue
                row = [str(n_agents)]
                for fw in self.frameworks.keys():
                    if fw in summary[key]:
                        row.append(f"{summary[key][fw].execution_time:.3f}")
                    else:
                        row.append("-")
                rows.append(row)
            
            if rows:
                if HAS_TABULATE:
                    md_content += tabulate(rows, headers, tablefmt="github") + "\n\n"
                else:
                    md_content += "| " + " | ".join(headers) + " |\n"
                    md_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                    for row in rows:
                        md_content += "| " + " | ".join(row) + " |\n"
                    md_content += "\n"
            
            # Memory usage table
            md_content += "### Peak Memory (MB)\n\n"
            rows = []
            
            for n_agents in self.AGENT_COUNTS:
                key = (model_name, n_agents)
                if key not in summary:
                    continue
                row = [str(n_agents)]
                for fw in self.frameworks.keys():
                    if fw in summary[key]:
                        row.append(f"{summary[key][fw].peak_memory_mb:.1f}")
                    else:
                        row.append("-")
                rows.append(row)
            
            if rows:
                if HAS_TABULATE:
                    md_content += tabulate(rows, headers, tablefmt="github") + "\n\n"
                else:
                    md_content += "| " + " | ".join(headers) + " |\n"
                    md_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                    for row in rows:
                        md_content += "| " + " | ".join(row) + " |\n"
                    md_content += "\n"
        
        # Calculate speedup comparison
        md_content += "## Performance Summary\n\n"
        md_content += self._calculate_speedup_summary()
        
        # Save markdown
        md_path = self.results_dir / 'summary_table.md'
        with open(md_path, 'w') as f:
            f.write(md_content)
        print(f"📄 Summary saved to: {md_path}")
    
    def _calculate_speedup_summary(self) -> str:
        """Calculate average speedup vs other frameworks."""
        
        # Group by model and agent count
        summary = {}
        for r in self.results:
            key = (r.model, r.n_agents)
            if key not in summary:
                summary[key] = {}
            summary[key][r.framework] = r.execution_time
        
        # Calculate relative speedups
        speedups = {'AMBER_vs_AgentPy': [], 'AMBER_vs_Mesa': []}
        
        for key, times in summary.items():
            if 'AMBER' in times and 'AgentPy' in times:
                speedups['AMBER_vs_AgentPy'].append(times['AgentPy'] / times['AMBER'])
            if 'AMBER' in times and 'Mesa' in times:
                speedups['AMBER_vs_Mesa'].append(times['Mesa'] / times['AMBER'])
        
        content = ""
        if speedups['AMBER_vs_AgentPy']:
            avg = np.mean(speedups['AMBER_vs_AgentPy'])
            content += f"- **AMBER vs AgentPy**: {avg:.2f}x {'faster' if avg > 1 else 'slower'}\n"
        if speedups['AMBER_vs_Mesa']:
            avg = np.mean(speedups['AMBER_vs_Mesa'])
            content += f"- **AMBER vs Mesa**: {avg:.2f}x {'faster' if avg > 1 else 'slower'}\n"
        
        return content or "No comparison data available.\n"
    
    def _generate_charts(self):
        """Generate performance comparison charts."""
        
        # Create figure with subplots for each model
        n_models = len(self.MODEL_CONFIGS)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        colors = {'AMBER': '#2563eb', 'AgentPy': '#dc2626', 'Mesa': '#16a34a'}
        
        for idx, model_name in enumerate(self.MODEL_CONFIGS.keys()):
            ax = axes[idx]
            
            for framework in self.frameworks.keys():
                model_results = [r for r in self.results 
                               if r.model == model_name and r.framework == framework]
                if model_results:
                    x = [r.n_agents for r in model_results]
                    y = [r.execution_time for r in model_results]
                    ax.plot(x, y, 'o-', label=framework, color=colors.get(framework, 'gray'), linewidth=2)
            
            ax.set_xlabel('Number of Agents')
            ax.set_ylabel('Execution Time (s)')
            ax.set_title(model_name.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        plt.tight_layout()
        chart_path = self.results_dir / 'scaling_chart.png'
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"📊 Chart saved to: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="AMBER vs AgentPy vs Mesa Benchmark")
    parser.add_argument('--quick', action='store_true', help='Quick test with small scale')
    parser.add_argument('--full', action='store_true', help='Full benchmark with all scales')
    parser.add_argument('--agents', type=int, nargs='+', help='Agent counts to test')
    parser.add_argument('--steps', type=int, default=100, help='Steps per simulation')
    parser.add_argument('--runs', type=int, default=3, help='Runs per configuration')
    parser.add_argument('--models', type=str, nargs='+', help='Models to benchmark')
    parser.add_argument('--frameworks', type=str, nargs='+', help='Frameworks to benchmark')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner()
    
    if args.quick:
        agent_counts = BenchmarkRunner.QUICK_AGENT_COUNTS
        n_steps = BenchmarkRunner.QUICK_STEPS
        n_runs = 1
    elif args.full:
        agent_counts = BenchmarkRunner.AGENT_COUNTS
        n_steps = BenchmarkRunner.DEFAULT_STEPS
        n_runs = 3
    else:
        agent_counts = args.agents or BenchmarkRunner.QUICK_AGENT_COUNTS
        n_steps = args.steps
        n_runs = args.runs
    
    runner.run_benchmarks(
        agent_counts=agent_counts,
        n_steps=n_steps,
        models=args.models,
        frameworks=args.frameworks,
        n_runs=n_runs
    )
    
    runner.save_results()
    
    print("\n✅ Benchmark complete! Check the results/ directory for output.")


if __name__ == '__main__':
    main()
