# AMBER vs AgentPy vs Mesa Performance Benchmark

This directory contains comprehensive performance benchmarks comparing the AMBER framework against two popular Python agent-based modeling (ABM) frameworks: **AgentPy** and **Mesa**.

## Quick Start

```bash
# Install benchmark dependencies
pip install -r requirements.txt

# Run quick benchmark (small scale)
python runner.py --quick

# Run full benchmark
python runner.py --full
```

## Metrics Measured

| Metric | Description |
|--------|-------------|
| **Execution Time** | Wall-clock time for complete simulation |
| **Memory Usage** | Peak memory consumption (MB) |
| **Time per Step** | Average time per simulation step |
| **Scaling Factor** | Performance change ratio vs. agent count |

## Models Compared

1. **Wealth Transfer** - Boltzmann wealth distribution model
2. **SIR Epidemic** - Spatial disease spread model
3. **Random Walk** - Basic 2D random walk

## Results

After running benchmarks, results are saved to:
- `results/benchmark_results.json` - Raw data
- `results/summary_table.md` - Markdown comparison
- `results/scaling_chart.png` - Visualization

## Architecture

```
benchmarks/
├── models/
│   ├── amber_models.py    # AMBER implementations
│   ├── agentpy_models.py  # AgentPy implementations
│   └── mesa_models.py     # Mesa implementations
├── runner.py              # Benchmark runner
├── requirements.txt       # Dependencies
└── results/               # Output folder
```
