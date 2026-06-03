# Benchmark Results

> Legacy four-framework output from `benchmarks/runner.py`, retained for
> historical memory comparisons. For the corrected seven-framework benchmark
> used by the README and paper, use `summary_table_all.md` and
> `benchmark_results_all.json`.

Generated: 2026-05-19 20:26:57

## Wealth Transfer

### Execution Time (seconds)

|   Agents |   AMBER |   AMBER (vectorized) |   AgentPy |   Mesa |
|----------|---------|----------------------|-----------|--------|
|      500 |   0.05  |                0.055 |     0.091 |  0.322 |
|     1000 |   0.106 |                0.125 |     0.187 |  1.09  |
|     5000 |   0.586 |                0.709 |     0.965 | 24.864 |

### Peak Memory (MB)

|   Agents |   AMBER |   AMBER (vectorized) |   AgentPy |   Mesa |
|----------|---------|----------------------|-----------|--------|
|      500 |     0.1 |                  0.1 |       0.2 |    0.2 |
|     1000 |     0.2 |                  0.1 |       0.3 |    0.4 |
|     5000 |     1   |                  0.6 |       1.5 |    1.9 |

## SIR Epidemic

### Execution Time (seconds)

|   Agents |   AMBER |   AMBER (vectorized) |   AgentPy |   Mesa |
|----------|---------|----------------------|-----------|--------|
|      500 |   0.948 |                0.176 |     1.066 |  0.95  |
|     1000 |   3.598 |                0.2   |     3.705 |  3.607 |
|     5000 |  24.586 |                0.832 |    26.165 | 32.568 |

### Peak Memory (MB)

|   Agents |   AMBER |   AMBER (vectorized) |   AgentPy |   Mesa |
|----------|---------|----------------------|-----------|--------|
|      500 |     0.2 |                  0.1 |       0.2 |    0.2 |
|     1000 |     0.4 |                  0.2 |       0.4 |    0.5 |
|     5000 |     1.8 |                  1   |       1.8 |    2.2 |

## Random Walk

### Execution Time (seconds)

|   Agents |   AMBER |   AMBER (vectorized) |   AgentPy |   Mesa |
|----------|---------|----------------------|-----------|--------|
|      500 |   0.133 |                0.009 |     0.074 |  0.068 |
|     1000 |   0.267 |                0.009 |     0.15  |  0.132 |
|     5000 |   1.341 |                0.016 |     0.769 |  0.677 |

### Peak Memory (MB)

|   Agents |   AMBER |   AMBER (vectorized) |   AgentPy |   Mesa |
|----------|---------|----------------------|-----------|--------|
|      500 |     0.2 |                  0.1 |       0.2 |    0.2 |
|     1000 |     0.3 |                  0.1 |       0.4 |    0.5 |
|     5000 |     1.4 |                  0.6 |       1.7 |    2.2 |

## Performance Summary

- **AMBER (vectorized) vs AgentPy**: 14.88x faster
- **AMBER (vectorized) vs Mesa**: 19.68x faster
- **AMBER (vectorized) vs AMBER**: 20.50x faster
