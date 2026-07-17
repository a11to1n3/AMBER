# Large-N multi-framework scaling

_NVIDIA RTX 5090, 50 steps, 10 runs (trimmed mean). AMBER (GPU) / AMBER (vectorized) use 0.4.3 native placement (`model.gpu().run()` / `cpu(mode="vectorized")`) on the same view-API classes. Other frameworks from the same host’s large-N sweep. Lower is better. Lines stop where a framework OOM’d or timed out._

**Chart:** [`scaling_chart.png`](scaling_chart.png)

## Wealth Transfer

| Framework | 1000 | 10000 | 100000 | 1000000 | 10000000 |
|---|---:|---:|---:|---:|---:|
| AMBER (GPU) | 23 ms | 47 ms | 334 ms | 3.91 s | 193 s |
| AMBER (vectorized) | 30 ms | 76 ms | 585 ms | 6.44 s | 214 s |
| AMBER (loop) | 120 ms | 1.19 s | 11.9 s | 98.3 s | — |
| mesa-frames | 54 ms | 168 ms | 1.36 s | 23.6 s | — |
| FLAME GPU 2 | 9.2 ms | **9.9 ms** | **12 ms** | **28 ms** | **226 ms** |
| Agents.jl | **2.1 ms** | 20 ms | 292 ms | 8.53 s | — |
| SimPy | 74 ms | 870 ms | 11.2 s | — | — |
| Melodie | 78 ms | 798 ms | 8.05 s | 84.9 s | — |
| AgentPy | 114 ms | 1.14 s | 11.3 s | 122 s | — |
| Mesa | 286 ms | 24.9 s | — | — | — |

## Random Walk

| Framework | 1000 | 10000 | 100000 | 1000000 | 10000000 |
|---|---:|---:|---:|---:|---:|
| AMBER (GPU) | 29 ms | 31 ms | 47 ms | 198 ms | 2.04 s |
| AMBER (vectorized) | 3.9 ms | 7.3 ms | 54 ms | 531 ms | 6.23 s |
| AMBER (loop) | 42 ms | 421 ms | 4.39 s | — | — |
| mesa-frames | 42 ms | 64 ms | 310 ms | 3.55 s | 20.8 s |
| FLAME GPU 2 | 4.8 ms | **5.4 ms** | **7.2 ms** | **20 ms** | **201 ms** |
| Agents.jl | **0.9 ms** | 14 ms | 156 ms | 4.41 s | — |
| SimPy | 97 ms | 1.13 s | 13.0 s | — | — |
| Melodie | 465 ms | 4.70 s | 46.3 s | — | — |
| AgentPy | 65 ms | 643 ms | 6.58 s | 69.3 s | — |
| Mesa | 56 ms | 538 ms | 5.54 s | 63.4 s | — |

## SIR Epidemic

| Framework | 1000 | 10000 | 100000 | 1000000 | 10000000 |
|---|---:|---:|---:|---:|---:|
| AMBER (GPU) | 82 ms | 82 ms | — | — | — |
| AMBER (vectorized) | 62 ms | 736 ms | — | — | — |
| AMBER (loop) | 1.56 s | — | — | — | — |
| mesa-frames | 333 ms | 2.83 s | — | — | — |
| FLAME GPU 2 | **12 ms** | **13 ms** | **20 ms** | **108 ms** | **3.80 s** |
| Agents.jl | 43 ms | 4.13 s | — | — | — |
| SimPy | 792 ms | 30.9 s | — | — | — |
| Melodie | 2.73 s | — | — | — | — |
| AgentPy | 1.50 s | — | — | — | — |
| Mesa | 1.43 s | — | — | — | — |

## Schelling Segregation

| Framework | 1000 | 10000 | 100000 | 1000000 | 10000000 |
|---|---:|---:|---:|---:|---:|
| AMBER (GPU) | 74 ms | 77 ms | **108 ms** | **428 ms** | **5.17 s** |
| AMBER (vectorized) | 12 ms | **24 ms** | 201 ms | 2.64 s | 59.8 s |
| AMBER (loop) | 90 ms | 934 ms | 12.3 s | — | — |
| mesa-frames | 60 ms | 90 ms | 420 ms | 4.33 s | 86.9 s |
| FLAME GPU 2 | 88 ms | 29 ms | 267 ms | 2.06 s | 20.8 s |
| Agents.jl | **5.2 ms** | 96 ms | 1.22 s | 54.4 s | — |
| SimPy | 137 ms | 1.58 s | 22.8 s | — | — |
| Melodie | 89 ms | 965 ms | 12.0 s | — | — |
| AgentPy | 92 ms | 1.02 s | 13.2 s | — | — |
| Mesa | 103 ms | 1.05 s | 13.6 s | — | — |

## Notes

- **AMBER (GPU)** is not a separate model rewrite: same `step` as vectorized, placed with `.gpu().run()` (CuPy / NVIDIA).
- **SIR (all-pairs):** AMBER vectorized/GPU stop at 10k (pair-matrix OOM). FLAME GPU 2 and others may use different contact representations at large N.
- Object-oriented frameworks often drop out above 100k–1M (budget/OOM).
- Reproduce: `python benchmarks/run_all_frameworks.py` with agents `1000 10000 100000 1000000 10000000`, then `python benchmarks/plot_scaling_with_gpu_schelling.py`.
