# Benchmark results — all frameworks

_Generated 2026-07-17 18:17:00 on `python 3.12`. Lower is better. Times are wall-clock, averaged per configuration._

## Wealth Transfer

| Framework | 10000000 |
|---|---|
| AMBER (GPU) | **94ms** |
| FLAME GPU 2 | 194ms |

## Random Walk

| Framework | 10000000 |
|---|---|
| AMBER (GPU) | **80ms** |
| FLAME GPU 2 | 161ms |

## SIR Epidemic

| Framework | 10000000 |
|---|---|
| AMBER (GPU) | **2.08s** |
| FLAME GPU 2 | 3.68s |

## Schelling Segregation

| Framework | 10000000 |
|---|---|
| AMBER (GPU) | **295ms** |
| FLAME GPU 2 | 18.72s |

## Speedup of AMBER (GPU) vs other frameworks

| Framework | wealth_transfer | random_walk | sir_epidemic | schelling |
|---|---|---|---|---|
| AMBER (vectorized) | — | — | — | — |
| AMBER (loop) | — | — | — | — |
| mesa-frames | — | — | — | — |
| FLAME GPU 2 | 2.1× | 2.0× | 1.8× | 63.4× |
| Agents.jl | — | — | — | — |
| SimPy | — | — | — | — |
| Melodie | — | — | — | — |
| AgentPy | — | — | — | — |
| Mesa | — | — | — | — |

