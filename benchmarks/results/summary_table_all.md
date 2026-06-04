# Benchmark results — all frameworks

_Generated 2026-06-04 03:01:06 on `python 3.12`. Lower is better. Times are wall-clock, averaged per configuration._

## Wealth Transfer

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| AMBER (vectorized) | 4.2ms | 6.0ms | 20ms |
| AMBER (loop) | 16ms | 32ms | 169ms |
| Agents.jl | **0.5ms** | **1.3ms** | **7.2ms** |
| SimPy | 18ms | 37ms | 216ms |
| Melodie | 18ms | 36ms | 177ms |
| AgentPy | 26ms | 51ms | 266ms |
| Mesa | 254ms | 969ms | 22.61s |

## Random Walk

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| AMBER (vectorized) | 2.4ms | 2.7ms | 4.8ms |
| AMBER (loop) | 33ms | 66ms | 332ms |
| Agents.jl | **0.2ms** | **0.3ms** | **1.6ms** |
| SimPy | 22ms | 45ms | 254ms |
| Melodie | 101ms | 205ms | 1.03s |
| AgentPy | 14ms | 28ms | 141ms |
| Mesa | 13ms | 25ms | 131ms |

## SIR Epidemic

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| AMBER (vectorized) | 88ms | 112ms | **497ms** |
| AMBER (loop) | 140ms | 799ms | 9.53s |
| Agents.jl | **4.2ms** | **37ms** | 813ms |
| SimPy | 107ms | 411ms | 4.67s |
| Melodie | 595ms | 1.98s | 20.09s |
| AgentPy | 197ms | 826ms | 10.98s |
| Mesa | 265ms | 1.07s | 16.63s |

## Speedup of AMBER (vectorized) vs other frameworks

| Framework | wealth_transfer | random_walk | sir_epidemic |
|---|---|---|---|
| AMBER (loop) | 5.9× | 35.7× | 9.3× |
| Agents.jl | 0.2× | 0.2× | 0.7× |
| SimPy | 7.1× | 26.2× | 4.8× |
| Melodie | 6.3× | 110.6× | 21.6× |
| AgentPy | 9.3× | 15.3× | 10.6× |
| Mesa | 450.9× | 14.0× | 15.4× |

