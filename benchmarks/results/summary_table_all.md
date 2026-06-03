# Benchmark results — all frameworks

_Generated 2026-06-03 09:20:49 on `python 3.12`. Lower is better. Times are wall-clock, averaged per configuration._

## Wealth Transfer

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| AMBER (vectorized) | 4.4ms | 6.3ms | 20ms |
| AMBER (loop) | 17ms | 34ms | 187ms |
| Agents.jl | **0.5ms** | **1.2ms** | **7.4ms** |
| SimPy | 18ms | 41ms | 241ms |
| Melodie | 18ms | 38ms | 207ms |
| AgentPy | 27ms | 60ms | 279ms |
| Mesa | 261ms | 1.03s | 25.89s |

## Random Walk

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| AMBER (vectorized) | 2.9ms | 3.4ms | 5.4ms |
| AMBER (loop) | 41ms | 80ms | 389ms |
| Agents.jl | **0.1ms** | **0.3ms** | **1.6ms** |
| SimPy | 24ms | 51ms | 314ms |
| Melodie | 112ms | 253ms | 1.19s |
| AgentPy | 17ms | 33ms | 169ms |
| Mesa | 14ms | 29ms | 147ms |

## SIR Epidemic

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| AMBER (vectorized) | 96ms | 134ms | **578ms** |
| AMBER (loop) | 157ms | 970ms | 10.16s |
| Agents.jl | **4.4ms** | **37ms** | 812ms |
| SimPy | 121ms | 488ms | 5.46s |
| Melodie | 676ms | 2.17s | 22.83s |
| AgentPy | 227ms | 975ms | 11.27s |
| Mesa | 307ms | 1.48s | 18.79s |

## Speedup of AMBER (vectorized) vs other frameworks

| Framework | wealth_transfer | random_walk | sir_epidemic |
|---|---|---|---|
| AMBER (loop) | 6.2× | 36.6× | 8.8× |
| Agents.jl | 0.2× | 0.1× | 0.6× |
| SimPy | 7.6× | 27.2× | 4.8× |
| Melodie | 6.9× | 111.5× | 20.9× |
| AgentPy | 9.9× | 15.7× | 9.7× |
| Mesa | 511.4× | 13.6× | 15.6× |

