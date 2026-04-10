# Benchmark results — all frameworks

_Generated 2026-04-09 22:21:00 on `python 3.13`. Lower is better. Times are wall-clock, averaged over multiple runs._

## Wealth Transfer

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| **AMBER (vectorized)** | 4ms | 5ms | 17ms |
| AMBER (loop) | 9ms | 17ms | 89ms |
| Agents.jl | 1ms | 1ms | 7ms |
| SimPy | 18ms | 37ms | 205ms |
| Melodie | 17ms | 33ms | 168ms |
| AgentPy | 26ms | 51ms | 266ms |
| Mesa | 35ms | 116ms | 2.87s |

## Random Walk

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| **AMBER (vectorized)** | 3ms | 2ms | 5ms |
| AMBER (loop) | 8ms | 16ms | 79ms |
| Agents.jl | 1ms | 1ms | 7ms |
| SimPy | 20ms | 40ms | 209ms |
| Melodie | 96ms | 190ms | 963ms |
| AgentPy | 10ms | 20ms | 98ms |
| Mesa | 9ms | 18ms | 87ms |

## Sir Epidemic

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| **AMBER (vectorized)** | 92ms | 141ms | 608ms |
| AMBER (loop) | 191ms | 857ms | 12.23s |
| Agents.jl | 5ms | 40ms | 892ms |
| SimPy | 100ms | 528ms | 6.75s |
| Melodie | 374ms | 1.17s | 11.21s |
| AgentPy | 115ms | 809ms | 8.78s |
| Mesa | 188ms | 728ms | 9.18s |

## Speedup of AMBER (vectorized) vs other frameworks

| Framework | wealth_transfer | random_walk | sir_epidemic |
|---|---|---|---|
| AMBER (loop) | 3.6× | 8.6× | 9.4× |
| Agents.jl | 0.3× | 0.7× | 0.6× |
| SimPy | 8.0× | 22.1× | 5.3× |
| Melodie | 6.9× | 103.3× | 10.3× |
| AgentPy | 10.7× | 10.6× | 7.1× |
| Mesa | 66.8× | 9.5× | 7.4× |

