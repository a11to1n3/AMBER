# Benchmark results — all frameworks

_Generated 2026-05-09 23:02:33 on `python 3.12`. Lower is better. Times are wall-clock, averaged over multiple runs._

## Wealth Transfer

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| **AMBER (vectorized)** | 4ms | 6ms | 20ms |
| AMBER (loop) | 16ms | 33ms | 173ms |
| Agents.jl | 1ms | 1ms | 7ms |
| SimPy | 18ms | 37ms | 215ms |
| Melodie | 18ms | 35ms | 178ms |
| AgentPy | 26ms | 52ms | 262ms |
| Mesa | 259ms | 1.00s | 24.40s |

## Random Walk

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| **AMBER (vectorized)** | 2ms | 3ms | 6ms |
| AMBER (loop) | 32ms | 64ms | 319ms |
| Agents.jl | 1ms | 1ms | 7ms |
| SimPy | 23ms | 47ms | 243ms |
| Melodie | 101ms | 200ms | 1.02s |
| AgentPy | 14ms | 28ms | 137ms |
| Mesa | 13ms | 26ms | 127ms |

## Sir Epidemic

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| **AMBER (vectorized)** | 88ms | 112ms | 551ms |
| AMBER (loop) | 185ms | 824ms | 9.36s |
| Agents.jl | 5ms | 40ms | 809ms |
| SimPy | 120ms | 424ms | 5.13s |
| Melodie | 556ms | 1.87s | 16.21s |
| AgentPy | 188ms | 897ms | 11.05s |
| Mesa | 244ms | 1.14s | 17.07s |

## Speedup of AMBER (vectorized) vs other frameworks

| Framework | wealth_transfer | random_walk | sir_epidemic |
|---|---|---|---|
| AMBER (loop) | 5.9× | 30.7× | 8.8× |
| Agents.jl | 0.2× | 0.7× | 0.6× |
| SimPy | 7.0× | 22.9× | 4.8× |
| Melodie | 6.2× | 97.6× | 17.5× |
| AgentPy | 9.2× | 13.2× | 10.1× |
| Mesa | 480.1× | 12.3× | 14.7× |

