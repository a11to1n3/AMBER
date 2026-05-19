# Benchmark results — all frameworks

_Generated 2026-05-19 20:19:14 on `python 3.12`. Lower is better. Times are wall-clock, averaged per configuration._

## Wealth Transfer

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| **AMBER (vectorized)** | 4.4ms | 6.4ms | 21ms |
| AMBER (loop) | 16ms | 33ms | 171ms |
| Agents.jl | 0.6ms | 1.4ms | 7.2ms |
| SimPy | 18ms | 38ms | 218ms |
| Melodie | 19ms | 36ms | 188ms |
| AgentPy | 27ms | 54ms | 272ms |
| Mesa | 261ms | 996ms | 23.92s |

## Random Walk

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| **AMBER (vectorized)** | 2.7ms | 2.7ms | 6.2ms |
| AMBER (loop) | 36ms | 69ms | 352ms |
| Agents.jl | 0.1ms | 0.3ms | 1.6ms |
| SimPy | 25ms | 48ms | 261ms |
| Melodie | 110ms | 216ms | 1.10s |
| AgentPy | 16ms | 31ms | 150ms |
| Mesa | 16ms | 28ms | 141ms |

## Sir Epidemic

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| **AMBER (vectorized)** | 95ms | 119ms | 687ms |
| AMBER (loop) | 174ms | 817ms | 10.57s |
| Agents.jl | 4.2ms | 36ms | 808ms |
| SimPy | 119ms | 497ms | 5.51s |
| Melodie | 472ms | 2.13s | 21.26s |
| AgentPy | 163ms | 973ms | 11.99s |
| Mesa | 250ms | 1.15s | 17.97s |

## Speedup of AMBER (vectorized) vs other frameworks

| Framework | wealth_transfer | random_walk | sir_epidemic |
|---|---|---|---|
| AMBER (loop) | 5.6× | 31.9× | 8.0× |
| Agents.jl | 0.2× | 0.1× | 0.5× |
| SimPy | 6.8× | 23.1× | 4.5× |
| Melodie | 6.3× | 99.6× | 17.9× |
| AgentPy | 9.1× | 14.0× | 9.1× |
| Mesa | 444.4× | 13.0× | 12.8× |

