# AMBER native GPU vs vectorized scaling

_Generated 2026-07-17. NVIDIA RTX 5090, 50 steps, 10 runs (trimmed mean). AMBER (vectorized) = `model.cpu(mode="vectorized").run()`; AMBER (GPU) = same classes via `model.gpu().run()` (0.4.3 native view API). Lower is better._

**Protocol:** Native API rerun: AMBER (vectorized)=cpu(mode=vectorized), AMBER (GPU)=same classes via model.gpu().run(). SIR vectorized capped at 10k (100k OOMs all-pairs).

## Wealth Transfer

| Framework | 1000 | 10000 | 100000 | 1000000 | 10000000 |
|---|---:|---:|---:|---:|---:|
| AMBER (GPU) | **23ms** | **47ms** | **334ms** | **3.91s** | **193s** |
| AMBER (vectorized) | 30ms | 76ms | 585ms | 6.44s | 214s |

## Random Walk

| Framework | 1000 | 10000 | 100000 | 1000000 | 10000000 |
|---|---:|---:|---:|---:|---:|
| AMBER (GPU) | 29ms | 31ms | **47ms** | **198ms** | **2.04s** |
| AMBER (vectorized) | **3.9ms** | **7.3ms** | 54ms | 531ms | 6.23s |

## SIR Epidemic

| Framework | 1000 | 10000 | 100000 | 1000000 | 10000000 |
|---|---:|---:|---:|---:|---:|
| AMBER (GPU) | 82ms | **82ms** | — | — | — |
| AMBER (vectorized) | **62ms** | 736ms | — | — | — |

## Schelling Segregation

| Framework | 1000 | 10000 | 100000 | 1000000 | 10000000 |
|---|---:|---:|---:|---:|---:|
| AMBER (GPU) | 74ms | 77ms | **108ms** | **428ms** | **5.17s** |
| AMBER (vectorized) | **12ms** | **24ms** | 201ms | 2.64s | 59.8s |

## GPU vs vectorized (vectorized time / GPU time)

| Model | 100k | 1M | 10M |
|---|---:|---:|---:|
| Wealth Transfer | 1.7× | 1.6× | 1.1× |
| Random Walk | 1.2× | 2.7× | 3.0× |
| SIR Epidemic | — | — | — |
| Schelling Segregation | 1.9× | 6.2× | 11.6× |

## Notes

- **Wealth transfer:** GPU is only modestly faster than vectorized CPU; device sync / launch overhead dominates this light kernel.
- **Random walk / Schelling:** GPU wins clearly at ≥100k–1M agents (~2.7× and ~6× at 1M; ~3× and ~12× at 10M).
- **SIR epidemic (all-pairs):** vectorized capped at 10k agents (100k OOMs on pair matrix). GPU measured at 1k/10k only; 100k OOMs (~7.4 GB allocation failed on this host).
- Chart: [`scaling_chart_native_gpu.png`](scaling_chart_native_gpu.png).
- Raw JSON is machine-local / gitignored; regenerate with `python benchmarks/run_all_frameworks.py --frameworks "AMBER (GPU)" "AMBER (vectorized)" ...`.
