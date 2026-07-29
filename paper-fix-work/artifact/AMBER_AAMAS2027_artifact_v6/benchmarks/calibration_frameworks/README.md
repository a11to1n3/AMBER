# Cross-framework calibration & validation benchmark

Compares ABM frameworks on their **calibration/validation ability**: how fast and
how accurately each one recovers a model's parameters from observed data, and how
the recovered parameters validate out-of-sample.

## Task (identical for every framework)

Recover a well-mixed SIR's `(beta, gamma)` from an observed infected-fraction
curve. The dynamics are fixed so the task is portable:

> `status ∈ {S,I,R}`, `k = i0_frac·N` initially infected. Each step (using the
> step-start infected fraction): `S→I` w.p. `beta·I_frac`, `I→R` w.p. `gamma`;
> record the infected fraction. ([`task.py`](task.py))

Each framework implements this on its own engine ([`frameworks.py`](frameworks.py)).

## How each framework is driven

- **Common optimiser (fair throughput).** Every framework evaluates the *same*
  seeded set of candidate `(beta, gamma)` pairs (a shared random search) and
  keeps the best. Recovery accuracy is therefore ~equal across frameworks; the
  wall-clock difference reflects each framework's per-evaluation throughput.
- **Native tooling (capability).** Where a framework ships calibration, we also
  run it: AMBER's **GPU batched ensemble** (`smac_batch_calibrate` — SMAC
  proposes a batch, the GPU evaluates all candidates in one `(B,steps)` tensor
  pass). Most frameworks have no built-in calibrator (you bring your own
  optimiser), which is itself part of the "ability" picture.

Validation re-runs each framework's model at its recovered parameters on
**held-out seeds** and reports out-of-sample loss.

## Frameworks

AMBER (columnar CPU), AMBER GPU ensemble, AMBER native SMAC+GPU, mesa-frames,
Mesa, agentpy, **Agents.jl** (Julia 1.9 / Agents 5.17, run as a self-contained
subprocess with JIT warmup), **FLAME GPU 2** (pyflamegpu RTC, one GPU simulation
per candidate). *Melodie* is in the suite but could not be imported on the
benchmark host (a runtime error at import), so it is omitted here.

## Run

```bash
cd benchmarks/calibration_frameworks
export CUDA_PATH=$HOME/cuda-12.0           # for FLAME GPU 2 RTC
AMBER_SUPPRESS_DEPRECATIONS=1 python run.py --budget 128
```

Outputs (in `benchmarks/results/`): `summary_table_calibration_frameworks.md`,
`calibration_frameworks_results.json`, `calibration_frameworks_curves.png`.

## Headline result (128 candidates, N=3000, 50 steps)

| Framework | Mode | Recovery err | Evals/s | Speedup |
|-----------|------|-------------:|--------:|--------:|
| **AMBER (GPU ensemble)** | batched | 0.065 | **970** | **270×** |
| Agents.jl | common-opt | 0.065 | 303 | 84× |
| AMBER | common-opt | 0.065 | 73 | 20× |
| agentpy | common-opt | 0.038 | 50 | 14× |
| mesa-frames | common-opt | 0.065 | 49 | 14× |
| Mesa | common-opt | 0.065 | 30 | 8× |
| AMBER (native SMAC+GPU) | native | 0.024–0.07 | 5 | 1.3× |
| FLAME GPU 2 | common-opt | 0.050 | 3.6 | 1× |

- **AMBER's GPU batched ensemble dominates calibration throughput** — ~970
  evals/s, 3× faster than Julia (the fastest CPU framework) and ~270× faster
  than the slowest. It batches every candidate into one GPU pass, which is
  exactly the calibration regime (many small replicate runs).
- **The GPU-vs-GPU comparison is the point.** FLAME GPU 2 is the *other* GPU
  framework, yet it is the **slowest** here: it runs one simulation per
  candidate (per-simulation instantiation overhead, no cross-candidate
  batching). FLAME GPU 2 is built for a few very large simulations, not
  thousands of tiny calibration runs — the opposite of AMBER's ensemble.
- **AMBER also leads the Python CPU frameworks** (73 vs 49 mesa-frames /
  30 Mesa evals/s), and its **native SMAC+GPU calibrator is the most
  sample-efficient** (lowest recovery error) though its optimiser overhead
  dominates when each evaluation is cheap.
- Validation losses track training losses, so recovered parameters generalise.

Numbers are seeded/reproducible on the benchmark GPU host; absolute wall-clock
varies with hardware. FLAME GPU 2's figure includes per-simulation setup; a
reuse-optimised FLAME GPU calibration would be faster but remains sequential
(no batching).
