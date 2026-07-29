# Calibration & validation benchmark

Compares every parameter-calibration method AMBER supports on a **ground-truth
recovery** task across three model families, then **validates** the recovered
parameters out-of-sample.

## Task

For each model family we pick known ground-truth parameters θ\*, generate a
synthetic *observed* summary trajectory by simulating at θ\* (averaged over
training seeds), then each method recovers θ̂ by minimising the SSE between its
simulated trajectory and the observed one under a fixed evaluation budget.
Finally we re-simulate at θ̂ on **held-out seeds** and measure out-of-sample
loss (and the overfitting gap).

| Model | Recover | Observed statistic |
|-------|---------|--------------------|
| `sir` | infection rate β, recovery rate γ | infected-fraction curve |
| `wealth` | transfer probability | Gini-coefficient curve |
| `schelling` | tolerance | segregation-index curve (tolerance-independent) |

## Methods

All share one loss; each is faithfully the named method ([`methods.py`](methods.py)):

- **grid** — dense grid over the bounds
- **random** — uniform random search
- **bayesian** — Gaussian-process Bayesian optimisation (scikit-optimize)
- **smac** — SMAC (random-forest surrogate), via AMBER's `SMACOptimizer` stack
- **gpu_ensemble** — AMBER's GPU batched ensemble (`smac_batch_calibrate`):
  SMAC proposes a whole batch of candidates each round and the GPU evaluates
  **all of them in one `(B, steps)` tensor pass**. Available for `sir` (the
  batched model `BatchedWellMixedSIR`); the other families use the CPU methods.

## Run

```bash
cd benchmarks/calibration
AMBER_SUPPRESS_DEPRECATIONS=1 python run_calibration_benchmark.py --budget 48
# --quick for a fast 16-eval smoke run
```

Outputs (in `benchmarks/results/`): `summary_table_calibration.md`,
`calibration_results.json`, `calibration_curves.png` (best-loss-vs-evaluations).

## Headline findings (budget 48)

- **SIR (the hard 2-parameter problem).** Bayesian recovers most accurately
  (normalised error **0.012**) but costs **8.7 s**. The **GPU ensemble reaches
  near-Bayesian accuracy (0.031) in 0.38 s** — ~23× faster than Bayesian and
  ~50× faster than SMAC — because it evaluates 32 candidates per batched GPU
  pass (≈85 evals/s vs SMAC's ≈3). This is the calibration-throughput win.
- **1-parameter problems (wealth, schelling).** Cheap grid/random are
  competitive and the surrogate methods' per-evaluation overhead doesn't pay
  off at low dimension. The wealth Gini-loss is weakly identifiable near the
  optimum (low validation loss despite a larger recovery error).
- **Validation.** Out-of-sample losses track training losses with small
  overfit gaps, so the recovered parameters generalise to unseen seeds.

Numbers are reproducible (seeded) on the benchmark GPU host; absolute
wall-clock varies with hardware.
