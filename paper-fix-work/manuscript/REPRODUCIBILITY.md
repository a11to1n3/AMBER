# Reproducibility map

This package separates manuscript source, raw evidence, executable checks, and
the AMBER code snapshot used for the final revision.

## Headline 10-million-agent benchmark

Raw artifact: `artifacts/benchmark_results_snapshot_correct_10run_10m.json`.
It contains all ten wall-clock samples for every AMBER GPU and FLAME GPU 2 row,
plus means, medians, IQRs, bootstrap median intervals, software versions, GPU,
driver, CUDA, and source-revision provenance. Each configuration used one
untimed warm-up; no timed sample was removed. Timed scope includes model
construction, setup, 50 steps, and result assembly. AMBER explicitly
synchronizes before return, while FLAME returns at a completed simulation
boundary.

The corresponding runner and model sources are under
`reproducibility/code_snapshot/`. On the recorded GPU host, the campaign was:

```bash
python benchmarks/run_all_frameworks.py \
  --agents 10000000 --steps 50 --runs 10 \
  --frameworks "AMBER (GPU)" "FLAME GPU 2" \
  --tag snapshot_correct_10run_10m --budget 120
```

The benchmark is an implementation comparison. The SIR and Schelling
transitions are related but not byte-identical across frameworks. The exact
AMBER SIR cell-binned join has output-sensitive work `O(N + C + P)` and a
quadratic worst case as density grows in a fixed domain; the 10M result is an
empirical measurement, not an asymptotic linearity claim.
The timed FLAME Schelling implementation also performs Python per-agent
initialization inside the measured region. Its 63.4x ratio is therefore a
setup-inclusive exploratory implementation result; the directly comparable
headline range is 1.77x--2.05x across wealth, random walk, and SIR.

Figure 5 also uses `artifacts/benchmark_results_all5090.json`, the archived
all-framework sweep. Run
`scripts/reconcile_all_framework_benchmark.py` before rendering: it preserves
all 142 historical rows and replaces exactly the eight AMBER-GPU/FLAME-GPU-2
10M rows with the authoritative records above, writing
`artifacts/benchmark_results_all5090_reconciled.json`. The output records every
old/new value and both timing conventions. Historical line segments retain the
archived convention (mean after trimming the slowest sample when at least three
runs were present), while each final segment leads to an outlined
endpoint that is the arithmetic mean of all ten retained final runs. AgentTorch
appears in the capability comparison but had no matched timing row.

## Scientific and monitor artifacts

| Claim area | Executable source | Raw artifact |
|---|---|---|
| Controlled SIR crossing | `scripts/emergence_threshold_controlled.py` | `artifacts/emergence_threshold_controlled.json` |
| mesa-frames local blocking | `scripts/mf_granularity.py` | `artifacts/mf_granularity.json` |
| Current monitor cost | `scripts/monitor_cost_current.py` | `artifacts/monitor_cost_current.json` |
| Sufficient-theorem checks | `scripts/theorem_referee.py` | `artifacts/theorem_referee_results.json` |
| Topological staging control | `scripts/topological_staging_experiment.py` | `artifacts/topological_staging_results.json` |
| Exploratory SIR phase boundary | released experiment snapshot | `artifacts/emergence_science_ext.json` |
| Exploratory Schelling sensitivity | released experiment snapshot | `artifacts/schelling_gpu.json` |
| Exploratory consensus sensitivity | released experiment snapshot | `artifacts/coordination_gpu.json` |
| CPU distribution checks | released benchmark snapshot | `artifacts/accuracy_results.json` |
| Historical public-example SLOC | released source audit | `artifacts/usability_results.json` |
| Calibration comparison | released calibration snapshot | `artifacts/calibration_gpu.json` |

Every mesa-frames granularity variant performs one final framework `set` per
step. The varied quantity is the number of local NumPy update blocks before
that commit. The topological-staging experiment concerns one sequential
topological reference; it is not a lower-bound proof or a statement about
global step-entry snapshot semantics.

The formal theorem concerns an event multiset expanded once from step-entry
state. Event identities, targets, read sets, and stochastic draws must therefore
be fixed independently of event order. The in-place SIR sensitivity program can
change later same-step infection opportunities and is intentionally presented
as a comparison of activation regimes, not as a theorem instance. Its headline
CPU pairing shares the initial state, per-agent random draws, and an activation
order reshuffled at every step; bootstrap resampling preserves each seed
trajectory across all transmissibility values. A single fixed order reused for
the full trajectory is retained only as a schedule-robustness condition.

## Figure regeneration

`scripts/render_publication_figures.py` is the authoritative renderer for every
figure included by `amber_aamas.tex`; `scripts/publication_figure_style.py`
defines the shared print dimensions, typography, and editorial visual system.
Run the reconciliation and renderer from the package root:

```bash
python scripts/reconcile_all_framework_benchmark.py
python scripts/render_publication_figures.py
```

The script reads only packaged source artifacts, preserves every experimental
observation, and writes deterministic vector PDFs plus 240-dpi PNG previews to
`figs/`. The manuscript includes the PDF exports. `FIGURE_MANIFEST.md` maps
each figure to its source artifact, analytical question, and chart form. Result
figures read packaged JSON; Figures 1 and 2 are deterministic vector
schematics and encode no experimental observations. Figure 2 is derived from
the public execution lifecycle and contract implementation; Algorithm 1 states
the same lifecycle. Private optimized hooks require a non-empty per-instance
`approve_fast_path(evidence)` label. The label records a caller decision;
`run()` checks its presence but does not validate the evidence. The benchmark's
released basis is a small workload smoke/invariant suite, not a matched
equivalence test of every timed kernel.

## Verification

The AMBER repository revision represented here completed 419 tests with 15
optional/GPU-dependent skips in the sandbox. The two semaphore-dependent
multiprocessing cases failed only because sandbox access was denied and passed
with normal host permissions, so all 421 collected non-skipped tests passed.
The targeted monitor/execution/TensorLane/GPU suite passed 57 tests with 10
GPU-dependent skips. The packaged code snapshot, raw artifacts, and figure
scripts provide the corresponding executable and evidence record.

## Experimental improvement campaign (post-review package)

A separate campaign implements the ChatGPT AAMAS *Experimental improvement plan*
minimum package (semantic attestation, monitor coverage/overhead, activation
effects, cold/warm native performance). Code lives under repository
`experiments/`; the recorded RTX 3090 host run is on `duypham-Z590` at
`~/AMBER_aamas_exp/experiments/raw/`. Summary:
`experiments/raw/REPORT_duypham_z590.md`. Manuscript integration notes:
`paper-fix-work/EXPERIMENT_CAMPAIGN_INTEGRATION.md`.

| Claim | Runner | Artifact |
|---|---|---|
| C1 semantic parity + negatives | `experiments/semantic/run_attestation.py` | `experiments/raw/semantic/attestation_duypham_z590.json` |
| C2 monitor coverage | `experiments/monitor/run_coverage.py` | `experiments/raw/monitor/coverage_duypham_z590.json` |
| C2 monitor overhead N,q,c | `experiments/monitor/run_overhead.py` | `experiments/raw/monitor/overhead_duypham_z590.json` |
| C3 SIR + Schelling activation | `experiments/benchmarks/run_activation.py` | `experiments/raw/semantic/activation_duypham_z590.json` |
| C4 performance (+ FLAME) | `experiments/benchmarks/run_performance.py` | `experiments/raw/performance/performance_duypham_z590.json` |

Reproduce on a CUDA host with the project venv:

```bash
export PYTHONPATH=$PWD/src:$PWD
export CUDA_PATH=/usr/local/cuda   # required for FLAME RTC
python experiments/run_all.py --tag local --out experiments/raw
python experiments/analysis/build_report.py local
```
