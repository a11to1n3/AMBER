# Artifact manifest (campaign cleanup)

All paths relative to `<artifact-root>/` (repository root or release package root).

## Authoritative

| File | Role | Hardware label |
|---|---|---|
| `experiments/raw/semantic/production_attestation_host_a.json` | P1 production-kernel attestations | host_a (RTX 3090) |
| `experiments/raw/semantic/attestations/*.json` | Per-workload immutable attestation records | host_a |
| `experiments/raw/semantic/sir_crossing_host_a.json` | P2 attack-rate crossing (when complete) | host_a |
| `experiments/raw/monitor/boundary_matrix_host_a.json` | P4 report boundary matrix | host_a |
| `experiments/raw/monitor/overhead_host_a.json` | Overhead surface; variable `q` means commit count **k** | host_a |
| `paper-fix-work/artifacts/benchmark_results_all5090_reconciled.json` | All-framework + reconciled 10M endpoints | host_b (RTX 5090) |
| `paper-fix-work/campaign_results/figs_all_frameworks/panelA_*.png` | Historical multi-framework context | host_b |
| `paper-fix-work/campaign_results/figs_all_frameworks/panelB_*.png` | Authoritative 10M endpoints | host_b |
| `experiments/raw/CLAIM_DASHBOARD.md` | Honest claim status | — |

## Superseded / non-authoritative

| File | Disposition |
|---|---|
| `experiments/raw/semantic/attestation_smoke.json` | Move to `experiments/raw/development_prechecks/` |
| `experiments/raw/semantic/attestation_host_a.json` | Prior GPU-style (not production-kernel) parity; retain as development evidence |
| Tags containing personal hostnames | Prefer `host_a` / `host_b` in new artifacts |

## Manuscript mapping (v4 draft)

| Paper element | Artifact |
|---|---|
| §5.1 Semantic attestation table | `production_attestation_host_a.json` |
| §5.2 SIR crossing | `sir_crossing_host_a.json` |
| §5.3 Report boundary + overhead | `boundary_matrix_*.json`, overhead with **k** notation |
| §5.4 Performance panels | panelA + panelB figures; 3090 crossover sentence from performance JSON |

## Overhead notation correction (no rerun)

Measured factors:

- \(N\) = population size
- \(c\) = schema column count
- \(k\) = number of observed **column commits** per step (formerly labeled `q` in JSON)

The theorem’s \(q\) = buffered **cell writes**. The vectorized experiment does **not** evaluate the \(q=\Theta(Nc)\) OOP regime.

## Reproduce commands

```bash
export PYTHONPATH=<artifact-root>/src:<artifact-root>
export CUDA_PATH=/usr/local/cuda
python experiments/semantic/run_production_attestation.py --tag host_a
python experiments/benchmarks/run_sir_crossing.py --tag host_a
python experiments/monitor/run_boundary_matrix.py --tag host_a
python experiments/analysis/plot_performance_two_panel.py
```
