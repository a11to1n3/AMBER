# Artifact v6 repair (reviewer reproducibility)

## Problem (v5)
- `python 08_scripts/prepare_v5.py` failed: expected `data/sir_crossing_host_b.json` but archive used numbered dirs only.
- Name mismatches (`benchmark_results_host_b_10m.json` vs `06_performance/benchmark_results_10m.json`).
- Missing: `src/`, experiment runners, benchmark models, specs, `reproduce.sh`, locked full baseline env.
- Figures/hashes only — not executable evidence chain.

## Repair package

**Directory:** `paper-fix-work/artifacts/AMBER_AAMAS2027_artifact_v6/`  
**Zip:** `paper-fix-work/artifacts/AMBER_AAMAS2027_artifact_v6.zip` (~2.8 MB)

### Now includes
| Path | Role |
|------|------|
| `src/` | AMBER library |
| `experiments/` | Conformance, monitor, activation, SIR crossing, analysis |
| `benchmarks/` | Multi-framework models + runners |
| `specs/` | Semantic YAML |
| `data/` | Flat canonical JSON (also rebuilt by prepare) |
| `02_rng`…`06_performance/` | Numbered Host-B evidence tree |
| `08_scripts/prepare_v5.py` | Maps numbered dirs → `data/` (both layouts) |
| `make_figures.py` | Regenerates all package figures from JSON |
| `reproduce.sh` | `./reproduce.sh figures` entry point |
| `MANIFEST.json` | SHA-256 inventory |
| `requirements-lock.txt` / `environment.yml` | Figure regen env |
| `00_environment/environment_full.txt` | GPU, thread policy, baseline packages |

### Verified clean-room
```bash
unzip AMBER_AAMAS2027_artifact_v6.zip
cd AMBER_AAMAS2027_artifact_v6
pip install matplotlib numpy   # or requirements-lock.txt
./reproduce.sh figures         # EXIT 0 → 11 figure pairs (png+pdf)
```

### Reviewer mapping
| Reviewer expectation | v6 |
|----------------------|-----|
| `data/sir_crossing_host_b.json` | Prepared from `04_activation/` |
| `benchmark_results_host_b_10m.json` | Alias of `06_performance/benchmark_results_10m.json` |
| Source + runners + models + YAML | Present |
| `./reproduce.sh figures` | Implemented & tested |
| Full baseline versions | Documented in `00_environment/environment_full.txt` |

### Residual note
Exact **patch** versions for Mesa / mesa-frames / AgentPy / Melodie / SimPy should be re-exported via `pip freeze` on the campaign Host-B venv if still available and pasted into `environment_full.txt`. Core versions (Python 3.12.3, numpy 2.5.1, cupy 14.1.1, polars 1.43.0, pyflamegpu 2.0.0rc4+cuda130, Julia 1.11.5, Agents.jl 7.0.3) are already recorded from the campaign.
