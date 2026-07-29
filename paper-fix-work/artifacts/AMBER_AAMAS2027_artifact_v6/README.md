# AMBER AAMAS 2027 — Anonymous Computational Artifact (v6)

This package reconstructs the **Host-B (RTX 5090)** experimental evidence chain
and regenerates every package figure from the included JSON.

## Layout

```
src/                 AMBER library (ambr)
experiments/         Semantic / monitor / activation / analysis runners
benchmarks/          Multi-framework scale models + run_all_frameworks.py
specs/               Semantic YAML for wealth / walk / SIR / Schelling
data/                Flat canonical JSON (filled by prepare_v5.py)
02_rng/ … 06_performance/   Numbered evidence tree (Host-B campaign)
00_environment/      Recorded software / GPU / thread settings
08_scripts/          prepare_v5.py (path normalizer)
make_figures.py      Figure regeneration from data/*.json
figs/                Submitted figures (PNG + PDF)
reproduce.sh         Entry point
requirements-lock.txt
environment.yml
MANIFEST.json
```

## Minimal review command (no GPU)

From a **clean extracted** directory:

```bash
python3 -m venv .venv && source .venv/bin/activate   # or your env
pip install -r requirements-lock.txt                 # or: pip install matplotlib numpy
./reproduce.sh figures
```

This must succeed with only packaged JSON (no re-run of GPU experiments).

## Full evidence re-run (GPU host, optional)

```bash
export PYTHONPATH=$PWD/src:$PWD
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Conformance
python experiments/semantic/run_attestation.py --tag host_b --out raw_rerun/conformance_gpu
python experiments/semantic/run_production_attestation.py --tag host_b --out raw_rerun/conformance_native
# Monitor
python experiments/monitor/run_boundary_matrix.py --tag host_b --out raw_rerun/monitor
python experiments/monitor/run_overhead.py --tag host_b --out raw_rerun/monitor
# Activation + SIR crossing
python experiments/benchmarks/run_activation.py --tag host_b --out raw_rerun/activation
python experiments/benchmarks/run_sir_crossing.py --tag host_b --out raw_rerun/activation
# 10M endpoints
python benchmarks/run_all_frameworks.py \
  --agents 10000000 --steps 50 --runs 10 --budget 1200 \
  --frameworks "AMBER (GPU)" "FLAME GPU 2" \
  --tag host_b_10m
```

See `00_environment/environment_full.txt` for locked baseline versions
(mesa, mesa-frames, AgentPy, Melodie, SimPy, Julia, Agents.jl, CuPy, pyflamegpu, …)
and thread/affinity notes.

## Anonymity

This package uses only host tags `host_b` / `host_a`. Personal hostnames and
account paths are stripped from packaged text.
