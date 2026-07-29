#!/usr/bin/env bash
# AMBER variant benchmark on CUDA host (RTX 5090).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.9}"
export PYTHONPATH="${ROOT}/../src:${ROOT}"
export PYTHONUNBUFFERED=1

cd "$ROOT"
mkdir -p results
python3 run_all_frameworks.py \
  --frameworks "AMBER (loop)" "AMBER (vectorized)" "AMBER (GPU)" \
  --agents 1000 10000 100000 1000000 10000000 \
  --steps 50 \
  --runs 10 \
  --budget 120 \
  --tag amber_rerun5090