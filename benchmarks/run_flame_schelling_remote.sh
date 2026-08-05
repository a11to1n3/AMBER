#!/usr/bin/env bash
# FLAME GPU 2 Schelling benchmark on a CUDA machine (large-N).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.9}"
export FLAMEGPU_SHARE_USAGE_STATISTICS=False
export FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE=true
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:${LD_LIBRARY_PATH:-}"

# NVRTC needs curand headers under CUDA_PATH/include (some cloud CUDA images
# only ship them inside the pip nvidia-* packages).
CURAND_INC="/usr/local/lib/python3.12/dist-packages/nvidia/curand/include"
if [[ -d "$CURAND_INC" ]]; then
  for hdr in "$CURAND_INC"/*.h; do
    ln -sf "$hdr" "$CUDA_PATH/include/$(basename "$hdr")" 2>/dev/null || true
  done
fi

cd "$ROOT"
# run_all_frameworks.py lives in benchmarks/; keep cwd here.
mkdir -p results
python3 run_all_frameworks.py \
  --frameworks "FLAME GPU 2" \
  --models schelling \
  --agents 1000 10000 100000 1000000 10000000 \
  --steps 50 \
  --runs 10 \
  --budget 120 \
  --tag flame_schelling_cuda
