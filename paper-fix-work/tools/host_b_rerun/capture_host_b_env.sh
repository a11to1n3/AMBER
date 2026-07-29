#!/usr/bin/env bash
# Capture Host-B hardware/software environment for the unified AAMAS campaign.
set -euo pipefail

OUT="${1:-./00_environment}"
mkdir -p "$OUT"

{
  echo "=== timestamp_utc ==="
  date -u +%Y-%m-%dT%H:%M:%SZ
  echo "=== hostname ==="
  hostname
  echo "=== uname ==="
  uname -a
  echo "=== os-release ==="
  cat /etc/os-release 2>/dev/null || true
} > "$OUT/os.txt"

{
  echo "=== nvidia-smi -L ==="
  nvidia-smi -L 2>&1 || true
  echo "=== nvidia-smi full ==="
  nvidia-smi 2>&1 || true
  echo "=== nvidia-smi query ==="
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.free,compute_cap,clocks.sm,clocks.mem,power.limit,temperature.gpu,pstate --format=csv 2>&1 || true
} > "$OUT/gpu.txt"

{
  echo "=== CUDA_PATH ==="
  echo "${CUDA_PATH:-}"
  echo "=== nvcc ==="
  nvcc --version 2>&1 || true
  echo "=== cuda libs ==="
  ls -la "${CUDA_PATH:-/usr/local/cuda}/lib64" 2>/dev/null | head -30 || true
} > "$OUT/cuda.txt"

{
  echo "=== cpuinfo model ==="
  grep -m1 'model name' /proc/cpuinfo || true
  echo "=== nproc ==="
  nproc
  echo "=== lscpu ==="
  lscpu 2>&1 || true
  echo "=== meminfo ==="
  free -h
  echo "=== numa ==="
  numactl --hardware 2>&1 || true
  echo "=== governor ==="
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "n/a"
} > "$OUT/cpu_mem.txt"

{
  echo "=== python ==="
  which python
  python --version
  python - <<'PY'
import sys, json
mods = {}
for name in ("numpy","cupy","polars","pytest","numba","pyflamegpu","ambr"):
    try:
        m = __import__(name)
        mods[name] = getattr(m, "__version__", getattr(m, "__file__", "ok"))
    except Exception as e:
        mods[name] = f"MISSING: {e}"
print(json.dumps(mods, indent=2))
try:
    import cupy as cp
    print("cupy_device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    print("cupy_runtime:", cp.cuda.runtime.runtimeGetVersion())
except Exception as e:
    print("cupy_probe_error:", e)
PY
  echo "=== pip freeze ==="
  pip freeze 2>/dev/null || true
} > "$OUT/python.txt"

{
  echo "=== julia ==="
  which julia 2>&1 || true
  julia --version 2>&1 || true
} > "$OUT/julia.txt"

{
  echo "=== env ==="
  env | sort | grep -E 'CUDA|CUPY|PYTHON|OMP|MKL|NUMBA|AMBER|PATH|LD_LIBRARY' || true
} > "$OUT/env.txt"

{
  echo "=== git ==="
  git -C "${AMBER_REPO:-.}" rev-parse HEAD 2>/dev/null || echo "no-git"
  git -C "${AMBER_REPO:-.}" status --porcelain 2>/dev/null || true
} > "$OUT/git.txt"

# Source hashes for kernels/experiments
if command -v sha256sum >/dev/null; then
  (
    cd "${AMBER_REPO:-.}"
    find experiments src/ambr/gpu_kernels.py benchmarks/models -type f \( -name '*.py' -o -name '*.cuh' -o -name '*.cu' \) 2>/dev/null \
      | sort | xargs -r sha256sum
  ) > "$OUT/source_hashes.sha256" || true
fi

echo "wrote environment capture under $OUT"
