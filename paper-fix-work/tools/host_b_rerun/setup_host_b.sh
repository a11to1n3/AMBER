#!/usr/bin/env bash
# Bootstrap Python venv + deps on Host B (vast.ai RTX 5090).
set -euo pipefail
REPO="${1:-/workspace/AMBER_aamas_exp}"
cd "$REPO"

export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.9}"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel setuptools

# Core
pip install -e ".[dev]" 2>/dev/null || pip install -e .
pip install -r requirements.txt 2>/dev/null || true
pip install -r requirements-dev.txt 2>/dev/null || true
pip install -r benchmarks/requirements.txt 2>/dev/null || true

# CuPy for CUDA 12.x
pip install "cupy-cuda12x" || pip install cupy-cuda12x

# Scientific / experiment deps
pip install numba pytest ruff polars pyarrow matplotlib

# Optional frameworks for full multi-framework campaign
pip install mesa mesa-frames agentpy simpy 2>/dev/null || true
# Melodie may be heavy; try
pip install Melodie 2>/dev/null || true

# FLAME GPU 2: official wheelhouse (not on PyPI). Prefer CUDA 13 non-vis wheel
# for RTX 5090 / driver 13.x; fall back to default index.
# Needs: libcurand-dev (headers for RTC), and pip nvidia-cuda-nvrtc / nvidia-nvjitlink
# for libnvrtc.so.13 when the system toolkit is still CUDA 12.x.
apt-get install -y libcurand-dev-12-9 libcurand-12-9 2>/dev/null \
  || apt-get install -y libcurand-dev-13-0 2>/dev/null \
  || echo "WARN: could not apt-install libcurand-dev (RTC may fail)"
pip install "nvidia-cuda-nvrtc>=13" "nvidia-nvjitlink>=13" 2>/dev/null || true
if ! pip install --extra-index-url https://whl.flamegpu.com/whl/cuda130/ pyflamegpu; then
  pip install --extra-index-url https://whl.flamegpu.com/whl/ pyflamegpu \
    || echo "WARN: pyflamegpu not installed from flamegpu wheelhouse"
fi

# Prepend pip CUDA-13 libs so pyflamegpu can resolve libnvrtc.so.13 at import time.
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
for d in "$SITE/nvidia/cu13/lib" "$SITE/nvidia/nvjitlink/lib" "$SITE/nvidia/cuda_nvrtc/lib"; do
  if [[ -d "$d" ]]; then
    export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
  fi
done

python - <<'PY'
import cupy as cp
print("CuPy OK:", cp.__version__)
print("Device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
import ambr
print("AMBER:", getattr(ambr, "__version__", ambr.__file__))
try:
    import pyflamegpu
    print("pyflamegpu OK:", getattr(pyflamegpu, "__version__", pyflamegpu.__file__))
except Exception as exc:
    print("WARN: pyflamegpu import failed:", exc)
PY

echo "setup complete"
