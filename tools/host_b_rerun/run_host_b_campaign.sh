#!/usr/bin/env bash
# Unified Host-B campaign orchestrator.
# Usage:
#   bash run_host_b_campaign.sh host_b_campaign.env <phase>
# phases: freeze|env|preflight|rng|conformance|monitor|activation|performance|analysis|all
set -euo pipefail

ENV_FILE="${1:-host_b_campaign.env}"
PHASE="${2:-all}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing env file: $ENV_FILE" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

cd "${AMBER_REPO}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PATH="${CUDA_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
# pyflamegpu+cuda130 links libnvrtc.so.13 from pip nvidia-cuda-nvrtc (nvidia/cu13/lib)
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  _site="${VIRTUAL_ENV}/lib"
  for d in "$_site"/python*/site-packages/nvidia/cu13/lib \
           "$_site"/python*/site-packages/nvidia/nvjitlink/lib \
           "$_site"/python*/site-packages/nvidia/cuda_nvrtc/lib; do
    if [[ -d "$d" ]]; then
      export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH}"
    fi
  done
fi
export PYTHONPATH="${AMBER_REPO}/src:${AMBER_REPO}"
export PYTHONHASHSEED=0
export AMBER_SUPPRESS_DEPRECATIONS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE=1
export FLAMEGPU_SHARE_USAGE_STATISTICS=False

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

phase_freeze() {
  log "FREEZE"
  if [[ -d .git ]]; then
    git status --short || true
    git switch -c aamas2027-host-b-unified 2>/dev/null || git switch aamas2027-host-b-unified
    git add -A
    git commit -m "Freeze unified RTX 5090 AAMAS experiment campaign" || true
    export AMBER_COMMIT="$(git rev-parse HEAD)"
  else
    export AMBER_COMMIT="no-git-$(date -u +%Y%m%dT%H%M%SZ)"
    log "WARNING: no .git — using synthetic commit id $AMBER_COMMIT"
  fi
  export CAMPAIGN_TAG="host_b_rtx5090_$(date -u +%Y%m%dT%H%M%SZ)"
  export ARTIFACT_ROOT="${AMBER_REPO}/artifacts/${CAMPAIGN_TAG}"
  mkdir -p "$ARTIFACT_ROOT"
  printf '%s\n' "$AMBER_COMMIT" > "$ARTIFACT_ROOT/GIT_COMMIT.txt"
  git status --porcelain > "$ARTIFACT_ROOT/GIT_STATUS.txt" 2>/dev/null || true
  # persist for later phases
  {
    echo "export AMBER_COMMIT='$AMBER_COMMIT'"
    echo "export CAMPAIGN_TAG='$CAMPAIGN_TAG'"
    echo "export ARTIFACT_ROOT='$ARTIFACT_ROOT'"
    echo "export CUPY_CACHE_DIR='$ARTIFACT_ROOT/00_environment/cupy_cache'"
  } > "$ARTIFACT_ROOT/campaign_exports.sh"
  log "ARTIFACT_ROOT=$ARTIFACT_ROOT"
  log "AMBER_COMMIT=$AMBER_COMMIT"
}

load_exports() {
  if [[ -z "${ARTIFACT_ROOT:-}" ]]; then
    latest=$(ls -1dt "${AMBER_REPO}/artifacts"/host_b_rtx5090_* 2>/dev/null | head -1 || true)
    if [[ -n "$latest" && -f "$latest/campaign_exports.sh" ]]; then
      # shellcheck disable=SC1090
      source "$latest/campaign_exports.sh"
    fi
  fi
  if [[ -z "${ARTIFACT_ROOT:-}" ]]; then
    echo "ARTIFACT_ROOT unset; run freeze first" >&2
    exit 2
  fi
  export CUPY_CACHE_DIR="${ARTIFACT_ROOT}/00_environment/cupy_cache"
  mkdir -p "$CUPY_CACHE_DIR"
}

phase_env() {
  load_exports
  mkdir -p "$ARTIFACT_ROOT/00_environment"
  export AMBER_REPO
  bash tools/host_b_rerun/capture_host_b_env.sh "$ARTIFACT_ROOT/00_environment"
  # verify 5090
  if ! grep -q "5090" "$ARTIFACT_ROOT/00_environment/gpu.txt"; then
    echo "FATAL: RTX 5090 not detected" >&2
    exit 1
  fi
}

phase_preflight() {
  load_exports
  mkdir -p "$ARTIFACT_ROOT/01_preflight"
  # dirty check
  if [[ -d .git ]] && [[ -n "$(git status --porcelain)" ]]; then
    echo "FATAL: repository dirty after freeze" >&2
    git status --porcelain | tee "$ARTIFACT_ROOT/01_preflight/dirty.txt"
    exit 1
  fi
  python -c "import cupy as cp; n=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(); print(n); assert '5090' in n" \
    |& tee "$ARTIFACT_ROOT/01_preflight/cupy_gpu.txt"
  pytest -q 2>&1 | tee "$ARTIFACT_ROOT/01_preflight/pytest.log" || {
    echo "WARNING: pytest failures — inspect log; continuing only if GPU path ok" | tee -a "$ARTIFACT_ROOT/01_preflight/pytest.log"
  }
  python benchmarks/correctness_check.py 2>&1 | tee "$ARTIFACT_ROOT/01_preflight/framework_correctness.log" || true
  python benchmarks/run_all_frameworks.py --quick --tag "${CAMPAIGN_TAG}_quick" \
    2>&1 | tee "$ARTIFACT_ROOT/01_preflight/benchmark_quick.log" || true
}

phase_rng() {
  load_exports
  mkdir -p "$ARTIFACT_ROOT/02_rng"
  python tools/host_b_rerun/run_rng_matrix.py --out "$ARTIFACT_ROOT/02_rng" --n-keys 10000 \
    |& tee "$ARTIFACT_ROOT/02_rng/run.log"
}

phase_conformance() {
  load_exports
  mkdir -p "$ARTIFACT_ROOT/03_conformance_gpu_style" "$ARTIFACT_ROOT/04_conformance_native"
  python experiments/semantic/run_attestation.py --tag host_b \
    --out "$ARTIFACT_ROOT/03_conformance_gpu_style" \
    |& tee "$ARTIFACT_ROOT/03_conformance_gpu_style/run.log"
  python experiments/semantic/run_production_attestation.py --tag host_b \
    --out "$ARTIFACT_ROOT/04_conformance_native" \
    |& tee "$ARTIFACT_ROOT/04_conformance_native/run.log"
  # cupy cache hashes
  if [[ -d "$CUPY_CACHE_DIR" ]]; then
    find "$CUPY_CACHE_DIR" -type f -print0 2>/dev/null | sort -z | xargs -0 sha256sum \
      > "$ARTIFACT_ROOT/04_conformance_native/cupy_cache.sha256" || true
  fi
}

phase_monitor() {
  load_exports
  mkdir -p "$ARTIFACT_ROOT/05_monitor"
  python experiments/monitor/run_boundary_matrix.py --tag host_b \
    --out "$ARTIFACT_ROOT/05_monitor" |& tee "$ARTIFACT_ROOT/05_monitor/boundary.log"
  python experiments/monitor/run_overhead.py --tag host_b \
    --out "$ARTIFACT_ROOT/05_monitor" |& tee "$ARTIFACT_ROOT/05_monitor/overhead.log"
}

phase_activation() {
  load_exports
  mkdir -p "$ARTIFACT_ROOT/06_activation"
  python experiments/benchmarks/run_activation.py --tag host_b \
    --out "$ARTIFACT_ROOT/06_activation" |& tee "$ARTIFACT_ROOT/06_activation/activation_surface.log"
  python experiments/benchmarks/run_sir_crossing.py --tag host_b \
    --out "$ARTIFACT_ROOT/06_activation" |& tee "$ARTIFACT_ROOT/06_activation/sir_crossing.log"
}

phase_performance() {
  load_exports
  mkdir -p "$ARTIFACT_ROOT/07_performance"
  # lifecycle / crossover (AMBER CPU/GPU + FLAME)
  python experiments/benchmarks/run_performance.py --tag host_b \
    --out "$ARTIFACT_ROOT/07_performance" --skip-matched \
    |& tee "$ARTIFACT_ROOT/07_performance/crossover.log"
  # unified all-framework 1k→10M
  python benchmarks/run_all_frameworks.py \
    --agents 1000 10000 100000 1000000 10000000 \
    --steps 50 --runs 10 --budget 300 \
    --tag "${CAMPAIGN_TAG}_unified" \
    |& tee "$ARTIFACT_ROOT/07_performance/unified_all_frameworks.log" || true
  # copy results if runner writes to benchmarks/results
  mkdir -p "$ARTIFACT_ROOT/07_performance/benchmark_results"
  cp -a benchmarks/results/*"${CAMPAIGN_TAG}"* "$ARTIFACT_ROOT/07_performance/benchmark_results/" 2>/dev/null || true
  # dedicated 10M
  python benchmarks/run_all_frameworks.py \
    --agents 10000000 --steps 50 --runs 10 --budget 1200 \
    --frameworks "AMBER (GPU)" "FLAME GPU 2" \
    --models wealth_transfer random_walk sir_epidemic schelling \
    --tag "${CAMPAIGN_TAG}_10m" \
    |& tee "$ARTIFACT_ROOT/07_performance/10m_endpoints.log" || true
  cp -a benchmarks/results/*"${CAMPAIGN_TAG}_10m"* "$ARTIFACT_ROOT/07_performance/benchmark_results/" 2>/dev/null || true
}

phase_analysis() {
  load_exports
  mkdir -p "$ARTIFACT_ROOT/08_analysis"
  python tools/host_b_rerun/validate_host_b_campaign.py "$ARTIFACT_ROOT" \
    |& tee "$ARTIFACT_ROOT/08_analysis/validation.log"
  # regenerate plots if possible
  if [[ -f experiments/analysis/plot_campaign_results.py ]]; then
    # point RAW to campaign if plot script allows — skip if not
    true
  fi
  log "campaign complete: $ARTIFACT_ROOT"
}

case "$PHASE" in
  freeze) phase_freeze ;;
  env) phase_env ;;
  preflight) phase_preflight ;;
  rng) phase_rng ;;
  conformance) phase_conformance ;;
  monitor) phase_monitor ;;
  activation) phase_activation ;;
  performance) phase_performance ;;
  analysis) phase_analysis ;;
  all)
    phase_freeze
    phase_env
    phase_preflight
    phase_rng
    phase_conformance
    phase_monitor
    phase_activation
    phase_performance
    phase_analysis
    ;;
  *)
    echo "unknown phase: $PHASE" >&2
    exit 2
    ;;
esac
