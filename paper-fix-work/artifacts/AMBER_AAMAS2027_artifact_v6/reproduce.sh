#!/usr/bin/env bash
# Anonymous AAMAS artifact entry point.
# Usage:
#   ./reproduce.sh figures     # regenerate all package figures from data/*.json
#   ./reproduce.sh prepare     # normalize data/ layout only
#   ./reproduce.sh check       # prepare + import smoke (no GPU required for figures)
#   ./reproduce.sh help
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

cmd="${1:-figures}"

run_prepare() {
  python3 08_scripts/prepare_v5.py
}

run_figures() {
  run_prepare
  python3 make_figures.py
  echo "OK: figures written under figs/"
  ls -1 figs/*.{png,pdf} 2>/dev/null | head -40 || true
}

run_check() {
  run_prepare
  python3 - <<'PY'
import json
from pathlib import Path
required = [
    "sir_crossing_host_b.json",
    "attestation_host_b.json",
    "activation_host_b.json",
    "boundary_matrix_host_b.json",
    "overhead_host_b.json",
    "performance_host_b.json",
    "benchmark_results_host_b_10m.json",
]
data = Path("data")
for name in required:
    p = data / name
    assert p.is_file(), name
    json.loads(p.read_text())
print("check: required JSON present and parseable")
# import package surface
import sys
sys.path.insert(0, "src")
import ambr  # noqa: F401
print("check: import ambr OK")
PY
}

case "$cmd" in
  figures) run_figures ;;
  prepare) run_prepare ;;
  check) run_check ;;
  help|-h|--help)
    sed -n '1,12p' "$0"
    echo
    echo "Full experiment re-runs (GPU host; optional):"
    echo "  export PYTHONPATH=\$PWD/src:\$PWD"
    echo "  python experiments/semantic/run_attestation.py --tag host_b --out raw_rerun/03"
    echo "  python experiments/semantic/run_production_attestation.py --tag host_b --out raw_rerun/04"
    echo "  python experiments/monitor/run_boundary_matrix.py --tag host_b --out raw_rerun/05"
    echo "  python experiments/monitor/run_overhead.py --tag host_b --out raw_rerun/05"
    echo "  python experiments/benchmarks/run_activation.py --tag host_b --out raw_rerun/06"
    echo "  python experiments/benchmarks/run_sir_crossing.py --tag host_b --out raw_rerun/06"
    echo "  python benchmarks/run_all_frameworks.py --agents 10000000 --steps 50 --runs 10 \\"
    echo "      --frameworks \"AMBER (GPU)\" \"FLAME GPU 2\" --tag host_b_10m --budget 1200"
    ;;
  *)
    echo "unknown command: $cmd (try: figures|prepare|check|help)" >&2
    exit 2
    ;;
esac
