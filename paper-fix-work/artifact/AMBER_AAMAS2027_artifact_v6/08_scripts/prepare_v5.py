#!/usr/bin/env python3
"""Normalize packaged evidence into a flat data/ layout for figure scripts.

Accepts either layout:

  (A) Numbered campaign dirs (as in Host-B pull):
        02_rng/, 03_conformance*/, 04_activation/, 05_monitor/, 06_performance/

  (B) Already-flat data/ with canonical names.

Always writes/refreshes ROOT/data/ with stable filenames expected by
make_figures.py and by the reviewer checklist.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


# (canonical data/ name, candidate relative paths inside artifact)
MAP: list[tuple[str, list[str]]] = [
    (
        "sir_crossing_host_b.json",
        [
            "data/sir_crossing_host_b.json",
            "04_activation/sir_crossing_host_b.json",
            "06_activation/sir_crossing_host_b.json",
            "04_activation/sir_crossing_host_a.json",
        ],
    ),
    (
        "attestation_host_b.json",
        [
            "data/attestation_host_b.json",
            "03_conformance/attestation_host_b.json",
            "03_conformance_gpu_style/attestation_host_b.json",
        ],
    ),
    (
        "production_attestation_host_b.json",
        [
            "data/production_attestation_host_b.json",
            "03_conformance/production_attestation_host_b.json",
            "04_conformance_native/production_attestation_host_b.json",
        ],
    ),
    (
        "activation_host_b.json",
        [
            "data/activation_host_b.json",
            "04_activation/activation_host_b.json",
            "06_activation/activation_host_b.json",
        ],
    ),
    (
        "boundary_matrix_host_b.json",
        [
            "data/boundary_matrix_host_b.json",
            "05_monitor/boundary_matrix_host_b.json",
        ],
    ),
    (
        "overhead_host_b.json",
        [
            "data/overhead_host_b.json",
            "05_monitor/overhead_host_b.json",
        ],
    ),
    (
        "performance_host_b.json",
        [
            "data/performance_host_b.json",
            "06_performance/performance_host_b.json",
            "07_performance/performance_host_b.json",
        ],
    ),
    (
        "benchmark_results_host_b_10m.json",
        [
            "data/benchmark_results_host_b_10m.json",
            "data/benchmark_results_10m.json",
            "06_performance/benchmark_results_10m.json",
            "06_performance/benchmark_results_host_b_10m.json",
            "07_performance/benchmark_results/benchmark_results_host_b_rtx5090_20260727T071054Z_10m.json",
        ],
    ),
    (
        "benchmark_results_10m.json",
        [
            "data/benchmark_results_10m.json",
            "data/benchmark_results_host_b_10m.json",
            "06_performance/benchmark_results_10m.json",
        ],
    ),
    (
        "sir_crossing_seed_level_host_b.json",
        [
            "data/sir_crossing_seed_level_host_b.json",
            "04_activation/sir_crossing_seed_level_host_b.json",
            "06_activation/sir_crossing_seed_level_host_b.json",
        ],
    ),
    (
        "benchmark_results_wealth_clean.json",
        [
            "data/benchmark_results_wealth_clean.json",
            "06_performance/benchmark_results_wealth_clean.json",
        ],
    ),
    (
        "benchmark_results_walk_clean.json",
        [
            "data/benchmark_results_walk_clean.json",
            "06_performance/benchmark_results_walk_clean.json",
        ],
    ),
    (
        "benchmark_results_agentsjl.json",
        [
            "data/benchmark_results_agentsjl.json",
            "06_performance/benchmark_results_agentsjl.json",
        ],
    ),
]


def resolve(cands: list[str]) -> Path | None:
    for rel in cands:
        p = ROOT / rel
        if p.is_file():
            return p
    return None


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    copied = 0
    for dest_name, cands in MAP:
        src = resolve(cands)
        if src is None:
            # only required keys fail hard later in make_figures
            missing.append(dest_name)
            continue
        dest = DATA / dest_name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        copied += 1
        print(f"ok  {dest_name}  <-  {src.relative_to(ROOT)}")

    # dual alias for 10m
    a = DATA / "benchmark_results_host_b_10m.json"
    b = DATA / "benchmark_results_10m.json"
    if a.is_file() and not b.is_file():
        shutil.copy2(a, b)
    if b.is_file() and not a.is_file():
        shutil.copy2(b, a)

    required = {
        "sir_crossing_host_b.json",
        "attestation_host_b.json",
        "activation_host_b.json",
        "boundary_matrix_host_b.json",
        "overhead_host_b.json",
        "performance_host_b.json",
        "benchmark_results_host_b_10m.json",
    }
    still_missing = [n for n in required if not (DATA / n).is_file()]
    if still_missing:
        print("ERROR: required data files still missing:", still_missing, file=sys.stderr)
        print("Searched from", ROOT, file=sys.stderr)
        return 1
    print(f"prepared {copied} files into {DATA}")
    if missing:
        print("optional missing:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
