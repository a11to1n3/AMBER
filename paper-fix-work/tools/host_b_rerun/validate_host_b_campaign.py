#!/usr/bin/env python3
"""Validate Host-B unified campaign artifact tree."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def load_json(path: Path):
    text = path.read_text()
    if "NaN" in text or "Infinity" in text or "-Infinity" in text:
        # allow only if proper JSON null used; bare NaN is invalid
        if ": NaN" in text or ": Infinity" in text or ": -Infinity" in text:
            raise ValueError(f"non-standard NaN/Infinity in {path}")
    return json.loads(text)


def check_no_nan(obj, path=""):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        raise ValueError(f"NaN/Inf at {path}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            check_no_nan(v, f"{path}.{k}")
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            check_no_nan(v, f"{path}[{i}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_root", type=Path)
    args = ap.parse_args()
    root = args.artifact_root
    errors = []
    warnings = []

    required = [
        "00_environment/gpu.txt",
        "00_environment/python.txt",
        "00_environment/GIT_COMMIT.txt" if (root / "GIT_COMMIT.txt").exists() else "GIT_COMMIT.txt",
        "02_rng/rng_test_vectors_host_b.json",
        "02_rng/rng_cuda_kernel.json",
        "03_conformance_gpu_style",
        "04_conformance_native",
        "05_monitor",
        "06_activation",
        "07_performance",
    ]
    # flexible required
    if not (root / "GIT_COMMIT.txt").exists() and not (root / "00_environment" / "git.txt").exists():
        errors.append("missing GIT_COMMIT / git capture")

    for rel in [
        "00_environment/gpu.txt",
        "02_rng/rng_test_vectors_host_b.json",
        "02_rng/rng_cuda_kernel.json",
    ]:
        if not (root / rel).exists():
            errors.append(f"missing {rel}")

    # GPU must be 5090
    gpu_txt = root / "00_environment" / "gpu.txt"
    if gpu_txt.exists():
        g = gpu_txt.read_text()
        if "5090" not in g:
            errors.append("GPU capture does not mention RTX 5090")

    # Scan JSON artifacts
    for path in root.rglob("*.json"):
        try:
            data = load_json(path)
            check_no_nan(data, str(path))
        except Exception as e:
            errors.append(f"{path}: {e}")

    # Native attestation
    native_hits = list((root / "04_conformance_native").rglob("production_attestation*.json"))
    if not native_hits:
        warnings.append("no production_attestation*.json under 04_conformance_native")
    else:
        d = load_json(native_hits[0])
        gate = d.get("submission_gate", {})
        if not gate.get("exact_timed_private_kernels_zero_mismatch"):
            errors.append("production attestation gate false")
        for w in d.get("workloads", []):
            if not w.get("production_kernel_attested"):
                errors.append(f"workload not attested: {w.get('workload')}")

    # GPU-style attestation
    style_hits = list((root / "03_conformance_gpu_style").rglob("attestation*.json"))
    if not style_hits:
        warnings.append("no attestation*.json under 03_conformance_gpu_style")
    else:
        d = load_json(style_hits[0])
        if not d.get("positive_backends_zero_mismatch"):
            errors.append("gpu-style positive backends not zero-mismatch")
        if not d.get("negative_all_detected"):
            errors.append("gpu-style negatives not all detected")

    # Crossing
    cross = list((root / "06_activation").rglob("sir_crossing*.json"))
    if not cross:
        warnings.append("missing sir_crossing JSON")
    else:
        d = load_json(cross[0])
        if not d.get("acceptance", {}).get("crossing_0.5_shift_excludes_zero"):
            warnings.append("crossing@0.5 does not exclude zero (report honestly)")

    report = {
        "artifact_root": str(root),
        "errors": errors,
        "warnings": warnings,
        "ok": len(errors) == 0,
    }
    print(json.dumps(report, indent=2))
    (root / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
