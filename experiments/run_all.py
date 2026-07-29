#!/usr/bin/env python3
"""Run the minimum AAMAS experimental package and write a consolidated report."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_step(label, cmd, env=None):
    print(f"\n######## {label} ########", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT.parent), env=env)
    return {
        "label": label,
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "raw")
    ap.add_argument("--tag", default="host_a")
    ap.add_argument("--quick", action="store_true", help="smaller budgets for smoke")
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--skip-activation", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    tag = args.tag
    quick = ["--quick"] if args.quick else []

    steps = []
    steps.append(run_step(
        "E1_semantic_attestation",
        [py, str(ROOT / "semantic" / "run_attestation.py"), "--out", str(args.out / "semantic"), "--tag", tag, *quick],
    ))
    steps.append(run_step(
        "E2a_monitor_coverage",
        [py, str(ROOT / "monitor" / "run_coverage.py"), "--out", str(args.out / "monitor"), "--tag", tag],
    ))
    steps.append(run_step(
        "E2b_monitor_overhead",
        [py, str(ROOT / "monitor" / "run_overhead.py"), "--out", str(args.out / "monitor"), "--tag", tag, *quick],
    ))
    if not args.skip_activation:
        steps.append(run_step(
            "E3_activation",
            [py, str(ROOT / "benchmarks" / "run_activation.py"), "--out", str(args.out / "semantic"), "--tag", tag, *quick],
        ))
    if not args.skip_perf:
        steps.append(run_step(
            "E4_performance",
            [py, str(ROOT / "benchmarks" / "run_performance.py"), "--out", str(args.out / "performance"), "--tag", tag, *quick],
        ))

    # Load artifacts
    def load(p):
        path = Path(p)
        if path.exists():
            return json.loads(path.read_text())
        return None

    att = load(args.out / "semantic" / f"attestation_{tag}.json")
    cov = load(args.out / "monitor" / f"coverage_{tag}.json")
    ovh = load(args.out / "monitor" / f"overhead_{tag}.json")
    act = load(args.out / "semantic" / f"activation_{tag}.json")
    perf = load(args.out / "performance" / f"performance_{tag}.json")

    claims = {
        "C1_semantic_parity": (att or {}).get("acceptance", {}).get("C1_semantic_parity"),
        "C1_negative_controls": (att or {}).get("acceptance", {}).get("negative_controls"),
        "C2_monitor_boundary": (cov or {}).get("acceptance", {}).get("C2_monitor_boundary_reported"),
        "C2_hazards_detected": (cov or {}).get("acceptance", {}).get("all_public_hazards_detected"),
        "C3_sir_activation": (act or {}).get("acceptance", {}).get("C3_activation_effect_sir"),
        "C3_schelling_activation": (act or {}).get("acceptance", {}).get("C3_activation_effect_schelling"),
        "C4_performance_campaign": perf is not None,
    }

    report = {
        "tag": tag,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "quick": args.quick,
        "steps": steps,
        "claims": claims,
        "artifacts": {
            "attestation": str(args.out / "semantic" / f"attestation_{tag}.json"),
            "coverage": str(args.out / "monitor" / f"coverage_{tag}.json"),
            "overhead": str(args.out / "monitor" / f"overhead_{tag}.json"),
            "activation": str(args.out / "semantic" / f"activation_{tag}.json"),
            "performance": str(args.out / "performance" / f"performance_{tag}.json"),
        },
        "attestation_summary": {
            "positive_ok": (att or {}).get("positive_backends_zero_mismatch"),
            "negatives": (att or {}).get("negative_controls_detected"),
        } if att else None,
        "monitor_summary": (cov or {}).get("summary"),
        "activation_summary": (act or {}).get("acceptance"),
        "performance_speedups": (perf or {}).get("speedups_gpu_vs_reference"),
        "overhead_points": len((ovh or {}).get("overhead") or []),
    }

    out_json = args.out / f"REPORT_{tag}.json"
    out_md = args.out / f"REPORT_{tag}.md"
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        f"# AAMAS Experimental Package Report — `{tag}`",
        "",
        f"- Host: `{report['host']}`",
        f"- Platform: `{report['platform']}`",
        f"- Quick mode: `{args.quick}`",
        "",
        "## Claims",
        "",
        "| Claim | Result |",
        "|---|---|",
    ]
    for k, v in claims.items():
        lines.append(f"| {k} | `{v}` |")
    lines += [
        "",
        "## Step return codes",
        "",
    ]
    for s in steps:
        lines.append(f"- **{s['label']}**: rc={s['returncode']} ({s['elapsed_s']:.1f}s)")
    lines += [
        "",
        "## Semantic attestation",
        "",
        f"- Positive backends zero-mismatch: `{report['attestation_summary']}`",
        "",
        "## Monitor",
        "",
        f"```json\n{json.dumps(report.get('monitor_summary'), indent=2)}\n```",
        "",
        "## Activation",
        "",
        f"```json\n{json.dumps(report.get('activation_summary'), indent=2)}\n```",
        "",
        "## Performance speedups (GPU-style vs reference)",
        "",
        f"```json\n{json.dumps(report.get('performance_speedups'), indent=2)}\n```",
        "",
        "## Artifacts",
        "",
    ]
    for k, v in report["artifacts"].items():
        lines.append(f"- {k}: `{v}`")
    out_md.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {out_json}\nwrote {out_md}")
    # Non-zero if any step failed
    return 0 if all(s["returncode"] == 0 for s in steps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
