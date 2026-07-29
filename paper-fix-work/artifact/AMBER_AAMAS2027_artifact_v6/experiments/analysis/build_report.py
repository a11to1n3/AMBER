#!/usr/bin/env python3
"""Build consolidated REPORT_*.md/json from raw artifacts."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"


def load(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def main(tag: str = "host_a") -> int:
    att = load(RAW / "semantic" / f"attestation_{tag}.json")
    cov = load(RAW / "monitor" / f"coverage_{tag}.json")
    ovh = load(RAW / "monitor" / f"overhead_{tag}.json")
    act = load(RAW / "semantic" / f"activation_{tag}.json")
    perf = load(RAW / "performance" / f"performance_{tag}.json")

    claims = {
        "C1_semantic_parity": (att or {}).get("acceptance", {}).get("C1_semantic_parity"),
        "C1_negative_controls": (att or {}).get("acceptance", {}).get("negative_controls"),
        "C2_monitor_boundary": (cov or {}).get("acceptance", {}).get("C2_monitor_boundary_reported"),
        "C2_hazards_detected": (cov or {}).get("acceptance", {}).get("all_public_hazards_detected"),
        "C2_no_false_positives": (cov or {}).get("acceptance", {}).get("no_false_positives_on_safe"),
        "C3_sir_activation": (act or {}).get("acceptance", {}).get("C3_activation_effect_sir"),
        "C3_schelling_activation": (act or {}).get("acceptance", {}).get("C3_activation_effect_schelling"),
        "C4_performance_campaign": bool(perf and perf.get("rows")),
    }

    native = []
    if perf:
        for r in perf.get("rows") or []:
            if r.get("status") == "success" and r.get("track") == "native_idiom":
                native.append(
                    {
                        "framework": r.get("framework"),
                        "workload": r.get("workload"),
                        "population": r.get("population"),
                        "warm_median_s": r.get("warm_median_s"),
                    }
                )

    ov = (ovh or {}).get("overhead") or []
    sir = (act or {}).get("sir") or {}
    host = platform.node()

    report = {
        "tag": tag,
        "host": host,
        "platform": platform.platform(),
        "gpu": "NVIDIA GeForce RTX 3090",
        "claims": claims,
        "attestation": {
            "positive_ok": (att or {}).get("positive_backends_zero_mismatch"),
            "negatives": (att or {}).get("negative_controls_detected"),
            "elapsed_s": (att or {}).get("elapsed_s"),
            "summaries": (att or {}).get("summaries"),
        },
        "monitor_coverage": (cov or {}).get("summary"),
        "monitor_overhead": ov,
        "activation_acceptance": (act or {}).get("acceptance"),
        "activation_sir": {
            "primary_tau": sir.get("primary_tau"),
            "primary_final_I_diff": sir.get("primary_final_I_diff"),
            "n": sir.get("n"),
            "steps": sir.get("steps"),
            "seeds": sir.get("seeds"),
            "rows": sir.get("rows"),
        },
        "activation_schelling": (act or {}).get("schelling"),
        "performance_speedups": (perf or {}).get("speedups_gpu_vs_reference"),
        "amber_vs_flame": (perf or {}).get("amber_vs_flame"),
        "performance_native": native,
        "performance_rows_n": len((perf or {}).get("rows") or []),
    }

    out_json = RAW / f"REPORT_{tag}.json"
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        f"# AAMAS Experimental Package Report — {tag}",
        "",
        f"- **Host:** {host} (SSH: host_a / z590)",
        f"- **GPU:** {report['gpu']}",
        f"- **Platform:** {report['platform']}",
        "- **Plan:** ChatGPT AAMAS Submission Review — Experimental improvement plan",
        "- **Tree:** <artifact-root>_aamas_exp/experiments/",
        "",
        "## Evidential chain",
        "",
        "declared activation semantics → reference transition → cross-path semantic",
        "validation → runtime-report characterization → validated fast implementation →",
        "fair performance measurement",
        "",
        "## Claims (minimum package)",
        "",
        "| Claim | Result |",
        "|---|---|",
    ]
    for k, v in claims.items():
        lines.append(f"| `{k}` | **{v}** |")

    lines += [
        "",
        "## C1 — Semantic attestation",
        "",
        f"- Positive backends zero-mismatch: **{report['attestation']['positive_ok']}**",
        f"- Negative controls: **{report['attestation']['negatives']}**",
        f"- Elapsed: {report['attestation']['elapsed_s']} s",
        "",
        "Backends: reference, vectorized_numpy, private_gpu_style (CuPy).",
        "Workloads: wealth, random walk, SIR ring, Schelling.",
        "Shared counter RNG keys draws by (seed, step, event, agent, partner).",
        "",
        "## C2 — Monitor coverage and overhead",
        "",
        "### Coverage",
        "",
        "```json",
        json.dumps(report["monitor_coverage"], indent=2),
        "```",
        "",
        "### Overhead surface (median; 20 steps; 5 retained runs)",
        "",
        "| N | c | q | off (s) | check (s) | overhead/step (ms) | ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for o in ov:
        lines.append(
            f"| {o['n']} | {o['c']} | {o['q']} | {o['median_off_s']:.4f} | "
            f"{o['median_check_s']:.4f} | {o['per_step_ms']:.3f} | {o['ratio']:.2f} |"
        )

    sch = report.get("activation_schelling") or {}
    lines += [
        "",
        "## C3 — Activation effects",
        "",
        f"- SIR N={sir.get('n')}, steps={sir.get('steps')}, seeds={sir.get('seeds')}",
        f"- Primary tau={sir.get('primary_tau')}: "
        f"Δ final I (seq−snap) = {json.dumps(sir.get('primary_final_I_diff'))}",
        f"- SIR excludes zero: **{claims['C3_sir_activation']}**",
        f"- Schelling contrast: {sch.get('contrast', 'n/a')}",
        f"- Schelling primary ({sch.get('primary_outcome', 'n/a')}): "
        f"{json.dumps(sch.get('primary'))} "
        f"(excludes zero: **{claims['C3_schelling_activation']}**)",
        "",
        "### SIR τ sweep",
        "",
        "| tau | mean Δ final I | CI lo | CI hi |",
        "|---:|---:|---:|---:|",
    ]
    for row in sir.get("rows") or []:
        d = row["final_I_diff"]
        lines.append(
            f"| {row['tau']:.3f} | {d['mean']:.4f} | {d['lo']:.4f} | {d['hi']:.4f} |"
        )

    lines += [
        "",
        "### Schelling grids",
        "",
        "| side | cell disagreement | CI lo | CI hi | Δ same-neighbor |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in sch.get("rows") or []:
        d = row.get("cell_disagreement") or {}
        sn = row.get("same_neighbor_diff_seq_minus_sync") or {}
        lines.append(
            f"| {row.get('side')} | {d.get('mean', float('nan')):.4f} | "
            f"{d.get('lo', float('nan')):.4f} | {d.get('hi', float('nan')):.4f} | "
            f"{sn.get('mean', float('nan')):.4f} |"
        )

    lines += [
        "",
        "## C4 — Performance",
        "",
        f"- Timed rows: **{report['performance_rows_n']}** (no trimming; cold/warm/steady scopes)",
        "",
        "### Semantics-matched GPU-style vs reference",
        "",
        "| Workload | N | reference (s) | GPU-style (s) | speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in report.get("performance_speedups") or []:
        lines.append(
            f"| {s['workload']} | {s['population']} | {s['reference_s']:.4f} | "
            f"{s['private_gpu_style_s']:.4f} | {s['speedup']:.3f}× |"
        )

    lines += [
        "",
        "### Native AMBER API track",
        "",
        "| Framework | workload | N | warm median (s) |",
        "|---|---|---:|---:|",
    ]
    for r in native:
        lines.append(
            f"| {r['framework']} | {r.get('workload', 'wealth')} | "
            f"{r['population']} | {r['warm_median_s']} |"
        )

    avf = report.get("amber_vs_flame") or (perf or {}).get("amber_vs_flame") or []
    lines += [
        "",
        "### AMBER GPU vs FLAME GPU 2 (native-idiom, warm median)",
        "",
        "| Workload | N | AMBER GPU (s) | FLAME (s) | AMBER/FLAME |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in avf:
        lines.append(
            f"| {s['workload']} | {s['population']} | {s['AMBER_gpu_s']:.4f} | "
            f"{s['FLAME_GPU_2_s']:.4f} | {s.get('amber_over_flame') or float('nan'):.3f}× |"
        )

    lines += [
        "",
        "Note: semantics-matched GPU-style still uses host counter-RNG for exact",
        "attestation; native AMBER/FLAME tracks use each framework's idiomatic RNG.",
        "",
        "## Artifacts on host_a",
        "",
        "```",
        "<artifact-root>_aamas_exp/experiments/raw/",
        f"  REPORT_{tag}.md",
        f"  REPORT_{tag}.json",
        f"  semantic/attestation_{tag}.json",
        f"  semantic/activation_{tag}.json",
        f"  monitor/coverage_{tag}.json",
        f"  monitor/overhead_{tag}.json",
        f"  performance/performance_{tag}.json",
        "```",
        "",
        "## Not run (stronger package)",
        "",
        "- ABMax ports",
        "- second-GPU hardware replication",
        "- library-integrated evidence-bound approve_fast_path binding",
        "- production-scale HazardBench generator",
        "- full profiler kernel ablations (C5)",
        "",
    ]

    out_md = RAW / f"REPORT_{tag}.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(claims, indent=2))
    print("wrote", out_md)
    print("wrote", out_json)
    return 0


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "host_a"
    raise SystemExit(main(tag))
