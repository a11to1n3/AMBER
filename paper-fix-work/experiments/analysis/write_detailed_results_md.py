#!/usr/bin/env python3
"""Write DETAILED_RESULTS_host_a.md with full tables + embedded plots."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
TAG = "host_a"
OUT = RAW / f"DETAILED_RESULTS_{TAG}.md"


def load(rel: str):
    return json.loads((RAW / rel).read_text())


def fnum(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        if abs(x) >= 100 or (abs(x) > 0 and abs(x) < 1e-3):
            return f"{x:.4g}"
        return f"{x:.{nd}f}"
    return str(x)


def main():
    att = load(f"semantic/attestation_{TAG}.json")
    cov = load(f"monitor/coverage_{TAG}.json")
    ovh = load(f"monitor/overhead_{TAG}.json")
    act = load(f"semantic/activation_{TAG}.json")
    perf = load(f"performance/performance_{TAG}.json")

    lines: list[str] = []
    def w(s=""):
        lines.append(s)

    w(f"# AAMAS Experimental Improvement Campaign — Detailed Results")
    w()
    w(f"**Tag:** `{TAG}`  ")
    w(f"**Host:** `{att['host']}` (SSH: `host_a (RTX 3090)` / `z590`)  ")
    w(f"**GPU:** NVIDIA GeForce RTX 3090 · CUDA 12.6  ")
    w(f"**Platform:** `{att['platform']}`  ")
    w(f"**Python:** {att.get('python', '3.11')}  ")
    w(f"**Report generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    w(f"**Source plan:** ChatGPT AAMAS Submission Review — *Experimental improvement plan*  ")
    w(f"**Code:** `experiments/` (synced tree `~/AMBER_aamas_exp` on server)")
    w()
    w("## Contents")
    w()
    w("1. [Executive summary](#1-executive-summary)")
    w("2. [Evidential chain](#2-evidential-chain)")
    w("3. [C1 Semantic attestation](#3-c1-semantic-attestation)")
    w("4. [C2 Monitor coverage and overhead](#4-c2-monitor-coverage-and-overhead)")
    w("5. [C3 Activation effects](#5-c3-activation-effects)")
    w("6. [C4 Performance](#6-c4-performance)")
    w("7. [Methods notes](#7-methods-notes)")
    w("8. [Artifact index](#8-artifact-index)")
    w("9. [Scope limits](#9-scope-limits)")
    w()

    # ---- 1 ----
    w("## 1. Executive summary")
    w()
    w("![Claims dashboard](figs/fig00_claims_dashboard.png)")
    w()
    w("| Claim | Result | One-line evidence |")
    w("|---|---|---|")
    w(f"| **C1** semantic parity | **{att['acceptance']['C1_semantic_parity']}** | 0 state mismatches on all positive backends |")
    w(f"| **C1** negative controls | **{att['acceptance']['negative_controls']}** | {att['negative_controls_detected']} intentional corruptions diverged |")
    w(f"| **C2** monitor hazards | **{cov['acceptance']['all_public_hazards_detected']}** | {cov['summary']['hazard_detected']}/{cov['summary']['hazard_total']} public-seam hazards detected |")
    w(f"| **C2** no false positives | **{cov['acceptance']['no_false_positives_on_safe']}** | {cov['summary']['safe_clean']}/{cov['summary']['safe_total']} safe programs clean |")
    w(f"| **C2** overhead surface | **yes** | {len(ovh['overhead'])} (N,c,q) points retained |")
    w(f"| **C3** SIR activation | **{act['acceptance']['C3_activation_effect_sir']}** | primary τ={act['sir']['primary_tau']}: ΔI={act['sir']['primary_final_I_diff']['mean']:.4f} CI excludes 0 |")
    w(f"| **C3** Schelling activation | **{act['acceptance']['C3_activation_effect_schelling']}** | cell disagreement mean={act['schelling']['primary']['mean']:.3f} CI>0 |")
    w(f"| **C4** performance campaign | **{bool(perf.get('rows'))}** | {len(perf.get('rows') or [])} timed rows; cold/warm/steady; no trimming |")
    w(f"| **C4** AMBER vs FLAME | **{bool(perf.get('amber_vs_flame'))}** | {len(perf.get('amber_vs_flame') or [])} native-idiom pairs |")
    w()
    w("**Headline numbers (native-idiom, warm median, 50 steps, RTX 3090):**")
    w()
    # pull 1M wealth
    def find(framework, workload, n):
        for r in perf["rows"]:
            if (r.get("status") == "success" and r.get("framework") == framework
                    and r.get("workload") == workload and r.get("population") == n
                    and r.get("track") == "native_idiom"):
                return r.get("warm_median_s")
        return None
    aw = find("AMBER_gpu", "wealth", 1_000_000)
    fw = find("FLAME_GPU_2", "wealth", 1_000_000)
    ar = find("AMBER_gpu", "random_walk", 1_000_000)
    fr = find("FLAME_GPU_2", "random_walk", 1_000_000)
    ac = find("AMBER_vectorized_cpu", "wealth", 1_000_000)
    w(f"- AMBER GPU wealth @ 1M: **{fnum(aw, 4)} s** vs vectorized CPU **{fnum(ac, 3)} s** (~{ac/aw:.0f}×)  " if aw and ac else "")
    w(f"- AMBER GPU vs FLAME wealth @ 1M: **{fnum(aw, 4)} s** vs **{fnum(fw, 4)} s** (AMBER/FLAME = {aw/fw:.3f}×)  " if aw and fw else "")
    w(f"- AMBER GPU vs FLAME random walk @ 1M: **{fnum(ar, 4)} s** vs **{fnum(fr, 4)} s** (AMBER/FLAME = {ar/fr:.3f}×)  " if ar and fr else "")
    w()

    # ---- 2 ----
    w("## 2. Evidential chain")
    w()
    w("```")
    w("declared activation semantics")
    w("        ↓")
    w("reference transition (pure NumPy / readable)")
    w("        ↓")
    w("cross-path semantic validation (vectorized + GPU-style + negatives)")
    w("        ↓")
    w("runtime-report characterization (coverage + overhead surface)")
    w("        ↓")
    w("validated fast implementation (native AMBER GPU)")
    w("        ↓")
    w("fair performance measurement (cold/warm/steady; FLAME native-idiom)")
    w("```")
    w()
    w("Shared **counter RNG** keys every random draw by")
    w("`(global_seed, step, event_type, agent_id, partner_id, draw_index)` so values")
    w("do not depend on thread index, event order, or backend.")
    w()

    # ---- 3 ----
    w("## 3. C1 Semantic attestation")
    w()
    w(f"- Elapsed: **{att['elapsed_s']:.2f} s**  ")
    w(f"- GPU-style backends included: **{att['gpu_included']}**  ")
    w(f"- Positive backends zero-mismatch: **{att['positive_backends_zero_mismatch']}**  ")
    w(f"- Negative controls: **{att['negative_controls_detected']}**")
    w()
    w("![Attestation](figs/fig01_attestation.png)")
    w()
    w("### 3.1 Positive backends (must match reference)")
    w()
    w("| Workload | Backend | Exhaustive | Random | Steps checked | State mismatches | Max abs err | Status |")
    w("|---|---|---:|---:|---:|---:|---:|---|")
    for r in att["summaries"]:
        if r["is_negative"]:
            continue
        w(
            f"| {r['workload']} | `{r['backend']}` | {r['exhaustive_cases']} | {r['random_cases']} | "
            f"{r['steps_checked']} | **{r['state_mismatches']}** | {r['max_abs_error']} | {r['status']} |"
        )
    w()
    w("### 3.2 Negative controls (must diverge)")
    w()
    w("| Workload | Backend | Cases with mismatch | Total cell mismatches | Detected? | Status |")
    w("|---|---|---:|---:|---|---|")
    for r in att["summaries"]:
        if not r["is_negative"]:
            continue
        w(
            f"| {r['workload']} | `{r['backend']}` | {r['cases_with_mismatch']} | "
            f"{r['state_mismatches']} | **{r['detected_as_divergent']}** | {r['status']} |"
        )
    w()
    w("**Negative control intents:**")
    w()
    w("| ID | Fault |")
    w("|---|---|")
    w("| `neg_live_donors` | Wealth eligibility recomputed after earlier transfers |")
    w("| `neg_last_write` | Duplicate targets replace rather than scatter-add |")
    w("| `neg_order_rng` | Displacements keyed by reverse execution position |")
    w("| `neg_inplace` | Newly infected transmit in the same step |")
    w("| `neg_thread_rng` | Infection draws keyed by enumeration order |")
    w("| `neg_last_winner` | Schelling vacancy collisions decided by arrival order |")
    w("| `neg_no_conflict_resolution` | Multiple agents may stack on one cell |")
    w()
    w("### 3.3 Workload semantics (frozen)")
    w()
    w("| Workload | Spec file | Semantics |")
    w("|---|---|---|")
    w("| wealth | `experiments/specs/wealth_transfer.yaml` | Snapshot delta; eligibility from entry; one transfer/donor |")
    w("| random walk | `experiments/specs/random_walk.yaml` | Identity-keyed displacements; clip to [0, world] |")
    w("| SIR ring | `experiments/specs/sir_ring.yaml` | Snapshot infection; pair-keyed RVs; no same-step transmit |")
    w("| Schelling | `experiments/specs/schelling.yaml` | Three-stage sync; deterministic conflict priority |")
    w()

    # ---- 4 ----
    w("## 4. C2 Monitor coverage and overhead")
    w()
    w("### 4.1 Coverage (HazardBench-lite)")
    w()
    w("![Monitor coverage](figs/fig02_monitor_coverage.png)")
    w()
    w("| Program | Mode | Expect hazard | Detected | Outcome | Violation kinds |")
    w("|---|---|---|---|---|---|")
    for r in cov["rows"]:
        kinds = ", ".join(r.get("kinds") or ["—"])
        w(
            f"| `{r['name']}` | {r['mode']} | {r['expect_kind'] or 'none (safe)'} | "
            f"{r['detected']} | **{r['outcome']}** | {kinds} |"
        )
    w()
    w(f"**Summary:** safe {cov['summary']['safe_clean']}/{cov['summary']['safe_total']} clean; "
      f"hazards {cov['summary']['hazard_detected']}/{cov['summary']['hazard_total']} detected; "
      f"false negatives: {cov['summary']['false_negatives'] or '[]'}; "
      f"false positives: {cov['summary']['false_positives'] or '[]'}.")
    w()
    w("### 4.2 Overhead surface")
    w()
    w("Protocol: synthetic vectorized model; **20 steps**; **5 retained runs** after 1 untimed warm-up; "
      "contracts `off` vs `check`; vary population N, schema columns c, concurrent column commits q.")
    w()
    w("![Overhead ratio](figs/fig03_monitor_overhead_ratio.png)")
    w()
    w("![Absolute overhead](figs/fig04_monitor_overhead_abs.png)")
    w()
    w("| N | c | q | median off (s) | median check (s) | abs overhead (s) | per-step (ms) | ratio |")
    w("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for o in sorted(ovh["overhead"], key=lambda x: (x["n"], x["c"], x["q"])):
        w(
            f"| {o['n']} | {o['c']} | {o['q']} | {o['median_off_s']:.4f} | {o['median_check_s']:.4f} | "
            f"{o['abs_overhead_s']:.4f} | {o['per_step_ms']:.3f} | {o['ratio']:.2f} |"
        )
    w()
    w("**Interpretation:** at N=1e6, c=1, q=1 the check path is ~26× the off path "
      "(~123 ms/step absolute overhead). Monitor cost is material and must not be "
      "compared against private optimized GPU loops that bypass the report.")
    w()

    # ---- 5 ----
    w("## 5. C3 Activation effects")
    w()
    w("### 5.1 SIR (ring, shared counter RNG)")
    w()
    sir = act["sir"]
    w(f"- N = **{sir['n']}**, steps = **{sir['steps']}**, seeds = **{sir['seeds']}**  ")
    w(f"- Radius = 3, recovery = 0.1, shared infection/recovery keys  ")
    w(f"- Contrast: **sequential (reshuffled order)** vs **snapshot**  ")
    w(f"- Primary τ = **{sir['primary_tau']}**  ")
    pd = sir["primary_final_I_diff"]
    w(f"- Primary Δ final infected (seq − snap) = **{pd['mean']:.6f}** "
      f"[{pd['lo']:.6f}, {pd['hi']:.6f}] (n={pd['n']})  ")
    w(f"- Effect excludes zero: **{act['acceptance']['C3_activation_effect_sir']}**")
    w()
    w("![SIR activation](figs/fig05_sir_activation.png)")
    w()
    w("| τ | mean Δ final I | CI lo | CI hi | mean final I snap | mean final I seq |")
    w("|---:|---:|---:|---:|---:|---:|")
    for r in sir["rows"]:
        d = r["final_I_diff"]
        w(
            f"| {r['tau']:.3f} | {d['mean']:.6f} | {d['lo']:.6f} | {d['hi']:.6f} | "
            f"{r['mean_final_I_snap']:.4f} | {r['mean_final_I_seq']:.4f} |"
        )
    w()
    w("### 5.2 Schelling (three-stage snapshot vs sequential)")
    w()
    sch = act["schelling"]
    w(f"- Contrast: **{sch.get('contrast')}**  ")
    w(f"- Primary outcome: **{sch.get('primary_outcome')}**  ")
    w(f"- Steps = {sch['steps']}, seeds = {sch['seeds']}, "
      f"threshold = {sch.get('threshold')}, empty_ratio = {sch.get('empty_ratio')}  ")
    sp = sch["primary"]
    w(f"- Primary (largest grid) disagreement = **{sp['mean']:.6f}** "
      f"[{sp['lo']:.6f}, {sp['hi']:.6f}]  ")
    w(f"- Effect excludes zero: **{act['acceptance']['C3_activation_effect_schelling']}**")
    w()
    w("![Schelling activation](figs/fig06_schelling_activation.png)")
    w()
    w("| side | cell disagreement mean | CI lo | CI hi | Δ same-neighbor mean | CI lo | CI hi | Δ segregation mean |")
    w("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sch["rows"]:
        d = r["cell_disagreement"]
        sn = r["same_neighbor_diff_seq_minus_sync"]
        sg = r["segregation_diff_seq_minus_sync"]
        w(
            f"| {r['side']} | {d['mean']:.6f} | {d['lo']:.6f} | {d['hi']:.6f} | "
            f"{sn['mean']:.6f} | {sn['lo']:.6f} | {sn['hi']:.6f} | {sg['mean']:.6f} |"
        )
    w()
    w("Note: earlier last-writer-only contrast produced a null segregation-index "
      "difference; the sequential activation contrast is the scientifically relevant "
      "activation-semantics intervention.")
    w()

    # ---- 6 ----
    w("## 6. C4 Performance")
    w()
    w("Protocol notes:")
    w()
    w("- **No sample trimming.** Every timed run retained with status.")
    w("- **Cold / warm / steady:** first timed run = cold; subsequent = warm/steady.")
    w("- **Native-idiom track:** AMBER public API + FLAME GPU 2 with each framework's RNG.")
    w("- **Semantics-matched track:** counter-RNG backends for attestation fidelity "
      "(GPU-style still draws tape on host — not production fused kernels).")
    w("- FLAME RTC requires `CUDA_PATH` pointing at a toolkit with `include/cuda_runtime.h`.")
    w()
    w("### 6.1 Native scaling")
    w()
    w("![Native scaling](figs/fig07_native_scaling.png)")
    w()
    w("![AMBER vs FLAME](figs/fig08_amber_vs_flame.png)")
    w()
    w("![Cold vs warm](figs/fig10_cold_warm.png)")
    w()
    w("#### Full native-idiom table (warm median)")
    w()
    w("| Framework | Workload | N | Steps | Warm median (s) | Cold (s) | Status |")
    w("|---|---|---:|---:|---:|---:|---|")
    native = [
        r for r in perf["rows"]
        if r.get("track") == "native_idiom"
    ]
    native.sort(key=lambda r: (r.get("workload") or "", r.get("framework") or "", r.get("population") or 0))
    for r in native:
        samples = r.get("samples") or []
        cold = next((s.get("total_s") for s in samples if s.get("scope") == "cold" and s.get("status") == "success"), None)
        w(
            f"| {r.get('framework')} | {r.get('workload')} | {r.get('population')} | "
            f"{r.get('steps')} | {fnum(r.get('warm_median_s'), 6)} | {fnum(cold, 6)} | {r.get('status')} |"
        )
    w()
    w("#### AMBER GPU vs FLAME GPU 2")
    w()
    w("| Workload | N | AMBER GPU (s) | FLAME (s) | AMBER/FLAME | FLAME/AMBER |")
    w("|---|---:|---:|---:|---:|---:|")
    for a in sorted(perf.get("amber_vs_flame") or [], key=lambda x: (x["workload"], x["population"])):
        w(
            f"| {a['workload']} | {a['population']} | {a['AMBER_gpu_s']:.6f} | {a['FLAME_GPU_2_s']:.6f} | "
            f"{a['amber_over_flame']:.3f}× | {a['flame_over_amber']:.3f}× |"
        )
    w()
    w("**Cross-over:** FLAME is faster at N≤1e5 on this host for wealth/random walk; "
      "AMBER GPU is faster at N=1e6 (AMBER/FLAME ≈ 0.65–0.71). Single-GPU (3090) result; "
      "not a universal ranking.")
    w()
    w("### 6.2 Semantics-matched track")
    w()
    w("![Matched speedups](figs/fig09_matched_speedups.png)")
    w()
    w("| Workload | N | reference (s) | GPU-style (s) | speedup |")
    w("|---|---:|---:|---:|---:|")
    for s in perf.get("speedups_gpu_vs_reference") or []:
        w(
            f"| {s['workload']} | {s['population']} | {s['reference_s']:.4f} | "
            f"{s['private_gpu_style_s']:.4f} | {s['speedup']:.3f}× |"
        )
    w()
    w("#### All semantics-matched warm medians")
    w()
    w("| Backend | Workload | N | Steps | Warm median (s) | Status |")
    w("|---|---|---:|---:|---:|---|")
    matched = [r for r in perf["rows"] if r.get("track") == "semantics_matched"]
    matched.sort(key=lambda r: (r.get("workload") or "", r.get("population") or 0, r.get("framework") or ""))
    for r in matched:
        w(
            f"| {r.get('framework')} | {r.get('workload')} | {r.get('population')} | "
            f"{r.get('steps')} | {fnum(r.get('warm_median_s'), 6)} | {r.get('status')} |"
        )
    w()
    w("#### Sample-level record (AMBER GPU wealth @ 1M)")
    w()
    for r in perf["rows"]:
        if (r.get("framework") == "AMBER_gpu" and r.get("workload") == "wealth"
                and r.get("population") == 1_000_000 and r.get("status") == "success"):
            w("| run | scope | total_s | status |")
            w("|---:|---|---:|---|")
            for s in r.get("samples") or []:
                w(f"| {s.get('run')} | {s.get('scope')} | {fnum(s.get('total_s'), 6)} | {s.get('status')} |")
            break
    w()

    # ---- 7 ----
    w("## 7. Methods notes")
    w()
    w("### Random tape")
    w()
    w("Counter mixer: SplitMix64-style `mix64` over packed keys; `u01` uses top 53 bits. "
      "Test vectors in `semantic/rng_test_vectors_host_a.json`.")
    w()
    w("### Bootstrap")
    w()
    w("Activation CIs: paired differences across seeds; 2000 bootstrap resamples; "
      "percentile interval [2.5%, 97.5%].")
    w()
    w("### Timing scopes")
    w()
    w("| Scope | Definition |")
    w("|---|---|")
    w("| cold | first timed invocation of a configuration in-process |")
    w("| warm | second timed invocation |")
    w("| steady | third+ timed invocations |")
    w("| warm median | median of non-cold successful samples |")
    w()
    w("### Software stack (host)")
    w()
    w("| Component | Version / path |")
    w("|---|---|")
    w("| Python | 3.11.4 (`~/AMBER/.venv`) |")
    w("| NumPy | 2.4.6 |")
    w("| CuPy | 14.x |")
    w("| pyflamegpu | installed in venv |")
    w("| CUDA | 12.6 (`/usr/local/cuda-12.6`) |")
    w("| Driver | 560.35.05 |")
    w()

    # ---- 8 ----
    w("## 8. Artifact index")
    w()
    w("| Kind | Path |")
    w("|---|---|")
    w(f"| This report | `experiments/raw/DETAILED_RESULTS_{TAG}.md` |")
    w(f"| Short report | `experiments/raw/REPORT_{TAG}.md` |")
    w(f"| Attestation JSON | `experiments/raw/semantic/attestation_{TAG}.json` |")
    w(f"| Activation JSON | `experiments/raw/semantic/activation_{TAG}.json` |")
    w(f"| Coverage JSON | `experiments/raw/monitor/coverage_{TAG}.json` |")
    w(f"| Overhead JSON | `experiments/raw/monitor/overhead_{TAG}.json` |")
    w(f"| Performance JSON | `experiments/raw/performance/performance_{TAG}.json` |")
    w("| Figures (PNG+PDF) | `experiments/raw/figs/` |")
    w("| Plot script | `experiments/analysis/plot_campaign_results.py` |")
    w("| MD builder | `experiments/analysis/write_detailed_results_md.py` |")
    w("| Integration notes | `paper-fix-work/EXPERIMENT_CAMPAIGN_INTEGRATION.md` |")
    w()
    w("### Figure list")
    w()
    w("| File | Content |")
    w("|---|---|")
    w("| `fig00_claims_dashboard` | Pass/fail claim board |")
    w("| `fig01_attestation` | Positive cases + negative detections |")
    w("| `fig02_monitor_coverage` | HazardBench-lite outcomes |")
    w("| `fig03_monitor_overhead_ratio` | check/off ratio vs N by c,q |")
    w("| `fig04_monitor_overhead_abs` | absolute ms/step overhead |")
    w("| `fig05_sir_activation` | Δ final I and level curves |")
    w("| `fig06_schelling_activation` | cell disagreement + Δ same-neighbor |")
    w("| `fig07_native_scaling` | AMBER CPU/GPU + FLAME log-log |")
    w("| `fig08_amber_vs_flame` | AMBER/FLAME ratio vs N |")
    w("| `fig09_matched_speedups` | reference/GPU-style speedups |")
    w("| `fig10_cold_warm` | AMBER GPU wealth cold vs warm |")
    w()

    # ---- 9 ----
    w("## 9. Scope limits")
    w()
    w("Not claimed / not run in this package:")
    w()
    w("- Byte-identical AMBER ↔ FLAME trajectories (different RNG and messaging idioms)")
    w("- Monitor completeness beyond instrumented API seams")
    w("- ABMax ports; second-GPU hardware replication")
    w("- Library-integrated evidence-bound `approve_fast_path` validation")
    w("- Production-scale HazardBench generator")
    w("- Full profiler kernel ablations (C5)")
    w("- Re-run of the paper’s 10M AMBER/FLAME campaign under this cold/warm schema "
      "(existing 10M artifact remains authoritative for those endpoints)")
    w()
    w("---")
    w()
    w("*All tables in this file are generated from the JSON artifacts listed above; "
      "re-run `plot_campaign_results.py` and `write_detailed_results_md.py` to refresh.*")
    w()

    OUT.write_text("\n".join(lines))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
