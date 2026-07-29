#!/usr/bin/env python3
"""Publication-style plots from the AAMAS experimental improvement campaign."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
FIGS = RAW / "figs"
TAG = "host_a"

# Editorial palette (aligned with paper Atlantic-blue / copper system)
BLUE = "#1f4e79"
BLUE_LIGHT = "#5b8fbf"
COPPER = "#c4713b"
COPPER_LIGHT = "#e0a070"
GREEN = "#2f6b4f"
RED = "#a33b3b"
GRAY = "#5c6670"
GRAY_LIGHT = "#c5ccd3"
BG = "#fbfaf8"


def style():
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": "#1a1a1a",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRAY_LIGHT,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "lines.linewidth": 1.8,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": BG,
    })


def load(rel: str):
    return json.loads((RAW / rel).read_text())


def save(fig, name: str):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIGS / f"{name}.{ext}"
        fig.savefig(path)
    plt.close(fig)
    print("wrote", FIGS / f"{name}.png")


def plot_attestation():
    att = load(f"semantic/attestation_{TAG}.json")
    rows = att["summaries"]
    positives = [r for r in rows if not r["is_negative"]]
    negatives = [r for r in rows if r["is_negative"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})

    # Left: cases per positive backend
    labels = [f"{r['workload']}\n{r['backend']}" for r in positives]
    exh = [r["exhaustive_cases"] for r in positives]
    rnd = [r["random_cases"] for r in positives]
    y = np.arange(len(labels))
    axes[0].barh(y, exh, color=BLUE, label="exhaustive", height=0.4)
    axes[0].barh(y, rnd, left=exh, color=BLUE_LIGHT, label="random", height=0.4)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Cases")
    axes[0].set_title("C1 positive backends (all 0 mismatches)")
    axes[0].legend(loc="lower right", frameon=False)
    for i, r in enumerate(positives):
        axes[0].text(
            exh[i] + rnd[i] + 2, i, f"mm={r['state_mismatches']}",
            va="center", fontsize=7, color=GREEN,
        )

    # Right: negative controls
    nlabels = [f"{r['workload']}\n{r['backend']}" for r in negatives]
    ny = np.arange(len(nlabels))
    colors = [GREEN if r["detected_as_divergent"] else RED for r in negatives]
    axes[1].barh(ny, [r["cases_with_mismatch"] for r in negatives], color=colors, height=0.55)
    axes[1].set_yticks(ny)
    axes[1].set_yticklabels(nlabels, fontsize=7.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Cases with ≥1 state mismatch")
    axes[1].set_title(f"Negative controls detected {att['negative_controls_detected']}")
    for i, r in enumerate(negatives):
        axes[1].text(
            r["cases_with_mismatch"] + 1, i,
            "detected" if r["detected_as_divergent"] else "MISSED",
            va="center", fontsize=7, color=colors[i],
        )

    fig.suptitle("Semantic attestation — host_a (RTX 3090)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig01_attestation")


def plot_monitor_coverage():
    cov = load(f"monitor/coverage_{TAG}.json")
    rows = cov["rows"]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    y = np.arange(len(rows))
    colors = []
    for r in rows:
        if r["outcome"] == "true_positive":
            colors.append(GREEN)
        elif r["outcome"] == "true_negative":
            colors.append(BLUE)
        elif r["outcome"] == "false_negative":
            colors.append(RED)
        else:
            colors.append(COPPER)
    ax.barh(y, [1] * len(rows), color=colors, height=0.65)
    ax.set_yticks(y)
    labels = []
    for r in rows:
        kind = r["expect_kind"] or "safe"
        labels.append(f"{r['name']}  [{kind}] → {r['outcome']}")
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xticks([])
    ax.set_title("C2 ContractReport coverage (HazardBench-lite)")
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color=GREEN, label="true positive"),
            Patch(color=BLUE, label="true negative"),
            Patch(color=RED, label="false negative"),
            Patch(color=COPPER, label="false positive"),
        ],
        loc="lower right", frameon=False, ncol=2,
    )
    for i, r in enumerate(rows):
        kinds = ", ".join(r.get("kinds") or ["—"])
        ax.text(1.02, i, kinds[:40], va="center", fontsize=7, color=GRAY)
    fig.tight_layout()
    save(fig, "fig02_monitor_coverage")


def plot_monitor_overhead():
    ovh = load(f"monitor/overhead_{TAG}.json")
    ov = ovh["overhead"]
    # Facet by c, lines by q, x=N
    qs = sorted({o["q"] for o in ov})
    cs = sorted({o["c"] for o in ov})
    fig, axes = plt.subplots(1, len(cs), figsize=(11.5, 3.8), sharey=True)
    if len(cs) == 1:
        axes = [axes]
    cmap = {q: c for q, c in zip(qs, [BLUE, COPPER, GREEN, RED][: len(qs)])}
    for ax, c in zip(axes, cs):
        for q in qs:
            pts = sorted([o for o in ov if o["c"] == c and o["q"] == q], key=lambda x: x["n"])
            if not pts:
                continue
            ns = [p["n"] for p in pts]
            ratios = [p["ratio"] for p in pts]
            ax.plot(ns, ratios, "o-", color=cmap[q], label=f"q={q}", markersize=5)
        ax.set_xscale("log")
        ax.set_title(f"schema columns c={c}")
        ax.set_xlabel("N agents")
        ax.axhline(1.0, color=GRAY, ls="--", lw=0.9)
    axes[0].set_ylabel("median time ratio  check / off")
    axes[-1].legend(frameon=False, title="writes/step")
    fig.suptitle("Monitor overhead surface (20 steps, 5 retained runs)", fontsize=12, y=1.03)
    fig.tight_layout()
    save(fig, "fig03_monitor_overhead_ratio")

    # Absolute per-step ms at large N
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for q in qs:
        pts = sorted([o for o in ov if o["c"] == max(cs) and o["q"] == q], key=lambda x: x["n"])
        if not pts:
            continue
        ax.plot(
            [p["n"] for p in pts],
            [p["per_step_ms"] for p in pts],
            "o-", color=cmap[q], label=f"q={q}", markersize=5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N agents")
    ax.set_ylabel("overhead per step (ms)  [c=8]")
    ax.set_title("Absolute monitor overhead (check − off) / steps")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig04_monitor_overhead_abs")


def plot_sir_activation():
    act = load(f"semantic/activation_{TAG}.json")
    rows = act["sir"]["rows"]
    taus = [r["tau"] for r in rows]
    means = [r["final_I_diff"]["mean"] for r in rows]
    los = [r["final_I_diff"]["lo"] for r in rows]
    his = [r["final_I_diff"]["hi"] for r in rows]
    yerr = np.array([
        [m - lo for m, lo in zip(means, los)],
        [hi - m for m, hi in zip(means, his)],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.errorbar(taus, means, yerr=yerr, fmt="o-", color=BLUE, ecolor=BLUE_LIGHT,
                capsize=3, markersize=5, label="Δ final I (seq − snap)")
    ax.axhline(0, color=GRAY, ls="--", lw=1)
    ax.set_xlabel("transmission τ")
    ax.set_ylabel("Δ final infected fraction")
    ax.set_title("SIR activation effect (shared counter RNG)")
    ax.legend(frameon=False)

    # mark primary
    pt = act["sir"]["primary_tau"]
    pd = act["sir"]["primary_final_I_diff"]
    ax.axvline(pt, color=COPPER, ls=":", lw=1.2)
    ax.annotate(
        f"primary τ={pt}\n{pd['mean']:.4f} [{pd['lo']:.4f}, {pd['hi']:.4f}]",
        xy=(pt, pd["mean"]),
        xytext=(pt + 0.04, pd["mean"] - 0.002),
        fontsize=8, color=COPPER,
        arrowprops=dict(arrowstyle="->", color=COPPER, lw=0.9),
    )

    ax2 = axes[1]
    snap = [r["mean_final_I_snap"] for r in rows]
    seq = [r["mean_final_I_seq"] for r in rows]
    ax2.plot(taus, snap, "s-", color=BLUE, label="snapshot", markersize=4)
    ax2.plot(taus, seq, "o-", color=COPPER, label="sequential", markersize=4)
    ax2.set_xlabel("transmission τ")
    ax2.set_ylabel("mean final infected fraction")
    ax2.set_title(f"N={act['sir']['n']}, steps={act['sir']['steps']}, seeds={act['sir']['seeds']}")
    ax2.legend(frameon=False)

    fig.suptitle("C3 — Activation semantics (SIR ring)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig05_sir_activation")


def plot_schelling_activation():
    act = load(f"semantic/activation_{TAG}.json")
    rows = act["schelling"]["rows"]
    sides = [r["side"] for r in rows]
    means = [r["cell_disagreement"]["mean"] for r in rows]
    los = [r["cell_disagreement"]["lo"] for r in rows]
    his = [r["cell_disagreement"]["hi"] for r in rows]
    yerr = np.array([
        [m - lo for m, lo in zip(means, los)],
        [hi - m for m, hi in zip(means, his)],
    ])
    d_same = [r["same_neighbor_diff_seq_minus_sync"]["mean"] for r in rows]
    d_same_lo = [r["same_neighbor_diff_seq_minus_sync"]["lo"] for r in rows]
    d_same_hi = [r["same_neighbor_diff_seq_minus_sync"]["hi"] for r in rows]
    yerr2 = np.array([
        [m - lo for m, lo in zip(d_same, d_same_lo)],
        [hi - m for m, hi in zip(d_same, d_same_hi)],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    axes[0].errorbar(sides, means, yerr=yerr, fmt="o-", color=BLUE, ecolor=BLUE_LIGHT,
                     capsize=3, markersize=6)
    axes[0].set_xlabel("grid side")
    axes[0].set_ylabel("cell disagreement fraction")
    axes[0].set_title("Primary: sync vs sequential finals")
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0, color=GRAY, ls="--", lw=0.8)

    axes[1].errorbar(sides, d_same, yerr=yerr2, fmt="s-", color=COPPER, ecolor=COPPER_LIGHT,
                     capsize=3, markersize=6)
    axes[1].axhline(0, color=GRAY, ls="--", lw=0.8)
    axes[1].set_xlabel("grid side")
    axes[1].set_ylabel("Δ mean same-neighbor frac (seq − sync)")
    axes[1].set_title("Secondary metric")

    thr = act["schelling"].get("threshold", 0.6)
    empty = act["schelling"].get("empty_ratio", 0.3)
    fig.suptitle(
        f"C3 — Schelling three-stage snapshot vs sequential "
        f"(threshold={thr}, empty={empty})",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()
    save(fig, "fig06_schelling_activation")


def plot_performance():
    perf = load(f"performance/performance_{TAG}.json")
    rows = [r for r in perf["rows"] if r.get("status") == "success" and r.get("warm_median_s") is not None]

    # Native scaling: AMBER cpu/gpu + FLAME
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=False)
    for ax, workload in zip(axes, ("wealth", "random_walk")):
        for framework, color, marker, label in [
            ("AMBER_vectorized_cpu", GRAY, "s", "AMBER vectorized CPU"),
            ("AMBER_gpu", BLUE, "o", "AMBER GPU"),
            ("FLAME_GPU_2", COPPER, "D", "FLAME GPU 2"),
        ]:
            pts = sorted(
                [r for r in rows if r.get("track") == "native_idiom"
                 and r.get("workload") == workload and r.get("framework") == framework],
                key=lambda x: x["population"],
            )
            if not pts:
                continue
            ax.plot(
                [p["population"] for p in pts],
                [p["warm_median_s"] for p in pts],
                marker=marker, color=color, label=label, markersize=6,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N agents")
        ax.set_ylabel("warm median wall time (s)")
        ax.set_title(f"{workload} · 50 steps")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("C4 — Native-idiom performance (host_a (RTX 3090), RTX 3090)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig07_native_scaling")

    # AMBER vs FLAME ratio
    avf = perf.get("amber_vs_flame") or []
    if avf:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for workload, color, marker in [("wealth", BLUE, "o"), ("random_walk", COPPER, "s")]:
            pts = sorted([a for a in avf if a["workload"] == workload], key=lambda x: x["population"])
            if not pts:
                continue
            ax.plot(
                [p["population"] for p in pts],
                [p["amber_over_flame"] for p in pts],
                marker=marker, color=color, label=workload, markersize=6,
            )
        ax.axhline(1.0, color=GRAY, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("N agents")
        ax.set_ylabel("AMBER GPU / FLAME  (warm median)")
        ax.set_title("< 1 means AMBER faster")
        ax.legend(frameon=False)
        # annotate
        ax.text(0.02, 0.95, "FLAME faster", transform=ax.transAxes, color=GRAY, va="top", fontsize=8)
        ax.text(0.02, 0.05, "AMBER faster", transform=ax.transAxes, color=GRAY, va="bottom", fontsize=8)
        fig.tight_layout()
        save(fig, "fig08_amber_vs_flame")

    # Matched track speedups
    su = perf.get("speedups_gpu_vs_reference") or []
    if su:
        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        labels = [f"{s['workload']}\nN={s['population']}" for s in su]
        vals = [s["speedup"] for s in su]
        colors = [BLUE if v >= 1 else COPPER for v in vals]
        x = np.arange(len(labels))
        ax.bar(x, vals, color=colors, width=0.7)
        ax.axhline(1.0, color=GRAY, ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel("speedup  reference / GPU-style")
        ax.set_title("Semantics-matched track (host counter-RNG; not production kernels)")
        fig.tight_layout()
        save(fig, "fig09_matched_speedups")

    # Cold vs warm for AMBER GPU wealth
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    pts = sorted(
        [r for r in rows if r.get("framework") == "AMBER_gpu" and r.get("workload") == "wealth"],
        key=lambda x: x["population"],
    )
    if pts:
        ns = [p["population"] for p in pts]
        cold = []
        warm = []
        for p in pts:
            samples = p.get("samples") or []
            c = next((s["total_s"] for s in samples if s.get("scope") == "cold" and s.get("status") == "success"), None)
            wvals = [s["total_s"] for s in samples if s.get("scope") != "cold" and s.get("status") == "success" and s.get("total_s") is not None]
            cold.append(c if c is not None else float("nan"))
            warm.append(float(np.median(wvals)) if wvals else float("nan"))
        ax.plot(ns, cold, "o--", color=COPPER, label="cold (1st timed run)", markersize=6)
        ax.plot(ns, warm, "o-", color=BLUE, label="warm median", markersize=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N agents")
        ax.set_ylabel("wall time (s)")
        ax.set_title("AMBER GPU wealth — cold vs warm separation")
        ax.legend(frameon=False)
        fig.tight_layout()
        save(fig, "fig10_cold_warm")


def plot_claims_dashboard():
    """Compact traffic-light summary of all claims."""
    att = load(f"semantic/attestation_{TAG}.json")
    cov = load(f"monitor/coverage_{TAG}.json")
    act = load(f"semantic/activation_{TAG}.json")
    perf = load(f"performance/performance_{TAG}.json")

    claims = [
        ("C1 semantic parity", att["acceptance"]["C1_semantic_parity"]),
        ("C1 negative controls", att["acceptance"]["negative_controls"]),
        ("C2 monitor boundary", cov["acceptance"]["C2_monitor_boundary_reported"]),
        ("C2 hazards detected", cov["acceptance"]["all_public_hazards_detected"]),
        ("C2 no false positives", cov["acceptance"]["no_false_positives_on_safe"]),
        ("C3 SIR activation", act["acceptance"]["C3_activation_effect_sir"]),
        ("C3 Schelling activation", act["acceptance"]["C3_activation_effect_schelling"]),
        ("C4 performance rows", bool(perf.get("rows"))),
        ("C4 AMBER↔FLAME pairs", bool(perf.get("amber_vs_flame"))),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(claims))
    colors = [GREEN if v else RED for _, v in claims]
    ax.barh(y, [1] * len(claims), color=colors, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}:  {'PASS' if v else 'FAIL'}" for n, v in claims], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Campaign claim dashboard — host_a (RTX 3090) / RTX 3090")
    fig.tight_layout()
    save(fig, "fig00_claims_dashboard")


def main():
    style()
    FIGS.mkdir(parents=True, exist_ok=True)
    plot_claims_dashboard()
    plot_attestation()
    plot_monitor_coverage()
    plot_monitor_overhead()
    plot_sir_activation()
    plot_schelling_activation()
    plot_performance()
    print("all figures in", FIGS)


if __name__ == "__main__":
    main()
