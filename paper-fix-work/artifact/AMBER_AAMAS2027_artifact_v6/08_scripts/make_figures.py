#!/usr/bin/env python3
"""Regenerate campaign figures from packaged JSON under data/.

Works after ``python 08_scripts/prepare_v5.py`` or when ``data/`` is already
populated. Does not require GPU for figure regeneration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGS = ROOT / "figs"

BLUE = "#1f4e79"
BLUE_LIGHT = "#5b8fbf"
COPPER = "#c4713b"
COPPER_LIGHT = "#e0a070"
GREEN = "#2f6b4f"
RED = "#a33b3b"
GRAY = "#5c6670"
GRAY_LIGHT = "#c5ccd3"
BG = "#fbfaf8"


def style() -> None:
    mpl.rcParams.update(
        {
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
        }
    )


def load(name: str) -> dict:
    path = DATA / name
    if not path.is_file():
        # common aliases
        alts = {
            "sir_crossing_host_b.json": [
                "sir_crossing_host_b.json",
                "../04_activation/sir_crossing_host_b.json",
            ],
            "benchmark_results_host_b_10m.json": [
                "benchmark_results_host_b_10m.json",
                "benchmark_results_10m.json",
                "../06_performance/benchmark_results_10m.json",
            ],
        }
        for rel in alts.get(name, []):
            cand = DATA / rel if not rel.startswith("..") else ROOT / rel.lstrip("./")
            if cand.is_file():
                path = cand
                break
        else:
            raise FileNotFoundError(
                f"missing {name} under {DATA} (run: python 08_scripts/prepare_v5.py)"
            )
    return json.loads(path.read_text())


def save(fig: plt.Figure, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", FIGS / f"{name}.png")


def plot_attestation() -> None:
    att = load("attestation_host_b.json")
    rows = att["summaries"]
    positives = [r for r in rows if not r["is_negative"]]
    negatives = [r for r in rows if r["is_negative"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
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

    nlabels = [f"{r['workload']}\n{r['backend']}" for r in negatives]
    ny = np.arange(len(nlabels))
    colors = [GREEN if r["detected_as_divergent"] else RED for r in negatives]
    axes[1].barh(ny, [r["cases_with_mismatch"] for r in negatives], color=colors, height=0.55)
    axes[1].set_yticks(ny)
    axes[1].set_yticklabels(nlabels, fontsize=7.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Cases with ≥1 state mismatch")
    axes[1].set_title(f"Negative controls {att.get('negative_controls_detected', '')}")
    fig.suptitle("Semantic attestation — Host B (RTX 5090)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig01_attestation")


def plot_boundary() -> None:
    bm = load("boundary_matrix_host_b.json")
    rows = bm["rows"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(rows))
    colors = [GREEN if r.get("match_expect") else RED for r in rows]
    ax.barh(y, [1] * len(rows), color=colors, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r['name']} [{r.get('semantic_category','')}] → {r.get('operational_outcome','')}" for r in rows],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xticks([])
    summ = bm.get("summary", {})
    ax.set_title(
        f"Monitor boundary matrix — {summ.get('match_expect', '?')}/{summ.get('n_cases', '?')} match expect"
    )
    fig.tight_layout()
    save(fig, "fig02_monitor_coverage")


def plot_overhead() -> None:
    ovh = load("overhead_host_b.json")
    ov = ovh["overhead"]
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
            ax.plot([p["n"] for p in pts], [p["ratio"] for p in pts], "o-", color=cmap[q], label=f"k={q}", markersize=5)
        ax.set_xscale("log")
        ax.set_title(f"schema columns c={c}")
        ax.set_xlabel("N agents")
        ax.axhline(1.0, color=GRAY, ls="--", lw=0.9)
    axes[0].set_ylabel("median time ratio  check / off")
    axes[-1].legend(frameon=False, title="commits/step k")
    fig.suptitle("Monitor overhead surface (Host B)", fontsize=12, y=1.03)
    fig.tight_layout()
    save(fig, "fig03_monitor_overhead_ratio")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for q in qs:
        pts = sorted([o for o in ov if o["c"] == max(cs) and o["q"] == q], key=lambda x: x["n"])
        if not pts:
            continue
        ax.plot([p["n"] for p in pts], [p["per_step_ms"] for p in pts], "o-", color=cmap[q], label=f"k={q}", markersize=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N agents")
    ax.set_ylabel("overhead per step (ms)")
    ax.set_title(f"Absolute monitor overhead (c={max(cs)})")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig04_monitor_overhead_abs")


def plot_activation() -> None:
    act = load("activation_host_b.json")
    rows = act["sir"]["rows"]
    # Host-B activation rows may use final_I_diff or nested structures
    taus, means, los, his = [], [], [], []
    for r in rows:
        taus.append(r["tau"])
        d = r.get("final_I_diff") or r.get("delta") or {}
        if isinstance(d, dict) and "mean" in d:
            means.append(d["mean"])
            los.append(d.get("lo", d["mean"]))
            his.append(d.get("hi", d["mean"]))
        else:
            means.append(float("nan"))
            los.append(float("nan"))
            his.append(float("nan"))
    if any(np.isfinite(means)):
        yerr = np.array([[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]])
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        ax.errorbar(taus, means, yerr=yerr, fmt="o-", color=BLUE, ecolor=BLUE_LIGHT, capsize=3, markersize=5)
        ax.axhline(0, color=GRAY, ls="--", lw=1)
        ax.set_xlabel("transmission τ")
        ax.set_ylabel("Δ final infected (seq − snap)")
        ax.set_title("SIR activation effect (Host B)")
        fig.tight_layout()
        save(fig, "fig05_sir_activation")

    # Schelling
    sc = act.get("schelling") or {}
    srows = sc.get("rows") or []
    if srows:
        sides = [r["side"] for r in srows]
        means = [r["cell_disagreement"]["mean"] for r in srows]
        los = [r["cell_disagreement"]["lo"] for r in srows]
        his = [r["cell_disagreement"]["hi"] for r in srows]
        yerr = np.array([[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]])
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.errorbar(sides, means, yerr=yerr, fmt="o-", color=BLUE, ecolor=BLUE_LIGHT, capsize=3)
        ax.set_xlabel("grid side")
        ax.set_ylabel("cell disagreement fraction")
        ax.set_title("Schelling activation (Host B)")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        save(fig, "fig06_schelling_activation")


def plot_sir_crossing() -> None:
    cross = load("sir_crossing_host_b.json")
    rows = cross["rows"]
    taus = [r["tau"] for r in rows]
    snap = [r["mean_A_snap"] for r in rows]
    seq = [r["mean_A_seq"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(taus, snap, "s-", color=BLUE, label="snapshot", markersize=4)
    ax.plot(taus, seq, "o-", color=COPPER, label="sequential", markersize=4)
    ax.set_xlabel("τ")
    ax.set_ylabel("mean cumulative attack $A_T$")
    ax.set_title("SIR cumulative attack curves (Host B)")
    ax.legend(frameon=False)
    # annotate primary crossing
    c = cross.get("crossings", {}).get("0.5", {})
    sh = c.get("paired_shift_seq_minus_snap", {})
    if sh:
        ax.text(
            0.98,
            0.05,
            f"crossing@0.5 shift={sh.get('median', float('nan')):.3f}\n"
            f"CI=[{sh.get('lo', float('nan')):.3f},{sh.get('hi', float('nan')):.3f}]",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=COPPER,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRAY_LIGHT),
        )
    fig.tight_layout()
    save(fig, "fig05b_sir_crossing")


def plot_performance() -> None:
    perf = load("performance_host_b.json")
    rows = [r for r in perf.get("rows", []) if r.get("status") == "success" and r.get("warm_median_s") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for ax, workload in zip(axes, ("wealth", "random_walk")):
        for framework, color, marker, label in [
            ("AMBER_vectorized_cpu", GRAY, "s", "AMBER vectorized CPU"),
            ("AMBER_gpu", BLUE, "o", "AMBER GPU"),
            ("FLAME_GPU_2", COPPER, "D", "FLAME GPU 2"),
        ]:
            pts = sorted(
                [
                    r
                    for r in rows
                    if r.get("track") == "native_idiom"
                    and r.get("workload") == workload
                    and r.get("framework") == framework
                ],
                key=lambda x: x["population"],
            )
            if not pts:
                continue
            ax.plot(
                [p["population"] for p in pts],
                [p["warm_median_s"] for p in pts],
                marker=marker,
                color=color,
                label=label,
                markersize=6,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N agents")
        ax.set_ylabel("warm median wall time (s)")
        ax.set_title(f"{workload} · 50 steps")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Native-idiom performance — Host B (RTX 5090)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig07_native_scaling")

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
                marker=marker,
                color=color,
                label=workload,
                markersize=6,
            )
        ax.axhline(1.0, color=GRAY, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("N agents")
        ax.set_ylabel("AMBER GPU / FLAME (warm median)")
        ax.set_title("< 1 means AMBER faster")
        ax.legend(frameon=False)
        fig.tight_layout()
        save(fig, "fig08_amber_vs_flame")


def plot_10m_endpoints() -> None:
    d = load("benchmark_results_host_b_10m.json")
    results = d.get("results") or []
    # group by model
    models = []
    for r in results:
        if r.get("model") not in models:
            models.append(r["model"])
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * max(len(models), 1), 4.0), sharey=False)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        rows = [r for r in results if r["model"] == model]
        labels = [r["framework"] for r in rows]
        vals = [r["execution_time"] for r in rows]
        colors = [BLUE if "AMBER" in f else COPPER for f in labels]
        x = np.arange(len(labels))
        ax.bar(x, vals, color=colors, width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("wall time (s)")
        ax.set_title(model.replace("_", " "))
        ax.set_yscale("log")
    fig.suptitle("10M agents · 50 steps · 10 runs (Host B)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig11_10m_endpoints")


def plot_claims_dashboard() -> None:
    att = load("attestation_host_b.json")
    bm = load("boundary_matrix_host_b.json")
    act = load("activation_host_b.json")
    cross = load("sir_crossing_host_b.json")
    claims = [
        ("C1 GPU-style zero mismatch", att.get("positive_backends_zero_mismatch", False)),
        ("C1 negatives detected", att.get("negative_all_detected", False)),
        ("C2 boundary match", bm.get("summary", {}).get("match_expect", 0) == bm.get("summary", {}).get("n_cases", -1)),
        ("C3 SIR activation", act.get("acceptance", {}).get("C3_activation_effect_sir", False)),
        ("C3 Schelling activation", act.get("acceptance", {}).get("C3_activation_effect_schelling", False)),
        (
            "C3 SIR crossing@0.5",
            cross.get("acceptance", {}).get("crossing_0.5_shift_excludes_zero", False),
        ),
        ("C4 10M endpoints present", (DATA / "benchmark_results_host_b_10m.json").is_file()),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    y = np.arange(len(claims))
    colors = [GREEN if v else RED for _, v in claims]
    ax.barh(y, [1] * len(claims), color=colors, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}:  {'PASS' if v else 'FAIL'}" for n, v in claims], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Host-B campaign claim dashboard")
    fig.tight_layout()
    save(fig, "fig00_claims_dashboard")


def main() -> int:
    if not DATA.is_dir() or not any(DATA.iterdir()):
        prep = ROOT / "08_scripts" / "prepare_v5.py"
        if prep.is_file():
            print("data/ empty — running prepare_v5.py …")
            import runpy

            runpy.run_path(str(prep), run_name="__main__")
        else:
            print("error: data/ missing and no prepare_v5.py", file=sys.stderr)
            return 2
    style()
    FIGS.mkdir(parents=True, exist_ok=True)
    plot_claims_dashboard()
    plot_attestation()
    plot_boundary()
    plot_overhead()
    plot_activation()
    plot_sir_crossing()
    plot_performance()
    plot_10m_endpoints()
    print(f"figures written to {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
