#!/usr/bin/env python3
"""Priority 3 — Split multi-framework historical context from authoritative endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "paper-fix-work" / "artifacts" / "benchmark_results_all5090_reconciled.json"
OUT = REPO / "paper-fix-work" / "campaign_results" / "figs_all_frameworks"

BLUE = "#1f4e79"
COPPER = "#c4713b"
GRAY = "#8a9099"
BG = "#fbfaf8"

FW_MUTED = {
    "AMBER (GPU)": ("#1f4e79", 2.0),
    "AMBER (vectorized)": ("#7aa0c4", 1.2),
    "AMBER (loop)": ("#a8c0d4", 1.0),
    "FLAME GPU 2": ("#c4713b", 2.0),
    "mesa-frames": ("#6a9a7a", 1.0),
    "Agents.jl": ("#8a7aaa", 1.0),
    "SimPy": ("#b09a7a", 0.9),
    "Melodie": ("#7a9a98", 0.9),
    "AgentPy": ("#9a7a88", 0.9),
    "Mesa": ("#999999", 0.9),
}
MODELS = ["wealth_transfer", "random_walk", "sir_epidemic", "schelling"]
TITLES = {
    "wealth_transfer": "Wealth",
    "random_walk": "Random walk",
    "sir_epidemic": "SIR",
    "schelling": "Schelling (setup-inclusive)",
}


def style():
    mpl.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#d0d4d8", "grid.linewidth": 0.5,
        "font.family": "DejaVu Sans", "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def main():
    style()
    data = json.loads(SRC.read_text())
    rows = data["results"]
    recon = data.get("reconciliation") or {}

    # Panel A: historical curves excluding 10M outlined join
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    axes = axes.ravel()
    for ax, model in zip(axes, MODELS):
        for fw, (color, lw) in FW_MUTED.items():
            pts = [
                (r["n_agents"], r["execution_time"], r.get("runs", 1))
                for r in rows
                if r["model"] == model and r["framework"] == fw and r["n_agents"] < 10_000_000
            ]
            pts.sort()
            if not pts:
                continue
            ns = [p[0] for p in pts]
            ts = [p[1] for p in pts]
            runs = [p[2] for p in pts]
            ax.plot(ns, ts, "-", color=color, lw=lw, alpha=0.85, label=fw)
            # single-run markers
            for n, t, run in pts:
                if run is not None and int(run) <= 1:
                    ax.plot(n, t, "x", color=color, markersize=6)
                else:
                    ax.plot(n, t, "o", color=color, markersize=3, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(TITLES[model])
        ax.set_xlabel("N agents")
        ax.set_ylabel("wall time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    # de-dup
    seen = {}
    for h, lab in zip(handles, labels):
        seen[lab] = h
    fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=7.5)
    fig.suptitle(
        "Panel A — Multi-framework scaling context (historical summary; 10M endpoints not joined)\n"
        "Timing convention: mean after trimming slowest sample when runs≥3; × = single-run cells",
        fontsize=10, y=1.08,
    )
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "panelA_historical_scaling.png")
    fig.savefig(OUT / "panelA_historical_scaling.pdf")
    plt.close(fig)
    print("wrote panel A")

    # Panel B: authoritative 10M only AMBER GPU + FLAME
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    models_b = []
    amber_t, flame_t = [], []
    amber_samples, flame_samples = [], []
    for model in MODELS:
        a = next((r for r in rows if r["model"] == model and r["framework"] == "AMBER (GPU)" and r["n_agents"] == 10_000_000), None)
        f = next((r for r in rows if r["model"] == model and r["framework"] == "FLAME GPU 2" and r["n_agents"] == 10_000_000), None)
        if not a and not f:
            continue
        models_b.append(TITLES[model])
        amber_t.append(a["execution_time"] if a else np.nan)
        flame_t.append(f["execution_time"] if f else np.nan)
        amber_samples.append(a.get("raw_samples") or [] if a else [])
        flame_samples.append(f.get("raw_samples") or [] if f else [])

    x = np.arange(len(models_b))
    w = 0.36
    ax.bar(x - w / 2, amber_t, width=w, color=BLUE, label="AMBER (GPU) mean", alpha=0.85)
    ax.bar(x + w / 2, flame_t, width=w, color=COPPER, label="FLAME GPU 2 mean", alpha=0.85)
    # raw samples + median
    for i, (asamps, fsamps) in enumerate(zip(amber_samples, flame_samples)):
        if asamps:
            ax.scatter(np.full(len(asamps), i - w / 2), asamps, color="white", edgecolor=BLUE, s=18, zorder=3)
            ax.plot(i - w / 2, np.median(asamps), "_", color=BLUE, markersize=18, markeredgewidth=2)
        if fsamps:
            ax.scatter(np.full(len(fsamps), i + w / 2), fsamps, color="white", edgecolor=COPPER, s=18, zorder=3)
            ax.plot(i + w / 2, np.median(fsamps), "_", color=COPPER, markersize=18, markeredgewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(models_b)
    ax.set_ylabel("wall time (s)")
    ax.set_yscale("log")
    ax.set_title(
        "Panel B — Authoritative 10M endpoint campaign (all 10 runs retained; arithmetic mean bars)\n"
        "Dots = raw samples; ticks = median. Schelling is setup-inclusive implementation comparison."
    )
    ax.legend(frameon=False)
    # annotate ratios for non-schelling
    for i, name in enumerate(models_b):
        if "Schelling" in name:
            continue
        if np.isfinite(amber_t[i]) and np.isfinite(flame_t[i]) and amber_t[i] > 0:
            ratio = flame_t[i] / amber_t[i]
            ax.text(i, max(amber_t[i], flame_t[i]) * 1.15, f"FLAME/AMBER={ratio:.2f}×", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "panelB_authoritative_10m.png")
    fig.savefig(OUT / "panelB_authoritative_10m.pdf")
    plt.close(fig)
    print("wrote panel B")

    # Write caption helper JSON
    meta = {
        "panel_A": "historical multi-framework context; no 10M line join",
        "panel_B": "authoritative 10M AMBER GPU vs FLAME; raw samples shown",
        "schelling_note": "setup-inclusive implementation comparison; do not put 63.4× in abstract",
        "timing_rename": {
            "cold": "first invocation in campaign process",
            "warm": "subsequent invocation",
            "steady": "third and later invocation",
        },
        "reconciliation": recon,
    }
    (OUT / "panel_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
