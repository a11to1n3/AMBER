#!/usr/bin/env python3
"""Regenerate plot07 (Figure 6) from benchmark_results_all5090.json.

Focused layout: full-width 4-panel log-log scaling with AMBER (GPU) emphasized.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from amber_figure_style import (  # noqa: E402
    FRAMEWORK_COLORS,
    FULL_W,
    PANEL_H,
    apply_figure_style,
    panel_label,
    save_figure,
    style_axes,
)

JSON_PATH = SCRIPT_DIR.parent / "artifacts" / "benchmark_results_all5090.json"
OUT_PATH = SCRIPT_DIR.parent / "figs" / "plot07.png"

MODEL_ORDER = [
    ("wealth_transfer", "Wealth transfer"),
    ("random_walk", "Random walk"),
    ("sir_epidemic", "SIR epidemic"),
    ("schelling", "Schelling"),
]

FRAMEWORK_ORDER = [
    "AMBER (GPU)",
    "FLAME GPU 2",
    "mesa-frames",
    "AMBER (vectorized)",
    "AMBER (loop)",
    "Agents.jl",
    "AgentPy",
    "Mesa",
    "Melodie",
    "SimPy",
]

HIGHLIGHT = {"AMBER (GPU)", "FLAME GPU 2"}
SECONDARY = {"mesa-frames", "AMBER (vectorized)"}


def load_results(path: Path) -> tuple[dict, list[int]]:
    data = json.loads(path.read_text())
    grid: dict[tuple[str, str, int], float] = {}
    for row in data["results"]:
        grid[(row["framework"], row["model"], row["n_agents"])] = row["execution_time"]
    return grid, data["agent_counts"]


def main() -> None:
    apply_figure_style()
    grid, agent_counts = load_results(JSON_PATH)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(FULL_W, PANEL_H),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.28, wspace=0.28)

    handles: dict[str, object] = {}
    for ax, (model_key, model_label), panel in zip(axes, MODEL_ORDER, "abcd"):
        panel_label(ax, panel, x=-0.14, y=1.08)
        for framework in FRAMEWORK_ORDER:
            xs, ys = [], []
            for n in agent_counts:
                t = grid.get((framework, model_key, n))
                if t is not None and t > 0:
                    xs.append(n)
                    ys.append(t)
            if not xs:
                continue

            color = FRAMEWORK_COLORS.get(framework, "#9ca3af")
            if framework == "AMBER (GPU)":
                lw, alpha, ms, z = 3.2, 1.0, 6, 10
            elif framework in HIGHLIGHT:
                lw, alpha, ms, z = 2.2, 0.95, 5, 6
            elif framework in SECONDARY:
                lw, alpha, ms, z = 1.6, 0.85, 4, 4
            else:
                lw, alpha, ms, z = 1.2, 0.55, 3.5, 2

            (line,) = ax.plot(
                xs,
                ys,
                color=color,
                linewidth=lw,
                alpha=alpha,
                marker="o",
                markersize=ms,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.4,
                zorder=z,
                label=framework,
            )
            handles.setdefault(framework, line)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(model_label, fontsize=9.5, pad=4)
        ax.set_xlabel("Agents")
        if ax is axes[0]:
            ax.set_ylabel("Wall time (s)")
        style_axes(ax)

    legend_order = [f for f in FRAMEWORK_ORDER if f in handles]
    fig.legend(
        [handles[f] for f in legend_order],
        legend_order,
        loc="lower center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.0,
        handlelength=2.2,
    )
    fig.suptitle(
        "ABM framework scaling on RTX 5090 (Blackwell) — 10 frameworks, up to 10M agents",
        fontsize=10.5,
        y=0.96,
    )
    save_figure(fig, OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
