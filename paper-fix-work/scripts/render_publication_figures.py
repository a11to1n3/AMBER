#!/usr/bin/env python3
"""Render every figure used by the AMBER AAMAS manuscript.

Each function has a single analytical job and reads a released JSON artifact.
Exports are deterministic vector PDFs plus PNG previews.  The renderer does
not alter, interpolate, remove, or reorder experimental observations.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figs"
sys.path.insert(0, str(SCRIPT_DIR))

from publication_figure_style import (  # noqa: E402
    AMBER,
    AMBER_DARK,
    AMBER_LIGHT,
    AMBER_PALE,
    BLUE,
    BLUE_DARK,
    BLUE_LIGHT,
    BLUE_MID,
    BLUE_PALE,
    CATEGORICAL,
    CHARCOAL,
    COLUMN_WIDTH,
    FULL_WIDTH,
    INK,
    LIGHT_GREY,
    MID_GREY,
    PALE_GREY,
    PLUM,
    PLUM_LIGHT,
    SLATE,
    TEAL,
    TEAL_LIGHT,
    WHITE,
    apply_style,
    clean_axes,
    compact_seconds,
    panel_label,
    save_both,
)


def load_json(name: str):
    return json.loads((ARTIFACTS / name).read_text())


def _box(ax, xy, width, height, title, body, *, edge, face, title_size=8.4, body_size=7.0):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        ec=edge,
        fc=face,
        lw=1.15,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.025,
        y + height - 0.045,
        title,
        ha="left",
        va="top",
        color=edge,
        fontsize=title_size,
        fontweight="bold",
    )
    ax.text(
        x + 0.025,
        y + height - 0.115,
        body,
        ha="left",
        va="top",
        color=INK,
        fontsize=body_size,
        linespacing=1.28,
    )
    return patch


def _arrow(ax, start, end, *, color=SLATE, width=1.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            lw=width,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def render_plot01() -> None:
    """Restrained vector schematic of AMBER's hybrid execution position."""
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.plot([0.145, 0.855], [0.895, 0.895], color=LIGHT_GREY, lw=1.0, zorder=0)
    ax.scatter([0.145, 0.5, 0.855], [0.895, 0.895, 0.895], s=[12, 20, 12], color=[BLUE, INK, AMBER], zorder=1)
    ax.text(0.145, 0.940, "agent-centric emphasis", ha="center", va="bottom", fontsize=5.3, color=BLUE_DARK)
    ax.text(0.855, 0.940, "array / accelerator emphasis", ha="center", va="bottom", fontsize=5.3, color=AMBER_DARK)

    side_specs = [
        (0.030, "Agent-centric / OOP", "AgentPy · Mesa · Agents.jl\nMelodie · SimPy", "object/struct per agent\n+ scheduler loop", BLUE_DARK, BLUE_PALE),
        (0.730, "Columnar / accelerator", "mesa-frames · FLAME GPU 2\nAgentTorch*", "column, tensor, or\ndevice execution", AMBER_DARK, AMBER_PALE),
    ]
    for x, title, examples, footer, color, face in side_specs:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.280),
                0.240,
                0.455,
                boxstyle="round,pad=0.009,rounding_size=0.012",
                ec=color,
                fc=WHITE,
                lw=0.8,
                zorder=2,
            )
        )
        ax.add_patch(Rectangle((x + 0.001, 0.605), 0.238, 0.129, ec="none", fc=face, zorder=2.5))
        ax.text(x + 0.018, 0.685, title, ha="left", va="center", fontsize=6.9, color=color, fontweight="bold", zorder=4)
        ax.text(x + 0.018, 0.535, examples, ha="left", va="center", fontsize=5.55, color=CHARCOAL, linespacing=1.22, zorder=4)
        ax.text(x + 0.120, 0.365, footer, ha="center", va="center", fontsize=5.45, color=color, linespacing=1.18, zorder=4)

    ax.add_patch(
        FancyBboxPatch(
            (0.300, 0.215),
            0.400,
            0.585,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            ec=INK,
            fc=WHITE,
            lw=1.2,
            zorder=2,
        )
    )
    ax.add_patch(Rectangle((0.301, 0.665), 0.398, 0.134, ec="none", fc=INK, zorder=2.5))
    ax.text(0.325, 0.745, "AMBER", ha="left", va="center", fontsize=8.5, color=WHITE, fontweight="bold", zorder=4)
    ax.text(0.420, 0.745, "one model / run API", ha="left", va="center", fontsize=6.0, color=BLUE_LIGHT, zorder=4)
    ax.text(0.500, 0.690, 'cpu(mode="oop")  ·  cpu(mode="vectorized")  ·  gpu()', ha="center", va="center", fontsize=5.35, color=WHITE, zorder=4)

    modes = [
        (0.325, 0.095, "OOP", BLUE_PALE, BLUE_DARK),
        (0.430, 0.165, "vectorized\ncolumnar", "#E7F0F4", BLUE_DARK),
        (0.605, 0.070, "GPU", AMBER_PALE, AMBER_DARK),
    ]
    for x, mode_width, label, face, color in modes:
        ax.add_patch(FancyBboxPatch((x, 0.485), mode_width, 0.115, boxstyle="round,pad=0.006,rounding_size=0.009", ec=color, fc=face, lw=0.75, zorder=3))
        ax.text(x + mode_width / 2, 0.543, label, ha="center", va="center", fontsize=5.7, color=color, fontweight="bold", linespacing=1.0, zorder=4)
    ax.plot([0.333, 0.585], [0.445, 0.445], color=BLUE_MID, lw=1.0)
    ax.text(0.459, 0.418, "instrumented development paths", ha="center", va="top", fontsize=5.1, color=BLUE_DARK)
    ax.plot([0.608, 0.672], [0.445, 0.445], color=AMBER_LIGHT, lw=1.0)
    ax.text(0.640, 0.418, "approval-\ngated", ha="center", va="top", fontsize=4.8, color=AMBER_DARK, linespacing=1.0)
    ax.text(0.500, 0.300, "High throughput  +  explicit activation alignment", ha="center", va="center", fontsize=6.2, color=INK, fontweight="bold")

    ax.text(
        0.500,
        0.080,
        "Placement describes released examples, not universal capability limits.  *AgentTorch: capability-only; no matched timing row.",
        ha="center",
        va="bottom",
        fontsize=5.15,
        color=SLATE,
    )

    save_both(fig, FIGURES, "plot01")


def render_plot02() -> None:
    """Decision-driven AMBER development-to-deployment flowchart."""
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.18))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def process(cx, cy, width, height, title, body="", *, edge=BLUE_DARK, face=WHITE, title_size=6.1, body_size=4.8):
        ax.add_patch(
            FancyBboxPatch(
                (cx - width / 2, cy - height / 2),
                width,
                height,
                boxstyle="round,pad=0.005,rounding_size=0.008",
                ec=edge,
                fc=face,
                lw=0.9,
                zorder=3,
            )
        )
        ax.text(
            cx,
            cy + (0.026 if body else 0),
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=edge,
            fontweight="bold",
            linespacing=1.0,
            zorder=4,
        )
        if body:
            ax.text(
                cx,
                cy - 0.035,
                body,
                ha="center",
                va="center",
                fontsize=body_size,
                color=CHARCOAL,
                linespacing=1.08,
                zorder=4,
            )

    def decision(cx, cy, width, height, label, *, edge=INK):
        points = [
            (cx, cy + height / 2),
            (cx + width / 2, cy),
            (cx, cy - height / 2),
            (cx - width / 2, cy),
        ]
        ax.add_patch(Polygon(points, closed=True, ec=edge, fc=WHITE, lw=1.0, zorder=3))
        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=5.1,
            color=edge,
            fontweight="bold",
            linespacing=1.0,
            zorder=4,
        )

    def terminator(cx, cy, width, height, label, *, face=INK):
        ax.add_patch(
            FancyBboxPatch(
                (cx - width / 2, cy - height / 2),
                width,
                height,
                boxstyle=f"round,pad=0.006,rounding_size={height / 2}",
                ec=face,
                fc=face,
                lw=0.9,
                zorder=3,
            )
        )
        ax.text(cx, cy, label, ha="center", va="center", fontsize=5.4, color=WHITE, fontweight="bold", linespacing=1.0, zorder=4)

    def flow_arrow(start, end, *, color=SLATE, width=0.78, curvature=0.0):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                connectionstyle=f"arc3,rad={curvature}",
                arrowstyle="-|>",
                mutation_scale=7.2,
                shrinkA=0.6,
                shrinkB=0.8,
                lw=width,
                capstyle="round",
                joinstyle="round",
                color=color,
                zorder=2,
            )
        )

    def path_arrow(vertices, codes, *, color=SLATE, width=0.78):
        ax.add_patch(
            FancyArrowPatch(
                path=MplPath(vertices, codes),
                arrowstyle="-|>",
                mutation_scale=7.2,
                lw=width,
                capstyle="round",
                joinstyle="round",
                color=color,
                zorder=2,
            )
        )

    def branch_label(x, y, label, *, color=SLATE):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=4.5,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.08", fc=WHITE, ec="none", alpha=0.96),
            zorder=5,
        )

    ax.text(0.50, 0.958, "AMBER DEVELOPMENT-TO-DEPLOYMENT FLOW", ha="center", va="center", fontsize=7.2, color=INK, fontweight="bold")
    ax.text(0.50, 0.895, "Native placement changes storage and dispatch; approval does not prove equivalence.", ha="center", va="center", fontsize=5.2, color=SLATE)
    ax.plot([0.025, 0.975], [0.855, 0.855], color=LIGHT_GREY, lw=0.7, zorder=0)

    # Main development and validation spine.
    terminator(0.065, 0.690, 0.095, 0.120, "Define\nintent", face=BLUE_DARK)
    process(0.230, 0.690, 0.200, 0.180, "Select native placement", 'cpu(mode="oop")\ncpu(mode="vectorized") · gpu()', face=BLUE_PALE, body_size=4.35)
    process(0.440, 0.690, 0.162, 0.155, "Run with contract", "check · warn · raise\n→ runtime report", face=BLUE_PALE)
    decision(0.590, 0.690, 0.110, 0.160, "Hazard\nreported?", edge=BLUE_DARK)
    process(0.744, 0.690, 0.158, 0.155, "Workload evidence", "coupled / distributional\nor invariants", edge=INK)
    decision(0.920, 0.690, 0.110, 0.160, "Caller\napproves?", edge=INK)

    flow_arrow((0.113, 0.690), (0.130, 0.690), color=BLUE_DARK, width=0.78)
    flow_arrow((0.330, 0.690), (0.355, 0.690), color=BLUE_DARK, width=0.78)
    flow_arrow((0.521, 0.690), (0.535, 0.690), color=BLUE_DARK, width=0.78)
    flow_arrow((0.645, 0.690), (0.665, 0.690), color=BLUE_DARK, width=0.78)
    flow_arrow((0.823, 0.690), (0.865, 0.690), color=INK, width=0.78)
    branch_label(0.653, 0.725, "no", color=BLUE_DARK)

    # Local correction loops are visually subordinate to the forward spine.
    process(0.500, 0.445, 0.140, 0.095, "Repair or stage", "writes / schedule", edge=BLUE_DARK, title_size=5.3, body_size=4.25)
    flow_arrow((0.590, 0.610), (0.552, 0.493), color=BLUE_MID, width=0.70)
    branch_label(0.585, 0.555, "yes", color=BLUE_DARK)
    path_arrow(
        [(0.430, 0.445), (0.423, 0.445), (0.410, 0.445), (0.410, 0.458), (0.410, 0.610)],
        [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO],
        color=BLUE_MID,
        width=0.70,
    )

    # A failed reference returns locally after model/kernel revision.
    flow_arrow((0.920, 0.770), (0.744, 0.770), color=SLATE, width=0.70, curvature=0.34)
    branch_label(0.830, 0.815, "no · revise")

    # A passed reference reaches the deployment gate, then both runtime paths
    # reconverge at the synchronized public-API exit.
    decision(0.600, 0.270, 0.140, 0.160, "Fast path\neligible?", edge=AMBER_DARK)
    path_arrow(
        [
            (0.920, 0.610),
            (0.920, 0.548),
            (0.920, 0.535),
            (0.907, 0.535),
            (0.613, 0.535),
            (0.600, 0.535),
            (0.600, 0.522),
            (0.600, 0.350),
        ],
        [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO],
        color=AMBER_DARK,
        width=0.80,
    )
    branch_label(0.938, 0.575, "yes", color=AMBER_DARK)

    process(0.770, 0.380, 0.140, 0.100, "General runner", "selected native mode", edge=BLUE_DARK, face=BLUE_PALE, title_size=5.35, body_size=4.05)
    process(0.770, 0.160, 0.140, 0.100, "Private optimized loop", "approval-gated", edge=AMBER_DARK, face=AMBER_PALE, title_size=5.0, body_size=4.05)
    flow_arrow((0.670, 0.290), (0.700, 0.360), color=BLUE_DARK, width=0.76)
    flow_arrow((0.670, 0.250), (0.700, 0.180), color=AMBER_DARK, width=0.76)
    branch_label(0.674, 0.385, "no", color=BLUE_DARK)
    branch_label(0.674, 0.160, "yes", color=AMBER_DARK)

    terminator(0.925, 0.270, 0.140, 0.140, "Sync + assemble\nretain timing", face=INK)
    flow_arrow((0.840, 0.380), (0.855, 0.310), color=BLUE_DARK, width=0.76)
    flow_arrow((0.840, 0.160), (0.855, 0.230), color=AMBER_DARK, width=0.76)

    save_both(fig, FIGURES, "plot02")


def render_plot03() -> None:
    """Controlled finite-horizon SIR crossing experiment."""
    apply_style()
    data = load_json("emergence_threshold_controlled.json")
    taus = np.asarray(data["protocol"]["taus"], float)
    curves = data["curves"]
    crossings = data["tau_c"]

    series = [
        ("rowwise_snapshot_mean", "rowwise_snapshot_sd", CHARCOAL, "o", "-", "Row-wise snapshot"),
        ("batched_gpu_snapshot_mean", "batched_gpu_snapshot_sd", BLUE, "s", "-", "Batched GPU snapshot"),
        ("inplace_ordered_mean", "inplace_ordered_sd", AMBER, "^", "--", "In-place ordered"),
    ]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.68))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.82)
    for mean_key, sd_key, color, marker, line_style, label in series:
        y = np.asarray(curves[mean_key], float)
        sd = np.asarray(curves[sd_key], float)
        ax.fill_between(taus, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1), color=color, alpha=0.08, lw=0)
        ax.plot(
            taus,
            y,
            color=color,
            marker=marker,
            markevery=2,
            linestyle=line_style,
            lw=1.65,
            label=label,
            zorder=3,
        )

    rowwise = crossings["rowwise_snapshot"]
    gpu = crossings["batched_gpu_snapshot"]
    ordered = crossings["inplace_ordered"]
    ax.axhline(0.5, color=SLATE, ls=":", lw=0.9)
    ax.text(0.402, 0.5, "attack = 0.5", ha="right", va="bottom", fontsize=6.7, color=SLATE)
    ax.axvspan(rowwise["lo"], rowwise["hi"], color=CHARCOAL, alpha=0.08, lw=0)
    ax.axvline(rowwise["median"], color=CHARCOAL, ls=":", lw=1.0)
    ax.axvline(ordered["median"], color=AMBER, ls=":", lw=1.0)
    ax.text(rowwise["median"], 0.98, f"{rowwise['median']:.3f}", ha="center", va="top", fontsize=6.8, color=CHARCOAL)
    ax.text(ordered["median"], 0.91, f"{ordered['median']:.3f}", ha="center", va="top", fontsize=6.8, color=AMBER_DARK)
    ax.annotate(
        "",
        xy=(ordered["median"], 0.50),
        xytext=(rowwise["median"], 0.50),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0),
    )
    shift = 100 * (rowwise["median"] - ordered["median"]) / rowwise["median"]
    ax.text(
        (ordered["median"] + rowwise["median"]) / 2,
        0.54,
        f"{shift:.0f}% lower τ",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=INK,
        fontweight="bold",
        bbox=dict(fc=WHITE, ec="none", pad=1.0, alpha=0.9),
    )
    if max(rowwise["lo"], gpu["lo"]) <= min(rowwise["hi"], gpu["hi"]):
        ax.text(0.99, 0.88, "row-wise and GPU crossing intervals overlap", transform=ax.transAxes, ha="right", color=BLUE_DARK, fontsize=6.8)

    ax.set_xlabel(r"Per-contact transmissibility $\tau$")
    ax.set_ylabel("Final attack rate")
    ax.set_xlim(taus[0] - 0.003, taus[-1] + 0.005)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", ncol=1)
    clean_axes(ax, grid="both")
    save_both(fig, FIGURES, "plot03")


def render_plot07() -> None:
    """All-framework scaling landscape with final 10M endpoints reconciled."""
    apply_style()
    data = load_json("benchmark_results_all5090_reconciled.json")
    rows = data["results"]
    models = [
        ("wealth_transfer", "Wealth transfer"),
        ("random_walk", "Random walk"),
        ("sir_epidemic", "SIR epidemic"),
        ("schelling", "Schelling"),
    ]
    frameworks = [
        "AMBER (GPU)",
        "AMBER (vectorized)",
        "AMBER (loop)",
        "FLAME GPU 2",
        "mesa-frames",
        "Agents.jl",
        "AgentPy",
        "Mesa",
        "Melodie",
        "SimPy",
    ]
    # AMBER and FLAME are the analytical comparison.  The six remaining
    # implementations stay identifiable, but use neutral context styling so
    # ten saturated series do not compete for attention.
    styles = {
        "AMBER (GPU)": dict(color=BLUE_DARK, marker="o", ls="-", lw=2.20, alpha=1.00),
        "AMBER (vectorized)": dict(color=BLUE, marker="s", ls="--", lw=1.45, alpha=0.98),
        "AMBER (loop)": dict(color=BLUE_MID, marker="^", ls=":", lw=1.35, alpha=0.98),
        "FLAME GPU 2": dict(color=AMBER, marker="D", ls="-", lw=1.95, alpha=1.00),
        "mesa-frames": dict(color="#596772", marker="h", ls="-", lw=1.00, alpha=0.86),
        "Agents.jl": dict(color="#74818B", marker="v", ls="--", lw=0.95, alpha=0.84),
        "AgentPy": dict(color="#89959E", marker="X", ls=":", lw=0.90, alpha=0.82),
        "Mesa": dict(color="#9DA8B0", marker="P", ls="-.", lw=0.90, alpha=0.80),
        "Melodie": dict(color="#B0B9C0", marker=">", ls="--", lw=0.90, alpha=0.82),
        "SimPy": dict(color="#C0C8CE", marker="<", ls=":", lw=0.90, alpha=0.88),
    }
    primary = ["AMBER (GPU)", "AMBER (vectorized)", "AMBER (loop)", "FLAME GPU 2"]
    context = ["mesa-frames", "Agents.jl", "AgentPy", "Mesa", "Melodie", "SimPy"]
    draw_order = context + ["AMBER (loop)", "AMBER (vectorized)", "FLAME GPU 2", "AMBER (GPU)"]

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 3.36), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.090, right=0.992, bottom=0.135, top=0.790, hspace=0.42, wspace=0.18)

    handles: dict[str, Line2D] = {}
    for panel_index, (ax, (model_key, model_label)) in enumerate(zip(axes.flat, models)):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(800, 1.35e7)
        ax.set_ylim(5e-4, 2e2)
        ax.set_title(
            f"({chr(ord('a') + panel_index)})  {model_label}",
            loc="left",
            fontsize=8.4,
            fontweight="semibold",
            pad=5.5,
        )

        final_times: dict[str, float] = {}
        for framework in draw_order:
            framework_rows = sorted(
                (row for row in rows if row["framework"] == framework and row["model"] == model_key),
                key=lambda row: int(row["n_agents"]),
            )
            if not framework_rows:
                continue
            style = styles[framework]
            final_rows = [
                row for row in framework_rows
                if row.get("source_campaign") == "final corrected ten-run 10M campaign"
            ]
            archived_rows = [row for row in framework_rows if row not in final_rows]

            if archived_rows:
                (line,) = ax.plot(
                    [int(row["n_agents"]) for row in archived_rows],
                    [float(row["execution_time"]) for row in archived_rows],
                    color=style["color"],
                    marker=style["marker"],
                    ls=style["ls"],
                    lw=style["lw"],
                    alpha=style["alpha"],
                    ms=4.0 if framework in {"AMBER (GPU)", "FLAME GPU 2"} else 3.1,
                    markerfacecolor=WHITE if framework == "FLAME GPU 2" else style["color"],
                    markeredgecolor=style["color"],
                    markeredgewidth=0.70,
                    zorder=12 if framework == "AMBER (GPU)" else 10 if framework == "FLAME GPU 2" else 6 if framework in primary else 3,
                    label=framework,
                )
                handles.setdefault(framework, line)
            else:
                handles.setdefault(
                    framework,
                    Line2D([0], [0], color=style["color"], marker=style["marker"], ls=style["ls"]),
                )

            # The corrected endpoints use all-ten arithmetic means, whereas
            # historical points use a trimmed convention. Connect the final
            # point with the normal series style for trajectory readability;
            # the outlined marker discloses the campaign/statistic change.
            for row in final_rows:
                final_times[framework] = float(row["execution_time"])
                if archived_rows:
                    previous = archived_rows[-1]
                    ax.plot(
                        [previous["n_agents"], row["n_agents"]],
                        [previous["execution_time"], row["execution_time"]],
                        color=style["color"],
                        linestyle="-",
                        linewidth=style["lw"],
                        alpha=style["alpha"],
                        zorder=11 if framework == "AMBER (GPU)" else 9,
                    )
                ax.plot(
                    [row["n_agents"]],
                    [row["execution_time"]],
                    marker=style["marker"],
                    markersize=6.3,
                    markerfacecolor=WHITE if framework == "FLAME GPU 2" else style["color"],
                    markeredgecolor=INK,
                    markeredgewidth=0.95,
                    linestyle="none",
                    zorder=20,
                )

        clean_axes(ax, grid=None)
        ax.grid(which="major", axis="both", color=LIGHT_GREY, lw=0.45, alpha=0.72)
        ax.grid(which="minor", axis="both", visible=False)
        ax.tick_params(which="minor", length=0)

        if {"AMBER (GPU)", "FLAME GPU 2"} <= final_times.keys():
            ratio = final_times["FLAME GPU 2"] / final_times["AMBER (GPU)"]
            digits = 1 if ratio >= 10 else 2
            ax.text(
                0.975,
                0.065,
                (
                    f"10M · {ratio:.{digits}f}× (setup incl.)"
                    if model_key == "schelling"
                    else f"10M · {ratio:.{digits}f}× speedup"
                ),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.25,
                color=BLUE_DARK,
                fontweight="semibold",
                bbox=dict(boxstyle="round,pad=0.24", fc=WHITE, ec=BLUE_LIGHT, lw=0.55, alpha=0.96),
                zorder=30,
            )

    fig.text(0.022, 0.458, "Wall time (seconds)", rotation=90, ha="center", va="center", fontsize=7.25, color=CHARCOAL)
    fig.text(0.545, 0.038, "Agents", ha="center", va="center", fontsize=7.15, color=CHARCOAL)

    main_legend = fig.legend(
        [handles[name] for name in primary],
        primary,
        loc="upper left",
        ncol=4,
        bbox_to_anchor=(0.087, 0.995),
        frameon=False,
        fontsize=7.25,
        columnspacing=1.45,
        handlelength=2.0,
        handletextpad=0.45,
        borderaxespad=0,
    )
    for text_item in main_legend.get_texts():
        text_item.set_color(INK)
        text_item.set_fontweight("semibold")

    context_legend = fig.legend(
        [handles[name] for name in context],
        context,
        loc="upper left",
        ncol=6,
        bbox_to_anchor=(0.087, 0.922),
        frameon=False,
        fontsize=6.55,
        columnspacing=1.35,
        handlelength=1.65,
        handletextpad=0.38,
        borderaxespad=0,
    )
    for text_item in context_legend.get_texts():
        text_item.set_color(SLATE)

    save_both(fig, FIGURES, "plot07", pad=0.04)


def render_plot11() -> None:
    """Generated-DAG staged-execution control without lower-bound framing."""
    apply_style()
    data = load_json("topological_staging_results.json")
    rows = data["rows"]
    depths = sorted({int(row["ell"]) for row in rows})
    grouped = {depth: [row for row in rows if int(row["ell"]) == depth] for depth in depths}
    full_exact = [mean([math.isclose(row["full_correct"], 1.0) for row in grouped[d]]) for d in depths]
    short_exact = [mean([math.isclose(row["short_correct"], 1.0) for row in grouped[d]]) for d in depths]
    full_cell = [mean([row["full_correct"] for row in grouped[d]]) for d in depths]
    short_cell = [mean([row["short_correct"] for row in grouped[d]]) for d in depths]
    short_sd = [pstdev([row["short_correct"] for row in grouped[d]]) for d in depths]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.42), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.22, top=0.76, wspace=0.30)
    ax = axes[0]
    ax.plot(depths, full_exact, color=BLUE, marker="o", label="Longest-path staging")
    ax.plot(depths, short_exact, color=AMBER, marker="s", markerfacecolor=WHITE, ls="--", label="Deepest layers merged")
    ax.text(depths[-1] + 0.08, full_exact[-1], "18/18", ha="left", va="center", color=BLUE_DARK, fontsize=6.8)
    ax.text(depths[-1] + 0.08, short_exact[-1], "0/18", ha="left", va="center", color=AMBER_DARK, fontsize=6.8)
    ax.set_title("Graph-level exact matches")
    ax.set_ylabel("Fraction of generated graphs")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks(depths)
    panel_label(ax, "a")
    clean_axes(ax, grid="y")

    ax = axes[1]
    ax.plot(depths, full_cell, color=BLUE, marker="o")
    ax.errorbar(
        depths,
        short_cell,
        yerr=short_sd,
        color=AMBER,
        marker="s",
        markerfacecolor=WHITE,
        ls="--",
        capsize=2.5,
    )
    ax.set_title("Cell-level agreement")
    ax.set_ylabel("Sequential-reference agreement")
    ax.set_ylim(0.62, 1.02)
    ax.set_xticks(depths)
    panel_label(ax, "b")
    clean_axes(ax, grid="y")

    for ax in axes:
        ax.set_xlabel("Longest directed-path length")
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.text(0.99, 0.01, "Whiskers: population SD across 18 graphs per depth", ha="right", va="bottom", fontsize=6.3, color=SLATE)
    save_both(fig, FIGURES, "plot11")


def render_plot17() -> None:
    """mesa-frames local NumPy update-block sensitivity."""
    apply_style()
    data = load_json("mf_granularity.json")
    taus = np.asarray(data["taus"], float)
    blocks = [int(value) for value in data["nblocks"]]
    colors = [BLUE_LIGHT, "#95B7CF", BLUE_MID, BLUE, BLUE_DARK]
    line_styles = [(0, (1, 1)), (0, (4, 2)), (0, (6, 2, 1, 2)), "--", "-"]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.55), gridspec_kw={"width_ratios": [1.18, 1]})
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.21, top=0.78, wspace=0.27)
    ax = axes[0]
    for block_count, color, line_style in zip(blocks, colors, line_styles):
        curve = np.asarray(data["by_nblocks"][str(block_count)]["curve"], float)
        ax.plot(taus, curve, color=color, ls=line_style, lw=1.7, label=f"{block_count} local block" + ("" if block_count == 1 else "s"))
    ax.axhline(0.5, color=SLATE, ls=":", lw=0.8)
    ax.set_xlabel(r"Transmissibility $\tau$")
    ax.set_ylabel("Final attack rate")
    ax.set_xlim(taus[0], taus[-1])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", ncol=2, columnspacing=0.8, handlelength=2.0)
    panel_label(ax, "a")
    clean_axes(ax, grid="both")

    ax = axes[1]
    crossings = [float(data["by_nblocks"][str(block_count)]["tc"]) for block_count in blocks]
    reference = data["by_nblocks"]["1"]
    ax.axhspan(reference["tc_ci"][0], reference["tc_ci"][1], color=BLUE_PALE, zorder=0)
    ax.axhline(reference["tc"], color=CHARCOAL, ls=":", lw=0.9, label="One-block estimate")
    ax.plot(blocks, crossings, color=BLUE_DARK, marker="o", lw=1.7)
    ax.set_xscale("log")
    ax.set_xticks(blocks)
    ax.set_xticklabels([str(value) for value in blocks])
    ax.set_xlabel("Local update blocks per step")
    ax.set_ylabel(r"Operational crossing $\tau_c$")
    ax.legend(loc="lower left")
    panel_label(ax, "b")
    clean_axes(ax, grid="both")
    save_both(fig, FIGURES, "plot17")


def render_plot18() -> None:
    """Crossing sensitivity at attack-rate levels 0.3, 0.5, and 0.7."""
    apply_style()
    data = load_json("emergence_threshold_controlled.json")["crossing_sensitivity"]
    methods = [
        ("rowwise_snapshot", "Row-wise snapshot", CHARCOAL, "o"),
        ("batched_gpu_snapshot", "Batched GPU snapshot", BLUE, "s"),
        ("inplace_ordered", "In-place ordered", AMBER, "^"),
    ]
    levels = ["0.3", "0.5", "0.7"]
    y_positions = [2, 1, 0]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.20))
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.23, top=0.75, wspace=0.40)
    for panel_index, (ax, level) in enumerate(zip(axes, levels)):
        values = []
        for y, (key, label, color, marker) in zip(y_positions, methods):
            row = data[level][key]
            values.extend([row["lo"], row["hi"]])
            ax.errorbar(
                row["median"],
                y,
                xerr=[[row["median"] - row["lo"]], [row["hi"] - row["median"]]],
                fmt=marker,
                color=color,
                markerfacecolor=color if marker != "s" else WHITE,
                markeredgewidth=1.0,
                ms=5.0,
                capsize=2.5,
                lw=1.25,
            )
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * 0.18, 0.004)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.55, 2.55)
        ax.set_title(f"({chr(ord('a') + panel_index)})  Attack crossing = {level}")
        ax.set_xlabel(r"$\tau_c$ (median, 95% CI)")
        ax.set_yticks(y_positions)
        if panel_index == 0:
            ax.set_yticklabels([label for _, label, _, _ in methods])
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        clean_axes(ax, grid="x")
    save_both(fig, FIGURES, "plot18")


def render_plot21() -> None:
    """Three exploratory semantic-sensitivity prototypes."""
    apply_style()
    sir = load_json("emergence_science_ext.json")
    schelling = load_json("schelling_gpu.json")
    coordination = load_json("coordination_gpu.json")

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.30))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.23, top=0.73, wspace=0.37)
    snapshot_style = dict(color=BLUE, marker="o", lw=1.55, label="Snapshot / staged")
    ordered_style = dict(color=AMBER, marker="s", markerfacecolor=WHITE, ls="--", lw=1.45, label="In-place ordered")

    ax = axes[0]
    radii = np.asarray(sir["radii"], float)
    staged = np.asarray(sir["tc_staged"], float)
    ordered = np.asarray(sir["tc_fused"], float)
    ax.errorbar(
        radii,
        staged,
        yerr=[staged - np.asarray(sir["boot_staged"]["tc_lo"]), np.asarray(sir["boot_staged"]["tc_hi"]) - staged],
        capsize=2,
        **snapshot_style,
    )
    ax.errorbar(
        radii,
        ordered,
        yerr=[ordered - np.asarray(sir["boot_fused"]["tc_lo"]), np.asarray(sir["boot_fused"]["tc_hi"]) - ordered],
        capsize=2,
        **ordered_style,
    )
    ax.set_xscale("log")
    ax.set_xticks([2, 4, 8, 12])
    ax.set_xticklabels(["2", "4", "8", "12"])
    ax.minorticks_off()
    ax.set_title("(a)  Ring-lattice SIR")
    ax.set_xlabel(r"Contact radius $r$")
    ax.set_ylabel(r"Operational crossing $\tau_c$")
    ax.text(0.03, 0.06, f"slopes {sir['boot_staged']['slope_med']:.2f} / {sir['boot_fused']['slope_med']:.2f}", transform=ax.transAxes, color=SLATE, fontsize=6.3)
    clean_axes(ax, grid="both")

    ax = axes[1]
    ax.plot(schelling["taus"], schelling["S_cert"], **snapshot_style)
    ax.plot(schelling["taus"], schelling["S_free"], **ordered_style)
    ax.set_title("(b)  Schelling segregation", fontsize=7.7)
    ax.set_xlabel(r"Tolerance $\tau$")
    ax.set_ylabel("Segregation index")
    ax.set_ylim(0.50, 1.01)
    clean_axes(ax, grid="both")

    ax = axes[2]
    ax.plot(coordination["eps"], coordination["consensus_cert"], **snapshot_style)
    ax.plot(coordination["eps"], coordination["consensus_free"], **ordered_style)
    ax.set_title("(c)  Bounded-confidence\nconsensus", fontsize=7.6)
    ax.set_xlabel(r"Confidence bound $\varepsilon$")
    ax.set_ylabel("Largest-cluster fraction")
    ax.set_ylim(0.85, 1.005)
    ax.text(0.02, 0.05, "N=20k / N=2k", transform=ax.transAxes, color=SLATE, fontsize=6.3)
    clean_axes(ax, grid="both")

    handles = [
        Line2D([0], [0], color=BLUE, marker="o", label="Snapshot / staged"),
        Line2D([0], [0], color=AMBER, marker="s", markerfacecolor=WHITE, ls="--", label="In-place ordered"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99))
    save_both(fig, FIGURES, "plot21")


def _ecdf(values):
    x = np.sort(np.asarray(values, float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def render_plot13() -> None:
    """Small-model CPU distribution checks as three ECDF panels."""
    apply_style()
    data = load_json("accuracy_results.json")
    engines = [
        ("amber_vec", "AMBER vectorized"),
        ("amber_loop", "AMBER loop"),
        ("agentpy", "AgentPy"),
        ("mesa", "Mesa"),
        ("mesa_frames", "mesa-frames"),
    ]
    models = [
        ("wealth_gini", "Wealth transfer", "Gini coefficient"),
        ("random_walk_msd", "Random walk", "Mean squared displacement"),
        ("sir_peak", "Well-mixed SIR", "Peak infected fraction"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.40), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.24, top=0.70, wspace=0.28)
    for panel_index, (ax, (model_key, title, x_label)) in enumerate(zip(axes, models)):
        for series_index, (engine_key, engine_label) in enumerate(engines):
            x, y = _ecdf(data["distributions"][model_key][engine_key])
            ax.step(
                x,
                y,
                where="post",
                color=CATEGORICAL[series_index],
                lw=1.9 if engine_key == "amber_vec" else 1.15,
                label=engine_label,
            )
        relevant = [test["p_value"] for test in data["tests"] if test["model"] == model_key]
        ax.text(0.03, 0.96, f"min p={min(relevant):.2f}; n=120", transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color=SLATE)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylim(0, 1.0)
        if panel_index == 0:
            ax.set_ylabel("Empirical CDF")
        panel_label(ax, chr(ord("a") + panel_index))
        clean_axes(ax, grid="both")
    legend_handles = [
        Line2D([0], [0], color=CATEGORICAL[index], lw=1.9 if key == "amber_vec" else 1.15, label=label)
        for index, (key, label) in enumerate(engines)
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.99), columnspacing=0.9)
    save_both(fig, FIGURES, "plot13")


def render_plot14() -> None:
    """Historical public-example SLOC matrix with every observed value shown."""
    apply_style()
    data = load_json("usability_results.json")["per_framework"]
    frameworks = [
        "AMBER (GPU)",
        "AMBER (vectorized)",
        "AMBER (loop)",
        "AgentPy",
        "Mesa",
        "mesa-frames",
        "Melodie",
        "SimPy",
        "FLAME GPU 2",
    ]
    models = [
        ("wealth_transfer", "Wealth"),
        ("random_walk", "Random walk"),
        ("sir_epidemic", "SIR"),
        ("schelling", "Schelling"),
    ]
    matrix = np.full((len(frameworks), len(models)), np.nan)
    for row_index, framework in enumerate(frameworks):
        per_model = data[framework]["per_model"]
        for column_index, (model_key, _) in enumerate(models):
            if model_key in per_model:
                matrix[row_index, column_index] = per_model[model_key]["sloc"]

    cmap = LinearSegmentedColormap.from_list("amber_sloc", [WHITE, BLUE_PALE, BLUE_LIGHT, BLUE_DARK])
    cmap.set_bad(PALE_GREY)
    norm = Normalize(vmin=0, vmax=np.nanmax(matrix))
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.05))
    fig.subplots_adjust(left=0.25, right=0.985, bottom=0.12, top=0.88)
    ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([label for _, label in models])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=5, length=0)
    ax.set_yticks(np.arange(len(frameworks)))
    ax.set_yticklabels(frameworks)
    ax.tick_params(axis="y", length=0)
    for label in ax.get_yticklabels()[:3]:
        label.set_color(BLUE_DARK)
        label.set_fontweight("bold")
    for row_index in range(len(frameworks)):
        for column_index in range(len(models)):
            value = matrix[row_index, column_index]
            if np.isnan(value):
                text = "—"
                color = SLATE
            else:
                text = f"{int(value)}"
                color = WHITE if value >= 75 else INK
            ax.text(column_index, row_index, text, ha="center", va="center", fontsize=7.2, color=color, fontweight="bold" if not np.isnan(value) else "normal")
    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(frameworks), 1), minor=True)
    ax.grid(which="minor", color=WHITE, linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.985, 0.02, "Logical source lines in historical public benchmark examples; — = not implemented", ha="right", va="bottom", fontsize=6.4, color=SLATE)
    save_both(fig, FIGURES, "plot14")


def render_plot15() -> None:
    """Calibration error and wall time at matched evaluation counts."""
    apply_style()
    data = load_json("calibration_gpu.json")
    methods = [
        ("smac", "Batched SMAC", BLUE, "o", "-"),
        ("random", "Random search", AMBER, "s", "--"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.55), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.22, top=0.76, wspace=0.30)
    for method_key, label, color, marker, line_style in methods:
        rows = data["methods"][method_key]
        evaluations = [row["mean_evals"] for row in rows]
        axes[0].errorbar(
            evaluations,
            [row["mean_l2"] for row in rows],
            yerr=[row["std_l2"] for row in rows],
            color=color,
            marker=marker,
            markerfacecolor=color if method_key == "smac" else WHITE,
            ls=line_style,
            capsize=2.5,
            label=label,
        )
        axes[1].errorbar(
            evaluations,
            [row["mean_wall_s"] for row in rows],
            yerr=[row["std_wall_s"] for row in rows],
            color=color,
            marker=marker,
            markerfacecolor=color if method_key == "smac" else WHITE,
            ls=line_style,
            capsize=2.5,
            label=label,
        )

    axes[0].set_title(r"Recovered-parameter $L_2$ error")
    axes[0].set_ylabel(r"Mean $L_2$ error (SD)")
    axes[0].set_yscale("log")
    axes[1].set_title("Wall-clock cost")
    axes[1].set_ylabel("Mean wall time, seconds (SD)")
    axes[1].set_yscale("log")
    for panel_index, ax in enumerate(axes):
        ax.set_xscale("log", base=2)
        ax.set_xticks([96, 192, 384, 768])
        ax.set_xticklabels(["96", "192", "384", "768"])
        ax.set_xlabel("Forward-model evaluations")
        panel_label(ax, chr(ord("a") + panel_index))
        clean_axes(ax, grid="both")
    legend_handles = [
        Line2D([0], [0], color=BLUE, marker="o", label="Batched SMAC"),
        Line2D([0], [0], color=AMBER, marker="s", markerfacecolor=WHITE, ls="--", label="Random search"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99))
    fig.text(0.99, 0.01, "Four trials per point; identical evaluation counts within each pair", ha="right", va="bottom", fontsize=6.3, color=SLATE)
    save_both(fig, FIGURES, "plot15")


def render_plot12() -> None:
    """Current q=0 monitor overhead: paired added cost and check/off ratio."""
    apply_style()
    data = load_json("monitor_cost_current.json")
    rows = data["rows"]
    styles = [(1, "1 column", BLUE, "o", "-"), (8, "8 columns", AMBER, "s", "--")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.45))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.22, top=0.75, wspace=0.30)

    for columns, label, color, marker, line_style in styles:
        selected = sorted((row for row in rows if row["columns"] == columns), key=lambda row: row["n"])
        populations = np.asarray([row["n"] for row in selected], float)
        centers, lower, upper = [], [], []
        for row in selected:
            paired = 1000.0 / row["steps"] * (np.asarray(row["check"]["raw_seconds"]) - np.asarray(row["off"]["raw_seconds"]))
            center = float(np.median(paired))
            q1, q3 = np.quantile(paired, [0.25, 0.75])
            centers.append(center)
            lower.append(center - q1)
            upper.append(q3 - center)
        axes[0].errorbar(
            populations,
            centers,
            yerr=[lower, upper],
            color=color,
            marker=marker,
            markerfacecolor=color if columns == 1 else WHITE,
            ls=line_style,
            capsize=2.5,
            label=label,
        )
        axes[1].plot(
            populations,
            [row["mean_ratio"] for row in selected],
            color=color,
            marker=marker,
            markerfacecolor=color if columns == 1 else WHITE,
            ls=line_style,
            label=label,
        )

    axes[0].set_title("Paired added cost")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Median added ms / step (IQR)")
    axes[1].set_title("Relative wall time")
    axes[1].set_xscale("log")
    axes[1].set_ylabel("Mean check / off ratio")
    axes[1].axhline(1.0, color=SLATE, ls=":", lw=0.8)
    axes[1].set_ylim(0, 20)
    for panel_index, ax in enumerate(axes):
        ax.set_xlabel("Agents N")
        panel_label(ax, chr(ord("a") + panel_index))
        clean_axes(ax, grid="both")
    legend_handles = [
        Line2D([0], [0], color=BLUE, marker="o", label="1 column"),
        Line2D([0], [0], color=AMBER, marker="s", markerfacecolor=WHITE, ls="--", label="8 columns"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99))
    save_both(fig, FIGURES, "plot12")


RENDERERS = [
    render_plot01,
    render_plot02,
    render_plot03,
    render_plot07,
    render_plot11,
    render_plot17,
    render_plot18,
    render_plot21,
    render_plot13,
    render_plot14,
    render_plot15,
    render_plot12,
]


def main() -> None:
    for renderer in RENDERERS:
        renderer()
        print(renderer.__name__.removeprefix("render_"))


if __name__ == "__main__":
    main()
