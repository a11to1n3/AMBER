#!/usr/bin/env python3
"""Regenerate plot07, plot18, plot12 with clearer, focused layouts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FIGS = SCRIPT_DIR.parent / "figs"
DATA = SCRIPT_DIR.parent / "artifacts"

sys.path.insert(0, str(SCRIPT_DIR))
from amber_figure_style import (  # noqa: E402
    CERT,
    FRAMEWORK_COLORS,
    FREE,
    FULL_W,
    MUTED,
    SEQ,
    apply_figure_style,
    panel_label,
    save_figure,
    style_axes,
)

# ---------------------------------------------------------------------------
# Figure 6 — scaling (plot07): heatmap matrix, all ten frameworks × five scales
# ---------------------------------------------------------------------------

MODELS = [
    ("wealth_transfer", "Wealth transfer"),
    ("random_walk", "Random walk"),
    ("sir_epidemic", "SIR epidemic"),
    ("schelling", "Schelling"),
]

N_TICKS = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
BUDGET_S = 120.0

FRAMEWORKS = [
    "AMBER (GPU)",
    "FLAME GPU 2",
    "AMBER (vectorized)",
    "mesa-frames",
    "AMBER (loop)",
    "Agents.jl",
    "AgentPy",
    "Mesa",
    "Melodie",
    "SimPy",
]

def _compact_time(seconds: float) -> str:
    if seconds >= BUDGET_S:
        return "T/O"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 10:
        return f"{seconds:.1f}s"
    return f"{seconds:.0f}s"


def _label_color(seconds: float, cmap, norm) -> str:
    rgba = cmap(norm(min(seconds, BUDGET_S)))
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if lum < 0.52 else "#1f2937"


def _draw_scaling_heatmap(ax, model_key: str, grid: dict, cmap, norm) -> None:
    n_rows = len(FRAMEWORKS)
    n_cols = len(N_TICKS)
    x_labels = [f"{int(n / 1e6)}M" if n >= 1e6 else f"{int(n / 1e3)}k" for n in N_TICKS]

    matrix = np.full((n_rows, n_cols), np.nan)
    raw = [[None] * n_cols for _ in range(n_rows)]
    for i, fw in enumerate(FRAMEWORKS):
        for j, n in enumerate(N_TICKS):
            t = grid.get((fw, model_key, n))
            if t is None or t <= 0:
                continue
            raw[i][j] = t
            matrix[i, j] = min(t, BUDGET_S)

    ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        cmap=cmap,
        norm=norm,
        origin="upper",
        interpolation="nearest",
    )

    for i, fw in enumerate(FRAMEWORKS):
        for j in range(n_cols):
            t = raw[i][j]
            if t is None:
                continue
            ax.text(
                j,
                i,
                _compact_time(t),
                ha="center",
                va="center",
                fontsize=6.8,
                color=_label_color(t, cmap, norm),
                fontweight="bold" if fw == "AMBER (GPU)" else "normal",
                zorder=4,
            )

    for y in (1.5, 4.5):
        ax.axhline(y - 0.5, color="#e2e8f0", lw=0.8, zorder=2)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, fontsize=7.5)
    ax.set_yticks(range(n_rows))
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5, zorder=3)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_plot07() -> None:
    from matplotlib import colors as mcolors

    apply_figure_style()
    data = json.loads((DATA / "benchmark_results_all5090.json").read_text())
    grid = {
        (r["framework"], r["model"], r["n_agents"]): r["execution_time"]
        for r in data["results"]
    }

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "amber_scale",
        ["#f0fdf4", "#86efac", "#fcd34d", "#f97316", "#991b1b"],
    )
    cmap.set_bad("#f1f5f9")
    norm = mcolors.LogNorm(vmin=0.004, vmax=BUDGET_S)

    # 2×2 heatmaps with aspect="auto" (never set_aspect — that collapses panels).
    fig = plt.figure(figsize=(FULL_W, 5.45))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.04],
        height_ratios=[1.0, 1.0],
        left=0.21,
        right=0.93,
        top=0.92,
        bottom=0.11,
        wspace=0.16,
        hspace=0.18,
    )
    panel_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    cax = fig.add_subplot(gs[:, 2])

    for ax, (model_key, title), panel in zip(panel_axes, MODELS, "abcd"):
        ax.set_title(f"({panel}) {title}", fontsize=8, loc="left", color="#111827", pad=2)
        _draw_scaling_heatmap(ax, model_key, grid, cmap, norm)
        if ax in (panel_axes[0], panel_axes[2]):
            ax.set_yticklabels(FRAMEWORKS, fontsize=6.8)
            for tick, fw in zip(ax.get_yticklabels(), FRAMEWORKS):
                if fw == "AMBER (GPU)":
                    tick.set_fontweight("bold")
                    tick.set_color(CERT)
        else:
            ax.set_yticklabels([])
        if ax in (panel_axes[2], panel_axes[3]):
            ax.set_xlabel("Agents", fontsize=8)

    panel_axes[0].set_ylabel("Framework", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Wall time (s)", fontsize=8)
    cbar.ax.set_yticks([0.01, 0.1, 1, 10, BUDGET_S])
    cbar.ax.set_yticklabels(["10ms", "100ms", "1s", "10s", "120s"])
    cbar.outline.set_linewidth(0.6)

    fig.savefig(FIGS / "plot07.png", dpi=200, facecolor="#ffffff")
    plt.close(fig)
    print("plot07")


# ---------------------------------------------------------------------------
# Figure 9 — crossing sensitivity (plot18): bar chart of tau_c
# ---------------------------------------------------------------------------

def render_plot18() -> None:
    from matplotlib.lines import Line2D

    apply_figure_style()
    d = json.loads((SCRIPT_DIR.parent / "emergence_threshold_controlled.json").read_text())
    levels = ["0.3", "0.5", "0.7"]
    paths = [
        ("Row-wise snapshot", "rowwise_snapshot", SEQ, "o"),
        ("Batched GPU snapshot", "batched_gpu_snapshot", CERT, "s"),
        ("In-place ordered", "inplace_ordered", FREE, "^"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.4), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.78, bottom=0.20, wspace=0.38)

    y_pos = np.arange(len(paths))

    for ax, lv, panel in zip(axes, levels, "abc"):
        panel_label(ax, panel, x=-0.28, y=1.08)
        xs, xerr_lo, xerr_hi = [], [], []
        for row, (_name, key, color, marker) in enumerate(paths):
            entry = d["crossing_sensitivity"][lv][key]
            tc, lo, hi = entry["median"], entry["lo"], entry["hi"]
            xs.append(tc)
            xerr_lo.append(tc - lo)
            xerr_hi.append(hi - lo)
            ax.errorbar(
                tc,
                row,
                xerr=[[tc - lo], [hi - tc]],
                fmt=marker,
                color=color,
                ms=7.5,
                mew=0.9,
                mec="white",
                capsize=3.5,
                elinewidth=1.2,
                zorder=4,
            )
            ax.text(
                hi + 0.004,
                row,
                f"{tc:.3f}",
                va="center",
                ha="left",
                fontsize=7.5,
                color="#111827",
            )

        pad = max(max(xerr_hi), 0.012)
        ax.set_xlim(min(xs) - pad, max(xs) + pad + 0.045)
        ax.set_xlabel(rf"$\tau_c$ (attack $={lv}$)")
        ax.set_yticks(y_pos)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.32)
        style_axes(ax)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticklabels([name for name, *_ in paths], fontsize=8)
    for ax in axes[1:]:
        ax.set_yticklabels([])

    handles = [
        Line2D([0], [0], marker=marker, color=color, linestyle="", markersize=7, label=name)
        for name, _key, color, marker in paths
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.52, 0.98),
        columnspacing=1.2,
    )
    save_figure(fig, FIGS / "plot18.png")
    print("plot18")


# ---------------------------------------------------------------------------
# Figure 12 — current q=0 monitor-cost microbenchmark
# ---------------------------------------------------------------------------

def render_plot12() -> None:
    apply_figure_style()
    d = json.loads((SCRIPT_DIR.parent / "monitor_cost_current.json").read_text())
    rows = d["rows"]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.85))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.22, wspace=0.28)

    styles = [(1, CERT, "o"), (8, FREE, "s")]

    # (a) absolute added cost
    ax = axes[0]
    panel_label(ax, "a", x=-0.14, y=1.08)
    for column_count, color, marker in styles:
        selected = sorted(
            (r for r in rows if r["columns"] == column_count),
            key=lambda r: r["n"],
        )
        ns = np.array([r["n"] for r in selected], float)
        centers, lower, upper = [], [], []
        for row in selected:
            paired = 1000.0 / row["steps"] * (
                np.array(row["check"]["raw_seconds"])
                - np.array(row["off"]["raw_seconds"])
            )
            center = float(np.median(paired))
            q1, q3 = np.quantile(paired, [0.25, 0.75])
            centers.append(center)
            lower.append(center - q1)
            upper.append(q3 - center)
        ax.errorbar(
            ns, centers, yerr=[lower, upper], color=color, marker=marker,
            lw=1.8, capsize=3, label=f"{column_count} column" + ("" if column_count == 1 else "s"),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Agents $N$")
    ax.set_ylabel("Median added ms/step (IQR)")
    ax.legend(loc="upper left", fontsize=7.5, frameon=True)
    style_axes(ax)

    # (b) relative cost against each workload's off time
    ax = axes[1]
    panel_label(ax, "b", x=-0.14, y=1.08)
    for column_count, color, marker in styles:
        selected = sorted(
            (r for r in rows if r["columns"] == column_count),
            key=lambda r: r["n"],
        )
        ax.plot(
            [r["n"] for r in selected],
            [r["mean_ratio"] for r in selected],
            color=color, marker=marker, lw=1.8,
            label=f"{column_count} column" + ("" if column_count == 1 else "s"),
        )
    ax.axhline(1.0, color=MUTED, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Agents $N$")
    ax.set_ylabel("Mean wall-time ratio (check/off)")
    ax.set_ylim(0, 20)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True)
    style_axes(ax)

    save_figure(fig, FIGS / "plot12.png")
    print("plot12")


# ---------------------------------------------------------------------------
# Figure 3 — epidemic threshold (plot03): single annotated S-curve panel
# ---------------------------------------------------------------------------

def render_plot03() -> None:
    apply_figure_style()
    d = json.loads((SCRIPT_DIR.parent / "emergence_threshold_controlled.json").read_text())
    taus = np.array(d["protocol"]["taus"], float)
    c = d["curves"]
    tc = d["tau_c"]

    series = [
        ("rowwise_snapshot_mean", "rowwise_snapshot_sd", SEQ, "o", "Row-wise snapshot", tc["rowwise_snapshot"]),
        ("batched_gpu_snapshot_mean", "batched_gpu_snapshot_sd", CERT, "s", "Batched GPU snapshot", tc["batched_gpu_snapshot"]),
        ("inplace_ordered_mean", "inplace_ordered_sd", FREE, "^", "In-place ordered", tc["inplace_ordered"]),
    ]

    fig, ax = plt.subplots(figsize=(FULL_W, 3.0))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.14)

    for mean_k, sd_k, color, marker, label, _ in series:
        y = np.array(c[mean_k], float)
        sd = np.array(c[sd_k], float)
        ax.plot(
            taus,
            y,
            color=color,
            marker=marker,
            markevery=2,
            label=label,
            zorder=4,
            clip_on=False,
        )
        ax.fill_between(taus, y - sd, y + sd, color=color, alpha=0.10, linewidth=0, zorder=2)

    ax.axhline(0.5, color=MUTED, ls=":", lw=1.1, zorder=1)
    ax.text(
        taus[-1] + 0.004,
        0.5,
        "attack = 0.5",
        fontsize=7.5,
        color=MUTED,
        va="center",
        ha="left",
    )

    seq = tc["rowwise_snapshot"]
    seq_m = seq["median"]
    ax.axvspan(seq["lo"], seq["hi"], color=MUTED, alpha=0.10, zorder=0)
    ax.text(
        0.02,
        0.97,
        "row-wise 95% bootstrap interval",
        transform=ax.transAxes,
        fontsize=7.5,
        color=MUTED,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e5e7eb", lw=0.7),
    )

    fused_m = tc["inplace_ordered"]["median"]
    ax.axvline(fused_m, color=FREE, ls="--", lw=1.2, alpha=0.85, zorder=3)
    ax.axvline(seq_m, color=SEQ, ls="--", lw=1.2, alpha=0.85, zorder=3)
    ax.text(
        fused_m,
        0.91,
        rf"$\tau_c={fused_m:.3f}$",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=FREE,
        fontweight="bold",
    )
    ax.text(
        seq_m,
        0.985,
        rf"$\tau_c={seq_m:.3f}$",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=SEQ,
        fontweight="bold",
    )
    gpu_tc = tc["batched_gpu_snapshot"]
    if max(seq["lo"], gpu_tc["lo"]) <= min(seq["hi"], gpu_tc["hi"]):
        ax.text(
            0.98,
            0.88,
            "snapshot intervals overlap",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color=CERT,
            linespacing=1.15,
        )

    shift_pct = 100 * (seq_m - fused_m) / seq_m
    ax.annotate(
        "",
        xy=(fused_m, 0.5),
        xytext=(seq_m, 0.5),
        arrowprops=dict(arrowstyle="<->", color="#374151", lw=1.5, shrinkA=0, shrinkB=0),
    )
    ax.text(
        (fused_m + seq_m) / 2,
        0.535,
        rf"{shift_pct:.0f}\% lower $\tau$",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#111827",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
    )

    ax.set_xlabel(r"Per-contact transmissibility $\tau$")
    ax.set_ylabel("Final attack rate")
    ax.set_xlim(taus[0] - 0.005, taus[-1] + 0.03)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=True, fontsize=8, handlelength=1.6)
    style_axes(ax)
    save_figure(fig, FIGS / "plot03.png")
    print("plot03")


def main() -> None:
    render_plot03()
    render_plot07()
    render_plot18()
    render_plot12()


if __name__ == "__main__":
    main()
