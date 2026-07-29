"""Unified matplotlib style for AMBER/AAMAS paper figures."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Semantic palette (color-blind friendly, print-safe)
SEQ = "#1a1a1a"
CERT = "#c0392b"
FREE = "#6a3d9a"
SAFE = "#2c7d3f"
COND = "#b8860b"
MUTED = "#6b7280"
GRID = "#d1d5db"
BG = "#ffffff"

FRAMEWORK_COLORS = {
    "AMBER (GPU)": "#b91c1c",
    "AMBER (vectorized)": "#2563eb",
    "AMBER (loop)": "#60a5fa",
    "FLAME GPU 2": "#059669",
    "mesa-frames": "#7c3aed",
    "Agents.jl": "#16a34a",
    "SimPy": "#a855f7",
    "Melodie": "#f97316",
    "AgentPy": "#ef4444",
    "Mesa": "#78716c",
}

DPI = 200
FULL_W = 7.5   # \textwidth in two-column sigconf (~7.5 in)
COL_W = 3.35
PANEL_H = 2.6


def apply_figure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#374151",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#e5e7eb",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "svg.fonttype": "none",
        }
    )


def style_axes(ax, *, hide_top_right: bool = True) -> None:
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        color="#111827",
    )


def save_figure(fig, path, *, facecolor: str = BG) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=facecolor)
    plt.close(fig)