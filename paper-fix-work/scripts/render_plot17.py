#!/usr/bin/env python3
"""Render the mesa-frames local-update-granularity sensitivity figure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from amber_figure_style import (  # noqa: E402
    FULL_W,
    apply_figure_style,
    panel_label,
    save_figure,
    style_axes,
)

INPUT = ROOT / "artifacts" / "mf_granularity.json"
OUTPUT = ROOT / "figs" / "plot17.png"
PURPLES = ["#c8c8df", "#aaa8cf", "#8d87bd", "#7563ad", "#542498"]


def main() -> None:
    apply_figure_style()
    data = json.loads(INPUT.read_text())
    taus = np.asarray(data["taus"], dtype=float)
    blocks = np.asarray(data["nblocks"], dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.45), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.22, wspace=0.28)

    ax = axes[0]
    for index, (nblocks, color) in enumerate(zip(blocks, PURPLES)):
        row = data["by_nblocks"][str(nblocks)]
        label = (
            "1 local block (snapshot-like)"
            if nblocks == 1
            else f"{nblocks} local blocks"
        )
        ax.plot(
            taus,
            row["curve"],
            color=color,
            linewidth=1.7 if index < len(blocks) - 1 else 2.2,
            label=label,
        )
        ax.axvline(row["tc"], color=color, linestyle=":", linewidth=1.0, alpha=0.75)
    ax.axhline(0.5, color="#4b5563", linestyle="--", linewidth=0.9)
    ax.set_xlabel(r"Transmissibility $\tau$")
    ax.set_ylabel("Final attack rate")
    ax.set_xlim(float(taus.min()), float(taus.max()))
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=6.8, handlelength=2.2)
    style_axes(ax)
    panel_label(ax, "a", x=-0.15, y=1.07)

    ax = axes[1]
    crossings = np.asarray(
        [data["by_nblocks"][str(nblocks)]["tc"] for nblocks in blocks]
    )
    baseline = data["by_nblocks"][str(blocks[0])]
    low, high = baseline["tc_ci"]
    ax.axhspan(low, high, color="#6b7280", alpha=0.12, label="One-block 95% CI")
    ax.axhline(
        baseline["tc"], color="#1f2937", linestyle="--", linewidth=1.0,
        label="One-block estimate",
    )
    ax.plot(blocks, crossings, color="#6a3d9a", marker="o", linewidth=2.0)
    ax.set_xscale("log")
    ax.set_xlabel("Local update blocks per step")
    ax.set_ylabel(r"Operational crossing $\tau_c$")
    ax.set_xticks(blocks)
    ax.set_xticklabels([str(value) for value in blocks])
    ax.legend(loc="lower left", fontsize=7.0)
    style_axes(ax)
    panel_label(ax, "b", x=-0.15, y=1.07)

    save_figure(fig, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
