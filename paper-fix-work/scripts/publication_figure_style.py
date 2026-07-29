"""Publication figure system for the AMBER AAMAS manuscript.

The palette uses one blue root, one amber comparator root, and neutrals.  All
dimensions and type sizes are set for the final ACM two-column page rather than
for a notebook or screen preview.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# Editorial palette: a deep Atlantic blue, warm copper comparator, and quiet
# mineral neutrals.  Additional roots are used only when category identity is
# the analytical point (the all-framework landscape and five-way ECDF).
INK = "#17232E"
CHARCOAL = "#43505C"
SLATE = "#71808E"
MID_GREY = "#AAB5BE"
LIGHT_GREY = "#DFE6EB"
PALE_GREY = "#F5F7F8"
BLUE = "#1F6387"
BLUE_DARK = "#123F59"
BLUE_MID = "#4F8EAC"
BLUE_LIGHT = "#A7C7D8"
BLUE_PALE = "#ECF4F7"
AMBER = "#C9752E"
AMBER_DARK = "#914817"
AMBER_LIGHT = "#EDB47D"
AMBER_PALE = "#FCF0E5"
TEAL = "#397A70"
TEAL_LIGHT = "#9AC2BB"
PLUM = "#75608E"
PLUM_LIGHT = "#B9ACC8"
WHITE = "#FFFFFF"

# Additional roots only for the five-framework ECDF where identity is the point.
CATEGORICAL = [BLUE, AMBER, TEAL, PLUM, "#68737C"]

FULL_WIDTH = 7.15
COLUMN_WIDTH = 3.42
DPI = 240


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.facecolor": WHITE,
            "axes.facecolor": "#FBFCFD",
            "font.family": ["Avenir Next", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.3,
            "axes.titleweight": "bold",
            "axes.titlepad": 7.0,
            "axes.labelcolor": INK,
            "axes.edgecolor": CHARCOAL,
            "axes.linewidth": 0.8,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.color": "#E2E8EC",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.9,
            "lines.linewidth": 1.75,
            "lines.markersize": 4.5,
            "legend.frameon": False,
            "legend.handlelength": 1.7,
            "legend.columnspacing": 1.1,
            "legend.handletextpad": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axes(ax, *, grid: str | None = "y", keep_top_right: bool = False) -> None:
    if not keep_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if grid == "x":
        ax.grid(axis="x")
    elif grid == "y":
        ax.grid(axis="y")
    elif grid == "both":
        ax.grid(axis="both")
    else:
        ax.grid(False)
    ax.set_axisbelow(True)


def panel_label(ax, label: str, *, x: float = 0.0, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=INK,
        fontsize=8.8,
        fontweight="bold",
        clip_on=False,
    )


def save_both(fig, output_dir: Path, stem: str, *, pad: float = 0.02) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "bbox_inches": "tight",
        "pad_inches": pad,
        "facecolor": WHITE,
        "metadata": {"Creator": "AMBER deterministic figure renderer"},
    }
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    pdf_temp = output_dir / f".{stem}.pdf.tmp"
    png_temp = output_dir / f".{stem}.png.tmp"
    fig.savefig(pdf_temp, format="pdf", **common)
    fig.savefig(png_temp, format="png", dpi=DPI, **common)
    pdf_temp.replace(pdf_path)
    png_temp.replace(png_path)
    plt.close(fig)


def compact_seconds(value: float) -> str:
    if value < 1e-3:
        return f"{value * 1e6:.0f} µs"
    if value < 1:
        return f"{value * 1e3:.1f} ms"
    if value < 10:
        return f"{value:.2f} s"
    return f"{value:.1f} s"
