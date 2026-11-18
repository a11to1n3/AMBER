"""Plot the recorded automatic batching benchmark for AMBER 0.6.0."""

import argparse
from io import StringIO
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, StrMethodFormatter


RESULTS = Path(__file__).resolve().parent / "results"
DATA = RESULTS / "automatic_batching_0_6_0.json"
LABELS = {"aging": "Integer aging", "walk": "Keyed random walk"}
LANES = (
    ("scalar_cpu", "Scalar CPU", "#666666", "o"),
    ("default_cpu", "Default CPU", "#0072B2", "s"),
    ("gpu", "RTX 5090", "#D55E00", "D"),
)


def plot(report):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "svg.hashsalt": "ambr-automatic-batching-0.6.0",
    })
    figure, axis = plt.subplots(figsize=(6.75, 3.2), layout="constrained")
    cases = report["results"]
    rows = np.arange(len(cases))
    for offset, (key, label, color, marker) in zip((-0.22, 0, 0.22), LANES):
        samples = [case["statistics"][key] for case in cases]
        median = np.array([sample["median_s"] for sample in samples])
        xerr = [median - [sample["min_s"] for sample in samples],
                [sample["max_s"] for sample in samples] - median]
        axis.errorbar(median, rows + offset, xerr=xerr, fmt=marker, markersize=4.5,
                      color=color, capsize=2.5, linewidth=1.2, label=label)

    axis.set_yticks(rows, [f"{LABELS[case['workload']]} ({case['n']:,})" for case in cases])
    axis.set_ylim(len(cases) - 0.48, -0.52)
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(FixedLocator([0.03, 0.1, 0.3, 1, 3, 6]))
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("End-to-end time (s, log scale)")
    axis.tick_params(axis="y", length=0, pad=6)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    for boundary in rows[:-1] + 0.5:
        axis.axhline(boundary, color="#EBEBEB", linewidth=0.6, zorder=0)
    figure.legend(loc="outside lower center", ncol=3, frameon=False)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA.with_suffix(".svg"))
    args = parser.parse_args()
    report = json.loads(DATA.read_text())
    figure = plot(report)
    with StringIO() as svg:
        figure.savefig(svg, format="svg", metadata={"Date": report["started_utc"]})
        args.output.write_text("\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n")
    plt.close(figure)


if __name__ == "__main__":
    main()
