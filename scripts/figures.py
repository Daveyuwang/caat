"""Plot the possible final error counts for the WinoGrande audit."""

import argparse
from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt

from common import ROOT, read_csv, read_json


def setup_style():
    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#263c49",
            "axes.edgecolor": "#c5d1d8",
            "text.color": "#142b39",
            "xtick.color": "#415964",
            "ytick.color": "#415964",
            "svg.fonttype": "none",
            "svg.hashsalt": "caat-research",
        }
    )


def audit_stopping(output):
    archive = ROOT / "results/winogrande"
    plan = read_json(archive / "plan.json")
    full = read_json(archive / "full-result.json")
    cap, boundary = plan["accepted"], plan["min_errors_to_exceed"]
    figure, axis = plt.subplots(figsize=(8, 3.8), layout="constrained")
    graded, errors = 0, 0
    for _, block in groupby(
        read_csv(archive / "receipts.csv"), key=lambda row: row["block"]
    ):
        for row in block:
            graded += 1
            errors += int(row["correct"]) == 0
        upper = errors + cap - graded
        color = "#126681" if errors >= boundary else "#788b96"
        axis.vlines(graded, errors, upper, linewidth=2.2, color=color)
        axis.scatter([graded, graded], [errors, upper], marker="_", s=76, color=color)
    axis.axhline(boundary, color="#a24b1e", linestyle="--", linewidth=1)
    axis.annotate(
        f"{boundary} errors settle ‘exceeds’",
        (1, boundary),
        xytext=(0, -12),
        textcoords="offset points",
        color="#843d18",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    axis.annotate(
        f"After {graded} labels\n{errors} to {upper} possible errors",
        (graded, upper),
        xytext=(-4, 32),
        textcoords="offset points",
        ha="right",
        arrowprops={"arrowstyle": "-", "color": "#126681"},
    )
    axis.scatter(cap, full["errors"], color="#162f40", marker="D", s=30)
    axis.annotate(
        f"{full['errors']} at full draw",
        (cap, full["errors"]),
        xytext=(0, -24),
        textcoords="offset points",
        ha="right",
    )
    axis.set_title("WinoGrande fixed-budget audit", loc="left", weight="bold", pad=10)
    axis.set(
        xlabel="Labels graded",
        ylabel=f"Possible errors in the {cap}-label draw",
        xlim=(0, cap + 3),
        ylim=(0, cap),
        xticks=range(0, cap + 1, 10),
    )
    axis.grid(axis="y", linewidth=0.6, color="#e1e8eb")
    axis.set_axisbelow(True)
    figure.savefig(
        output / "winogrande-audit.svg", metadata={"Date": None, "Creator": "CAAT"}
    )
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    setup_style()
    audit_stopping(output)
