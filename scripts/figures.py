"""Draw the CAAT research figures from saved CSV tables."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from common import ARCHIVES, ROOT, read_csv


COLORS = ("#0072b2", "#b54e00", "#00835f", "#aa4b86", "#7458ad", "#83672d", "#4d6979")
STYLES = ("-", "--", ":", "-.")
MARKERS = ("o", "s", "^", "D", "P")
GRADER_NAMES = {
    "strict_pass": "Google strict",
    "helm_strict_pass": "HELM full pass",
    "correct": "HELM",
    "tiger_correct": "TIGER-Lab",
    "gpt": "GPT",
    "llama": "Llama",
    "claude": "Claude",
}


def setup_style():
    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
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


def export(figure, destination):
    svg = destination.with_suffix(".svg")
    figure.savefig(svg, metadata={"Date": None, "Creator": "CAAT"})
    lines = svg.read_text(encoding="utf-8").splitlines()
    svg.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    figure.savefig(
        destination.with_suffix(".pdf"),
        metadata={"CreationDate": None, "Creator": "CAAT"},
    )
    plt.close(figure)
    print(f"Wrote {destination.name}.svg and .pdf")


def risk_coverage(rows, output):
    for archive, (_, _, graders) in ARCHIVES.items():
        subset = [row for row in rows if row["benchmark"] == archive]
        methods = list(dict.fromkeys(row["method"] for row in subset))
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.5), sharey=True)
        fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.30, wspace=0.17)
        for ax, grader in zip(axes, graders, strict=True):
            for i, method in enumerate(methods):
                points = [
                    row
                    for row in subset
                    if row["method"] == method and row["label_source"] == grader
                ]
                ax.plot(
                    [float(row["coverage"]) for row in points],
                    [float(row["error_rate"]) for row in points],
                    color=COLORS[i % len(COLORS)],
                    linestyle=STYLES[(i // len(COLORS)) % len(STYLES)],
                    marker=MARKERS[i % len(MARKERS)],
                    markersize=3.3,
                    linewidth=1.5,
                    label=method.replace("_", " "),
                )
            ax.set_title(GRADER_NAMES[grader])
            ax.set_xlabel("Coverage")
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis="y", color="#e0e8ec", linewidth=0.8)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Accepted error rate")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            frameon=False,
            fontsize=8,
            ncols=3,
        )
        export(fig, output / f"risk-coverage-{archive.replace('_', '-')}")


def short_model(model):
    return model.split("_", 1)[-1]


def wildbench_shifts(rows, output):
    pairs = list(
        dict.fromkeys((row["first_judge"], row["second_judge"]) for row in rows)
    )
    models = sorted({row["model"] for row in rows}, reverse=True)
    fig, axes = plt.subplots(
        1, len(pairs), figsize=(11.7, 5.8), sharey=True, sharex=True
    )
    fig.subplots_adjust(left=0.28, right=0.98, top=0.89, bottom=0.13, wspace=0.18)
    for index, (ax, pair) in enumerate(zip(axes, pairs, strict=True)):
        ax.axvline(0, color="#67818f", linewidth=1, linestyle="--")
        cells = {
            row["model"]: row
            for row in rows
            if (row["first_judge"], row["second_judge"]) == pair
        }
        for y, model in enumerate(models):
            row = cells[model]
            lower, point, upper = (
                100 * float(row[key])
                for key in ("lower_95", "excess_shift", "upper_95")
            )
            ax.hlines(y, lower, upper, linewidth=1.3, color=COLORS[index])
            ax.scatter(
                point, y, s=19, marker=MARKERS[index], color=COLORS[index], zorder=2
            )
        ax.set_title(f"{GRADER_NAMES[pair[0]]} to {GRADER_NAMES[pair[1]]}")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks([-5, 0, 5, 10, 25, 50], ["−5", "0", "5", "10", "25", "50"])
        ax.grid(axis="y", color="#e7edef", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.yaxis.set_tick_params(length=0)
    axes[0].set_yticks(
        range(len(models)), [short_model(model) for model in models], fontsize=8
    )
    fig.supxlabel("Excess shift in percentage points", x=0.63, y=0.03)
    export(fig, output / "wildbench-grader-shifts")


def wildbench_auroc(rows, output):
    data = [
        row
        for row in rows
        if row["scenario"] == "wildbench" and row["outcome_cutoff"] == "7"
    ]
    judges = list(dict.fromkeys(row["judge"] for row in data))
    scores = list(dict.fromkeys(row["score"] for row in data))
    models = sorted({row["model"] for row in data}, reverse=True)
    fig, axes = plt.subplots(
        1, len(judges), figsize=(12.5, 6.4), sharey=True, sharex=True
    )
    fig.subplots_adjust(left=0.26, right=0.98, top=0.91, bottom=0.21, wspace=0.16)
    for ax, judge in zip(axes, judges, strict=True):
        ax.axvline(0.5, color="#67818f", linewidth=1, linestyle="--")
        pairs = {
            (row["model"], row["score"]): row["auroc"]
            for row in data
            if row["judge"] == judge
        }
        for index, score in enumerate(scores):
            available = [
                (y + (index - 2) * 0.11, float(pairs[model, score]))
                for y, model in enumerate(models)
                if pairs[model, score]
            ]
            y, x = zip(*available)
            ax.scatter(
                x,
                y,
                label=score.replace("_", " "),
                marker=MARKERS[index],
                s=23,
                color=COLORS[index],
                zorder=2,
            )
        ax.set_title(GRADER_NAMES[judge])
        ax.set_xlabel("AUROC")
        ax.set_xlim(-0.025, 1.025)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.grid(axis="y", color="#e7edef", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.yaxis.set_tick_params(length=0)
    axes[0].set_yticks(
        range(len(models)), [short_model(model) for model in models], fontsize=8
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.58, 0.035),
        frameon=False,
        ncols=3,
    )
    export(fig, output / "wildbench-auroc")


def calibration_replay(rows, output):
    examples = (
        ("fever", "qwen_self_report", "FEVER · Qwen self-report"),
        ("jigsaw", "deberta_probability_of_qwen_label", "Jigsaw · DeBERTa"),
        (
            "omni_math",
            "geometric_mean_token_probability",
            "Omni-MATH · Token probability",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.7), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.24, top=0.86, wspace=0.21)
    for axis, (archive, score, title) in zip(axes, examples, strict=True):
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#67818f")
        for index, method in enumerate(("raw", "platt")):
            bins = [
                row
                for row in rows
                if row["archive"] == archive
                and row["score"] == score
                and row["method"] == method
                and int(row["count"]) > 0
            ]
            axis.scatter(
                [float(row["mean_probability"]) for row in bins],
                [float(row["accuracy"]) for row in bins],
                s=[18 + 0.09 * int(row["count"]) for row in bins],
                label={"raw": "Raw score", "platt": "Platt"}[method],
                color=COLORS[index],
                marker=MARKERS[index],
                alpha=0.72,
            )
        axis.set(
            title=title,
            xlim=(-0.04, 1.04),
            ylim=(-0.04, 1.04),
            xticks=(0, 0.5, 1),
            yticks=(0, 0.5, 1),
        )
        axis.set_aspect("equal")
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Observed correctness")
    figure.supxlabel("Mean predicted probability", y=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=2, frameon=False)
    export(figure, output / "calibration-replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=ROOT / "replay")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "figures")
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Plot the bundled calibration table only.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    calibration_replay(
        read_csv(ROOT / "results/calibration/reliability.csv"), args.output_dir
    )
    if args.calibration_only:
        return
    risk_coverage(read_csv(ROOT / "results/module_study/ranking.csv"), args.output_dir)
    records = args.replay_dir / "judge_study"
    wildbench_shifts(read_csv(records / "gate_cells.csv"), args.output_dir)
    wildbench_auroc(read_csv(records / "auroc.csv"), args.output_dir)


if __name__ == "__main__":
    main()
