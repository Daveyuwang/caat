"""Draw the CAAT research figures from saved CSV tables."""

import argparse
from itertools import groupby
from math import isnan
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

from common import ARCHIVES, ROOT, read_csv, read_json


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
            "pdf.fonttype": 42,
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


def score_matrix(axis, values, row_names, column_names, palette, fmt):
    # Fixed limits keep a given shade comparable across graders and archives.
    axis.imshow(values, cmap=palette, vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(column_names)), column_names)
    axis.set_yticks(range(len(row_names)), row_names)
    axis.tick_params(length=0, pad=5)
    axis.spines[:].set_visible(False)
    for y, row in enumerate(values):
        for x, value in enumerate(row):
            text = "n/a" if isnan(value) else format(value, fmt)
            axis.text(x, y, text, ha="center", va="center", fontsize=8)
    for position in range(len(row_names) - 1):
        axis.axhline(position + 0.5, color="white", linewidth=1)
    for position in range(len(column_names) - 1):
        axis.axvline(position + 0.5, color="white", linewidth=1)


def risk_coverage(rows, output):
    palette = LinearSegmentedColormap.from_list(
        "accepted_error", ["#ffffff", "#e5b690"]
    )
    for archive, (_, _, graders) in ARCHIVES.items():
        subset = [row for row in rows if row["benchmark"] == archive]
        methods = list(dict.fromkeys(row["method"] for row in subset))
        budgets = sorted({float(row["target_coverage"]) for row in subset})
        fig, axes = plt.subplots(
            1, 2, figsize=(9.1, 1.8 + len(methods) * 0.34), sharey=True
        )
        fig.subplots_adjust(left=0.26, right=0.98, top=0.87, bottom=0.21, wspace=0.1)
        for ax, grader in zip(axes, graders, strict=True):
            risks = {
                (row["method"], float(row["target_coverage"])): float(row["error_rate"])
                for row in subset
                if row["label_source"] == grader
            }
            score_matrix(
                ax,
                [[risks[method, budget] for budget in budgets] for method in methods],
                [method.replace("_", " ") for method in methods],
                [f"{budget:.0%}" for budget in budgets],
                palette,
                ".1%",
            )
            ax.set_title(GRADER_NAMES[grader])
        fig.supxlabel("Target coverage", x=0.62, y=0.035)
        dataset, model = archive.rsplit("_", 1)
        dataset = {"ifeval": "IFEval", "mmlu_pro": "MMLU-Pro"}[dataset]
        model = {"qwen": "Qwen 2.5 7B", "llama": "Llama 3.1 8B"}[model]
        fig.text(0.018, 0.94, f"{dataset} · {model}", weight="bold")
        fig.text(0.018, 0.87, "Accepted error rate")
        export(fig, output / f"risk-coverage-{archive.replace('_', '-')}")


def short_model(model):
    return model.split("_", 1)[-1]


def wildbench_shifts(rows, output):
    pairs = list(
        dict.fromkeys((row["first_judge"], row["second_judge"]) for row in rows)
    )
    models = sorted({row["model"] for row in rows})
    fig, axes = plt.subplots(
        1, len(pairs), figsize=(12.2, 6.5), sharey=True, sharex=True
    )
    fig.subplots_adjust(left=0.31, right=0.98, top=0.91, bottom=0.15, wspace=0.12)
    for ax, pair in zip(axes, pairs, strict=True):
        ax.axvline(0, color="#556777", linewidth=1, linestyle="--")
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
            ax.hlines(y, lower, upper, linewidth=1.5, color="#286b83")
            ax.scatter([lower, upper], [y, y], marker="|", color="#286b83", s=20)
            ax.scatter(point, y, s=20, color="#143e51", zorder=3)
        ax.set_title(f"{GRADER_NAMES[pair[0]]} to {GRADER_NAMES[pair[1]]}")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks([-5, 0, 5, 10, 25, 50], ["−5", "0", "5", "10", "25", "50"])
        ax.grid(axis="y", color="#e4e9ec", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    totals = {row["model"]: int(row["total"]) for row in rows}
    labels = [f"{short_model(model)}  ({totals[model]:,})" for model in models]
    axes[0].set_yticks(range(len(models)), labels, fontsize=8.5)
    axes[0].invert_yaxis()
    fig.text(0.018, 0.945, "Model (paired rows)", weight="bold")
    fig.supxlabel(
        "Excess grader shift (percentage points), paired 95% interval", x=0.64, y=0.045
    )
    export(fig, output / "wildbench-grader-shifts")


def wildbench_auroc(rows, output):
    data = [
        row
        for row in rows
        if row["scenario"] == "wildbench" and row["outcome_cutoff"] == "7"
    ]
    judges = list(dict.fromkeys(row["judge"] for row in data))
    names = {
        "mean_token_logprob": "Mean\nlog p",
        "sequence_logprob": "Seq.\nlog p",
        "negative_mean_token_logprob": "Neg. mean\nlog p",
        "low_token_logprob": "Low\nlog p",
        "shorter_response": "Shorter\nresponse",
    }
    scores = list(dict.fromkeys(row["score"] for row in data))
    models = sorted({row["model"] for row in data})
    palette = LinearSegmentedColormap.from_list(
        "auroc", ["#e3af83", "#ffffff", "#84bfd0"]
    )
    fig, axes = plt.subplots(1, len(judges), figsize=(14.4, 6.3), sharey=True)
    fig.subplots_adjust(left=0.215, right=0.99, top=0.92, bottom=0.13, wspace=0.08)
    for ax, judge in zip(axes, judges, strict=True):
        values = {
            (row["model"], row["score"]): float(row["auroc"] or "nan")
            for row in data
            if row["judge"] == judge
        }
        score_matrix(
            ax,
            [[values[model, score] for score in scores] for model in models],
            [short_model(model) for model in models],
            [names[score] for score in scores],
            palette,
            ".2f",
        )
        ax.set_title(GRADER_NAMES[judge])
        ax.tick_params(labelsize=8)
    fig.text(0.015, 0.95, "AUROC by confidence score", weight="bold")
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
    figure, axes = plt.subplots(2, 3, figsize=(9.2, 5.9), sharex=True, sharey=True)
    figure.subplots_adjust(
        left=0.14, right=0.985, bottom=0.13, top=0.90, wspace=0.18, hspace=0.15
    )
    for column, (archive, score, title) in enumerate(examples):
        axes[0, column].set_title(title)
        for index, method in enumerate(("raw", "platt")):
            axis = axes[index, column]
            bins = [
                row
                for row in rows
                if row["archive"] == archive
                and row["score"] == score
                and row["method"] == method
                and int(row["count"]) > 0
            ]
            axis.plot([0, 1], [0, 1], color="#748694", linestyle="--", linewidth=1)
            axis.scatter(
                [float(row["mean_probability"]) for row in bins],
                [float(row["accuracy"]) for row in bins],
                s=[14 + 0.12 * int(row["count"]) for row in bins],
                facecolors="none",
                edgecolors="#126681",
                linewidths=1.4,
            )
            axis.set(
                xlim=(-0.06, 1.06),
                ylim=(-0.06, 1.06),
                xticks=(0, 0.5, 1),
                yticks=(0, 0.5, 1),
                aspect="equal",
            )
            axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            axis.grid(color="#e1e8eb", linewidth=0.65)
            axis.set_axisbelow(True)
    for index, method in enumerate(("Raw", "Platt")):
        axes[index, 0].set_ylabel(method, weight="bold", labelpad=8)
    figure.supylabel("Observed correctness", x=0.024)
    figure.supxlabel("Mean predicted probability", x=0.55, y=0.035)
    export(figure, output / "calibration-replay")


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
    export(figure, output / "winogrande-audit")


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
    audit_stopping(args.output_dir)
    risk_coverage(read_csv(ROOT / "results/module_study/ranking.csv"), args.output_dir)
    records = args.replay_dir / "judge_study"
    wildbench_shifts(read_csv(records / "gate_cells.csv"), args.output_dir)
    wildbench_auroc(read_csv(records / "auroc.csv"), args.output_dir)


if __name__ == "__main__":
    main()
