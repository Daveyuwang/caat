"""Replay selective annotation policies on the bundled prediction CSVs."""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator, PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
# Let the course replay run directly from a checkout.
sys.path.insert(0, str(ROOT / "src"))
from caat.evaluation import grouped_counts, threshold_points

DATASETS = {"imdb": "IMDb", "jigsaw": "Jigsaw", "fever": "FEVER"}


@dataclass(frozen=True)
class Prediction:
    gold: str
    gold_class: str
    llm: str
    deberta: str
    trust_score: float
    deberta_score: float
    balanced_accept: bool


def load_predictions(path):
    with path.open(encoding="utf-8", newline="") as source:
        return [
            Prediction(
                gold=row["gold_label"],
                gold_class=row["gold_label_str"],
                llm=row["llm_label"],
                deberta=row["deberta_predicted_label"],
                trust_score=float(row["trust_score_calibrated"]),
                deberta_score=float(row[f"prob_class{row['deberta_predicted_label']}"]),
                balanced_accept=row["policy_balanced_decision"] == "1",
            )
            for row in csv.DictReader(source)
        ]


def counts_by_class(predictions, labels, accept):
    overall, by_class = grouped_counts(
        (row.gold_class, accepted, label != row.gold)
        for row, label, accepted in zip(predictions, labels, accept, strict=True)
    )
    return {"overall": overall, **by_class}


def evaluate(predictions):
    llm_labels = [row.llm for row in predictions]
    deberta_labels = [row.deberta for row in predictions]
    balanced = [row.balanced_accept for row in predictions]
    agreement = [row.llm == row.deberta for row in predictions]
    trust_curve = threshold_points(
        (row.trust_score, row.llm != row.gold) for row in predictions
    )
    deberta_curve = threshold_points(
        (row.deberta_score, row.deberta != row.gold) for row in predictions
    )

    # Match the saved policy's acceptance count without splitting score ties.
    target = sum(balanced)
    threshold = min(
        (score for score, counts in deberta_curve if counts.accepted <= target),
        default=float("inf"),
    )
    matched = [row.deberta_score >= threshold for row in predictions]
    operating_points = {
        ("trust_scorer", "balanced"): counts_by_class(predictions, llm_labels, balanced),
        ("agreement", "labels_agree"): counts_by_class(predictions, llm_labels, agreement),
        ("deberta_only", "matched_trust_coverage"): counts_by_class(
            predictions, deberta_labels, matched
        ),
        ("deberta_only", "full_coverage"): counts_by_class(
            predictions, deberta_labels, [True] * len(predictions)
        ),
    }
    return trust_curve, deberta_curve, operating_points


def plot_panel(ax, title, trust_curve, deberta_curve, operating_points):
    trust_color = "#24518C"
    deberta_color = "#B4511E"
    agreement_color = "#08745C"
    for curve, label, color, style, marker in (
        (trust_curve, "Trust scorer", trust_color, "-", "."),
        (deberta_curve, "DeBERTa-only", deberta_color, "--", None),
    ):
        ax.plot(
            [counts.coverage for _, counts in curve],
            [counts.error_rate for _, counts in curve],
            label=label, color=color, linestyle=style, linewidth=1.8,
            marker=marker, markersize=3,
        )

    for key, label, color, marker, face, size in (
        (("trust_scorer", "balanced"), "Trust scorer (balanced)", trust_color, "o", trust_color, 5),
        (("deberta_only", "matched_trust_coverage"), "DeBERTa-only (matched coverage)",
         deberta_color, "s", "white", 5.5),
        (("agreement", "labels_agree"), "Two-model agreement", agreement_color, "D", "none", 8),
    ):
        counts = operating_points[key]["overall"]
        ax.plot(
            [counts.coverage] if counts.accepted else [],
            [counts.error_rate] if counts.accepted else [],
            label=label, color=color, marker=marker, markerfacecolor=face, markersize=size,
            markeredgewidth=1.4, linestyle="none", zorder=4,
        )

    ax.set_title(f"{title} (n={trust_curve[-1][1].total:,})", loc="left", fontweight="bold")
    ax.set(xlim=(0, 1.025), ylim=(0, None))
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.24)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "offline_eval")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(11.7, 3.7))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(DATASETS))
    figure.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.30, wspace=0.32)
    figure.supylabel("Error rate among accepted labels", x=0.022)
    figure.supxlabel("Coverage (share of examples accepted)", y=0.156)

    table_path = args.output_dir / "class_errors.csv"
    with table_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["dataset", "method", "policy", "gold_class", "total", "accepted", "errors",
             "coverage", "error_rate"]
        )
        for (dataset, title), ax in zip(DATASETS.items(), axes, strict=True):
            path = ROOT / "results" / "trust_scorer" / f"{dataset}_lr_predictions.csv"
            trust_curve, deberta_curve, operating_points = evaluate(load_predictions(path))
            plot_panel(ax, title, trust_curve, deberta_curve, operating_points)
            for (method, policy), groups in operating_points.items():
                for gold_class, counts in groups.items():
                    writer.writerow([
                        dataset, method, policy, gold_class, counts.total, counts.accepted, counts.errors,
                        counts.coverage, counts.error_rate,
                    ])

    handles, labels = axes[0].get_legend_handles_labels()
    # Keep each curve and its operating point in the same legend column.
    order = [0, 2, 1, 3, 4]
    figure.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False,
    )
    figure_path = args.output_dir / "risk_coverage.png"
    figure.savefig(figure_path, dpi=180)
    print(f"Wrote {table_path} and {figure_path}")


if __name__ == "__main__":
    main()
