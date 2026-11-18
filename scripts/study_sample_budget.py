"""Compare the held-out value of additional saved generations."""

import argparse
from math import ceil, isfinite
from pathlib import Path
from random import Random
from statistics import fmean

from caat.audit import percentile_interval
from caat.metrics import ranking_metrics, threshold_points
from caat.scores import _ScoreRanking

from common import ROOT, read_csv, write_report
from prepare_sample_budget import (
    BASELINES,
    BUDGETS,
    FROZEN,
    REPLAY,
    SAMPLE_SCORES,
    SOURCE_FOLDERS,
    prefix_column,
)
from study_black_box import BOOTSTRAP_SEED, RESAMPLES
from study_model_families import read_partitions


COVERAGE = 0.75
COUNTS = (0, *BUDGETS)


def select_score(rows, design, budget):
    errors = [rows[item]["correct"] == "0" for item in design]
    candidates = BASELINES if budget == 0 else SAMPLE_SCORES
    best = None
    for name in candidates:
        column = prefix_column(name, budget) if budget else name
        values = [float(rows[item][column]) for item in design]
        if not all(map(isfinite, values)):
            raise ValueError(f"Nonfinite design value in {column}")
        gain = ranking_metrics(threshold_points(zip(values, errors, strict=True)))[
            "normalized_aurc_gain"
        ]
        if gain is None:
            raise ValueError("Design rows need both correct and incorrect answers")
        if best is None or gain > best[2]:
            best = name, column, gain
    return best


def model_rankings(rows, design, heldout):
    errors = [rows[item]["correct"] == "0" for item in heldout]
    ranks, selections = [], []
    for budget in COUNTS:
        name, column, gain = select_score(rows, design, budget)
        values = [float(rows[item][column]) for item in heldout]
        if not all(map(isfinite, values)):
            raise ValueError(f"Nonfinite holdout value in {column}")
        ranks.append(_ScoreRanking.build(values, errors))
        pairs = 0
        if name == "negative_semantic_entropy":
            pairs = sum(
                int(rows[item][prefix_column("nli_pairs", budget)]) for item in heldout
            )
        selections.append(
            {
                "extra_prompt_evaluations_per_answer": budget,
                "selected_score": name,
                "design_normalized_aurc_gain": gain,
                "holdout_nli_directional_pairs": pairs,
            }
        )
    return ranks, selections


def compare_budgets(rankings, size, accepted):
    def rates(weights):
        return [
            [rank.risks(weights, (accepted,))[0] for rank in model]
            for model in rankings
        ]

    point = rates([1] * size)
    point_means = [fmean(values) for values in zip(*point, strict=True)]
    baseline_draws = [[] for _ in COUNTS]
    previous_draws = [[] for _ in COUNTS]
    rng = Random(BOOTSTRAP_SEED)
    for _ in range(RESAMPLES):
        weights = [0] * size
        for index in rng.choices(range(size), k=size):
            weights[index] += 1
        means = [fmean(values) for values in zip(*rates(weights), strict=True)]
        for index, value in enumerate(means):
            baseline_draws[index].append(value - means[0])
            previous_draws[index].append(value - means[max(0, index - 1)])

    summary = []
    for index, (budget, rate) in enumerate(zip(COUNTS, point_means, strict=True)):
        baseline_low, baseline_high = percentile_interval(baseline_draws[index])
        previous_low, previous_high = percentile_interval(previous_draws[index])
        summary.append(
            {
                "extra_prompt_evaluations_per_answer": budget,
                "error_rate_75": rate,
                "delta_vs_zero": rate - point_means[0],
                "delta_vs_zero_ci_low": baseline_low,
                "delta_vs_zero_ci_high": baseline_high,
                "delta_vs_previous": rate - point_means[max(0, index - 1)],
                "delta_vs_previous_ci_low": previous_low,
                "delta_vs_previous_ci_high": previous_high,
            }
        )
    return point, summary


def study(source, output):
    partitions = read_partitions(FROZEN / "split.csv")
    tables = {
        (dataset, model): {} for dataset in partitions for model in SOURCE_FOLDERS
    }
    for row in read_csv(source):
        item = row["id"]
        table = tables[(row["dataset"], row["model"])]
        if not item or item in table or row["correct"] not in {"0", "1"}:
            raise ValueError(f"Invalid score record for {item!r}")
        table[item] = row

    model_rows, task_rows = [], []
    for dataset, assignments in partitions.items():
        design = sorted(item for item, fold in assignments.items() if fold == "design")
        heldout = sorted(
            item for item, fold in assignments.items() if fold == "heldout"
        )
        active = set(design) | set(heldout)
        accepted = ceil(len(heldout) * COVERAGE)
        rankings, selections = [], []
        for model in SOURCE_FOLDERS:
            rows = tables[(dataset, model)]
            if rows.keys() != active:
                raise ValueError(f"{dataset}/{model} does not match the frozen split")
            ranked, selected = model_rankings(rows, design, heldout)
            rankings.append(ranked)
            selections.append(selected)
        points, summaries = compare_budgets(rankings, len(heldout), accepted)
        counts = {
            "heldout_questions_per_model": len(heldout),
            "accepted_per_model": accepted,
        }
        for model, chosen, rates in zip(
            SOURCE_FOLDERS, selections, points, strict=True
        ):
            for selection, rate in zip(chosen, rates, strict=True):
                model_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        **counts,
                        **selection,
                        "error_rate_75": rate,
                    }
                )
        for summary, chosen in zip(
            summaries, zip(*selections, strict=True), strict=True
        ):
            task_rows.append(
                {
                    "dataset": dataset,
                    "models": len(SOURCE_FOLDERS),
                    **counts,
                    "holdout_nli_directional_pairs": sum(
                        row["holdout_nli_directional_pairs"] for row in chosen
                    ),
                    **summary,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    write_report(output / "model_results.csv", model_rows)
    write_report(output / "task_results.csv", task_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "results/sample_budget/prefix_scores.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=REPLAY)
    args = parser.parse_args()
    try:
        study(args.input, args.output_dir)
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
