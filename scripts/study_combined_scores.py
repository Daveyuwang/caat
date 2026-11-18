"""Fit and test score combinations on the frozen three-family benchmarks."""

import argparse
from importlib.metadata import version
from itertools import product
from math import ceil
from pathlib import Path
from random import Random
from statistics import fmean

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from caat.audit import percentile_interval
from caat.metrics import ranking_metrics, risk_at_count, threshold_points
from caat.scores import _ScoreRanking

from common import ROOT, write_report
from study_black_box import BOOTSTRAP_SEED, RESAMPLES, write_json
from study_model_families import OPTIONAL, matched_ids, read_models, read_partitions


MODELS = ("qwen2_5_7b", "phi4_mini", "smollm2")
TRANSFORMS = ("standardized", "quantile")
STRENGTHS = (0.01, 0.1, 1.0)
COVERAGE = 0.75
CV_SEED = 42


def normalized_gain(scores, errors):
    return ranking_metrics(threshold_points(zip(scores, errors, strict=True)))[
        "normalized_aurc_gain"
    ]


def combiner(transform, strength, size):
    preprocessing = []
    if transform == "quantile":
        preprocessing.append(
            QuantileTransformer(
                n_quantiles=size, subsample=None, output_distribution="uniform"
            )
        )
    return make_pipeline(
        *preprocessing,
        StandardScaler(),
        LogisticRegression(C=strength, l1_ratio=0.0, solver="lbfgs", max_iter=1000),
    )


def fit_combiner(design, labels):
    count = min(5, int(np.bincount(labels, minlength=2).min()))
    if count < 2:
        raise ValueError(
            "Design needs at least two rows of each outcome to tune the combination"
        )
    folds = list(
        StratifiedKFold(count, shuffle=True, random_state=CV_SEED).split(design, labels)
    )
    candidates = []
    for transform, strength in product(TRANSFORMS, STRENGTHS):
        gains = []
        for train, validate in folds:
            model = combiner(transform, strength, len(train)).fit(
                design[train], labels[train]
            )
            scores = model.decision_function(design[validate])
            gains.append(normalized_gain(scores, labels[validate] == 0))
        candidates.append(
            {"transform": transform, "C": strength, "cv_gain": fmean(gains)}
        )
    selected = max(candidates, key=lambda candidate: candidate["cv_gain"])
    model = combiner(selected["transform"], selected["C"], len(design)).fit(
        design, labels
    )
    return model, selected, candidates, count


def matrix(table, ids, columns):
    return np.array(
        [[float(table[item_id][name]) for name in columns] for item_id in ids]
    )


def evaluate_model(table, columns, design_ids, heldout_ids):
    design = matrix(table, design_ids, columns)
    labels = np.array([int(table[item_id]["correct"]) for item_id in design_ids])
    fitted, selected, candidates, folds = fit_combiner(design, labels)
    training_gains = [normalized_gain(values, labels == 0) for values in design.T]
    baseline = max(range(len(columns)), key=training_gains.__getitem__)

    heldout = matrix(table, heldout_ids, columns)
    combined = fitted.decision_function(heldout)
    correct = [int(table[item_id]["correct"]) for item_id in heldout_ids]
    errors = [label == 0 for label in correct]
    single = heldout[:, baseline]
    count = ceil(len(heldout_ids) * COVERAGE)
    single_risks = [
        risk_at_count(threshold_points(zip(values, errors, strict=True)), count)
        for values in heldout.T
    ]
    lowest_observed = min(range(len(columns)), key=single_risks.__getitem__)
    ranks = (_ScoreRanking.build(single, errors), _ScoreRanking.build(combined, errors))
    rows = [
        {
            "id": item_id,
            "correct": label,
            "single_score": float(score),
            "combined_score": float(value),
        }
        for item_id, label, score, value in zip(
            heldout_ids, correct, single, combined, strict=True
        )
    ]
    fit = {
        **selected,
        "folds": folds,
        "design_rows": len(design_ids),
        "design_correct": int(labels.sum()),
        "selected_single": columns[baseline],
        "single_design_gain": training_gains[baseline],
    }
    diagnostic = {
        "observed_best_single": columns[lowest_observed],
        "observed_best_single_risk_75": single_risks[lowest_observed],
    }
    return rows, ranks, fit, diagnostic, candidates


def paired_results(ranks, size):
    accepted = ceil(size * COVERAGE)

    def risks(pair, weights):
        return tuple(ranking.risks(weights, (accepted,))[0] for ranking in pair)

    samples = [[] for _ in ranks]
    task_deltas = []
    rng = Random(BOOTSTRAP_SEED)
    for _ in range(RESAMPLES):
        weights = [0] * size
        for index in rng.choices(range(size), k=size):
            weights[index] += 1
        changes = []
        for pair, draws in zip(ranks, samples, strict=True):
            single, combined = risks(pair, weights)
            change = combined - single
            draws.append(change)
            changes.append(change)
        task_deltas.append(fmean(changes))

    def summarize(single, combined, deltas):
        low, high = percentile_interval(deltas)
        return {
            "heldout_questions": size,
            "accepted_per_model": accepted,
            "single_error_75": single,
            "combined_error_75": combined,
            "delta_75": combined - single,
            "delta_ci_low": low,
            "delta_ci_high": high,
        }

    points = [risks(pair, [1] * size) for pair in ranks]
    details = [
        summarize(*point, draws) for point, draws in zip(points, samples, strict=True)
    ]
    return details, summarize(*map(fmean, zip(*points, strict=True)), task_deltas)


def study(archive, output):
    model_results, task_results, choices, heldout_scores = [], [], [], {}
    for dataset, assignments in read_partitions(archive / "split.csv").items():
        models, names = read_models(
            [archive / model for model in MODELS], dataset, assignments
        )
        columns = tuple(name for name in names if name not in (*OPTIONAL, "random"))
        design = sorted(
            item_id for item_id, fold in assignments.items() if fold == "design"
        )
        heldout = sorted(
            item_id for item_id, fold in assignments.items() if fold == "heldout"
        )
        active = {item_id: assignments[item_id] for item_id in (*design, *heldout)}
        if not heldout or matched_ids(models, active, columns) != active.keys():
            raise ValueError(
                f"{dataset} needs complete primary scores on the frozen comparison rows"
            )
        ranks, summaries = [], []
        for name, table in models.items():
            rows, pair, fit, diagnostic, candidates = evaluate_model(
                table, columns, design, heldout
            )
            identity = {"dataset": dataset, "model": name}
            ranks.append(pair)
            summaries.append({**identity, **fit, **diagnostic})
            choices.append(
                {**identity, "columns": columns, "candidates": candidates, **fit}
            )
            heldout_scores[(name, dataset)] = rows
        models_at_budget, task_at_budget = paired_results(ranks, len(heldout))
        model_results.extend(
            {**summary, **budget}
            for summary, budget in zip(summaries, models_at_budget, strict=True)
        )
        task_results.append(
            {"dataset": dataset, "models": len(models), **task_at_budget}
        )

    output.mkdir(parents=True, exist_ok=True)
    for (model, dataset), rows in heldout_scores.items():
        folder = output / model
        folder.mkdir(exist_ok=True)
        write_report(folder / f"{dataset}_heldout.csv", rows)
    write_report(output / "model_results.csv", model_results)
    write_report(output / "task_results.csv", task_results)
    write_json(
        output / "selection.json",
        {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": version("scipy"),
            "cv_seed": CV_SEED,
            "coverage": COVERAGE,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": RESAMPLES,
            "fits": choices,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "results/black_box_families"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "replay/combined_scores"
    )
    args = parser.parse_args()
    try:
        study(args.archive, args.output_dir)
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
