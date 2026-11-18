"""Compare saved confidence scores on the same questions across model families."""

import argparse
from pathlib import Path

from caat.metrics import calibration_metrics
from caat.scores import DEFAULT_COVERAGES, ScoredRow, compare_scores

from common import read_csv, write_report
from study_black_box import BASELINE, BOOTSTRAP_SEED, RESAMPLES, TOKEN_LIMITS


OPTIONAL = ("verbalized_confidence", "answer_span_logprob")
PARTITIONS = ("design", "calibration", "heldout")


def read_partitions(path):
    assignments = {dataset: {} for dataset in TOKEN_LIMITS}
    for row in read_csv(path):
        dataset, item_id, fold = row["dataset"], row["id"], row["split"]
        if dataset not in assignments or not item_id or fold not in PARTITIONS:
            raise ValueError(f"Invalid split assignment for {dataset}/{item_id}")
        if item_id in assignments[dataset]:
            raise ValueError(f"Duplicate split assignment for {dataset}/{item_id}")
        assignments[dataset][item_id] = fold
    return assignments


def read_models(directories, dataset, assignments):
    models, names = {}, ()
    for path in directories:
        rows = read_csv(path / f"{dataset}_scores.csv")
        indexed = {row["id"]: row for row in rows}
        if (
            not rows
            or len(indexed) != len(rows)
            or indexed.keys() != assignments.keys()
        ):
            raise ValueError(f"{path.name}/{dataset} does not match the frozen IDs")
        columns = tuple(column for column in rows[0] if column not in {"id", "correct"})
        if BASELINE not in columns or (names and set(names) != set(columns)):
            raise ValueError(f"{path.name}/{dataset} has different score columns")
        if any(row["correct"] not in {"0", "1"} for row in rows):
            raise ValueError(f"{path.name}/{dataset} has invalid correctness labels")
        names = names or columns
        models[path.name] = indexed
    return models, names


def matched_ids(models, assignments, names):
    return {
        item_id
        for item_id in assignments
        if all(table[item_id][name] for table in models.values() for name in names)
    }


def compare(table, ids, names, *, bootstrap):
    scored = [
        ScoredRow(
            item_id,
            tuple(float(table[item_id][name]) for name in names),
            (table[item_id]["correct"] == "1",),
        )
        for item_id in ids
    ]
    return compare_scores(
        scored,
        names=names,
        graders=("correct",),
        baseline=BASELINE,
        coverages=DEFAULT_COVERAGES,
        bootstrap=bootstrap,
        seed=BOOTSTRAP_SEED,
        split="all",
    )["graders"]["correct"]["scores"]


def comparison_rows(directories, partitions):
    for dataset, assignments in partitions.items():
        models, names = read_models(directories, dataset, assignments)
        core = tuple(name for name in names if name not in OPTIONAL)
        common = matched_ids(models, assignments, core)
        design_ids = [item_id for item_id in common if assignments[item_id] == "design"]
        if not design_ids or not any(assignments[i] == "heldout" for i in common):
            raise ValueError(f"{dataset} needs shared design and held-out scores")

        cohorts = [(core, common)]
        for name in OPTIONAL:
            if name in names:
                pair = (BASELINE, name)
                cohorts.append((pair, matched_ids(models, assignments, pair)))

        for model, table in models.items():
            design = compare(table, design_ids, core, bootstrap=0)
            gains = {name: design[name]["normalized_aurc_gain"] for name in core}
            selected = max(
                (name for name in core if gains[name] is not None and gains[name] > 0),
                key=gains.get,
                default=None,
            )
            for split in ("all", "heldout"):
                ids = [
                    item_id
                    for item_id, fold in assignments.items()
                    if split == "all" or split == fold
                ]
                for index, (columns, shared) in enumerate(cohorts):
                    eligible = [item_id for item_id in ids if item_id in shared]
                    metrics = (
                        compare(table, eligible, columns, bootstrap=RESAMPLES)
                        if eligible
                        else {}
                    )
                    labels = [table[item_id]["correct"] == "1" for item_id in eligible]
                    for name in columns if index == 0 else columns[1:]:
                        row = result_row(table, ids, eligible, labels, metrics, name)
                        yield {
                            "model": model,
                            "dataset": dataset,
                            "split": split,
                            "method": name,
                            "design_normalized_aurc_gain": gains.get(name),
                            "selected_on_design": int(name == selected),
                            **row,
                        }


def result_row(table, ids, eligible, labels, metrics, name):
    ranking = metrics.get(name, {})
    budgets = ranking.get("budgets", ())
    at_75 = budgets[-1] if budgets else {}
    bounds = at_75.get("risk_delta_95") or (None, None)
    calibration = (
        calibration_metrics([float(table[i][name]) for i in eligible], labels)
        if eligible and name in {"p_true", "verbalized_confidence"}
        else {}
    )
    return {
        "total_rows": len(ids),
        "scored_rows": sum(bool(table[item_id][name]) for item_id in ids),
        "matched_rows": len(eligible),
        "matched_correct": sum(labels),
        "auroc": ranking.get("auroc"),
        "normalized_aurc_gain": ranking.get("normalized_aurc_gain"),
        "brier_score": calibration.get("brier_score"),
        "ece": calibration.get("ece"),
        **{
            f"risk_{round(coverage * 100)}": budget.get("error_rate")
            for coverage, budget in zip(
                DEFAULT_COVERAGES, budgets or [{}] * len(DEFAULT_COVERAGES), strict=True
            )
        },
        "accepted_75": at_75.get("accepted"),
        "mean_risk_75_on_matched_rows": metrics[BASELINE]["budgets"][-1]["error_rate"]
        if metrics
        else None,
        "delta_75_vs_mean": at_75.get("risk_delta"),
        "delta_75_ci_low": bounds[0],
        "delta_75_ci_high": bounds[1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores_dirs", type=Path, nargs="+", metavar="DIR")
    parser.add_argument("--split", type=Path, required=True, help="Frozen split CSV.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.scores_dirs) < 2 or len({d.name for d in args.scores_dirs}) != len(
        args.scores_dirs
    ):
        parser.error("provide at least two score directories with distinct names")
    inputs = [
        args.split,
        *(
            folder / f"{task}_scores.csv"
            for folder in args.scores_dirs
            for task in TOKEN_LIMITS
        ),
    ]
    if args.output.resolve() in {path.resolve() for path in inputs}:
        parser.error("output would overwrite an input file")
    try:
        partitions = read_partitions(args.split)
        rows = list(comparison_rows(args.scores_dirs, partitions))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_report(args.output, rows)
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
