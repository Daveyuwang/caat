"""Compare saved ranking scores on the same outcomes and review budgets."""

from dataclasses import dataclass
from math import ceil, isfinite
from random import Random
from statistics import mean

from .audit import percentile_interval
from .features import FEATURES
from .metrics import ranking_metrics, risk_at_count, threshold_points
from .policy import split_rows
from .table import open_predictions_csv, parse_correct, parse_score


DEFAULT_COVERAGES = (0.25, 0.5, 0.75)
COSTS = ("no-extra-call", "one-extra-call", "n-samples", "external", "unknown")


@dataclass(frozen=True)
class ScoredRow:
    id: str
    scores: tuple[float, ...]
    outcomes: tuple[bool, ...]
    length: float | None = None


def read_score_rows(path, names, graders, *, id_column="id", length_column=None):
    needed = {id_column, *names, *graders}
    if length_column is not None:
        needed.add(length_column)
    with open_predictions_csv(path, needed, None) as (reader, _):
        return [
            ScoredRow(
                row[id_column],
                tuple(parse_score(row[name], reader.line_num) for name in names),
                tuple(parse_correct(row[name], reader.line_num) for name in graders),
                parse_score(row[length_column], reader.line_num)
                if length_column is not None
                else None,
            )
            for row in reader
        ]


def coverage_budgets(points, coverages):
    total = points[-1][1].total
    for target in coverages:
        count = ceil(total * target)
        yield {
            "target_coverage": target,
            "accepted": count,
            "coverage": count / total,
            "error_rate": risk_at_count(points, count),
        }


def paired_risk_intervals(score_lists, errors, reference, coverages, replicates, seed):
    """Resample complete prediction rows and rerank each score at each budget."""
    rng = Random(seed)
    n = len(errors)
    order = range(n)
    targets = tuple(ceil(n * target) for target in coverages)
    differences = [[[] for _ in targets] for _ in score_lists]
    for _ in range(replicates if len(score_lists) > 1 else 0):
        sampled = rng.choices(order, k=n)

        def risks(scores):
            points = threshold_points((scores[i], errors[i]) for i in sampled)
            return [risk_at_count(points, target) for target in targets]

        baseline = risks(score_lists[reference])
        for i, values in enumerate(score_lists):
            if i != reference:
                for sample, value, initial in zip(
                    differences[i], risks(values), baseline, strict=True
                ):
                    sample.append(value - initial)
    return [
        [
            ([0.0, 0.0] if i == reference else percentile_interval(sample))
            if replicates
            else None
            for sample in group
        ]
        for i, group in enumerate(differences)
    ]


def compare_scores(
    rows,
    *,
    names,
    graders,
    baseline=None,
    length_control=False,
    costs=None,
    coverages=DEFAULT_COVERAGES,
    bootstrap=500,
    seed=42,
    split="heldout",
):
    """Report frozen score comparisons and a design-only recommendation."""
    names, graders, rows = tuple(names), tuple(graders), list(rows)
    if not names or len(set(names)) != len(names) or not all(names):
        raise ValueError("provide distinct nonempty score columns")
    if not graders or len(set(graders)) != len(graders) or not all(graders):
        raise ValueError("provide distinct nonempty correctness columns")
    if (
        not rows
        or len({row.id for row in rows}) != len(rows)
        or any(not row.id for row in rows)
    ):
        raise ValueError("score rows must have unique nonempty IDs")
    if any(
        len(row.scores) != len(names)
        or not all(map(isfinite, row.scores))
        or len(row.outcomes) != len(graders)
        or any(type(y) is not bool for y in row.outcomes)
        for row in rows
    ):
        raise ValueError(
            "each score row needs finite scores and one boolean outcome per grader"
        )
    if length_control and any(
        row.length is None or not isfinite(row.length) or row.length < 0 for row in rows
    ):
        raise ValueError("length must be finite and nonnegative")
    if type(bootstrap) is not int or bootstrap < 0:
        raise ValueError("bootstrap replicates must be nonnegative")
    coverages = tuple(sorted(set(coverages)))
    if not coverages or any(not 0 < c <= 1 for c in coverages):
        raise ValueError("coverages must be between 0 (exclusive) and 1")
    costs = costs or {}
    if set(costs) - set(names) or any(cost not in COSTS for cost in costs.values()):
        raise ValueError(
            "costs must name supplied score columns and valid cost classes"
        )
    baseline = baseline or names[0]
    if baseline not in names:
        raise ValueError("baseline must be among the supplied score columns")
    reference = names.index(baseline)
    rows.sort(key=lambda row: row.id)
    if split == "heldout":
        if len(rows) < 5:
            raise ValueError(
                "at least five rows are required for the design, calibration, and held-out splits"
            )
        design, calibration, target = split_rows(rows, seed)
    elif split == "all":
        design, calibration, target = [], [], rows
    else:
        raise ValueError("comparison split must be heldout or all")

    def scores_for(data):
        return tuple(zip(*(row.scores for row in data), strict=True))

    design_gains = {}
    if design:
        for name, values in zip(names, scores_for(design), strict=True):
            gains = []
            for outcomes in zip(*(row.outcomes for row in design), strict=True):
                points = threshold_points(
                    (score, not y) for score, y in zip(values, outcomes, strict=True)
                )
                gain = ranking_metrics(points)["normalized_aurc_gain"]
                if gain is not None:
                    gains.append(gain)
            design_gains[name] = mean(gains) if gains else None
    recommendation = max(
        (name for name, gain in design_gains.items() if gain is not None and gain > 0),
        key=design_gains.get,
        default=None,
    )

    scores = scores_for(target)
    control_scores = {"random": [0.0] * len(target)}
    if length_control:
        control_scores["length_only"] = [-row.length for row in target]
    by_grader = {}
    for name, outcomes in zip(
        graders, zip(*(row.outcomes for row in target), strict=True), strict=True
    ):
        errors = [not outcome for outcome in outcomes]
        intervals = paired_risk_intervals(
            scores, errors, reference, coverages, bootstrap, seed
        )

        def summary(values):
            points = threshold_points(zip(values, errors, strict=True))
            result = ranking_metrics(points)
            result["budgets"] = list(coverage_budgets(points, coverages))
            return result

        candidates = {}
        for column, values in zip(names, scores, strict=True):
            entry = summary(values)
            auroc = entry["auroc"]
            entry["direction_warning"] = auroc is not None and auroc <= 0.5
            candidates[column] = entry
        for entry, bounds in zip(candidates.values(), intervals, strict=True):
            for budget, comparison, interval in zip(
                entry["budgets"], candidates[baseline]["budgets"], bounds, strict=True
            ):
                budget["risk_delta"] = budget["error_rate"] - comparison["error_rate"]
                budget["risk_delta_95"] = interval
        by_grader[name] = {
            "scores": candidates,
            "controls": {
                key: summary(values) for key, values in control_scores.items()
            },
        }
    return {
        "baseline": baseline,
        "costs": {
            name: costs.get(
                name, FEATURES[name].cost if name in FEATURES else "unknown"
            )
            for name in names
        },
        "split": {
            "seed": seed,
            "comparison": split,
            "design": len(design),
            "calibration": len(calibration),
            "reported": len(target),
        },
        "bootstrap": {"replicates": bootstrap, "seed": seed},
        "design_normalized_aurc_gain": design_gains,
        "recommended_score": recommendation,
        "graders": by_grader,
    }
