"""Compare saved ranking scores on the same outcomes and review budgets."""

import os
from bisect import bisect_left, bisect_right
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from itertools import accumulate, groupby, repeat, starmap
from math import ceil, isfinite
from random import Random
from statistics import mean

from .audit import percentile_interval
from .features import FEATURES
from .metrics import ranking_metrics, risk_at_count, threshold_points
from .policy import split_rows
from .table import exact_match, open_selected_predictions, parse_correct, parse_score


DEFAULT_COVERAGES = (0.25, 0.5, 0.75)
COSTS = ("no-extra-call", "one-extra-call", "n-samples", "external", "unknown")


@dataclass(frozen=True)
class ScoredRow:
    id: str
    scores: tuple[float | None, ...]
    outcomes: tuple[bool, ...]
    length: float | None = None
    output_key: str | None = None


def read_score_rows(
    path,
    names,
    graders,
    *,
    id_column="id",
    length_column=None,
    output_key_column=None,
    selection=None,
    source_rows=None,
    negated_scores=None,
    review_missing_scores=False,
    label_columns=None,
):
    """Read score columns with 0/1 grader columns or one exact label match.

    label_columns is a (prediction, reference) pair that replaces graders.
    """
    negated_scores = negated_scores or {}
    columns = tuple(
        (negated_scores.get(name, name), name in negated_scores) for name in names
    )
    needed = {id_column, *graders, *(label_columns or ())}
    needed.update(column for column, _ in columns)
    needed.update(
        column for column in (length_column, output_key_column) if column is not None
    )
    with open_selected_predictions(path, needed, None, selection) as (
        source,
        reader,
        _,
    ):
        rows = []
        for row in source:
            rows.append(
                ScoredRow(
                    row[id_column],
                    tuple(
                        None
                        if review_missing_scores and row[column] == ""
                        else parse_score(row[column], reader.line_num, negate=negate)
                        for column, negate in columns
                    ),
                    (
                        (exact_match(row[id_column], *(row[c] for c in label_columns)),)
                        if label_columns
                        else ()
                    )
                    + tuple(
                        parse_correct(row[name], reader.line_num) for name in graders
                    ),
                    parse_score(row[length_column], reader.line_num)
                    if length_column is not None
                    else None,
                    row[output_key_column] if output_key_column is not None else None,
                )
            )
            if source_rows is not None:
                source_rows.append(row)
    if output_key_column is not None:
        for row in rows:
            if not row.output_key:
                raise ValueError(f"{path}: missing output key for {row.id!r}")
    return rows


def _coverage_count(total, target):
    # A float product can round just above an integral decimal budget.
    fraction = Fraction(str(target)) if isinstance(target, float) else target
    return ceil(total * fraction)


def coverage_budgets(points, coverages, *, total=None):
    scored = points[-1][1].total if points else 0
    total = scored if total is None else total
    if not total:
        raise ValueError("coverage budgets need a nonempty cohort")
    for target in coverages:
        count = _coverage_count(total, target)
        feasible = count <= scored
        yield {
            "target_coverage": target,
            "accepted": count if feasible else None,
            "coverage": count / total if feasible else None,
            "error_rate": risk_at_count(points, count) if feasible else None,
        }


@dataclass(frozen=True)
class _ScoreRanking:
    order: list[int]
    ties: list[tuple[int, int]]
    error_positions: list[int]
    error_rows: list[int]

    @classmethod
    def build(cls, scores, errors, *, review_missing_scores=False):
        indices = (
            (i for i, score in enumerate(scores) if score is not None)
            if review_missing_scores
            else range(len(scores))
        )
        order = sorted(indices, key=scores.__getitem__, reverse=True)
        ties = []
        start = 0
        for _, group in groupby(order, key=scores.__getitem__):
            end = start + sum(1 for _ in group)
            if end > start + 1:
                ties.append((start, end))
            start = end
        error_positions, error_rows = [], []
        for position, row in enumerate(order):
            if errors[row]:
                error_positions.append(position)
                error_rows.append(row)
        return cls(order, ties, error_positions, error_rows)

    def risks(self, weights, targets):
        # Row multiplicities change on each draw. Scores and tie groups stay fixed.
        counts = list(accumulate(map(weights.__getitem__, self.order), initial=0))
        errors = list(accumulate(map(weights.__getitem__, self.error_rows), initial=0))
        risks = []
        for target in targets:
            if target > counts[-1]:
                risks.append(None)
                continue
            position = bisect_left(counts, target) - 1
            start, end = position, position + 1
            tie = bisect_right(self.ties, (position, len(self.order))) - 1
            if tie >= 0 and position < self.ties[tie][1]:
                start, end = self.ties[tie]
            earlier_errors = errors[bisect_left(self.error_positions, start)]
            tied_errors = (
                errors[bisect_left(self.error_positions, end)] - earlier_errors
            )
            share = (target - counts[start]) / (counts[end] - counts[start])
            risks.append((earlier_errors + share * tied_errors) / target)
        return risks


def _bootstrap_chunk(task):
    rankings, targets, reference, state, replicates, row_count = task
    rng = Random()
    rng.setstate(state)
    order = range(row_count)
    differences = [[[] for _ in targets] for _ in rankings]
    for _ in range(replicates):
        weights = [0] * len(order)
        for row in rng.choices(order, k=len(order)):
            weights[row] += 1
        baseline = rankings[reference].risks(weights, targets)
        for i, ranking in enumerate(rankings):
            risks = baseline if i == reference else ranking.risks(weights, targets)
            for sample, value, initial in zip(
                differences[i], risks, baseline, strict=True
            ):
                if value is not None and initial is not None:
                    sample.append(value - initial)
    return differences


def _bootstrap_tasks(
    rankings, targets, reference, seed, replicates, workers, row_count
):
    rng = Random(seed)
    size, remainder = divmod(replicates, workers)
    for i in range(workers):
        count = size + (i < remainder)
        yield rankings, targets, reference, rng.getstate(), count, row_count
        if i + 1 < workers:
            # Unweighted choices consumes one random() call per sampled row.
            deque(starmap(rng.random, repeat((), row_count * count)), maxlen=0)


def _paired_risk_stats(
    score_lists,
    errors,
    reference,
    coverages,
    replicates,
    seed,
    *,
    workers=1,
    review_missing_scores=False,
):
    """Return the interval and number of joint feasible draws at each budget."""
    if not replicates:
        return [[(None, 0)] * len(coverages) for _ in score_lists]
    if len(score_lists) == 1 and None not in score_lists[0]:
        return [[([0.0, 0.0], replicates) for _ in coverages]]
    n = len(errors)
    if workers is None:
        workers = 1
        if n * len(score_lists) * replicates >= 20_000_000:
            cpu_count = getattr(os, "process_cpu_count", os.cpu_count)
            workers = min(8, cpu_count() or 1)
    workers = min(workers, replicates)
    targets = tuple(_coverage_count(n, target) for target in coverages)
    rankings = [
        _ScoreRanking.build(scores, errors, review_missing_scores=review_missing_scores)
        for scores in score_lists
    ]
    tasks = _bootstrap_tasks(rankings, targets, reference, seed, replicates, workers, n)
    if workers == 1:
        chunks = [_bootstrap_chunk(next(tasks))]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(_bootstrap_chunk, tasks))
    statistics = []
    for i in range(len(score_lists)):
        by_budget = []
        for j in range(len(targets)):
            deltas = [delta for chunk in chunks for delta in chunk[i][j]]
            by_budget.append((percentile_interval(deltas), len(deltas)))
        statistics.append(by_budget)
    return statistics


def paired_risk_intervals(
    score_lists, errors, reference, coverages, replicates, seed, *, workers=1
):
    """Resample complete rows and reuse each score's fixed order and ties."""
    stats = _paired_risk_stats(
        score_lists, errors, reference, coverages, replicates, seed, workers=workers
    )
    return [[interval for interval, _ in column] for column in stats]


def design_recommendation(design, names, *, review_missing_scores=False):
    design_gains, design_scored = {}, {}
    if design:
        for name, values in zip(
            names, zip(*(row.scores for row in design), strict=True), strict=True
        ):
            if review_missing_scores:
                design_scored[name] = len(values) - values.count(None)
                if design_scored[name] < len(design):
                    design_gains[name] = None
                    continue
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

    return design_gains, design_scored, recommendation


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
    workers=1,
    split_sizes=None,
    review_missing_scores=False,
):
    """Report frozen score comparisons and a design-only recommendation.

    With split_sizes, rows arrive in design, calibration, and holdout order.
    """
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
        or not all(
            isfinite(score) if score is not None else review_missing_scores
            for score in row.scores
        )
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
    if workers is not None and (type(workers) is not int or workers < 1):
        raise ValueError("bootstrap workers must be a positive integer")
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
    if split_sizes is not None:
        if (
            split != "heldout"
            or len(split_sizes) != 3
            or any(type(size) is not int or size < 1 for size in split_sizes)
            or sum(split_sizes) != len(rows)
        ):
            raise ValueError(
                "split sizes must cover the design, calibration, and holdout rows"
            )
        first = split_sizes[0]
        second = first + split_sizes[1]
        design, calibration, target = rows[:first], rows[first:second], rows[second:]
    elif split == "heldout":
        if len(rows) < 5:
            raise ValueError(
                "at least five rows are required for the design, calibration, and held-out splits"
            )
        design, calibration, target = split_rows(rows, seed)
    elif split == "all":
        rows.sort(key=lambda row: row.id)
        design, calibration, target = [], [], rows
    else:
        raise ValueError("comparison split must be heldout or all")

    def scores_for(data):
        return tuple(zip(*(row.scores for row in data), strict=True))

    design_gains, design_scored, recommendation = design_recommendation(
        design, names, review_missing_scores=review_missing_scores
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
        intervals = _paired_risk_stats(
            scores,
            errors,
            reference,
            coverages,
            bootstrap,
            seed,
            workers=workers,
            review_missing_scores=review_missing_scores,
        )

        def summary(values):
            pairs = zip(values, errors, strict=True)
            if review_missing_scores:
                pairs = ((score, error) for score, error in pairs if score is not None)
            points = threshold_points(pairs)
            result = ranking_metrics(points)
            result["budgets"] = list(
                coverage_budgets(points, coverages, total=len(target))
            )
            return result

        candidates = {}
        for column, values in zip(names, scores, strict=True):
            entry = summary(values)
            auroc = entry["auroc"]
            entry["direction_warning"] = auroc is not None and auroc <= 0.5
            if review_missing_scores:
                entry["scored"] = len(values) - values.count(None)
                entry["maximum_coverage"] = entry["scored"] / len(values)
            candidates[column] = entry
        for entry, bounds in zip(candidates.values(), intervals, strict=True):
            for budget, comparison, (interval, paired_draws) in zip(
                entry["budgets"], candidates[baseline]["budgets"], bounds, strict=True
            ):
                risk, baseline_risk = budget["error_rate"], comparison["error_rate"]
                comparable = risk is not None and baseline_risk is not None
                budget["risk_delta"] = risk - baseline_risk if comparable else None
                budget["risk_delta_95"] = interval if comparable else None
                if review_missing_scores:
                    budget["risk_delta_paired_draws"] = paired_draws
        by_grader[name] = {
            "scores": candidates,
            "controls": {
                key: summary(values) for key, values in control_scores.items()
            },
        }
    result = {
        "baseline": baseline,
        "costs": {
            name: costs.get(
                name, FEATURES[name].cost if name in FEATURES else "unknown"
            )
            for name in names
        },
        "split": {
            "seed": seed if split_sizes is None else None,
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
    if review_missing_scores:
        result["design_scored"] = design_scored
    return result
