from collections import Counter
from math import ceil, isfinite
from random import Random
from typing import Any, Iterable, Sequence

from .metrics import grouped_counts
from .rules import assess_accepted_risk, validate_holdout_confidence
from .table import GradedRow, UnscoredRow, accepted_at, validated_rows, write_rows


def percentile_interval(values):
    if not values:
        return None
    ordered = sorted(values)
    return [ordered[ceil(fraction * len(ordered)) - 1] for fraction in (0.025, 0.975)]


def _bootstrap_shifts(rows, threshold, replicates, seed):
    pattern_ids, population = {}, []
    for row in rows:
        pattern = (accepted_at(row, threshold), tuple(row.outcomes))
        population.append(pattern_ids.setdefault(pattern, len(pattern_ids)))
    patterns = [
        (kept, tuple(outcomes[0] - bit for bit in outcomes[1:]))
        for kept, outcomes in pattern_ids
    ]
    comparisons = len(rows[0].outcomes) - 1
    intervals = [([], [], []) for _ in range(comparisons)]
    rng = Random(seed)
    for _ in range(replicates):
        # Keep the seeded row draws and count identical outcomes once per draw.
        counts = Counter(rng.choices(population, k=len(population)))
        accepted = 0
        total_deltas = [0] * comparisons
        accepted_deltas = [0] * comparisons
        for index_of_pattern, count in counts.items():
            kept, changes = patterns[index_of_pattern]
            accepted += kept * count
            for index, change in enumerate(changes):
                delta = change * count
                total_deltas[index] += delta
                if kept:
                    accepted_deltas[index] += delta
        for index, (risk_deltas, whole_deltas, excess_deltas) in enumerate(intervals):
            whole = total_deltas[index] / len(population)
            whole_deltas.append(whole)
            if accepted:
                risk = accepted_deltas[index] / accepted
                risk_deltas.append(risk)
                excess_deltas.append(risk - whole)
    return [
        {
            "risk_change": percentile_interval(risk_deltas),
            "overall_risk_change": percentile_interval(whole_deltas),
            "excess_risk_change": percentile_interval(excess_deltas),
            "replicates_with_acceptance": len(risk_deltas),
        }
        for risk_deltas, whole_deltas, excess_deltas in intervals
    ]


def audit(
    rows: Iterable[GradedRow | UnscoredRow],
    *,
    graders: Sequence[str],
    threshold: float,
    risk_target: float | None = None,
    holdout_confidence: float | None = None,
    accepted_population: int | None = None,
    bootstrap: int | None = None,
    seed: int = 42,
    review_missing_scores: bool = False,
) -> dict[str, Any]:
    """Replay one cutoff against paired correctness outcomes on the same outputs.

    With an accepted population, the labeled accepted rows must be a uniform
    sample without replacement from that finite population.
    """
    if not isfinite(threshold):
        raise ValueError("threshold must be finite")
    if risk_target is not None and not 0 <= risk_target <= 1:
        raise ValueError("risk target must be between 0 and 1")
    validate_holdout_confidence(
        holdout_confidence, accepted_population=accepted_population
    )
    if bootstrap is not None and bootstrap < 1:
        raise ValueError("bootstrap count must be positive")
    if bootstrap is not None and accepted_population is not None:
        raise ValueError(
            "bootstrap assumes independent rows and needs an independent holdout"
        )
    rows = validated_rows(
        rows, graders, minimum_graders=2, review_missing_scores=review_missing_scores
    )

    accepted = [row for row in rows if accepted_at(row, threshold)]
    results = {}
    for index, name in enumerate(graders):
        overall, groups = grouped_counts(
            (row.group, accepted_at(row, threshold), not row.outcomes[index])
            for row in rows
        )
        results[name] = {
            **assess_accepted_risk(
                overall,
                risk_target,
                holdout_confidence,
                accepted_population=accepted_population,
            ),
            "by_group": {group: counts.metrics() for group, counts in groups.items()},
        }

    comparisons = {}
    for index, name in enumerate(graders[1:], start=1):
        newly_failed = sum(
            row.outcomes[0] and not row.outcomes[index] for row in accepted
        )
        newly_passed = sum(
            not row.outcomes[0] and row.outcomes[index] for row in accepted
        )
        primary = results[graders[0]]["meets_target"]
        alternate = results[name]["meets_target"]
        risk_change = (
            (newly_failed - newly_passed) / len(accepted) if accepted else None
        )
        overall_change = sum(
            row.outcomes[0] - row.outcomes[index] for row in rows
        ) / len(rows)
        comparisons[name] = {
            "newly_failed": newly_failed,
            "newly_passed": newly_passed,
            "risk_change": risk_change,
            "overall_risk_change": overall_change,
            "excess_risk_change": risk_change - overall_change
            if risk_change is not None
            else None,
            "target_decision_changed": primary != alternate
            if primary is not None
            else None,
        }
    if bootstrap is not None:
        intervals = _bootstrap_shifts(rows, threshold, bootstrap, seed)
        for comparison, interval in zip(comparisons.values(), intervals, strict=True):
            comparison["bootstrap_95"] = interval

    discordant = [
        row for row in rows if any(bit != row.outcomes[0] for bit in row.outcomes[1:])
    ]
    report = {
        "schema_version": 1,
        "threshold": threshold,
        "risk_target": risk_target,
        "primary_grader": graders[0],
        "graders": results,
        "accepted_comparisons": comparisons,
        "discordant": [
            {
                "id": row.id,
                "group": row.group,
                "accepted": accepted_at(row, threshold),
                "outcomes": dict(zip(graders, row.outcomes, strict=True)),
            }
            for row in sorted(discordant, key=lambda row: row.id)
        ],
    }
    if bootstrap is not None:
        report["bootstrap"] = {"replicates": bootstrap, "seed": seed}
    if holdout_confidence is not None:
        report["holdout_confidence"] = holdout_confidence
    if accepted_population is not None:
        report["accepted_population"] = accepted_population
    if review_missing_scores:
        report["scored"] = sum(row.score is not None for row in rows)
    return report


def write_disagreements(output, rows, report, *, include_output_key=False):
    """Export the audited disagreements with the original scores and output keys."""
    columns = ["id", "score", "decision", "group"]
    if include_output_key:
        columns.append("output_key")
    columns.extend(f"grade_{grader}" for grader in report["graders"])
    by_id = {row.id: row for row in rows}

    def records():
        for disagreement in report["discordant"]:
            row = by_id[disagreement["id"]]
            decision = "accept" if disagreement["accepted"] else "review"
            values = [row.id, row.score, decision, row.group]
            if include_output_key:
                values.append(row.output_key)
            values.extend(int(outcome) for outcome in row.outcomes)
            yield values

    write_rows(output, columns, records())
