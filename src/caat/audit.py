from math import ceil, isfinite
from random import Random
from typing import Iterable

from .metrics import grouped_counts
from .table import GradedRow, validated_rows


AuditRow = GradedRow


def percentile_interval(values):
    if not values:
        return None
    ordered = sorted(values)
    return [ordered[ceil(fraction * len(ordered)) - 1] for fraction in (0.025, 0.975)]


def bootstrap_shifts(pairs, replicates, seed):
    # Resample the full rows so the accepted and whole-set risks remain paired.
    rng = Random(seed)
    risk_deltas, whole_deltas, excess_deltas = [], [], []
    for _ in range(replicates):
        accepted = accepted_delta = total_delta = 0
        for kept, change in rng.choices(pairs, k=len(pairs)):
            total_delta += change
            if kept:
                accepted += 1
                accepted_delta += change
        whole = total_delta / len(pairs)
        whole_deltas.append(whole)
        if accepted:
            risk = accepted_delta / accepted
            risk_deltas.append(risk)
            excess_deltas.append(risk - whole)
    return {
        "risk_change": percentile_interval(risk_deltas),
        "overall_risk_change": percentile_interval(whole_deltas),
        "excess_risk_change": percentile_interval(excess_deltas),
        "replicates_with_acceptance": len(risk_deltas),
    }


def audit(
    rows: Iterable[GradedRow],
    *,
    graders: tuple[str, ...],
    threshold: float,
    risk_target: float | None = None,
    bootstrap: int | None = None,
    seed: int = 42,
) -> dict:
    """Replay one cutoff against paired correctness outcomes on the same outputs."""
    if not isfinite(threshold):
        raise ValueError("threshold must be finite")
    if risk_target is not None and not 0 <= risk_target <= 1:
        raise ValueError("risk target must be between 0 and 1")
    if bootstrap is not None and bootstrap < 1:
        raise ValueError("bootstrap count must be positive")
    rows = validated_rows(rows, graders, minimum_graders=2)

    accepted = [row for row in rows if row.score >= threshold]
    results = {}
    for index, name in enumerate(graders):
        overall, groups = grouped_counts(
            (row.group, row.score >= threshold, not row.outcomes[index]) for row in rows
        )
        risk = overall.error_rate
        results[name] = {
            **overall.metrics(),
            "meets_target": risk <= risk_target if risk is not None and risk_target is not None else None,
            "by_group": {group: counts.metrics() for group, counts in groups.items()},
        }

    comparisons = {}
    for index, name in enumerate(graders[1:], start=1):
        newly_failed = sum(row.outcomes[0] and not row.outcomes[index] for row in accepted)
        newly_passed = sum(not row.outcomes[0] and row.outcomes[index] for row in accepted)
        primary = results[graders[0]]["meets_target"]
        alternate = results[name]["meets_target"]
        risk_change = (newly_failed - newly_passed) / len(accepted) if accepted else None
        changes = [(row.score >= threshold, row.outcomes[0] - row.outcomes[index]) for row in rows]
        overall_change = sum(delta for _, delta in changes) / len(rows)
        comparisons[name] = {
            "newly_failed": newly_failed,
            "newly_passed": newly_passed,
            "risk_change": risk_change,
            "overall_risk_change": overall_change,
            "excess_risk_change": risk_change - overall_change if risk_change is not None else None,
            "target_decision_changed": primary != alternate if primary is not None else None,
        }
        if bootstrap is not None:
            comparisons[name]["bootstrap_95"] = bootstrap_shifts(changes, bootstrap, seed)

    discordant = [row for row in rows if any(bit != row.outcomes[0] for bit in row.outcomes[1:])]
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
                "accepted": row.score >= threshold,
                "outcomes": dict(zip(graders, row.outcomes, strict=True)),
            }
            for row in sorted(discordant, key=lambda row: row.id)
        ],
    }
    if bootstrap is not None:
        report["bootstrap"] = {"replicates": bootstrap, "seed": seed}
    return report
