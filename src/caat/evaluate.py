"""Evaluate saved predictions under a frozen threshold."""

from math import isfinite
from typing import Any, Iterable

from .metrics import (
    Counts,
    METRICS,
    grouped_counts,
    threshold_points,
)
from .rules import assess_accepted_risk, validate_holdout_confidence
from .table import GradedRow, UnscoredRow, accepted_at, validated_rows


def evaluate(
    rows: Iterable[GradedRow | UnscoredRow],
    *,
    threshold: float,
    risk_target: float | None = None,
    holdout_confidence: float | None = None,
    accepted_population: int | None = None,
    probability: bool = False,
    bins: int = 10,
    extra_metrics: Iterable[str] = (),
    review_missing_scores: bool = False,
) -> tuple[dict[str, Any], list[tuple[float, Counts]]]:
    """Return metrics at a fixed threshold and the attainable risk–coverage points.

    A holdout confidence level adds an exact upper risk bound. Providing an
    accepted population uses sampling without replacement from a fixed batch.
    """
    if not isfinite(threshold):
        raise ValueError("threshold must be finite")
    if risk_target is not None and not 0 <= risk_target <= 1:
        raise ValueError("risk target must be between 0 and 1")
    validate_holdout_confidence(
        holdout_confidence, accepted_population=accepted_population
    )
    rows = validated_rows(
        rows, ("correct",), review_missing_scores=review_missing_scores
    )
    scored = (
        [row for row in rows if row.score is not None]
        if review_missing_scores
        else rows
    )
    if probability:
        for row in scored:
            if not 0 <= row.score <= 1:
                raise ValueError(f"score for id {row.id!r} must be between 0 and 1")
    errors = [not row.outcomes[0] for row in rows]
    accepted = [accepted_at(row, threshold) for row in rows]
    overall, by_reference = grouped_counts(
        (row.reference, accept, error)
        for row, accept, error in zip(rows, accepted, errors, strict=True)
    )
    _, by_group = grouped_counts(
        (row.group, accept, error)
        for row, accept, error in zip(rows, accepted, errors, strict=True)
    )
    metrics = {
        "threshold": threshold,
        "overall": (
            assess_accepted_risk(
                overall,
                risk_target,
                holdout_confidence,
                accepted_population=accepted_population,
            )
            if risk_target is not None or holdout_confidence is not None
            else overall.metrics()
        ),
        "by_reference": {
            label: counts.metrics() for label, counts in by_reference.items()
        },
        "by_group": {group: counts.metrics() for group, counts in by_group.items()},
        "brier_score": None,
    }
    if risk_target is not None:
        metrics["risk_target"] = risk_target
    if holdout_confidence is not None:
        metrics["holdout_confidence"] = holdout_confidence
    if accepted_population is not None:
        metrics["accepted_population"] = accepted_population
    if review_missing_scores:
        metrics["scored"] = len(scored)
    if probability:
        metrics.update(
            METRICS["calibration"].compute(
                scored, bins=bins, allow_empty=review_missing_scores
            )
        )
    if extra_metrics:
        metrics["extra_metrics"] = {}
        for name in extra_metrics:
            if name in metrics["extra_metrics"]:
                raise ValueError(f"duplicate metric {name!r}")
            if name not in METRICS:
                raise ValueError(f"unknown metric {name!r}")
            metric = METRICS[name]
            if metric.requires_probability and not probability:
                raise ValueError(f"metric {name!r} requires --probability")
            metrics["extra_metrics"][name] = metric.compute(scored) if scored else None
    curve = threshold_points((row.score, not row.outcomes[0]) for row in scored)
    metrics.update(METRICS["ranking"].compute(scored, points=curve))
    if len(scored) != len(rows):
        curve = [
            (score, Counts(len(rows), counts.accepted, counts.errors))
            for score, counts in curve
        ]
    return metrics, curve
