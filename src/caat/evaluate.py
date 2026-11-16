"""Evaluate saved predictions under a frozen threshold."""

from math import isfinite
from typing import Iterable

from .metrics import (
    Counts,
    calibration_metrics,
    grouped_counts,
    ranking_metrics,
    threshold_points,
)
from .table import GradedRow, validated_rows


def evaluate(
    rows: Iterable[GradedRow],
    *,
    threshold: float,
    probability: bool = False,
    bins: int = 10,
) -> tuple[dict, list[tuple[float, Counts]]]:
    """Return metrics at a fixed threshold and the attainable risk–coverage points."""
    if not isfinite(threshold):
        raise ValueError("threshold must be finite")
    rows = validated_rows(rows, ("correct",))
    if probability:
        for row in rows:
            if not 0 <= row.score <= 1:
                raise ValueError(f"score for id {row.id!r} must be between 0 and 1")
    errors = [not row.outcomes[0] for row in rows]
    accepted = [row.score >= threshold for row in rows]
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
        "overall": overall.metrics(),
        "by_reference": {
            label: counts.metrics() for label, counts in by_reference.items()
        },
        "by_group": {group: counts.metrics() for group, counts in by_group.items()},
        "brier_score": None,
    }
    if probability:
        metrics.update(
            calibration_metrics(
                (row.score for row in rows), (not error for error in errors), bins=bins
            )
        )
    curve = threshold_points(
        (row.score, error) for row, error in zip(rows, errors, strict=True)
    )
    return {**metrics, **ranking_metrics(curve)}, curve
