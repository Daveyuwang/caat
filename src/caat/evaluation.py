from dataclasses import asdict, dataclass
from itertools import groupby
from math import isfinite
from operator import itemgetter
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class Prediction:
    id: str
    prediction: str
    reference: str
    score: float
    group: str | None = None


@dataclass(frozen=True)
class Counts:
    total: int
    accepted: int
    errors: int

    @property
    def coverage(self) -> float:
        return self.accepted / self.total

    @property
    def error_rate(self) -> float | None:
        return self.errors / self.accepted if self.accepted else None

    def metrics(self) -> dict:
        return {**asdict(self), "coverage": self.coverage, "error_rate": self.error_rate}


def threshold_points(scored_errors: Iterable[tuple[float, bool]]) -> list[tuple[float, Counts]]:
    """Count accepted rows and errors at each distinct score threshold."""
    ranked = sorted(scored_errors, key=itemgetter(0), reverse=True)
    accepted = errors = 0
    points = []
    for threshold, tied_rows in groupby(ranked, key=itemgetter(0)):
        for _, is_error in tied_rows:
            accepted += 1
            errors += is_error
        points.append((threshold, Counts(len(ranked), accepted, errors)))
    return points


def grouped_counts(
    decisions: Iterable[tuple[str | None, bool, bool]],
) -> tuple[Counts, dict[str, Counts]]:
    """Count decisions overall and by group. Each row has group, acceptance, and error."""
    total = accepted_total = errors_total = 0
    groups = {}
    for name, accepted, is_error in decisions:
        error = accepted and is_error
        total += 1
        accepted_total += accepted
        errors_total += error
        if name is not None:
            counts = groups.setdefault(name, [0, 0, 0])
            counts[0] += 1
            counts[1] += accepted
            counts[2] += error
    return Counts(total, accepted_total, errors_total), {
        name: Counts(*counts) for name, counts in groups.items()
    }


def evaluate(
    predictions: Iterable[Prediction], *, threshold: float, probability: bool = False
) -> tuple[dict, list[tuple[float, Counts]]]:
    """Return metrics at a fixed threshold and the attainable risk–coverage points."""
    if not isfinite(threshold):
        raise ValueError("threshold must be finite")
    rows = list(predictions)
    if not rows:
        raise ValueError("at least one prediction is required")

    seen = set()
    for row in rows:
        if not (row.id and row.prediction and row.reference):
            raise ValueError("id, prediction, and reference must be nonempty")
        if row.id in seen:
            raise ValueError(f"duplicate id: {row.id!r}")
        seen.add(row.id)
        if not isfinite(row.score) or (probability and not 0 <= row.score <= 1):
            expected = "between 0 and 1" if probability else "finite"
            raise ValueError(f"score for id {row.id!r} must be {expected}")

    errors = [row.prediction != row.reference for row in rows]
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
        "by_reference": {label: counts.metrics() for label, counts in by_reference.items()},
        "by_group": {group: counts.metrics() for group, counts in by_group.items()},
        "brier_score": (
            mean((row.score - (not error)) ** 2 for row, error in zip(rows, errors, strict=True))
            if probability else None
        ),
    }
    curve = threshold_points((row.score, error) for row, error in zip(rows, errors, strict=True))
    return metrics, curve
