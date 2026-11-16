"""Counts and ranking metrics for saved prediction policies."""

from dataclasses import asdict, dataclass
from itertools import groupby
from math import fsum
from operator import itemgetter
from statistics import mean
from typing import Iterable


def probability_bin(probability, bins):
    return min(int(probability * bins), bins - 1)


def calibration_metrics(probabilities, outcomes, *, bins=10):
    """Measure binary correctness calibration. Keep tied scores together in ACE."""
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    pairs = list(zip(probabilities, outcomes, strict=True))
    if not pairs or any(not 0 <= p <= 1 for p, _ in pairs):
        raise ValueError("calibration metrics need probabilities between 0 and 1")
    fixed = [[] for _ in range(bins)]
    adaptive = [[] for _ in range(min(bins, len(pairs)))]
    for pair in pairs:
        fixed[probability_bin(pair[0], bins)].append(pair)
    seen = 0
    for _, tied in groupby(sorted(pairs, key=itemgetter(0)), key=itemgetter(0)):
        block = list(tied)
        index = (2 * seen + len(block)) * len(adaptive) // (2 * len(pairs))
        adaptive[index].extend(block)
        seen += len(block)

    def summarize(group):
        count = len(group)
        return {
            "count": count,
            "mean_probability": fsum(p for p, _ in group) / count if count else None,
            "accuracy": sum(y for _, y in group) / count if count else None,
        }

    reliability = [
        {"lower": i / bins, "upper": (i + 1) / bins, **summarize(group)}
        for i, group in enumerate(fixed)
    ]
    adaptive_gaps = [
        abs(fsum(p - y for p, y in group)) / len(group) for group in adaptive if group
    ]
    return {
        "brier_score": mean((p - y) ** 2 for p, y in pairs),
        "ece": fsum(abs(fsum(p - y for p, y in group)) for group in fixed) / len(pairs),
        "ace": fsum(adaptive_gaps) / len(adaptive_gaps),
        "reliability_bins": reliability,
    }


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
        return {
            **asdict(self),
            "coverage": self.coverage,
            "error_rate": self.error_rate,
        }


def threshold_points(
    scored_errors: Iterable[tuple[float, bool]],
) -> list[tuple[float, Counts]]:
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


def ranking_metrics(points: list[tuple[float, Counts]]) -> dict:
    """Use the expected prefix risk under uniform tie-breaking."""
    total = points[-1][1]
    correct = total.total - total.errors
    previous_accepted = previous_errors = 0
    pair_wins = []
    prefix_risks = []
    for _, counts in points:
        size = counts.accepted - previous_accepted
        tied_errors = counts.errors - previous_errors
        pair_wins.append(
            (size - tied_errors) * (total.errors - counts.errors + tied_errors / 2)
        )
        prefix_risks.extend(
            (previous_errors + tied_errors * i / size) / (previous_accepted + i)
            for i in range(1, size + 1)
        )
        previous_accepted = counts.accepted
        previous_errors = counts.errors
    aurc = fsum(prefix_risks) / total.total
    gain = None
    if correct and total.errors:
        random_aurc = total.errors / total.total
        oracle_aurc = (
            fsum((k - correct) / k for k in range(correct + 1, total.total + 1))
            / total.total
        )
        gain = (random_aurc - aurc) / (random_aurc - oracle_aurc)
    return {
        "auroc": fsum(pair_wins) / (correct * total.errors)
        if correct and total.errors
        else None,
        "aurc": aurc,
        "normalized_aurc_gain": gain,
    }


def risk_at_count(points, target):
    """Expected prefix risk when the budget ends inside a tie."""
    if target < 1:
        raise ValueError("coverage budget must accept at least one row")
    previous_accepted = previous_errors = 0
    for _, counts in points:
        if counts.accepted >= target:
            share = (target - previous_accepted) / (counts.accepted - previous_accepted)
            return (
                previous_errors + share * (counts.errors - previous_errors)
            ) / target
        previous_accepted, previous_errors = counts.accepted, counts.errors
    raise ValueError("coverage budget exceeds the row count")


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
