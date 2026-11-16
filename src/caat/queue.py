"""Prioritize saved predictions for review under a row budget."""

import csv
from collections import Counter
from random import Random

from .table import validated_rows


def review_queue(rows, *, budget, direction="higher", quotas=None, graders=(), seed=42):
    """Return a confidence-ordered queue, a random queue, and their results."""
    rows = validated_rows(rows, graders, minimum_graders=0)
    if type(budget) is not int or not 1 <= budget <= len(rows):
        raise ValueError("queue budget must be between 1 and the number of rows")
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be higher or lower")
    quotas = {} if quotas is None else dict(quotas)
    available = Counter(row.group for row in rows)
    for group, limit in quotas.items():
        if not group or group not in available:
            raise ValueError(f"quota names an unknown group: {group!r}")
        if type(limit) is not int or limit < 0:
            raise ValueError("quota limits must be nonnegative integers")
    if (
        sum(min(count, quotas.get(group, count)) for group, count in available.items())
        < budget
    ):
        raise ValueError("group quotas cannot fill the requested budget")

    def take(ordered):
        counts, selected = Counter(), []
        for row in ordered:
            if counts[row.group] < quotas.get(row.group, budget):
                selected.append(row)
                counts[row.group] += 1
                if len(selected) == budget:
                    break
        return selected, dict(
            (group, count) for group, count in counts.items() if group is not None
        )

    side = 1 if direction == "higher" else -1
    chosen, by_group = take(sorted(rows, key=lambda row: (side * row.score, row.id)))
    shuffled = sorted(rows, key=lambda row: row.id)
    Random(seed).shuffle(shuffled)
    random_rows, random_groups = take(shuffled)
    report = {
        "total": len(rows),
        "budget": budget,
        "direction": direction,
        "quotas": quotas,
        "by_group": by_group,
        "random_baseline": {"seed": seed, "by_group": random_groups},
        "graders": {
            name: {
                "errors_found": sum(not row.outcomes[i] for row in chosen),
                "random_errors_found": sum(not row.outcomes[i] for row in random_rows),
                "random_expected_errors": budget
                * sum(not row.outcomes[i] for row in rows)
                / len(rows)
                if not quotas
                else None,
            }
            for i, name in enumerate(graders)
        },
    }
    return chosen, random_rows, report


def write_queue(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("id", "score", "group"))
        writer.writerows((row.id, row.score, row.group) for row in rows)
