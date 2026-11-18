"""Prioritize saved predictions for review under a row budget."""

import csv
from collections import Counter
from random import Random

from .table import atomic_output, open_prediction_table, validated_rows


_QUEUE_COLUMNS = ("id", "score", "group")


def shuffled_by_id(rows, seed, *, key=None):
    """Shuffle after sorting by ID so the draw ignores input order."""
    ordered = sorted(rows, key=key)
    Random(seed).shuffle(ordered)
    return ordered


def review_queue(
    rows,
    *,
    budget,
    direction="higher",
    quotas=None,
    graders=(),
    seed=42,
    review_missing_scores=False,
):
    """Return a confidence-ordered queue, a random queue, and their results."""
    rows = validated_rows(
        rows, graders, minimum_graders=0, review_missing_scores=review_missing_scores
    )
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

    def priority(row):
        score = row.score
        return score is not None, side * score if score is not None else 0, row.id

    chosen, by_group = take(sorted(rows, key=priority))
    random_rows, random_groups = take(
        shuffled_by_id(rows, seed, key=lambda row: row.id)
    )
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


def read_sample_columns(
    path,
    rows,
    columns,
    *,
    id_column,
    output_key_column,
    fallback_id_column=None,
    fallback_output_key_column=None,
):
    """Read the requested fields for each sampled ID and output key."""
    if not all(columns) or len(set(columns)) != len(columns):
        raise ValueError("sample column names must be distinct and nonempty")
    if conflicts := set(columns) & {*_QUEUE_COLUMNS, "output_key"}:
        raise ValueError(
            f"sample columns already present: {', '.join(sorted(conflicts))}"
        )
    expected = {row.id: row.output_key for row in rows}
    if not all(expected.values()):
        raise ValueError("--sample-source needs output keys in the batch decisions")
    values = {}
    required = {
        *columns,
        *(name for name in (id_column, output_key_column) if name is not None),
    }
    with open_prediction_table(path, required, None) as (records, _):
        fields = records.fieldnames
        if id_column is None:
            id_column = "id" if "id" in fields else fallback_id_column or "id"
        if output_key_column is None:
            output_key_column = (
                "output_sha256"
                if "output_sha256" in fields
                else fallback_output_key_column or "output_sha256"
            )
        if missing := {id_column, output_key_column}.difference(fields):
            raise ValueError(
                f"{path}: missing sample source columns: {', '.join(sorted(missing))}"
            )
        for row in records:
            item_id = row[id_column]
            if expected.get(item_id) != row[output_key_column]:
                continue
            if item_id in values:
                raise ValueError(
                    f"{path}: multiple source rows for sampled ID {item_id!r} and its output key"
                )
            values[item_id] = tuple(row[name] for name in columns)
    if missing := expected.keys() - values.keys():
        raise ValueError(
            f"{path}: no source row matching the output key for sampled ID {min(missing)!r}"
        )
    return values, id_column, output_key_column


def write_queue(
    path, rows, *, include_output_key=False, extra_columns=(), extra_values=None
):
    columns = list(_QUEUE_COLUMNS)
    if include_output_key:
        columns.append("output_key")
    columns.extend(extra_columns)
    with atomic_output(path, newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        quoted = csv.writer(stream, lineterminator="\n", quoting=csv.QUOTE_ALL)
        writer.writerow(columns)
        for row in rows:
            fields = [row.id, row.score, row.group]
            if include_output_key:
                fields.append(row.output_key)
            if extra_values is not None:
                fields.extend(extra_values[row.id])
            # Older Python CSV writers need quoting for a bare carriage return.
            if any(isinstance(field, str) and "\r" in field for field in fields):
                quoted.writerow(fields)
            else:
                writer.writerow(fields)
