"""Load and align saved predictions and their grading outcomes."""

import csv
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class GradedRow:
    id: str
    score: float
    outcomes: tuple[bool, ...]
    group: str | None = None
    output_key: str | None = None
    reference: str | None = None


def validated_rows(rows, graders, *, minimum_graders=1):
    if len(graders) < minimum_graders or len(graders) != len(set(graders)) or not all(graders):
        raise ValueError(f"provide at least {minimum_graders} distinct grader names")
    rows = list(rows)
    if not rows:
        raise ValueError("at least one prediction is required")
    seen = set()
    for row in rows:
        if not row.id or row.id in seen:
            raise ValueError(f"empty or duplicate id: {row.id!r}")
        seen.add(row.id)
        if not isfinite(row.score):
            raise ValueError(f"score for id {row.id!r} must be finite")
        if len(row.outcomes) != len(graders) or any(type(bit) is not bool for bit in row.outcomes):
            raise ValueError(f"id {row.id!r} must have one boolean result per grader")
    return rows


@contextmanager
def open_predictions_csv(path: Path, required: set[str], group_column: str | None):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        header = reader.fieldnames or []
        if len(header) != len(set(header)):
            raise ValueError("CSV column names must be unique")
        if group_column is not None:
            required = required | {group_column}
        missing = sorted(required.difference(header))
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(missing)}. The file has: {', '.join(header)}")
        if group_column is None and "group" in header:
            group_column = "group"
        yield reader, group_column


def parse_score(value, line):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid score at CSV line {line}") from exc


def parse_correct(value, line):
    if value not in ("0", "1"):
        raise ValueError(f"correctness must be 0 or 1 at CSV line {line}")
    return value == "1"


def exact_match(item_id, prediction, reference):
    if not prediction or not reference:
        raise ValueError(f"id {item_id!r} needs a prediction and reference or a correctness result")
    return prediction == reference


def read_label_records(path, columns, group_column):
    with open_predictions_csv(path, set(columns.values()), group_column) as (reader, group_column):
        for row in reader:
            values = {name: row[column] for name, column in columns.items()}
            score = parse_score(values["score"], reader.line_num)
            correct = parse_correct(values["correct"], reader.line_num) if "correct" in columns else None
            group = (row[group_column] or None) if group_column is not None else None
            yield values, score, correct, group


def load_predictions(path, columns, group_column):
    for values, score, correct, group in read_label_records(path, columns, group_column):
        item_id, reference = values["id"], values.get("reference")
        if correct is None:
            correct = exact_match(item_id, values.get("prediction"), reference)
        yield GradedRow(item_id, score, (correct,), group, reference=reference)


@contextmanager
def open_selected_predictions(path, required, group_column, selection):
    if selection is not None:
        required = required | {selection[0]}
    with open_predictions_csv(path, required, group_column) as (reader, group_column):
        rows = reader
        if selection is not None:
            column, expected = selection
            rows = (row for row in reader if row[column] == expected)
        yield rows, reader, group_column


def read_graded_rows(path, id_column, score_column, graders, *, group_column=None, selection=None):
    required = {id_column, score_column, *graders}
    with open_selected_predictions(path, required, group_column, selection) as (rows, reader, group_column):
        return [
            GradedRow(
                row[id_column], parse_score(row[score_column], reader.line_num),
                tuple(parse_correct(row[name], reader.line_num) for name in graders),
                (row[group_column] or None) if group_column is not None else None,
            )
            for row in rows
        ], group_column


def read_design_scores(path, score_column):
    with open_predictions_csv(path, {score_column}, None) as (reader, _):
        return [parse_score(row[score_column], reader.line_num) for row in reader]


def join_grader_files(path, grader_files, *, id_column, score_column, output_key_column,
                      group_column=None, selection=None):
    required = {id_column, score_column, output_key_column}
    with open_selected_predictions(path, required, group_column, selection) as (rows, reader, group_column):
        predictions = {}
        for row in rows:
            item_id, key = row[id_column], row[output_key_column]
            if not item_id or item_id in predictions:
                raise ValueError(f"{path}: empty or duplicate id: {item_id!r}")
            if not key:
                raise ValueError(f"{path}: missing output key for {item_id!r}")
            predictions[item_id] = (
                parse_score(row[score_column], reader.line_num),
                key,
                (row[group_column] or None) if group_column is not None else None,
            )
    if not predictions:
        raise ValueError("at least one prediction is required")

    outcomes = []
    for name, grader_path in grader_files:
        bits = {}
        with open_predictions_csv(grader_path, {id_column, output_key_column, "correct"}, None) as (reader, _):
            for row in reader:
                item_id = row[id_column]
                if item_id not in predictions:
                    raise ValueError(f"{name}: unexpected id {item_id!r}")
                if item_id in bits:
                    raise ValueError(f"{name}: duplicate id {item_id!r}")
                if row[output_key_column] != predictions[item_id][1]:
                    raise ValueError(f"{name}: output key mismatch for {item_id!r}")
                bits[item_id] = parse_correct(row["correct"], reader.line_num)
        missing = predictions.keys() - bits.keys()
        if missing:
            raise ValueError(f"{name}: missing {len(missing)} ids, including {min(missing)!r}")
        outcomes.append(bits)
    return [
        GradedRow(item_id, score, tuple(bits[item_id] for bits in outcomes), group, key)
        for item_id, (score, key, group) in predictions.items()
    ], group_column
