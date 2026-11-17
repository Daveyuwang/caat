"""Load and align saved predictions and their grading outcomes."""

import csv
import json
from contextlib import closing, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from itertools import chain
from math import isfinite
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import ClassVar, Iterable


@dataclass(frozen=True)
class GradedRow:
    id: str
    score: float
    outcomes: tuple[bool, ...]
    group: str | None = None
    output_key: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class UnscoredRow:
    id: str
    outcomes: tuple[bool, ...]
    group: str | None = None
    output_key: str | None = None
    reference: str | None = None
    score: ClassVar[None] = None


def graded_row(item_id, score, outcomes, group=None, output_key=None, reference=None):
    if score is None:
        return UnscoredRow(item_id, outcomes, group, output_key, reference)
    return GradedRow(item_id, score, outcomes, group, output_key, reference)


def accepted_at(row, threshold):
    score = row.score
    return score is not None and score >= threshold


def output_fingerprint(item_id, output_key):
    if not item_id or not output_key:
        raise ValueError(f"sample {item_id!r} needs a nonempty ID and output key")
    pair = json.dumps((item_id, output_key), separators=(",", ":"))
    return sha256(pair.encode("utf-8")).hexdigest()


def output_fingerprints(rows):
    return {output_fingerprint(row.id, row.output_key): row.id for row in rows}


def validated_rows(rows, graders, *, minimum_graders=1, review_missing_scores=False):
    return list(
        iter_validated_rows(
            rows,
            graders,
            minimum_graders=minimum_graders,
            review_missing_scores=review_missing_scores,
        )
    )


def iter_validated_rows(
    rows, graders, *, minimum_graders=1, review_missing_scores=False
):
    """Validate and yield each prediction while tracking unique IDs."""
    if (
        len(graders) < minimum_graders
        or len(graders) != len(set(graders))
        or not all(graders)
    ):
        raise ValueError(f"provide at least {minimum_graders} distinct grader names")
    seen = set()
    for row in rows:
        if not row.id or row.id in seen:
            raise ValueError(f"empty or duplicate id: {row.id!r}")
        seen.add(row.id)
        score = row.score
        if not (review_missing_scores if score is None else isfinite(score)):
            raise ValueError(f"score for id {row.id!r} must be finite")
        if len(row.outcomes) != len(graders) or any(
            type(bit) is not bool for bit in row.outcomes
        ):
            raise ValueError(f"id {row.id!r} must have one boolean result per grader")
        yield row
    if not seen:
        raise ValueError("at least one prediction is required")


class _CsvRows:
    """Read CSV records with one field per header column."""

    def __init__(self, source):
        self._name = source.name
        self._reader = csv.reader(source, strict=True)
        self._records = self._read_records()
        self.fieldnames = next(self._records, [])

    @property
    def line_num(self):
        return self._reader.line_num

    def _read_records(self):
        try:
            yield from filter(None, self._reader)
        except csv.Error as exc:
            raise ValueError(f"{self._name}: CSV line {self.line_num}: {exc}") from exc

    def __iter__(self):
        width = len(self.fieldnames)
        for fields in self._records:
            if len(fields) != width:
                raise ValueError(
                    f"{self._name}: CSV record ending at line {self.line_num} "
                    f"has {len(fields)} fields. Expected {width} from the header."
                )
            yield dict(zip(self.fieldnames, fields))


def _unique_json_fields(pairs):
    record = dict(pairs)
    if len(record) != len(pairs):
        raise ValueError("duplicate field name")
    return record


def _reject_json_constant(value):
    raise ValueError(f"invalid JSON number {value}")


class _JsonlRows:
    """Provide the fields and line numbers used by the CSV table readers."""

    def __init__(self, source):
        self.line_num = 0
        remaining = self.read_rows(source)
        first = next(remaining, None)
        self.empty = first is None
        self.fieldnames = list(first) if first is not None else []
        self.fields = set(self.fieldnames)
        self.rows = chain(() if first is None else (first,), remaining)

    def read_rows(self, source):
        for self.line_num, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    parse_int=str,
                    parse_float=str,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_unique_json_fields,
                )
                if not isinstance(row, dict):
                    raise ValueError("expected an object")
                for name, value in row.items():
                    if value is None:
                        row[name] = ""
                    elif type(value) is bool:
                        row[name] = "1" if value else "0"
                    elif not isinstance(value, str):
                        raise ValueError(f"field {name!r} needs a scalar value")
            except (ValueError, RecursionError) as exc:
                raise ValueError(f"JSONL line {self.line_num}: {exc}") from exc
            yield row

    def __iter__(self):
        for row in self.rows:
            if row.keys() != self.fields:
                missing = sorted(self.fields - row.keys())
                extra = sorted(row.keys() - self.fields)
                raise ValueError(
                    f"JSONL line {self.line_num} differs from the first object's fields "
                    f"(missing {missing}, extra {extra})"
                )
            yield row


@contextmanager
def open_prediction_table(
    path: Path, required: set[str], group_column: str | None, *, allow_empty_jsonl=False
):
    path = Path(path)
    jsonl = path.suffix.lower() == ".jsonl"
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = _JsonlRows(source) if jsonl else _CsvRows(source)
        header = reader.fieldnames or []
        if len(header) != len(set(header)):
            raise ValueError("CSV column names must be unique")
        if group_column is not None:
            required = required | {group_column}
        missing = sorted(required.difference(header))
        if missing and not (
            allow_empty_jsonl and isinstance(reader, _JsonlRows) and reader.empty
        ):
            kind = "JSONL" if jsonl else "CSV"
            raise ValueError(
                f"missing {kind} columns: {', '.join(missing)}. The file has: {', '.join(header)}"
            )
        if group_column is None and "group" in header:
            group_column = "group"
        yield reader, group_column


def parse_score(value, line, *, negate=False, review_missing_scores=False):
    if review_missing_scores and value == "":
        return None
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid score at input line {line}") from exc
    return -score if negate else score


@contextmanager
def atomic_output(output, *, newline):
    """Write beside the destination and replace it after a successful write."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", dir=output.parent, encoding="utf-8", newline=newline, delete=False
        ) as handle:
            temporary = Path(handle.name)
            yield handle
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_rows(output, columns, rows):
    """Replace a CSV after every input row has been validated."""
    with atomic_output(output, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        count = 0
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_holdout_rows(output, source_rows, heldout, *, id_column, grader_columns=None):
    ids = {row.id for row in heldout}
    columns = tuple(source_rows[0])
    grader_columns = grader_columns or {}
    if overlap := grader_columns.keys() & set(columns):
        raise ValueError(
            f"holdout grader columns already exist: {', '.join(sorted(overlap))}"
        )
    return write_rows(
        output,
        (*columns, *grader_columns),
        (
            (
                *(source[column] for column in columns),
                *(int(bits[source[id_column]]) for bits in grader_columns.values()),
            )
            for source in source_rows
            if source[id_column] in ids
        ),
    )


def read_external_scores(path, score_column, columns, *, review_missing_scores=False):
    *key_columns, digest_column = columns
    scores = {}
    with open_prediction_table(
        path, {*columns, score_column}, None, allow_empty_jsonl=review_missing_scores
    ) as (reader, _):
        for row in reader:
            identity = tuple(row[column] for column in key_columns)
            if not all(identity) or identity in scores:
                raise ValueError(f"empty or duplicate score identity: {identity!r}")
            score = parse_score(
                row[score_column],
                reader.line_num,
                review_missing_scores=review_missing_scores,
            )
            if score is not None and not isfinite(score):
                raise ValueError(f"score for sample {row['id']!r} must be finite")
            scores[identity] = row[digest_column], score
    return scores


def negated_score_name(name):
    return f"negative_{name}"


def negated_score_columns(names, requested):
    """Return the positions and names of the requested confidence columns."""
    names = tuple(names)
    requested = set(requested)
    if unknown := requested.difference(names):
        raise ValueError(
            f"--lower-is-better needs an external score column: {', '.join(sorted(unknown))}"
        )
    return tuple(
        (index, negated_score_name(name))
        for index, name in enumerate(names)
        if name in requested
    )


def write_external_score_join(
    log,
    scores_file,
    output,
    samples,
    *,
    identity_columns,
    grade_columns,
    score_column,
    score_files=(),
    lower_is_better=(),
    review_missing_scores=False,
):
    """Verify returned scores against the saved outputs and write the joined CSV."""
    sources = tuple(score_files)
    if scores_file is not None:
        sources = ((score_column, scores_file), *sources)
    if output.resolve() in {log.resolve(), *(path.resolve() for _, path in sources)}:
        raise ValueError("output would overwrite an input file")
    names = tuple(name for name, _ in sources)
    negated = negated_score_columns(names, lower_is_better)
    columns = (
        *identity_columns,
        *names,
        *(name for _, name in negated),
        *grade_columns,
    )
    if (
        not sources
        or not grade_columns
        or not all(columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("provide distinct grader, score, and identity column names")

    tables = []
    for name, path in sources:
        try:
            scores = read_external_scores(
                path,
                name,
                identity_columns,
                review_missing_scores=review_missing_scores,
            )
        except ValueError as exc:
            raise ValueError(f"{path} ({name}): {exc}") from exc
        tables.append((name, scores))
    id_position = identity_columns.index("id")

    def rows():
        for identity, grades in samples:
            key = tuple(map(str, identity[:-1]))
            item = identity[id_position]
            values = []
            for name, scores in tables:
                stored = scores.pop(key, None)
                if stored is None:
                    if not review_missing_scores:
                        raise ValueError(f"{name}: missing score for sample {item!r}")
                    values.append(None)
                    continue
                digest, score = stored
                if digest != identity[-1]:
                    raise ValueError(f"{name}: output key mismatch for sample {item!r}")
                values.append(score)
            yield (
                *identity,
                *values,
                *(
                    -values[index] if values[index] is not None else None
                    for index, _ in negated
                ),
                *grades,
            )
        for name, scores in tables:
            if scores:
                raise ValueError(
                    f"{name}: score file has an unexpected sample identity: {next(iter(scores))!r}"
                )

    with closing(samples):
        return write_rows(output, columns, rows())


def parse_correct(value, line):
    if value not in ("0", "1"):
        raise ValueError(f"correctness must be 0 or 1 at input line {line}")
    return value == "1"


def exact_match(item_id, prediction, reference):
    if not prediction or not reference:
        raise ValueError(
            f"id {item_id!r} needs a prediction and reference or a correctness result"
        )
    return prediction == reference


def parse_prediction_filter(value):
    if isinstance(value, str):
        column, separator, expected = value.partition("=")
        if column and separator:
            return column, expected
    raise ValueError("prediction_filter must use COLUMN=VALUE")


@contextmanager
def open_selected_predictions(path, required, group_column, selection):
    if selection is not None:
        required = required | {selection[0]}
    with open_prediction_table(path, required, group_column) as (reader, group_column):
        rows = reader
        if selection is not None:
            column, expected = selection
            rows = (row for row in reader if row[column] == expected)
        yield rows, reader, group_column


def orient_score_table(
    source, output, requested, *, selection=None, review_missing_scores=False
):
    """Keep the input table and append negations of the named scores."""
    if source.resolve() == output.resolve():
        raise ValueError("output would overwrite the score input")
    requested = set(requested)
    with open_selected_predictions(source, requested, None, selection) as (
        selected,
        reader,
        _,
    ):
        columns = tuple(reader.fieldnames)
        negated = negated_score_columns(columns, requested)
        added = tuple(name for _, name in negated)
        if collisions := set(added).intersection(columns):
            raise ValueError(
                f"derived score columns already exist: {', '.join(sorted(collisions))}"
            )

        def rows():
            for row in selected:
                scores = []
                for index, _ in negated:
                    column = columns[index]
                    try:
                        value = parse_score(
                            row[column],
                            reader.line_num,
                            negate=True,
                            review_missing_scores=review_missing_scores,
                        )
                    except ValueError as exc:
                        raise ValueError(f"{source} ({column}): {exc}") from exc
                    if value is not None and not isfinite(value):
                        raise ValueError(
                            f"{source}: score {column!r} must be finite at input line {reader.line_num}"
                        )
                    scores.append(value)
                yield (*(row[name] for name in columns), *scores)

        return write_rows(output, (*columns, *added), rows())


@contextmanager
def open_table_rows(
    path,
    *,
    id_column="id",
    score_column="score",
    correct_columns=(),
    group_column=None,
    output_key_column=None,
    prediction_column=None,
    reference_column=None,
    selection=None,
    require_labels=True,
    source_rows=None,
    negate_score=False,
    review_missing_scores=False,
):
    """Yield parsed rows and the resolved group column while the input is open."""
    if not correct_columns and require_labels:
        prediction_column = prediction_column or "prediction"
        reference_column = reference_column or "reference"
    required = {id_column, score_column, *correct_columns}
    required.update(
        column
        for column in (prediction_column, reference_column, output_key_column)
        if column
    )
    path = Path(path)
    make_row = graded_row if review_missing_scores else GradedRow
    with open_selected_predictions(path, required, group_column, selection) as (
        source,
        reader,
        group_column,
    ):

        def rows():
            for row in source:
                item_id = row[id_column]
                score = parse_score(
                    row[score_column],
                    reader.line_num,
                    negate=negate_score,
                    review_missing_scores=review_missing_scores,
                )
                reference = (
                    row[reference_column] if reference_column is not None else None
                )
                group = (
                    (row[group_column] or None) if group_column is not None else None
                )
                key = row[output_key_column] if output_key_column is not None else None
                if output_key_column is not None and not key:
                    raise ValueError(f"{path}: missing output key for {item_id!r}")
                if correct_columns:
                    outcomes = tuple(
                        parse_correct(row[column], reader.line_num)
                        for column in correct_columns
                    )
                elif require_labels:
                    outcomes = (
                        exact_match(item_id, row[prediction_column], reference),
                    )
                else:
                    outcomes = ()
                if source_rows is not None:
                    source_rows.append(row)
                yield make_row(item_id, score, outcomes, group, key, reference)

        yield rows(), group_column


def read_table_rows(path, **options):
    with open_table_rows(path, **options) as (rows, group_column):
        return list(rows), group_column


def load_csv(
    path: str | PathLike[str],
    *,
    id_column: str = "id",
    score_column: str = "score",
    correct_columns: Iterable[str] = (),
    group_column: str | None = None,
    output_key_column: str | None = None,
    prediction_column: str | None = None,
    reference_column: str | None = None,
) -> list[GradedRow]:
    """Read CSV or JSONL, using correctness columns or exact label matching.

    Outcome order follows correct_columns. The default exact match uses
    prediction and reference columns and produces one outcome per row.
    """
    columns = tuple(correct_columns)
    rows, _ = read_table_rows(
        path,
        id_column=id_column,
        score_column=score_column,
        correct_columns=columns,
        group_column=group_column,
        output_key_column=output_key_column,
        prediction_column=prediction_column,
        reference_column=reference_column,
    )
    return validated_rows(rows, columns or ("correct",))


def read_design_scores(
    path,
    score_column,
    *,
    calibration,
    id_column,
    output_key_column=None,
    selection=None,
    negate_score=False,
    review_missing_scores=False,
):
    with open_selected_predictions(path, {score_column}, None, selection) as (
        rows,
        reader,
        _,
    ):
        calibration_keys = (
            {row.id: row.output_key for row in calibration}
            if id_column in reader.fieldnames
            else {}
        )
        scores = []
        for row in rows:
            item_id = row.get(id_column)
            if item_id in calibration_keys:
                design_key = row.get(output_key_column)
                calibration_key = calibration_keys[item_id]
                if (
                    not design_key
                    or not calibration_key
                    or design_key == calibration_key
                ):
                    raise ValueError(
                        f"{path}: design and calibration reuse sample ID {item_id!r}"
                    )
            scores.append(
                parse_score(
                    row[score_column],
                    reader.line_num,
                    negate=negate_score,
                    review_missing_scores=review_missing_scores,
                )
            )
        return scores


def read_grader_file(
    path,
    name,
    output_keys,
    *,
    id_column,
    output_key_column,
    require_all=True,
    allow_extra=False,
):
    bits = {}
    other_versions = set()
    required = {id_column, output_key_column, "correct"}
    with open_prediction_table(
        path, required, None, allow_empty_jsonl=not require_all
    ) as (reader, _):
        for row in reader:
            item_id = row[id_column]
            if item_id not in output_keys:
                if allow_extra:
                    continue
                raise ValueError(f"{name}: unexpected id {item_id!r}")
            matches = row[output_key_column] == output_keys[item_id]
            if allow_extra and not matches:
                if require_all:
                    other_versions.add(item_id)
                continue
            if item_id in bits:
                raise ValueError(f"{name}: duplicate id {item_id!r}")
            if not matches:
                raise ValueError(f"{name}: output key mismatch for {item_id!r}")
            bits[item_id] = parse_correct(row["correct"], reader.line_num)
    if unmatched := other_versions - bits.keys():
        raise ValueError(f"{name}: output key mismatch for {min(unmatched)!r}")
    if require_all and (missing := output_keys.keys() - bits.keys()):
        raise ValueError(
            f"{name}: missing {len(missing)} ids, including {min(missing)!r}"
        )
    return bits


def join_grader_files(
    path,
    grader_files,
    *,
    id_column,
    score_column,
    output_key_column,
    group_column=None,
    selection=None,
    grader_id_column=None,
    grader_output_key_column=None,
    allow_extra_grades=False,
    check_columns=(),
    reference_column=None,
    source_rows=None,
    review_missing_scores=False,
    negate_score=False,
):
    required = {id_column, score_column, output_key_column, *check_columns}
    if reference_column is not None:
        required.add(reference_column)
    with open_selected_predictions(path, required, group_column, selection) as (
        rows,
        reader,
        group_column,
    ):
        predictions = {}
        for row in rows:
            item_id, key = row[id_column], row[output_key_column]
            if not item_id or item_id in predictions:
                raise ValueError(f"{path}: empty or duplicate id: {item_id!r}")
            if not key:
                raise ValueError(f"{path}: missing output key for {item_id!r}")
            predictions[item_id] = (
                parse_score(
                    row[score_column],
                    reader.line_num,
                    negate=negate_score,
                    review_missing_scores=review_missing_scores,
                ),
                key,
                (row[group_column] or None) if group_column is not None else None,
                tuple(
                    parse_correct(row[name], reader.line_num) for name in check_columns
                ),
                row[reference_column] if reference_column is not None else None,
            )
            if source_rows is not None:
                source_rows.append(row)
    if not predictions:
        raise ValueError("at least one prediction is required")

    output_keys = {item: key for item, (_, key, _, _, _) in predictions.items()}
    outcomes = [
        read_grader_file(
            grader_path,
            name,
            output_keys,
            id_column=grader_id_column or id_column,
            output_key_column=grader_output_key_column or output_key_column,
            allow_extra=allow_extra_grades,
        )
        for name, grader_path in grader_files
    ]
    make_row = graded_row if review_missing_scores else GradedRow
    return [
        make_row(
            item_id,
            score,
            tuple(bits[item_id] for bits in outcomes) + checks,
            group,
            key,
            reference,
        )
        for item_id, (score, key, group, checks, reference) in predictions.items()
    ], group_column
