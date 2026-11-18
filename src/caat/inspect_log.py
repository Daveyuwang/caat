"""Join external confidence scores to saved Inspect completions."""

import json
from hashlib import sha256
from itertools import chain, islice
from zipfile import BadZipFile

from .table import (
    atomic_output,
    open_prediction_table,
    write_external_score_join,
    write_rows,
)


IDENTITY = ("eval_id", "id", "epoch", "output_sha256")


def _read_json_log(path, epoch):
    from inspect_ai.log import read_eval_log

    # Read both views from the same file. The header preserves the declared
    # IDs, which Inspect fills from the samples on a full read if absent.
    with path.open("rb") as source:
        header = read_eval_log(source, header_only=True, format="json")
        source.seek(0)
        samples = read_eval_log(source, format="json").samples or ()

    declared = header.eval.dataset.sample_ids
    if declared is None:
        return header, samples
    seed_order = {str(item_id): position for position, item_id in enumerate(declared)}
    if len(seed_order) != len(declared):
        raise ValueError(f"{path}: duplicate sample ID in the Inspect log header")
    selected = [sample for sample in samples if sample.epoch == epoch]
    if epoch <= (header.eval.config.epochs or 1):
        missing = seed_order.keys() - {str(sample.id) for sample in selected}
        if missing:
            raise IndexError(f"sample {min(missing)!r} is missing from epoch {epoch}")
    return header, sorted(
        selected, key=lambda sample: seed_order.get(str(sample.id), len(seed_order))
    )


def _indexed_samples(
    path, header, epoch, *, include_attachments, exclude_fields, requested_ids
):
    from inspect_ai.log import (
        read_eval_log_sample_summaries,
        read_eval_log_samples,
        read_eval_log_samples_by_id,
    )

    available = dict.fromkeys(
        (str(sample.id), sample.epoch)
        for sample in read_eval_log_sample_summaries(path)
    )
    seeded = [
        (str(item_id), number)
        for item_id in header.eval.dataset.sample_ids
        for number in range(1, (header.eval.config.epochs or 1) + 1)
    ]
    seeded_keys = set(seeded)
    if not available or seeded_keys.difference(available):
        yield from read_eval_log_samples(path, exclude_fields=exclude_fields)
        return

    # Keep the seed order and include samples added by a running task.
    keys = chain(seeded, (key for key in available if key not in seeded_keys))
    requested = (key for key in keys if key[1] == epoch)
    if requested_ids and all(
        (item_id, epoch) in available for item_id in requested_ids
    ):
        requested = (key for key in requested if key[0] in requested_ids)
    # Limit how many retained prompts and media payloads are buffered at once.
    batch_size = 16 if include_attachments else 64
    while batch := list(islice(requested, batch_size)):
        samples = read_eval_log_samples_by_id(
            path, batch, exclude_fields=exclude_fields
        )
        for key, sample in zip(batch, samples, strict=True):
            if (str(sample.id), sample.epoch) != key:
                raise ValueError(
                    f"Inspect sample does not match its index entry {key!r}"
                )
            yield sample


def samples_from_log(path, epoch, *, include_attachments=False, requested_ids=None):
    try:
        from inspect_ai.log import read_eval_log, read_eval_log_samples
    except ModuleNotFoundError as exc:
        if exc.name != "inspect_ai":
            raise
        raise ValueError(
            "install caat-annotation[inspect] to read Inspect logs"
        ) from exc
    if epoch < 1:
        raise ValueError("epoch must be positive")
    excluded = {"events", "messages", "store", "error_retries"}
    if not include_attachments:
        excluded.add("attachments")
    seen = set()
    extension = path.suffix.lower()
    try:
        if extension == ".json":
            header, samples = _read_json_log(path, epoch)
        else:
            header = read_eval_log(path, header_only=True)
        if (
            header.status != "success"
            or getattr(header, "invalidated", False)
            or not header.eval.eval_id
        ):
            raise ValueError("expected a completed Inspect log with an evaluation ID")
        if extension != ".json":
            if header.eval.dataset.sample_ids is None:
                samples = read_eval_log(path, exclude_fields=excluded).samples or ()
            elif extension == ".eval":
                samples = _indexed_samples(
                    path,
                    header,
                    epoch,
                    include_attachments=include_attachments,
                    exclude_fields=excluded,
                    requested_ids=requested_ids,
                )
            else:
                samples = read_eval_log_samples(path, exclude_fields=excluded)
        for sample in samples:
            if sample.epoch != epoch:
                continue
            item_id = str(sample.id)
            if not item_id or item_id in seen:
                raise ValueError(
                    f"empty or duplicate sample ID in epoch {epoch}: {item_id!r}"
                )
            seen.add(item_id)
            if requested_ids is not None and item_id not in requested_ids:
                continue
            digest = sha256(sample.output.completion.encode("utf-8")).hexdigest()
            yield (header.eval.eval_id, item_id, str(epoch), digest), sample
    except (BadZipFile, EOFError, IndexError, KeyError) as exc:
        raise ValueError(f"could not read Inspect log {path}: {exc}") from exc
    if not seen:
        raise ValueError(f"no samples found in epoch {epoch}")


def export_for_scoring(log, output, *, epoch=1):
    if output.resolve() == log.resolve():
        raise ValueError("output would overwrite the Inspect log")

    def rows():
        for identity, sample in samples_from_log(log, epoch, include_attachments=True):
            prompt = sample.input
            if sample.attachments:
                from inspect_ai.log import resolve_sample_attachments

                prompt = resolve_sample_attachments(sample, "core").input
            if not isinstance(prompt, str):
                prompt = json.dumps(
                    [message.model_dump(mode="json") for message in prompt],
                    ensure_ascii=False,
                )
            yield (*identity, prompt, sample.output.completion)

    columns = (*IDENTITY, "input", "completion")
    if output.suffix.lower() != ".jsonl":
        return write_rows(output, columns, rows())
    count = 0
    with atomic_output(output, newline="\n") as handle:
        for row in rows():
            handle.write(json.dumps(dict(zip(columns, row)), ensure_ascii=False) + "\n")
            count += 1
    return count


def binary_grade(sample, grader):
    score = (sample.scores or {}).get(grader)
    if score is None:
        available = ", ".join(sorted(sample.scores or {})) or "none"
        raise ValueError(
            f"sample {sample.id!r} has no score for grader {grader!r}. "
            f"Scorers in this sample: {available}"
        )
    if score.value in ("C", 1):
        return 1
    if score.value in ("I", 0):
        return 0
    raise ValueError(f"sample {sample.id!r} has a non-binary score for {grader!r}")


def export_grades(log, predictions, output, *, grader, epoch=1):
    if output.resolve() in {log.resolve(), predictions.resolve()}:
        raise ValueError("output would overwrite an input file")
    expected = {}
    with open_prediction_table(predictions, set(IDENTITY), None) as (reader, _):
        for row in reader:
            item_id = row["id"]
            identity = tuple(row[column] for column in IDENTITY)
            if not all(identity) or item_id in expected:
                raise ValueError(
                    f"{predictions}: empty identity or duplicate id {item_id!r}"
                )
            expected[item_id] = identity

    def rows():
        for identity, sample in samples_from_log(
            log, epoch, requested_ids=set(expected)
        ):
            _, item_id, _, digest = identity
            if expected.pop(item_id) != identity:
                raise ValueError(
                    f"Inspect identity mismatch for {item_id!r} "
                    "(evaluation ID, epoch, or completion hash)"
                )
            yield item_id, digest, binary_grade(sample, grader)
        if expected:
            raise ValueError(
                f"missing {len(expected)} prediction IDs in the log, including {min(expected)!r}"
            )

    return write_rows(output, ("id", "output_sha256", "correct"), rows())


def join_scores(
    log,
    scores_file,
    output,
    *,
    graders,
    score_column="confidence",
    epoch=1,
    score_files=(),
    lower_is_better=(),
    review_missing_scores=False,
    scores_id_column=None,
):
    graders = tuple(graders)

    def rows():
        for identity, sample in samples_from_log(log, epoch):
            yield identity, (binary_grade(sample, name) for name in graders)

    try:
        return write_external_score_join(
            log,
            scores_file,
            output,
            rows(),
            identity_columns=IDENTITY,
            grade_columns=graders,
            score_column=score_column,
            score_files=score_files,
            lower_is_better=lower_is_better,
            review_missing_scores=review_missing_scores,
            score_id_column=scores_id_column,
        )
    except ValueError as exc:
        if scores_id_column is None and "identity columns" in str(exc):
            raise ValueError(
                f"{exc}, or match on the sample ID alone with --scores-id-column"
            ) from exc
        raise
