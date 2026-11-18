"""Read sample JSONL written by lm-evaluation-harness."""

import json
from hashlib import sha256
from math import isfinite

from .table import write_external_score_join, write_rows


IDENTITY = ("id", "filter", "prompt_hash", "output_sha256")


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def read_samples(path, sample_filter):
    seen = set()
    chosen = sample_filter
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid sample JSON at line {line_number}") from exc
            if not isinstance(sample, dict) or not isinstance(
                sample.get("filter"), str
            ):
                raise ValueError(
                    f"expected a sample with a filter at line {line_number}"
                )
            current = sample["filter"]
            if not current:
                raise ValueError(f"empty filter at line {line_number}")
            if chosen is None:
                chosen = current
            if current != chosen:
                if sample_filter is None:
                    raise ValueError(
                        "samples contain several filters. Select one with --filter"
                    )
                continue
            item_id = sample.get("doc_id")
            if type(item_id) is not int or item_id < 0 or item_id in seen:
                raise ValueError(
                    f"invalid or duplicate doc_id at line {line_number}: {item_id!r}"
                )
            seen.add(item_id)
            yield sample
    if not seen:
        raise ValueError(f"no samples for filter {chosen!r}")


def argument_records(sample):
    arguments = sample.get("arguments")
    if isinstance(arguments, dict):
        try:
            arguments = [
                arguments[f"gen_args_{index}"] for index in range(len(arguments))
            ]
            arguments = [
                [record[f"arg_{index}"] for index in range(len(record))]
                for record in arguments
            ]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"invalid arguments for doc_id {sample['doc_id']}"
            ) from exc
    if (
        not isinstance(arguments, list)
        or not arguments
        or any(
            not isinstance(record, list)
            or len(record) != 2
            or not isinstance(record[0], str)
            for record in arguments
        )
    ):
        raise ValueError(f"invalid arguments for doc_id {sample['doc_id']}")
    if digest(arguments[0][0]) != sample.get("prompt_hash"):
        raise ValueError(f"prompt hash mismatch for doc_id {sample['doc_id']}")
    return arguments


def grade(sample, name):
    value = sample.get(name)
    if value not in (0, 1):
        raise ValueError(f"doc_id {sample['doc_id']} needs a binary {name!r} result")
    return int(value)


def identity(sample, completion):
    return (
        sample["doc_id"],
        sample["filter"],
        sample["prompt_hash"],
        digest(completion),
    )


def generations(path, sample_filter):
    for sample in read_samples(path, sample_filter):
        arguments = argument_records(sample)
        resps = sample.get("resps")
        if (
            len(arguments) != 1
            or not isinstance(resps, list)
            or len(resps) != 1
            or not isinstance(resps[0], list)
            or len(resps[0]) != 1
            or not isinstance(resps[0][0], str)
        ):
            raise ValueError(f"doc_id {sample['doc_id']} needs one saved generation")
        completion = resps[0][0]
        yield identity(sample, completion), arguments[0][0], completion, sample


def export_generations(log, output, *, sample_filter=None):
    if output.resolve() == log.resolve():
        raise ValueError("output would overwrite the sample log")
    return write_rows(
        output,
        (*IDENTITY, "input", "completion"),
        (
            (*key, prompt, completion)
            for key, prompt, completion, _ in generations(log, sample_filter)
        ),
    )


def join_scores(
    log,
    scores_file,
    output,
    *,
    graders,
    score_column="confidence",
    sample_filter=None,
    score_files=(),
    lower_is_better=(),
    review_missing_scores=False,
):
    graders = tuple(graders)

    def rows():
        for key, _, _, sample in generations(log, sample_filter):
            yield key, (grade(sample, name) for name in graders)

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
    )


def choice_scores(log, output, *, sample_filter=None):
    if output.resolve() == log.resolve():
        raise ValueError("output would overwrite the sample log")

    def rows():
        for sample in read_samples(log, sample_filter):
            arguments = argument_records(sample)
            responses = sample.get("filtered_resps")
            item_id = sample["doc_id"]
            if (
                len(arguments) < 2
                or any(not isinstance(record[1], str) for record in arguments)
                or not isinstance(responses, list)
                or len(responses) != len(arguments)
                or any(
                    not isinstance(response, list) or len(response) != 2
                    for response in responses
                )
            ):
                raise ValueError(f"doc_id {item_id} needs per-choice log-likelihoods")
            try:
                scores = [float(response[0]) for response in responses]
                target = int(str(sample.get("target")))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid choice scores or target for doc_id {item_id}"
                ) from exc
            if not all(
                isfinite(score) and score <= 0 for score in scores
            ) or not 0 <= target < len(scores):
                raise ValueError(
                    f"invalid choice scores or target for doc_id {item_id}"
                )
            order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
            winner, second = order[:2]
            correct = grade(sample, "acc")
            if correct != (winner == target):
                raise ValueError(
                    f"acc disagrees with raw likelihood ranking for doc_id {item_id}"
                )
            completion = arguments[winner][1]
            yield (
                *identity(sample, completion),
                scores[winner],
                scores[winner] - scores[second],
                correct,
            )

    return write_rows(
        output, (*IDENTITY, "max_loglikelihood", "loglikelihood_margin", "acc"), rows()
    )
