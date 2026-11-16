"""Compute scores from a saved response and its token log probabilities."""

import csv
import json
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, fsum, isfinite
from statistics import mean
from typing import Callable, Sequence


@dataclass(frozen=True)
class SavedResponse:
    text: str
    tokens: Sequence[dict]
    logprobs: Sequence[float]
    answer_span: tuple[int, int] | None = None


def token_logprobs(completion, item_id):
    scores = [token["logprob"] for token in completion["tokens"]]
    if not scores or any(type(score) not in (int, float) or not isfinite(score) or score > 0 for score in scores):
        raise ValueError(f"Invalid token log probabilities for item {item_id}")
    return scores


def mean_token_logprob(response):
    return mean(response.logprobs)


def sequence_logprob(response):
    return fsum(response.logprobs)


def low_token_logprob(response):
    size = ceil(len(response.logprobs) / 10)
    return mean(sorted(response.logprobs)[:size])


def token_count(response):
    return len(response.logprobs)


def shorter_response(response):
    return -token_count(response)


def negative_mean_token_logprob(response):
    return -mean_token_logprob(response)


def answer_span_logprob(response):
    if response.answer_span is None:
        return None
    start, end = response.answer_span
    if not "".join(token["text"] for token in response.tokens).startswith(response.text):
        return None
    offset = 0
    overlaps = []
    for token, score in zip(response.tokens, response.logprobs, strict=True):
        boundary = offset + len(token["text"])
        if offset < end and boundary > start and boundary > offset:
            overlaps.append(score)
        offset = boundary
        if offset >= end:
            break
    return fsum(overlaps) if overlaps else None


@dataclass(frozen=True)
class Feature:
    compute: Callable[[SavedResponse], float | None]
    direction: str = "higher"
    requires: tuple[str, ...] = ("tokens",)


FEATURES = {
    "mean_token_logprob": Feature(mean_token_logprob),
    "sequence_logprob": Feature(sequence_logprob),
    "low_token_logprob": Feature(low_token_logprob),
    "token_count": Feature(token_count, "lower"),
    "shorter_response": Feature(shorter_response),
    "negative_mean_token_logprob": Feature(negative_mean_token_logprob),
    "answer_span_logprob": Feature(answer_span_logprob, requires=("text", "tokens", "answer_span")),
}

FULL_SCORE_FIELDS = ("sequence_logprob", "token_count", "low_token_logprob", "output_sha256")


def full_scores(response):
    return (
        sequence_logprob(response), token_count(response), low_token_logprob(response),
        sha256(response.text.encode("utf-8")).hexdigest(),
    )


def read_response(record):
    item_id = record["id"]
    text = record["text"]
    tokens = record["tokens"]
    if not isinstance(item_id, str) or not item_id or not isinstance(text, str) or not isinstance(tokens, list):
        raise ValueError("each response needs a nonempty string id, text, and a token array")
    if not all(isinstance(token, dict) and isinstance(token.get("text"), str) for token in tokens):
        raise ValueError("each token needs text and a logprob")
    span = record.get("answer_span")
    if span is not None and (
        not isinstance(span, list) or len(span) != 2 or any(type(index) is not int for index in span)
        or not 0 <= span[0] < span[1] <= len(text)
    ):
        raise ValueError("answer_span must be a valid half-open character range")
    return item_id, SavedResponse(text, tokens, token_logprobs(record, item_id), tuple(span) if span else None)


def write_feature_csv(source, output, names):
    if source.resolve() == output.resolve():
        raise ValueError("output would overwrite the input JSONL file")
    if len(set(names)) != len(names):
        raise ValueError("feature names must be distinct")
    rows = {}
    with source.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item_id, response = read_response(json.loads(line))
                if item_id in rows:
                    raise ValueError(f"duplicate id {item_id!r}")
                rows[item_id] = tuple(FEATURES[name].compute(response) for name in names)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"JSONL line {line_number}: {exc}") from exc
    if not rows:
        raise ValueError("at least one saved response is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("id", *names))
        writer.writerows((item_id, *scores) for item_id, scores in rows.items())
    return len(rows)
