"""Compute confidence scores from saved responses."""

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, exp, fsum, isfinite, log
from statistics import mean
from typing import Callable, Sequence
from unicodedata import normalize

from .registry import Registry
from .table import atomic_output


@dataclass(frozen=True)
class SavedResponse:
    text: str
    tokens: Sequence[dict] | None = None
    logprobs: Sequence[float] = ()
    answer_span: tuple[int, int] | None = None
    p_true: float | None = None
    verbalized_confidence: str | float | None = None
    samples: tuple[str, ...] | None = None
    clusters: tuple[str, ...] | None = None
    sample_logprobs: tuple[float, ...] | None = None


def token_logprobs(completion, item_id):
    scores = [token["logprob"] for token in completion["tokens"]]
    if not scores or any(
        type(score) not in (int, float) or not isfinite(score) or score > 0
        for score in scores
    ):
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
    if response.answer_span is None or response.tokens is None:
        return None
    start, end = response.answer_span
    if not "".join(token["text"] for token in response.tokens).startswith(
        response.text
    ):
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


WORD_CONFIDENCE = {
    "very low": 0.1,
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "very high": 0.9,
}


def parse_confidence(value):
    if isinstance(value, str):
        text = " ".join(value.casefold().split()).removeprefix("confidence:").strip()
        if text in WORD_CONFIDENCE:
            return WORD_CONFIDENCE[text]
        if not re.fullmatch(r"(?:\d+(?:\.\d+)?|\.\d+)\s*%?", text):
            raise ValueError(
                "confidence must be a probability, percentage, or documented word"
            )
        value = float(text[:-1]) / 100 if text.endswith("%") else float(text)
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError("confidence probability must be between 0 and 1")
    return float(value)


def sample_agreement(response):
    return max(Counter(response.samples).values()) / len(response.samples)


def normalized_sample_agreement(response):
    samples = (
        " ".join(normalize("NFKC", text).casefold().split()).strip(" .?!")
        for text in response.samples
    )
    return max(Counter(samples).values()) / len(response.samples)


def semantic_entropy(response):
    masses = Counter()
    if response.sample_logprobs is None:
        masses.update(response.clusters)
    else:
        anchor = max(response.sample_logprobs)
        for cluster, value in zip(
            response.clusters, response.sample_logprobs, strict=True
        ):
            masses[cluster] += exp(value - anchor)
    total = fsum(masses.values())
    probabilities = [mass / total for mass in masses.values() if mass]
    return fsum(-p * log(p) for p in probabilities)


@dataclass(frozen=True)
class Feature:
    compute: Callable[[SavedResponse], float | None]
    direction: str = "higher"
    requires: tuple[str, ...] = ("tokens",)
    cost: str = "no-extra-call"


_TOKEN_FEATURES = {
    "mean_token_logprob": Feature(mean_token_logprob),
    "sequence_logprob": Feature(sequence_logprob),
    "low_token_logprob": Feature(low_token_logprob),
    "token_count": Feature(token_count, "lower"),
    "shorter_response": Feature(shorter_response),
    "negative_mean_token_logprob": Feature(negative_mean_token_logprob),
    "answer_span_logprob": Feature(
        answer_span_logprob, requires=("text", "tokens", "answer_span")
    ),
}
DEFAULT_FEATURES = tuple(_TOKEN_FEATURES)
FEATURES = Registry(
    "features",
    Feature,
    {
        **_TOKEN_FEATURES,
        "p_true": Feature(
            lambda response: response.p_true,
            requires=("p_true",),
            cost="one-extra-call",
        ),
        "verbalized_confidence": Feature(
            lambda response: parse_confidence(response.verbalized_confidence),
            requires=("verbalized_confidence",),
            cost="one-extra-call",
        ),
        "sample_agreement": Feature(
            sample_agreement, requires=("samples",), cost="n-samples"
        ),
        "normalized_sample_agreement": Feature(
            normalized_sample_agreement, requires=("samples",), cost="n-samples"
        ),
        "semantic_entropy": Feature(
            semantic_entropy,
            direction="lower",
            requires=("clusters",),
            cost="n-samples",
        ),
    },
)

FULL_SCORE_FIELDS = (
    "sequence_logprob",
    "token_count",
    "low_token_logprob",
    "output_sha256",
)


def full_scores(response):
    return (
        sequence_logprob(response),
        token_count(response),
        low_token_logprob(response),
        sha256(response.text.encode("utf-8")).hexdigest(),
    )


def read_response(record):
    item_id = record["id"]
    text = record["text"]
    tokens = record.get("tokens")
    if (
        not isinstance(item_id, str)
        or not item_id
        or not isinstance(text, str)
        or (tokens is not None and not isinstance(tokens, list))
    ):
        raise ValueError("each response needs a nonempty string id and response text")
    if not all(
        isinstance(token, dict) and isinstance(token.get("text"), str)
        for token in (tokens or ())
    ):
        raise ValueError("each token needs text and a logprob")
    span = record.get("answer_span")
    if span is not None and (
        not isinstance(span, list)
        or len(span) != 2
        or any(type(index) is not int for index in span)
        or not 0 <= span[0] < span[1] <= len(text)
    ):
        raise ValueError("answer_span must be a valid half-open character range")
    samples = record.get("samples")
    clusters = record.get("clusters")
    for field, sequence in (("samples", samples), ("clusters", clusters)):
        if sequence is not None and (
            not isinstance(sequence, list)
            or not sequence
            or any(
                not isinstance(item, str) or (field == "clusters" and not item.strip())
                for item in sequence
            )
        ):
            raise ValueError(f"{field} must be a nonempty array of strings")
    if samples is not None and clusters is not None and len(samples) != len(clusters):
        raise ValueError("clusters must identify each sample")
    weights = record.get("sample_logprobs")
    if weights is not None and (
        clusters is None
        or not isinstance(weights, list)
        or len(weights) != len(clusters)
        or any(type(x) not in (int, float) or not isfinite(x) or x > 0 for x in weights)
    ):
        raise ValueError(
            "sample_logprobs must give one finite nonpositive value per cluster assignment"
        )
    p_true = record.get("p_true")
    if p_true is not None and (
        type(p_true) not in (int, float) or not 0 <= p_true <= 1
    ):
        raise ValueError("p_true must be a probability")
    return item_id, SavedResponse(
        text=text,
        tokens=tokens,
        logprobs=token_logprobs(record, item_id) if tokens is not None else (),
        answer_span=tuple(span) if span else None,
        p_true=p_true,
        verbalized_confidence=record.get("verbalized_confidence"),
        samples=tuple(samples) if samples is not None else None,
        clusters=tuple(clusters) if clusters is not None else None,
        sample_logprobs=tuple(weights) if weights is not None else None,
    )


def compute_feature(feature, response):
    if any(getattr(response, name) is None for name in feature.requires):
        return None
    score = feature.compute(response)
    if score is not None and not isfinite(score):
        raise ValueError("feature score must be finite")
    return score


def write_feature_csv(source, output, names, *, include_output_key=False):
    if source.resolve() == output.resolve():
        raise ValueError("output would overwrite the input JSONL file")
    names = tuple(names)
    identity = ("id", "output_sha256") if include_output_key else ("id",)
    columns = (*identity, *names)
    if len(set(columns)) != len(columns):
        raise ValueError(
            "feature names must be distinct from each other and the identity columns"
        )
    rows = {}
    with source.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item_id, response = read_response(json.loads(line))
                if item_id in rows:
                    raise ValueError(f"duplicate id {item_id!r}")
                key = (
                    (sha256(response.text.encode("utf-8")).hexdigest(),)
                    if include_output_key
                    else ()
                )
                rows[item_id] = (
                    *key,
                    *(compute_feature(FEATURES[name], response) for name in names),
                )
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"JSONL line {line_number}: {exc}") from exc
    if not rows:
        raise ValueError("at least one saved response is required")
    with atomic_output(output, newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows((item_id, *scores) for item_id, scores in rows.items())
    return len(rows)
