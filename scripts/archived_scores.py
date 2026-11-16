"""Compute confidence features and output identities from archived completions."""

from hashlib import sha256
from math import ceil, fsum, isfinite
from statistics import mean


def token_logprobs(completion, item_id):
    scores = [token["logprob"] for token in completion["tokens"]]
    if not scores or any(not isfinite(score) or score > 0 for score in scores):
        raise ValueError(f"Invalid token log probabilities for item {item_id}")
    return scores


FULL_SCORE_FIELDS = ("sequence_logprob", "token_count", "low_token_logprob", "output_sha256")


def full_scores(completion, scores):
    bottom_count = ceil(len(scores) / 10)
    return (
        fsum(scores),
        len(scores),
        mean(sorted(scores)[:bottom_count]),
        sha256(completion["text"].encode("utf-8")).hexdigest(),
    )
