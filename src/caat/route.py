"""Apply a frozen cutoff to scored predictions."""

import json
from hashlib import sha256
from math import inf, isfinite
from typing import NamedTuple

from .queue import shuffled_by_id
from .table import (
    GradedRow,
    open_prediction_table,
    open_scored_predictions,
    parse_score,
    write_rows,
)


class BatchDecision(NamedTuple):
    id: str
    score: float | None
    accepted: bool
    group: str | None
    output_key: str | None


def iter_batch_decisions(path, *, threshold=None, require_output_key=False):
    """Yield ID, score, acceptance, group, and output key in decision order."""
    if threshold is not None and not isfinite(threshold):
        raise ValueError("threshold must be finite")
    required = {"id", "score", "decision"}
    if require_output_key:
        required.add("output_key")
    seen = set()
    min_accept, max_review = inf, -inf
    with open_prediction_table(path, required, None) as (reader, _):
        keyed = "output_key" in reader.fieldnames
        for row in reader:
            item_id = row["id"]
            if not item_id or item_id in seen:
                raise ValueError(f"{path}: empty or duplicate id: {item_id!r}")
            seen.add(item_id)
            score = parse_score(row["score"], reader.line_num) if row["score"] else None
            if score is not None and not isfinite(score):
                raise ValueError(f"{path}: score for id {item_id!r} must be finite")
            decision = row["decision"]
            if decision == "accept":
                if score is None:
                    raise ValueError(
                        f"{path}: accepted prediction {item_id!r} needs a score"
                    )
                min_accept = min(min_accept, score)
            elif decision != "review":
                raise ValueError(
                    f"{path}: invalid decision for id {item_id!r}: {decision!r}"
                )
            elif score is not None:
                max_review = max(max_review, score)
            key = row.get("output_key")
            if keyed and not key:
                raise ValueError(f"{path}: missing output key for {item_id!r}")
            yield item_id, score, decision == "accept", row.get("group") or None, key
    if max_review >= min_accept:
        raise ValueError(f"{path}: decisions are inconsistent with a single cutoff")
    if threshold is not None and not max_review < threshold <= min_accept:
        raise ValueError(f"{path}: decisions do not match the cutoff {threshold!r}")


def read_batch_decisions(
    path,
    *,
    threshold=None,
    sample=(),
    sample_seed=None,
    fingerprint=False,
    accepted_rows=None,
    allow_negated_sample_scores=False,
):
    """Check the batch and return its accepted count, digest, and sample negation."""
    pending = {row.id: row for row in sample}
    accepted_ids = []
    sample_ids = set(pending) if sample_seed is not None else set()
    accepted_records = [] if fingerprint and accepted_rows is None else None
    accepted = 0
    scores_match = True
    negated_scores_match = allow_negated_sample_scores
    for item_id, score, is_accepted, group, key in iter_batch_decisions(
        path, threshold=threshold
    ):
        if is_accepted:
            accepted += 1
            if sample_seed is not None:
                accepted_ids.append(item_id)
            if accepted_rows is not None:
                accepted_rows.append(GradedRow(item_id, score, (), group, key))
            elif accepted_records is not None:
                accepted_records.append((item_id, score, key))
        labeled = pending.pop(item_id, None)
        if labeled is not None:
            if score is None or labeled.score is None:
                if score is not labeled.score:
                    raise ValueError(f"{path}: score mismatch for sample {item_id!r}")
            else:
                scores_match = scores_match and score == labeled.score
                negated_scores_match = negated_scores_match and score == -labeled.score
            if not (scores_match or negated_scores_match):
                raise ValueError(f"{path}: score mismatch for sample {item_id!r}")
            if key is not None and labeled.output_key is None:
                raise ValueError(
                    f"{path}: sample {item_id!r} needs an output key. "
                    "Use --output-key-column to load it."
                )
            if key != labeled.output_key:
                raise ValueError(f"{path}: output key mismatch for sample {item_id!r}")
    if pending:
        raise ValueError(
            f"{path}: sample {min(pending)!r} is missing from the batch decisions"
        )
    if not accepted:
        raise ValueError(f"{path}: batch decisions have no accepted rows")
    if sample_seed is not None:
        budget = len(sample_ids)
        drawn = shuffled_by_id(accepted_ids, sample_seed)[:budget]
        if set(drawn) != sample_ids:
            raise ValueError(
                f"{path}: sample does not match the draw for seed {sample_seed} "
                f"and {budget} labels"
            )
    digest = None
    if fingerprint:
        records = (
            accepted_records
            if accepted_rows is None
            else ((row.id, row.score, row.output_key) for row in accepted_rows)
        )
        hasher = sha256()
        for item_id, score, key in sorted(records):
            # Encode the parsed score so formatting differences preserve identity.
            encoded = json.dumps(
                (item_id, (score + 0.0).hex(), key), separators=(",", ":")
            )
            hasher.update(encoded.encode("ascii") + b"\n")
        digest = hasher.hexdigest()
    return accepted, digest, not scores_match


def route_batch(
    path,
    output,
    *,
    threshold,
    id_column,
    score_column,
    group_column,
    output_key_column,
    fallback_output_key_column,
    selection,
    negate_score,
    review_missing_scores,
    keep_columns=(),
    scorer=None,
):
    """Stream predictions through a cutoff and write all batch decisions.

    keep_columns are copied from the input after the decision columns.
    """
    if not isfinite(threshold):
        raise ValueError("threshold must be finite")
    keep_columns = tuple(keep_columns)
    reserved = {"id", "score", "decision", "group", "output_key"}
    if len(set(keep_columns)) != len(keep_columns) or reserved & set(keep_columns):
        raise ValueError(
            "--keep-column names must be distinct and differ from "
            "id, score, decision, group, and output_key"
        )
    required = {id_column, *keep_columns}
    if output_key_column is not None:
        required.add(output_key_column)
    accepted = missing_scores = 0
    with open_scored_predictions(
        path,
        required,
        group_column,
        selection,
        score_column=score_column,
        negate_score=negate_score,
        review_missing_scores=review_missing_scores,
        scorer=scorer,
    ) as (
        rows,
        reader,
        group_column,
    ):
        if (
            output_key_column is None
            and fallback_output_key_column in reader.fieldnames
        ):
            output_key_column = fallback_output_key_column
        columns = ["id", "score", "decision"]
        if group_column is not None:
            columns.append("group")
        if output_key_column is not None:
            columns.append("output_key")
        columns.extend(keep_columns)

        def decisions():
            nonlocal accepted, missing_scores
            seen = set()
            for row, score in rows:
                item_id = row[id_column]
                if not item_id or item_id in seen:
                    raise ValueError(f"empty or duplicate id: {item_id!r}")
                seen.add(item_id)
                if score is None:
                    missing_scores += 1
                elif not isfinite(score):
                    raise ValueError(f"score for id {item_id!r} must be finite")
                meets_cutoff = score is not None and score >= threshold
                accepted += meets_cutoff
                fields = [item_id, score, "accept" if meets_cutoff else "review"]
                if group_column is not None:
                    fields.append(row[group_column])
                if output_key_column is not None:
                    key = row[output_key_column]
                    if not key:
                        raise ValueError(f"{path}: missing output key for {item_id!r}")
                    fields.append(key)
                fields.extend(row[name] for name in keep_columns)
                yield fields
            if not seen:
                raise ValueError("at least one prediction is required")

        total = write_rows(output, columns, decisions())
    overall = {
        "total": total,
        "accepted": accepted,
        "review": total - accepted,
        "coverage": accepted / total,
    }
    if review_missing_scores:
        overall["missing_scores"] = missing_scores
    return overall, group_column, output_key_column
