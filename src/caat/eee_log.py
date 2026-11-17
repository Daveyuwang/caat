"""Read single-turn sample records from Every Eval Ever."""

import json
from hashlib import sha256

from .table import write_external_score_join, write_rows


IDENTITY = ("evaluation_id", "result", "id", "output_sha256")


def samples(path, result):
    seen = set()
    run = None
    chosen = result
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid sample JSON at line {line_number}") from exc
            if not isinstance(sample, dict):
                raise ValueError(f"expected a sample object at line {line_number}")
            current_run = sample.get("evaluation_id"), sample.get("model_id")
            name = sample.get("evaluation_result_id") or sample.get("evaluation_name")
            if not all(
                isinstance(key, str) and key.strip() for key in (*current_run, name)
            ):
                raise ValueError(
                    f"missing evaluation, model, or result identity at line {line_number}"
                )
            if run is None:
                run = current_run
            if current_run != run:
                raise ValueError(
                    "the sample file must contain one evaluation and model"
                )
            if chosen is None:
                chosen = name
            if name != chosen:
                if result is None:
                    raise ValueError(
                        "samples contain several results. Select one with --result"
                    )
                continue
            item_id = sample.get("sample_id")
            if not isinstance(item_id, str) or not item_id.strip() or item_id in seen:
                raise ValueError(
                    f"invalid or duplicate sample_id at line {line_number}: {item_id!r}"
                )
            seen.add(item_id)
            if sample.get("interaction_type") != "single_turn":
                raise ValueError(f"sample {item_id!r} needs a single-turn interaction")
            output = sample.get("output")
            completions = output.get("raw") if isinstance(output, dict) else None
            if (
                not isinstance(completions, list)
                or len(completions) != 1
                or not isinstance(completions[0], str)
            ):
                raise ValueError(f"sample {item_id!r} needs one saved text response")
            completion = completions[0]
            inputs = sample.get("input")
            if not isinstance(inputs, dict) or not isinstance(inputs.get("raw"), str):
                raise ValueError(f"sample {item_id!r} needs a raw text prompt")
            prompt = inputs.get("formatted")
            if prompt is None:
                prompt = inputs["raw"]
            if not isinstance(prompt, str):
                raise ValueError(f"sample {item_id!r} needs a text prompt")
            evaluation = sample.get("evaluation")
            correct = (
                evaluation.get("is_correct") if isinstance(evaluation, dict) else None
            )
            if type(correct) is not bool:
                raise ValueError(
                    f"sample {item_id!r} needs a boolean evaluation.is_correct"
                )
            key = run[0], name, item_id, sha256(completion.encode("utf-8")).hexdigest()
            yield key, prompt, completion, int(correct)
    if not seen:
        raise ValueError(f"no samples for result {chosen!r}")


def export_generations(log, output, *, result=None):
    if output.resolve() == log.resolve():
        raise ValueError("output would overwrite the sample log")
    return write_rows(
        output,
        (*IDENTITY, "input", "completion"),
        (
            (*key, prompt, completion)
            for key, prompt, completion, _ in samples(log, result)
        ),
    )


def join_scores(
    log,
    scores_file,
    output,
    *,
    score_column="confidence",
    result=None,
    score_files=(),
    lower_is_better=(),
    review_missing_scores=False,
):
    def rows():
        for key, _, _, grade in samples(log, result):
            yield key, (grade,)

    return write_external_score_join(
        log,
        scores_file,
        output,
        rows(),
        identity_columns=IDENTITY,
        grade_columns=("correct",),
        score_column=score_column,
        score_files=score_files,
        lower_is_better=lower_is_better,
        review_missing_scores=review_missing_scores,
    )
