"""Read one completed prediction per instance from a saved HELM run."""

import json

from .features import (
    FULL_SCORE_FIELDS,
    SavedResponse,
    full_scores,
    mean_token_logprob,
    token_logprobs,
)
from .table import write_rows


SCORE_COLUMNS = ("mean_token_logprob", *FULL_SCORE_FIELDS)


def load_json(path):
    with path.open(encoding="utf-8-sig") as source:
        return json.load(source)


def requests(path):
    state = load_json(path)
    rows = state.get("request_states") if isinstance(state, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("scenario_state.json needs nonempty request_states")
    seen = set()
    for row in rows:
        try:
            item_id = row["instance"]["id"]
            trial = row["train_trial_index"]
            result = row["result"]
            completions = result["completions"]
            success = result["success"]
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid request state in scenario_state.json") from exc
        if not isinstance(item_id, str) or not item_id.strip() or item_id in seen:
            raise ValueError(f"empty or duplicate HELM instance ID: {item_id!r}")
        seen.add(item_id)
        if type(trial) is not int or trial < 0:
            raise ValueError(f"invalid training trial for instance {item_id!r}")
        if (
            success is not True
            or not isinstance(completions, list)
            or len(completions) != 1
        ):
            raise ValueError(f"instance {item_id!r} needs one successful completion")
        completion = completions[0]
        try:
            text = completion["text"]
            logprobs = token_logprobs(completion, item_id)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid token log probabilities or text for {item_id!r}"
            ) from exc
        if not isinstance(text, str):
            raise ValueError(f"instance {item_id!r} needs a text completion")
        if not any(logprobs):
            raise ValueError(
                f"instance {item_id!r} has all-zero token log probabilities"
            )
        yield item_id, trial, row, SavedResponse(text, logprobs=logprobs)


def published_grades(path, metrics):
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError("per_instance_stats.json needs a list of per-instance stats")
    grades = {}
    wanted = set(metrics)
    for row in records:
        try:
            item_id = row["instance_id"]
            trial = row["train_trial_index"]
            stats = [(stat["name"]["name"], stat) for stat in row["stats"]]
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid row in per_instance_stats.json") from exc
        if (
            not isinstance(item_id, str)
            or not item_id.strip()
            or type(trial) is not int
            or trial < 0
        ):
            raise ValueError("invalid instance ID or trial in per_instance_stats.json")
        for name, stat in stats:
            if not isinstance(name, str):
                raise ValueError(f"invalid metric name for instance {item_id!r}")
            if name not in wanted:
                continue
            key = (item_id, trial, name)
            if key in grades:
                raise ValueError(
                    f"duplicate metric {name!r} for instance {item_id!r}, trial {trial}"
                )
            score = stat.get("sum")
            if (
                type(stat.get("count")) is not int
                or stat["count"] != 1
                or type(score) not in (int, float)
                or not 0 <= score <= 1
            ):
                raise ValueError(
                    f"metric {name!r} for {item_id!r} needs one score in [0, 1]"
                )
            grades[key] = int(score == 1)
    return grades


def annotation_grades(row, selectors, item_id):
    for name, group, field in selectors:
        try:
            value = row["annotations"][group][field]
        except (KeyError, TypeError):
            value = None
        if type(value) is not bool:
            raise ValueError(
                f"instance {item_id!r} needs a boolean annotation {name!r}"
            )
        yield int(value)


def export_run(run, output, *, metrics=(), annotations=()):
    metrics = tuple(metrics)
    annotations = tuple(annotations)
    columns = ("id", *SCORE_COLUMNS, *metrics, *annotations)
    if (
        not (metrics or annotations)
        or not all(columns)
        or len(columns) != len(set(columns))
    ):
        raise ValueError("provide at least one distinct grader metric or annotation")
    selectors = [(name, *name.split(".", 1)) for name in annotations]
    if any(len(parts) != 3 or not all(parts) for parts in selectors):
        raise ValueError("use GROUP.FIELD for --annotation")
    state = run / "scenario_state.json"
    stats = run / "per_instance_stats.json"
    if output.resolve() in {state.resolve(), stats.resolve()}:
        raise ValueError("output would overwrite a HELM input file")
    published = published_grades(stats, metrics) if metrics else {}

    def rows():
        for item_id, trial, request, response in requests(state):
            try:
                grades = [published.pop((item_id, trial, name)) for name in metrics]
            except KeyError as exc:
                raise ValueError(
                    f"missing metric {exc.args[0][2]!r} for {item_id!r}, trial {trial}"
                ) from exc
            grades.extend(annotation_grades(request, selectors, item_id))
            yield (
                item_id,
                mean_token_logprob(response),
                *full_scores(response),
                *grades,
            )
        if published:
            item_id, trial, name = next(iter(published))
            raise ValueError(
                f"unexpected metric {name!r} for {item_id!r}, trial {trial}"
            )

    return write_rows(output, columns, rows())
