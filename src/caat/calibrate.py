"""Fit and replay binary correctness probability maps."""

import csv
import json
from bisect import bisect_right
from dataclasses import dataclass
from itertools import groupby
from math import exp, fsum, isfinite, log, log1p
from operator import itemgetter
from pathlib import Path
from typing import Callable

from .metrics import probability_bin
from .registry import Registry
from .report import read_negated_score_source, read_review_missing_scores
from .table import (
    atomic_output,
    open_selected_predictions,
    output_fingerprint,
    parse_score,
)


def sigmoid(value):
    tail = exp(-abs(value))
    return 1 / (1 + tail) if value >= 0 else tail / (1 + tail)


def logit(probability):
    probability = min(max(probability, 1e-12), 1 - 1e-12)
    return log(probability) - log1p(-probability)


def input_values(score, scale):
    score = float(score)
    if not isfinite(score):
        raise ValueError("calibration scores must be finite")
    if scale == "probability" and not 0 <= score <= 1:
        raise ValueError("calibration probabilities must be between 0 and 1")
    return score


def fit_temperature(scores, outcomes, *, bins=10):
    # The derivative of the binary log loss increases with inverse temperature.
    if not any(scores):
        return {"inverse_temperature": 1.0}
    lo, hi = 1e-6, 100.0
    scale = max(map(abs, scores))
    for _ in range(60):
        midpoint = (lo + hi) / 2
        gradient = fsum(
            (x / scale) * (sigmoid(x * midpoint) - y)
            for x, y in zip(scores, outcomes, strict=True)
        )
        if gradient < 0:
            lo = midpoint
        else:
            hi = midpoint
    return {"inverse_temperature": (lo + hi) / 2}


def fit_platt(scores, outcomes, *, bins=10):
    scale = max(map(abs, scores)) or 1.0
    pairs = [(score / scale, y) for score, y in zip(scores, outcomes, strict=True)]
    positive = sum(outcomes)
    intercept = log(positive + 1) - log(len(outcomes) - positive + 1)
    if positive in (0, len(outcomes)):
        return {"scale": scale, "slope": 0.0, "intercept": intercept}

    def loss(slope, intercept):
        logits = [(slope * x + intercept) * (1 - 2 * y) for x, y in pairs]
        return slope**2 / 2 + fsum(max(z, 0) + log1p(exp(-abs(z))) for z in logits)

    slope = 0.0
    for _ in range(100):
        probabilities = [sigmoid(slope * x + intercept) for x, _ in pairs]
        residuals = [p - y for (_, y), p in zip(pairs, probabilities, strict=True)]
        weights = [p * (1 - p) for p in probabilities]
        gradient_slope = slope + fsum(
            x * r for (x, _), r in zip(pairs, residuals, strict=True)
        )
        gradient_intercept = fsum(residuals)
        if max(abs(gradient_slope), abs(gradient_intercept)) < 1e-9:
            break
        ss = 1 + fsum(x * x * w for (x, _), w in zip(pairs, weights, strict=True))
        si = fsum(x * w for (x, _), w in zip(pairs, weights, strict=True))
        ii = fsum(weights)
        determinant = ss * ii - si**2
        ds = (ii * gradient_slope - si * gradient_intercept) / determinant
        di = (ss * gradient_intercept - si * gradient_slope) / determinant
        previous = loss(slope, intercept)
        step = 1.0
        while step > 1e-8 and loss(slope - step * ds, intercept - step * di) > previous:
            step /= 2
        slope -= step * ds
        intercept -= step * di
    return {"scale": scale, "slope": slope, "intercept": intercept}


def fit_isotonic(scores, outcomes, *, bins=10):
    knots, blocks = [], []
    pairs = sorted(zip(scores, outcomes, strict=True), key=itemgetter(0))
    for score, tied in groupby(pairs, key=itemgetter(0)):
        labels = [y for _, y in tied]
        knots.append(score)
        blocks.append((1, len(labels), sum(labels)))
        while len(blocks) > 1:
            n_left, count_left, positives_left = blocks[-2]
            n_right, count_right, positives_right = blocks[-1]
            if positives_left * count_right <= positives_right * count_left:
                break
            blocks[-2:] = [
                (
                    n_left + n_right,
                    count_left + count_right,
                    positives_left + positives_right,
                )
            ]
    return {
        "scores": knots,
        "probabilities": [
            positives / count for size, count, positives in blocks for _ in range(size)
        ],
    }


def fit_histogram(scores, outcomes, *, bins=10):
    totals, positive = [0] * bins, [0] * bins
    for score, outcome in zip(scores, outcomes, strict=True):
        index = probability_bin(score, bins)
        totals[index] += 1
        positive[index] += outcome
    rate = sum(positive) / len(scores)
    return {
        "probabilities": [
            p / n if n else rate for p, n in zip(positive, totals, strict=True)
        ]
    }


@dataclass(frozen=True)
class Calibrator:
    fit: Callable
    restore: Callable
    monotone: bool
    input_scale: str = "score"
    logits: bool = False
    probability_only: bool = False


def finite_parameter(value):
    if type(value) not in (float, int) or not isfinite(value):
        raise ValueError("map parameters must be finite numbers")
    return value


def saved_probabilities(params):
    values = [finite_parameter(y) for y in params["probabilities"]]
    if not values or any(not 0 <= y <= 1 for y in values):
        raise ValueError("map needs probabilities between 0 and 1")
    return values


def restore_temperature(params):
    slope = finite_parameter(params["inverse_temperature"])
    if slope <= 0:
        raise ValueError("inverse temperature must be positive")
    return lambda x: sigmoid(x * slope)


def restore_platt(params):
    scale, slope, intercept = (
        finite_parameter(params[key]) for key in ("scale", "slope", "intercept")
    )
    if scale <= 0:
        raise ValueError("Platt scale must be positive")
    return lambda x: sigmoid((x / scale) * slope + intercept if slope else intercept)


def restore_histogram(params):
    values = saved_probabilities(params)
    return lambda x: values[probability_bin(x, len(values))]


def restore_isotonic(params):
    values = saved_probabilities(params)
    knots = [finite_parameter(x) for x in params["scores"]]
    if (
        len(knots) != len(values)
        or any(a >= b for a, b in zip(knots, knots[1:]))
        or values != sorted(values)
    ):
        raise ValueError("invalid isotonic knots")

    def interpolate(x):
        right = bisect_right(knots, x)
        if right == 0:
            return values[0]
        if right == len(knots):
            return values[-1]
        left = right - 1
        a, b = knots[left], knots[right]
        fraction = (
            (x / 2 - a / 2) / (b / 2 - a / 2)
            if b - a == float("inf")
            else (x - a) / (b - a)
        )
        return values[left] + fraction * (values[right] - values[left])

    return interpolate


CALIBRATORS = Registry(
    "calibrators",
    Calibrator,
    {
        "temperature": Calibrator(
            fit_temperature, restore_temperature, True, "probability", logits=True
        ),
        "platt": Calibrator(fit_platt, restore_platt, False, logits=True),
        "isotonic": Calibrator(fit_isotonic, restore_isotonic, True),
        "histogram": Calibrator(
            fit_histogram,
            restore_histogram,
            False,
            "probability",
            probability_only=True,
        ),
    },
)


def default_scale(method):
    if method not in CALIBRATORS:
        raise ValueError(f"unknown calibration method: {method!r}")
    return CALIBRATORS[method].input_scale


def check_scale(component, input_scale):
    if input_scale not in {"score", "probability"}:
        raise ValueError("input scale must be score or probability")
    if component.probability_only and input_scale != "probability":
        raise ValueError("this calibration method requires probabilities")


def fit_calibration(scores, outcomes, *, method, input_scale=None, bins=10):
    """Return a JSON-serializable map trained only on the supplied rows."""
    if method not in CALIBRATORS:
        raise ValueError(f"unknown calibration method: {method!r}")
    component = CALIBRATORS[method]
    input_scale = input_scale or component.input_scale
    check_scale(component, input_scale)
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    pairs = sorted(zip(scores, outcomes, strict=True))
    if not pairs or any(y not in (0, 1) for _, y in pairs):
        raise ValueError("calibration requires rows with binary correctness outcomes")
    scores = [input_values(x, input_scale) for x, _ in pairs]
    outcomes = [int(y) for _, y in pairs]
    if input_scale == "probability" and component.logits:
        scores = [logit(x) for x in scores]
    params = component.fit(scores, outcomes, bins=bins)
    if not isinstance(params, dict):
        raise ValueError("a calibrator must return a dictionary of map parameters")
    return {
        "schema_version": 1,
        "method": method,
        "input_scale": input_scale,
        "parameters": params,
    }


def calibration_transform(mapping):
    """Validate a saved map and return its score-to-probability function."""

    try:
        if mapping["schema_version"] != 1:
            raise ValueError("unsupported calibration map schema")
        method, scale, params = (
            mapping[k] for k in ("method", "input_scale", "parameters")
        )
        if method not in CALIBRATORS:
            raise ValueError(f"unknown calibration method: {method!r}")
        component = CALIBRATORS[method]
        check_scale(component, scale)
        predict = component.restore(params)
        if not callable(predict):
            raise ValueError(
                "a calibrator must restore a score-to-probability callable"
            )
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("invalid calibration map") from exc

    def transform(score):
        x = input_values(score, scale)
        if component.logits and scale == "probability":
            x = logit(x)
        probability = float(predict(x))
        if not 0 <= probability <= 1:
            raise ValueError("a calibrated probability must be between 0 and 1")
        return probability

    return transform


def apply_csv(
    path,
    output,
    mapping,
    score_column,
    probability_column,
    *,
    negate_score=False,
    review_missing_scores=False,
    selection=None,
    fit_outputs=None,
    id_column="id",
):
    transform = calibration_transform(mapping)
    required = {score_column}
    fitted = None
    if fit_outputs is not None:
        key_column = fit_outputs["key_column"]
        fitted = set(fit_outputs["sha256"])
        required.update((id_column, key_column))
    with (
        atomic_output(Path(output).resolve(), newline="") as destination,
        open_selected_predictions(path, required, None, selection) as (
            selected,
            reader,
            _,
        ),
    ):
        fields = reader.fieldnames
        if probability_column in fields or not probability_column:
            raise ValueError("probability output column must be new and nonempty")
        writer = csv.DictWriter(destination, fieldnames=[*fields, probability_column])
        writer.writeheader()
        count = 0
        for row in selected:
            if fitted is not None:
                item_id = row[id_column]
                if output_fingerprint(item_id, row[key_column]) in fitted:
                    raise ValueError(
                        f"holdout reuses output from the fit for sample {item_id!r}"
                    )
            score = parse_score(
                row[score_column],
                reader.line_num,
                negate=negate_score,
                review_missing_scores=review_missing_scores,
            )
            row[probability_column] = transform(score) if score is not None else None
            writer.writerow(row)
            count += 1
        if not count:
            raise ValueError("at least one prediction is required")
    return count


def read_calibration(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_saved_calibration(path, *, grader=None):
    source = read_calibration(path)
    command = source.get("command") if isinstance(source, dict) else None
    inputs = source.get("input") if isinstance(source, dict) else None
    if command not in ("policy", "run"):
        if grader is not None:
            raise ValueError("--grader requires a policy or run report")
        if not isinstance(inputs, dict):
            return source, None, None, None, {}
        review_missing_scores = read_review_missing_scores(path, source)
        score_column = inputs.get("score_column")
        if (score_column is not None or review_missing_scores) and (
            not isinstance(score_column, str) or not score_column.strip()
        ):
            raise ValueError(f"{path}: probability map needs a score column name")
        return (
            source,
            score_column,
            read_negated_score_source(path, source, score_column),
            review_missing_scores,
            inputs,
        )

    policy = source.get("policy") if command == "run" else source
    try:
        maps = policy["probability_calibration"]["graders"]
    except (KeyError, TypeError):
        maps = None
    if not isinstance(maps, dict) or not maps:
        raise ValueError(f"{path}: {command} report has no fitted probability maps")
    if grader is None:
        if len(maps) != 1:
            raise ValueError(f"{path}: select a map with --grader ({', '.join(maps)})")
        grader = next(iter(maps))
    if grader not in maps:
        raise ValueError(f"{path}: no fitted probability map for grader {grader!r}")
    try:
        mapping = maps[grader]["map"]
        score_column = (
            source["selected_score"]
            if command == "run"
            else source["input"]["score_column"]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: invalid saved probability map") from exc
    if not isinstance(score_column, str) or not score_column.strip():
        raise ValueError(f"{path}: probability map needs a score column name")
    return (
        mapping,
        score_column,
        read_negated_score_source(path, source, score_column),
        read_review_missing_scores(path, source),
        inputs if isinstance(inputs, dict) else {},
    )
