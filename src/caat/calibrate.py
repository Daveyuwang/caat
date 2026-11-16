"""Fit and replay binary correctness probability maps."""

import csv
import json
from bisect import bisect_right
from itertools import groupby
from math import exp, fsum, isfinite, log, log1p
from operator import itemgetter
from pathlib import Path

from .metrics import probability_bin
from .table import open_predictions_csv, parse_score


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


def fit_temperature(scores, outcomes):
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


def fit_platt(scores, outcomes):
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


def fit_isotonic(scores, outcomes):
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


def fit_histogram(scores, outcomes, bins):
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


CALIBRATORS = {
    "temperature": fit_temperature,
    "platt": fit_platt,
    "isotonic": fit_isotonic,
    "histogram": fit_histogram,
}


def default_scale(method):
    return "probability" if method in {"temperature", "histogram"} else "score"


def fit_calibration(scores, outcomes, *, method, input_scale=None, bins=10):
    """Return a JSON-serializable map trained only on the supplied rows."""
    if method not in CALIBRATORS:
        raise ValueError(f"unknown calibration method: {method!r}")
    input_scale = input_scale or default_scale(method)
    if input_scale not in {"score", "probability"}:
        raise ValueError("input scale must be score or probability")
    if method == "histogram" and input_scale != "probability":
        raise ValueError("histogram calibration requires probabilities")
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    pairs = sorted(zip(scores, outcomes, strict=True))
    if not pairs or any(y not in (0, 1) for _, y in pairs):
        raise ValueError("calibration requires rows with binary correctness outcomes")
    scores = [input_values(x, input_scale) for x, _ in pairs]
    outcomes = [int(y) for _, y in pairs]
    if input_scale == "probability" and method in {"temperature", "platt"}:
        scores = [logit(x) for x in scores]
    fit = CALIBRATORS[method]
    params = (
        fit(scores, outcomes, bins) if method == "histogram" else fit(scores, outcomes)
    )
    return {
        "schema_version": 1,
        "method": method,
        "input_scale": input_scale,
        "parameters": params,
    }


def calibration_transform(mapping):
    """Validate a saved map and return its score-to-probability function."""

    def number(value):
        if type(value) not in (float, int) or not isfinite(value):
            raise ValueError("map parameters must be finite numbers")
        return value

    try:
        if mapping["schema_version"] != 1:
            raise ValueError("unsupported calibration map schema")
        method, scale, params = (
            mapping[k] for k in ("method", "input_scale", "parameters")
        )
        if scale not in {"score", "probability"} or method not in CALIBRATORS:
            raise ValueError("unknown calibration method or input scale")
        if method == "histogram" and scale != "probability":
            raise ValueError("histogram calibration requires probabilities")
        if method == "temperature":
            slope = number(params["inverse_temperature"])
            if slope <= 0:
                raise ValueError("inverse temperature must be positive")
        elif method == "platt":
            divisor, slope, intercept = (
                number(params[k]) for k in ("scale", "slope", "intercept")
            )
            if divisor <= 0:
                raise ValueError("Platt scale must be positive")
        else:
            values = [number(y) for y in params["probabilities"]]
            if not values or any(not 0 <= y <= 1 for y in values):
                raise ValueError("map needs probabilities between 0 and 1")
            if method == "isotonic":
                knots = [number(x) for x in params["scores"]]
                if (
                    len(knots) != len(values)
                    or any(a >= b for a, b in zip(knots, knots[1:]))
                    or values != sorted(values)
                ):
                    raise ValueError("invalid isotonic knots")
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("invalid calibration map") from exc

    def transform(score):
        x = input_values(score, scale)
        if method in {"temperature", "platt"} and scale == "probability":
            x = logit(x)
        if method == "temperature":
            return sigmoid(x * slope)
        if method == "platt":
            return sigmoid((x / divisor) * slope + intercept if slope else intercept)
        if method == "histogram":
            return values[probability_bin(x, len(values))]
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

    return transform


def apply_csv(path, output, mapping, score_column, probability_column):
    transform = calibration_transform(mapping)
    with open_predictions_csv(path, {score_column}, None) as (reader, _):
        fields = reader.fieldnames
        if probability_column in fields or not probability_column:
            raise ValueError("probability output column must be new and nonempty")
        rows = []
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(
                    f"incomplete or extra CSV fields at line {reader.line_num}"
                )
            row[probability_column] = transform(
                parse_score(row[score_column], reader.line_num)
            )
            rows.append(row)
    if not rows:
        raise ValueError("at least one prediction is required")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=[*fields, probability_column])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_calibration(path):
    return json.loads(path.read_text(encoding="utf-8"))
