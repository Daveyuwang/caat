"""Exact two-look designs for uniform sampling from a finite batch."""

from fractions import Fraction
from math import ceil, comb, floor, isfinite

from .finite import hypergeometric_cdf
from .plan import (
    forecast_verdict,
    pending_probability,
    sample_error_limits,
    verdict_ranges,
)


def _mass(size, errors, cap):
    return [comb(errors, y) * comb(size - errors, cap - y) for y in range(cap + 1)]


def _crossing_counts(cap, look, maximum):
    """Number of interim subsets crossing the lower boundary, given each final count."""
    ways = comb(cap, look)
    return [
        int(ways * hypergeometric_cdf(cap, y, look, maximum)) for y in range(cap + 1)
    ]


def _last_boundary(mass, crossed, ways, alpha):
    # Count the union of a first-look crossing and a final crossing.
    # Ignoring opposite-direction stops gives a conservative directional test.
    spent = sum(w * e for w, e in zip(mass, crossed))
    allowance = floor(alpha * sum(mass) * ways)
    for y, (w, e) in enumerate(zip(mass, crossed)):
        spent += w * (ways - e)
        if spent > allowance:
            return y - 1
    return len(mass) - 1


def _stage(n, low, high):
    return dict(
        labels=n,
        max_errors_to_meet=low if low >= 0 else None,
        min_errors_to_exceed=high if high <= n else None,
    )


def validate_stages(stages, size, cap, target, confidence):
    if (
        not isinstance(stages, list)
        or len(stages) != 2
        or not all(isinstance(s, dict) for s in stages)
    ):
        raise ValueError("expected two frozen checkpoints")
    first = stages[0].get("labels")
    final = stages[1].get("labels")
    if (
        type(first) is not int
        or not 0 < first < cap
        or type(final) is not int
        or final != cap
    ):
        raise ValueError("invalid checkpoint label counts")
    boundaries = []
    for s in stages:
        n = s["labels"]
        low, high = s.get("max_errors_to_meet"), s.get("min_errors_to_exceed")
        if set(s) != {"labels", "max_errors_to_meet", "min_errors_to_exceed"} or any(
            x is not None and (type(x) is not int or not 0 <= x <= n)
            for x in (low, high)
        ):
            raise ValueError("invalid checkpoint boundaries")
        low, high = (-1 if low is None else low), (n + 1 if high is None else high)
        if low >= high:
            raise ValueError("overlapping checkpoint boundaries")
        boundaries.append((low, high))
    (a, b), (c, d) = boundaries
    allowed = floor(Fraction(str(target)) * size)
    if allowed >= size:
        raise ValueError("two-stage targets must have a violating population")
    ways = comb(cap, first)
    alpha = 1 - Fraction(str(confidence))
    for k, initial, last in ((allowed + 1, a, c), (size - allowed, first - b, cap - d)):
        mass = _mass(size, k, cap)
        cross = _crossing_counts(cap, first, initial)
        used = sum(w * (ways if y <= last else cross[y]) for y, w in enumerate(mass))
        if used > alpha * sum(mass) * ways:
            raise ValueError(
                "checkpoint boundaries exceed the directional error allowance"
            )


def _outcomes(mass, early_low, early_high, ways, first, final, low, high):
    total = sum(mass) * ways
    meets = exceeds = continued = 0
    for y, weight in enumerate(mass):
        left, right = early_low[y], early_high[y]
        more = ways - left - right
        continued += weight * more
        meets += weight * (left + (more if y <= low else 0))
        exceeds += weight * (right + (more if y >= high else 0))
    return dict(
        meets=Fraction(meets, total),
        exceeds=Fraction(exceeds, total),
        inconclusive=Fraction(total - meets - exceeds, total),
        expected_labels=first + (final - first) * Fraction(continued, total),
        expected_requests=1 + Fraction(continued, total),
    )


def _single_comparison(size, cap, limits, rates, price):
    """Compare all constant block sizes using the same pending probabilities."""
    scenarios = []
    schedules = []
    ranges = verdict_ranges(cap, limits)
    for rate in rates:
        result = forecast_verdict(size, cap, rate, limits)
        k = result["expected_population_errors"]
        pending = [pending_probability(size, k, t, cap - t, ranges) for t in range(cap)]
        costs = []
        for b in range(1, cap + 1):
            requests = sum(pending[::b])
            labels = sum(min(b, cap - t) * pending[t] for t in range(0, cap, b))
            costs.append((labels + price * requests, labels, requests))
        schedules.append(costs)
        scenarios.append(dict(rate=rate, errors=k, **result["verdict_probabilities"]))
    idx = min(range(cap), key=lambda i: max(s[i][0] for s in schedules))
    for s, costs in zip(scenarios, schedules):
        s["cost"], s["expected_labels"], s["expected_requests"] = costs[idx]
    return dict(block_size=idx + 1, scenarios=scenarios)


def design(size, cap, first, target, confidence, rates, power, price):
    if not 0 < first < cap <= size:
        raise ValueError(
            "interim labels must be strictly between zero and the effective cap"
        )
    if not 0 <= rates[0] < target < rates[1] <= 1:
        raise ValueError("error scenarios must bracket the saved target")
    if not 0 < power < 1 or not isfinite(price) or price < 0:
        raise ValueError(
            "use power between 0 and 1 and a finite nonnegative request cost"
        )
    alpha = 1 - Fraction(str(confidence))
    allowed = floor(Fraction(str(target)) * size)
    counts = [ceil(Fraction(str(rate)) * size) for rate in rates]
    null_low = _mass(size, allowed + 1, cap)
    null_correct = _mass(size, size - allowed, cap)
    populations = [_mass(size, k, cap) for k in counts]
    ways = comb(cap, first)
    best = None
    for share in map(Fraction, ("1/10", "1/5", "2/5", "1/2", "4/5")):
        early = sample_error_limits(size, first, target, 1 - alpha * share)
        a = early["max_errors_to_meet"]
        b = early["min_errors_to_exceed"]
        left = _crossing_counts(cap, first, -1 if a is None else a)
        correct = _crossing_counts(cap, first, -1 if b is None else first - b)
        low = _last_boundary(null_low, left, ways, alpha)
        high = cap - _last_boundary(null_correct, correct, ways, alpha)
        if low >= high:
            raise ValueError("two-stage directional boundaries overlap")
        scenarios = [
            _outcomes(mass, left, correct[::-1], ways, first, cap, low, high)
            for mass in populations
        ]
        attained = min(scenarios[0]["meets"], scenarios[1]["exceeds"])
        feasible = attained >= Fraction(str(power))
        costs = [
            s["expected_labels"] + Fraction(str(price)) * s["expected_requests"]
            for s in scenarios
        ]
        rank = (not feasible, 0 if feasible else -attained, max(costs))
        if best is None or rank < best[0]:
            rows = [
                dict(
                    rate=r,
                    errors=k,
                    cost=float(c),
                    **{n: float(v) for n, v in s.items()},
                )
                for r, k, c, s in zip(rates, counts, costs, scenarios)
            ]
            best = (
                rank,
                dict(
                    qualifies=feasible,
                    stages=[dict(labels=first, **early), _stage(cap, low, high)],
                    scenarios=rows,
                ),
            )
    result = best[1]
    limits = sample_error_limits(size, cap, target, confidence)
    result["single_stage"] = dict(
        **limits, **_single_comparison(size, cap, limits, rates, price)
    )
    return result
