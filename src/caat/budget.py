"""Compare executable finite-population audit designs under a hard label ceiling."""

from fractions import Fraction
from math import ceil, floor

from .finite import hypergeometric_cdf
from .plan import forecast_block_cost, sample_error_limits
from .two_stage import checkpoint_design, single_comparison, validate_settings


def _fixed_qualifies(size, cap, limits, counts, power):
    low, high = limits["max_errors_to_meet"], limits["min_errors_to_exceed"]
    return (
        low is not None
        and high is not None
        and hypergeometric_cdf(size, counts[0], cap, low) >= power
        and hypergeometric_cdf(size, size - counts[1], cap, cap - high) >= power
    )


def _near_target(size, errors, offer):
    cap = offer["cap"]
    if offer["mode"] == "fixed-block":
        cost = forecast_block_cost(
            size, errors, cap, offer["limits"], offer["block_size"]
        )
        return {k: cost[k] for k in ("expected_labels", "expected_requests")}
    first = offer["stages"][0]
    n = first["labels"]
    pending = Fraction(1)
    if (low := first["max_errors_to_meet"]) is not None:
        pending -= hypergeometric_cdf(size, errors, n, low)
    if (high := first["min_errors_to_exceed"]) is not None:
        pending -= hypergeometric_cdf(size, size - errors, n, n - high)
    return dict(
        expected_labels=float(n + (cap - n) * pending),
        expected_requests=float(1 + pending),
    )


def compare_budgets(size, ceiling, target, confidence, rates, power, price):
    validate_settings(target, rates, power, price)
    if ceiling < 1 or not 0.5 < confidence < 1:
        raise ValueError(
            "use a positive budget ceiling and confidence between 0.5 and 1"
        )
    limit = min(size, ceiling)
    counts = [ceil(Fraction(str(r)) * size) for r in rates]
    required = Fraction(str(power))
    caps = set(range(200, limit + 1, 25))
    if limit:
        caps.add(limit)
    if limit < 200:
        caps.update(range(25, limit, 25))
    if limit <= 25:
        caps.update(range(1, limit + 1))
    baseline = None
    for cap in range(1, limit + 1):
        limits = sample_error_limits(size, cap, target, confidence)
        if _fixed_qualifies(size, cap, limits, counts, required):
            baseline = cap
            caps.add(cap)
            break
    offers = {}
    ranks = {}

    def consider(cap, mode, design):
        scenarios = design["scenarios"]
        cost = max(s["cost"] for s in scenarios)
        requests = max(s["expected_requests"] for s in scenarios)
        for name, rank in (
            ("smallest-cap", (cap, cost, requests)),
            ("lowest-cost", (cost, cap, requests)),
        ):
            if name not in ranks or rank < ranks[name]:
                ranks[name] = rank
                offers[name] = dict(mode=mode, cap=cap, worst_cost=cost, **design)

    for cap in sorted(caps):
        limits = sample_error_limits(size, cap, target, confidence)
        if (
            baseline is not None
            and cap >= baseline
            and _fixed_qualifies(size, cap, limits, counts, required)
        ):
            consider(
                cap,
                "fixed-block",
                dict(
                    limits=limits, **single_comparison(size, cap, limits, rates, price)
                ),
            )
        firsts = set(range(50, cap, 25)) | {2 * cap // 3}
        if limit <= 25:
            firsts.update(range(1, cap))
        for first in sorted(firsts - {0}):
            design = checkpoint_design(
                size, cap, first, target, confidence, rates, power, price
            )
            if design.pop("qualifies"):
                consider(cap, "two-stage", design)

    allowed = floor(Fraction(str(target)) * size)
    for offer in offers.values():
        offer["near_target"] = dict(
            errors=allowed, **_near_target(size, allowed, offer)
        )
    return dict(
        qualifies=bool(offers),
        caps_tested=sorted(caps),
        minimum_single_stage_cap=baseline,
        offers=offers,
        extra_reserve=offers["lowest-cost"]["cap"] - offers["smallest-cap"]["cap"]
        if offers
        else None,
    )
