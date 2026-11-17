"""Exact error-count probabilities and fixed-population bounds."""

from bisect import bisect_left
from fractions import Fraction
from math import ceil, comb, nextafter
from struct import pack, unpack


def validate_population(size):
    if type(size) is not int or size < 1:
        raise ValueError("accepted population must be a positive integer")


def round_rate_up(rate):
    """Return the smallest float whose decimal spelling is at least the rate."""
    rounded = float(rate)
    return nextafter(rounded, 1.0) if Fraction(str(rounded)) < rate else rounded


def hypergeometric_cdf(size, errors, labels, observed):
    """Return the exact probability of at most observed errors without replacement."""
    correct = size - errors
    lower, upper = max(0, labels - correct), min(errors, labels)
    if observed < lower:
        return Fraction(0)
    if observed >= upper:
        return Fraction(1)
    if observed == 0 and errors < labels:
        # Place every error in the unlabeled part of the population.
        return Fraction(comb(size - labels, errors), comb(size, errors))
    if observed - lower + 1 > upper - observed:
        # Sum the shorter tail by counting correct labels instead.
        return 1 - hypergeometric_cdf(size, correct, labels, labels - observed - 1)

    mass = total = comb(errors, observed) * comb(correct, labels - observed)
    for count in range(observed, lower, -1):
        # Ratio of adjacent counts of samples with the indicated errors.
        mass = (
            mass
            * count
            * (correct - labels + count)
            // ((errors - count + 1) * (labels - count + 1))
        )
        total += mass
    return Fraction(total, comb(size, labels))


def _tail_bounds(lower, upper, mode, observed, step_ratio, *, precision=80):
    """Return rational lower and upper bounds on the same tail probability."""
    if observed < lower:
        return (0, 1), (0, 1)
    if observed >= upper:
        return (1, 1), (1, 1)

    scale = 1 << precision
    total_low = total_high = scale
    tail_low = tail_high = scale if mode <= observed else 0
    for endpoint, step in ((lower, -1), (upper, 1)):
        mass_low = mass_high = scale
        for count in range(mode, endpoint, step):
            numerator, denominator = step_ratio(count, step)
            mass_low = mass_low * numerator // denominator
            mass_high = (mass_high * numerator + denominator - 1) // denominator
            total_low += mass_low
            total_high += mass_high
            next_count = count + step
            if next_count <= observed:
                tail_low += mass_low
                tail_high += mass_high
            if mass_high <= 1:
                # Each remaining term is at most one unit. Include this
                # entire allowance in both upper bounds.
                remaining = abs(endpoint - next_count)
                total_high += remaining
                tail_high += remaining
                break

    return (tail_low, total_high), (min(tail_high, total_low), total_low)


def _hypergeometric_bounds(size, errors, labels, observed):
    correct = size - errors
    lower, upper = max(0, labels - correct), min(errors, labels)

    def step_ratio(count, step):
        if step < 0:
            return count * (correct - labels + count), (errors - count + 1) * (
                labels - count + 1
            )
        return (errors - count) * (labels - count), (count + 1) * (
            correct - labels + count + 1
        )

    # This is where the adjacent mass ratio crosses one.
    mode = min(upper, max(lower, (labels + 1) * (errors + 1) // (size + 2)))
    return _tail_bounds(lower, upper, mode, observed, step_ratio)


def _compare_bounds(bounds, level):
    lower, upper = bounds
    numerator, denominator = level.as_integer_ratio()
    upper_difference = upper[0] * denominator - upper[1] * numerator
    lower_difference = lower[0] * denominator - lower[1] * numerator
    if upper_difference < 0:
        return -1
    if lower_difference > 0:
        return 1
    if lower_difference == upper_difference == 0:
        return 0
    return None


def _round_bounds(bounds):
    lower, upper = bounds
    rounded = float(Fraction(*lower))
    return rounded if rounded == float(Fraction(*upper)) else None


def compare_hypergeometric_cdf(size, errors, labels, observed, level):
    """Compare the exact lower-tail probability with a level, returning -1, 0, or 1."""
    comparison = _compare_bounds(
        _hypergeometric_bounds(size, errors, labels, observed), level
    )
    if comparison is not None:
        return comparison
    probability = hypergeometric_cdf(size, errors, labels, observed)
    return (probability > level) - (probability < level)


def hypergeometric_cdf_float(size, errors, labels, observed):
    """Round the exact lower-tail probability to a float."""
    rounded = _round_bounds(_hypergeometric_bounds(size, errors, labels, observed))
    return (
        float(hypergeometric_cdf(size, errors, labels, observed))
        if rounded is None
        else rounded
    )


def _binomial_bounds(labels, observed, rate, *, precision=80):
    errors, denominator = rate.as_integer_ratio()
    correct = denominator - errors
    if not errors or not correct:
        outcome = labels if errors else 0
        bound = (int(observed >= outcome), 1)
        return bound, bound

    def step_ratio(count, step):
        if step < 0:
            return count * correct, (labels - count + 1) * errors
        return (labels - count) * errors, (count + 1) * correct

    mode = (labels + 1) * errors // denominator
    return _tail_bounds(0, labels, mode, observed, step_ratio, precision=precision)


def _binomial_cdf(labels, observed, rate):
    if observed < 0:
        return Fraction(0)
    if observed >= labels or rate == 0:
        return Fraction(1)
    if rate == 1:
        return Fraction(0)
    if observed + 1 > labels - observed:
        return 1 - _binomial_cdf(labels, labels - observed - 1, 1 - rate)
    errors, denominator = rate.as_integer_ratio()
    correct = denominator - errors
    mass = total = (
        comb(labels, observed) * errors**observed * correct ** (labels - observed)
    )
    for count in range(observed, 0, -1):
        mass = mass * count * correct // ((labels - count + 1) * errors)
        total += mass
    return Fraction(total, denominator**labels)


def _binomial_precision(labels, level):
    numerator, denominator = level.as_integer_ratio()
    edge = min(numerator, denominator - numerator)
    if edge <= 0:
        return 80
    # Allow for relative rounding near zero or one and for the number of terms.
    return max(
        80, 64 + labels.bit_length() + denominator.bit_length() - edge.bit_length()
    )


def _binomial_test(labels, observed, rate, level, precision):
    """Return the exact comparison and a rational endpoint for interpolation."""
    bounds = _binomial_bounds(labels, observed, rate, precision=precision)
    comparison = _compare_bounds(bounds, level)
    if comparison is not None:
        return comparison, bounds[0]
    probability = _binomial_cdf(labels, observed, rate)
    comparison = (probability > level) - (probability < level)
    return comparison, bounds[1] if comparison > 0 else bounds[0]


def compare_binomial_cdf(labels, observed, rate, level):
    """Compare an exact binomial lower-tail probability with a level."""
    comparison, _ = _binomial_test(
        labels, observed, rate, level, _binomial_precision(labels, level)
    )
    return comparison


def _binomial_tail_below_rounding(labels, observed, rate, bits):
    numerator, denominator = rate.as_integer_ratio()
    mean = labels * numerator
    distance = mean - observed * denominator
    # The Chernoff bound is exp(-distance**2 / (2 * mean * denominator)).
    # An exponent of at least bits makes the bound smaller than 2**-bits.
    return distance > 0 and distance**2 >= 2 * bits * mean * denominator


def binomial_cdf_float(labels, observed, rate):
    """Round the exact binomial lower-tail probability to a float."""
    if _binomial_tail_below_rounding(labels, observed, rate, 1075):
        return 0.0
    if _binomial_tail_below_rounding(labels, labels - observed - 1, 1 - rate, 54):
        return 1.0
    precision = 80
    while True:
        bounds = _binomial_bounds(labels, observed, rate, precision=precision)
        rounded = _round_bounds(bounds)
        if rounded is not None:
            return rounded
        numerator, denominator = bounds[0]
        exponent = denominator.bit_length() - numerator.bit_length()
        # Refine even subnormal tails. Fall back to an exact sum near a rounding tie.
        bits = min(1075, exponent + 54) if numerator else 1075
        limit = bits + 2 * labels.bit_length() + 32
        if precision >= limit:
            return float(_binomial_cdf(labels, observed, rate))
        precision = min(precision * 2, limit)


def upper_binomial_rate(labels, observed, level):
    """Find the smallest representable rate rejected by the exact lower tail."""
    precision = _binomial_precision(labels, level)
    # Nonnegative binary64 values are ordered by their unsigned bits.
    lower, upper = 0, int.from_bytes(pack(">d", 1.0), "big")
    previous_point, previous_gap = lower, 1 - level
    point = (lower + upper) // 2
    secant_steps = 0
    while lower + 1 < upper:
        rate = unpack(">d", point.to_bytes(8, "big"))[0]
        comparison, probability = _binomial_test(
            labels, observed, Fraction(str(rate)), level, precision
        )
        if comparison == 0:
            return rate
        if comparison > 0:
            lower = point
        else:
            upper = point
        if lower + 1 == upper:
            break

        gap = Fraction(*probability) - level
        midpoint = (lower + upper) // 2
        proposal = midpoint
        # Spacing between bit patterns is constant within one exponent.
        # Take a bisection step after at most two secant probes.
        if (
            secant_steps < 2
            and point >> 52 == previous_point >> 52
            and gap != previous_gap
        ):
            nearby = ceil(point - gap * (point - previous_point) / (gap - previous_gap))
            if lower <= nearby <= upper and nearby >> 52 == point >> 52:
                proposal = max(lower + 1, min(upper - 1, nearby))
        secant_steps = 0 if proposal == midpoint else secant_steps + 1
        previous_point, previous_gap = point, gap
        point = proposal
    return unpack(">d", upper.to_bytes(8, "big"))[0]


def upper_population_errors(size, labels, observed, failure_rate):
    """Find the greatest total error count retained by the exact lower-tail test."""
    possible = range(observed, size - labels + observed + 1)
    first_rejected = bisect_left(
        possible,
        True,
        key=lambda errors: compare_hypergeometric_cdf(
            size, errors, labels, observed, failure_rate
        )
        <= 0,
    )
    return possible[first_rejected - 1]
