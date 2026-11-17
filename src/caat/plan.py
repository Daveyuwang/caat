"""Plan labels for a frozen acceptance policy."""

from bisect import bisect_left
from fractions import Fraction
from math import ceil, floor

from .finite import (
    binomial_cdf_float,
    compare_binomial_cdf,
    compare_hypergeometric_cdf,
    hypergeometric_cdf_float,
    round_rate_up,
    upper_population_errors,
    validate_population,
)
from .metrics import Counts
from .report import check_fit_overlap
from .route import accepted_population_digest
from .rules import binomial_upper_bound
from .scorer import LinearScorer
from .table import (
    accepted_at,
    iter_validated_rows,
    open_prediction_table,
    open_table_rows,
)


DEFAULT_MAX_ACCEPTED = 100_000


def read_candidate_batch(path, saved, *, selection=None):
    """Apply a saved cutoff for sampling. Keep only the rows eligible for labels."""
    key = saved["output_key_column"]
    fit_outputs = saved["fit_outputs"]
    if fit_outputs is None:
        with open_prediction_table(path, set(), None) as (reader, _):
            if (key or "output_sha256") in reader.fieldnames:
                key = key or "output_sha256"
            else:
                key = None
    source = saved.get("negated_score_source")
    review_missing = bool(saved["review_missing_scores"])
    with open_table_rows(
        path,
        id_column=saved["id_column"],
        score_column=source or saved["score_column"],
        output_key_column=key,
        scorer=LinearScorer.from_dict(saved["scorer"]) if "scorer" in saved else None,
        negate_score=source is not None,
        review_missing_scores=review_missing,
        require_labels=False,
        selection=selection,
    ) as (rows, _):
        batch = list(
            iter_validated_rows(
                rows, (), minimum_graders=0, review_missing_scores=review_missing
            )
        )
    if fit_outputs is not None:
        check_fit_overlap(batch, fit_outputs, cohort="audit batch")
    candidates = [row for row in batch if accepted_at(row, saved["threshold"])]
    digest = accepted_population_digest(
        (r.id, r.score, r.output_key) for r in candidates
    )
    return candidates, len(batch), digest, key


def check_label_plan(plan, *, threshold, accepted, accepted_population):
    cutoff = plan["threshold"]
    if cutoff is not None and threshold != cutoff:
        raise ValueError(f"cutoff does not match the label plan ({cutoff!r})")
    if accepted_population != plan["accepted_population"]:
        raise ValueError("accepted population does not match the label plan")
    if accepted != plan["accepted"]:
        raise ValueError(
            f"label plan requires {plan['accepted']} accepted labels, found {accepted}"
        )


def _error_quantile(compare_expected, labels):
    return bisect_left(
        range(labels + 1), True, key=lambda count: compare_expected(labels, count) >= 0
    )


def _error_limit(compare_null, labels):
    return (
        bisect_left(
            range(labels + 1), True, key=lambda count: compare_null(labels, count) > 0
        )
        - 1
    )


def _first_unexcluded_budget(could_qualify, excluded, limit):
    """Advance only past budgets for which the entire prefix is ruled out."""
    high = min(max(1, 2 * excluded), limit)
    while not could_qualify(high):
        excluded = high
        if high == limit:
            return limit + 1
        high = min(2 * high, limit)
    while excluded + 1 < high:
        middle = (excluded + high) // 2
        if could_qualify(middle):
            high = middle
        else:
            excluded = middle
    return excluded + 1


def _minimum_label_budget(
    compare_null, compare_expected, probability, limit, *, bound_prefix
):
    first = _first_unexcluded_budget(
        lambda labels: compare_null(labels, 0) <= 0, 0, limit
    )
    if first > limit:
        return None, None, None
    if compare_expected(first, 0) >= 0:
        return first, 0, probability(first, 0)
    start = first + 1
    if bound_prefix and start <= limit:

        def could_qualify_prefix(labels):
            # The optimal randomized test uses the lowest error counts
            # first. Including the entire first count past the exact
            # cutoff bounds its power from above. A larger sample can
            # reproduce any smaller test by taking a uniform subset.
            boundary = _error_limit(compare_null, labels) + 1
            return compare_expected(labels, boundary) >= 0

        start = _first_unexcluded_budget(could_qualify_prefix, first, limit)
    if start > limit:
        return None, None, None

    max_errors = _error_limit(compare_null, start)
    for labels in range(start, limit + 1):
        if compare_expected(labels, max_errors) >= 0:
            return labels, max_errors, probability(labels, max_errors)
        # One more label can add at most one error, so only the next
        # count can become eligible at the next budget.
        if labels < limit and compare_null(labels + 1, max_errors + 1) <= 0:
            max_errors += 1
    return None, None, None


def _plan_population(
    size, expected_error, risk_target, accepted, confidence, power, limit
):
    failure_rate = 1 - Fraction(str(confidence))
    expected_errors = ceil(Fraction(str(expected_error)) * size)
    power = Fraction(str(power))

    def compare_expected(labels, count):
        return compare_hypergeometric_cdf(size, expected_errors, labels, count, power)

    def probability(labels, count):
        return hypergeometric_cdf_float(size, expected_errors, labels, count)

    if accepted is not None:
        if accepted > size:
            raise ValueError("accepted label count exceeds the accepted population")
        observed = _error_quantile(compare_expected, accepted)
        upper = upper_population_errors(size, accepted, observed, failure_rate)
        risk_target = round_rate_up(Fraction(upper, size))
    # Test the smallest total error count that would miss the target.
    violating_errors = floor(Fraction(str(risk_target)) * size) + 1

    def compare_null(labels, count):
        return compare_hypergeometric_cdf(
            size, violating_errors, labels, count, failure_rate
        )

    if accepted is None:
        errors = certification_probability = None
        if violating_errors > size:
            accepted, errors, certification_probability = 1, 1, 1.0
        elif expected_errors < violating_errors or power <= failure_rate:
            accepted, errors, certification_probability = _minimum_label_budget(
                compare_null,
                compare_expected,
                probability,
                min(limit, size),
                bound_prefix=expected_errors < violating_errors,
            )
    else:
        errors = accepted
        if violating_errors <= size:
            errors = _error_limit(compare_null, accepted)
        certification_probability = probability(accepted, errors)
    return {
        "accepted_population": size,
        "expected_population_errors": expected_errors,
        "risk_target": risk_target,
        "accepted": accepted,
        "max_errors": errors,
        "certification_probability": certification_probability,
    }


def plan_labels(
    *,
    expected_error,
    risk_target=None,
    accepted=None,
    holdout_confidence=0.95,
    power=0.8,
    max_accepted=DEFAULT_MAX_ACCEPTED,
    accepted_population=None,
):
    """Find the minimum accepted label count or the smallest certifiable target."""
    if (risk_target is None) == (accepted is None):
        raise ValueError("provide exactly one of risk target and accepted count")
    if not 0 <= expected_error <= 1:
        raise ValueError("expected error must be between 0 and 1")
    if not 0 < holdout_confidence < 1 or not 0 < power < 1:
        raise ValueError(
            "holdout confidence and power must be strictly between 0 and 1"
        )
    if accepted is not None:
        if type(accepted) is not int or accepted < 1:
            raise ValueError("accepted count must be a positive integer")
    else:
        if not 0 <= risk_target <= 1:
            raise ValueError("risk target must be between 0 and 1")
        if type(max_accepted) is not int or max_accepted < 1:
            raise ValueError("max accepted must be a positive integer")
    if accepted_population is not None:
        validate_population(accepted_population)
        return _plan_population(
            accepted_population,
            expected_error,
            risk_target,
            accepted,
            holdout_confidence,
            power,
            max_accepted,
        )

    power = Fraction(str(power))
    expected_rate = (
        expected_error if expected_error in (0, 1) else Fraction(str(expected_error))
    )

    def compare_expected(labels, count):
        return compare_binomial_cdf(labels, count, expected_rate, power)

    def probability(labels, count):
        return binomial_cdf_float(labels, count, expected_rate)

    if accepted is not None:
        errors = _error_quantile(compare_expected, accepted)
        risk_target = binomial_upper_bound(
            Counts(accepted, accepted, errors), holdout_confidence
        )
        certification_probability = probability(accepted, errors)
    elif risk_target == 1:
        accepted, errors, certification_probability = 1, 1, 1.0
    else:
        errors = certification_probability = None
        failure_rate = 1 - Fraction(str(holdout_confidence))
        target_rate = Fraction(str(risk_target)) if risk_target else Fraction(0)
        if expected_rate < target_rate or power <= failure_rate:

            def compare_null(labels, count):
                return compare_binomial_cdf(labels, count, target_rate, failure_rate)

            accepted, errors, certification_probability = _minimum_label_budget(
                compare_null,
                compare_expected,
                probability,
                max_accepted,
                bound_prefix=expected_rate < target_rate,
            )
    return {
        "risk_target": risk_target,
        "accepted": accepted,
        "max_errors": errors,
        "certification_probability": certification_probability,
    }
