"""Selection rules and the exact binomial test they use."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import groupby
from typing import Callable

from .finite import (
    binomial_cdf_float,
    compare_binomial_cdf,
    round_rate_up,
    upper_binomial_rate,
    upper_population_errors,
    validate_population,
)
from .metrics import Counts
from .registry import Registry


def binomial_pvalue(counts: Counts, target: float) -> float:
    """Exact binomial lower-tail probability for the observed accepted errors."""
    n, k = counts.accepted, counts.errors
    if not (0 <= k <= n and 0 <= target <= 1):
        raise ValueError("invalid binomial count or probability")
    return binomial_cdf_float(n, k, Fraction(str(target)))


def validate_holdout_confidence(confidence, *, accepted_population=None):
    if confidence is not None and not 0 < confidence < 1:
        raise ValueError("holdout confidence must be strictly between 0 and 1")
    if accepted_population is not None:
        validate_population(accepted_population)
        if confidence is None:
            raise ValueError("accepted population requires holdout confidence")


def binomial_upper_bound(counts: Counts, confidence: float) -> float | None:
    """Return the smallest representable risk rejected by the exact test."""
    if not counts.accepted:
        return None
    if counts.errors == counts.accepted:
        return 1.0
    failure_rate = 1 - Fraction(str(confidence))
    return upper_binomial_rate(counts.accepted, counts.errors, failure_rate)


def assess_accepted_risk(
    counts: Counts, risk_target, holdout_confidence, *, accepted_population=None
):
    """Report observed risk and check the requested quantity against the target."""
    report = counts.metrics()
    risk = None
    if accepted_population is not None:
        if counts.accepted > accepted_population:
            raise ValueError("accepted label count exceeds the accepted population")
        upper = (
            upper_population_errors(
                accepted_population,
                counts.accepted,
                counts.errors,
                1 - Fraction(str(holdout_confidence)),
            )
            if counts.accepted
            else None
        )
        report["risk_upper_errors"] = upper
        risk = Fraction(upper, accepted_population) if upper is not None else None
        report["risk_upper_bound"] = round_rate_up(risk) if risk is not None else None
    elif holdout_confidence is not None:
        bound = binomial_upper_bound(counts, holdout_confidence)
        report["risk_upper_bound"] = bound
        risk = Fraction(str(bound)) if bound is not None else None
    elif counts.accepted:
        risk = Fraction(counts.errors, counts.accepted)
    report["meets_target"] = (
        risk <= Fraction(str(risk_target))
        if risk is not None and risk_target is not None
        else None
    )
    return report


@dataclass(frozen=True)
class RuleResult:
    """Acceptance decisions in threshold order. None marks an untested candidate."""

    decisions: tuple[bool | None, ...]
    per_candidate_level: float | None = None
    controls_risk: bool = False

    @property
    def selected_index(self):
        return max(
            (index for index, passed in enumerate(self.decisions) if passed),
            default=None,
        )


def empirical(candidates, risk_target, failure_rate):
    numerator, denominator = Fraction(str(risk_target)).as_integer_ratio()
    return RuleResult(
        tuple(
            candidate["accepted"] > 0
            and max(candidate["errors"].values()) * denominator
            <= candidate["accepted"] * numerator
            for candidate in candidates
        )
    )


def _passes_binomial_test(candidate, target, level):
    if not candidate["accepted"]:
        return False
    rounded_level = float(level)
    if candidate["pvalue"] != rounded_level:
        # Rounding preserves strict order when the reported values differ.
        return candidate["pvalue"] < rounded_level
    return (
        compare_binomial_cdf(
            candidate["accepted"], max(candidate["errors"].values()), target, level
        )
        <= 0
    )


def bonferroni(candidates, risk_target, failure_rate):
    level = Fraction(str(failure_rate)) / len(candidates)
    target = Fraction(str(risk_target))
    decisions = tuple(
        _passes_binomial_test(candidate, target, level) for candidate in candidates
    )
    return RuleResult(decisions, float(level), controls_risk=True)


def fixed_sequence(candidates, risk_target, failure_rate):
    level, target = Fraction(str(failure_rate)), Fraction(str(risk_target))
    decisions = []
    for candidate in candidates:
        passed = _passes_binomial_test(candidate, target, level)
        decisions.append(passed)
        if not passed:
            break
    untested = len(candidates) - len(decisions)
    return RuleResult(
        tuple(decisions) + (None,) * untested, failure_rate, controls_risk=True
    )


def holm(candidates, risk_target, failure_rate):
    """Test the smallest p-values first at Holm's step-down levels."""
    error_budget, target = Fraction(str(failure_rate)), Fraction(str(risk_target))
    remaining = len(candidates)

    def reported_pvalue(index):
        return candidates[index]["pvalue"]

    order = sorted(range(remaining), key=reported_pvalue)
    decisions = [None] * len(candidates)
    for _, tied in groupby(order, key=reported_pvalue):
        pending = list(tied)
        while pending:
            level = error_budget / remaining
            failed = []
            # Rounded ties may straddle this level. Test every candidate that
            # can pass before raising the level for the rest of the group.
            for index in pending:
                if _passes_binomial_test(candidates[index], target, level):
                    decisions[index] = True
                else:
                    failed.append(index)
            passed = len(pending) - len(failed)
            if not passed:
                decisions[failed[0]] = False
                return RuleResult(tuple(decisions), controls_risk=True)
            remaining -= passed
            pending = failed
    return RuleResult(tuple(decisions), controls_risk=True)


@dataclass(frozen=True)
class Rule:
    choose: Callable
    assumptions: str | None = None


LTT_ASSUMPTIONS = (
    "IID calibration from the target population and independent candidate thresholds"
)
RULES = Registry(
    "rules",
    Rule,
    {
        "empirical": Rule(empirical),
        "ltt-bonferroni": Rule(bonferroni, LTT_ASSUMPTIONS),
        "ltt-fixed-sequence": Rule(fixed_sequence, LTT_ASSUMPTIONS),
        "ltt-holm": Rule(holm, LTT_ASSUMPTIONS),
    },
)
