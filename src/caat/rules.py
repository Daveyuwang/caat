"""Selection rules and the exact binomial test they use."""

from dataclasses import dataclass
from math import exp, expm1, fsum, lgamma, log, log1p

from .metrics import Counts


def log_lower_tail(n, k, log_p, log_q):
    # Scale by the probability of the boundary count to avoid overflow.
    term = mass = 1.0
    for successes in range(k, 0, -1):
        term *= exp(log(successes / (n - successes + 1)) + log_q - log_p)
        mass += term
        if term == 0:
            break
    return fsum(
        (
            lgamma(n + 1),
            -lgamma(k + 1),
            -lgamma(n - k + 1),
            k * log_p,
            (n - k) * log_q,
            log(mass),
        )
    )


def binomial_pvalue(counts: Counts, target: float) -> float:
    """Exact binomial lower-tail probability for the observed accepted errors."""
    n, k = counts.accepted, counts.errors
    if not (0 <= k <= n and 0 <= target <= 1):
        raise ValueError("invalid binomial count or probability")
    if k == n or target == 0:
        return 1.0
    if target == 1:
        return 0.0
    log_p, log_q = log(target), log1p(-target)
    if k < n * target:
        return exp(min(0.0, log_lower_tail(n, k, log_p, log_q)))
    return -expm1(min(0.0, log_lower_tail(n, n - k - 1, log_q, log_p)))


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
    return RuleResult(
        tuple(
            candidate["accepted"] > 0 and candidate["worst_error_rate"] <= risk_target
            for candidate in candidates
        )
    )


def bonferroni(candidates, risk_target, failure_rate):
    level = failure_rate / len(candidates)
    decisions = tuple(
        candidate["accepted"] > 0 and candidate["pvalue"] <= level
        for candidate in candidates
    )
    return RuleResult(decisions, level, controls_risk=True)


def fixed_sequence(candidates, risk_target, failure_rate):
    decisions = []
    for candidate in candidates:
        passed = candidate["accepted"] > 0 and candidate["pvalue"] <= failure_rate
        decisions.append(passed)
        if not passed:
            break
    untested = len(candidates) - len(decisions)
    return RuleResult(
        tuple(decisions) + (None,) * untested, failure_rate, controls_risk=True
    )


RULES = {
    "empirical": empirical,
    "ltt-bonferroni": bonferroni,
    "ltt-fixed-sequence": fixed_sequence,
}
