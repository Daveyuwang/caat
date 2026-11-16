from math import ceil, exp, expm1, fsum, isfinite, lgamma, log, log1p
from operator import attrgetter

from .audit import validated_rows
from .evaluation import Counts


RULES = ("empirical", "ltt-bonferroni", "ltt-fixed-sequence")


def log_lower_tail(n, k, log_p, log_q):
    # Scale by the probability of the boundary count to avoid overflow.
    term = mass = 1.0
    for successes in range(k, 0, -1):
        term *= exp(log(successes / (n - successes + 1)) + log_q - log_p)
        mass += term
        if term == 0:
            break
    return fsum((lgamma(n + 1), -lgamma(k + 1), -lgamma(n - k + 1),
                 k * log_p, (n - k) * log_q, log(mass)))


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


def design_thresholds(scores) -> list[float]:
    scores = list(scores)
    if not scores or any(not isfinite(score) for score in scores):
        raise ValueError("design needs at least one score and all design scores must be finite")
    scores.sort(reverse=True)
    return sorted({scores[ceil(len(scores) * i / 10) - 1] for i in range(1, 11)}, reverse=True)


def select_threshold(rows, *, graders, thresholds, risk_target, failure_rate=0.05, rule="ltt-bonferroni"):
    """Select a cutoff using a fixed candidate list and paired calibration labels."""
    if not 0 <= risk_target <= 1:
        raise ValueError("risk target must be between 0 and 1")
    if not 0 < failure_rate < 1:
        raise ValueError("failure rate must be strictly between 0 and 1")
    if rule not in RULES:
        raise ValueError(f"unknown selection rule {rule!r}")
    thresholds = sorted(set(thresholds), reverse=True)
    if not thresholds or any(not isfinite(threshold) for threshold in thresholds):
        raise ValueError("provide at least one finite threshold")
    rows = validated_rows(rows, graders)
    rows.sort(key=attrgetter("score"), reverse=True)
    limit = failure_rate / len(thresholds) if rule == "ltt-bonferroni" else failure_rate
    accepted, errors = 0, [0] * len(graders)
    selected, stopped = None, False
    candidates = []
    for threshold in thresholds:
        while accepted < len(rows) and rows[accepted].score >= threshold:
            for index, correct in enumerate(rows[accepted].outcomes):
                errors[index] += not correct
            accepted += 1
        pvalue = binomial_pvalue(Counts(len(rows), accepted, max(errors)), risk_target)
        tested = not stopped
        passes = None
        if tested:
            criterion_met = max(errors) / accepted <= risk_target if rule == "empirical" and accepted else pvalue <= limit
            passes = accepted > 0 and criterion_met
            if passes:
                selected = threshold
            elif rule == "ltt-fixed-sequence":
                stopped = True
        candidates.append({
            "threshold": threshold,
            "accepted": accepted,
            "coverage": accepted / len(rows),
            "errors": dict(zip(graders, errors, strict=True)),
            "worst_error_rate": max(errors) / accepted if accepted else None,
            "pvalue": pvalue,
            "tested": tested,
            "passes": passes,
        })
    return {
        "schema_version": 1,
        "rule": rule,
        "graders": list(graders),
        "risk_target": risk_target,
        "failure_rate": failure_rate,
        "per_candidate_level": limit if rule != "empirical" else None,
        "guarantee": {
            "confidence": 1 - failure_rate,
            "assumptions": "IID calibration from the target population and independent candidate thresholds",
        } if rule != "empirical" and selected is not None else None,
        "calibration_total": len(rows),
        "selected_threshold": selected,
        "candidates": candidates,
    }
