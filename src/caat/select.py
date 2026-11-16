"""Build candidate thresholds and apply a registered selection rule."""

from math import ceil, isfinite
from operator import attrgetter

from .metrics import Counts
from .rules import RULES, binomial_pvalue
from .table import validated_rows


def design_thresholds(scores) -> list[float]:
    scores = list(scores)
    if not scores or any(not isfinite(score) for score in scores):
        raise ValueError(
            "design needs at least one score and all design scores must be finite"
        )
    scores.sort(reverse=True)
    return sorted(
        {scores[ceil(len(scores) * i / 10) - 1] for i in range(1, 11)}, reverse=True
    )


def select(
    rows, *, graders, thresholds, risk_target, failure_rate=0.05, rule="ltt-bonferroni"
):
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
    accepted, errors = 0, [0] * len(graders)
    candidates = []
    for threshold in thresholds:
        while accepted < len(rows) and rows[accepted].score >= threshold:
            for index, correct in enumerate(rows[accepted].outcomes):
                errors[index] += not correct
            accepted += 1
        pvalue = binomial_pvalue(Counts(len(rows), accepted, max(errors)), risk_target)
        candidates.append(
            {
                "threshold": threshold,
                "accepted": accepted,
                "coverage": accepted / len(rows),
                "errors": dict(zip(graders, errors, strict=True)),
                "worst_error_rate": max(errors) / accepted if accepted else None,
                "pvalue": pvalue,
            }
        )
    decision = RULES[rule](candidates, risk_target, failure_rate)
    for candidate, passes in zip(candidates, decision.decisions, strict=True):
        candidate.update(tested=passes is not None, passes=passes)
    index = decision.selected_index
    selected = candidates[index]["threshold"] if index is not None else None
    return {
        "schema_version": 1,
        "rule": rule,
        "graders": list(graders),
        "risk_target": risk_target,
        "failure_rate": failure_rate,
        "per_candidate_level": decision.per_candidate_level,
        "guarantee": {
            "confidence": 1 - failure_rate,
            "assumptions": "IID calibration from the target population and independent candidate thresholds",
        }
        if decision.controls_risk and selected is not None
        else None,
        "calibration_total": len(rows),
        "selected_threshold": selected,
        "candidates": candidates,
    }
