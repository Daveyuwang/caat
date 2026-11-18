"""Build candidate thresholds and apply a registered selection rule."""

from math import isfinite
from operator import attrgetter
from typing import Any, Iterable, Sequence

from .metrics import Counts
from .rules import RULES, RuleResult, binomial_pvalue
from .table import GradedRow, UnscoredRow, validated_rows


DEFAULT_DESIGN_GRID_SIZE = 10


def design_thresholds(
    scores, *, grid_size: int = DEFAULT_DESIGN_GRID_SIZE, review_missing_scores=False
) -> list[float]:
    """Return distinct cutoffs at evenly spaced ranks in the design scores."""
    if type(grid_size) is not int or grid_size < 1:
        raise ValueError("design grid size must be a positive integer")
    scores = list(scores)
    if not scores or any(
        not (review_missing_scores if score is None else isfinite(score))
        for score in scores
    ):
        raise ValueError(
            "design needs at least one score and all design scores must be finite"
        )
    if review_missing_scores:
        scores = [score for score in scores if score is not None]
        if not scores:
            return []
    scores.sort(reverse=True)
    count = len(scores)
    grid_size = min(grid_size, count)
    ranks = ((count * i - 1) // grid_size for i in range(1, grid_size + 1))
    return sorted({scores[rank] for rank in ranks}, reverse=True)


def build_threshold_candidates(
    rows, *, graders, thresholds, risk_target, review_missing_scores=False
):
    """Count calibration outcomes for a fixed score and grid before testing."""
    thresholds = sorted(set(thresholds), reverse=True)
    if (not thresholds and not review_missing_scores) or any(
        not isfinite(threshold) for threshold in thresholds
    ):
        raise ValueError("provide at least one finite threshold")
    rows = validated_rows(rows, graders, review_missing_scores=review_missing_scores)
    total = len(rows)
    if review_missing_scores:
        rows = [row for row in rows if row.score is not None]
    rows.sort(key=attrgetter("score"), reverse=True)
    accepted, errors = 0, [0] * len(graders)
    candidates = []
    for threshold in thresholds:
        while accepted < len(rows) and rows[accepted].score >= threshold:
            for index, correct in enumerate(rows[accepted].outcomes):
                errors[index] += not correct
            accepted += 1
        pvalue = binomial_pvalue(Counts(total, accepted, max(errors)), risk_target)
        candidates.append(
            {
                "threshold": threshold,
                "accepted": accepted,
                "coverage": accepted / total,
                "errors": dict(zip(graders, errors, strict=True)),
                "worst_error_rate": max(errors) / accepted if accepted else None,
                "pvalue": pvalue,
            }
        )
    return candidates, total


def test_candidates(candidates, *, rule, risk_target, failure_rate):
    """Apply one registered rule to the complete family and record its decisions."""
    component = RULES[rule]
    if not candidates:
        return RuleResult(())
    decision = component.choose(candidates, risk_target, failure_rate)
    if (
        not isinstance(decision, RuleResult)
        or len(decision.decisions) != len(candidates)
        or any(
            value is not None and type(value) is not bool
            for value in decision.decisions
        )
        or (decision.controls_risk and not component.assumptions)
    ):
        raise ValueError(f"selection rule {rule!r} returned invalid decisions")
    for candidate, passes in zip(candidates, decision.decisions, strict=True):
        candidate.update(tested=passes is not None, passes=passes)
    return decision


def select(
    rows: Iterable[GradedRow | UnscoredRow],
    *,
    graders: Sequence[str],
    thresholds: Iterable[float],
    risk_target: float,
    failure_rate: float = 0.05,
    rule: str = "ltt-bonferroni",
    review_missing_scores: bool = False,
) -> dict[str, Any]:
    """Select a cutoff using a fixed candidate list and paired calibration labels."""
    if not 0 <= risk_target <= 1:
        raise ValueError("risk target must be between 0 and 1")
    if not 0 < failure_rate < 1:
        raise ValueError("failure rate must be strictly between 0 and 1")
    if rule not in RULES:
        raise ValueError(f"unknown selection rule {rule!r}")
    candidates, total = build_threshold_candidates(
        rows,
        graders=graders,
        thresholds=thresholds,
        risk_target=risk_target,
        review_missing_scores=review_missing_scores,
    )
    decision = test_candidates(
        candidates, rule=rule, risk_target=risk_target, failure_rate=failure_rate
    )
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
            "assumptions": RULES[rule].assumptions,
        }
        if decision.controls_risk and selected is not None
        else None,
        "calibration_total": total,
        "selected_threshold": selected,
        "candidates": candidates,
    }
