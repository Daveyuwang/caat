from operator import attrgetter
from random import Random
from typing import Any, Iterable, Sequence

from .audit import audit
from .calibrate import calibration_transform, default_scale, fit_calibration
from .metrics import Counts, calibration_metrics, ranking_metrics, threshold_points
from .rules import assess_accepted_risk, validate_holdout_confidence
from .select import DEFAULT_DESIGN_GRID_SIZE, design_thresholds, select
from .table import GradedRow, UnscoredRow, accepted_at, graded_row, validated_rows


def split_rows(rows, seed, *, id_key=attrgetter("id")):
    """Sort by ID, shuffle, and split about 20/40/40 into design, calibration, and held-out rows."""
    rows = sorted(rows, key=id_key)
    Random(seed).shuffle(rows)
    first, second = len(rows) // 5, 3 * len(rows) // 5
    return rows[:first], rows[first:second], rows[second:]


def policy(
    rows: Iterable[GradedRow | UnscoredRow],
    *,
    graders: Sequence[str],
    check_graders: Sequence[str] = (),
    risk_target: float,
    holdout_confidence: float | None = None,
    failure_rate: float = 0.05,
    rule: str = "ltt-bonferroni",
    seed: int = 42,
    calibrate: str | None = None,
    calibration_input: str | None = None,
    design_grid_size: int = DEFAULT_DESIGN_GRID_SIZE,
    review_missing_scores: bool = False,
) -> dict[str, Any]:
    """Select a cutoff on calibration rows and replay it unchanged on held-out rows.

    Each row holds one outcome per grader, selection graders first. The check
    graders take no part in selection and are reported on the held-out rows.
    """
    validate_holdout_confidence(holdout_confidence)
    everyone = (*graders, *check_graders)
    rows = validated_rows(rows, everyone, review_missing_scores=review_missing_scores)
    if not graders:
        raise ValueError("provide at least one grader for selection")
    if len(rows) < 5:
        raise ValueError(
            "at least five rows are required for the design, calibration, and held-out splits"
        )
    design, calibration, heldout = split_rows(rows, seed)
    return _policy_for_splits(
        design,
        calibration,
        heldout,
        graders=graders,
        check_graders=check_graders,
        calibration_graders=everyone,
        risk_target=risk_target,
        holdout_confidence=holdout_confidence,
        failure_rate=failure_rate,
        rule=rule,
        split_seed=seed,
        calibrate=calibrate,
        calibration_input=calibration_input,
        design_grid_size=design_grid_size,
        review_missing_scores=review_missing_scores,
    )


def _policy_for_splits(
    design,
    calibration,
    heldout,
    *,
    graders,
    check_graders,
    calibration_graders,
    risk_target,
    holdout_confidence,
    failure_rate,
    rule,
    split_seed,
    calibrate,
    calibration_input,
    design_grid_size,
    selection=None,
    review_missing_scores=False,
):
    """Fit and check a policy using validated partitions."""
    everyone = (*graders, *check_graders)
    if selection is None:
        thresholds = design_thresholds(
            (row.score for row in design),
            grid_size=design_grid_size,
            review_missing_scores=review_missing_scores,
        )
        selection = select(
            (
                graded_row(row.id, row.score, row.outcomes[: len(graders)])
                for row in calibration
            ),
            graders=graders,
            thresholds=thresholds,
            risk_target=risk_target,
            failure_rate=failure_rate,
            rule=rule,
            review_missing_scores=review_missing_scores,
        )
    scored_calibration = (
        [row for row in calibration if row.score is not None]
        if review_missing_scores
        else calibration
    )
    scored_heldout = (
        [row for row in heldout if row.score is not None]
        if review_missing_scores
        else heldout
    )
    ranking = {
        name: ranking_metrics(
            threshold_points(
                (row.score, not row.outcomes[index]) for row in scored_calibration
            )
        )
        for index, name in enumerate(calibration_graders)
    }
    threshold = selection["selected_threshold"]
    check = None
    if threshold is not None and len(everyone) > 1:
        paired = audit(
            heldout,
            graders=everyone,
            threshold=threshold,
            risk_target=risk_target,
            holdout_confidence=holdout_confidence,
            review_missing_scores=review_missing_scores,
        )
        check = {
            "graders": {
                name: {
                    key: value for key, value in summary.items() if key != "by_group"
                }
                for name, summary in paired["graders"].items()
            },
            "accepted_comparisons": paired["accepted_comparisons"],
        }
    elif threshold is not None:
        accepted = [row for row in heldout if accepted_at(row, threshold)]
        counts = Counts(
            len(heldout),
            len(accepted),
            sum(not row.outcomes[0] for row in accepted),
        )
        check = {
            "graders": {
                everyone[0]: assess_accepted_risk(
                    counts, risk_target, holdout_confidence
                )
            }
        }
    report = {
        "schema_version": 1,
        "split": {
            "seed": split_seed,
            "design": len(design),
            "calibration": len(calibration),
            "heldout": len(heldout),
        },
        "check_graders": list(check_graders),
        "ranking": ranking,
        "selection": selection,
        "heldout": check,
        "confirmed": check is not None
        and all(result["meets_target"] is True for result in check["graders"].values()),
    }
    if holdout_confidence is not None:
        report["holdout_confidence"] = holdout_confidence
    if review_missing_scores:
        report["scored"] = {
            "design": sum(row.score is not None for row in design),
            "calibration": len(scored_calibration),
            "heldout": len(scored_heldout),
        }
    if calibrate is not None and scored_calibration:
        scale = calibration_input or default_scale(calibrate)
        estimates = {}
        for index, name in enumerate(graders):
            mapping = fit_calibration(
                (row.score for row in scored_calibration),
                (row.outcomes[index] for row in scored_calibration),
                method=calibrate,
                input_scale=scale,
            )
            transform = calibration_transform(mapping)
            grades = [row.outcomes[index] for row in scored_heldout]
            scores = [row.score for row in scored_heldout]
            estimates[name] = {
                "map": mapping,
                "heldout_uncalibrated": calibration_metrics(
                    scores, grades, allow_empty=review_missing_scores
                )
                if scale == "probability"
                else None,
                "heldout_calibrated": calibration_metrics(
                    map(transform, scores),
                    grades,
                    allow_empty=review_missing_scores,
                ),
            }
        report["probability_calibration"] = {
            "method": calibrate,
            "threshold_scale": "raw",
            "graders": estimates,
        }
    return report
