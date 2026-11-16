from operator import attrgetter
from random import Random

from .audit import audit
from .calibrate import calibration_transform, default_scale, fit_calibration
from .metrics import Counts, calibration_metrics, ranking_metrics, threshold_points
from .select import design_thresholds, select
from .table import GradedRow, validated_rows


def split_rows(rows, seed, *, id_key=attrgetter("id")):
    """Sort by ID, shuffle, and split about 20/40/40 into design, calibration, and held-out rows."""
    rows = sorted(rows, key=id_key)
    Random(seed).shuffle(rows)
    first, second = len(rows) // 5, 3 * len(rows) // 5
    return rows[:first], rows[first:second], rows[second:]


def policy(
    rows,
    *,
    graders,
    check_graders=(),
    risk_target,
    failure_rate=0.05,
    rule="ltt-bonferroni",
    seed=42,
    calibrate=None,
    calibration_input=None,
) -> dict:
    """Select a cutoff on calibration rows and replay it unchanged on held-out rows.

    Each row holds one outcome per grader, selection graders first. The check
    graders take no part in selection and are reported on the held-out rows.
    """
    everyone = (*graders, *check_graders)
    rows = validated_rows(rows, everyone)
    if not graders:
        raise ValueError("provide at least one grader for selection")
    if len(rows) < 5:
        raise ValueError(
            "at least five rows are required for the design, calibration, and held-out splits"
        )
    design, calibration, heldout = split_rows(rows, seed)
    ranking = {
        name: ranking_metrics(
            threshold_points(
                (row.score, not row.outcomes[index]) for row in calibration
            )
        )
        for index, name in enumerate(everyone)
    }
    selection = select(
        (
            GradedRow(row.id, row.score, row.outcomes[: len(graders)])
            for row in calibration
        ),
        graders=graders,
        thresholds=design_thresholds(row.score for row in design),
        risk_target=risk_target,
        failure_rate=failure_rate,
        rule=rule,
    )
    threshold = selection["selected_threshold"]
    check = None
    if threshold is not None:
        accepted = [row for row in heldout if row.score >= threshold]
        results = {}
        for index, name in enumerate(everyone):
            counts = Counts(
                len(heldout),
                len(accepted),
                sum(not row.outcomes[index] for row in accepted),
            )
            risk = counts.error_rate
            results[name] = {
                **counts.metrics(),
                "meets_target": risk <= risk_target if risk is not None else None,
            }
        check = {"graders": results}
        if len(everyone) > 1:
            check["accepted_comparisons"] = audit(
                heldout, graders=everyone, threshold=threshold, risk_target=risk_target
            )["accepted_comparisons"]
    report = {
        "schema_version": 1,
        "split": {
            "seed": seed,
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
    if calibrate is not None:
        scale = calibration_input or default_scale(calibrate)
        estimates = {}
        for index, name in enumerate(graders):
            mapping = fit_calibration(
                (row.score for row in calibration),
                (row.outcomes[index] for row in calibration),
                method=calibrate,
                input_scale=scale,
            )
            transform = calibration_transform(mapping)
            grades = [row.outcomes[index] for row in heldout]
            scores = [row.score for row in heldout]
            estimates[name] = {
                "map": mapping,
                "heldout_uncalibrated": calibration_metrics(scores, grades)
                if scale == "probability"
                else None,
                "heldout_calibrated": calibration_metrics(
                    map(transform, scores), grades
                ),
            }
        report["probability_calibration"] = {
            "method": calibrate,
            "threshold_scale": "raw",
            "graders": estimates,
        }
    return report
