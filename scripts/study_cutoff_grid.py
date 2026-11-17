"""Replay the candidate-grid and pointwise-test differences in threshold fitting."""

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path

from calibroute import PredictionRecord, clopper_pearson_upper, fit_policy

from caat import GradedRow, select
from caat.select import design_thresholds

from common import ARCHIVES, ROOT, write_report
from study_decision_baselines import (
    CONFIDENCE,
    FAMILY_WISE_FAILURE_RATE,
    OMNI,
    PARTITIONS,
    load_case,
    load_inspect,
    measure,
)


BONFERRONI = "ltt-bonferroni"
HOLM = "ltt-holm"
DESIGN_POINTWISE = "design-pointwise-clopper-pearson"
CALIBROUTE = "calibroute-clopper-pearson"


def design_pointwise(calibration, thresholds, target):
    cutoff = None
    for threshold in thresholds:
        counts = measure(calibration, threshold, "correct", target)
        if counts["accepted"]:
            bound = clopper_pearson_upper(
                counts["errors"], counts["accepted"], CONFIDENCE
            )
            if not math.isclose(
                counts["risk_upper_bound"], bound, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(f"Clopper–Pearson bounds disagree at {threshold}")
            if bound <= target:
                cutoff = threshold
    return cutoff


def native_cutoff(calibration, target):
    records = [
        PredictionRecord(row["id"], row["confidence"], row["correct"] == "1")
        for row in calibration
    ]
    try:
        policy = fit_policy(
            records,
            max_risk=target,
            min_coverage=1 / len(records),
            risk_method="clopper_pearson",
            confidence_level=CONFIDENCE,
        )
    except ValueError as exc:
        if not str(exc).startswith("no validation operating point "):
            raise
        return None
    return policy.accept_threshold


def holdout_record(case, rule, threshold, target, calibration, holdout):
    fitted = measure(calibration, threshold, "correct", target)
    primary = measure(holdout, threshold, "correct", target)
    alternate = measure(holdout, threshold, "alternate_correct", target)
    return {
        "case": case,
        "rule": rule,
        "target": target,
        "threshold": threshold,
        "calibration_accepted": fitted["accepted"],
        "calibration_errors": fitted["errors"],
        "calibration_upper_95": fitted["risk_upper_bound"],
        "holdout": primary["total"],
        "accepted": primary["accepted"],
        "coverage": primary["coverage"],
        "primary_errors": primary["errors"],
        "primary_risk": primary["error_rate"],
        "primary_upper_95": primary["risk_upper_bound"],
        "primary_met_target": primary["meets_target"],
        "alternate_errors": alternate["errors"],
        "alternate_risk": alternate["error_rate"],
        "alternate_upper_95": alternate["risk_upper_bound"],
        "alternate_met_target": alternate["meets_target"],
    }


def compare(case, splits, target):
    design, calibration, holdout = (splits[part] for part in PARTITIONS)
    thresholds = design_thresholds(row["confidence"] for row in design)
    graded = [
        GradedRow(row["id"], row["confidence"], (row["correct"] == "1",))
        for row in calibration
    ]
    cutoffs = {}
    for rule in (BONFERRONI, HOLM):
        result = select(
            graded,
            graders=("correct",),
            thresholds=thresholds,
            risk_target=target,
            failure_rate=FAMILY_WISE_FAILURE_RATE,
            rule=rule,
        )
        cutoffs[rule] = result["selected_threshold"]
    cutoffs[DESIGN_POINTWISE] = design_pointwise(calibration, thresholds, target)
    cutoffs[CALIBROUTE] = native_cutoff(calibration, target)
    rows = [
        holdout_record(case, rule, cutoff, target, calibration, holdout)
        for rule, cutoff in cutoffs.items()
    ]
    accepted = {row["rule"]: row["accepted"] for row in rows}
    pointwise = accepted[DESIGN_POINTWISE]
    native = accepted[CALIBROUTE]
    changes = {
        "case": case,
        "target": target,
        "holdout": len(holdout),
        "bonferroni_accepted": accepted[BONFERRONI],
        "holm_accepted": accepted[HOLM],
        "design_pointwise_accepted": pointwise,
        "calibroute_accepted": native,
        "pointwise_minus_bonferroni": pointwise - accepted[BONFERRONI],
        "pointwise_minus_holm": pointwise - accepted[HOLM],
        "calibration_grid_minus_design_grid": native - pointwise,
    }
    return rows, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspect-splits",
        type=Path,
        help="Original Inspect design, calibration, and holdout CSVs.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replay/cutoff-grid")
    args = parser.parse_args()
    protocol = {
        "caat-annotation": version("caat-annotation"),
        "calibroute-ai": version("calibroute-ai"),
        "helm_score": "exp(mean_token_logprob)",
        "caat_failure_rate": FAMILY_WISE_FAILURE_RATE,
        "pointwise_confidence": CONFIDENCE,
        "holdout_confidence": CONFIDENCE,
        "cases": {},
    }
    results, changes = [], []
    cases = {case: load_case(case) for case in (*ARCHIVES, OMNI)}
    if args.inspect_splits is not None:
        cases["inspect_triviaqa"] = load_inspect(args.inspect_splits)
    for case, (splits, details) in cases.items():
        rows, differences = compare(case, splits, details["risk_target"])
        protocol["cases"][case] = details
        results.extend(rows)
        changes.append(differences)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_report(args.output_dir / "comparison.csv", results)
    write_report(args.output_dir / "coverage_changes.csv", changes)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
