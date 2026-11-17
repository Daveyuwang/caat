"""Compare Learn then Test controllers on shared HELM calibration cutoffs."""

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np
from mapie.risk_control import BinaryClassificationController
from mapie.risk_control.fwer_control import control_fwer
from mapie.risk_control.methods import compute_hoeffding_bentkus_p_value
from scipy.stats import binom

from caat import GradedRow, select
from caat.select import design_thresholds

from common import ARCHIVES, ROOT, write_report
from study_decision_baselines import (
    CONFIDENCE,
    FAMILY_WISE_FAILURE_RATE,
    OMNI,
    load_case,
    measure,
)


CORRECTIONS = {"ltt-bonferroni": "bonferroni", "ltt-holm": "bonferroni_holm"}


def probabilities(values):
    score = np.asarray(values).ravel()
    return np.column_stack((1 - score, score))


def holdout_row(case, rule, tool, cutoff, qualified, holdout, target):
    primary = measure(holdout, cutoff, "correct", target)
    alternate = measure(holdout, cutoff, "alternate_correct", target)
    return {
        "case": case,
        "rule": rule,
        "tool": tool,
        "target": target,
        "qualified_cutoffs": qualified,
        "threshold": cutoff,
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


def compare_case(case, partitions, target):
    design, calibration, holdout = (
        partitions[part] for part in ("design", "calibration", "holdout")
    )
    thresholds = np.array(
        sorted(design_thresholds(row["confidence"] for row in design))
    )
    scores = np.array([row["confidence"] for row in calibration])
    correct = np.array([row["correct"] == "1" for row in calibration])
    test_scores = np.array([row["confidence"] for row in holdout])
    accepted = scores[None, :] >= thresholds[:, None]
    n = accepted.sum(axis=1)
    errors = (accepted & ~correct[None, :]).sum(axis=1)
    exact = binom.cdf(errors, n, target)
    precision = np.divide(
        n - errors, n, out=np.zeros_like(n, dtype=float), where=n != 0
    )
    risk = 1 - precision
    reconstructed_errors = np.ceil(n * risk).astype(int)
    reconstructed_p = compute_hoeffding_bentkus_p_value(
        risk, n, target, binary=True
    ).ravel()
    rows = [
        GradedRow(row["id"], row["confidence"], (row["correct"] == "1",))
        for row in calibration
    ]
    policies, candidates = [], []
    for rule, correction in CORRECTIONS.items():
        caat = select(
            rows,
            graders=("correct",),
            thresholds=thresholds.tolist(),
            risk_target=target,
            failure_rate=FAMILY_WISE_FAILURE_RATE,
            rule=rule,
        )
        exact_qualified = set(control_fwer(exact, FAMILY_WISE_FAILURE_RATE, correction))
        mapie = BinaryClassificationController(
            probabilities,
            "precision",
            list_predict_params=thresholds,
            target_level=1 - target,
            confidence_level=CONFIDENCE,
            fwer_method=correction,
            best_predict_param_choice="abstention_rate",
        ).calibrate(scores[:, None], correct.astype(int))
        native_p = mapie.p_values.ravel()
        np.testing.assert_allclose(native_p, reconstructed_p, atol=1e-12, rtol=0)
        qualified = set(mapie.valid_predict_params)
        for index, (threshold, candidate) in enumerate(
            zip(thresholds, reversed(caat["candidates"]), strict=True)
        ):
            caat_pass = candidate["passes"] is True
            if (
                candidate["threshold"] != threshold
                or candidate["accepted"] != n[index]
                or candidate["errors"]["correct"] != errors[index]
                or not math.isclose(
                    candidate["pvalue"], exact[index], rel_tol=0, abs_tol=1e-12
                )
                or caat_pass != (index in exact_qualified)
            ):
                raise ValueError(f"{case}/{rule}: reference disagrees at {threshold}")
            candidates.append(
                {
                    "case": case,
                    "rule": rule,
                    "threshold": float(threshold),
                    "calibration_accepted": int(n[index]),
                    "calibration_errors": int(errors[index]),
                    "mapie_reconstructed_errors": int(reconstructed_errors[index]),
                    "exact_pvalue": float(exact[index]),
                    "mapie_pvalue": float(native_p[index]),
                    "caat_pass": caat_pass,
                    "mapie_pass": threshold in qualified,
                }
            )
        native_threshold = mapie.best_predict_param
        if native_threshold is not None:
            if int(np.sum(scores >= native_threshold)) != max(
                int(np.sum(scores >= cutoff)) for cutoff in qualified
            ):
                raise ValueError(
                    f"{case}/{rule}: MAPIE did not select for maximum coverage"
                )
            np.testing.assert_array_equal(
                mapie.predict(test_scores[:, None]),
                (test_scores >= native_threshold).astype(int),
            )
            native_threshold = float(native_threshold)
        for tool, cutoff, count in (
            ("caat", caat["selected_threshold"], len(exact_qualified)),
            ("mapie", native_threshold, len(qualified)),
        ):
            policies.append(
                holdout_row(case, rule, tool, cutoff, count, holdout, target)
            )
    return policies, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "replay/mapie-baselines"
    )
    args = parser.parse_args()
    protocol = {
        "versions": {
            name: version(name)
            for name in ("caat-annotation", "mapie", "numpy", "scipy", "scikit-learn")
        },
        "score": "exp(mean_token_logprob)",
        "grid": "design-score deciles",
        "caat_failure_rate": FAMILY_WISE_FAILURE_RATE,
        "mapie_confidence": CONFIDENCE,
        "holdout_confidence": CONFIDENCE,
        "cases": {},
    }
    comparisons, candidates = [], []
    for case in (*ARCHIVES, OMNI):
        partitions, metadata = load_case(case)
        policies, tested = compare_case(case, partitions, metadata["risk_target"])
        protocol["cases"][case] = metadata
        comparisons.extend(policies)
        candidates.extend(tested)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_report(args.output_dir / "comparison.csv", comparisons)
    write_report(args.output_dir / "candidates.csv", candidates)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
