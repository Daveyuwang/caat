"""Compare primary, per-grader, and either-grader calibration targets."""

import argparse
import json
import math
import sys
from fractions import Fraction
from importlib.metadata import version
from itertools import accumulate, product
from pathlib import Path

from common import ARCHIVES, ROOT, write_report
from study_decision_baselines import (
    CONFIDENCE,
    FAMILY_WISE_FAILURE_RATE,
    OMNI,
    PARTITIONS,
    call,
    load_case,
    load_inspect,
    measure,
    verify_counts,
)


DESIGN_QUANTILES = (10, 20)
RULES = ("ltt-bonferroni", "ltt-holm")
FIT_COLUMNS = {
    "primary-only": ("correct",),
    "each-grader": ("correct", "alternate_correct"),
    "either-grader-error": ("both_correct",),
}
AUDIT_COLUMNS = {
    "primary": "correct",
    "alternate": "alternate_correct",
    "either": "both_correct",
}
CLI = (
    sys.executable,
    "-I",
    "-c",
    "from caat.cli import main; raise SystemExit(main())",
)


def exact_pvalue(count, errors, target):
    if errors == count:
        return Fraction(1)
    rate = Fraction(str(target))
    return sum(
        math.comb(count, k) * rate**k * (1 - rate) ** (count - k)
        for k in range(errors + 1)
    )


def check_fit(report, calibration, design, columns, size, rule, target):
    scores = sorted((row["confidence"] for row in design), reverse=True)
    thresholds = sorted(
        {
            scores[math.ceil(len(scores) * index / size) - 1]
            for index in range(1, size + 1)
        },
        reverse=True,
    )
    candidates = report["candidates"]
    pvalues = []
    for cutoff, candidate in zip(thresholds, candidates, strict=True):
        kept = [row for row in calibration if row["confidence"] >= cutoff]
        errors = {name: sum(row[name] == "0" for row in kept) for name in columns}
        pvalue = max(
            exact_pvalue(len(kept), count, target) for count in errors.values()
        )
        verify_counts(
            f"{rule} candidate {cutoff}",
            candidate,
            {
                "threshold": cutoff,
                "accepted": len(kept),
                "errors": errors,
                "pvalue": float(pvalue),
            },
        )
        pvalues.append(pvalue)

    n = len(pvalues)
    error_budget = Fraction(str(FAMILY_WISE_FAILURE_RATE))
    if rule == "ltt-holm":
        order = sorted(range(n), key=pvalues.__getitem__)
        adjusted = accumulate(
            ((n - rank) * pvalues[index] for rank, index in enumerate(order)), max
        )
        qualified = {
            index for index, p in zip(order, adjusted, strict=True) if p <= error_budget
        }
    else:
        qualified = {index for index, p in enumerate(pvalues) if n * p <= error_budget}
    assert {
        index for index, candidate in enumerate(candidates) if candidate["passes"]
    } == qualified
    expected = thresholds[max(qualified)] if qualified else None
    assert report["selected_threshold"] == expected


def grader_options(columns):
    return [flag for name in columns for flag in ("--correct-column", name)]


def evaluate_case(case, directory, partitions, target):
    directory.mkdir(parents=True, exist_ok=True)
    paths = {part: directory / f"{part}.csv" for part in PARTITIONS}
    for part, rows in partitions.items():
        for row in rows:
            row["both_correct"] = str(
                int(row["correct"] == row["alternate_correct"] == "1")
            )
        write_report(paths[part], rows)
    design, calibration, holdout = (partitions[part] for part in PARTITIONS)

    for size, rule, mode in product(DESIGN_QUANTILES, RULES, FIT_COLUMNS):
        label = f"{size}-{rule}-{mode}"
        fit_path = directory / f"{label}-fit.json"
        columns = FIT_COLUMNS[mode]
        fit = call(
            *CLI,
            "select",
            paths["calibration"],
            "--design",
            paths["design"],
            "--design-grid-size",
            str(size),
            "--score-column",
            "confidence",
            *grader_options(columns),
            "--rule",
            rule,
            "--risk-target",
            str(target),
            "--failure-rate",
            str(FAMILY_WISE_FAILURE_RATE),
            "--output-key-column",
            "output_sha256",
            "--track-fit-outputs",
            "--output",
            fit_path,
            "--json",
            statuses=(0, 1),
        )
        selection = json.loads(fit.stdout)
        check_fit(selection, calibration, design, columns, size, rule, target)
        cutoff = selection["selected_threshold"]
        assert fit.returncode == int(cutoff is None)
        fitted = {
            key: measure(calibration, cutoff, column, target)
            for key, column in AUDIT_COLUMNS.items()
        }
        held = {
            key: measure(holdout, cutoff, column, target)
            for key, column in AUDIT_COLUMNS.items()
        }
        if cutoff is not None:
            status = int(any(not entry["meets_target"] for entry in held.values()))
            audit = json.loads(
                call(
                    *CLI,
                    "audit",
                    paths["holdout"],
                    "--threshold-report",
                    fit_path,
                    *grader_options(AUDIT_COLUMNS.values()),
                    "--holdout-confidence",
                    str(CONFIDENCE),
                    "--output",
                    directory / f"{label}-audit.json",
                    "--json",
                    statuses=(status,),
                ).stdout
            )
            assert audit["input"]["fit_outputs_checked"] is True
            for key, column in AUDIT_COLUMNS.items():
                verify_counts(
                    f"{case} {label} {key}", audit["graders"][column], held[key]
                )

        yield {
            "case": case,
            "design_quantiles": size,
            "distinct_cutoffs": len(selection["candidates"]),
            "rule": rule,
            "fit_target": mode,
            "risk_target": target,
            "cutoff": cutoff,
            "calibration_rows": len(calibration),
            "alternate_calibration_labels": len(calibration)
            if mode != "primary-only"
            else 0,
            "calibration_accepted": fitted["primary"]["accepted"],
            **{
                f"calibration_{key}_errors": row["errors"]
                for key, row in fitted.items()
            },
            "holdout_rows": len(holdout),
            "accepted": held["primary"]["accepted"],
            "coverage": held["primary"]["coverage"],
            **{
                f"{key}_{field}": row[column]
                for key, row in held.items()
                for field, column in (
                    ("errors", "errors"),
                    ("risk", "error_rate"),
                    ("upper_95", "risk_upper_bound"),
                )
            },
            "both_grader_targets_met": held["primary"]["meets_target"]
            and held["alternate"]["meets_target"],
            "either_error_target_met": held["either"]["meets_target"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspect-splits", type=Path, help="Original Inspect split CSVs."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "replay/paired-calibration"
    )
    args = parser.parse_args()
    cases = {name: load_case(name) for name in (*ARCHIVES, OMNI)}
    if args.inspect_splits is not None:
        cases["inspect_triviaqa"] = load_inspect(args.inspect_splits)
    output = args.output_dir
    results = []
    for name, (splits, details) in cases.items():
        results.extend(
            evaluate_case(
                name, output / "reports" / name, splits, details["risk_target"]
            )
        )
    write_report(output / "comparison.csv", results)
    protocol = {
        "caat-annotation": version("caat-annotation"),
        "design_quantiles": DESIGN_QUANTILES,
        "rules": RULES,
        "fit_columns": FIT_COLUMNS,
        "confidence": CONFIDENCE,
        "family_wise_failure_rate": FAMILY_WISE_FAILURE_RATE,
        "cases": {name: details for name, (_, details) in cases.items()},
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
