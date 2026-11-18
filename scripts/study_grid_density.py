"""Compare design quantile counts on frozen calibration and holdout splits."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from caat import GradedRow, select
from caat.select import design_thresholds

from common import ARCHIVES, ROOT, write_report
from study_cutoff_grid import BONFERRONI, HOLM, holdout_record
from study_decision_baselines import (
    CONFIDENCE,
    FAMILY_WISE_FAILURE_RATE,
    OMNI,
    PARTITIONS,
    load_case,
    load_inspect,
)


GRID_SIZES = (10, 20, 50)


def compare_case(case, partitions, target):
    design, calibration, holdout = (partitions[part] for part in PARTITIONS)
    scores = [row["confidence"] for row in design]
    candidates = {
        size: design_thresholds(scores, grid_size=size) for size in GRID_SIZES
    }
    baseline = set(candidates[10])
    if any(not baseline.issubset(grid) for grid in candidates.values()):
        raise ValueError(f"{case}: finer grid dropped a design decile")
    rows = [
        GradedRow(row["id"], row["confidence"], (row["correct"] == "1",))
        for row in calibration
    ]
    for rule in (BONFERRONI, HOLM):
        for size, thresholds in candidates.items():
            fit = select(
                rows,
                graders=("correct",),
                thresholds=thresholds,
                risk_target=target,
                failure_rate=FAMILY_WISE_FAILURE_RATE,
                rule=rule,
            )
            yield {
                "design_grid_size": size,
                "distinct_cutoffs": len(thresholds),
                **holdout_record(
                    case, rule, fit["selected_threshold"], target, calibration, holdout
                ),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspect-splits", type=Path, help="Original Inspect split CSVs."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "replay/design-grid-density"
    )
    args = parser.parse_args()
    cases = {name: load_case(name) for name in (*ARCHIVES, OMNI)}
    if args.inspect_splits is not None:
        cases["inspect_triviaqa"] = load_inspect(args.inspect_splits)
    results = []
    for name, (splits, metadata) in cases.items():
        results.extend(compare_case(name, splits, metadata["risk_target"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_report(args.output_dir / "comparison.csv", results)
    protocol = {
        "caat-annotation": version("caat-annotation"),
        "requested_design_quantiles": GRID_SIZES,
        "rules": (BONFERRONI, HOLM),
        "failure_rate": FAMILY_WISE_FAILURE_RATE,
        "holdout_confidence": CONFIDENCE,
        "cases": {name: metadata for name, (_, metadata) in cases.items()},
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
