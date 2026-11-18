"""Compare design and calibration score choice on frozen risk targets."""

import argparse
import json
from fractions import Fraction
from hashlib import sha256
from importlib.metadata import version
from itertools import chain
from math import inf
from pathlib import Path

from caat import GradedRow, select
from caat.rules import RULES
from caat.scores import ScoredRow, compare_scores
from caat.select import design_thresholds

from common import ROOT, read_csv, write_report
from study_decision_baselines import (
    CONFIDENCE,
    FAMILY_WISE_FAILURE_RATE,
    PARTITIONS,
    load_case,
    load_inspect,
    measure,
)
from study_model_families import OPTIONAL, matched_ids, read_models, read_partitions
from study_paired_calibration import check_fit, exact_pvalue


RULE = "ltt-bonferroni"
GRID_SIZE = 10
SHORT_TARGETS = (0.2, 0.4)
CHOICE_TARGETS = (0.4, 0.8)
PAIRED_GRADERS = ("correct", "alternate_correct")


def attach_scores(partitions, source, columns):
    indexed = {row["id"]: row for row in source}
    ids = {row["id"] for row in chain.from_iterable(partitions.values())}
    if indexed.keys() != ids or len(indexed) != len(source):
        raise ValueError("score export must cover the same distinct IDs")
    for rows in partitions.values():
        for row in rows:
            stored = indexed[row["id"]]
            if stored["output_sha256"] != row["output_sha256"]:
                raise ValueError(f"output key mismatch for {row['id']!r}")
            row.update({name: float(stored[name]) for name in columns})


def cases(inspect_splits, inspect_scores):
    for name in ("ifeval_qwen", "ifeval_llama"):
        partitions, metadata = load_case(name)
        columns = ("mean_token_logprob", "sequence_logprob")
        attach_scores(partitions, read_csv(ROOT / metadata["prepared_source"]), columns)
        yield name, partitions, columns, PAIRED_GRADERS, SHORT_TARGETS, metadata

    if inspect_splits is not None:
        partitions, metadata = load_inspect(inspect_splits)
        columns = ("confidence", "negative_token_count")
        source = read_csv(inspect_scores)
        original = {row["id"]: row for row in chain.from_iterable(partitions.values())}
        for row in source:
            saved = original[row["id"]]
            if (
                float(row["confidence"]) != saved["confidence"]
                or row["triviaqa_alias"] != saved["correct"]
                or row["canonical_exact"] != saved["alternate_correct"]
            ):
                raise ValueError(f"Inspect score export differs for {row['id']!r}")
        attach_scores(partitions, source, columns)
        metadata["score_export_sha256"] = sha256(
            inspect_scores.read_bytes()
        ).hexdigest()
        yield (
            "inspect_triviaqa",
            partitions,
            columns,
            PAIRED_GRADERS,
            SHORT_TARGETS,
            metadata,
        )

    root = ROOT / "results/black_box_families"
    assignments = read_partitions(root / "split.csv")
    models = ("qwen2_5_7b", "phi4_mini", "smollm2")
    for dataset, folds in assignments.items():
        tables, columns = read_models([root / name for name in models], dataset, folds)
        columns = tuple(name for name in columns if name not in (*OPTIONAL, "random"))
        shared = matched_ids(tables, folds, columns)
        targets = SHORT_TARGETS if dataset == "triviaqa" else CHOICE_TARGETS
        for model, table in tables.items():
            partitions = {name: [] for name in PARTITIONS}
            for item, fold in folds.items():
                if item in shared:
                    row = table[item]
                    partitions["holdout" if fold == "heldout" else fold].append(
                        {
                            "id": item,
                            "correct": row["correct"],
                            **{name: float(row[name]) for name in columns},
                        }
                    )
            metadata = {
                "score_export": f"results/black_box_families/{model}/{dataset}_scores.csv",
                "score_export_sha256": sha256(
                    (root / model / f"{dataset}_scores.csv").read_bytes()
                ).hexdigest(),
                "split_sha256": sha256((root / "split.csv").read_bytes()).hexdigest(),
                "counts": {name: len(rows) for name, rows in partitions.items()},
            }
            yield (
                f"{dataset}_{model}",
                partitions,
                columns,
                ("correct",),
                targets,
                metadata,
            )


def rows_for_score(data, name):
    return [{**row, "confidence": row[name]} for row in data]


def max_design_acceptance(design, grid, risk):
    target = Fraction(str(risk))
    largest = 0
    for threshold in grid:
        accepted = [row for row in design if row["confidence"] >= threshold]
        errors = sum(row["correct"] == "0" for row in accepted)
        if accepted and Fraction(errors, len(accepted)) <= target:
            largest = max(largest, len(accepted))
    return largest


def fit_scores(by_score, grids, risk):
    reports = {}
    for name, (design, calibration, _) in by_score.items():
        records = [
            GradedRow(row["id"], row["confidence"], (row["correct"] == "1",))
            for row in calibration
        ]
        report = select(
            records,
            graders=("correct",),
            thresholds=grids[name],
            risk_target=risk,
            rule=RULE,
        )
        check_fit(report, calibration, design, ("correct",), GRID_SIZE, RULE, risk)
        reports[name] = report
    return reports


def joint_choice(fits, risk):
    pairs = [
        (name, candidate)
        for name, report in fits.items()
        for candidate in report["candidates"]
    ]
    candidates = [candidate for _, candidate in pairs]
    corrected = RULES[RULE].choose(candidates, risk, FAMILY_WISE_FAILURE_RATE)
    level = Fraction(str(FAMILY_WISE_FAILURE_RATE)) / len(candidates)
    assert corrected.per_candidate_level == float(level)
    assert corrected.decisions == tuple(
        candidate["accepted"] > 0
        and exact_pvalue(candidate["accepted"], max(candidate["errors"].values()), risk)
        <= level
        for candidate in candidates
    )
    best = max(
        (
            pair
            for pair, passed in zip(pairs, corrected.decisions, strict=True)
            if passed
        ),
        key=lambda pair: (pair[1]["accepted"], -pair[1]["pvalue"]),
        default=None,
    )
    return (
        (best[0], best[1]["threshold"], len(pairs))
        if best
        else (None, None, len(pairs))
    )


def evaluate_case(case, partitions, columns, graders, risks):
    design, calibration, holdout = (partitions[name] for name in PARTITIONS)
    comparison = compare_scores(
        [
            ScoredRow(
                row["id"],
                tuple(row[name] for name in columns),
                (row["correct"] == "1",),
            )
            for row in chain(design, calibration, holdout)
        ],
        names=columns,
        graders=("correct",),
        bootstrap=0,
        split_sizes=(len(design), len(calibration), len(holdout)),
    )
    baseline = comparison["recommended_score"] or columns[0]
    gains = comparison["design_normalized_aurc_gain"]
    by_score = {
        name: tuple(rows_for_score(partitions[part], name) for part in PARTITIONS)
        for name in columns
    }
    grids = {
        name: design_thresholds(row["confidence"] for row in rows)
        for name, (rows, _, _) in by_score.items()
    }

    for risk in risks:
        design_counts = {
            name: max_design_acceptance(rows, grids[name], risk)
            for name, (rows, _, _) in by_score.items()
        }
        target_score = baseline
        if any(design_counts.values()):
            target_score = max(
                columns,
                key=lambda name: (
                    design_counts[name],
                    gains[name] if gains[name] is not None else -inf,
                ),
            )
        fits = fit_scores(by_score, grids, risk)
        choices = {
            "design-aurc": (
                baseline,
                fits[baseline]["selected_threshold"],
                len(grids[baseline]),
            ),
            "design-target-coverage": (
                target_score,
                fits[target_score]["selected_threshold"],
                len(grids[target_score]),
            ),
            "joint-calibration": joint_choice(fits, risk),
        }
        for method, (name, threshold, candidates) in choices.items():
            _, fitted, held = by_score[name or baseline]
            cal = measure(fitted, threshold, "correct", risk)
            stats = [measure(held, threshold, grader, risk) for grader in graders]
            primary, alternate = stats[0], stats[1] if len(stats) > 1 else {}
            confirmed = all(row["meets_target"] for row in stats)
            yield {
                "case": case,
                "risk_target": risk,
                "selection": method,
                "score": name,
                "threshold": threshold,
                "candidates": candidates,
                "design_aurc_gain": gains.get(name),
                "design_accepted_at_target": design_counts.get(name),
                "calibration_accepted": cal["accepted"],
                "calibration_errors": cal["errors"],
                "holdout_rows": len(held),
                "accepted": primary["accepted"],
                "primary_errors": primary["errors"],
                "primary_upper_95": primary["risk_upper_bound"],
                "alternate_errors": alternate.get("errors"),
                "alternate_upper_95": alternate.get("risk_upper_bound"),
                "confirmed": confirmed,
                "accepted_if_confirmed": primary["accepted"] if confirmed else 0,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-splits", type=Path)
    parser.add_argument("--inspect-scores", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replay/score-choice")
    args = parser.parse_args()
    if (args.inspect_splits is None) != (args.inspect_scores is None):
        parser.error("provide both --inspect-splits and --inspect-scores")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    results, sources = [], {}
    for name, partitions, columns, graders, risks, metadata in cases(
        args.inspect_splits, args.inspect_scores
    ):
        sources[name] = {
            **metadata,
            "scores": columns,
            "graders": graders,
            "risk_targets": risks,
        }
        results.extend(evaluate_case(name, partitions, columns, graders, risks))
    write_report(output / "comparison.csv", results)
    protocol = {
        "caat-annotation": version("caat-annotation"),
        "rule": RULE,
        "design_quantiles": GRID_SIZE,
        "failure_rate": FAMILY_WISE_FAILURE_RATE,
        "holdout_confidence": CONFIDENCE,
        "cases": sources,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
