"""Compare additive and proportional grader-risk predictions on held-out WildBench rows."""

import argparse
import json
import re
from math import isfinite
from pathlib import Path
from statistics import median
from urllib.parse import quote

import numpy as np

from caat.policy import split_rows
from common import (
    JUDGES,
    MATCHED_TOKEN_TEXT,
    SEED as SPLIT_SEED,
    read_csv,
    write_report,
)
from study_judges import BOOTSTRAPS, MEAN, PAIRS, SEED


def completed_runs(path):
    return [
        run
        for run in read_csv(path)
        if run["scenario"] == "wildbench" and run["status"] == "complete"
    ]


def run_key(run):
    return run["suite"], run["run"]


def archive_rows(directory, run):
    path = directory / "wildbench" / f"{quote(run['model'], safe='-_.')}.csv"
    source = read_csv(path)
    ids = {row["id"] for row in source}
    if not all(ids) or len(ids) != len(source) or len(source) != int(run["kept"]):
        raise ValueError(f"{path}: question IDs do not match the inventory")
    return path, source


def response_identities(path, source):
    identities = {}
    for row in source:
        digest = row.get("output_sha256") or ""
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(
                f"{path}: invalid or missing output_sha256 for {row['id']}. "
                "Prepare the archive with prepare_helm_judges.py."
            )
        identities[row["id"]] = (row["id"], digest)
    return identities


def aligned_rows(path, source, run):
    aligned = {}
    for row in source:
        if row["token_alignment"] not in MATCHED_TOKEN_TEXT:
            continue
        score = float(row[MEAN])
        grades = [float(row[f"{judge}_score"]) for judge in JUDGES]
        if not isfinite(score) or score >= 0 or any(not 1 <= x <= 10 for x in grades):
            raise ValueError(f"{path}: invalid confidence or grades for {row['id']}")
        aligned[row["id"]] = (score, tuple(int(grade < 7) for grade in grades))
    if len(aligned) != int(run["token_text_verified"]):
        raise ValueError(
            f"{path}: token-text verification count differs from inventory"
        )
    return aligned


def confirmation_runs(directory, candidates, discovery_dir, discovery):
    seen = set()
    for run in discovery:
        path, source = archive_rows(discovery_dir, run)
        seen.update(response_identities(path, source).values())

    runs, overlap = [], []
    for run in sorted(candidates, key=run_key):
        path, source = archive_rows(directory, run)
        identities = response_identities(path, source)
        aligned = aligned_rows(path, source, run)
        rows = {
            item: row for item, row in aligned.items() if identities[item] not in seen
        }
        overlap.append(
            {
                **{key: run[key] for key in ("suite", "run", "model")},
                "reused_rows": len(aligned) - len(rows),
                "included_rows": len(rows),
            }
        )
        if rows:
            runs.append((run, rows))
            seen.update(identities[item] for item in rows)
    return runs, overlap


def question_counts(ids, rows):
    empty = (0,) * (1 + len(JUDGES))
    return np.array(
        [(1, *rows[item][1]) if item in rows else empty for item in ids],
        dtype=np.int64,
    ).reshape(-1, len(empty))


def resample_weights(rng, count):
    if not count:
        return np.zeros((BOOTSTRAPS, 0), dtype=np.int64)
    return rng.multinomial(count, np.full(count, 1 / count), size=BOOTSTRAPS)


def predictions(a, r, s):
    return np.clip(a + (s - r), 0, 1), np.clip(a * s / r, 0, 1)


def risk(errors, total):
    return float(errors / total) if total else None


def analyze_run(run, rows, splits, weights):
    design, calibration, holdout = splits
    design_scores = [rows[item][0] for item in design if item in rows]
    cutoff = median(design_scores) if design_scores else None
    accepted = {
        item: row
        for item, row in rows.items()
        if cutoff is not None and row[0] >= cutoff
    }
    cal, test = question_counts(calibration, rows), question_counts(holdout, accepted)
    cal_point, test_point = cal.sum(axis=0), test.sum(axis=0)
    sampled_cal, sampled_test = weights[0] @ cal, weights[1] @ test
    holdout_total = sum(item in rows for item in holdout)
    common = {
        **{key: run[key] for key in ("suite", "run", "model")},
        "archive_rows": int(run["kept"]),
        "aligned_rows": len(rows),
        "design_rows": len(design_scores),
        "calibration_rows": int(cal_point[0]),
        "holdout_rows": holdout_total,
        "holdout_accepted": int(test_point[0]),
        "holdout_coverage": risk(test_point[0], holdout_total),
        "threshold": cutoff,
    }
    cells, distributions = [], []
    for first, second in PAIRS:
        old, new = JUDGES.index(first) + 1, JUDGES.index(second) + 1
        r, s = (risk(cal_point[i], cal_point[0]) for i in (old, new))
        a, observed = (risk(test_point[i], test_point[0]) for i in (old, new))
        undefined = None
        if cutoff is None:
            undefined = "no_design_scores"
        elif a is None:
            undefined = "no_accepted_holdout"
        elif r is None:
            undefined = "no_calibration_rows"
        elif r == 0:
            undefined = "zero_old_calibration_risk"
        cell = {
            **common,
            "first_judge": first,
            "second_judge": second,
            "old_calibration_risk": r,
            "new_calibration_risk": s,
            "old_accepted_risk": a,
            "new_accepted_risk": observed,
            "additive_prediction": None,
            "proportional_prediction": None,
            "additive_absolute_error": None,
            "proportional_absolute_error": None,
            "absolute_error_gain": None,
            "undefined_reason": undefined,
            "undefined_bootstrap_replicates": None,
        }
        cells.append(cell)
        if undefined is not None:
            continue
        additive, proportional = map(float, predictions(a, r, s))
        additive_error = abs(additive - observed)
        proportional_error = abs(proportional - observed)
        valid = (sampled_cal[:, old] > 0) & (sampled_test[:, 0] > 0)
        c, h = sampled_cal[valid], sampled_test[valid]
        add, prop = predictions(
            h[:, old] / h[:, 0], c[:, old] / c[:, 0], c[:, new] / c[:, 0]
        )
        actual = h[:, new] / h[:, 0]
        gains = np.full(BOOTSTRAPS, np.nan)
        gains[valid] = np.abs(add - actual) - np.abs(prop - actual)
        distributions.append(gains)
        cell.update(
            additive_prediction=additive,
            proportional_prediction=proportional,
            additive_absolute_error=additive_error,
            proportional_absolute_error=proportional_error,
            absolute_error_gain=additive_error - proportional_error,
            undefined_bootstrap_replicates=BOOTSTRAPS - int(valid.sum()),
        )
    return cells, distributions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--release", required=True)
    parser.add_argument("--discovery-inventory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--verify-discovery",
        action="store_true",
        help="Exercise the protocol with the already analyzed archives.",
    )
    args = parser.parse_args()

    discovery = completed_runs(args.discovery_inventory)
    used = {run_key(run) for run in discovery}
    if not used:
        parser.error("the discovery inventory has no complete WildBench runs")
    eligible = [
        run
        for run in completed_runs(args.input_dir / "inventory.csv")
        if run["release"] == args.release
    ]
    selected = [
        run for run in eligible if (run_key(run) in used) == args.verify_discovery
    ]
    if not selected:
        mode = (
            "from the discovery inventory"
            if args.verify_discovery
            else "outside the discovery inventory"
        )
        parser.error(f"no complete WildBench runs {mode} for {args.release}")
    if len({run_key(run) for run in selected}) != len(selected):
        parser.error("duplicate WildBench archive identities")
    try:
        if args.verify_discovery:
            runs = []
            for run in selected:
                path, source = archive_rows(args.input_dir, run)
                runs.append((run, aligned_rows(path, source, run)))
        else:
            runs, overlap = confirmation_runs(
                args.input_dir, selected, args.discovery_inventory.parent, discovery
            )
            if not runs:
                reused = sum(item["reused_rows"] for item in overlap)
                parser.error(f"no unseen aligned WildBench responses ({reused} reused)")
    except ValueError as exc:
        parser.error(str(exc))
    questions = set().union(*(rows.keys() for _, rows in runs))
    splits = split_rows(questions, SPLIT_SEED, id_key=str)
    rng = np.random.default_rng(SEED)
    weights = [resample_weights(rng, len(items)) for items in splits[1:]]
    cells, distributions = [], []
    for run, rows in runs:
        entries, sampled = analyze_run(run, rows, splits, weights)
        cells.extend(entries)
        distributions.extend(sampled)
    effects = [
        cell["absolute_error_gain"]
        for cell in cells
        if cell["undefined_reason"] is None
    ]
    draws = np.mean(distributions, axis=0) if distributions else np.array([])
    defined = draws[np.isfinite(draws)]
    interval = (
        np.percentile(defined, [2.5, 97.5]).tolist() if len(defined) >= 2 else None
    )
    partitions = dict(zip(("design", "calibration", "holdout"), splits, strict=True))
    summary = {
        "analysis": "protocol_verification"
        if args.verify_discovery
        else "confirmation",
        "release": args.release,
        "archives": len(runs),
        "discovery_archives": len(used),
        "split_seed": SPLIT_SEED,
        "partition_sizes": {
            partition: len(items) for partition, items in partitions.items()
        },
        "cells": len(cells),
        "undefined_cells": len(cells) - len(effects),
        "additive_minus_proportional_mae": sum(effects) / len(effects)
        if effects
        else None,
        "paired_percentile_interval_95": interval,
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "undefined_bootstrap_replicates": BOOTSTRAPS - len(defined),
        "numpy_version": np.__version__,
    }
    if not args.verify_discovery:
        summary["response_overlap"] = overlap
        summary["supports_proportional_comparison"] = (
            interval is not None and interval[0] > 0
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_report(args.output_dir / "cells.csv", cells)
    write_report(
        args.output_dir / "split.csv",
        (
            {"id": item, "partition": name}
            for name, items in partitions.items()
            for item in items
        ),
        fields=("id", "partition"),
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
