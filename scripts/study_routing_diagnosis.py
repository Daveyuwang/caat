"""Separate grid and calibration variability, and check the native cutoff decisions."""

from argparse import ArgumentParser
from math import log
from pathlib import Path
from random import Random
from statistics import fmean

from mapie.risk_control.fwer_control import control_fwer
import numpy as np
from scipy.stats import binom

from caat import GradedRow, load_csv, select
from caat.policy import split_rows
from caat.select import design_thresholds

from common import ROOT, read_csv, write_report
from study_routing_stability import shortlist, score_vectors, FEATURES, MODELS
from study_model_families import read_models, read_partitions


def check_grid(calibration, thresholds, target):
    if not thresholds:
        return 0
    fit = select(
        calibration, graders=["correct"], thresholds=thresholds, risk_target=target
    )
    counts = []
    for t in thresholds:
        labels = [r.outcomes[0] for r in calibration if r.score >= t]
        counts.append((len(labels), len(labels) - sum(labels)))
    n, k = np.asarray(counts).T
    passed = set(control_fwer(binom.cdf(k, n, target), 0.05, "bonferroni"))
    actual = {r["threshold"] for r in fit["candidates"] if r["passes"]}
    if actual != {thresholds[i] for i in passed}:
        raise ValueError("MAPIE with independent exact binomial p-values disagrees")
    return len(thresholds)


def check_features(table, assignment, names, target, label):
    ids = sorted(table)
    x = np.array([[float(table[i][n]) for n in names] for i in ids])
    y = np.array([table[i][label] == "1" for i in ids])
    d = [j for j, i in enumerate(ids) if assignment[i] == "design"]
    c = [j for j, i in enumerate(ids) if assignment[i] == "calibration"]
    checked = 0
    for _, _, scores, _ in score_vectors(x, y, d, names):
        design = [GradedRow(ids[j], scores[j], (bool(y[j]),)) for j in d]
        cal = [GradedRow(ids[j], scores[j], (bool(y[j]),)) for j in c]
        for grid in (
            design_thresholds(r.score for r in design),
            design_thresholds((r.score for r in design), grid_size=3),
            shortlist(design, target),
        ):
            checked += check_grid(cal, grid, target)
    return checked


def diagnosis(name, rows, transform):
    design, cal, held = split_rows(rows, 42)
    pool = sorted(design + cal, key=lambda r: r.id)
    records = []
    checked = check_grid(
        cal, design_thresholds((r.score for r in design), grid_size=3), 0.4
    )
    for grid_name, propose in (
        ("ten", lambda rows: design_thresholds(r.score for r in rows)),
        ("shortlist", lambda rows: shortlist(rows, 0.4)),
    ):
        checked += check_grid(cal, propose(design), 0.4)
        for variable in ("design", "calibration"):
            decisions = []
            for seed in range(200):
                d = Random(seed if variable == "design" else 42).sample(
                    design, len(design) * 4 // 5
                )
                c = Random(seed if variable == "calibration" else 42).sample(
                    cal, len(cal) * 4 // 5
                )
                decisions.append(decide(c, propose(d), held))
            records.append(summary(name, grid_name, variable, decisions, len(held)))
    external = [transform((i + 0.5) / 10) for i in range(10)]
    checked += check_grid(cal, external, 0.4)
    decisions = []
    for seed in range(200):
        draw = pool.copy()
        Random(seed).shuffle(draw)
        n = len(design)
        labelled = draw[: n * 4 // 5] + draw[n : n + len(cal) * 4 // 5]
        decisions.append(decide(labelled, external, held))
    records.append(
        summary(name, "prespecified", "80% of fitting pool", decisions, len(held))
    )
    return records, checked


def decide(cal, grid, held):
    if not grid:
        return None
    t = select(cal, graders=["correct"], thresholds=grid, risk_target=0.4)[
        "selected_threshold"
    ]
    return None if t is None else {r.id for r in held if r.score >= t}


def summary(case, grid, variable, decisions, n):
    sets = [s or set() for s in decisions]
    return dict(
        case=case,
        grid=grid,
        varied=variable,
        flip_rate=fmean(
            len(a ^ b) / n for a, b in zip(sets[::2], sets[1::2], strict=True)
        ),
        coverage=fmean(len(s) / n for s in sets),
        no_cutoff=decisions.count(None),
        refits=len(sets),
    )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "replay/routing-diagnosis.csv"
    )
    args = parser.parse_args()
    results = []
    total = 0
    for case, path, score, grader, transform in (
        (
            "inspect_triviaqa",
            "results/routing_stability/inspect_scores.csv",
            "confidence",
            "correct",
            float,
        ),
        (
            "ifeval",
            "examples/ifeval_graders.csv",
            "mean_token_logprob",
            "strict_pass",
            log,
        ),
    ):
        rows = load_csv(ROOT / path, score_column=score, correct_columns=[grader])
        records, count = diagnosis(case, rows, transform)
        results.extend(records)
        total += count
    archive = ROOT / "results/black_box_families"
    for task, assignment in read_partitions(archive / "split.csv").items():
        tables, _ = read_models([archive / name for name in MODELS], task, assignment)
        for table in tables.values():
            total += check_features(table, assignment, FEATURES, 0.4, "correct")
    arc = {
        r["id"]: r for r in read_csv(ROOT / "results/routing_stability/arc_scores.csv")
    }
    total += check_features(
        arc,
        {i: r["split"] for i, r in arc.items()},
        ("max_loglikelihood", "loglikelihood_margin"),
        0.2,
        "acc",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output, results)
    print(f"Verified {total} candidate decisions with SciPy and MAPIE.")


if __name__ == "__main__":
    main()
