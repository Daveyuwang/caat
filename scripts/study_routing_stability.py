"""Compare single and fitted scores with three independently calibrated cutoff grids."""

import argparse
import csv
import json
from collections import defaultdict
from math import ceil
from pathlib import Path
from random import Random
from statistics import fmean, median, quantiles
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from caat import GradedRow, select
from caat.metrics import Counts, ranking_metrics, threshold_points
from caat.policy import split_rows
from caat.rules import binomial_upper_bound
from caat.select import design_thresholds

from common import ROOT, read_csv, write_report
from study_combined_scores import MODELS, paired_results
from study_model_families import read_models, read_partitions
from caat.scores import _ScoreRanking


FEATURES = ("mean_token_logprob", "sequence_logprob", "shorter_response")


def shortlist(rows, target):
    """Propose distinct tied-score thresholds using design outcomes alone."""
    fractions = []
    rng = Random(42)
    for _ in range(25):
        sample = sorted(
            rng.sample(rows, max(1, len(rows) * 4 // 5)),
            key=lambda row: row.score,
            reverse=True,
        )
        thresholds = design_thresholds(row.score for row in sample)
        accepted = errors = 0
        best = 0
        for cutoff in thresholds:
            while accepted < len(sample) and sample[accepted].score >= cutoff:
                errors += not sample[accepted].outcomes[0]
                accepted += 1
            if errors <= target * accepted:
                best = accepted
        fractions.append(best / len(sample))
    scores = sorted((row.score for row in rows), reverse=True)
    return sorted(
        {
            scores[ceil(frac * len(scores)) - 1]
            for frac in quantiles(fractions, method="inclusive")
            if frac > 0
        },
        reverse=True,
    )


def score_vectors(x, y, design, features):
    gains = [
        ranking_metrics(threshold_points(zip(x[design, col], ~y[design], strict=True)))[
            "normalized_aurc_gain"
        ]
        for col in range(len(features))
    ]
    best = max(
        range(len(features)),
        key=lambda i: gains[i] if gains[i] is not None else -float("inf"),
    )
    yield "single", features[best], x[:, best], 0.0
    if len(features) > 1:
        start = perf_counter()
        if len(set(y[design])) == 1:
            fitted = np.zeros(len(x))
        else:
            model = make_pipeline(
                StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)
            )
            model.fit(x[design], y[design])
            fitted = model.decision_function(x)
        yield "fitted", "logistic", fitted, perf_counter() - start


def measure(case, table, assignment, features, label, target, seeds, fraction, output):
    ids = sorted(table)
    x = np.array([[float(table[i][name]) for name in features] for i in ids])
    if not np.isfinite(x).all() or any(table[i][label] not in ("0", "1") for i in ids):
        raise ValueError(f"{case}: invalid score or grade")
    y = np.array([table[i][label] == "1" for i in ids])
    original = {
        s: [j for j, i in enumerate(ids) if assignment[i] == s]
        for s in ("design", "calibration", "heldout")
    }
    held = original["heldout"]
    pool = sorted(original["design"] + original["calibration"])
    size = len(original["design"])
    if min(size, len(original["calibration"]), len(held)) < 1:
        raise ValueError(f"{case}: empty partition")
    stats = defaultdict(list)
    primary = []
    ranks = []
    location = output / case
    location.mkdir(parents=True, exist_ok=True)
    with (
        (location / "splits.jsonl").open("w", encoding="utf-8") as splits,
        (location / "decisions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as decisions,
    ):
        writer = csv.writer(decisions, lineterminator="\n")
        writer.writerow(
            [
                "seed",
                "score",
                "feature",
                "grid",
                "cutoff",
                "accepted_bits_in_holdout_order",
            ]
        )
        (location / "holdout.json").write_text(
            json.dumps([ids[i] for i in held]), encoding="utf-8"
        )
        for seed in range(-1, seeds):
            if seed == -1:
                design, calibration = original["design"], original["calibration"]
            else:
                draw = pool.copy()
                Random(seed).shuffle(draw)
                design = draw[: max(1, int(size * fraction))]
                calibration = draw[
                    size : size + max(1, int((len(pool) - size) * fraction))
                ]
            splits.write(
                json.dumps(
                    {
                        "seed": seed,
                        "design": [ids[i] for i in design],
                        "calibration": [ids[i] for i in calibration],
                    }
                )
                + "\n"
            )
            for kind, feature, scores, seconds in score_vectors(x, y, design, features):
                if seed == -1:
                    ranks.append(_ScoreRanking.build(scores[held], ~y[held]))
                rows = [
                    GradedRow(ids[i], float(v), (bool(b),))
                    for i, (v, b) in enumerate(zip(scores, y, strict=True))
                ]
                drows = [rows[i] for i in design]
                crows = [rows[i] for i in calibration]
                grids = {
                    "ten": design_thresholds(row.score for row in drows),
                    "three": design_thresholds(
                        (row.score for row in drows), grid_size=3
                    ),
                    "shortlist": shortlist(drows, target),
                }
                for grid, thresholds in grids.items():
                    cutoff = None
                    if thresholds:
                        cutoff = select(
                            crows,
                            graders=["correct"],
                            thresholds=thresholds,
                            risk_target=target,
                        )["selected_threshold"]
                    mask = (
                        np.zeros(len(held), dtype=bool)
                        if cutoff is None
                        else scores[held] >= cutoff
                    )
                    accepted = int(mask.sum())
                    bits = "".join("1" if v else "0" for v in mask)
                    writer.writerow([seed, kind, feature, grid, cutoff, bits])
                    if seed >= 0:
                        stats[kind, grid].append((mask, cutoff is None))
                    else:
                        errors = int((mask & ~y[held]).sum())
                        primary.append(
                            dict(
                                case=case,
                                score=kind,
                                feature=feature,
                                grid=grid,
                                cutoff=cutoff,
                                holdout=len(held),
                                accepted=accepted,
                                errors=errors,
                                risk=errors / accepted if accepted else None,
                                upper_95=binomial_upper_bound(
                                    Counts(len(held), accepted, errors), 0.95
                                ),
                                fit_seconds=seconds,
                            )
                        )
    summaries = []
    for (kind, grid), runs in stats.items():
        counts = [int(mask.sum()) for mask, _ in runs]
        flips = [
            float(np.mean(a[0] != b[0]))
            for a, b in zip(runs[::2], runs[1::2], strict=True)
        ]
        deciles = quantiles(counts, n=10, method="inclusive")
        summaries.append(
            dict(
                case=case,
                score=kind,
                grid=grid,
                seeds=seeds,
                heldout=len(held),
                median_accepted=median(counts),
                flip_rate=fmean(flips),
                coverage=fmean(counts) / len(held),
                coverage_p10=deciles[0] / len(held),
                coverage_p90=deciles[-1] / len(held),
                no_cutoff=sum(missing for _, missing in runs),
            )
        )
    return summaries, primary, ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "replay" / "routing-stability"
    )
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--fit-fraction", type=float, default=1.0)
    parser.add_argument(
        "--arc",
        type=Path,
        help="Prepared ARC train and test feature table with split column",
    )
    args = parser.parse_args()
    if args.seeds < 2 or args.seeds % 2 or not 0 < args.fit_fraction <= 1:
        parser.error("provide an even number of seeds >=2 and a fit fraction in (0,1]")
    args.output.mkdir(exist_ok=True, parents=True)
    stability, audits, quality = [], [], []

    def run(case, table, assignments, features, label, target):
        summary, primary, ranks = measure(
            case,
            table,
            assignments,
            features,
            label,
            target,
            args.seeds,
            args.fit_fraction,
            args.output,
        )
        stability.extend(summary)
        audits.extend(primary)
        return ranks

    if args.arc:
        data = read_csv(args.arc)
        table = {r["id"]: r for r in data}
        if len(table) != len(data) or any(
            r["split"] not in ("design", "calibration", "heldout") for r in data
        ):
            raise ValueError("invalid ARC split or duplicate ID")
        ranks = run(
            "arc_challenge",
            table,
            {i: r["split"] for i, r in table.items()},
            ("max_loglikelihood", "loglikelihood_margin"),
            "acc",
            0.2,
        )
        _, result = paired_results([ranks], audits[0]["holdout"])
        quality.append({"dataset": "arc_challenge", **result})
    else:
        for case, path, feature, label in (
            (
                "inspect_triviaqa",
                "results/routing_stability/inspect_scores.csv",
                "confidence",
                "correct",
            ),
            (
                "ifeval",
                "examples/ifeval_graders.csv",
                "mean_token_logprob",
                "strict_pass",
            ),
        ):
            data = read_csv(ROOT / path)
            table = {row["id"]: row for row in data}
            parts = split_rows(list(table), 42, id_key=str)
            assignment = {
                i: s
                for s, part in zip(
                    ("design", "calibration", "heldout"), parts, strict=True
                )
                for i in part
            }
            run(case, table, assignment, [feature], label, 0.4)
        archive = ROOT / "results/black_box_families"
        for dataset, assignment in read_partitions(archive / "split.csv").items():
            tables, _ = read_models(
                [archive / name for name in MODELS], dataset, assignment
            )
            pairs = [
                run(f"{dataset}_{name}", table, assignment, FEATURES, "correct", 0.4)
                for name, table in tables.items()
            ]
            _, result = paired_results(
                pairs, sum(v == "heldout" for v in assignment.values())
            )
            quality.append({"dataset": dataset, **result})
    write_report(args.output / "stability.csv", stability)
    write_report(args.output / "audits.csv", audits)
    write_report(args.output / "quality.csv", quality)


if __name__ == "__main__":
    main()
