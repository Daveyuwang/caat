"""Recompute the 0.5 planning comparison from the two public 0.4 inputs."""

import csv
from math import log
from random import Random

import numpy as np
from scipy.stats import binom


from common import ROOT
from caat import GradedRow, select
from mapie.risk_control import BinaryClassificationController


def quantiles(scores):
    ranked = np.sort(scores)[::-1]
    return np.unique([ranked[(len(ranked) * j - 1) // 10] for j in range(1, 11)])


def acceptance(score, wrong, grid, calibration, holdout, check_native):
    grid = np.sort(grid)
    mask = score[calibration, None] >= grid
    accepted = mask.sum(axis=0)
    errors = (mask * wrong[calibration, None]).sum(axis=0)
    passed = (accepted > 0) & (binom.cdf(errors, accepted, 0.4) <= 0.05 / len(grid))
    rows = [GradedRow(str(i), score[i], (not wrong[i],)) for i in calibration]
    result = select(
        rows, graders=["correct"], thresholds=grid.tolist(), risk_target=0.4
    )
    actual = list(reversed(result["candidates"]))
    np.testing.assert_array_equal([r["accepted"] for r in actual], accepted)
    np.testing.assert_array_equal([r["errors"]["correct"] for r in actual], errors)
    np.testing.assert_allclose(
        [r["pvalue"] for r in actual], binom.cdf(errors, accepted, 0.4), atol=1e-12
    )
    np.testing.assert_array_equal([r["passes"] for r in actual], passed)
    if check_native:
        # An order-preserving encoding keeps the exact masks on log and probability scales.
        support = np.unique(np.r_[grid, score[calibration]])
        values = (1 + np.searchsorted(support, score[calibration])) / (len(support) + 1)
        bounds = (1 + np.searchsorted(support, grid)) / (len(support) + 1)
        controller = BinaryClassificationController(
            lambda x: np.column_stack((1 - x.ravel(), x.ravel())),
            "precision",
            list_predict_params=bounds,
            target_level=0.6,
            confidence_level=0.95,
            fwer_method="bonferroni",
            best_predict_param_choice="abstention_rate",
        ).calibrate(values[:, None], (~wrong[calibration]).astype(int))
        print(
            "MAPIE seed 42",
            len(grid),
            "candidates;",
            sum(
                (t in controller.valid_predict_params) != ok
                for t, ok in zip(bounds, passed)
            ),
            "different decisions; max p-value difference",
            max(abs(controller.p_values.ravel() - binom.cdf(errors, accepted, 0.4))),
        )
    threshold = min(grid[passed], default=float("inf"))
    return score[holdout] >= threshold, not passed.any()


def replay(case, path, column, label, transform):
    with (ROOT / path).open(encoding="utf-8", newline="") as source:
        rows = sorted(csv.DictReader(source), key=lambda row: row["id"])
    assert len({r["id"] for r in rows}) == len(rows)
    score = np.array([float(r[column]) for r in rows])
    wrong = np.array([r[label] == "0" for r in rows])
    assignment = list(range(len(rows)))
    Random(42).shuffle(assignment)
    d, dc = len(rows) // 5, 3 * len(rows) // 5
    holdout, pool = assignment[dc:], sorted(assignment[:dc])
    nd, nc = d * 4 // 5, (dc - d) * 4 // 5
    fixed = np.array([transform((j + 0.5) / 10) for j in range(10)])
    runs = {}
    for seed in range(200):
        draw = pool.copy()
        Random(seed).shuffle(draw)
        design, cal = draw[:nd], draw[d : d + nc]
        reserve = draw[nd:d] + draw[d + nc :]
        larger = cal + design
        ref_grid = quantiles(score[reserve])
        for name, grid, labels in (
            ("A. Original quantiles", quantiles(score[design]), cal),
            ("B. Fixed, smaller calibration", fixed, cal),
            ("C. Fixed, larger calibration", fixed, larger),
            ("D. Reference quantiles, smaller calibration", ref_grid, cal),
            ("E. Reference quantiles, larger calibration", ref_grid, larger),
        ):
            runs.setdefault(name, []).append(
                acceptance(score, wrong, grid, labels, holdout, seed == 42)
            )
    for name, outcomes in runs.items():
        masks = np.array([m for m, _ in outcomes])
        changed = (masks[::2] != masks[1::2]).mean()
        print(
            f"{case:8s} {name:44s} {changed:.3%} {masks.mean():.3%} "
            f"{sum(empty for _, empty in outcomes):3d}/200"
        )


if __name__ == "__main__":
    print(
        "Case     Arm                                          Changed Coverage No cutoff"
    )
    replay(
        "Inspect",
        "results/routing_stability/inspect_scores.csv",
        "confidence",
        "correct",
        float,
    )
    replay(
        "IFEval",
        "examples/ifeval_graders.csv",
        "mean_token_logprob",
        "strict_pass",
        log,
    )
