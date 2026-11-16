"""Cross-check archived CAAT reports against SciPy, scikit-learn, and MAPIE."""

import argparse
import json
import math
import subprocess
import sys
import warnings
from operator import itemgetter
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

import numpy as np
from mapie.risk_control import BinaryClassificationController
from mapie.risk_control.risks import BinaryRisk
from scipy.optimize import minimize_scalar
from scipy.stats import binom, rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from caat import load_csv, select
from caat.calibrate import calibration_transform, fit_calibration
from caat.metrics import Counts, calibration_metrics
from caat.policy import split_rows
from caat.rules import binomial_pvalue
from caat.select import design_thresholds

from common import ARCHIVES, ROOT, SEED, read_csv, write_report
from study_calibration import METHODS, archives
from study_modules import course_predictions, load_archive, score_archive


SOURCES = {
    archive: (directory, graders)
    for archive, (directory, _, graders) in ARCHIVES.items()
}


def write_subset(path, rows, fields):
    write_report(path, [{key: row[key] for key in fields} for row in rows], fields)


def invoke(*arguments, code=0):
    cli = [sys.executable, "-c", "from caat.cli import main; raise SystemExit(main())"]
    result = subprocess.run(
        [*cli, *(str(argument) for argument in arguments), "--json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != code:
        raise ValueError(
            f"Expected exit {code}, got {result.returncode}: {result.stderr}"
        )
    return json.loads(result.stdout)


def verify_binomial():
    for n in (0, 1, 5, 216, 400, 10000, 100000):
        for k in sorted({0, n // 10, n // 2, n - 1, n} & set(range(n + 1))):
            for p in (0.0, 1e-6, 0.05, 0.2, 0.3, 0.5, 0.9999, 1.0):
                ours = binomial_pvalue(Counts(n, n, k), p)
                reference = binom.cdf(k, n, p)
                assert math.isclose(ours, reference, rel_tol=4e-9, abs_tol=1e-11), (
                    n,
                    k,
                    p,
                    ours,
                    reference,
                )


def verify_policies(replay, scratch):
    policies = read_csv(ROOT / "results/module_study/policies.csv")
    count = 0
    pvalues = 0
    for name, (folder, graders) in SOURCES.items():
        design_rows, calibration, heldout = split_rows(
            read_csv(replay / folder / "predictions_full.csv"),
            SEED,
            id_key=itemgetter("id"),
        )
        score = "mean_token_logprob"
        design, cal, test = (
            scratch / f"{name}_{part}.csv" for part in ("design", "cal", "test")
        )
        write_subset(design, design_rows, [score])
        write_subset(cal, calibration, ["id", score, *graders])
        write_subset(test, heldout, ["id", score, *graders])
        for expected in policies:
            if expected["archive"] != name:
                continue
            selected_graders = (
                graders[:1] if expected["scope"] == "primary" else graders
            )
            args = sum(
                (["--correct-column", grader] for grader in selected_graders), []
            )
            threshold = float(expected["threshold"]) if expected["threshold"] else None
            target = float(expected["risk_target"])
            actual = invoke(
                "select",
                cal,
                "--score-column",
                score,
                "--design",
                design,
                "--risk-target",
                target,
                *args,
                "--rule",
                "empirical" if expected["rule"] == "empirical" else "ltt-bonferroni",
                code=int(threshold is None),
            )
            assert actual["selected_threshold"] == threshold, (expected, actual)
            for candidate in actual["candidates"]:
                mask = [
                    row
                    for row in calibration
                    if float(row[score]) >= candidate["threshold"]
                ]
                errors = {
                    g: sum(row[g] == "0" for row in mask) for g in selected_graders
                }
                assert (
                    candidate["accepted"] == len(mask) and candidate["errors"] == errors
                )
                pvalue = max(
                    binom.cdf(errors[g], len(mask), target) for g in selected_graders
                )
                assert math.isclose(
                    candidate["pvalue"], pvalue, rel_tol=1e-10, abs_tol=1e-13
                )
                pvalues += 1
            if threshold is not None:
                selected = next(
                    c for c in actual["candidates"] if c["threshold"] == threshold
                )
                assert selected["accepted"] == int(expected["cal_accepted"])
                assert math.isclose(
                    selected["pvalue"],
                    float(expected["cal_pvalue"]),
                    rel_tol=1e-10,
                    abs_tol=1e-13,
                )
            else:
                threshold = math.nextafter(
                    max(float(row[score]) for row in heldout), math.inf
                )
            audited = invoke(
                "audit",
                test,
                "--score-column",
                score,
                f"--threshold={threshold}",
                "--correct-column",
                graders[0],
                "--correct-column",
                graders[1],
            )
            assert (
                int(expected["accepted"]) == audited["graders"][graders[0]]["accepted"]
            )
            assert (
                int(expected["primary_errors"])
                == audited["graders"][graders[0]]["errors"]
            )
            assert (
                int(expected["alternate_errors"])
                == audited["graders"][graders[1]]["errors"]
            )
            count += 1
    return count, pvalues


def verify_auroc(replay):
    checks = 0
    for row in read_csv(replay / "judge_study/auroc.csv"):
        if row["scenario"] == "deterministic":
            folder, graders = SOURCES[row["model"]]
            archive = read_csv(replay / folder / "predictions_full.csv")
            grades = [int(item[row["judge"]]) for item in archive]
        else:
            name = quote(row["model"], safe="-_.")
            archive = read_csv(replay / "helm_judges" / row["scenario"] / f"{name}.csv")
            suffix = (
                "score" if row["scenario"] == "wildbench" else "equivalence_judgement"
            )
            grades = [
                int(
                    float(item[f"{row['judge']}_{suffix}"])
                    >= int(row["outcome_cutoff"])
                )
                for item in archive
            ]
        key = row["score"]
        if key == "shorter_response":
            scores = [-float(item["token_count"]) for item in archive]
        elif key == "negative_mean_token_logprob":
            scores = [-float(item["mean_token_logprob"]) for item in archive]
        else:
            scores = [float(item[key]) for item in archive]
        if len(set(grades)) == 1:
            assert not row["auroc"]
        else:
            assert math.isclose(
                float(row["auroc"]), roc_auc_score(grades, scores), abs_tol=1e-12
            )
        checks += 1
    return checks


def probabilities(data):
    score = data[:, 0]
    return np.column_stack((1 - score, score))


def rejected(y_true, y_pred):
    return y_pred == 0


def all_rows(y_true, y_pred):
    return np.ones(len(y_pred), dtype=bool)


def verify_mapie(replay):
    # Compare a nontrivial first-pass, second-fail sequence with the same fixed grid.
    archive = replay / "ifeval_llama/predictions_full.csv"
    rows = load_csv(
        archive, score_column="mean_token_logprob", correct_columns=("strict_pass",)
    )
    design, calibration, _ = split_rows(rows, SEED)
    thresholds = design_thresholds(row.score for row in design)
    scores = np.array([row.score for row in calibration])
    labels = np.array([row.outcomes[0] for row in calibration], dtype=int)
    ranks = np.unique(np.concatenate((scores, thresholds)))
    parameters = (np.searchsorted(ranks, thresholds) + 1) / (len(ranks) + 1)
    probability_scores = (np.searchsorted(ranks, scores) + 1) / (len(ranks) + 1)
    target = 0.42
    controller = BinaryClassificationController(
        probabilities,
        "precision",
        list_predict_params=parameters,
        target_level=1 - target,
        confidence_level=0.95,
        fwer_method="fixed_sequence",
        best_predict_param_choice=BinaryRisk(rejected, all_rows, False),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Fixed sequence testing requires a monotonic risk.*"
        )
        controller.calibrate(probability_scores.reshape(-1, 1), labels)
    report = select(
        calibration,
        graders=("strict_pass",),
        thresholds=thresholds,
        risk_target=target,
        rule="ltt-fixed-sequence",
    )
    valid = [
        threshold
        for threshold, param in zip(thresholds, parameters, strict=True)
        if param in controller.valid_predict_params
    ]
    assert valid == [report["selected_threshold"]] == [-0.14207283069189022]
    assert [candidate["passes"] for candidate in report["candidates"][:3]] == [
        True,
        False,
        None,
    ]
    return report["selected_threshold"]


def verify_calibration(replay):
    scenarios = 0
    for _, _, rows in archives(replay):
        _, calibration, heldout = split_rows(rows, SEED)
        x = np.array([row.score for row in calibration])
        y = np.array([row.outcomes[0] for row in calibration], dtype=int)
        z = np.array([row.score for row in heldout])
        labels = np.array([row.outcomes[0] for row in heldout], dtype=int)
        logits = np.log(
            np.clip(x, 1e-12, 1 - 1e-12) / (1 - np.clip(x, 1e-12, 1 - 1e-12))
        )
        z_clipped = np.clip(z, 1e-12, 1 - 1e-12)
        z_logits = np.log(z_clipped / (1 - z_clipped))
        scale = np.abs(logits).max() or 1
        for method in ("raw", *METHODS):
            if method == "raw":
                values = z
            else:
                mapping = fit_calibration(
                    x, y.tolist(), method=method, input_scale="probability"
                )
                transform = calibration_transform(mapping)
                values = np.array([transform(p) for p in z])
            if method == "platt":
                model = LogisticRegression(C=1, tol=1e-12, max_iter=1000)
                model.fit((logits / scale).reshape(-1, 1), y)
                reference = model.predict_proba((z_logits / scale).reshape(-1, 1))[:, 1]
                np.testing.assert_allclose(values, reference, rtol=0, atol=1e-6)
            elif method == "isotonic":
                model = IsotonicRegression(out_of_bounds="clip")
                np.testing.assert_allclose(
                    values, model.fit(x, y).predict(z), atol=1e-12
                )
            elif method == "temperature":

                def loss(slope):
                    return np.logaddexp(0, (1 - 2 * y) * logits * slope).mean()

                optimum = minimize_scalar(loss, bounds=(1e-6, 100), method="bounded")
                assert (
                    loss(mapping["parameters"]["inverse_temperature"])
                    <= optimum.fun + 1e-8
                )
            elif method == "histogram":
                fit_bin = np.minimum((x * 10).astype(int), 9)
                frequencies = [
                    y[fit_bin == i].mean() if (fit_bin == i).any() else y.mean()
                    for i in range(10)
                ]
                np.testing.assert_allclose(
                    values, np.array(frequencies)[np.minimum((z * 10).astype(int), 9)]
                )

            actual = calibration_metrics(values, labels)
            assert math.isclose(
                actual["brier_score"], brier_score_loss(labels, values), abs_tol=1e-12
            )
            fixed = np.minimum((values * 10).astype(int), 9)
            adaptive = (
                (rankdata(values) - 0.5) * min(len(values), 10) / len(values)
            ).astype(int)
            counts = np.bincount(fixed, minlength=10)
            fixed_gaps = np.abs(np.bincount(fixed, values - labels, minlength=10))
            _, counts_adaptive = np.unique(adaptive, return_counts=True)
            adaptive_gaps = np.abs(np.bincount(adaptive, values - labels))[
                np.unique(adaptive)
            ]
            np.testing.assert_array_equal(
                [entry["count"] for entry in actual["reliability_bins"]], counts
            )
            assert math.isclose(
                actual["ece"], fixed_gaps.sum() / len(values), abs_tol=1e-12
            )
            assert math.isclose(
                actual["ace"], (adaptive_gaps / counts_adaptive).mean(), abs_tol=1e-12
            )
            scenarios += 1
    return scenarios


def verify_scores(replay, directory):
    tables = [
        {
            (
                item["benchmark"],
                item["method"],
                item["label_source"],
                float(item["target_coverage"]),
            ): item
            for item in read_csv(ROOT / "results" / table)
        }
        for table in ("module_study/ranking.csv", "course/course.csv")
    ]
    expected_module, expected_course = tables
    budgets = (
        "--coverage",
        "0.25",
        "--coverage",
        "0.5",
        "--coverage",
        "0.75",
        "--coverage",
        "1",
    )

    def compare(path, methods, grader_columns, split):
        args = ["scores", path, "--split", split, "--bootstrap", "0", *budgets]
        for method in methods:
            args.extend(("--score-column", method))
        for label in grader_columns:
            args.extend(("--correct-column", label))
        return invoke(*args)["graders"]

    def check(result, expected, archive):
        for label, grader in result.items():
            for method, score in grader["scores"].items():
                for entry in score["budgets"]:
                    key = (archive, method, label, entry["target_coverage"])
                    saved = expected.pop(key)
                    assert entry["accepted"] == int(saved["accepted"]), key
                    assert entry["coverage"] == float(saved["coverage"]), key
                    assert entry["error_rate"] == float(saved["error_rate"]), key

    module_count, course_count = len(expected_module), len(expected_course)
    for name, (folder, count, graders) in ARCHIVES.items():
        source = load_archive(replay / folder / "predictions_full.csv", count, graders)
        design, calibration, heldout = split_rows(source, SEED, id_key=itemgetter("id"))
        rows = design + calibration + heldout
        scores, _ = score_archive(rows, len(design), graders[0])
        path = directory / f"scores_{name}.csv"
        write_report(
            path,
            [
                {
                    "id": row["id"],
                    **{label: int(row[label]) for label in graders},
                    **{key: values[i] for key, values in scores.items()},
                }
                for i, row in enumerate(rows)
            ],
        )
        check(compare(path, scores, graders, "heldout"), expected_module, name)

    for archive, method, label, rows, _ in course_predictions():
        path = directory / f"scores_{archive}_{method}.csv"
        write_report(
            path,
            [
                {"id": row.id, method: row.score, label: int(row.outcomes[0])}
                for row in rows
            ],
        )
        check(compare(path, [method], [label], "all"), expected_course, archive)
    assert not expected_module and not expected_course
    return module_count, course_count


def verify_black_box():
    root = ROOT / "results/black_box"
    report = read_csv(root / "comparison.csv")
    for dataset in ("triviaqa", "mmlu_pro"):
        archived = read_csv(root / f"{dataset}_scores.csv")
        rows = [row for row in report if row["dataset"] == dataset]
        names = [row["method"] for row in rows]
        matched = [row for row in archived if all(row[name] for name in names)]
        labels = np.array([int(row["correct"]) for row in matched])
        for row in rows:
            name = row["method"]
            assert int(row["total_rows"]) == len(archived)
            assert int(row["scored_rows"]) == sum(bool(item[name]) for item in archived)
            assert int(row["matched_rows"]) == len(matched)
            assert int(row["matched_correct"]) == labels.sum()
            scores = np.array([float(item[name]) for item in matched])
            assert math.isclose(
                float(row["auroc"]), roc_auc_score(labels, scores), abs_tol=5.1e-11
            ), (dataset, name)
            for coverage in (25, 50, 75):
                accepted = math.ceil(len(scores) * coverage / 100)
                cut_index = len(scores) - accepted
                cutoff = np.partition(scores, cut_index)[cut_index]
                above, tied = scores > cutoff, scores == cutoff
                errors = (1 - labels[above]).sum()
                errors += (accepted - above.sum()) * (1 - labels[tied]).mean()
                assert math.isclose(
                    float(row[f"risk_{coverage}"]), errors / accepted, abs_tol=5.1e-11
                ), (dataset, name, coverage)
            if row["brier_score"]:
                assert math.isclose(
                    float(row["brier_score"]),
                    brier_score_loss(labels, scores),
                    abs_tol=5.1e-11,
                ), (dataset, name)
    return len(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=ROOT / "replay")
    args = parser.parse_args()
    verify_binomial()
    with TemporaryDirectory() as directory:
        count, pvalues = verify_policies(args.replay_dir, Path(directory))
        module_scores, course_scores = verify_scores(args.replay_dir, Path(directory))
    checks = verify_auroc(args.replay_dir)
    cutoff = verify_mapie(args.replay_dir)
    calibration = verify_calibration(args.replay_dir)
    black_box = verify_black_box()
    print(
        f"{count} policy replays, {pvalues} calibration tails, and {checks} AUROC checks matched."
    )
    print(f"MAPIE selected the same IFEval cutoff: {cutoff!r}.")
    print(
        f"{calibration} held-out calibration scenarios matched independent references."
    )
    print(
        f"{module_scores} module and {course_scores} course score budgets matched the frozen tables."
    )
    print(
        f"{black_box} Qwen score comparisons matched NumPy and scikit-learn references."
    )


if __name__ == "__main__":
    main()
