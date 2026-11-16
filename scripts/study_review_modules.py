"""Compare archived confidence features and single- or multi-grader thresholds."""

import argparse
import csv
from math import ceil, isfinite, log
from pathlib import Path
from random import Random

from numpy.random import default_rng
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from caat.audit import AuditRow, audit
from caat.evaluation import Prediction, threshold_points

from compare_baselines import FAILURE_RATE, ROOT, SEED, ranking_rows, risk_at_count, write_report

ARCHIVES = {
    "ifeval_qwen": ("ifeval", 541, ("strict_pass", "helm_strict_pass")),
    "ifeval_llama": ("ifeval_llama", 541, ("strict_pass", "helm_strict_pass")),
    "mmlu_pro_qwen": ("mmlu_pro", 1000, ("correct", "tiger_correct")),
    "mmlu_pro_llama": ("mmlu_pro_llama", 1000, ("correct", "tiger_correct")),
}
ARCHIVED_FEATURES = ("mean_token_logprob", "low_token_logprob", "log_token_count")
RAW_SCORES = ("mean_token_logprob", "sequence_logprob", "low_token_logprob", "shorter_response")
BASELINE = "mean_token_logprob"
TARGETS = (0.20, 0.30, 0.40)
RESAMPLES = 2000
BOOTSTRAP_SEED = 20261115


def load_archive(path, expected, graders):
    with path.open(encoding="utf-8", newline="") as source:
        rows = sorted(csv.DictReader(source), key=lambda row: row["id"])
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise ValueError(f"{path} must contain {expected} distinct outputs")
    for row in rows:
        for grader in graders:
            if row[grader] not in ("0", "1"):
                raise ValueError(f"{grader} must be a bit for {row['id']}")
            row[grader] = row[grader] == "1"
        for score in ("mean_token_logprob", "sequence_logprob", "low_token_logprob"):
            row[score] = float(row[score])
            if not isfinite(row[score]):
                raise ValueError(f"{score} must be finite for {row['id']}")
        row["log_token_count"] = log(int(row["token_count"]))
        row["shorter_response"] = -row["log_token_count"]
        if "answer_token_logprob" in row:
            value = row["answer_token_logprob"]
            row["answer_present"] = float(value != "")
            row["answer_feature"] = float(value) if value else 0.0
            row["answer_token_logprob"] = float(value) if value else -1e6
    Random(SEED).shuffle(rows)
    return rows


def fitted_scores(rows, design_count, target, features):
    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    matrix = [[row[name] for name in features] for row in rows]
    estimator.fit(matrix[:design_count], target)
    return [float(value) for value in estimator.predict_proba(matrix)[:, 1]]


def score_archive(rows, design_count, primary):
    scores = {name: [row[name] for row in rows] for name in RAW_SCORES}
    scores["random"] = [0.0] * len(rows)
    labels = [row[primary] for row in rows[:design_count]]
    probabilities = {
        "design_base_rate": [sum(labels) / len(labels)] * len(rows),
        "logistic_mean": fitted_scores(rows, design_count, labels, (BASELINE,)),
        "logistic_archived": fitted_scores(rows, design_count, labels, ARCHIVED_FEATURES),
    }
    if "answer_token_logprob" in rows[0]:
        scores["answer_present"] = [row["answer_present"] for row in rows]
        scores["answer_token_logprob"] = [row["answer_token_logprob"] for row in rows]
        probabilities["logistic_with_answer_presence"] = fitted_scores(
            rows, design_count, labels, (*ARCHIVED_FEATURES, "answer_present")
        )
        probabilities["logistic_with_answer"] = fitted_scores(
            rows, design_count, labels, (*ARCHIVED_FEATURES, "answer_present", "answer_feature")
        )
    return {**scores, **probabilities}, set(probabilities)


def budget_risk(scores, errors, indices):
    points = threshold_points((scores[i], errors[i]) for i in indices)
    return risk_at_count(points, ceil(len(indices) * 0.75))


def bootstrap_interval(replicates):
    ordered = sorted(replicates)
    size = len(ordered)
    return ordered[ceil(0.025 * size) - 1], ordered[ceil(0.975 * size) - 1]


def budget_deltas(name, graders, heldout, scores):
    n = len(heldout)
    comparisons = [(method, BASELINE) for method in (
        "sequence_logprob", "answer_token_logprob", "logistic_archived", "logistic_with_answer",
    ) if method in scores]
    if "answer_token_logprob" in scores:
        comparisons.append(("answer_token_logprob", "answer_present"))
    for grader in graders:
        errors = [not row[grader] for row in heldout]
        for method, reference in comparisons:
            baseline, candidate = scores[reference], scores[method]
            observed = budget_risk(candidate, errors, range(n)) - budget_risk(baseline, errors, range(n))
            rng = default_rng(BOOTSTRAP_SEED)
            replicates = []
            for indices in rng.integers(0, n, size=(RESAMPLES, n)).tolist():
                replicates.append(budget_risk(candidate, errors, indices) - budget_risk(baseline, errors, indices))
            lower, upper = bootstrap_interval(replicates)
            yield {
                "archive": name, "grader": grader, "method": method, "baseline": reference,
                "coverage": 0.75, "risk_delta": observed,
                "bootstrap_lower": lower, "bootstrap_upper": upper,
            }


def audit_rows(rows, scores, graders):
    return [
        AuditRow(row["id"], score, tuple(row[grader] for grader in graders), row.get("group"))
        for row, score in zip(rows, scores, strict=True)
    ]


def compare_policies(name, rows, scores, graders, design_count, calibration_end):
    design = sorted(scores[:design_count], reverse=True)
    thresholds = sorted({design[ceil(len(design) * i / 10) - 1] for i in range(1, 11)})
    calibration = audit_rows(rows[design_count:calibration_end], scores[design_count:calibration_end], graders)
    heldout = audit_rows(rows[calibration_end:], scores[calibration_end:], graders)
    candidates = []
    for threshold in thresholds:
        report = audit(calibration, graders=graders, threshold=threshold)
        first, second = (report["graders"][grader] for grader in graders)
        if first["accepted"]:
            candidates.append((threshold, first["accepted"], first["errors"], second["errors"]))
    for target in TARGETS:
        for scope in ("primary", "both"):
            for rule in ("empirical", "learn_then_test"):
                selected = None
                for threshold, n, first_errors, second_errors in candidates:
                    # The larger count gives the larger binomial p-value for the same n and target.
                    errors = first_errors if scope == "primary" else max(first_errors, second_errors)
                    pvalue = float(binom.cdf(errors, n, target))
                    if (errors / n <= target if rule == "empirical" else pvalue <= FAILURE_RATE / len(thresholds)):
                        selected = (threshold, n, first_errors, second_errors, pvalue)
                        break
                yield policy_result(name, graders, heldout, selected, target, scope, rule)


def policy_result(name, graders, heldout, selected, target, scope, rule):
    threshold, cal_n, cal_first, cal_second, pvalue = selected or (None, 0, 0, 0, None)
    accepted = first_errors = second_errors = newly_failed = newly_passed = 0
    primary_risk = alternate_risk = delta = lower = upper = target_disagrees = None
    if threshold is not None:
        report = audit(heldout, graders=graders, threshold=threshold, risk_target=target)
        first, second = (report["graders"][grader] for grader in graders)
        compared = report["accepted_comparisons"][graders[1]]
        accepted, first_errors, second_errors = first["accepted"], first["errors"], second["errors"]
        primary_risk, alternate_risk = first["error_rate"], second["error_rate"]
        newly_failed, newly_passed = compared["newly_failed"], compared["newly_passed"]
        delta, target_disagrees = compared["risk_change"], compared["target_decision_changed"]
        if accepted:
            rng = default_rng(BOOTSTRAP_SEED)
            counts = rng.multinomial(
                accepted, [newly_failed / accepted, newly_passed / accepted,
                           (accepted - newly_failed - newly_passed) / accepted], size=RESAMPLES,
            )
            lower, upper = bootstrap_interval((counts[:, 0] - counts[:, 1]) / accepted)
    return {
        "archive": name, "primary_grader": graders[0], "alternate_grader": graders[1],
        "risk_target": target, "rule": rule, "scope": scope, "threshold": threshold,
        "cal_accepted": cal_n, "cal_primary_errors": cal_first, "cal_alternate_errors": cal_second,
        "cal_pvalue": pvalue, "heldout_total": len(heldout), "accepted": accepted,
        "coverage": accepted / len(heldout), "primary_errors": first_errors, "alternate_errors": second_errors,
        "primary_risk": primary_risk, "alternate_risk": alternate_risk, "newly_failed": newly_failed,
        "newly_passed": newly_passed, "risk_delta": delta, "bootstrap_lower": lower,
        "bootstrap_upper": upper, "target_decision_changed": target_disagrees,
    }


def study(replay_dir, output):
    rankings, paired_deltas, policies, disagreements = [], [], [], []
    for name, (directory, expected, graders) in ARCHIVES.items():
        rows = load_archive(replay_dir / directory / "predictions_full.csv", expected, graders)
        design_count, calibration_end = expected // 5, 3 * expected // 5
        scores, probabilities = score_archive(rows, design_count, graders[0])
        heldout = rows[calibration_end:]
        heldout_scores = {method: values[calibration_end:] for method, values in scores.items()}
        for method, values in heldout_scores.items():
            for grader in graders:
                predictions = [
                    Prediction(row["id"], None, None, score, correct=row[grader])
                    for row, score in zip(heldout, values, strict=True)
                ]
                rankings.extend(ranking_rows(name, method, grader, predictions, method in probabilities))
        paired_deltas.extend(budget_deltas(name, graders, heldout, heldout_scores))
        # Fix the score to the same recognized baseline for each threshold rule.
        policies.extend(compare_policies(name, rows, scores[BASELINE], graders, design_count, calibration_end))
        for split, subset in (("all", rows), ("heldout", heldout)):
            first, second = graders
            disagreements.append({
                "archive": name, "split": split, "primary_grader": first, "alternate_grader": second,
                "total": len(subset), "primary_pass": sum(row[first] for row in subset),
                "alternate_pass": sum(row[second] for row in subset),
                "newly_failed": sum(row[first] and not row[second] for row in subset),
                "newly_passed": sum(not row[first] and row[second] for row in subset),
            })
        write_report(output / f"{name}_heldout.csv", [
            {"id": row["id"], "output_sha256": row["output_sha256"], "group": row.get("group"),
             BASELINE: row[BASELINE], **{grader: int(row[grader]) for grader in graders}}
            for row in heldout
        ])
    for name, report in (
        ("ranking", rankings), ("ranking_deltas", paired_deltas), ("policies", policies),
        ("disagreements", disagreements),
    ):
        write_report(output / f"{name}.csv", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=ROOT / "replay")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replay" / "module_study")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    study(args.replay_dir, args.output_dir)


if __name__ == "__main__":
    main()
