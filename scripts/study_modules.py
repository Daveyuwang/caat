"""Replay archived score rankings, review policies, and course baselines."""

import argparse
from dataclasses import replace
from math import ceil, isfinite, log
from pathlib import Path
from operator import itemgetter

from numpy.random import default_rng
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from caat import GradedRow, audit, evaluate, load_csv, select
from caat.metrics import Counts, threshold_points
from caat.policy import split_rows
from caat.select import design_thresholds

from common import ARCHIVES, ROOT, SEED, read_csv, risk_at_count, write_report

HELM = {
    "ifeval_google": ("ifeval", 541, "strict_pass"),
    "ifeval_helm": ("ifeval", 541, "helm_strict_pass"),
    "mmlu_pro": ("mmlu_pro", 1000, "correct"),
}
SCORES = ("mean_token_logprob", "sequence_logprob")
BASELINE_TARGETS = (0.20, 0.30)


ARCHIVED_FEATURES = ("mean_token_logprob", "low_token_logprob", "log_token_count")
RAW_SCORES = ("mean_token_logprob", "sequence_logprob", "low_token_logprob", "shorter_response")
BASELINE = "mean_token_logprob"
TARGETS = (0.20, 0.30, 0.40)
RESAMPLES = 2000
BOOTSTRAP_SEED = 20261115


def load_archive(path, expected, graders):
    rows = read_csv(path)
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
        GradedRow(row["id"], score, tuple(row[grader] for grader in graders), row.get("group"))
        for row, score in zip(rows, scores, strict=True)
    ]


def compare_policies(name, rows, scores, graders, design_count, calibration_end):
    thresholds = design_thresholds(scores[:design_count])
    calibration = audit_rows(rows[design_count:calibration_end], scores[design_count:calibration_end], graders)
    heldout = audit_rows(rows[calibration_end:], scores[calibration_end:], graders)
    primary = [replace(row, outcomes=row.outcomes[:1]) for row in calibration]
    for target in TARGETS:
        for scope, selected_rows, selected_graders in (
            ("primary", primary, graders[:1]), ("both", calibration, graders),
        ):
            for rule, registered in (("empirical", "empirical"), ("learn_then_test", "ltt-bonferroni")):
                report = select(selected_rows, graders=selected_graders, thresholds=thresholds,
                                risk_target=target, rule=registered)
                threshold = report["selected_threshold"]
                selected = None
                if threshold is not None:
                    accepted = [row for row in calibration if row.score >= threshold]
                    n = len(accepted)
                    first, second = (sum(not row.outcomes[index] for row in accepted) for index in (0, 1))
                    errors = first if scope == "primary" else max(first, second)
                    # Preserve the published reference representation of the tail.
                    selected = (threshold, n, first, second, float(binom.cdf(errors, n, target)))
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
        archived = load_archive(replay_dir / directory / "predictions_full.csv", expected, graders)
        design, calibration, heldout = split_rows(archived, SEED, id_key=itemgetter("id"))
        rows = design + calibration + heldout
        design_count, calibration_end = len(design), len(design) + len(calibration)
        scores, probabilities = score_archive(rows, design_count, graders[0])
        heldout_scores = {method: values[calibration_end:] for method, values in scores.items()}
        for method, values in heldout_scores.items():
            for grader in graders:
                predictions = [
                    GradedRow(row["id"], score, (row[grader],))
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


def ranking_rows(benchmark, method, label_source, rows, probability=False):
    metrics, curve = evaluate(rows, threshold=min(row.score for row in rows), probability=probability)
    for coverage in (0.25, 0.50, 0.75, 1.00):
        accepted = ceil(len(rows) * coverage)
        yield {
            "benchmark": benchmark,
            "method": method,
            "label_source": label_source,
            "target_coverage": coverage,
            "total": len(rows),
            "accepted": accepted,
            "coverage": accepted / len(rows),
            "error_rate": risk_at_count(curve, accepted),
            "brier_score": metrics["brier_score"],
        }


def policy_rows(benchmark, score, design, calibration, heldout):
    thresholds = design_thresholds(row.score for row in design)
    for target in BASELINE_TARGETS:
        for method, rule in (("empirical", "empirical"), ("learn_then_test_bonferroni", "ltt-bonferroni")):
            report = select(calibration, graders=("grade",), thresholds=thresholds, risk_target=target, rule=rule)
            threshold = report["selected_threshold"]
            counts = errors = 0
            pvalue = None
            test = Counts(len(heldout), 0, 0).metrics()
            if threshold is not None:
                candidate = next(item for item in report["candidates"] if item["threshold"] == threshold)
                counts, errors, pvalue = candidate["accepted"], candidate["errors"]["grade"], candidate["pvalue"]
                test = evaluate(heldout, threshold=threshold)[0]["overall"]
            yield {
                "benchmark": benchmark, "score": score, "policy": method, "risk_target": target,
                "candidates": len(thresholds), "pvalue_cutoff": report["per_candidate_level"],
                "threshold": threshold, "calibration_total": len(calibration),
                "calibration_accepted": counts, "calibration_errors": errors, "calibration_pvalue": pvalue,
                **test,
            }


def compare_helm(replay_dir):
    ranking, policies = [], []
    heldout_rows = {}
    for benchmark, (source, count, grader) in HELM.items():
        path = replay_dir / source / "predictions_full.csv"
        for score in SCORES:
            rows = sorted(
                load_csv(path, score_column=score, correct_columns=(grader,)),
                key=lambda row: row.id,
            )
            if len(rows) != count or len({row.id for row in rows}) != count:
                raise ValueError(f"{path} must contain {count} distinct benchmark responses")
            design, calibration, heldout = split_rows(rows, SEED)
            heldout_rows[benchmark, score] = heldout
            ranking.extend(ranking_rows(benchmark, score, "saved_qwen", heldout))
            policies.extend(policy_rows(benchmark, score, design, calibration, heldout))
        random = [replace(row, score=0.0) for row in heldout]
        ranking.extend(ranking_rows(benchmark, "random", "saved_qwen", random))
    paired = []
    for policy in policies:
        if policy["benchmark"] != "ifeval_google" or policy["threshold"] is None:
            continue
        score, threshold = policy["score"], policy["threshold"]
        google = heldout_rows["ifeval_google", score]
        helm = heldout_rows["ifeval_helm", score]
        if [(row.id, row.score) for row in google] != [(row.id, row.score) for row in helm]:
            raise ValueError("IFEval graders must share the same held-out IDs and scores")
        accepted = [(g.outcomes[0], h.outcomes[0]) for g, h in zip(google, helm, strict=True)
                    if g.score >= threshold]
        paired.append({
            "score": score,
            "policy_selected_using_google": policy["policy"],
            "risk_target": policy["risk_target"],
            "threshold": threshold,
            "total": len(google),
            "accepted": len(accepted),
            "google_errors": policy["errors"],
            "helm_errors": sum(not h for _, h in accepted),
            "google_pass_helm_fail": sum(g and not h for g, h in accepted),
            "google_fail_helm_pass": sum(not g and h for g, h in accepted),
        })
    return ranking, policies, paired


def compare_course():
    for benchmark in ("imdb", "jigsaw", "fever"):
        path = ROOT / "results" / "trust_scorer" / f"{benchmark}_lr_predictions.csv"
        saved = read_csv(path)
        methods = {
            "random": (lambda row: 0.0, False),
            "qwen_self_report": (lambda row: float(row["llm_confidence_numeric"]), True),
            "model_agreement": (lambda row: float(row["llm_label"] == row["deberta_predicted_label"]), False),
            "deberta_probability_of_qwen_label": (lambda row: float(row[f"prob_class{row['llm_label']}"]), True),
            "course_trust_score": (lambda row: float(row["trust_score_calibrated"]), True),
            "deberta_maximum_softmax_probability": (
                lambda row: max(float(value) for key, value in row.items() if key.startswith("prob_class")),
                True,
            ),
        }
        for method, (score, probability) in methods.items():
            label = "deberta_predicted_label" if method == "deberta_maximum_softmax_probability" else "llm_label"
            rows = [
                GradedRow(row["example_id"], score(row), (row[label] == row["gold_label"],), reference=row["gold_label"])
                for row in saved
            ]
            yield from ranking_rows(benchmark, method, label, rows, probability)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=ROOT / "replay")
    parser.add_argument("--output-dir", type=Path)
    studies = parser.add_mutually_exclusive_group()
    studies.add_argument("--course", action="store_true", help="Replay the saved course predictions.")
    studies.add_argument("--baselines", action="store_true", help="Replay the original Qwen HELM baselines.")
    args = parser.parse_args()
    output = args.output_dir or args.replay_dir / ("baselines" if args.course or args.baselines else "module_study")
    output.mkdir(parents=True, exist_ok=True)
    if args.course:
        write_report(output / "course.csv", list(compare_course()))
    elif args.baselines:
        ranking, policies, paired = compare_helm(args.replay_dir)
        write_report(output / "ranking.csv", ranking)
        write_report(output / "policies.csv", policies)
        write_report(output / "paired_ifeval.csv", paired)
    else:
        study(args.replay_dir, output)


if __name__ == "__main__":
    main()
