"""Compare review policies on saved HELM responses or the course predictions."""

import argparse
import csv
from dataclasses import replace
from math import ceil
from pathlib import Path
from random import Random

from caat import GradedRow, evaluate, load_csv
from caat.metrics import Counts, grouped_counts
from caat.rules import binomial_pvalue

ROOT = Path(__file__).resolve().parents[1]
HELM = {
    "ifeval_google": ("ifeval", 541, "strict_pass"),
    "ifeval_helm": ("ifeval", 541, "helm_strict_pass"),
    "mmlu_pro": ("mmlu_pro", 1000, "correct"),
}
SCORES = ("mean_token_logprob", "sequence_logprob")
SEED = 42
RISK_TARGETS = (0.20, 0.30)
FAILURE_RATE = 0.05


def risk_at_count(points, target):
    previous_accepted = previous_errors = 0
    for _, counts in points:
        if counts.accepted >= target:
            # Include the expected errors from a uniform draw inside the boundary tie.
            share = (target - previous_accepted) / (counts.accepted - previous_accepted)
            return (previous_errors + share * (counts.errors - previous_errors)) / target
        previous_accepted, previous_errors = counts.accepted, counts.errors
    raise ValueError("Target exceeds the sample size")


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
    ranked = sorted((row.score for row in design), reverse=True)
    thresholds = sorted({ranked[ceil(len(ranked) * i / 10) - 1] for i in range(1, 11)})
    candidates = []
    for threshold in thresholds:
        counts, _ = grouped_counts((None, row.score >= threshold, not row.outcomes[0]) for row in calibration)
        candidates.append((threshold, counts))
    cutoff = FAILURE_RATE / len(candidates)
    for target in RISK_TARGETS:
        empirical = certified = None
        for threshold, counts in candidates:
            if not counts.accepted:
                continue
            if empirical is None and counts.error_rate <= target:
                empirical = (threshold, counts)
            if certified is None and binomial_pvalue(counts, target) <= cutoff:
                certified = (threshold, counts)
        for method, selected in (("empirical", empirical), ("learn_then_test_bonferroni", certified)):
            threshold = calibration_counts = pvalue = None
            test = Counts(len(heldout), 0, 0).metrics()
            if selected is not None:
                threshold, calibration_counts = selected
                pvalue = binomial_pvalue(calibration_counts, target)
                test = evaluate(heldout, threshold=threshold)[0]["overall"]
            yield {
                "benchmark": benchmark,
                "score": score,
                "policy": method,
                "risk_target": target,
                "candidates": len(candidates),
                "pvalue_cutoff": cutoff if method == "learn_then_test_bonferroni" else None,
                "threshold": threshold,
                "calibration_total": len(calibration),
                "calibration_accepted": calibration_counts.accepted if calibration_counts else 0,
                "calibration_errors": calibration_counts.errors if calibration_counts else 0,
                "calibration_pvalue": pvalue,
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
            Random(SEED).shuffle(rows)
            design_end, calibration_end = count // 5, 3 * count // 5
            heldout = rows[calibration_end:]
            heldout_rows[benchmark, score] = heldout
            ranking.extend(ranking_rows(benchmark, score, "saved_qwen", heldout))
            policies.extend(policy_rows(
                benchmark, score, rows[:design_end], rows[design_end:calibration_end], heldout,
            ))
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
        with path.open(encoding="utf-8", newline="") as source:
            saved = list(csv.DictReader(source))
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


def write_report(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=ROOT / "replay")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replay" / "baselines")
    parser.add_argument("--course", action="store_true", help="Compare the saved course predictions.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.course:
        write_report(args.output_dir / "course.csv", list(compare_course()))
    else:
        ranking, policies, paired = compare_helm(args.replay_dir)
        write_report(args.output_dir / "ranking.csv", ranking)
        write_report(args.output_dir / "policies.csv", policies)
        write_report(args.output_dir / "paired_ifeval.csv", paired, (
            "score", "policy_selected_using_google", "risk_target", "threshold", "total",
            "accepted", "google_errors", "helm_errors", "google_pass_helm_fail", "google_fail_helm_pass",
        ))


if __name__ == "__main__":
    main()
