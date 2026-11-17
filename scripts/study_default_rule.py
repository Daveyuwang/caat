"""Compare threshold selection rules over many random splits of saved archives.

For each archive, score, and risk target, the script repeats `caat.policy`
with different split seeds. For each rule it reports how often a cutoff was
selected, the mean held-out coverage, and how often the held-out error
exceeded the target. All inputs are checked into the repository.
"""

import argparse
import csv
from pathlib import Path
from statistics import mean

from caat import load_csv, policy

ROOT = Path(__file__).resolve().parents[1]
RULES = ("ltt-bonferroni", "ltt-holm", "ltt-fixed-sequence", "empirical")
SEEDS = range(200)
FAILURE_RATE = 0.05

# (name, file, load options, grader name, risk targets)
SETTINGS = (
    (
        "ifeval_qwen/mean_token_logprob",
        "examples/ifeval_graders.csv",
        {"score_column": "mean_token_logprob", "correct_columns": ("strict_pass",)},
        "strict_pass",
        (0.25, 0.30),
    ),
    (
        "triviaqa_qwen/sequence_logprob",
        "results/black_box_1k/triviaqa_scores.csv",
        {"score_column": "sequence_logprob", "correct_columns": ("correct",)},
        "correct",
        (0.30, 0.40),
    ),
    (
        "mmlu_pro_qwen/mean_token_logprob",
        "results/black_box_1k/mmlu_pro_scores.csv",
        {"score_column": "mean_token_logprob", "correct_columns": ("correct",)},
        "correct",
        (0.40, 0.50),
    ),
    (
        "jigsaw_course/trust_score",
        "results/course/jigsaw_lr_predictions.csv",
        {
            "id_column": "example_id",
            "score_column": "trust_score_calibrated",
            "prediction_column": "llm_label",
            "reference_column": "gold_label",
        },
        "gold_label",
        (0.05, 0.10),
    ),
    (
        "imdb_course/trust_score",
        "results/course/imdb_lr_predictions.csv",
        {
            "id_column": "example_id",
            "score_column": "trust_score_calibrated",
            "prediction_column": "llm_label",
            "reference_column": "gold_label",
        },
        "gold_label",
        (0.02, 0.03),
    ),
    (
        "fever_course/trust_score",
        "results/course/fever_lr_predictions.csv",
        {
            "id_column": "example_id",
            "score_column": "trust_score_calibrated",
            "prediction_column": "llm_label",
            "reference_column": "gold_label",
        },
        "gold_label",
        (0.30, 0.40),
    ),
)


def summarize(rows, grader, target, rule):
    selected, coverage, violations = 0, [], 0
    for seed in SEEDS:
        report = policy(
            rows,
            graders=(grader,),
            risk_target=target,
            failure_rate=FAILURE_RATE,
            rule=rule,
            seed=seed,
        )
        if report["heldout"] is None:
            coverage.append(0.0)
            continue
        selected += 1
        held = report["heldout"]["graders"][grader]
        coverage.append(held["coverage"])
        violations += held["error_rate"] is not None and held["error_rate"] > target
    return {
        "selected_share": selected / len(SEEDS),
        "mean_heldout_coverage": mean(coverage),
        "heldout_error_above_target_share": violations / len(SEEDS),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "decision_baselines" / "default_rule.csv",
    )
    output = parser.parse_args().output
    records = []
    for name, path, options, grader, targets in SETTINGS:
        rows = load_csv(ROOT / path, **options)
        for target in targets:
            for rule in RULES:
                records.append(
                    {
                        "setting": name,
                        "rows": len(rows),
                        "risk_target": target,
                        "rule": rule,
                        "splits": len(SEEDS),
                        **summarize(rows, grader, target, rule),
                    }
                )
            print(f"{name} at {target}: done", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as target_file:
        writer = csv.DictWriter(
            target_file, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
