"""Replay correctness calibration on saved course and HELM predictions."""

import argparse
from math import exp
from pathlib import Path

from caat import GradedRow, evaluate
from caat.calibrate import CALIBRATORS, calibration_transform, fit_calibration
from caat.policy import split_rows

from common import ROOT, SEED, read_csv, write_report


HELM_MODEL = "qwen_qwen2.5-7b-instruct-turbo"


def archives(replay):
    for archive in ("imdb", "jigsaw", "fever"):
        saved = read_csv(ROOT / "results/course" / f"{archive}_lr_predictions.csv")
        for method in ("qwen_self_report", "deberta_probability_of_qwen_label"):
            yield (
                archive,
                method,
                [
                    GradedRow(
                        row["example_id"],
                        float(row["llm_confidence_numeric"])
                        if method == "qwen_self_report"
                        else float(row[f"prob_class{row['llm_label']}"]),
                        (row["llm_label"] == row["gold_label"],),
                    )
                    for row in saved
                ],
            )
    for archive, label, cutoff in (
        ("wildbench", "gpt_score", 7),
        ("omni_math", "gpt_equivalence_judgement", 1),
    ):
        saved = read_csv(replay / "helm_judges" / archive / f"{HELM_MODEL}.csv")
        yield (
            archive,
            "geometric_mean_token_probability",
            [
                GradedRow(
                    row["id"],
                    exp(float(row["mean_token_logprob"])),
                    (float(row[label]) >= cutoff,),
                )
                for row in saved
            ],
        )


def replay_calibration(rows):
    design, calibration, heldout = split_rows(rows, SEED)
    scores = [row.score for row in calibration]
    outcomes = [row.outcomes[0] for row in calibration]
    raw, _ = evaluate(heldout, threshold=0, probability=True)
    counts = {
        "design": len(design),
        "calibration": len(calibration),
        "heldout": len(heldout),
    }
    yield "raw", counts, raw
    for method in CALIBRATORS:
        mapping = fit_calibration(
            scores, outcomes, method=method, input_scale="probability"
        )
        predict = calibration_transform(mapping)
        mapped = [
            GradedRow(row.id, predict(row.score), row.outcomes) for row in heldout
        ]
        metrics, _ = evaluate(mapped, threshold=0, probability=True)
        yield method, counts, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=ROOT / "replay")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/calibration")
    args = parser.parse_args()
    summary, reliability = [], []
    for archive, score, rows in archives(args.replay_dir):
        for method, counts, metrics in replay_calibration(rows):
            run = {"archive": archive, "score": score, "method": method}
            summary.append(
                {
                    **run,
                    **counts,
                    **{
                        key: metrics[key]
                        for key in ("brier_score", "ece", "ace", "auroc")
                    },
                }
            )
            reliability.extend({**run, **row} for row in metrics["reliability_bins"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_report(args.output_dir / "summary.csv", summary)
    write_report(args.output_dir / "reliability.csv", reliability)


if __name__ == "__main__":
    main()
