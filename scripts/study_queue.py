"""Compare low-confidence and random review queues on saved class labels."""

import argparse
from math import ceil
from pathlib import Path

from caat.queue import review_queue

from common import ROOT, SEED, write_report
from study_calibration import course_archives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/course/queue.csv"
    )
    args = parser.parse_args()
    summary = []
    for archive, score, rows in course_archives():
        if score != "deberta_probability_of_qwen_label":
            continue
        _, _, report = review_queue(
            rows, budget=ceil(len(rows) / 4), graders=("qwen_label",), seed=SEED
        )
        summary.append(
            {
                "archive": archive,
                "score": score,
                "total": report["total"],
                "budget": report["budget"],
                "seed": SEED,
                **report["graders"]["qwen_label"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output, summary)


if __name__ == "__main__":
    main()
