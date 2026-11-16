"""Replay IFEval, MMLU-Pro, and TriviaQA from the checked-in CSV samples."""

import argparse
import csv
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.cli import main as caat_main

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = {
    "ifeval": (
        "ifeval_strict.csv",
        ("--score-column", "mean_token_logprob", "--correct-column", "strict_pass", "--threshold", "-0.35"),
    ),
    "mmlu_pro": (
        "mmlu_pro.csv",
        ("--score-column", "mean_token_logprob", "--correct-column", "correct",
         "--reference-column", "reference", "--threshold", "-0.3"),
    ),
    "triviaqa": (
        "triviaqa.csv",
        ("--score-column", "response_prob", "--correct-column", "correct", "--threshold", "0.5"),
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replay" / "benchmarks")
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        for benchmark, (sample, flags) in BENCHMARKS.items():
            run = stage / benchmark
            with redirect_stdout(StringIO()):
                caat_main(["evaluate", str(ROOT / "examples" / sample), *flags, "--output-dir", str(run)])
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            rows.append({"benchmark": benchmark, "threshold": metrics["threshold"], **metrics["overall"]})

        summary = stage / "summary.csv"
        with summary.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        for benchmark in BENCHMARKS:
            destination = output / benchmark
            destination.mkdir(exist_ok=True)
            for artifact in (stage / benchmark).iterdir():
                artifact.replace(destination / artifact.name)
        summary.replace(output / summary.name)

    print(f"Wrote {output / summary.name} and outputs for {', '.join(BENCHMARKS)}")


if __name__ == "__main__":
    main()
