import argparse
import csv
import json
from pathlib import Path

from .evaluation import Prediction, evaluate


def read_predictions(path: Path, columns: dict[str, str], group_column: str | None):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        header = reader.fieldnames or []
        if len(header) != len(set(header)):
            raise ValueError("CSV column names must be unique")
        required = set(columns.values())
        if group_column is not None:
            required.add(group_column)
        missing = sorted(required.difference(header))
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(missing)}")
        if group_column is None and "group" in header:
            group_column = "group"
        for row in reader:
            values = {name: row[column] for name, column in columns.items()}
            try:
                values["score"] = float(values["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid score at CSV line {reader.line_num}") from exc
            group = (row[group_column] or None) if group_column is not None else None
            yield Prediction(**values, group=group)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="caat", description="Evaluate confidence-based annotation policies.")
    commands = parser.add_subparsers(required=True)
    command = commands.add_parser(
        "evaluate", help="Evaluate a fixed confidence threshold on a labeled CSV."
    )
    command.add_argument("predictions", type=Path)
    command.add_argument(
        "--threshold", type=float, required=True,
        help="Accept rows with a score at or above this value.",
    )
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--probability", action="store_true",
        help="Treat scores as correctness probabilities and report Brier score.",
    )
    fields = ("id", "prediction", "reference", "score")
    for field in fields:
        command.add_argument(f"--{field}-column", default=field)
    command.add_argument("--group-column", help="Group column to summarize. Defaults to 'group' if present.")
    args = parser.parse_args(argv)

    columns = {field: getattr(args, f"{field}_column") for field in fields}
    try:
        metrics, curve = evaluate(
            read_predictions(args.predictions, columns, args.group_column),
            threshold=args.threshold,
            probability=args.probability,
        )
        metrics_path = args.output_dir / "metrics.json"
        curve_path = args.output_dir / "risk_coverage.csv"
        if args.predictions.resolve() in {metrics_path.resolve(), curve_path.resolve()}:
            raise ValueError("output would overwrite the input CSV")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        with curve_path.open("w", encoding="utf-8", newline="") as output:
            fields = ["threshold", "total", "accepted", "errors", "coverage", "error_rate"]
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for threshold, counts in curve:
                writer.writerow({"threshold": threshold, **counts.metrics()})
    except (OSError, ValueError, csv.Error) as exc:
        command.error(str(exc))
    print(f"Wrote {metrics_path} and {curve_path}")
