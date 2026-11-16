import argparse
import csv
import json
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

from .audit import AuditRow, audit
from .evaluation import Prediction, evaluate
from .policy import build_policy, describe
from .selection import RULES, design_thresholds, select_threshold


@contextmanager
def open_predictions_csv(path: Path, required: set[str], group_column: str | None):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        header = reader.fieldnames or []
        if len(header) != len(set(header)):
            raise ValueError("CSV column names must be unique")
        if group_column is not None:
            required = required | {group_column}
        missing = sorted(required.difference(header))
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(missing)}. The file has: {', '.join(header)}")
        if group_column is None and "group" in header:
            group_column = "group"
        yield reader, group_column


def parse_score(value, line):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid score at CSV line {line}") from exc


def parse_correct(value, line):
    if value not in ("0", "1"):
        raise ValueError(f"correctness must be 0 or 1 at CSV line {line}")
    return value == "1"


def read_predictions(path: Path, columns: dict[str, str], group_column: str | None):
    with open_predictions_csv(path, set(columns.values()), group_column) as (reader, group_column):
        for row in reader:
            values = {name: row[column] for name, column in columns.items()}
            score = parse_score(values["score"], reader.line_num)
            correct = parse_correct(values["correct"], reader.line_num) if "correct" in columns else None
            group = (row[group_column] or None) if group_column is not None else None
            yield Prediction(
                values["id"], values.get("prediction"), values.get("reference"),
                score, group=group, correct=correct,
            )


def grader_source(value):
    name, _, path = value.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError("use NAME=CSV for --grader")
    return name, Path(path)


def prediction_filter(value):
    column, separator, expected = value.partition("=")
    if not column or not separator:
        raise argparse.ArgumentTypeError("use COLUMN=VALUE for --prediction-filter")
    return column, expected


@contextmanager
def open_audit_predictions(args, required):
    selection = args.prediction_filter
    if selection is not None:
        required = required | {selection[0]}
    with open_predictions_csv(args.predictions, required, args.group_column) as (reader, group_column):
        rows = reader
        if selection is not None:
            column, expected = selection
            rows = (row for row in reader if row[column] == expected)
        yield rows, reader, group_column


def join_grader_files(args):
    required = {args.id_column, args.score_column, args.output_key_column}
    with open_audit_predictions(args, required) as (rows, reader, group_column):
        predictions = {}
        for row in rows:
            item_id, key = row[args.id_column], row[args.output_key_column]
            if not item_id or item_id in predictions:
                raise ValueError(f"{args.predictions}: empty or duplicate id: {item_id!r}")
            if not key:
                raise ValueError(f"{args.predictions}: missing output key for {item_id!r}")
            predictions[item_id] = (
                parse_score(row[args.score_column], reader.line_num),
                key,
                (row[group_column] or None) if group_column is not None else None,
            )
    if not predictions:
        raise ValueError("at least one prediction is required")

    outcomes = []
    for name, path in args.grader:
        bits = {}
        with open_predictions_csv(path, {args.id_column, args.output_key_column, "correct"}, None) as (reader, _):
            for row in reader:
                item_id = row[args.id_column]
                if item_id not in predictions:
                    raise ValueError(f"{name}: unexpected id {item_id!r}")
                if item_id in bits:
                    raise ValueError(f"{name}: duplicate id {item_id!r}")
                if row[args.output_key_column] != predictions[item_id][1]:
                    raise ValueError(f"{name}: output key mismatch for {item_id!r}")
                bits[item_id] = parse_correct(row["correct"], reader.line_num)
        missing = predictions.keys() - bits.keys()
        if missing:
            raise ValueError(f"{name}: missing {len(missing)} ids, including {min(missing)!r}")
        outcomes.append(bits)
    rows = (
        AuditRow(item_id, score, tuple(bits[item_id] for bits in outcomes), group)
        for item_id, (score, _, group) in predictions.items()
    )
    return rows, group_column


def run_audit(args, command):
    if args.fail_on_target and args.risk_target is None:
        command.error("--fail-on-target requires --risk-target")
    try:
        options = {
            "threshold": args.threshold, "risk_target": args.risk_target,
            "bootstrap": args.bootstrap, "seed": args.seed,
        }
        if args.grader:
            graders = tuple(name for name, _ in args.grader)
            rows, group_column = join_grader_files(args)
            report = audit(rows, graders=graders, **options)
        else:
            graders = tuple(args.correct_column)
            required = {args.id_column, args.score_column, *graders}
            with open_audit_predictions(args, required) as (source_rows, reader, group_column):
                rows = (
                    AuditRow(
                        row[args.id_column], parse_score(row[args.score_column], reader.line_num),
                        tuple(parse_correct(row[name], reader.line_num) for name in graders),
                        (row[group_column] or None) if group_column is not None else None,
                    )
                    for row in source_rows
                )
                report = audit(rows, graders=graders, **options)
    except (OSError, ValueError, csv.Error) as exc:
        command.error(str(exc))
    report["input"] = {
        "path": str(args.predictions),
        "id_column": args.id_column,
        "score_column": args.score_column,
        "group_column": group_column,
    }
    if args.prediction_filter is not None:
        column, value = args.prediction_filter
        report["input"]["prediction_filter"] = {"column": column, "value": value}
    if args.grader:
        report["input"]["output_key_column"] = args.output_key_column
        report["input"]["grader_files"] = {name: str(path) for name, path in args.grader}
    print(json.dumps(report, indent=2))
    if args.fail_on_target and any(result["meets_target"] is not True for result in report["graders"].values()):
        return 1
    return 0


def read_graded_rows(path, id_column, score_column, graders):
    with open_predictions_csv(path, {id_column, score_column, *graders}, None) as (reader, _):
        return [
            AuditRow(row[id_column], parse_score(row[score_column], reader.line_num),
                     tuple(parse_correct(row[name], reader.line_num) for name in graders))
            for row in reader
        ]


def run_policy(args, command):
    graders, check_graders = tuple(args.correct_column), tuple(args.check_column or ())
    try:
        report = build_policy(
            read_graded_rows(args.predictions, args.id_column, args.score_column, (*graders, *check_graders)),
            graders=graders, check_graders=check_graders, risk_target=args.risk_target,
            failure_rate=args.failure_rate, rule=args.rule, seed=args.seed,
        )
    except (OSError, ValueError, csv.Error) as exc:
        command.error(str(exc))
    print(json.dumps(report, indent=2) if args.json else describe(report))
    return int(not report["confirmed"])


def run_select(args, command):
    graders = tuple(args.correct_column)
    try:
        thresholds = args.thresholds
        if args.design is not None:
            if args.design.resolve() == args.calibration.resolve():
                raise ValueError("design and calibration must be separate CSVs")
            with open_predictions_csv(args.design, {args.score_column}, None) as (reader, _):
                thresholds = design_thresholds(
                    parse_score(row[args.score_column], reader.line_num) for row in reader
                )
        report = select_threshold(
            read_graded_rows(args.calibration, args.id_column, args.score_column, graders),
            graders=graders, thresholds=thresholds, risk_target=args.risk_target,
            failure_rate=args.failure_rate, rule=args.rule,
        )
    except (OSError, ValueError, csv.Error) as exc:
        command.error(str(exc))
    print(json.dumps(report, indent=2))
    return int(report["selected_threshold"] is None)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="caat", description="Evaluate confidence-based annotation policies.")
    parser.add_argument("--version", action="version", version=f"caat {version('caat-annotation')}")
    commands = parser.add_subparsers(dest="command", required=True)
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
    command.add_argument("--id-column", default="id")
    command.add_argument("--score-column", default="score")
    command.add_argument("--prediction-column", help="Defaults to 'prediction' for exact matching.")
    command.add_argument("--reference-column", help="Defaults to 'reference' for exact matching.")
    command.add_argument("--correct-column", help="Read the evaluation outcome from a column of 0s and 1s.")
    command.add_argument("--group-column", help="Group column to summarize. Defaults to 'group' if present.")
    audit_command = commands.add_parser("audit", help="Compare graders at the same fixed threshold.")
    audit_command.add_argument("predictions", type=Path)
    audit_command.add_argument("--id-column", default="id")
    audit_command.add_argument("--score-column", default="score")
    audit_command.add_argument("--group-column", help="Group column to summarize. Defaults to 'group' if present.")
    audit_command.add_argument("--prediction-filter", type=prediction_filter, metavar="COLUMN=VALUE",
                               help="Read only matching prediction rows. Grader exports must cover those IDs.")
    graders = audit_command.add_mutually_exclusive_group(required=True)
    graders.add_argument("--correct-column", action="append",
                         help="Repeat for grader columns in an aligned CSV. The first is the baseline.")
    graders.add_argument("--grader", action="append", type=grader_source, metavar="NAME=CSV",
                         help="Repeat for separate grader exports keyed by ID and output.")
    audit_command.add_argument("--output-key-column", default="output_sha256",
                               help="Output identity column shared by predictions and grader files.")
    audit_command.add_argument("--threshold", type=float, required=True)
    audit_command.add_argument("--risk-target", type=float)
    audit_command.add_argument("--bootstrap", type=int, metavar="N", help="Paired bootstrap replicates for 95%% intervals.")
    audit_command.add_argument("--seed", type=int, default=42, help="Bootstrap seed. Defaults to 42.")
    audit_command.add_argument("--fail-on-target", action="store_true",
                               help="Return 1 if any grader exceeds the target or coverage is zero.")
    select_command = commands.add_parser("select", help="Choose a cutoff on labeled calibration data.")
    select_command.add_argument("calibration", type=Path)
    select_command.add_argument("--id-column", default="id")
    select_command.add_argument("--score-column", default="score")
    select_command.add_argument("--correct-column", action="append", required=True,
                                help="Repeat for each grader's outcome column.")
    select_command.add_argument("--risk-target", type=float, required=True)
    select_command.add_argument("--failure-rate", type=float, default=0.05)
    select_command.add_argument("--rule", choices=RULES, default="ltt-bonferroni")
    grid = select_command.add_mutually_exclusive_group(required=True)
    grid.add_argument("--thresholds", action="extend", nargs="+", type=float,
                      help="Fixed candidate cutoffs. Use --thresholds=-1e-5 for negative scientific notation.")
    grid.add_argument("--design", type=Path, help="Separate CSV with scores for a decile grid.")
    policy_command = commands.add_parser(
        "policy", help="Split one labeled CSV, choose a cutoff, and check it on held-out rows."
    )
    policy_command.add_argument("predictions", type=Path)
    policy_command.add_argument("--id-column", default="id")
    policy_command.add_argument("--score-column", default="score")
    policy_command.add_argument("--correct-column", action="append", required=True,
                                help="Grader outcome column the cutoff must satisfy. Repeat for each grader.")
    policy_command.add_argument("--check-column", action="append",
                                help="Grader outcome column reported on held-out rows only. Repeatable.")
    policy_command.add_argument("--risk-target", type=float, required=True)
    policy_command.add_argument("--failure-rate", type=float, default=0.05)
    policy_command.add_argument("--rule", choices=RULES, default="ltt-bonferroni")
    policy_command.add_argument("--seed", type=int, default=42, help="Split seed. Defaults to 42.")
    policy_command.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    args = parser.parse_args(argv)
    if args.command == "audit":
        return run_audit(args, audit_command)
    if args.command == "policy":
        return run_policy(args, policy_command)
    if args.command == "select":
        return run_select(args, select_command)

    columns = {"id": args.id_column, "score": args.score_column}
    if args.correct_column is None:
        columns["prediction"] = args.prediction_column or "prediction"
        columns["reference"] = args.reference_column or "reference"
    else:
        columns["correct"] = args.correct_column
        if args.prediction_column is not None:
            columns["prediction"] = args.prediction_column
        if args.reference_column is not None:
            columns["reference"] = args.reference_column
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
