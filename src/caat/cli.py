import argparse
import csv
from importlib.metadata import version
from pathlib import Path

from .audit import audit
from .evaluate import evaluate
from .policy import policy
from .report import emit, make_report, write_curve, write_report
from .rules import RULES
from .select import design_thresholds, select
from .table import join_grader_files, read_csv_rows, read_design_scores


def grader_source(value):
    name, _, path = value.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError("use NAME=CSV for --grader-file")
    return name, Path(path)


def prediction_filter(value):
    column, separator, expected = value.partition("=")
    if not column or not separator:
        raise argparse.ArgumentTypeError("use COLUMN=VALUE for --prediction-filter")
    return column, expected


def column_options(parser):
    parser.add_argument("predictions", type=Path, metavar="CSV")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--group-column", help="Defaults to 'group' when present.")
    parser.add_argument("--output-key-column", help="Column identifying the saved output.")
    parser.add_argument("--json", action="store_true", help="Print the JSON report.")
    parser.add_argument("--output", type=Path, metavar="FILE", help="Write the JSON report to a file.")


def selection_options(parser):
    parser.add_argument("--correct-column", action="append", required=True, help="Repeat for each selection grader.")
    parser.add_argument("--risk-target", type=float, required=True)
    parser.add_argument("--failure-rate", type=float, default=0.05)
    parser.add_argument("--rule", choices=RULES, default="ltt-bonferroni")


def evaluate_options(parser):
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, help="Also write metrics.json and risk_coverage.csv.")
    parser.add_argument("--probability", action="store_true", help="Report Brier score for correctness probabilities.")
    parser.add_argument("--prediction-column", help="Defaults to 'prediction' for exact matching.")
    parser.add_argument("--reference-column", help="Defaults to 'reference' for exact matching.")
    parser.add_argument("--correct-column", action="append", help="A column of 0/1 grades for one grader.")


def audit_options(parser):
    grades = parser.add_mutually_exclusive_group(required=True)
    grades.add_argument("--correct-column", action="append", help="Repeat for each grader in the same CSV.")
    grades.add_argument("--grader-file", action="append", type=grader_source, metavar="NAME=CSV",
                        help="Repeat for separate grader exports. Default output key is 'output_sha256'.")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--risk-target", type=float)
    parser.add_argument("--prediction-filter", type=prediction_filter, metavar="COLUMN=VALUE",
                        help="Read only matching prediction rows.")
    parser.add_argument("--bootstrap", type=int, metavar="N", help="Paired bootstrap replicates for 95%% intervals.")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap seed. Defaults to 42.")


def select_options(parser):
    selection_options(parser)
    grid = parser.add_mutually_exclusive_group(required=True)
    grid.add_argument("--thresholds", action="extend", nargs="+", type=float,
                      help="Fixed candidate cutoffs. Use --thresholds=-1e-5 for negative scientific notation.")
    grid.add_argument("--design", type=Path, help="Separate CSV with scores for a decile grid.")


def policy_options(parser):
    selection_options(parser)
    parser.add_argument("--seed", type=int, default=42, help="Split seed. Defaults to 42.")
    parser.add_argument("--check-column", action="append", help="Repeat for each held-out-only grader.")


def load_inputs(args):
    columns = tuple(args.correct_column or ())
    common = {"id_column": args.id_column, "score_column": args.score_column, "group_column": args.group_column}
    meta = {"path": str(args.predictions), "id_column": args.id_column, "score_column": args.score_column,
            "correct_columns": list(columns), "output_key_column": args.output_key_column}
    key = args.output_key_column
    selected = getattr(args, "prediction_filter", None)
    if selected is not None:
        meta["prediction_filter"] = dict(zip(("column", "value"), selected, strict=True))
    files = getattr(args, "grader_file", None)
    if files:
        key = key or "output_sha256"
        rows, group = join_grader_files(args.predictions, files, **common, output_key_column=key, selection=selected)
        meta.update(output_key_column=key, grader_files={name: str(path) for name, path in files})
    else:
        checks = tuple(getattr(args, "check_column", None) or ())
        if args.command == "policy":
            meta["check_columns"] = list(checks)
        prediction = getattr(args, "prediction_column", None)
        reference = getattr(args, "reference_column", None)
        if args.command == "evaluate":
            if len(columns) > 1:
                raise ValueError("evaluate needs one correctness column; use audit to compare graders")
            if not columns:
                prediction, reference = prediction or "prediction", reference or "reference"
            meta.update(prediction_column=prediction, reference_column=reference)
        rows, group = read_csv_rows(
            args.predictions, **common, correct_columns=(*columns, *checks),
            output_key_column=key, prediction_column=prediction, reference_column=reference, selection=selected,
        )
    meta["group_column"] = group
    return rows, meta


def run(args):
    rows, meta = load_inputs(args)
    graders = tuple(args.correct_column or ())
    curve = None
    exit_code = 0
    if args.command == "evaluate":
        result, curve = evaluate(rows, threshold=args.threshold, probability=args.probability)
    elif args.command == "audit":
        if args.grader_file:
            graders = tuple(name for name, _ in args.grader_file)
        result = audit(rows, graders=graders, threshold=args.threshold, risk_target=args.risk_target,
                       bootstrap=args.bootstrap, seed=args.seed)
        if args.risk_target is not None:
            exit_code = int(any(counts["meets_target"] is not True for counts in result["graders"].values()))
    elif args.command == "select":
        thresholds = args.thresholds
        if args.design is not None:
            if args.design.resolve() == args.predictions.resolve():
                raise ValueError("design and calibration must be separate CSVs")
            thresholds = design_thresholds(read_design_scores(args.design, args.score_column))
        meta["design_path"] = str(args.design) if args.design is not None else None
        result = select(rows, graders=graders, thresholds=thresholds, risk_target=args.risk_target,
                        failure_rate=args.failure_rate, rule=args.rule)
        exit_code = int(result["selected_threshold"] is None)
    else:
        result = policy(rows, graders=graders, check_graders=tuple(args.check_column or ()),
                        risk_target=args.risk_target, failure_rate=args.failure_rate, rule=args.rule, seed=args.seed)
        exit_code = int(not result["confirmed"])
    return make_report(args.command, meta, result), curve, exit_code


def output_paths(args):
    inputs = {args.predictions.resolve(), *(path.resolve() for _, path in (getattr(args, "grader_file", None) or ()))}
    design = getattr(args, "design", None)
    if design is not None:
        inputs.add(design.resolve())
    directory = getattr(args, "output_dir", None)
    artifacts = (directory / "metrics.json", directory / "risk_coverage.csv") if directory is not None else ()
    outputs = {*artifacts}
    if args.output is not None:
        if artifacts and args.output.resolve() == artifacts[1].resolve():
            raise ValueError("--output cannot replace risk_coverage.csv")
        outputs.add(args.output)
    if inputs.intersection(path.resolve() for path in outputs):
        raise ValueError("output would overwrite an input CSV")
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(prog="caat", description="Evaluate saved predictions and review policies.")
    parser.add_argument("--version", action="version", version=f"caat {version('caat-annotation')}")
    subcommands = parser.add_subparsers(dest="command", required=True)
    definitions = {
        "evaluate": ("Evaluate a fixed threshold.", evaluate_options),
        "audit": ("Compare graders at a fixed threshold.", audit_options),
        "select": ("Choose a cutoff on calibration data.", select_options),
        "policy": ("Split a CSV, choose a cutoff, and check it on held-out rows.", policy_options),
    }
    commands = {}
    for name, (help_text, configure) in definitions.items():
        command = subcommands.add_parser(name, help=help_text)
        column_options(command)
        configure(command)
        commands[name] = command
    args = parser.parse_args(argv)
    try:
        artifacts = output_paths(args)
        report, curve, status = run(args)
        if artifacts:
            artifact, curve_path = artifacts
            artifact.parent.mkdir(parents=True, exist_ok=True)
            write_curve(curve_path, curve)
            if args.output is None:
                args.output = artifact
        serialized = emit(report, json_output=args.json, output=args.output)
        if artifacts and args.output.resolve() != artifacts[0].resolve():
            write_report(artifacts[0], serialized)
        return status
    except (OSError, ValueError, csv.Error) as exc:
        commands[args.command].error(str(exc))
