import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from .audit import audit
from .calibrate import CALIBRATORS, apply_csv, fit_calibration, read_calibration
from .collect import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    TransformersBackend,
    collect_responses,
    read_prompts,
    write_responses,
)
from .evaluate import evaluate
from .features import DEFAULT_FEATURES, FEATURES, write_feature_csv
from .html_report import write_html
from .metrics import Counts, threshold_points
from .plots import reliability_diagram, write_plot, write_svg
from .policy import policy, split_rows
from .queue import review_queue, write_queue
from .report import emit, make_report, write_curve, write_report
from .rules import RULES
from .select import design_thresholds, select
from .table import join_grader_files, load_csv, read_csv_rows, read_design_scores


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
    parser.add_argument(
        "--output-key-column", help="Column identifying the saved output."
    )
    parser.add_argument("--json", action="store_true", help="Print the JSON report.")
    parser.add_argument(
        "--output", type=Path, metavar="FILE", help="Write the JSON report to a file."
    )


def selection_options(parser):
    parser.add_argument(
        "--correct-column",
        action="append",
        required=True,
        help="Repeat for each selection grader.",
    )
    parser.add_argument("--risk-target", type=float, required=True)
    parser.add_argument("--failure-rate", type=float, default=0.05)
    parser.add_argument("--rule", choices=RULES, default="ltt-bonferroni")


def evaluate_options(parser):
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument(
        "--output-dir", type=Path, help="Also write metrics.json and risk_coverage.csv."
    )
    parser.add_argument(
        "--probability",
        action="store_true",
        help="Report Brier score, ECE, ACE, and reliability bins for probabilities.",
    )
    parser.add_argument(
        "--bins", type=int, default=10, help="Number of calibration bins."
    )
    parser.add_argument(
        "--reliability-plot",
        type=Path,
        metavar="FILE.svg",
        help="Write a reliability SVG.",
    )
    parser.add_argument(
        "--prediction-column", help="Defaults to 'prediction' for exact matching."
    )
    parser.add_argument(
        "--reference-column", help="Defaults to 'reference' for exact matching."
    )
    parser.add_argument(
        "--correct-column",
        action="append",
        help="A column of 0/1 grades for one grader.",
    )


def audit_options(parser):
    grades = parser.add_mutually_exclusive_group(required=True)
    grades.add_argument(
        "--correct-column",
        action="append",
        help="Repeat for each grader in the same CSV.",
    )
    grades.add_argument(
        "--grader-file",
        action="append",
        type=grader_source,
        metavar="NAME=CSV",
        help="Repeat for separate grader exports. Default output key is 'output_sha256'.",
    )
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--risk-target", type=float)
    parser.add_argument(
        "--prediction-filter",
        type=prediction_filter,
        metavar="COLUMN=VALUE",
        help="Read only matching prediction rows.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        metavar="N",
        help="Paired bootstrap replicates for 95%% intervals.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Bootstrap seed. Defaults to 42."
    )


def select_options(parser):
    selection_options(parser)
    grid = parser.add_mutually_exclusive_group(required=True)
    grid.add_argument(
        "--thresholds",
        action="extend",
        nargs="+",
        type=float,
        help="Fixed candidate cutoffs. Use --thresholds=-1e-5 for negative scientific notation.",
    )
    grid.add_argument(
        "--design", type=Path, help="Separate CSV with scores for a decile grid."
    )


def policy_options(parser):
    selection_options(parser)
    parser.add_argument(
        "--seed", type=int, default=42, help="Split seed. Defaults to 42."
    )
    parser.add_argument(
        "--check-column", action="append", help="Repeat for each held-out-only grader."
    )
    parser.add_argument("--calibrate", choices=CALIBRATORS, metavar="METHOD")
    parser.add_argument("--calibration-input", choices=("score", "probability"))


def calibrate_options(parser):
    operations = parser.add_subparsers(dest="operation", required=True)
    fit = operations.add_parser("fit", help="Fit a map using labeled calibration rows.")
    apply = operations.add_parser("apply", help="Apply a saved map to a CSV.")
    for command in (fit, apply):
        command.add_argument("predictions", type=Path, metavar="CSV")
        command.add_argument("--score-column", default="score")
        command.add_argument("--output", type=Path, required=True)
    fit.add_argument("--method", required=True, choices=CALIBRATORS)
    fit.add_argument("--input-scale", choices=("score", "probability"))
    fit.add_argument("--id-column", default="id")
    fit.add_argument("--correct-column", required=True)
    fit.add_argument("--bins", type=int, default=10, help="Number of histogram bins.")
    apply.add_argument("--map", type=Path, required=True)
    apply.add_argument("--probability-column", default="calibrated_probability")


def run_calibrate(args):
    applying = args.operation == "apply"
    inputs = {args.predictions.resolve()}
    if applying:
        inputs.add(args.map.resolve())
    if args.output.resolve() in inputs:
        raise ValueError("output would overwrite a calibration input")
    if applying:
        count = apply_csv(
            args.predictions,
            args.output,
            read_calibration(args.map),
            args.score_column,
            args.probability_column,
        )
        print(f"Wrote {count} calibrated probabilities to {args.output}.")
    else:
        rows = load_csv(
            args.predictions,
            id_column=args.id_column,
            score_column=args.score_column,
            correct_columns=(args.correct_column,),
        )
        mapping = fit_calibration(
            (row.score for row in rows),
            (row.outcomes[0] for row in rows),
            method=args.method,
            input_scale=args.input_scale,
            bins=args.bins,
        )
        write_report(args.output, json.dumps(mapping, indent=2, allow_nan=False))
        print(
            f"Fitted {args.method} on {len(rows)} calibration rows. Wrote {args.output}."
        )
    return 0


def group_quota(value):
    group, separator, limit = value.partition("=")
    if not group or not separator or not limit.isdecimal():
        raise argparse.ArgumentTypeError("use GROUP=COUNT for --quota")
    return group, int(limit)


def queue_options(parser):
    parser.add_argument("predictions", type=Path, metavar="CSV")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--group-column")
    parser.add_argument(
        "--correct-column", action="append", help="Optional grades for replay."
    )
    parser.add_argument("--direction", choices=("higher", "lower"), default="higher")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument(
        "--quota", action="append", type=group_quota, metavar="GROUP=COUNT"
    )
    parser.add_argument("--output", type=Path, metavar="CSV", required=True)
    parser.add_argument("--random-output", type=Path, metavar="CSV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")


def run_queue(args):
    paths = [
        path.resolve()
        for path in (args.predictions, args.output, args.random_output)
        if path is not None
    ]
    if len(paths) != len(set(paths)):
        raise ValueError("input and queue output paths must be distinct")
    quotas = dict(args.quota or ())
    if len(quotas) != len(args.quota or ()):
        raise ValueError("give at most one quota per group")
    graders = tuple(args.correct_column or ())
    rows, group = read_csv_rows(
        args.predictions,
        id_column=args.id_column,
        score_column=args.score_column,
        group_column=args.group_column,
        correct_columns=graders,
        require_labels=False,
    )
    chosen, random_rows, result = review_queue(
        rows,
        budget=args.budget,
        direction=args.direction,
        quotas=quotas,
        graders=graders,
        seed=args.seed,
    )
    write_queue(args.output, chosen)
    if args.random_output is not None:
        write_queue(args.random_output, random_rows)
    meta = {
        "path": str(args.predictions),
        "id_column": args.id_column,
        "score_column": args.score_column,
        "group_column": group,
        "correct_columns": list(graders),
        "output_key_column": None,
    }
    emit(make_report("queue", meta, result), json_output=args.json)
    return 0


def collect_options(parser):
    parser.add_argument("prompts", type=Path, metavar="JSONL")
    parser.add_argument("--output", type=Path, metavar="JSONL", required=True)
    parser.add_argument("--metadata", type=Path, metavar="JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--gpu-memory", help="Limit GPU model weights and offload to CPU (e.g. 10GiB)."
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--p-true", action="store_true")
    parser.add_argument("--verbalized", action="store_true")
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)


def run_collect(args):
    paths = [
        path.resolve()
        for path in (args.prompts, args.output, args.metadata)
        if path is not None
    ]
    if len(paths) != len(set(paths)):
        raise ValueError("prompt, output, and metadata paths must be distinct")
    if args.max_new_tokens < 1 or args.samples < 0:
        raise ValueError(
            "collect needs a positive token budget and a nonnegative sample count"
        )
    prompts = read_prompts(args.prompts)
    revision = args.revision or (
        DEFAULT_REVISION if args.model == DEFAULT_MODEL else None
    )
    start = perf_counter()
    try:
        backend = TransformersBackend(
            args.model,
            revision,
            device=args.device,
            seed=args.seed,
            local_files_only=args.local_files_only,
            gpu_memory=args.gpu_memory,
        )
        load_seconds = perf_counter() - start
        records, timings = collect_responses(
            prompts,
            backend,
            max_new_tokens=args.max_new_tokens,
            p_true=args.p_true,
            verbalized=args.verbalized,
            samples=args.samples,
        )
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"local model failed: {exc}") from exc
    write_responses(args.output, records)
    if args.metadata is not None:
        meta = {
            **backend.metadata,
            "prompt_count": len(prompts),
            "max_new_tokens": args.max_new_tokens,
            "samples_per_prompt": args.samples,
            "model_load_seconds": load_seconds,
            "methods": timings,
        }
        write_report(args.metadata, json.dumps(meta, indent=2, allow_nan=False))
    print(f"Collected {len(records)} responses to {args.output}.")
    return 0


def load_inputs(args):
    columns = tuple(args.correct_column or ())
    common = {
        "id_column": args.id_column,
        "score_column": args.score_column,
        "group_column": args.group_column,
    }
    meta = {
        "path": str(args.predictions),
        "id_column": args.id_column,
        "score_column": args.score_column,
        "correct_columns": list(columns),
        "output_key_column": args.output_key_column,
    }
    key = args.output_key_column
    selected = getattr(args, "prediction_filter", None)
    if selected is not None:
        meta["prediction_filter"] = dict(
            zip(("column", "value"), selected, strict=True)
        )
    files = getattr(args, "grader_file", None)
    if files:
        key = key or "output_sha256"
        rows, group = join_grader_files(
            args.predictions, files, **common, output_key_column=key, selection=selected
        )
        meta.update(
            output_key_column=key,
            grader_files={name: str(path) for name, path in files},
        )
    else:
        checks = tuple(getattr(args, "check_column", None) or ())
        if args.command == "policy":
            meta["check_columns"] = list(checks)
        prediction = getattr(args, "prediction_column", None)
        reference = getattr(args, "reference_column", None)
        if args.command == "evaluate":
            if len(columns) > 1:
                raise ValueError(
                    "evaluate needs one correctness column; use audit to compare graders"
                )
            if not columns:
                prediction, reference = (
                    prediction or "prediction",
                    reference or "reference",
                )
            meta.update(prediction_column=prediction, reference_column=reference)
        rows, group = read_csv_rows(
            args.predictions,
            **common,
            correct_columns=(*columns, *checks),
            output_key_column=key,
            prediction_column=prediction,
            reference_column=reference,
            selection=selected,
        )
    meta["group_column"] = group
    return rows, meta


def run(args):
    rows, meta = load_inputs(args)
    graders = tuple(args.correct_column or ())
    curve = None
    exit_code = 0
    if args.command == "evaluate":
        if (
            args.reliability_plot is not None or args.bins != 10
        ) and not args.probability:
            raise ValueError(
                "calibration bins and reliability plots require --probability"
            )
        result, curve = evaluate(
            rows, threshold=args.threshold, probability=args.probability, bins=args.bins
        )
    elif args.command == "audit":
        if args.grader_file:
            graders = tuple(name for name, _ in args.grader_file)
        result = audit(
            rows,
            graders=graders,
            threshold=args.threshold,
            risk_target=args.risk_target,
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
        if args.risk_target is not None:
            exit_code = int(
                any(
                    counts["meets_target"] is not True
                    for counts in result["graders"].values()
                )
            )
    elif args.command == "select":
        thresholds = args.thresholds
        if args.design is not None:
            if args.design.resolve() == args.predictions.resolve():
                raise ValueError("design and calibration must be separate CSVs")
            thresholds = design_thresholds(
                read_design_scores(args.design, args.score_column)
            )
        meta["design_path"] = str(args.design) if args.design is not None else None
        result = select(
            rows,
            graders=graders,
            thresholds=thresholds,
            risk_target=args.risk_target,
            failure_rate=args.failure_rate,
            rule=args.rule,
        )
        exit_code = int(result["selected_threshold"] is None)
    else:
        if args.calibration_input is not None and args.calibrate is None:
            raise ValueError("--calibration-input requires --calibrate")
        result = policy(
            rows,
            graders=graders,
            check_graders=tuple(args.check_column or ()),
            risk_target=args.risk_target,
            failure_rate=args.failure_rate,
            rule=args.rule,
            seed=args.seed,
            calibrate=args.calibrate,
            calibration_input=args.calibration_input,
        )
        exit_code = int(not result["confirmed"])
        if getattr(args, "html", None) is not None:
            _, _, rows = split_rows(rows, args.seed)
            graders = (*graders, *(args.check_column or ()))
    if getattr(args, "html", None) is not None and args.command in {"audit", "policy"}:
        threshold = (
            result["threshold"]
            if args.command == "audit"
            else result["selection"]["selected_threshold"]
        )
        accepted = [
            row for row in rows if threshold is not None and row.score >= threshold
        ]
        curves = {}
        for index, name in enumerate(graders):
            view = {
                "threshold": threshold,
                "overall": Counts(
                    len(rows),
                    len(accepted),
                    sum(not row.outcomes[index] for row in accepted),
                ).metrics(),
            }
            curves[name] = (
                view,
                threshold_points((row.score, not row.outcomes[index]) for row in rows),
            )
    else:
        curves = None
    return make_report(args.command, meta, result), curve, curves, exit_code


def output_paths(args):
    inputs = {
        args.predictions.resolve(),
        *(path.resolve() for _, path in (getattr(args, "grader_file", None) or ())),
    }
    design = getattr(args, "design", None)
    if design is not None:
        inputs.add(design.resolve())
    directory = getattr(args, "output_dir", None)
    artifacts = (
        (directory / "metrics.json", directory / "risk_coverage.csv")
        if directory is not None
        else ()
    )
    outputs = {*artifacts}
    if args.output is not None:
        if artifacts and args.output.resolve() == artifacts[1].resolve():
            raise ValueError("--output cannot replace risk_coverage.csv")
        outputs.add(args.output)
    for option, suffix in (
        ("plot", ".svg"),
        ("reliability_plot", ".svg"),
        ("html", ".html"),
    ):
        path = getattr(args, option, None)
        if path is None:
            continue
        flag = "--" + option.replace("_", "-")
        if path.suffix.lower() != suffix:
            raise ValueError(f"{flag} needs a {suffix} file")
        if path.resolve() in {output.resolve() for output in outputs}:
            raise ValueError(f"{flag} must use a different path from report outputs")
        outputs.add(path)
    if inputs.intersection(path.resolve() for path in outputs):
        raise ValueError("output would overwrite an input CSV")
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="caat", description="Evaluate saved predictions and review policies."
    )
    parser.add_argument(
        "--version", action="version", version=f"caat {version('caat-annotation')}"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    definitions = {
        "evaluate": ("Evaluate a fixed threshold.", evaluate_options),
        "audit": ("Compare graders at a fixed threshold.", audit_options),
        "select": ("Choose a cutoff on calibration data.", select_options),
        "policy": (
            "Split a CSV, choose a cutoff, and check it on held-out rows.",
            policy_options,
        ),
    }
    commands = {}
    for name, (help_text, configure) in definitions.items():
        command = subcommands.add_parser(name, help=help_text)
        column_options(command)
        configure(command)
        if name in {"evaluate", "audit", "policy"}:
            command.add_argument(
                "--plot",
                type=Path,
                metavar="FILE.svg",
                help="Write an SVG view of the report.",
            )
            command.add_argument(
                "--html",
                type=Path,
                metavar="FILE.html",
                help="Write a standalone HTML report.",
            )
        commands[name] = command
    feature_command = subcommands.add_parser(
        "features", help="Compute scores from saved responses."
    )
    feature_command.add_argument("predictions", type=Path, metavar="JSONL")
    feature_command.add_argument("--output", type=Path, metavar="CSV", required=True)
    feature_command.add_argument(
        "--feature", action="append", choices=FEATURES, help="Repeat to choose scores."
    )
    commands["features"] = feature_command
    commands["calibrate"] = subcommands.add_parser(
        "calibrate", help="Fit and apply correctness probability maps."
    )
    calibrate_options(commands["calibrate"])
    commands["queue"] = subcommands.add_parser(
        "queue", help="Order saved predictions for review or labeling."
    )
    queue_options(commands["queue"])
    commands["collect"] = subcommands.add_parser(
        "collect", help="Save responses and confidence from a local Transformers model."
    )
    collect_options(commands["collect"])
    args = parser.parse_args(argv)
    try:
        if args.command == "features":
            count = write_feature_csv(
                args.predictions, args.output, args.feature or DEFAULT_FEATURES
            )
            print(f"Wrote scores for {count} responses to {args.output}.")
            return 0
        if args.command == "calibrate":
            return run_calibrate(args)
        if args.command == "queue":
            return run_queue(args)
        if args.command == "collect":
            return run_collect(args)
        artifacts = output_paths(args)
        report, curve, curves, status = run(args)
        plot = getattr(args, "plot", None)
        if plot is not None:
            write_plot(plot, report, curve)
        reliability = getattr(args, "reliability_plot", None)
        if reliability is not None:
            write_svg(reliability, reliability_diagram(report["reliability_bins"]))
        html = getattr(args, "html", None)
        if html is not None:
            write_html(html, report, curve, curves)
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
