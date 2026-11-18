import argparse
import csv
import json
import sys
from dataclasses import replace
from importlib.metadata import version
from operator import attrgetter
from os import environ
from pathlib import Path
from time import perf_counter

from . import eee_log, helm_log, lm_eval_log
from .adapters import ADAPTERS
from .artifacts import staged_outputs
from .audit import audit, write_disagreements
from .audit_blocks import block_options, run_block
from .calibrate import (
    CALIBRATORS,
    apply_csv,
    default_scale,
    fit_calibration,
    read_saved_calibration,
)
from .collect import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    TransformersBackend,
    collect_responses,
    read_prompts,
    write_responses,
)
from .collect_http import ChatCompletionsBackend
from .evaluate import evaluate
from .features import DEFAULT_FEATURES, FEATURES, write_feature_csv
from .inspect_log import export_for_scoring, export_grades, join_scores
from .metrics import METRICS, Counts, threshold_points
from .monitor import monitor_risk
from .plan import (
    DEFAULT_MAX_ACCEPTED,
    batch_verdict,
    check_candidate_sample,
    check_label_plan,
    forecast_block_cost,
    forecast_verdict,
    plan_labels,
    read_candidate_batch,
    sample_error_limits,
)
from .partial import evaluate_partial
from .two_stage import design as two_stage_design
from .plots import reliability_diagram, write_svg
from .plugins import catalog
from .policy import policy, split_rows
from .queue import read_sample_columns, review_queue, shuffled_by_id, write_queue
from .renderers import RENDERERS, render_report
from .report import (
    check_fit_overlap,
    emit,
    fit_outputs_metadata,
    make_report,
    read_fit_outputs,
    read_label_plan,
    read_threshold_report,
    write_curve,
    write_report,
)
from .route import (
    BatchDecision,
    iter_batch_decisions,
    read_batch_decisions,
    route_batch,
)
from .rules import RULES
from .scores import COSTS, DEFAULT_COVERAGES, compare_scores, read_score_rows
from .scorer import LinearScorer, configure_scorer, run_scorer
from .select import DEFAULT_DESIGN_GRID_SIZE, design_thresholds, select
from .table import (
    accepted_at,
    join_grader_files,
    negated_score_name,
    open_prediction_table,
    orient_score_table,
    parse_prediction_filter,
    read_design_scores,
    read_grader_file,
    read_table_rows,
    validated_rows,
    write_holdout_rows,
)
from .workflow import run_workflow


def named_file(value):
    name, _, path = value.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError("use NAME=FILE")
    return name, Path(path)


def prediction_filter(value):
    try:
        return parse_prediction_filter(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use COLUMN=VALUE for --prediction-filter"
        ) from error


def prediction_filter_option(parser):
    parser.add_argument(
        "--prediction-filter",
        type=prediction_filter,
        metavar="COLUMN=VALUE",
        help="Read only matching prediction rows.",
    )


def review_missing_scores_option(
    parser,
    *,
    help_text="Keep rows with empty scores in coverage and treat them as reviewed.",
):
    parser.add_argument(
        "--review-missing-scores",
        action="store_true",
        default=None,
        help=help_text,
    )


def column_options(parser):
    parser.add_argument("predictions", type=Path, metavar="CSV_OR_JSONL")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--group-column", help="Defaults to 'group' when present.")
    review_missing_scores_option(parser)
    parser.add_argument(
        "--adapter", choices=ADAPTERS, help="Read input through a registered adapter."
    )
    parser.add_argument("--render", choices=RENDERERS, metavar="NAME")
    parser.add_argument("--render-output", type=Path, metavar="FILE")
    parser.add_argument(
        "--output-key-column", help="Column identifying the saved output."
    )
    parser.add_argument("--json", action="store_true", help="Print the JSON report.")
    parser.add_argument(
        "--output", type=Path, metavar="FILE", help="Write the JSON report to a file."
    )


def score_direction_option(parser):
    parser.add_argument(
        "--direction",
        choices=("higher", "lower"),
        default="higher",
        help="Direction of higher confidence. Defaults to higher scores.",
    )


def selection_options(parser):
    prediction_filter_option(parser)
    score_direction_option(parser)
    grader_source_options(parser, labels=True)
    parser.add_argument("--risk-target", type=float, required=True)
    parser.add_argument("--failure-rate", type=float, default=0.05)
    parser.add_argument("--rule", choices=RULES, default="ltt-bonferroni")
    parser.add_argument(
        "--design-grid-size",
        type=int,
        metavar="N",
        help=f"Number of design-score quantiles (default {DEFAULT_DESIGN_GRID_SIZE}).",
    )
    parser.add_argument(
        "--track-fit-outputs",
        action="store_true",
        help="Save hashes of design and calibration identities for later holdout checks.",
    )


def raw_score_column_option(parser):
    parser.add_argument(
        "--raw-score-column",
        metavar="COLUMN",
        help="Read this column using the saved score direction.",
    )


def score_column_override(args):
    if args.raw_score_column is None:
        return args.score_column
    if args.score_column is not None:
        raise ValueError("use only one of --score-column and --raw-score-column")
    return args.raw_score_column


def threshold_options(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--threshold", type=float)
    source.add_argument(
        "--threshold-report",
        type=Path,
        metavar="JSON",
        help="Read a cutoff and settings from a select, policy, or run report.",
    )
    raw_score_column_option(parser)
    parser.set_defaults(id_column=None, score_column=None)


def use_plan_sample_columns(args, saved, sample_path):
    """Set sample columns and flag an ambiguous score direction."""
    if args.adapter not in (None, "csv") or all(
        name is not None
        for name in (args.id_column, args.score_column, args.output_key_column)
    ):
        return False
    with open_prediction_table(args.predictions, set(), None) as (reader, _):
        fields = set(reader.fieldnames or ())
    if not {"id", "score"} <= fields:
        return False

    # A labeled copy can retain the source columns. Prefer those when they
    # are all present unless this is the sample file the plan wrote.
    raw_id = args.id_column or saved.get("id_column", "id")
    raw_score = (
        args.score_column
        or saved.get("negated_score_source")
        or saved.get("score_column", "score")
    )
    raw_key = args.output_key_column or (saved.get("fit_outputs") or {}).get(
        "key_column"
    )
    if raw_key is None and "output_key" in fields:
        raw_key = saved.get("output_key_column") or "output_sha256"
    features = (
        LinearScorer.from_dict(saved["scorer"]).input_columns
        if args.score_column is None and "scorer" in saved
        else ()
    )
    original_columns = {raw_id, raw_key, *(features or (raw_score,))} - {None}
    if (
        args.predictions.resolve() != Path(sample_path).resolve()
        and original_columns <= fields
    ):
        return (
            args.score_column is None and saved.get("negated_score_source") == "score"
        )

    if args.id_column is None:
        args.id_column = "id"
    if args.score_column is None:
        args.score_column = "score"
    if args.output_key_column is None and "output_key" in fields:
        args.output_key_column = "output_key"
    return False


def load_threshold_options(args, *, require_confirmed=False, label_plan=None):
    if args.raw_score_column is not None and args.threshold_report is None:
        raise ValueError("--raw-score-column requires --threshold-report")
    args.score_column = score_column_override(args)
    saved = (
        read_threshold_report(
            args.threshold_report,
            require_confirmed=require_confirmed,
            fingerprint=getattr(args, "batch_predictions", None) is not None,
        )
        if args.threshold_report is not None
        else {}
    )
    args.allow_negated_sample_scores = False
    if label_plan is not None and (sample_path := label_plan["sample_path"]):
        args.allow_negated_sample_scores = use_plan_sample_columns(
            args, saved, sample_path
        )
    use_report_score = args.score_column is None
    args.scorer = (
        LinearScorer.from_dict(saved["scorer"])
        if use_report_score and "scorer" in saved
        else None
    )
    options = vars(args)
    for name in (
        "threshold",
        "risk_target",
        "holdout_confidence",
        "id_column",
        "score_column",
        "review_missing_scores",
    ):
        if name in options and options[name] is None:
            options[name] = saved.get(name)
    if args.id_column is None:
        args.id_column = "id"
    if args.score_column is None:
        args.score_column = "score"
    source = saved.get("negated_score_source")
    if args.raw_score_column is not None and source is not None:
        args.negated_score_source = args.raw_score_column
        args.score_column = negated_score_name(args.raw_score_column)
    else:
        args.negated_score_source = source if use_report_score else None
    return saved


def use_available_output_key_column(args, saved):
    key_column = saved.get("output_key_column")
    if (
        args.output_key_column is not None
        or key_column is None
        or getattr(args, "adapter", None) not in (None, "csv")
    ):
        return
    if (batch := getattr(args, "batch_decisions", None)) is not None:
        # Old decision files can omit output keys even when the source has them.
        with open_prediction_table(batch, set(), None) as (reader, _):
            if "output_key" not in reader.fieldnames:
                return
    with open_prediction_table(args.predictions, set(), None) as (reader, _):
        if key_column in reader.fieldnames:
            args.output_key_column = key_column


def threshold_input_paths(args):
    paths = set()
    for name in (
        "predictions",
        "threshold_report",
        "batch_decisions",
        "batch_predictions",
        "plan_report",
    ):
        if path := getattr(args, name, None):
            paths.add(path.resolve())
    return paths


def evaluate_options(parser):
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Check whether a prefix of a fixed-budget draw determines its final verdict.",
    )
    threshold_options(parser)
    prediction_filter_option(parser)
    parser.add_argument(
        "--risk-target", type=float, help="Maximum accepted error rate."
    )
    holdout_confidence_option(parser)
    accepted_population_options(parser, check_sample=True)
    parser.add_argument(
        "--metric", action="append", choices=METRICS, help="Add a registered metric."
    )
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
    grades = parser.add_mutually_exclusive_group()
    grades.add_argument(
        "--correct-column",
        action="append",
        help="A column of 0/1 grades for one grader.",
    )
    grades.add_argument(
        "--grader-file",
        action="append",
        type=named_file,
        metavar="NAME=FILE",
        help="Read one separate grader export. Default output key is 'output_sha256'.",
    )
    grader_export_options(parser)


def lower_is_better_option(parser, *, required=False):
    parser.add_argument(
        "--lower-is-better",
        action="append",
        required=required,
        metavar="COLUMN",
        help="Treat lower values in the named score column as higher confidence.",
    )


def external_score_options(parser):
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--scores", type=Path, metavar="CSV_OR_JSONL")
    sources.add_argument(
        "--score-file",
        action="append",
        type=named_file,
        metavar="COLUMN=FILE",
        help="Repeat to read named score columns from CSV or JSONL files.",
    )
    parser.add_argument(
        "--score-column", help="Column in --scores. Defaults to confidence."
    )
    lower_is_better_option(parser)
    review_missing_scores_option(
        parser, help_text="Keep rows with missing external scores as empty score cells."
    )


def external_score_settings(args):
    if args.score_file and args.score_column is not None:
        raise ValueError("--score-column is used with --scores")
    return {
        "score_column": "confidence"
        if args.score_column is None
        else args.score_column,
        "score_files": args.score_file or (),
        "lower_is_better": args.lower_is_better or (),
        "review_missing_scores": bool(args.review_missing_scores),
    }


def inspect_options(parser):
    commands = parser.add_subparsers(dest="inspect_command", required=True)
    export = commands.add_parser(
        "export", help="Export completions for external scoring."
    )
    join = commands.add_parser(
        "join", help="Match external scores and export grader columns."
    )
    grades = commands.add_parser(
        "grades", help="Match a saved scorer to a prediction table for audit."
    )
    for command in (export, join, grades):
        command.add_argument("log", type=Path, metavar="LOG")
        command.add_argument(
            "--output",
            type=Path,
            required=True,
            metavar="CSV_OR_JSONL" if command is export else "CSV",
        )
        command.add_argument(
            "--epoch", type=int, default=1, help="Epoch to export. Defaults to 1."
        )
    grades.add_argument(
        "--predictions", type=Path, required=True, metavar="CSV_OR_JSONL"
    )
    grades.add_argument(
        "--grader", required=True, help="Name of the saved binary scorer."
    )
    external_score_options(join)
    join.add_argument(
        "--scores-id-column",
        metavar="COLUMN",
        help="Match scores on the sample ID in this column alone. The score files "
        "then need no evaluation ID, epoch, or output hash, and CAAT cannot check "
        "that the scores were computed for the completions in this log.",
    )
    join.add_argument(
        "--grader",
        action="append",
        required=True,
        help="Repeat for each named binary scorer.",
    )


def run_inspect(args):
    if args.inspect_command == "export":
        count = export_for_scoring(args.log, args.output, epoch=args.epoch)
    elif args.inspect_command == "grades":
        count = export_grades(
            args.log,
            args.predictions,
            args.output,
            grader=args.grader,
            epoch=args.epoch,
        )
    else:
        count = join_scores(
            args.log,
            args.scores,
            args.output,
            graders=args.grader,
            epoch=args.epoch,
            scores_id_column=args.scores_id_column,
            **external_score_settings(args),
        )
    print(f"Wrote {count} rows to {args.output}.")
    if args.inspect_command == "join" and args.scores_id_column is not None:
        print(
            f"Scores were matched on {args.scores_id_column} alone. The output hash "
            "was not checked against the completions in the log."
        )
    return 0


def lm_eval_options(parser):
    commands = parser.add_subparsers(dest="lm_eval_command", required=True)
    export = commands.add_parser(
        "export", help="Export saved text for external scoring."
    )
    join = commands.add_parser("join", help="Match external scores with saved grades.")
    choices = commands.add_parser(
        "choices", help="Extract multiple-choice likelihood scores."
    )
    for command in (export, join, choices):
        command.add_argument("log", type=Path, metavar="SAMPLES.jsonl")
        command.add_argument("--output", type=Path, required=True, metavar="CSV")
        command.add_argument("--filter", help="Select one scoring filter in the log.")
    external_score_options(join)
    join.add_argument(
        "--grader",
        action="append",
        required=True,
        help="Repeat for each binary metric.",
    )


def run_lm_eval(args):
    options = {"sample_filter": args.filter}
    if args.lm_eval_command == "choices":
        count = lm_eval_log.choice_scores(args.log, args.output, **options)
    elif args.lm_eval_command == "export":
        count = lm_eval_log.export_generations(args.log, args.output, **options)
    else:
        count = lm_eval_log.join_scores(
            args.log,
            args.scores,
            args.output,
            graders=args.grader,
            **options,
            **external_score_settings(args),
        )
    print(f"Wrote {count} rows to {args.output}.")
    return 0


def eee_options(parser):
    commands = parser.add_subparsers(dest="eee_command", required=True)
    export = commands.add_parser(
        "export", help="Export saved text for external scoring."
    )
    join = commands.add_parser("join", help="Match external scores with saved grades.")
    for command in (export, join):
        command.add_argument("log", type=Path, metavar="SAMPLES.jsonl")
        command.add_argument("--output", type=Path, required=True, metavar="CSV")
        command.add_argument(
            "--result", help="Select a result ID or legacy evaluation name."
        )
    external_score_options(join)


def run_eee(args):
    if args.eee_command == "export":
        count = eee_log.export_generations(args.log, args.output, result=args.result)
    else:
        count = eee_log.join_scores(
            args.log,
            args.scores,
            args.output,
            result=args.result,
            **external_score_settings(args),
        )
    print(f"Wrote {count} rows to {args.output}.")
    return 0


def helm_options(parser):
    parser.add_argument("run", type=Path, metavar="RUN_DIRECTORY")
    parser.add_argument("--output", type=Path, required=True, metavar="CSV")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        metavar="NAME",
        help="Repeat to use full credit on a published per-instance metric as a grade.",
    )
    parser.add_argument(
        "--annotation",
        action="append",
        default=[],
        metavar="GROUP.FIELD",
        help="Repeat for each boolean annotation in the saved request state.",
    )


def holdout_confidence_option(parser):
    parser.add_argument(
        "--holdout-confidence",
        type=float,
        metavar="LEVEL",
        help="Check the risk target against a one-sided exact upper bound.",
    )


def accepted_population_options(parser, *, check_sample=False):
    population = parser.add_mutually_exclusive_group()
    population.add_argument(
        "--accepted-population",
        type=int,
        metavar="N",
        help="Total accepted rows in a fixed batch sampled uniformly without replacement.",
    )
    population.add_argument(
        "--batch-decisions",
        type=Path,
        metavar="FILE",
        help="Read the full fixed batch from a caat route decision table.",
    )
    population.add_argument(
        "--batch-predictions",
        type=Path,
        metavar="CSV_OR_JSONL",
        help="Score a new batch for a fixed-budget audit sample.",
    )
    if check_sample:
        parser.add_argument(
            "--plan-report",
            type=Path,
            metavar="JSON",
            help="Check the labeled sample against a saved caat plan report.",
        )
        parser.add_argument(
            "--sample-seed",
            type=int,
            metavar="SEED",
            help="Verify the random queue draw from --batch-decisions with this seed.",
        )


def grader_export_options(parser):
    parser.add_argument(
        "--grader-id-column",
        metavar="COLUMN",
        help="Read grader IDs from a different column than the prediction ID.",
    )
    parser.add_argument(
        "--grader-output-key-column",
        metavar="COLUMN",
        help="Read grader output keys from a different column than the prediction key.",
    )
    parser.add_argument(
        "--allow-extra-grades",
        action="store_true",
        help="Read full grader exports and keep only the selected IDs and output keys.",
    )


def monitor_options(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("predictions", nargs="?", type=Path, metavar="CSV_OR_JSONL")
    source.add_argument(
        "--batch-decisions",
        type=Path,
        metavar="FILE",
        help="Read a caat route decision file.",
    )
    parser.add_argument(
        "--id-column", help="ID column in predictions and grader files."
    )
    parser.add_argument("--score-column")
    labels = parser.add_mutually_exclusive_group()
    labels.add_argument("--correct-column")
    labels.add_argument(
        "--grader-file",
        type=named_file,
        metavar="NAME=FILE",
        help="Read arriving labels separately.",
    )
    parser.add_argument(
        "--output-key-column", help="Column identifying the saved output."
    )
    grader_export_options(parser)
    prediction_filter_option(parser)
    review_missing_scores_option(parser)
    threshold_options(parser)
    parser.add_argument("--risk-target", type=float)
    parser.add_argument(
        "--failure-rate", type=float, help="Sequential alert level. Defaults to 0.05."
    )
    parser.add_argument(
        "--coverage-window",
        type=int,
        metavar="ROWS",
        help="Check each window of this many stream rows.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        metavar="FRACTION",
        help="Minimum accepted fraction in each full coverage window.",
    )
    parser.add_argument("--json", action="store_true", help="Print the JSON report.")
    parser.add_argument(
        "--output", type=Path, metavar="FILE", help="Save the JSON report."
    )


def run_monitor(args):
    routed = args.batch_decisions is not None
    coverage_only = args.correct_column is None and args.grader_file is None
    if coverage_only:
        if args.coverage_window is None or args.min_coverage is None:
            raise ValueError(
                "coverage monitoring without labels requires --coverage-window and --min-coverage"
            )
        if args.risk_target is not None or args.failure_rate is not None:
            raise ValueError(
                "--risk-target and --failure-rate require --correct-column or --grader-file"
            )
    path = args.batch_decisions if routed else args.predictions
    if routed:
        if args.correct_column is not None:
            raise ValueError("--batch-decisions requires --grader-file to monitor risk")
        if any(
            value is not None
            for value in (
                args.score_column,
                args.raw_score_column,
                args.output_key_column,
                args.prediction_filter,
            )
        ):
            raise ValueError(
                "--score-column, --raw-score-column, --output-key-column, and "
                "--prediction-filter apply to prediction tables"
            )
    saved = load_threshold_options(args)
    fit_outputs = saved.get("fit_outputs")
    if args.grader_file is None:
        if (
            args.grader_id_column is not None
            or args.grader_output_key_column is not None
            or args.allow_extra_grades
        ):
            raise ValueError(
                "grader column options and --allow-extra-grades require --grader-file"
            )
        grader, grade_path = args.correct_column, None
        columns = () if coverage_only else (grader,)
    else:
        grader, grade_path = args.grader_file
        columns = ()
    if not coverage_only and args.risk_target is None:
        raise ValueError("monitor with --threshold requires --risk-target")
    if args.output is not None:
        inputs = threshold_input_paths(args)
        if grade_path is not None:
            inputs.add(grade_path.resolve())
        if args.output.resolve() in inputs:
            raise ValueError("output would overwrite a monitoring input")
    if routed:
        id_column, score_column = "id", "score"
        grade_key = (
            args.grader_output_key_column
            or saved.get("output_key_column")
            or "output_key"
        )
        rows = [
            BatchDecision(*decision)
            for decision in iter_batch_decisions(
                path,
                threshold=args.threshold,
                require_output_key=not coverage_only or fit_outputs is not None,
            )
        ]
        key_column = "output_key" if rows and rows[0].output_key is not None else None
    else:
        if fit_outputs is not None and args.output_key_column is None:
            args.output_key_column = fit_outputs["key_column"]
        use_available_output_key_column(args, saved)
        id_column, score_column, key_column = (
            args.id_column,
            args.score_column,
            args.output_key_column,
        )
        if grade_path is not None:
            key_column = key_column or "output_sha256"
        grade_key = args.grader_output_key_column or key_column
        rows, _ = read_table_rows(
            path,
            id_column=id_column,
            score_column=args.negated_score_source or score_column,
            correct_columns=columns,
            output_key_column=key_column,
            selection=args.prediction_filter,
            require_labels=bool(columns),
            negate_score=args.negated_score_source is not None,
            review_missing_scores=bool(args.review_missing_scores),
            scorer=args.scorer,
        )
    if fit_outputs is not None:
        check_fit_overlap(rows, fit_outputs, cohort="monitoring stream")
    labels = None
    if grade_path is not None:
        grade_id = args.grader_id_column or args.id_column
        labels = read_grader_file(
            grade_path,
            grader,
            {row.id: row.output_key for row in rows},
            id_column=grade_id,
            output_key_column=grade_key,
            require_all=False,
            allow_extra=args.allow_extra_grades,
        )
    contents = monitor_risk(
        rows,
        grader=grader,
        threshold=args.threshold,
        risk_target=args.risk_target,
        failure_rate=0.05 if args.failure_rate is None else args.failure_rate,
        labels=labels,
        routed=routed,
        review_missing_scores=bool(args.review_missing_scores),
        coverage_window=args.coverage_window,
        min_coverage=args.min_coverage,
    )
    metadata = {
        "path": str(path),
        "id_column": id_column,
        "score_column": score_column,
        "correct_columns": list(columns),
        "group_column": None,
        "output_key_column": key_column,
    }
    if routed:
        metadata["batch_decisions"] = str(path)
    if grade_path is not None:
        metadata["grader_files"] = {grader: str(grade_path)}
        if args.grader_id_column is not None or grade_id != id_column:
            metadata["grader_id_column"] = grade_id
        if args.grader_output_key_column is not None or (
            routed and grade_key != key_column
        ):
            metadata["grader_output_key_column"] = grade_key
        if args.allow_extra_grades:
            metadata["allow_extra_grades"] = True
    if args.prediction_filter is not None:
        column, value = args.prediction_filter
        metadata["prediction_filter"] = {"column": column, "value": value}
    if not routed and args.negated_score_source is not None:
        metadata["negated_score_source"] = args.negated_score_source
    if fit_outputs is not None:
        metadata["fit_outputs_checked"] = True
    if args.review_missing_scores:
        metadata["review_missing_scores"] = True
    if args.threshold_report is not None:
        metadata["threshold_report"] = str(args.threshold_report)
    report = make_report("monitor", metadata, contents)
    emit(report, json_output=args.json, output=args.output)
    coverage_alert = contents.get("coverage_check", {}).get("first_alert")
    return int(contents.get("first_alert") is not None or coverage_alert is not None)


def plan_options(parser):
    parser.add_argument(
        "--interim-labels", type=int, help="First checkpoint of a two-stage audit."
    )
    parser.add_argument(
        "--error-scenarios",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Assumed population error rates below and above the target.",
    )
    parser.add_argument(
        "--request-cost", type=float, help="Cost of one dispatch in labels (default 1)."
    )
    parser.add_argument(
        "--block-size",
        type=int,
        help="Forecast labels and requests for one grader per block of this size.",
    )
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Preview fixed-budget verdict probabilities without drawing a sample.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--risk-target", type=float, help="Find the minimum labels.")
    target.add_argument("--accepted", type=int, help="Find the minimum risk target.")
    target.add_argument(
        "--threshold-report",
        type=Path,
        metavar="JSON",
        help="Find the minimum labels for the risk target in a select, policy, or run report.",
    )
    parser.add_argument(
        "--expected-error",
        type=float,
        help="Assumed accepted error rate. Defaults to the held-out rate in --threshold-report.",
    )
    parser.add_argument(
        "--holdout-confidence",
        type=float,
        metavar="LEVEL",
        help="Confidence level. Uses the threshold report's level or 0.95 by default.",
    )
    accepted_population_options(parser)
    prediction_filter_option(parser)
    parser.add_argument(
        "--label-budget",
        type=int,
        metavar="N",
        help="Sample up to N eligible rows at the saved risk target.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        metavar="CSV",
        help="Write the uniform label sample for the fixed batch.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        metavar="SEED",
        help="Record the sample seed for a fixed batch (default 42 when writing a sample).",
    )
    parser.add_argument(
        "--sample-source",
        type=Path,
        metavar="CSV_OR_JSONL",
        help="Copy fields from the original outputs into the label sample.",
    )
    parser.add_argument(
        "--sample-column",
        action="append",
        metavar="COLUMN",
        help="Repeat for each source column to include in the sample.",
    )
    parser.add_argument("--id-column", help="Sample source ID column (default id).")
    parser.add_argument(
        "--output-key-column", help="Sample source output key (default output_sha256)."
    )
    parser.add_argument(
        "--power",
        type=float,
        help="Required chance of certification when sizing a plan (default 0.8).",
    )
    parser.add_argument(
        "--max-accepted",
        type=int,
        help=f"Label search limit when sizing a plan (default {DEFAULT_MAX_ACCEPTED}).",
    )
    parser.add_argument("--json", action="store_true", help="Print the JSON report.")
    parser.add_argument(
        "--output", type=Path, metavar="FILE", help="Write the JSON report."
    )


def run_plan(args):
    two_stage = args.interim_labels is not None
    if two_stage:
        if not args.batch_predictions or args.error_scenarios is None:
            raise ValueError(
                "--interim-labels requires --batch-predictions and --error-scenarios"
            )
        if args.expected_error is not None or args.block_size is not None:
            raise ValueError(
                "two-stage forecasts use --error-scenarios and checkpoint-sized requests"
            )
    elif args.error_scenarios is not None or args.request_cost is not None:
        raise ValueError(
            "--error-scenarios and --request-cost require --interim-labels"
        )
    if args.block_size is not None and not args.forecast:
        raise ValueError("--block-size requires --forecast")
    seed = args.sample_seed
    if args.sample_output is not None and seed is None:
        seed = 42
    fixed = args.batch_predictions is not None
    if args.forecast:
        if not fixed or (not two_stage and args.expected_error is None):
            raise ValueError(
                "--forecast requires --batch-predictions and --expected-error"
            )
        if args.sample_output is not None or seed is not None:
            raise ValueError(
                "--forecast does not draw a sample; omit --sample-output and --sample-seed"
            )
    if fixed:
        if args.threshold_report is None or args.label_budget is None:
            raise ValueError(
                "--batch-predictions requires --threshold-report and --label-budget"
            )
        if not args.forecast and args.sample_output is None:
            raise ValueError(
                "provide --sample-output to draw a sample or --forecast to preview the budget"
            )
        if args.label_budget < 1:
            raise ValueError("--label-budget must be positive")
        if (args.power is not None and not two_stage) or args.max_accepted is not None:
            raise ValueError("--power and --max-accepted apply to label sizing")
        if args.expected_error is not None and not args.forecast:
            raise ValueError(
                "--expected-error requires --forecast for a fixed-budget batch"
            )
    elif args.label_budget is not None or args.prediction_filter is not None:
        raise ValueError(
            "--label-budget and --prediction-filter require --batch-predictions"
        )
    elif seed is not None and args.batch_decisions is None:
        raise ValueError("--sample-output and --sample-seed require --batch-decisions")
    sample_columns = tuple(args.sample_column or ())
    if args.sample_source is not None:
        if args.sample_output is None or not sample_columns:
            raise ValueError(
                "--sample-source requires --sample-output and --sample-column"
            )
    elif (
        sample_columns
        or args.id_column is not None
        or args.output_key_column is not None
    ):
        raise ValueError(
            "--sample-column, --id-column, and --output-key-column require --sample-source"
        )
    destinations = [
        path.resolve() for path in (args.output, args.sample_output) if path is not None
    ]
    if len(destinations) != len(set(destinations)):
        raise ValueError("plan and sample output paths must be distinct")
    for path, name in (
        (args.batch_decisions, "batch decisions"),
        (args.batch_predictions, "batch predictions"),
        (args.threshold_report, "threshold report"),
        (args.sample_source, "sample source"),
    ):
        if path is not None and path.resolve() in destinations:
            raise ValueError(f"output would overwrite the {name}")
    saved = (
        read_threshold_report(args.threshold_report, fingerprint=fixed)
        if args.threshold_report is not None
        else {}
    )
    confidence = args.holdout_confidence
    if confidence is None:
        confidence = saved.get("holdout_confidence") or 0.95
    if fixed:
        if not 0.5 < confidence < 1:
            raise ValueError(
                "fixed-budget directional checks require confidence above 0.5 and below 1"
            )
        accepted_rows, total, digest, key = read_candidate_batch(
            args.batch_predictions, saved, selection=args.prediction_filter
        )
        population = len(accepted_rows)
        result = {
            "total": total,
            "accepted_population": population,
            "review": total - population,
            "accepted": min(args.label_budget, population),
            "risk_target": saved["risk_target"],
        }
        if population:
            result.update(
                sample_error_limits(
                    population, result["accepted"], saved["risk_target"], confidence
                )
            )
        options = {
            "threshold_report_sha256": saved["threshold_report_sha256"],
            "batch_predictions": str(args.batch_predictions),
            "label_budget": args.label_budget,
            "accepted_population_sha256": digest,
            "output_key_column": key,
            "fit_outputs_checked": saved["fit_outputs"] is not None,
            "holdout_confidence": confidence,
        }
        if args.prediction_filter is not None:
            column, value = args.prediction_filter
            options["prediction_filter"] = {"column": column, "value": value}
        if two_stage:
            options.update(
                forecast=args.forecast,
                interim_labels=args.interim_labels,
                error_scenarios=args.error_scenarios,
                power=0.8 if args.power is None else args.power,
                request_cost=1 if args.request_cost is None else args.request_cost,
            )
            result.update(
                two_stage_design(
                    population,
                    result["accepted"],
                    args.interim_labels,
                    saved["risk_target"],
                    confidence,
                    args.error_scenarios,
                    options["power"],
                    options["request_cost"],
                )
            )
            if not args.forecast and not result["qualifies"]:
                raise ValueError(
                    "insufficient power for this design; use --forecast to compare budgets"
                )
        elif args.forecast:
            options.update(forecast=True, expected_error=args.expected_error)
            result.update(
                forecast_verdict(
                    population, result["accepted"], args.expected_error, result
                )
            )
            if args.block_size is not None:
                options["block_size"] = args.block_size
                result["block_cost"] = forecast_block_cost(
                    population,
                    result["expected_population_errors"],
                    result["accepted"],
                    result,
                    args.block_size,
                )
    else:
        result, options, accepted_rows = size_label_plan(args, saved, confidence)
    if args.threshold_report is not None:
        options.update(
            threshold_report=str(args.threshold_report), threshold=saved["threshold"]
        )
    budget = result["accepted"]
    sample_path = args.sample_output if budget is not None else None
    if budget is not None and seed is not None:
        options["sample_seed"] = seed
    extra_values = None
    if sample_path is not None:
        result["sample_path"] = str(sample_path)
        sample = shuffled_by_id(accepted_rows, seed, key=attrgetter("id"))[:budget]
        if args.sample_source is not None:
            extra_values, source_id, source_key = read_sample_columns(
                args.sample_source,
                sample,
                sample_columns,
                id_column=args.id_column,
                output_key_column=args.output_key_column,
                fallback_id_column=saved.get("id_column"),
                fallback_output_key_column=saved.get("output_key_column"),
            )
            options.update(
                sample_source=str(args.sample_source),
                sample_columns=sample_columns,
                sample_source_id_column=source_id,
                sample_source_output_key_column=source_key,
            )
    report = make_report("two-stage-plan" if two_stage else "plan", options, result)
    serialized = render_report("json", report)
    paths = [path for path in (args.output, sample_path) if path is not None]
    with staged_outputs(paths) as staged:
        if args.output is not None:
            write_report(staged[args.output], serialized)
        if sample_path is not None:
            write_queue(
                staged[sample_path],
                sample,
                include_output_key=options["output_key_column"] is not None
                if fixed
                else sample[0].output_key is not None,
                extra_columns=sample_columns,
                extra_values=extra_values,
            )
    print(serialized if args.json else render_report("text", report))
    return int(not budget or not result.get("qualifies", True))


def size_label_plan(args, saved, confidence):
    expected = args.expected_error
    if expected is None:
        expected = saved.get("heldout_error_rate")
        if expected is None:
            raise ValueError(
                "give --expected-error, or a policy or run report whose held-out "
                "check accepted rows"
            )
    options = {
        "expected_error": expected,
        "holdout_confidence": confidence,
        "power": 0.8 if args.power is None else args.power,
    }
    population = args.accepted_population
    digest = None
    accepted_rows = [] if args.sample_output is not None else None
    if args.batch_decisions is not None:
        population, digest, _ = read_batch_decisions(
            args.batch_decisions,
            threshold=saved.get("threshold"),
            fingerprint=True,
            accepted_rows=accepted_rows,
        )
    if population is not None:
        options["accepted_population"] = population
    if args.accepted is not None:
        options["accepted"] = args.accepted
    else:
        options.update(
            risk_target=saved.get("risk_target", args.risk_target),
            max_accepted=DEFAULT_MAX_ACCEPTED
            if args.max_accepted is None
            else args.max_accepted,
        )
    result = plan_labels(**options)
    if args.expected_error is None:
        options["expected_error_source"] = "threshold_report"
    if args.batch_decisions is not None:
        options.update(
            batch_decisions=str(args.batch_decisions), accepted_population_sha256=digest
        )
    return result, options, accepted_rows


def grader_source_options(parser, *, multiple=True, labels=False):
    """Add grade inputs. With labels, an exact prediction-reference match also works."""
    grades = parser.add_mutually_exclusive_group(required=not labels)
    action = "append" if multiple else "store"
    if labels:
        parser.add_argument(
            "--prediction-column",
            help="Grade by exact match with the reference column. Defaults to 'prediction' without 0/1 grades.",
        )
        parser.add_argument(
            "--reference-column", help="Defaults to 'reference' for exact matching."
        )
    grades.add_argument(
        "--correct-column",
        action=action,
        help=(
            "Repeat for each grader in the same prediction file."
            if multiple
            else "Read correctness from a column in the prediction file."
        ),
    )
    grades.add_argument(
        "--grader-file",
        action=action,
        type=named_file,
        metavar="NAME=FILE",
        help=(
            "Repeat for separate grader exports. Default output key is 'output_sha256'."
            if multiple
            else "Read a separate grader export. Default output key is 'output_sha256'."
        ),
    )
    grader_export_options(parser)


def audit_options(parser):
    grader_source_options(parser)
    threshold_options(parser)
    parser.add_argument("--risk-target", type=float)
    holdout_confidence_option(parser)
    accepted_population_options(parser, check_sample=True)
    prediction_filter_option(parser)
    parser.add_argument(
        "--bootstrap",
        type=int,
        metavar="N",
        help="Paired bootstrap replicates for 95%% intervals.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Bootstrap seed. Defaults to 42."
    )
    parser.add_argument(
        "--disagreements-output",
        type=Path,
        metavar="CSV",
        help="Write rows whose graders disagree, with scores and cutoff decisions.",
    )


def select_options(parser):
    selection_options(parser)
    grid = parser.add_mutually_exclusive_group(required=True)
    grid.add_argument(
        "--thresholds",
        action="extend",
        nargs="+",
        type=float,
        help="Fixed candidate cutoffs in input score units. Use --thresholds=-1e-5 for negative scientific notation.",
    )
    grid.add_argument(
        "--design", type=Path, help="Separate CSV or JSONL scores for a quantile grid."
    )


def policy_options(parser):
    selection_options(parser)
    holdout_confidence_option(parser)
    parser.add_argument(
        "--seed", type=int, default=42, help="Split seed. Defaults to 42."
    )
    parser.add_argument(
        "--split-checks",
        type=int,
        default=20,
        metavar="N",
        help="Refit on the next N seeds and report how the accepted count moves. "
        "Defaults to 20. Use 0 to skip.",
    )
    parser.add_argument(
        "--check-column", action="append", help="Repeat for each held-out-only grader."
    )
    parser.add_argument(
        "--holdout-output", type=Path, metavar="CSV", help="Export the held-out rows."
    )
    parser.add_argument(
        "--calibrate",
        choices=CALIBRATORS,
        metavar="METHOD",
        help=f"Also fit a probability map: {', '.join(CALIBRATORS)}.",
    )
    parser.add_argument("--calibration-input", choices=("score", "probability"))


def calibrate_options(parser):
    operations = parser.add_subparsers(dest="operation", required=True)
    fit = operations.add_parser("fit", help="Fit a map using labeled calibration rows.")
    apply = operations.add_parser("apply", help="Apply a saved map and write a CSV.")
    for command in (fit, apply):
        command.add_argument("predictions", type=Path, metavar="CSV_OR_JSONL")
        command.add_argument("--score-column")
        command.add_argument("--output", type=Path, required=True)
        prediction_filter_option(command)
    fit.set_defaults(score_column="score")
    fit.add_argument("--method", required=True, choices=CALIBRATORS)
    fit.add_argument("--input-scale", choices=("score", "probability"))
    score_direction_option(fit)
    fit.add_argument("--id-column", default="id")
    fit.add_argument(
        "--output-key-column",
        help="Saved output identity column. Defaults to output_sha256 for grader joins and tracked fits.",
    )
    fit.add_argument(
        "--track-fit-outputs",
        action="store_true",
        help="Save identities of scored fitting rows for a holdout check.",
    )
    grader_source_options(fit, multiple=False)
    fit.add_argument("--bins", type=int, default=10, help="Number of histogram bins.")
    review_missing_scores_option(
        fit,
        help_text="Fit on scored rows and preserve the missing-score rule in the map.",
    )
    raw_score_column_option(apply)
    apply.add_argument(
        "--map",
        type=Path,
        required=True,
        help="Map JSON or a policy or run JSON report.",
    )
    apply.add_argument("--grader", help="Choose a grader's map from a report.")
    apply.add_argument("--probability-column", default="calibrated_probability")
    apply.add_argument(
        "--check-holdout",
        action="store_true",
        help="Reject rows that reuse a fitted output tracked in the map or report.",
    )
    apply.add_argument(
        "--id-column", help="Override the saved ID column for --check-holdout."
    )
    apply.add_argument(
        "--output-key-column",
        help="Override the saved output key column for --check-holdout.",
    )
    review_missing_scores_option(
        apply, help_text="Leave probabilities blank for missing scores."
    )


def run_calibrate(args):
    applying = args.operation == "apply"
    inputs = {args.predictions.resolve()}
    if applying:
        inputs.add(args.map.resolve())
    elif args.grader_file is not None:
        inputs.add(args.grader_file[1].resolve())
    if args.output.resolve() in inputs:
        raise ValueError("output would overwrite a calibration input")
    if applying:
        if not args.check_holdout and (
            args.id_column is not None or args.output_key_column is not None
        ):
            raise ValueError(
                "--id-column and --output-key-column require --check-holdout"
            )
        override = score_column_override(args)
        mapping, saved_score, saved_source, saved_review, saved_input = (
            read_saved_calibration(args.map, grader=args.grader)
        )
        fit_outputs = None
        id_column = "id"
        if args.check_holdout:
            fit_outputs = read_fit_outputs(
                args.map, saved_input, artifact="calibration map or report"
            )
            if fit_outputs is None:
                raise ValueError(
                    "--check-holdout requires fit output identities in --map"
                )
            id_column = (
                args.id_column
                if args.id_column is not None
                else saved_input.get("id_column")
            )
            if not isinstance(id_column, str) or not id_column.strip():
                raise ValueError("--check-holdout needs an ID column name")
            if args.output_key_column is not None:
                fit_outputs = {**fit_outputs, "key_column": args.output_key_column}
        source = saved_source if args.score_column is None else None
        score_column = (
            override if override is not None else source or saved_score or "score"
        )
        review_missing_scores = args.review_missing_scores or saved_review
        count = apply_csv(
            args.predictions,
            args.output,
            mapping,
            score_column,
            args.probability_column,
            negate_score=source is not None,
            review_missing_scores=review_missing_scores,
            selection=args.prediction_filter,
            fit_outputs=fit_outputs,
            id_column=id_column,
        )
        units = "rows" if review_missing_scores else "calibrated probabilities"
        print(f"Wrote {count} {units} to {args.output}.")
        return 0
    return run_calibrate_fit(args)


def run_calibrate_fit(args):
    negate_score = args.direction == "lower"
    if negate_score and (args.input_scale or default_scale(args.method)) != "score":
        raise ValueError("--direction lower requires score-scale calibration")
    review_missing_scores = bool(args.review_missing_scores)
    key = args.output_key_column
    if key is None and (args.track_fit_outputs or args.grader_file is not None):
        key = "output_sha256"
    table_options = {
        "id_column": args.id_column,
        "score_column": args.score_column,
        "output_key_column": key,
        "negate_score": negate_score,
        "review_missing_scores": review_missing_scores,
        "selection": args.prediction_filter,
    }
    map_input = {}
    if args.grader_file is None:
        if args.allow_extra_grades or any(
            column is not None
            for column in (
                args.grader_id_column,
                args.grader_output_key_column,
            )
        ):
            raise ValueError("grader join options require --grader-file")
        columns = (args.correct_column,)
        rows, _ = read_table_rows(
            args.predictions, correct_columns=columns, **table_options
        )
    else:
        grader, grade_path = args.grader_file
        columns = (grader,)
        rows, _ = join_grader_files(
            args.predictions,
            (args.grader_file,),
            grader_id_column=args.grader_id_column,
            grader_output_key_column=args.grader_output_key_column,
            allow_extra_grades=args.allow_extra_grades,
            **table_options,
        )
        map_input.update(
            path=str(args.predictions),
            id_column=args.id_column,
            output_key_column=key,
            grader_files={grader: str(grade_path)},
        )
        for name in ("grader_id_column", "grader_output_key_column"):
            if (value := getattr(args, name)) is not None:
                map_input[name] = value
        if args.allow_extra_grades:
            map_input["allow_extra_grades"] = True
    rows = validated_rows(rows, columns, review_missing_scores=review_missing_scores)
    total = len(rows)
    if review_missing_scores:
        rows = [row for row in rows if row.score is not None]
        if not rows:
            raise ValueError("calibration requires at least one scored row")
    mapping = fit_calibration(
        (row.score for row in rows),
        (row.outcomes[0] for row in rows),
        method=args.method,
        input_scale=args.input_scale,
        bins=args.bins,
    )
    if args.track_fit_outputs:
        map_input.update(
            id_column=args.id_column, fit_outputs=fit_outputs_metadata(key, rows)
        )
    if negate_score:
        map_input["negated_score_source"] = args.score_column
    if review_missing_scores:
        map_input["review_missing_scores"] = True
    if args.prediction_filter is not None:
        column, value = args.prediction_filter
        map_input["prediction_filter"] = {"column": column, "value": value}
    if map_input:
        score_column = (
            negated_score_name(args.score_column) if negate_score else args.score_column
        )
        mapping["input"] = {"score_column": score_column, **map_input}
        mapping["calibration_total"] = total
        mapping["scored"] = len(rows)
    serialized = json.dumps(mapping, indent=2, allow_nan=False)
    with staged_outputs((args.output,)) as staged:
        write_report(staged[args.output], serialized)
    count = f"{len(rows)} of {total}" if review_missing_scores else str(len(rows))
    print(f"Fitted {args.method} on {count} calibration rows. Wrote {args.output}.")
    return 0


def group_quota(value):
    group, separator, limit = value.partition("=")
    if not group or not separator or not limit.isdecimal():
        raise argparse.ArgumentTypeError("use GROUP=COUNT for --quota")
    return group, int(limit)


def queue_options(parser):
    parser.add_argument("predictions", type=Path, metavar="CSV_OR_JSONL")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--group-column")
    parser.add_argument(
        "--output-key-column", help="Column identifying the saved output."
    )
    prediction_filter_option(parser)
    review_missing_scores_option(
        parser, help_text="Place rows with empty scores at the front of the queue."
    )
    parser.add_argument(
        "--correct-column", action="append", help="Optional grades for replay."
    )
    score_direction_option(parser)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument(
        "--quota", action="append", type=group_quota, metavar="GROUP=COUNT"
    )
    parser.add_argument(
        "--output", type=Path, metavar="CSV", help="Write the score-ordered queue."
    )
    parser.add_argument(
        "--random-output",
        type=Path,
        metavar="CSV",
        help="Write the seeded random sample.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")


def run_queue(args):
    if args.output is None and args.random_output is None:
        raise ValueError("give --output, --random-output, or both")
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
    rows, group = read_table_rows(
        args.predictions,
        id_column=args.id_column,
        score_column=args.score_column,
        group_column=args.group_column,
        correct_columns=graders,
        output_key_column=args.output_key_column,
        selection=args.prediction_filter,
        require_labels=False,
        review_missing_scores=bool(args.review_missing_scores),
    )
    chosen, random_rows, result = review_queue(
        rows,
        budget=args.budget,
        direction=args.direction,
        quotas=quotas,
        graders=graders,
        seed=args.seed,
        review_missing_scores=bool(args.review_missing_scores),
    )
    keep_output_key = args.output_key_column is not None
    outputs = {
        path: rows
        for path, rows in ((args.output, chosen), (args.random_output, random_rows))
        if path is not None
    }
    with staged_outputs(outputs) as staged:
        for path, rows in outputs.items():
            write_queue(staged[path], rows, include_output_key=keep_output_key)
    meta = {
        "path": str(args.predictions),
        "id_column": args.id_column,
        "score_column": args.score_column,
        "group_column": group,
        "correct_columns": list(graders),
        "output_key_column": args.output_key_column,
    }
    if args.prediction_filter is not None:
        column, value = args.prediction_filter
        meta["prediction_filter"] = {"column": column, "value": value}
    if args.review_missing_scores:
        meta["review_missing_scores"] = True
    if args.output is None:
        result["sample_path"] = str(args.random_output)
    else:
        meta["output"] = str(args.output)
    emit(make_report("queue", meta, result), json_output=args.json)
    return 0


def route_options(parser):
    parser.add_argument("predictions", type=Path, metavar="CSV_OR_JSONL")
    parser.add_argument("--id-column")
    parser.add_argument("--score-column")
    parser.add_argument("--group-column")
    parser.add_argument(
        "--output-key-column", help="Column identifying the saved output."
    )
    threshold_options(parser)
    prediction_filter_option(parser)
    review_missing_scores_option(parser)
    parser.add_argument("--output", type=Path, metavar="CSV", required=True)
    parser.add_argument(
        "--keep-column",
        action="append",
        metavar="COLUMN",
        help="Copy this input column to the decisions file. Repeat for more.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the routing summary."
    )


def run_route(args):
    saved = load_threshold_options(args, require_confirmed=True)
    if args.output.resolve() in threshold_input_paths(args):
        raise ValueError("output would overwrite an input file")
    overall, group, key_column = route_batch(
        args.predictions,
        args.output,
        threshold=args.threshold,
        id_column=args.id_column,
        score_column=args.negated_score_source or args.score_column,
        group_column=args.group_column,
        output_key_column=args.output_key_column,
        fallback_output_key_column=saved.get("output_key_column"),
        selection=args.prediction_filter,
        negate_score=args.negated_score_source is not None,
        review_missing_scores=args.review_missing_scores,
        keep_columns=args.keep_column or (),
        scorer=args.scorer,
    )
    meta = {
        "path": str(args.predictions),
        "id_column": args.id_column,
        "score_column": args.score_column,
        "group_column": group,
        "output_key_column": key_column,
    }
    if args.keep_column:
        meta["keep_columns"] = list(args.keep_column)
    if args.negated_score_source is not None:
        meta["negated_score_source"] = args.negated_score_source
    if args.review_missing_scores:
        meta["review_missing_scores"] = True
    if args.threshold_report is not None:
        meta["threshold_report"] = str(args.threshold_report)
    if args.prediction_filter is not None:
        column, value = args.prediction_filter
        meta["prediction_filter"] = {"column": column, "value": value}
    result = {
        "threshold": args.threshold,
        "overall": overall,
        "decision_path": str(args.output),
    }
    emit(make_report("route", meta, result), json_output=args.json)
    return 0


def collect_options(parser):
    parser.add_argument("prompts", type=Path, metavar="JSONL")
    parser.add_argument("--output", type=Path, metavar="JSONL", required=True)
    parser.add_argument("--metadata", type=Path, metavar="JSON")
    parser.add_argument("--model", help="Local model or the ID served by --base-url.")
    parser.add_argument(
        "--base-url", help="Chat Completions API root (e.g. http://localhost:8000/v1)."
    )
    parser.add_argument(
        "--api-key-env",
        metavar="NAME",
        help="Environment variable holding the API bearer token.",
    )
    parser.add_argument(
        "--no-logprobs",
        action="store_true",
        help="Collect text-only signals from the API.",
    )
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Maximum prompts or samples per model call. Defaults to 1.",
    )
    parser.add_argument(
        "--serial-greedy",
        action="store_true",
        help="Collect primary greedy answers one prompt at a time.",
    )
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
    if args.batch_size < 1:
        raise ValueError("collect needs a positive batch size")
    prompts = read_prompts(args.prompts)
    use_api = args.base_url is not None
    start = perf_counter()
    try:
        if use_api:
            if not args.model or not args.model.strip():
                raise ValueError("--base-url requires an explicit --model")
            if args.batch_size != 1:
                raise ValueError("--base-url requires --batch-size 1")
            if (
                args.revision
                or args.local_files_only
                or args.gpu_memory
                or args.device != "auto"
            ):
                raise ValueError("local model options require the Transformers backend")
            api_key = None
            if args.api_key_env is not None:
                api_key = environ.get(args.api_key_env)
                if not api_key:
                    raise ValueError(
                        f"{args.api_key_env} needs to contain an API bearer token"
                    )
            backend = ChatCompletionsBackend(
                args.base_url,
                args.model,
                seed=args.seed,
                api_key=api_key,
                logprobs=not args.no_logprobs,
            )
        else:
            if args.api_key_env is not None or args.no_logprobs:
                raise ValueError("--api-key-env and --no-logprobs require --base-url")
            model = args.model or DEFAULT_MODEL
            revision = args.revision or (
                DEFAULT_REVISION if model == DEFAULT_MODEL else None
            )
            backend = TransformersBackend(
                model,
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
            batch_size=args.batch_size,
            serial_greedy=args.serial_greedy,
        )
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"collection failed: {exc}") from exc
    write_responses(args.output, records)
    if args.metadata is not None:
        meta = {
            **backend.metadata,
            "prompt_count": len(prompts),
            "max_new_tokens": args.max_new_tokens,
            "samples_per_prompt": args.samples,
            "batch_size": args.batch_size,
            "serial_greedy": args.serial_greedy,
            "model_load_seconds": load_seconds,
            "methods": timings,
        }
        if use_api:
            meta["probability_rows"] = {
                field: sum(field in record for record in records)
                for field in ("tokens", "p_true")
            }
        write_report(args.metadata, json.dumps(meta, indent=2, allow_nan=False))
    print(f"Collected {len(records)} responses to {args.output}.")
    return 0


def score_cost(value):
    column, separator, cost = value.partition("=")
    if not column or not separator or cost not in COSTS:
        raise argparse.ArgumentTypeError(
            f"use COLUMN=CLASS with one of {', '.join(COSTS)}"
        )
    return column, cost


def scores_options(parser):
    parser.add_argument("predictions", type=Path, metavar="CSV_OR_JSONL")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--score-column", action="append", required=True)
    lower_is_better_option(parser)
    parser.add_argument(
        "--output-key-column", help="Column identifying the saved output."
    )
    grader_source_options(parser, labels=True)
    prediction_filter_option(parser)
    review_missing_scores_option(parser)
    parser.add_argument("--baseline", help="Default is the first score column.")
    parser.add_argument("--length-column")
    parser.add_argument(
        "--class-column",
        help="Report review volume by predicted class at each global coverage.",
    )
    parser.add_argument(
        "--score-cost", action="append", type=score_cost, metavar="COLUMN=CLASS"
    )
    parser.add_argument(
        "--coverage",
        action="append",
        type=float,
        help="Coverage fraction. Repeat for multiple budgets.",
    )
    parser.add_argument("--split", choices=("heldout", "all"), default="heldout")
    parser.add_argument("--bootstrap", type=int, default=500, metavar="N")
    parser.add_argument(
        "--workers",
        type=int,
        metavar="N",
        help="Bootstrap processes. Large CLI runs use up to eight by default.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, metavar="JSON")


def run_scores(args, *, auto_workers=False):
    files = args.grader_file or ()
    if args.output is not None:
        destination = args.output.resolve()
        if destination == args.predictions.resolve():
            raise ValueError("output would overwrite the score input")
        if any(destination == path.resolve() for _, path in files):
            raise ValueError("output would overwrite a grader input")
    if not files and (
        args.grader_id_column is not None
        or args.grader_output_key_column is not None
        or args.allow_extra_grades
    ):
        raise ValueError(
            "grader column options and --allow-extra-grades require --grader-file"
        )
    key = args.output_key_column or ("output_sha256" if files else None)
    labels = None
    if not files and not args.correct_column:
        labels = (
            args.prediction_column or "prediction",
            args.reference_column or "reference",
        )
    elif args.prediction_column is not None:
        raise ValueError("--prediction-column applies to exact-match grading")
    graders = [name for name, _ in files] if files else args.correct_column
    if labels:
        graders = [labels[1]]
    costs = dict(args.score_cost or ())
    if len(costs) != len(args.score_cost or ()):
        raise ValueError("give at most one cost for each score column")
    lower = args.lower_is_better or ()
    if unknown := set(lower).difference(args.score_column):
        raise ValueError(
            f"--lower-is-better columns must appear in --score-column: {', '.join(sorted(unknown))}"
        )
    rows = read_score_rows(
        args.predictions,
        args.score_column,
        args.correct_column or (),
        id_column=args.id_column,
        length_column=args.length_column,
        class_column=args.class_column,
        output_key_column=key,
        selection=args.prediction_filter,
        negated_scores={name: name for name in lower},
        review_missing_scores=args.review_missing_scores,
        label_columns=labels,
    )
    if files:
        output_keys = {row.id: row.output_key for row in rows}
        if not output_keys or len(output_keys) != len(rows) or "" in output_keys:
            raise ValueError("score rows must have unique nonempty IDs")
        outcomes = [
            read_grader_file(
                path,
                name,
                output_keys,
                id_column=args.grader_id_column or args.id_column,
                output_key_column=args.grader_output_key_column or key,
                allow_extra=args.allow_extra_grades,
            )
            for name, path in files
        ]
        for index, row in enumerate(rows):
            rows[index] = replace(
                row, outcomes=tuple(bits[row.id] for bits in outcomes)
            )
    workers = args.workers
    if workers is None and not auto_workers:
        workers = 1
    result = compare_scores(
        rows,
        names=args.score_column,
        graders=graders,
        baseline=args.baseline,
        length_control=args.length_column is not None,
        costs=costs,
        coverages=args.coverage or DEFAULT_COVERAGES,
        bootstrap=args.bootstrap,
        seed=args.seed,
        split=args.split,
        workers=workers,
        review_missing_scores=args.review_missing_scores,
    )
    meta = {
        "path": str(args.predictions),
        "id_column": args.id_column,
        "score_columns": args.score_column,
        "length_column": args.length_column,
        "correct_columns": args.correct_column or [],
    }
    if args.class_column is not None:
        meta["class_column"] = args.class_column
    if labels:
        meta["prediction_column"], meta["reference_column"] = labels
    if key is not None:
        meta["output_key_column"] = key
    if files:
        meta["grader_files"] = {name: str(path) for name, path in files}
        for name in ("grader_id_column", "grader_output_key_column"):
            if (value := getattr(args, name)) is not None:
                meta[name] = value
        if args.allow_extra_grades:
            meta["allow_extra_grades"] = True
    if lower:
        meta["lower_is_better"] = list(lower)
    if args.review_missing_scores:
        meta["review_missing_scores"] = True
    if args.prediction_filter is not None:
        column, value = args.prediction_filter
        meta["prediction_filter"] = {"column": column, "value": value}
    emit(make_report("scores", meta, result), json_output=args.json, output=args.output)
    return 0


def load_inputs(args, *, source_rows=None):
    columns = tuple(args.correct_column or ())
    score_source = getattr(args, "negated_score_source", None)
    common = {
        "id_column": args.id_column,
        "score_column": score_source or args.score_column,
        "group_column": args.group_column,
    }
    meta = {
        "path": str(args.predictions),
        "id_column": args.id_column,
        "score_column": args.score_column,
        "correct_columns": list(columns),
        "output_key_column": args.output_key_column,
    }
    if getattr(args, "scorer", None) is not None:
        if args.adapter not in (None, "csv"):
            raise ValueError(
                "use --score-column with a precomputed score for a custom adapter"
            )
        common["scorer"] = args.scorer
    if args.review_missing_scores:
        common["review_missing_scores"] = True
        meta["review_missing_scores"] = True
    threshold_report = getattr(args, "threshold_report", None)
    if threshold_report is not None:
        meta["threshold_report"] = str(threshold_report)
    key = args.output_key_column
    selected = getattr(args, "prediction_filter", None)
    if selected is not None:
        meta["prediction_filter"] = dict(
            zip(("column", "value"), selected, strict=True)
        )
    files = getattr(args, "grader_file", None)
    checks = tuple(getattr(args, "check_column", None) or ())
    if args.command == "policy":
        meta["check_columns"] = list(checks)
    grader_id = getattr(args, "grader_id_column", None)
    grader_key = getattr(args, "grader_output_key_column", None)
    allow_extra = getattr(args, "allow_extra_grades", False)
    if (grader_id is not None or grader_key is not None or allow_extra) and not files:
        raise ValueError(
            "grader column options and --allow-extra-grades require --grader-file"
        )
    prediction = getattr(args, "prediction_column", None)
    reference = getattr(args, "reference_column", None)
    unnamed_labels = False
    if args.command == "evaluate":
        if len(columns) > 1:
            raise ValueError(
                "evaluate needs one correctness column; use audit to compare graders"
            )
        if len(files or ()) > 1:
            raise ValueError(
                "evaluate needs one grader file; use audit to compare graders"
            )
    if args.command in {"evaluate", "select", "policy"}:
        if files:
            if prediction is not None:
                raise ValueError(
                    "--prediction-column applies to exact-match evaluation"
                )
        elif columns:
            if prediction is not None and args.command != "evaluate":
                raise ValueError(
                    "use --correct-column for 0/1 grades or --prediction-column for label matching"
                )
        else:
            if checks:
                raise ValueError(
                    "--check-column needs --correct-column or --grader-file"
                )
            unnamed_labels = prediction is None and reference is None
            prediction, reference = prediction or "prediction", reference or "reference"
        if args.command == "evaluate" or not (files or columns):
            meta.update(prediction_column=prediction, reference_column=reference)
    if files:
        if args.adapter is not None:
            raise ValueError("separate grader files use the built-in table join")
        key = key or "output_sha256"
        rows, group = join_grader_files(
            args.predictions,
            files,
            **common,
            output_key_column=key,
            selection=selected,
            grader_id_column=grader_id,
            grader_output_key_column=grader_key,
            allow_extra_grades=allow_extra,
            check_columns=checks,
            reference_column=reference,
            source_rows=source_rows,
        )
        meta.update(
            output_key_column=key,
            grader_files={name: str(path) for name, path in files},
        )
        if grader_id is not None:
            meta["grader_id_column"] = grader_id
        if grader_key is not None:
            meta["grader_output_key_column"] = grader_key
        if allow_extra:
            meta["allow_extra_grades"] = True
    else:
        options = dict(
            **common,
            correct_columns=(*columns, *checks),
            output_key_column=key,
            prediction_column=prediction,
            reference_column=reference,
            selection=selected,
        )
        if source_rows is not None:
            if args.adapter not in (None, "csv"):
                raise ValueError("--holdout-output uses the built-in prediction reader")
            options["source_rows"] = source_rows
        if args.adapter is None:
            try:
                rows, group = read_table_rows(args.predictions, **options)
            except ValueError as exc:
                if unnamed_labels and " columns: " in str(exc):
                    raise ValueError(
                        f"{exc}. Name a 0/1 grade column with --correct-column, or label "
                        "columns with --prediction-column and --reference-column"
                    ) from exc
                raise
        else:
            rows = list(ADAPTERS[args.adapter].read(args.predictions, options))
            group = args.group_column or (
                "group" if any(row.group is not None for row in rows) else None
            )
            meta["adapter"] = args.adapter
    if score_source is not None:
        for index, row in enumerate(rows):
            if row.score is not None:
                rows[index] = replace(row, score=-row.score)
        meta["negated_score_source"] = score_source
    meta["group_column"] = group
    return rows, meta


def load_label_plan(args):
    if args.plan_report is None:
        if args.batch_predictions is not None:
            raise ValueError(
                "--batch-predictions requires a fixed-budget --plan-report"
            )
        return None
    plan = read_label_plan(
        args.plan_report, allow_empty=getattr(args, "partial", False)
    )
    if "threshold_report_sha256" in plan:
        if args.batch_predictions is None or args.threshold_report is None:
            raise ValueError(
                "a fixed-budget plan requires --batch-predictions and --threshold-report"
            )
    elif args.batch_predictions is not None:
        raise ValueError("--batch-predictions requires a fixed-budget plan")
    elif plan["sample_seed"] is not None and args.batch_decisions is None:
        raise ValueError("the saved label draw requires --batch-decisions")
    for name in ("risk_target", "holdout_confidence", "sample_seed"):
        if plan[name] is None:
            continue
        supplied = getattr(args, name)
        if supplied is None:
            setattr(args, name, plan[name])
        elif supplied != plan[name]:
            option = name.replace("_", "-")
            raise ValueError(
                f"--{option} does not match the label plan ({plan[name]!r})"
            )
    return plan


def run(args, *, staged, fit_outputs=None, label_plan=None, candidate_policy=None):
    if fit_outputs is not None and args.output_key_column is None:
        args.output_key_column = fit_outputs["key_column"]
    if args.command in {"select", "policy"} and args.direction == "lower":
        args.negated_score_source = args.score_column
        args.score_column = negated_score_name(args.score_column)
    grader_files = getattr(args, "grader_file", None) or ()
    review_missing_scores = bool(args.review_missing_scores)
    track_fit = getattr(args, "track_fit_outputs", False)
    if track_fit and not grader_files and not (args.output_key_column or "").strip():
        raise ValueError("--track-fit-outputs requires --output-key-column")
    requested_grid_size = getattr(args, "design_grid_size", None)
    if (
        requested_grid_size is not None
        and args.command == "select"
        and args.design is None
    ):
        raise ValueError("--design-grid-size requires --design")
    grid_size = (
        DEFAULT_DESIGN_GRID_SIZE if requested_grid_size is None else requested_grid_size
    )
    holdout_output = getattr(args, "holdout_output", None)
    source_rows = [] if holdout_output is not None else None
    batch = getattr(args, "batch_decisions", None)
    sample_seed = getattr(args, "sample_seed", None)
    if sample_seed is not None and batch is None and candidate_policy is None:
        raise ValueError("--sample-seed requires --batch-decisions")
    rows, meta = load_inputs(args, source_rows=source_rows)
    population = getattr(args, "accepted_population", None)
    if candidate_policy is not None:
        population = check_candidate_sample(
            args.batch_predictions, candidate_policy, label_plan, rows
        )
        meta.update(
            batch_predictions=str(args.batch_predictions),
            accepted_population_checked=True,
            threshold_report_sha256=candidate_policy["threshold_report_sha256"],
            sample_seed=sample_seed,
        )
    elif batch is not None:
        expected_digest = (
            label_plan["accepted_population_sha256"] if label_plan is not None else None
        )
        population, digest, negate_sample = read_batch_decisions(
            batch,
            threshold=args.threshold,
            sample=rows,
            sample_seed=sample_seed,
            fingerprint=expected_digest is not None,
            allow_negated_sample_scores=args.allow_negated_sample_scores,
        )
        if expected_digest is not None:
            if digest != expected_digest:
                raise ValueError("batch accepted outputs do not match the label plan")
            meta["accepted_population_checked"] = True
        if negate_sample:
            rows = [
                replace(row, score=-row.score) if row.score is not None else row
                for row in rows
            ]
            meta["score_column"] = meta.pop("negated_score_source")
        meta["batch_decisions"] = str(batch)
        if sample_seed is not None:
            meta["sample_seed"] = sample_seed
    elif population is None and label_plan is not None:
        population = label_plan["accepted_population"]
    if label_plan is not None:
        check_label_plan(
            label_plan,
            threshold=args.threshold,
            accepted=sum(accepted_at(row, args.threshold) for row in rows),
            accepted_population=population,
        )
        meta["plan_report"] = str(args.plan_report)
        meta["planned_labels"] = label_plan["accepted"]
    if requested_grid_size is not None:
        meta["design_grid_size"] = requested_grid_size
    if fit_outputs is not None:
        check_fit_overlap(rows, fit_outputs, cohort="holdout")
        meta["fit_outputs_checked"] = True
    graders = (
        tuple(name for name, _ in grader_files)
        if grader_files
        else tuple(args.correct_column or ())
    )
    if not graders and args.command in {"select", "policy"}:
        # Exact label matching has one grader, named after the reference column.
        graders = (meta["reference_column"],)
    curve = None
    exit_code = 0
    html = getattr(args, "html", None) is not None or args.render == "html"
    if args.command == "evaluate":
        if (
            args.reliability_plot is not None or args.bins != 10
        ) and not args.probability:
            raise ValueError(
                "calibration bins and reliability plots require --probability"
            )
        result, curve = evaluate(
            rows,
            threshold=args.threshold,
            risk_target=args.risk_target,
            holdout_confidence=args.holdout_confidence,
            accepted_population=population,
            probability=args.probability,
            bins=args.bins,
            extra_metrics=args.metric or (),
            review_missing_scores=review_missing_scores,
        )
        if args.risk_target is not None:
            exit_code = int(result["overall"]["meets_target"] is not True)
    elif args.command == "audit":
        result = audit(
            rows,
            graders=graders,
            threshold=args.threshold,
            risk_target=args.risk_target,
            holdout_confidence=args.holdout_confidence,
            accepted_population=population,
            bootstrap=args.bootstrap,
            seed=args.seed,
            review_missing_scores=review_missing_scores,
        )
        if args.disagreements_output is not None:
            write_disagreements(
                staged[args.disagreements_output],
                rows,
                result,
                include_output_key=meta["output_key_column"] is not None,
            )
            meta["disagreements_output"] = str(args.disagreements_output)
        if args.risk_target is not None:
            exit_code = int(
                any(
                    counts["meets_target"] is not True
                    for counts in result["graders"].values()
                )
            )
    elif args.command == "select":
        thresholds = args.thresholds
        score_source = getattr(args, "negated_score_source", None)
        negate = score_source is not None
        score_column = score_source or args.score_column
        design = ()
        if args.design is not None:
            if args.design.resolve() == args.predictions.resolve():
                raise ValueError("design and calibration need separate files")
            if track_fit:
                design, _ = read_table_rows(
                    args.design,
                    id_column=args.id_column,
                    score_column=score_column,
                    output_key_column=meta["output_key_column"],
                    selection=args.prediction_filter,
                    require_labels=False,
                    negate_score=negate,
                    review_missing_scores=review_missing_scores,
                )
                design = validated_rows(
                    design,
                    (),
                    minimum_graders=0,
                    review_missing_scores=review_missing_scores,
                )
                design_scores = (row.score for row in design)
            else:
                design_scores = read_design_scores(
                    args.design,
                    score_column,
                    calibration=rows,
                    id_column=args.id_column,
                    output_key_column=meta["output_key_column"],
                    selection=args.prediction_filter,
                    negate_score=negate,
                    review_missing_scores=review_missing_scores,
                )
            thresholds = design_thresholds(
                design_scores,
                grid_size=grid_size,
                review_missing_scores=review_missing_scores,
            )
        elif negate:
            thresholds = [-threshold for threshold in thresholds]
        meta["design_path"] = str(args.design) if args.design is not None else None
        if track_fit:
            meta["fit_outputs"] = fit_outputs_metadata(
                meta["output_key_column"], rows, design=design
            )
        result = select(
            rows,
            graders=graders,
            thresholds=thresholds,
            risk_target=args.risk_target,
            failure_rate=args.failure_rate,
            rule=args.rule,
            review_missing_scores=review_missing_scores,
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
            holdout_confidence=args.holdout_confidence,
            failure_rate=args.failure_rate,
            rule=args.rule,
            seed=args.seed,
            calibrate=args.calibrate,
            calibration_input=args.calibration_input,
            design_grid_size=grid_size,
            review_missing_scores=review_missing_scores,
            split_checks=args.split_checks,
        )
        exit_code = int(not result["confirmed"])
        if html or source_rows is not None or track_fit:
            design, calibration, rows = split_rows(rows, args.seed)
            graders = (*graders, *(args.check_column or ()))
            if track_fit:
                meta["fit_outputs"] = fit_outputs_metadata(
                    meta["output_key_column"], calibration, design=design
                )
        if source_rows is not None:
            joined_grades = {
                name: {row.id: row.outcomes[index] for row in rows}
                for index, (name, _) in enumerate(grader_files)
            }
            write_holdout_rows(
                staged[holdout_output],
                source_rows,
                rows,
                id_column=args.id_column,
                grader_columns=joined_grades,
            )
            meta["holdout_output"] = str(holdout_output)
    if html and args.command in {"audit", "policy"}:
        threshold = (
            result["threshold"]
            if args.command == "audit"
            else result["selection"]["selected_threshold"]
        )
        accepted = [
            row for row in rows if threshold is not None and accepted_at(row, threshold)
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
                threshold_points(
                    (
                        (row.score, not row.outcomes[index])
                        for row in rows
                        if row.score is not None
                    ),
                    total=len(rows),
                ),
            )
    else:
        curves = None
    if candidate_policy is not None:
        result.update(
            sample_error_limits(
                population, len(rows), args.risk_target, args.holdout_confidence
            )
        )
        assessments = (
            (result["overall"],)
            if args.command == "evaluate"
            else result["graders"].values()
        )
        for counts in assessments:
            counts.update(
                batch_verdict(
                    counts, population, args.risk_target, args.holdout_confidence
                )
            )
    return make_report(args.command, meta, result), curve, curves, exit_code


def output_paths(args):
    inputs = threshold_input_paths(args)
    inputs.update(
        path.resolve() for _, path in (getattr(args, "grader_file", None) or ())
    )
    if getattr(args, "design", None) is not None:
        inputs.add(args.design.resolve())
    directory = getattr(args, "output_dir", None)
    artifacts = (
        (directory / "metrics.json", directory / "risk_coverage.csv")
        if directory is not None
        else ()
    )
    if (args.render is None) != (args.render_output is None):
        raise ValueError("--render and --render-output are used together")
    outputs = list(artifacts[1:])
    if artifacts:
        if args.output is None:
            args.output = artifacts[0]
        if args.output.resolve() == artifacts[1].resolve():
            raise ValueError("--output cannot replace risk_coverage.csv")
        if args.output.resolve() != artifacts[0].resolve():
            outputs.append(artifacts[0])
    if args.output is not None:
        outputs.append(args.output)
    resolved = {path.resolve() for path in outputs}
    if len(resolved) != len(outputs):
        raise ValueError("report outputs must use different paths")
    for option, suffix in (
        ("plot", ".svg"),
        ("reliability_plot", ".svg"),
        ("html", ".html"),
        ("holdout_output", ".csv"),
        ("disagreements_output", ".csv"),
        ("render_output", None),
    ):
        path = getattr(args, option, None)
        if path is None:
            continue
        flag = "--" + option.replace("_", "-")
        if suffix is not None and path.suffix.lower() != suffix:
            raise ValueError(f"{flag} needs a {suffix} file")
        target = path.resolve()
        if target in resolved:
            raise ValueError(f"{flag} must use a different path from report outputs")
        resolved.add(target)
        outputs.append(path)
    if inputs.intersection(resolved):
        raise ValueError("output would overwrite an input file")
    if args.output is not None:
        outputs.remove(args.output)
        outputs.append(args.output)
    return artifacts, outputs


def write_analysis_outputs(args, artifacts, staged, report, curve, curves):
    plot = getattr(args, "plot", None)
    if plot is not None:
        write_svg(staged[plot], render_report("svg", report, curve=curve))
    reliability = getattr(args, "reliability_plot", None)
    if reliability is not None:
        write_svg(staged[reliability], reliability_diagram(report["reliability_bins"]))
    html = getattr(args, "html", None)
    if html is not None:
        write_svg(
            staged[html], render_report("html", report, curve=curve, curves=curves)
        )
    if args.render is not None:
        write_svg(
            staged[args.render_output],
            render_report(args.render, report, curve=curve, curves=curves),
        )
    if artifacts:
        write_curve(staged[artifacts[1]], curve)
    serialized = render_report("json", report)
    if args.output is not None:
        write_report(staged[args.output], serialized)
    if artifacts and artifacts[0] in staged and artifacts[0] != args.output:
        write_report(staged[artifacts[0]], serialized)
    return serialized if args.json else render_report("text", report)


def run_workflow_command(args):
    report = run_workflow(args.config, save_outputs=True)
    print(render_report("json" if args.json else "text", report))
    return int(not report["policy"]["confirmed"])


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="caat",
        description=(
            "Score, calibrate, evaluate, and use model confidence to decide "
            "which saved predictions to accept and which to review."
        ),
        epilog=(
            "Terms used in the reports: "
            "https://github.com/Daveyuwang/caat/blob/main/docs/guide/glossary.md"
        ),
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
            "Split saved predictions, choose a cutoff, and check it on held-out rows.",
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
    feature_command.add_argument(
        "--include-output-key",
        action="store_true",
        help="Include the SHA-256 of each saved response as output_sha256.",
    )
    commands["features"] = feature_command
    orient_command = subcommands.add_parser(
        "orient", help="Add confidence columns for lower-is-better scores."
    )
    orient_command.add_argument("predictions", type=Path, metavar="CSV_OR_JSONL")
    orient_command.add_argument("--output", type=Path, metavar="CSV", required=True)
    lower_is_better_option(orient_command, required=True)
    prediction_filter_option(orient_command)
    review_missing_scores_option(
        orient_command, help_text="Keep missing scores blank in derived columns."
    )
    commands["orient"] = orient_command
    commands["calibrate"] = subcommands.add_parser(
        "calibrate", help="Fit and apply correctness probability maps."
    )
    calibrate_options(commands["calibrate"])
    commands["queue"] = subcommands.add_parser(
        "queue", help="Order saved predictions for review or labeling."
    )
    queue_options(commands["queue"])
    commands["route"] = subcommands.add_parser(
        "route", help="Apply a frozen cutoff and write one decision per prediction."
    )
    route_options(commands["route"])
    commands["collect"] = subcommands.add_parser(
        "collect", help="Save responses and confidence from a model or chat API."
    )
    collect_options(commands["collect"])
    commands["scorer"] = subcommands.add_parser(
        "scorer", help="Fit or apply a portable confidence scorer."
    )
    configure_scorer(commands["scorer"])
    commands["scores"] = subcommands.add_parser(
        "scores", help="Compare saved scores on shared labels and coverage budgets."
    )
    scores_options(commands["scores"])
    commands["inspect"] = subcommands.add_parser(
        "inspect",
        help="Export Inspect completions and join external confidence scores.",
    )
    inspect_options(commands["inspect"])
    commands["lm-eval"] = subcommands.add_parser(
        "lm-eval", help="Read saved sample logs from lm-evaluation-harness."
    )
    lm_eval_options(commands["lm-eval"])
    commands["eee"] = subcommands.add_parser(
        "eee", help="Read saved Every Eval Ever sample records."
    )
    eee_options(commands["eee"])
    commands["helm"] = subcommands.add_parser(
        "helm", help="Export confidence and grades from a saved HELM run directory."
    )
    helm_options(commands["helm"])
    commands["monitor"] = subcommands.add_parser(
        "monitor", help="Check accepted risk as new labeled rows arrive."
    )
    monitor_options(commands["monitor"])
    commands["plan"] = subcommands.add_parser(
        "plan", help="Plan independent labels for a frozen policy."
    )
    plan_options(commands["plan"])
    commands["audit-block"] = subcommands.add_parser(
        "audit-block", help="Issue or resume one block of a frozen audit."
    )
    block_options(commands["audit-block"], named_file)
    commands["run"] = subcommands.add_parser(
        "run", help="Run a review workflow from a TOML file."
    )
    commands["run"].add_argument(
        "config", nargs="?", type=Path, default=Path("caat.toml")
    )
    commands["run"].add_argument(
        "--json", action="store_true", help="Print the JSON report."
    )
    commands["plugins"] = subcommands.add_parser(
        "plugins", help="List available components and their sources."
    )
    commands["plugins"].add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "features":
            count = write_feature_csv(
                args.predictions,
                args.output,
                args.feature or DEFAULT_FEATURES,
                include_output_key=args.include_output_key,
            )
            print(f"Wrote scores for {count} responses to {args.output}.")
            return 0
        if args.command == "orient":
            count = orient_score_table(
                args.predictions,
                args.output,
                args.lower_is_better,
                selection=args.prediction_filter,
                review_missing_scores=bool(args.review_missing_scores),
            )
            print(f"Wrote {count} rows to {args.output}.")
            return 0
        if args.command == "calibrate":
            return run_calibrate(args)
        if args.command == "queue":
            return run_queue(args)
        if args.command == "route":
            return run_route(args)
        if args.command == "collect":
            return run_collect(args)
        if args.command == "scores":
            console = Path(sys.argv[0]).name.lower() in {"caat", "caat.exe"}
            return run_scores(args, auto_workers=argv is None and console)
        if args.command == "inspect":
            return run_inspect(args)
        if args.command == "lm-eval":
            return run_lm_eval(args)
        if args.command == "eee":
            return run_eee(args)
        if args.command == "helm":
            count = helm_log.export_run(
                args.run, args.output, metrics=args.metric, annotations=args.annotation
            )
            print(f"Wrote {count} rows to {args.output}.")
            return 0
        if args.command == "monitor":
            return run_monitor(args)
        if args.command == "audit-block":
            return run_block(args)
        if args.command == "plan":
            return run_plan(args)
        if args.command == "scorer":
            return run_scorer(args)
        if args.command == "run":
            return run_workflow_command(args)
        if args.command == "plugins":
            emit(
                make_report("plugins", {}, {"stages": catalog()}), json_output=args.json
            )
            return 0
        fit_outputs = label_plan = candidate_policy = None
        if args.command in {"evaluate", "audit"}:
            label_plan = load_label_plan(args)
            if label_plan and "stages" in label_plan:
                if args.command != "evaluate":
                    raise ValueError("use evaluate for checkpoint verdicts")
                return evaluate_partial(args, label_plan)
            if getattr(args, "partial", False):
                return evaluate_partial(args, label_plan)
            saved = load_threshold_options(args, label_plan=label_plan)
            if args.batch_predictions is not None:
                candidate_policy = saved
            if args.holdout_confidence is not None:
                fit_outputs = saved.get("fit_outputs")
            if fit_outputs is None:
                use_available_output_key_column(args, saved)
        artifacts, outputs = output_paths(args)
        with staged_outputs(outputs) as staged:
            report, curve, curves, status = run(
                args,
                staged=staged,
                fit_outputs=fit_outputs,
                label_plan=label_plan,
                candidate_policy=candidate_policy,
            )
            stdout = write_analysis_outputs(
                args, artifacts, staged, report, curve, curves
            )
        print(stdout)
        return status
    except (OSError, ValueError, csv.Error) as exc:
        commands[args.command].error(str(exc))
