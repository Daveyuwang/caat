"""Run a review policy from one TOML configuration."""

from dataclasses import replace
from itertools import chain

from .artifacts import staged_outputs
from .calibrate import CALIBRATORS, check_scale, default_scale
from .policy import _policy_for_splits, split_rows
from .queue import review_queue
from .renderers import render_report
from .report import fit_outputs_metadata, make_report
from .rules import RULES, validate_holdout_confidence
from .scores import compare_scores, read_score_rows
from .select import (
    DEFAULT_DESIGN_GRID_SIZE,
    build_threshold_candidates,
    design_thresholds,
    test_candidates,
)
from .table import (
    accepted_at,
    graded_row,
    negated_score_columns,
    parse_prediction_filter,
    read_grader_file,
    write_holdout_rows,
)


REQUIRED = {"predictions", "score_columns", "risk_target"}
SPLITS = ("design", "calibration", "holdout")
GRADER_FILE_SETTINGS = ("grader_files", "holdout_grader_files")
DEFAULTS = {
    "id_column": "id",
    "output_key_column": None,
    "prediction_filter": None,
    "review_missing_scores": False,
    "correct_columns": (),
    "check_columns": (),
    "grader_files": {},
    "holdout_grader_files": {},
    "grader_id_column": None,
    "grader_output_key_column": None,
    "lower_is_better": (),
    "score_selection": "design-aurc",
    "rule": "ltt-bonferroni",
    "failure_rate": 0.05,
    "design_grid_size": None,
    "holdout_confidence": 0.95,
    "calibrate": "platt",
    "calibration_input": None,
    "seed": 42,
    "bootstrap": 500,
    "queue_budget": 10,
    "html": "caat-report.html",
    "json_output": None,
    "holdout_output": None,
}


def _prediction_paths(directory, value):
    if isinstance(value, dict):
        if value.keys() != set(SPLITS):
            raise ValueError(
                "predictions table needs design, calibration, and holdout paths"
            )
        paths = {}
        for name in SPLITS:
            path = value[name]
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"predictions.{name} must be a nonempty path")
            paths[name] = (directory / path).resolve()
        if len(set(paths.values())) != len(SPLITS):
            raise ValueError("each prediction split needs a separate input file")
        return paths
    if not isinstance(value, str) or not value.strip():
        raise ValueError("predictions must be a path or a table of split paths")
    return (directory / value).resolve()


def load_workflow(path):
    try:
        import tomllib as toml
    except ModuleNotFoundError:
        try:
            import tomli as toml
        except ModuleNotFoundError as exc:
            raise ValueError(
                "install caat-annotation[workflow] to read TOML on Python 3.10"
            ) from exc

    with path.open("rb") as source:
        config = toml.load(source)
    if unknown := config.keys() - (REQUIRED | DEFAULTS.keys()):
        raise ValueError(f"unknown workflow setting: {', '.join(sorted(unknown))}")
    if missing := REQUIRED - config.keys():
        raise ValueError(f"missing workflow setting: {', '.join(sorted(missing))}")
    settings = {**DEFAULTS, **config}
    if type(settings["review_missing_scores"]) is not bool:
        raise ValueError("review_missing_scores must be a boolean")
    for name in ("id_column", "score_selection", "rule", "calibrate"):
        value = settings[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    for name in ("output_key_column", "grader_id_column", "grader_output_key_column"):
        value = settings[name]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a nonempty string")
    for field in GRADER_FILE_SETTINGS:
        files = settings[field]
        if not isinstance(files, dict) or any(
            not name.strip() or not isinstance(value, str) or not value.strip()
            for name, value in files.items()
        ):
            raise ValueError(f"{field} must map grader names to file paths")
        settings[field] = {
            name: (path.parent / value).resolve() for name, value in files.items()
        }
    grader_files = settings["grader_files"]
    holdout_files = settings["holdout_grader_files"]
    if bool(grader_files) == ("correct_columns" in config):
        raise ValueError("set exactly one of correct_columns or grader_files")
    external_graders = (*grader_files, *holdout_files)
    if external_graders:
        key_column = settings["output_key_column"] or "output_sha256"
        if key_column in external_graders:
            raise ValueError("grader names must differ from the output key column")
        settings["output_key_column"] = key_column
    else:
        for name in ("grader_id_column", "grader_output_key_column"):
            if settings[name] is not None:
                raise ValueError(
                    f"{name} requires grader_files or holdout_grader_files"
                )
    if settings["prediction_filter"] is not None:
        settings["prediction_filter"] = parse_prediction_filter(
            settings["prediction_filter"]
        )
    for name in (
        "score_columns",
        "correct_columns",
        "check_columns",
        "lower_is_better",
    ):
        values = settings[name]
        optional = name in {"check_columns", "lower_is_better"} or (
            name == "correct_columns" and bool(grader_files)
        )
        if (
            not isinstance(values, (tuple, list))
            or (not values and not optional)
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ):
            qualifier = "" if optional else "nonempty "
            raise ValueError(f"{name} must be a {qualifier}array of column names")
        settings[name] = tuple(values)
    other_columns = (
        settings["id_column"],
        *settings["correct_columns"],
        *settings["check_columns"],
        *external_graders,
    )
    raw_scores = settings["score_columns"]
    columns = (*raw_scores, *other_columns)
    if len(set(columns)) != len(columns):
        raise ValueError("ID, score, and grader columns must be distinct")
    lower = settings["lower_is_better"]
    if unknown := set(lower).difference(raw_scores):
        raise ValueError(
            f"lower_is_better columns must appear in score_columns: {', '.join(sorted(unknown))}"
        )
    scores = list(raw_scores)
    negated = {}
    for index, name in negated_score_columns(raw_scores, lower):
        negated[name] = raw_scores[index]
        scores[index] = name
    oriented_columns = (*scores, *other_columns)
    if len(set(oriented_columns)) != len(oriented_columns):
        raise ValueError(
            "derived score names must be distinct from other scores, IDs, and graders"
        )
    settings["score_columns"] = tuple(scores)
    settings["negated_scores"] = negated

    for name in ("risk_target", "failure_rate", "holdout_confidence"):
        if type(settings[name]) not in (int, float):
            raise ValueError(f"{name} must be a number")
    if not 0 <= settings["risk_target"] <= 1:
        raise ValueError("risk_target must be between 0 and 1")
    if not 0 < settings["failure_rate"] < 1:
        raise ValueError("failure_rate must be strictly between 0 and 1")
    grid_size = settings["design_grid_size"]
    if grid_size is not None and (type(grid_size) is not int or grid_size < 1):
        raise ValueError("design_grid_size must be a positive integer")
    validate_holdout_confidence(settings["holdout_confidence"])
    for name in ("seed", "bootstrap", "queue_budget"):
        if type(settings[name]) is not int:
            raise ValueError(f"{name} must be an integer")
    if settings["bootstrap"] < 0 or settings["queue_budget"] < 1:
        raise ValueError(
            "bootstrap must be nonnegative and queue_budget must be positive"
        )
    if settings["rule"] not in RULES:
        raise ValueError(f"unknown selection rule: {settings['rule']!r}")
    if settings["score_selection"] not in {"design-aurc", "joint-calibration"}:
        raise ValueError(f"unknown score_selection: {settings['score_selection']!r}")
    if settings["score_selection"] == "joint-calibration" and settings["rule"] not in {
        "ltt-bonferroni",
        "ltt-holm",
    }:
        raise ValueError("joint-calibration requires ltt-bonferroni or ltt-holm")
    method = settings["calibrate"]
    scale = default_scale(method)
    if "calibration_input" in config:
        scale = settings["calibration_input"]
        if not isinstance(scale, str):
            raise ValueError("calibration_input must be score or probability")
        check_scale(CALIBRATORS[method], scale)
    settings["calibration_input"] = scale
    predictions = _prediction_paths(path.parent, settings["predictions"])
    settings["predictions"] = predictions
    inputs = predictions.values() if isinstance(predictions, dict) else (predictions,)
    paths = {path.resolve(), *inputs, *grader_files.values(), *holdout_files.values()}
    for name, suffix in (
        ("html", ".html"),
        ("json_output", ".json"),
        ("holdout_output", ".csv"),
    ):
        value = settings[name]
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
        output = (path.parent / value).resolve()
        if output in paths:
            raise ValueError(f"{name} would overwrite another workflow file")
        if output.suffix.lower() != suffix or output.is_dir():
            raise ValueError(f"{name} must name a {suffix} file")
        paths.add(output)
        settings[name] = output
    return settings


def _joint_calibration(design, calibration, settings, fallback, *, graders):
    """Test all score and cutoff pairs as one family."""
    risk_target = settings["risk_target"]
    failure_rate = settings["failure_rate"]
    rule = settings["rule"]
    grid_size = settings["design_grid_size"] or DEFAULT_DESIGN_GRID_SIZE
    review_missing_scores = settings["review_missing_scores"]
    by_score = {}
    for index, name in enumerate(settings["score_columns"]):
        thresholds = design_thresholds(
            (row.scores[index] for row in design),
            grid_size=grid_size,
            review_missing_scores=review_missing_scores,
        )
        by_score[name], _ = build_threshold_candidates(
            (
                graded_row(row.id, row.scores[index], row.outcomes[: len(graders)])
                for row in calibration
            ),
            graders=graders,
            thresholds=thresholds,
            risk_target=risk_target,
            review_missing_scores=review_missing_scores,
        )

    pairs = [
        (name, candidate)
        for name, candidates in by_score.items()
        for candidate in candidates
    ]
    decision = test_candidates(
        [candidate for _, candidate in pairs],
        rule=rule,
        risk_target=risk_target,
        failure_rate=failure_rate,
    )
    winner = max(
        (pair for pair in pairs if pair[1]["passes"]),
        key=lambda pair: (pair[1]["accepted"], -pair[1]["pvalue"]),
        default=None,
    )
    selected, threshold = (
        (winner[0], winner[1]["threshold"]) if winner else (fallback, None)
    )
    return selected, {
        "schema_version": 1,
        "rule": rule,
        "graders": list(graders),
        "risk_target": risk_target,
        "failure_rate": failure_rate,
        "calibration_total": len(calibration),
        "selected_threshold": threshold,
        "candidates": by_score[selected],
        "per_candidate_level": decision.per_candidate_level,
        "candidate_family_size": len(pairs),
        "candidate_thresholds_by_score": {
            name: [row["threshold"] for row in candidates]
            for name, candidates in by_score.items()
        },
        "candidates_by_score": by_score,
        "guarantee": {
            "confidence": 1 - failure_rate,
            "assumptions": "IID calibration from the target population and score-cutoff pairs fixed independently of calibration",
        }
        if winner and decision.controls_risk
        else None,
    }


def _read_grader_exports(files, rows, *, id_column, output_key_column):
    output_keys = {}
    for row in rows:
        if not row.id or row.id in output_keys:
            raise ValueError("score rows must have unique nonempty IDs")
        output_keys[row.id] = row.output_key
    return {
        name: read_grader_file(
            file,
            name,
            output_keys,
            id_column=id_column,
            output_key_column=output_key_column,
            allow_extra=True,
        )
        for name, file in files.items()
    }


def run_workflow(path, *, save_outputs=False):
    path = path.resolve()
    settings = load_workflow(path)
    scores = settings["score_columns"]
    inline_correct, inline_checks = (
        settings["correct_columns"],
        settings["check_columns"],
    )
    grader_files = settings["grader_files"]
    holdout_files = settings["holdout_grader_files"]
    graders = tuple(grader_files) or inline_correct
    checks = (*inline_checks, *holdout_files)
    inline_graders = (*inline_correct, *inline_checks)
    everyone = (*graders, *checks)
    seed = settings["seed"]
    id_column = settings["id_column"]
    key_column = settings["output_key_column"]
    grid_size = settings["design_grid_size"]
    holdout_output = settings["holdout_output"]
    json_output = settings["json_output"]
    source_rows = [] if save_outputs and holdout_output is not None else None
    predictions = settings["predictions"]
    prediction_filter = settings["prediction_filter"]
    review_missing_scores = settings["review_missing_scores"]
    provided = isinstance(predictions, dict)
    table_options = {
        "id_column": id_column,
        "output_key_column": key_column,
        "selection": prediction_filter,
        "negated_scores": settings["negated_scores"],
        "review_missing_scores": review_missing_scores,
    }
    if provided:
        partitions = []
        used_ids = set()
        for name, file in predictions.items():
            rows = read_score_rows(
                file,
                scores,
                inline_graders if name == "holdout" else inline_correct,
                source_rows=source_rows if name == "holdout" else None,
                **table_options,
            )
            if not rows:
                raise ValueError(f"predictions.{name} needs at least one row")
            ids = {row.id for row in rows}
            if overlap := ids & used_ids:
                raise ValueError(
                    f"predictions.{name} reuses sample ID {min(overlap)!r}"
                )
            used_ids.update(ids)
            partitions.append(rows)
    else:
        rows = read_score_rows(
            predictions,
            scores,
            inline_graders,
            source_rows=source_rows,
            **table_options,
        )
        partitions = split_rows(rows, seed)
    grader_options = {
        "id_column": settings["grader_id_column"] or id_column,
        "output_key_column": settings["grader_output_key_column"] or key_column,
    }
    primary_grades = {}
    if grader_files:
        primary_grades = _read_grader_exports(
            grader_files, chain.from_iterable(partitions), **grader_options
        )
        for partition in partitions:
            for index, row in enumerate(partition):
                partition[index] = replace(
                    row,
                    outcomes=tuple(bits[row.id] for bits in primary_grades.values())
                    + row.outcomes,
                )
    holdout_grades = {}
    if holdout_files:
        design_rows, calibration_rows, holdout_rows = partitions
        holdout_grades = _read_grader_exports(
            holdout_files, holdout_rows, **grader_options
        )
        partitions = (
            design_rows,
            calibration_rows,
            [
                replace(
                    row,
                    outcomes=(
                        *row.outcomes,
                        *(bits[row.id] for bits in holdout_grades.values()),
                    ),
                )
                for row in holdout_rows
            ],
        )
    comparison = compare_scores(
        (
            replace(row, outcomes=row.outcomes[: len(graders)])
            for row in chain.from_iterable(partitions)
        ),
        names=scores,
        graders=graders,
        seed=seed,
        bootstrap=settings["bootstrap"],
        split_sizes=tuple(map(len, partitions)) if provided else None,
        review_missing_scores=review_missing_scores,
    )
    selected_score = comparison["recommended_score"] or scores[0]
    selection = None
    if settings["score_selection"] == "joint-calibration":
        design_rows, calibration_rows, _ = partitions
        selected_score, selection = _joint_calibration(
            design_rows, calibration_rows, settings, selected_score, graders=graders
        )
    column = scores.index(selected_score)
    design, calibration, heldout = (
        [
            graded_row(
                row.id, row.scores[column], row.outcomes, output_key=row.output_key
            )
            for row in partition
        ]
        for partition in partitions
    )
    result = _policy_for_splits(
        design,
        calibration,
        heldout,
        graders=graders,
        check_graders=checks,
        calibration_graders=graders if provided else (*graders, *inline_checks),
        risk_target=settings["risk_target"],
        holdout_confidence=settings["holdout_confidence"],
        failure_rate=settings["failure_rate"],
        rule=settings["rule"],
        split_seed=None if provided else seed,
        calibrate=settings["calibrate"],
        calibration_input=settings["calibration_input"],
        design_grid_size=grid_size or DEFAULT_DESIGN_GRID_SIZE,
        selection=selection,
        review_missing_scores=review_missing_scores,
    )
    cutoff = result["selection"]["selected_threshold"] if result["confirmed"] else None
    eligible = [
        row for row in heldout if cutoff is None or not accepted_at(row, cutoff)
    ]
    budget = settings["queue_budget"]
    queued = []
    if eligible:
        queued, _, _ = review_queue(
            eligible,
            budget=min(budget, len(eligible)),
            graders=everyone,
            seed=seed,
            review_missing_scores=review_missing_scores,
        )
    prediction_inputs = (
        {"predictions": {name: str(file) for name, file in predictions.items()}}
        if provided
        else {"path": str(predictions)}
    )
    metadata = {
        "config": str(path),
        **prediction_inputs,
        "html": str(settings["html"]),
        "id_column": id_column,
        "score_columns": list(scores),
        "correct_columns": list(inline_correct),
        "check_columns": list(inline_checks),
    }
    for field in GRADER_FILE_SETTINGS:
        if files := settings[field]:
            metadata[field] = {name: str(file) for name, file in files.items()}
    if grader_files or holdout_files:
        for name in ("grader_id_column", "grader_output_key_column"):
            if settings[name] is not None:
                metadata[name] = settings[name]
    if settings["lower_is_better"]:
        metadata["lower_is_better"] = list(settings["lower_is_better"])
    if source := settings["negated_scores"].get(selected_score):
        metadata["negated_score_source"] = source
    if grid_size is not None:
        metadata["design_grid_size"] = grid_size
    if prediction_filter is not None:
        filter_column, filter_value = prediction_filter
        metadata["prediction_filter"] = {"column": filter_column, "value": filter_value}
    if review_missing_scores:
        metadata["review_missing_scores"] = True
    if selection is not None:
        metadata["score_selection"] = settings["score_selection"]
    if key_column is not None:
        metadata["fit_outputs"] = fit_outputs_metadata(
            key_column, calibration, design=design
        )
    if json_output is not None:
        metadata["json_output"] = str(json_output)
    if holdout_output is not None:
        metadata["holdout_output"] = str(holdout_output)
    report = make_report(
        "run",
        metadata,
        {
            "selected_score": selected_score,
            "score_comparison": comparison,
            "policy": result,
            "review_queue": {
                "eligible": len(eligible),
                "requested_budget": budget,
                "entries": [{"id": row.id, "score": row.score} for row in queued],
            },
        },
    )
    if save_outputs:
        html = settings["html"]
        outputs = [
            path for path in (holdout_output, html, json_output) if path is not None
        ]
        with staged_outputs(outputs) as staged:
            staged[html].write_text(
                render_report("html", report), encoding="utf-8", newline="\n"
            )
            if source_rows is not None:
                write_holdout_rows(
                    staged[holdout_output],
                    source_rows,
                    heldout,
                    id_column=id_column,
                    grader_columns={**primary_grades, **holdout_grades},
                )
            if json_output is not None:
                staged[json_output].write_text(
                    render_report("json", report) + "\n", encoding="utf-8", newline="\n"
                )
    return report
