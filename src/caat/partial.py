"""Read the grades available for a frozen draw and determine its full-budget verdict."""

from .artifacts import staged_outputs
from .metrics import Counts
from .plan import (
    batch_verdict,
    candidate_draw,
    check_draw_rows,
    sample_error_limits,
    verdict_ranges,
)
from .report import emit, make_report, read_threshold_report
from .rules import assess_accepted_risk
from .table import read_grader_file, read_table_rows


def checkpoint_progress(stages, bits):
    outcomes = list(bits.values())
    counts = dict(
        graded=len(outcomes),
        errors=len(outcomes) - sum(outcomes),
        decision_status="pending",
        batch_verdict=None,
    )
    for index, stage in enumerate(stages, 1):
        n = stage["labels"]
        if len(outcomes) < n:
            return dict(counts, next_checkpoint=n)
        errors = n - sum(outcomes[:n])
        verdict = next(
            name for name, lo, hi in verdict_ranges(n, stage) if lo <= errors <= hi
        )
        if verdict != "inconclusive" or index == len(stages):
            return dict(
                counts,
                decision_status="settled",
                batch_verdict=verdict,
                decision_stage=index,
                decision_errors=errors,
                decision_labels=n,
            )


def progress_report(plan, grades):
    size, cap = plan["accepted_population"], plan["accepted"]
    if "stages" in plan:
        results = {
            name: checkpoint_progress(plan["stages"], bits)
            for name, bits in grades.items()
        }
        return dict(
            accepted_population=size,
            label_budget=plan["label_budget"],
            planned_labels=cap,
            risk_target=plan["risk_target"],
            holdout_confidence=plan["holdout_confidence"],
            stages=plan["stages"],
            graders=results,
            decision_status="settled"
            if all(r["batch_verdict"] for r in results.values())
            else "pending",
        )
    target, confidence = plan["risk_target"], plan["holdout_confidence"]
    limits = sample_error_limits(size, cap, target, confidence) if size else {}

    ranges = verdict_ranges(cap, limits)

    results = {}
    for name, bits in grades.items():
        graded = len(bits)
        errors = graded - sum(bits.values())
        high = errors + cap - graded
        outcome = next(
            (name for name, first, last in ranges if first <= errors and high <= last),
            None,
        )
        counts = {
            "graded": graded,
            "errors": errors,
            "possible_final_errors": [errors, high],
            "batch_verdict": outcome,
            "decision_status": "settled" if outcome else "pending",
        }
        if size and graded == cap:
            full = assess_accepted_risk(
                Counts(cap, cap, errors), target, confidence, accepted_population=size
            )
            counts.update(full, **batch_verdict(full, size, target, confidence))
        results[name] = counts
    return {
        "accepted_population": size,
        "label_budget": plan["label_budget"],
        "planned_labels": cap,
        "risk_target": target,
        "holdout_confidence": confidence,
        **limits,
        "decision_status": "settled"
        if all(v["batch_verdict"] for v in results.values())
        else "pending",
        "graders": results,
    }


def evaluate_partial(args, plan):
    if plan is None or "threshold_report_sha256" not in plan:
        raise ValueError(
            "--partial requires a fixed-budget --plan-report, --threshold-report and --batch-predictions"
        )
    unsupported = (
        "metric",
        "probability",
        "output_dir",
        "render",
        "render_output",
        "plot",
        "html",
        "prediction_filter",
        "allow_extra_grades",
        "raw_score_column",
        "prediction_column",
        "reference_column",
        "reliability_plot",
        "review_missing_scores",
    )
    if (
        any(getattr(args, name) for name in unsupported)
        or args.bins != 10
        or args.adapter not in (None, "csv")
    ):
        raise ValueError(
            "--partial evaluates the frozen sample using keyed exports or inline grade columns"
        )
    files = args.grader_file or []
    names = [name for name, _ in files] or args.correct_column or []
    if not names or not all(names) or len(names) != len(set(names)):
        raise ValueError(
            "provide distinct --grader-file names or --correct-column names"
        )
    if not files and (args.grader_id_column or args.grader_output_key_column):
        raise ValueError("grader column overrides require --grader-file")
    saved = read_threshold_report(args.threshold_report, fingerprint=True)
    draw = candidate_draw(args.batch_predictions, saved, plan)
    keyed = bool(draw and draw[0].output_key is not None)
    key = args.output_key_column or ("output_key" if keyed else None)
    rows, _ = read_table_rows(
        args.predictions,
        id_column=args.id_column or "id",
        score_column=args.score_column or "score",
        group_column=args.group_column,
        output_key_column=key,
        correct_columns=args.correct_column or (),
        require_labels=False,
    )
    check_draw_rows(draw[: len(rows)], rows)
    if files:
        if draw and not keyed:
            raise ValueError(
                "separate grader files require output keys in the frozen draw"
            )
        keys = {r.id: r.output_key for r in rows}
        grades = {
            name: read_grader_file(
                path,
                name,
                keys,
                id_column=args.grader_id_column or args.id_column or "id",
                output_key_column=args.grader_output_key_column or key or "output_key",
                require_all=False,
            )
            for name, path in files
        }
        for name, bits in grades.items():
            if bits.keys() != {r.id for r in draw[: len(bits)]}:
                raise ValueError(
                    f"{name}: graded IDs must form a prefix of the frozen draw"
                )
    else:
        grades = {
            name: {r.id: r.outcomes[i] for r in rows} for i, name in enumerate(names)
        }
    if "stages" in plan:
        grades = {
            name: {r.id: bits[r.id] for r in draw[: len(bits)]}
            for name, bits in grades.items()
        }
    result = progress_report(plan, grades)
    inputs = [
        args.threshold_report,
        args.plan_report,
        args.batch_predictions,
        args.predictions,
        *(p for _, p in files),
    ]
    if args.output and args.output.resolve() in {p.resolve() for p in inputs}:
        raise ValueError("output would overwrite an audit input")
    meta = {
        "partial": True,
        "plan_report": str(args.plan_report),
        "threshold_report_sha256": plan["threshold_report_sha256"],
        "batch_predictions": str(args.batch_predictions),
        "sample_seed": plan["sample_seed"],
        "sample": str(args.predictions),
        "grader_files": {name: str(path) for name, path in files},
    }
    with staged_outputs([args.output] if args.output else []) as staged:
        emit(
            make_report("evaluate", meta, result),
            json_output=args.json,
            output=staged.get(args.output),
        )
    return int(any(r["batch_verdict"] != "meets" for r in result["graders"].values()))
