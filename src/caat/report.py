"""Read and render versioned CLI reports."""

import csv
import json
import re
from importlib.metadata import version
from math import isfinite

from .table import negated_score_name, output_fingerprints


SCHEMA_VERSION = 2


def make_report(command, input_meta, contents):
    return {
        **contents,
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "caat_version": version("caat-annotation"),
        "input": input_meta,
    }


def fit_outputs_metadata(key_column, calibration, *, design=()):
    calibration_keys = output_fingerprints(calibration)
    design_keys = output_fingerprints(design)
    overlap = calibration_keys.keys() & design_keys.keys()
    if overlap:
        item = min(calibration_keys[key] for key in overlap)
        raise ValueError(
            f"design and calibration reuse saved outputs, including {item!r}"
        )
    return {
        "key_column": key_column,
        "sha256": sorted(calibration_keys.keys() | design_keys.keys()),
    }


def check_fit_overlap(rows, fit_outputs, *, cohort):
    identities = output_fingerprints(rows)
    overlap = identities.keys() & set(fit_outputs["sha256"])
    if overlap:
        item = min(identities[key] for key in overlap)
        raise ValueError(
            f"{cohort} reuses output from the fit for sample {item!r} ({len(overlap)} total)"
        )


def _read_json_report(path):
    try:
        return json.loads(path.read_bytes())
    except ValueError as exc:
        raise ValueError(f"{path}: invalid JSON report") from exc


def _finite_number(value):
    try:
        return type(value) in (int, float) and isfinite(value)
    except OverflowError:
        return False


def read_negated_score_source(path, report, score_column):
    inputs = report.get("input")
    source = inputs.get("negated_score_source") if isinstance(inputs, dict) else None
    if source is not None and (
        not isinstance(source, str)
        or not source.strip()
        or score_column != negated_score_name(source)
    ):
        raise ValueError(f"{path}: invalid saved score direction")
    return source


def read_review_missing_scores(path, report):
    inputs = report.get("input")
    value = inputs.get("review_missing_scores") if isinstance(inputs, dict) else None
    if value is not None and type(value) is not bool:
        raise ValueError(f"{path}: invalid missing score option in report")
    return value


def read_fit_outputs(path, inputs, *, artifact="threshold report"):
    fit_outputs = inputs.get("fit_outputs")
    if fit_outputs is not None and (
        not isinstance(fit_outputs, dict)
        or not isinstance(fit_outputs.get("key_column"), str)
        or not fit_outputs["key_column"].strip()
        or not isinstance(fit_outputs.get("sha256"), list)
        or not fit_outputs["sha256"]
        or any(
            not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key)
            for key in fit_outputs["sha256"]
        )
    ):
        raise ValueError(f"{path}: invalid fit output identities in {artifact}")
    return fit_outputs


def read_threshold_report(path, *, require_confirmed=False):
    report = _read_json_report(path)
    error = f"{path}: expected a CAAT select, policy, or run JSON report"
    if not isinstance(report, dict) or report.get("command") not in (
        "select",
        "policy",
        "run",
    ):
        raise ValueError(error)
    command = report["command"]
    try:
        inputs = report["input"]
        policy = report["policy"] if command == "run" else report
        selection = policy if command == "select" else policy["selection"]
        cutoff = selection["selected_threshold"]
        risk = selection["risk_target"]
        confidence = policy.get("holdout_confidence")
        id_column = inputs["id_column"]
        score_column = (
            report["selected_score"] if command == "run" else inputs["score_column"]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(error) from exc
    if cutoff is None:
        raise ValueError(f"{path}: the report did not select a cutoff")
    if (
        require_confirmed
        and command != "select"
        and policy.get("confirmed") is not True
    ):
        raise ValueError(f"{path}: routing requires a confirmed {command} report")
    if not _finite_number(cutoff):
        raise ValueError(f"{path}: selected cutoff must be a finite number")
    if type(risk) not in (int, float) or not 0 <= risk <= 1:
        raise ValueError(f"{path}: invalid risk target in threshold report")
    if confidence is not None and (
        type(confidence) not in (int, float) or not 0 < confidence < 1
    ):
        raise ValueError(f"{path}: invalid holdout confidence in threshold report")
    if not all(
        isinstance(name, str) and name.strip() for name in (id_column, score_column)
    ):
        raise ValueError(
            f"{path}: the threshold report needs ID and score column names"
        )
    fit_outputs = read_fit_outputs(path, inputs)
    output_key_column = (
        fit_outputs["key_column"]
        if fit_outputs is not None
        else inputs.get("output_key_column")
    )
    if output_key_column is not None and (
        not isinstance(output_key_column, str) or not output_key_column.strip()
    ):
        raise ValueError(f"{path}: invalid output key column in threshold report")
    options = {
        "threshold": cutoff,
        "risk_target": risk,
        "holdout_confidence": confidence,
        "id_column": id_column,
        "score_column": score_column,
        "output_key_column": output_key_column,
        "review_missing_scores": read_review_missing_scores(path, report),
        "fit_outputs": fit_outputs,
    }
    if source := read_negated_score_source(path, report, score_column):
        options["negated_score_source"] = source
    return options


def read_label_plan(path):
    report = _read_json_report(path)
    if (
        not isinstance(report, dict)
        or report.get("command") != "plan"
        or not isinstance(report.get("input"), dict)
    ):
        raise ValueError(f"{path}: expected a CAAT plan JSON report")
    labels = report.get("accepted")
    population = report.get("accepted_population")
    risk = report.get("risk_target")
    confidence = report["input"].get("holdout_confidence")
    threshold = report["input"].get("threshold")
    digest = report["input"].get("accepted_population_sha256")
    sample_seed = report["input"].get("sample_seed")
    sample_path = report.get("sample_path")
    if type(labels) is not int or labels < 1:
        raise ValueError(f"{path}: label plan needs a positive accepted label count")
    if population is not None and (type(population) is not int or population < labels):
        raise ValueError(f"{path}: invalid accepted population in label plan")
    if type(risk) not in (int, float) or not 0 <= risk <= 1:
        raise ValueError(f"{path}: invalid risk target in label plan")
    if type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise ValueError(f"{path}: invalid holdout confidence in label plan")
    if threshold is not None and not _finite_number(threshold):
        raise ValueError(f"{path}: invalid cutoff in label plan")
    if digest is not None and (
        population is None
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise ValueError(f"{path}: invalid accepted population digest in label plan")
    if sample_seed is not None and (type(sample_seed) is not int or digest is None):
        raise ValueError(
            f"{path}: invalid sample seed or missing batch digest in label plan"
        )
    return {
        "accepted": labels,
        "accepted_population": population,
        "risk_target": risk,
        "holdout_confidence": confidence,
        "threshold": threshold,
        "accepted_population_sha256": digest,
        "sample_seed": sample_seed,
        "sample_path": sample_path
        if sample_seed is not None and isinstance(sample_path, str)
        else None,
    }


def write_report(path, serialized):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def write_curve(path, curve):
    with path.open("w", encoding="utf-8", newline="") as output:
        fields = ["threshold", "total", "accepted", "errors", "coverage", "error_rate"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for threshold, counts in curve:
            writer.writerow({"threshold": threshold, **counts.metrics()})


def emit(report, *, json_output, output=None):
    from .renderers import render_report

    serialized = render_report("json", report)
    if output is not None:
        write_report(output, serialized)
    print(serialized if json_output else render_report("text", report))
    return serialized


def describe(report):
    command = report["command"]
    if command == "plugins":
        return "\n".join(
            f"{stage}: "
            + ", ".join(f"{entry['name']} ({entry['source']})" for entry in components)
            for stage, components in report["stages"].items()
        )
    if command == "policy":
        return describe_policy(report)
    if command == "run":
        comparison = report["score_comparison"]
        selected = report["selected_score"]
        choice = (
            f"Selected {selected} on {comparison['split']['design']} design rows."
            if comparison["recommended_score"] is not None
            else f"Used the first configured score {selected}."
        )
        if origin := joint_calibration_summary(report["policy"]["selection"]):
            choice = f"{selected}. {origin}"
        queue = report["review_queue"]
        return "\n".join(
            (
                choice,
                describe_policy(report["policy"]),
                f"Queued {len(queue['entries'])} of {queue['eligible']} eligible held-out rows.",
                f"Wrote the HTML report to {report['input']['html']}.",
            )
        )
    if command == "plan":
        inputs = report["input"]
        target = report["risk_target"]
        population = report.get("accepted_population")
        batch = (
            f" The plan assumes {report['expected_population_errors']} errors among "
            f"{population} accepted rows in the fixed batch."
            if population is not None
            else ""
        )
        if report["accepted"] is None:
            limit = inputs["max_accepted"]
            if population is not None:
                limit = min(limit, population)
            return (
                f"No plan meets the requested power at {percent(target)} risk with up to "
                f"{limit} accepted labels.{batch}"
            )
        confidence = percent(inputs["holdout_confidence"])
        chance = percent(report["certification_probability"])
        summary = (
            f"With {report['accepted']} accepted labels, the smallest risk target is {target!r}."
            if "accepted" in inputs
            else f"Label {report['accepted']} accepted rows for the {percent(target)} risk target."
        )
        if sample_path := report.get("sample_path"):
            batch += f" Wrote the seed-{inputs['sample_seed']} sample to {sample_path}."
        return (
            f"{summary} Up to {report['max_errors']} errors certify it at {confidence} confidence. "
            f"The chance of certification is {chance} at the assumed error rate.{batch}"
        )
    if command == "evaluate":
        counts = report["overall"]
        summary = f"Threshold {report['threshold']!r}. {counts_summary(counts)}"
        if counts.get("risk_upper_bound") is not None:
            summary += " " + upper_bound_summary(
                counts, report["holdout_confidence"], report.get("accepted_population")
            )
        if "risk_target" in report:
            verdict = "Meets" if counts["meets_target"] else "Misses"
            summary += f" {verdict} the {percent(report['risk_target'])} error target."
        if "ece" in report:
            summary += "\n" + calibration_summary(report)
        return summary
    if command == "monitor":
        if "anytime_pvalue" in report:
            counts = counts_summary(report["overall"])
            pvalue = report["anytime_pvalue"]
            alert = report["first_alert"]
            verdict = (
                f"First alert at row {alert['row']} ({alert['id']})."
                if alert is not None
                else "No sequential alert."
            )
            target = percent(report["risk_target"])
            summary = (
                f"Frozen threshold {report['threshold']!r} for {target} risk. {counts}\n"
                f"Anytime p-value {pvalue:.4g}. {verdict}"
            )
        else:
            counts = report["overall"]
            summary = (
                f"Frozen threshold {report['threshold']!r}. Accepted {counts['accepted']} "
                f"of {counts['total']} rows ({percent(counts['coverage'])})."
            )
        if stream := report.get("stream"):
            summary += (
                f"\nChecked {report['overall']['total']} of {stream['total']} stream rows. "
                f"The policy accepts {stream['accepted']} ({percent(stream['coverage'])})."
            )
            if pending := stream["first_unlabeled"]:
                summary += (
                    f" {stream['unlabeled_accepted']} accepted rows await labels, starting with "
                    f"row {pending['row']} ({pending['id']})."
                )
        if missing := report["overall"].get("missing_scores"):
            summary += f"\n{missing} stream rows had no score."
        if coverage := report.get("coverage_check"):
            size = coverage["window_size"]
            floor = percent(coverage["min_coverage"])
            if hit := coverage["first_alert"]:
                summary += (
                    f"\nCoverage alert at rows {hit['start_row']} to {hit['end_row']} "
                    f"({hit['id']}). Accepted {hit['accepted']} of {size} "
                    f"({percent(hit['coverage'])}) with a {floor} minimum. "
                    f"{hit['missing_scores']} rows in the window had no score."
                )
            elif checked := coverage["windows_checked"]:
                summary += f"\nCoverage met the {floor} minimum in all {checked} windows of {size} rows."
            else:
                summary += (
                    f"\nThe coverage check needs {size} rows for its first window."
                )
            latest = coverage.get("latest_window")
            if latest is not None and (
                hit is None or latest["end_row"] != hit["end_row"]
            ):
                relation = "meets" if latest["meets_minimum"] else "falls below"
                summary += (
                    f"\nLatest window at rows {latest['start_row']} to {latest['end_row']} "
                    f"accepted {latest['accepted']} of {size} ({percent(latest['coverage'])}) "
                    f"and {relation} the {floor} minimum. "
                    f"{latest['missing_scores']} rows in the window had no score."
                )
        return summary
    if command == "route":
        counts = report["overall"]
        missing = counts.get("missing_scores", 0)
        detail = f" {missing} reviewed rows had no score." if missing else ""
        return (
            f"Accepted {counts['accepted']} of {counts['total']} rows at threshold "
            f"{report['threshold']!r}. Marked {counts['review']} for review."
            f"{detail} Wrote {report['decision_path']}."
        )
    if command == "queue":
        if "sample_path" in report:
            lines = [
                f"Sampled {report['budget']} of {report['total']} rows with seed "
                f"{report['random_baseline']['seed']}. Wrote {report['sample_path']}."
            ]
            lines.extend(
                f"{grader}: the sample contained {counts['random_errors_found']} errors."
                for grader, counts in report["graders"].items()
            )
            return "\n".join(lines)
        lines = [
            f"Queued {report['budget']} of {report['total']} rows in order of increasing confidence."
        ]
        for grader, counts in report["graders"].items():
            lines.append(
                f"{grader}: found {counts['errors_found']} errors. "
                f"The random queue found {counts['random_errors_found']}."
            )
        return "\n".join(lines)
    if command == "scores":
        recommended = report["recommended_score"]
        lines = [
            f"Compared scores on {report['split']['reported']} rows against {report['baseline']}."
        ]
        if report["split"]["comparison"] == "heldout":
            if recommended:
                lines.append(f"Design-only recommendation: {recommended}.")
            elif "design_scored" in report:
                lines.append("No complete design score improved on random by AURC.")
            else:
                lines.append("No score improved on random by design AURC.")
        for grader, comparison in report["graders"].items():
            lines.append(f"{grader}:")
            controls = [
                (f"control {key}", value, "")
                for key, value in comparison["controls"].items()
            ]
            candidates = [
                (name, value, f" ({report['costs'][name]})")
                for name, value in comparison["scores"].items()
            ]
            for name, score, cost in controls + candidates:
                auc = score["auroc"]
                roc = f"{auc:.3f}" if auc is not None else "undefined"
                risks = ", ".join(
                    f"{percent(entry['target_coverage'])} coverage: "
                    + (
                        f"{percent(entry['error_rate'])} error"
                        if entry["error_rate"] is not None
                        else "unreachable"
                    )
                    for entry in score["budgets"]
                )
                availability = (
                    f" Scored {score['scored']} of {report['split']['reported']} rows."
                    if "scored" in score
                    else ""
                )
                warning = (
                    " AUROC no better than random."
                    if score.get("direction_warning")
                    else ""
                )
                lines.append(
                    f"  {name}{cost}. AUROC {roc}.{availability} {risks}.{warning}"
                )
        return "\n".join(lines)
    if command == "audit":
        lines = [f"Threshold {report['threshold']!r}."]
        for grader, counts in report["graders"].items():
            details = counts_summary(counts)
            if counts.get("risk_upper_bound") is not None:
                details += " " + upper_bound_summary(
                    counts,
                    report["holdout_confidence"],
                    report.get("accepted_population"),
                )
            verdict = ""
            if report["risk_target"] is not None:
                verdict = (
                    " Meets target." if counts["meets_target"] else " Target not met."
                )
            lines.append(f"{grader}: {details}{verdict}")
        lines.append(f"{len(report['discordant'])} outputs have differing outcomes.")
        if path := report["input"].get("disagreements_output"):
            lines.append(f"Wrote the disagreements to {path}.")
        return "\n".join(lines)
    threshold = report["selected_threshold"]
    if threshold is None:
        return f"No candidate threshold met the {percent(report['risk_target'])} target ({report['rule']})."
    candidate = next(
        row for row in report["candidates"] if row["threshold"] == threshold
    )
    return (
        f"Selected threshold {threshold!r} ({report['rule']}). "
        f"Calibration accepted {candidate['accepted']} of {report['calibration_total']} rows."
    )


def counts_summary(counts):
    rate = counts["error_rate"]
    rate_text = "undefined risk" if rate is None else f"{percent(rate)} error"
    return f"Accepted {counts['accepted']} of {counts['total']} rows with {counts['errors']} errors ({rate_text})."


def percent(value):
    return f"{100 * value:.3g}%"


def upper_bound_summary(counts, confidence, population=None):
    upper = percent(counts["risk_upper_bound"])
    if population is not None:
        return (
            f"The one-sided {percent(confidence)} upper error bound for the fixed batch "
            f"is {upper} ({counts['risk_upper_errors']} errors among {population} accepted rows)."
        )
    return f"The one-sided {percent(confidence)} upper error bound is {upper}."


def joint_calibration_summary(selection):
    size = selection.get("candidate_family_size")
    if size is None:
        return None
    if selection["selected_threshold"] is None:
        return (
            f"No cutoff qualified among {size} score-cutoff pairs on calibration rows."
        )
    if selection["rule"] == "ltt-holm":
        return f"Selected from {size} score-cutoff pairs on calibration rows with Holm's step-down test."
    return (
        f"Selected on calibration from {size} score-cutoff pairs "
        f"at a per-pair level of {selection['per_candidate_level']:.4g}."
    )


def calibration_summary(metrics):
    if metrics["brier_score"] is None:
        return "No scored rows for calibration metrics."
    return (
        f"Brier {metrics['brier_score']:.4f}. "
        f"ECE {percent(metrics['ece'])}. ACE {percent(metrics['ace'])}."
    )


def describe_policy(report) -> str:
    """Summarize a policy report in plain text."""
    split, selection = report["split"], report["selection"]
    source = (
        f"Split with seed {split['seed']}: "
        if split["seed"] is not None
        else "Provided split has "
    )
    lines = [
        f"{source}{split['design']} design, "
        f"{split['calibration']} calibration, and {split['heldout']} held-out rows."
    ]
    if scored := report.get("scored"):
        lines.append(
            f"Scored {scored['design']} design, {scored['calibration']} calibration, "
            f"and {scored['heldout']} held-out rows."
        )
    for name, ranking in report["ranking"].items():
        auroc = ranking["auroc"]
        if auroc is None and "scored" in report:
            lines.append(f"AUROC unavailable for scored calibration rows under {name}.")
        elif auroc is None:
            lines.append(
                f"Score ranking under {name}: undefined, every calibration row has the same outcome."
            )
        elif auroc <= 0.5:
            lines.append(
                f"Score ranking under {name}: AUROC {auroc:.3f}, no better than random on the "
                "calibration rows. Larger scores must indicate greater confidence."
            )
        else:
            lines.append(f"Score ranking under {name}: AUROC {auroc:.3f}.")
    target, rule = percent(selection["risk_target"]), selection["rule"]
    graders = " and ".join(selection["graders"])
    threshold = selection["selected_threshold"]
    if "probability_calibration" in report:
        calibration = report["probability_calibration"]
        lines.append(
            f"Probability calibration uses {calibration['method']}. "
            "The policy threshold uses the raw score."
        )
        for name, estimates in calibration["graders"].items():
            before = estimates["heldout_uncalibrated"]
            if before is not None:
                lines.append(
                    f"Held-out raw probabilities under {name}. {calibration_summary(before)}"
                )
            lines.append(
                f"Held-out calibrated probabilities under {name}. "
                + calibration_summary(estimates["heldout_calibrated"])
            )
    if threshold is None:
        lines.append(
            f"No candidate threshold met the {target} risk target under {graders} ({rule})."
        )
        return "\n".join(lines)
    lines.append(
        f"Selected threshold {threshold!r} for a {target} risk target under {graders} ({rule})."
    )
    for name, result in report["heldout"]["graders"].items():
        role = " (check only)" if name in report["check_graders"] else ""
        if result["error_rate"] is None:
            lines.append(f"Held-out rows under {name}{role}: none accepted.")
            continue
        verdict = "meets" if result["meets_target"] else "exceeds"
        check = f", which {verdict} the target."
        if result.get("risk_upper_bound") is not None:
            bound = upper_bound_summary(result, report["holdout_confidence"])
            check = f". {bound} The bound {verdict} the target."
        lines.append(
            f"Held-out rows under {name}{role}: accepted {result['accepted']} of {result['total']} "
            f"with {result['errors']} errors ({percent(result['error_rate'])}){check}"
        )
    lines.append(
        "The held-out rows confirm the policy."
        if report["confirmed"]
        else "The held-out rows do not confirm the policy."
    )
    return "\n".join(lines)
