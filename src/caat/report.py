"""Render versioned CLI reports."""

import csv
import json

from importlib.metadata import version


SCHEMA_VERSION = 2


def make_report(command, input_meta, contents):
    return {
        **contents,
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "caat_version": version("caat-annotation"),
        "input": input_meta,
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
    serialized = json.dumps(report, indent=2)
    if output is not None:
        write_report(output, serialized)
    print(serialized if json_output else describe(report))
    return serialized


def describe(report):
    command = report["command"]
    if command == "policy":
        return describe_policy(report)
    if command == "evaluate":
        summary = (
            f"Threshold {report['threshold']!r}. {counts_summary(report['overall'])}"
        )
        if "ece" in report:
            summary += "\n" + calibration_summary(report)
        return summary
    if command == "queue":
        lines = [
            f"Queued {report['budget']} of {report['total']} rows in order of increasing confidence."
        ]
        for grader, counts in report["graders"].items():
            lines.append(
                f"{grader}: found {counts['errors_found']} errors. "
                f"The random queue found {counts['random_errors_found']}."
            )
        return "\n".join(lines)
    if command == "audit":
        lines = [f"Threshold {report['threshold']!r}."]
        for grader, counts in report["graders"].items():
            verdict = ""
            if report["risk_target"] is not None:
                verdict = (
                    " Meets target." if counts["meets_target"] else " Target not met."
                )
            lines.append(f"{grader}: {counts_summary(counts)}{verdict}")
        lines.append(f"{len(report['discordant'])} outputs have differing outcomes.")
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


def calibration_summary(metrics):
    return (
        f"Brier {metrics['brier_score']:.4f}. "
        f"ECE {percent(metrics['ece'])}. ACE {percent(metrics['ace'])}."
    )


def describe_policy(report) -> str:
    """Summarize a policy report in plain text."""
    split, selection = report["split"], report["selection"]
    lines = [
        f"Split with seed {split['seed']}: {split['design']} design, "
        f"{split['calibration']} calibration, and {split['heldout']} held-out rows."
    ]
    for name, ranking in report["ranking"].items():
        auroc = ranking["auroc"]
        if auroc is None:
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
        lines.append(
            f"Held-out rows under {name}{role}: accepted {result['accepted']} of {result['total']} "
            f"with {result['errors']} errors ({percent(result['error_rate'])}), which {verdict} the target."
        )
    lines.append(
        "The held-out rows confirm the policy."
        if report["confirmed"]
        else "The held-out rows do not confirm the policy."
    )
    return "\n".join(lines)
