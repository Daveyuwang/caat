"""Render self-contained reports for offline review."""

from html import escape

from .plots import grader_comparison, reliability_diagram, risk_coverage
from .report import describe, joint_calibration_summary, percent


def table(headers, rows, *, scroll_label=None):
    head = "".join(f'<th scope="col">{escape(header)}</th>' for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    region = (
        f' role="region" aria-label="{escape(scroll_label)}" tabindex="0"'
        if scroll_label
        else ""
    )
    return f'<div class="scroll"{region}><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def cutoff_table(selection, *, scroll_label=None):
    """Tabulate the calibration candidates with the test each had to pass."""
    if selection["rule"] == "empirical":
        test = "its calibration error is at or below the target"
    elif level := selection.get("per_candidate_level"):
        test = f"its p-value is at or below {level:.3g}"
    else:
        test = (
            "its p-value passes the step-down test at a "
            f"{percent(selection['failure_rate'])} failure rate"
        )
    note = (
        f"<p>Counts are from the calibration rows. A candidate passes when {test}. "
        "A strict cutoff that accepts few rows can fail while a looser one passes.</p>"
    )
    return note + table(
        (
            "Cutoff",
            "Accepted",
            "Coverage",
            *(f"Errors ({name})" for name in selection["graders"]),
            "p-value",
            "Decision",
        ),
        (
            (
                candidate["threshold"],
                candidate["accepted"],
                percent(candidate["coverage"]),
                *(candidate["errors"][key] for key in selection["graders"]),
                f"{candidate['pvalue']:.4g}",
                {None: "Untested", True: "Pass", False: "Fail"}[candidate["passes"]],
            )
            for candidate in selection["candidates"]
        ),
        scroll_label=scroll_label,
    )


def workflow_holdout(policy, sections, add_chart):
    sections.append("<h2>Held-out check</h2>")
    if policy["heldout"] is None:
        sections.append("<p>The held-out check needs a qualifying cutoff.</p>")
        return
    results = policy["heldout"]["graders"]
    counts = next(iter(results.values()))
    target = percent(policy["selection"]["risk_target"])
    verdict = (
        "Held-out check passed." if policy["confirmed"] else "Held-out check failed."
    )
    sections.append(
        f"<p><strong>{verdict}</strong> Accepted {counts['accepted']} of {counts['total']} "
        f"({percent(counts['coverage'])}) at a {target} risk target.</p>"
    )
    confidence = percent(policy["holdout_confidence"])
    sections.append('<div class="outcomes">')
    sections.append(
        table(
            ("Grader", "Errors (rate)", f"{confidence} upper bound", "Status"),
            (
                (
                    name,
                    f"{entry['errors']} ({entry['error_rate']:.2%})"
                    if entry["error_rate"] is not None
                    else "Undefined",
                    f"{entry['risk_upper_bound']:.2%}"
                    if entry["risk_upper_bound"] is not None
                    else "Undefined",
                    "Pass" if entry["meets_target"] else "Fail",
                )
                for name, entry in results.items()
            ),
        )
    )
    sections.append("</div>")
    add_chart(grader_comparison({**policy, "command": "policy"}))


def workflow_calibration(policy, sections, add_chart):
    sections.append("<h2>Probability calibration</h2>")
    calibration = policy.get("probability_calibration")
    if calibration is None:
        sections.append(
            "<p>No scored calibration rows for fitting a probability map.</p>"
        )
        return
    fitted = policy.get("scored", policy["split"])["calibration"]
    evaluation_rows = "scored held-out rows" if "scored" in policy else "held-out rows"
    sections.append(
        f"<p>Fitted {escape(calibration['method'])} on {fitted} calibration rows. "
        f"Metrics use {evaluation_rows}. The acceptance cutoff uses the raw score.</p>"
    )
    estimates = {
        grader: result["heldout_calibrated"]
        for grader, result in calibration["graders"].items()
    }
    sections.append(
        table(
            ("Grader", "Brier", "ECE", "ACE"),
            (
                (
                    grader,
                    f"{value['brier_score']:.4f}",
                    percent(value["ece"]),
                    percent(value["ace"]),
                )
                if value["brier_score"] is not None
                else (grader, "Undefined", "Undefined", "Undefined")
                for grader, value in estimates.items()
            ),
        )
    )
    for name, metrics in estimates.items():
        if metrics["reliability_bins"]:
            add_chart(reliability_diagram(metrics["reliability_bins"]), name)


def workflow_sections(report, sections, add_chart):
    policy = report["policy"]
    workflow_holdout(policy, sections, add_chart)
    comparison = report["score_comparison"]
    selected = escape(report["selected_score"])
    origin = joint_calibration_summary(policy["selection"]) or (
        f"Selected on {policy['split']['design']} design rows."
        if comparison["recommended_score"] is not None
        else "First score in the configuration."
    )
    sections.extend(
        (
            "<h2>Score choice</h2>",
            f"<p><code>{selected}</code>. {origin}</p>",
            table(
                ("Candidate score", "Design normalized AURC gain"),
                (
                    (name, f"{gain:.4f}" if gain is not None else "Undefined")
                    for name, gain in comparison["design_normalized_aurc_gain"].items()
                ),
            ),
        )
    )
    workflow_calibration(policy, sections, add_chart)

    selection = policy["selection"]
    cutoff = selection["selected_threshold"]
    rule = escape(selection["rule"])
    target = percent(selection["risk_target"])
    sections.append("<h2>Acceptance cutoff</h2>")
    sections.append(
        f"<p>Cutoff <code>{cutoff!r}</code> at a {target} risk target using {rule}.</p>"
        if cutoff is not None
        else f"<p>Zero cutoffs qualified at a {target} risk target using {rule}.</p>"
    )
    sections.extend(
        (
            "<details><summary>Calibration candidates</summary>",
            cutoff_table(selection, scroll_label="Calibration candidates"),
            "</details>",
        )
    )
    queue = report["review_queue"]
    entries = queue["entries"]
    queue_summary = (
        f"Queued {len(entries)} of {queue['eligible']} eligible held-out rows"
    )
    if report["input"].get("review_missing_scores"):
        queue_summary += (
            ". Unscored rows come first, followed by increasing confidence."
        )
    else:
        queue_summary += " in order of increasing confidence."
    sections.extend(("<h2>Review queue</h2>", f"<p>{queue_summary}</p>"))
    if not policy["confirmed"]:
        sections.append("<p>All held-out rows are eligible for review.</p>")
    if entries:
        sections.append(
            table(
                ("ID", "Raw score"),
                (
                    (
                        row["id"],
                        row["score"] if row["score"] is not None else "Unscored",
                    )
                    for row in entries
                ),
            )
        )


def render_html(report, curve=None, curves=None):
    command = report["command"]
    sections = [] if command == "run" else [f"<pre>{escape(describe(report))}</pre>"]
    charts = 0

    def add_chart(svg, heading=None):
        nonlocal charts
        charts += 1
        title_id = f"chart-{charts}"
        if heading is not None:
            sections.append(f"<h3>{escape(heading)}</h3>")
        sections.append(
            svg.replace('id="title"', f'id="{title_id}"', 1).replace(
                'aria-labelledby="title"', f'aria-labelledby="{title_id}"', 1
            )
        )

    if command == "run":
        workflow_sections(report, sections, add_chart)
    elif command == "evaluate":
        add_chart(risk_coverage(report, curve or ()))
        if "reliability_bins" in report:
            add_chart(reliability_diagram(report["reliability_bins"]))
            sections.append(
                table(
                    (
                        "Probability range",
                        "Rows",
                        "Mean probability",
                        "Observed correctness",
                    ),
                    (
                        (
                            f"{row['lower']:.0%} to {row['upper']:.0%}",
                            row["count"],
                            percent(row["mean_probability"]) if row["count"] else "",
                            percent(row["accuracy"]) if row["count"] else "",
                        )
                        for row in report["reliability_bins"]
                    ),
                )
            )
    else:
        add_chart(grader_comparison(report))
        if command == "policy":
            sections.append("<h2>Calibration candidates</h2>")
            sections.append(cutoff_table(report["selection"]))
            calibration = report.get("probability_calibration")
            if calibration:
                sections.append("<h2>Held-out calibration</h2>")
                for grader, estimates in calibration["graders"].items():
                    add_chart(
                        reliability_diagram(
                            estimates["heldout_calibrated"]["reliability_bins"]
                        ),
                        grader,
                    )
        if curves:
            title = (
                "Held-out risk and coverage"
                if command == "policy"
                else "Risk and coverage by grader"
            )
            sections.append(f"<h2>{title}</h2>")
            for name, (view, points) in curves.items():
                add_chart(risk_coverage(view, points), name)

    title = "CAAT review workflow" if command == "run" else f"CAAT {command} report"
    main_class = ' class="workflow"' if command == "run" else ""
    inputs = report.get("input") or {}
    if command != "run" and inputs.get("path"):
        source = f"Input: {inputs['path']}"
        if inputs.get("score_column"):
            source += f". Score column: {inputs['score_column']}"
        sections.insert(
            0, f"<p>{escape(source)}. CAAT {escape(report['caat_version'])}.</p>"
        )
    body = "\n".join(sections)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{color:#172b38;font:16px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:52rem;padding:0 1rem}}
h1 {{font-size:2rem}}
h2 {{font-size:1.5rem}}
h3 {{font-size:1.17rem}}
pre {{background:#f0f5f7;padding:1rem;white-space:pre-wrap;overflow-wrap:anywhere}}
svg {{display:block;width:100%;max-width:670px;height:auto}}
.scroll {{overflow-x:auto}}
table {{border-collapse:collapse;width:100%}}
td,th {{border-bottom:1px solid #dbe4e9;padding:.4rem .6rem;text-align:left}}
.workflow p {{max-width:70ch;overflow-wrap:anywhere}}
.workflow td {{font-variant-numeric:tabular-nums}}
.workflow summary {{font-weight:600;cursor:pointer}}
.workflow :focus-visible {{outline:2px solid #0072b2;outline-offset:2px}}
@media (max-width:30rem) {{
  .workflow th,.workflow td {{padding-inline:.35rem}}
  .workflow .outcomes th,.workflow .outcomes td {{overflow-wrap:anywhere}}
  .workflow .outcomes th:last-child {{white-space:nowrap}}
}}
</style>
</head>
<body><main{main_class}>
<h1>{title}</h1>
{body}
</main></body>
</html>
"""
