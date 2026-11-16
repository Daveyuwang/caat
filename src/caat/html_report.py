"""Render self-contained reports for offline review."""

from html import escape

from .plots import grader_comparison, reliability_diagram, risk_coverage
from .report import describe, percent


def table(headers, rows):
    head = "".join(f'<th scope="col">{escape(header)}</th>' for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_html(report, curve=None, curves=None):
    command = report["command"]
    sections = [f"<pre>{escape(describe(report))}</pre>"]
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

    if command == "evaluate":
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
            selection = report["selection"]
            sections.append(
                table(
                    (
                        "Cutoff",
                        "Accepted",
                        "Coverage",
                        *selection["graders"],
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
                            {None: "Untested", True: "Pass", False: "Fail"}[
                                candidate["passes"]
                            ],
                        )
                        for candidate in selection["candidates"]
                    ),
                )
            )
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

    title = f"CAAT {command} report"
    body = "\n".join(sections)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{color:#172b38;font:16px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:52rem;padding:0 1rem}}
pre {{background:#f0f5f7;padding:1rem;white-space:pre-wrap;overflow-wrap:anywhere}}
svg {{display:block;width:100%;max-width:670px;height:auto}}
.scroll {{overflow-x:auto}}
table {{border-collapse:collapse;width:100%}}
td,th {{border-bottom:1px solid #dbe4e9;padding:.4rem .6rem;text-align:left}}
</style>
</head>
<body><main>
<h1>{title}</h1>
{body}
</main></body>
</html>
"""
