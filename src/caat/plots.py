"""Write dependency-free SVG views of CAAT reports."""

from html import escape
from math import ceil


INK = "#172b38"
MUTED = "#496170"
GRID = "#dbe4e9"
BLUE = "#0072b2"
ORANGE = "#b54e00"


def label(x, y, value, *, size=13, anchor="start", color=INK, weight="400"):
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>'
    )


def line(x1, y1, x2, y2, color=GRID, dashed=False):
    style = ' stroke-dasharray="5 4"' if dashed else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}"{style}/>'


def canvas(width, height, title, marks):
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">',
            f'<title id="title">{escape(title)}</title>',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            '<g font-family="system-ui, sans-serif">',
            *marks,
            "</g>",
            "</svg>",
            "",
        ]
    )


def risk_coverage(report, curve):
    width, height = 670, 350
    left, right, upper, lower = 60, 642, 68, 285

    def x(value):
        return round(left + (right - left) * value, 2)

    def y(value):
        return round(lower - (lower - upper) * value, 2)

    threshold = report["threshold"]
    marks = [
        label(left, 27, "Risk and coverage", size=18, weight="650"),
    ]
    title = "Risk and coverage"
    if threshold is not None:
        title = f"Risk and coverage at cutoff {threshold:.8g}"
        marks.append(label(left, 47, f"Selected cutoff {threshold:.8g}", color=MUTED))
    for value in (0, 0.25, 0.5, 0.75, 1):
        marks.extend(
            [
                line(left, y(value), right, y(value)),
                label(
                    left - 10, y(value) + 4, f"{value:.0%}", anchor="end", color=MUTED
                ),
                label(
                    x(value), lower + 20, f"{value:.0%}", anchor="middle", color=MUTED
                ),
            ]
        )
    marks.extend(
        [
            label(left, upper - 8, "Accepted error rate", size=12, color=MUTED),
            label(
                (left + right) / 2,
                lower + 44,
                "Coverage",
                size=12,
                anchor="middle",
                color=MUTED,
            ),
        ]
    )
    points = " ".join(
        f"{x(counts.coverage)},{y(counts.error_rate)}" for _, counts in curve
    )
    marks.append(
        f'<polyline points="{points}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>'
    )
    overall = report["overall"]
    if threshold is not None:
        selected = x(overall["coverage"])
        marks.append(line(selected, upper, selected, lower, ORANGE, dashed=True))
        if overall["error_rate"] is not None:
            marks.append(
                f'<circle cx="{selected}" cy="{y(overall["error_rate"])}" r="5" fill="{ORANGE}"/>'
            )
    return canvas(width, height, title, marks)


def reliability_diagram(bins):
    title = "Reliability diagram"
    left, top, side = 64, 81, 240
    right, bottom = left + side, top + side
    occupied = [row for row in bins if row["count"]]
    rows = sum(row["count"] for row in occupied)
    marks = [
        label(20, 27, title, size=18, weight="650"),
        label(20, 48, f"{rows:,} rows in {len(occupied)} occupied bins", color=MUTED),
        label(left, top - 11, "Observed correctness", color=MUTED),
        label(
            left + side / 2,
            bottom + 45,
            "Mean predicted probability",
            anchor="middle",
            color=MUTED,
        ),
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1):
        x, y = left + side * tick, bottom - side * tick
        marks.extend(
            [
                line(left, y, right, y),
                line(x, top, x, bottom),
                label(left - 10, y + 4, f"{tick:.0%}", anchor="end", color=MUTED),
                label(x, bottom + 20, f"{tick:.0%}", anchor="middle", color=MUTED),
            ]
        )
    marks.append(line(left, bottom, right, top, MUTED, dashed=True))
    for row in occupied:
        probability, accuracy = row["mean_probability"], row["accuracy"]
        x, y = left + side * probability, bottom - side * accuracy
        description = f"Bin size {row['count']}, mean probability {probability:.2%}, observed correctness {accuracy:.2%}"
        marks.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{BLUE}" stroke="white">'
            f"<title>{description}</title></circle>"
        )
    marks.extend(
        [
            line(104, bottom + 71, 129, bottom + 71, MUTED, dashed=True),
            label(136, bottom + 75, "Perfect agreement", size=12, color=MUTED),
        ]
    )
    return canvas(350, bottom + 92, title, marks)


def grader_comparison(report):
    if report["command"] == "policy":
        threshold = report["selection"]["selected_threshold"]
        target = report["selection"]["risk_target"]
        graders = report["heldout"]["graders"] if report["heldout"] else {}
        title = "Held-out risk by grader"
    else:
        threshold = report["threshold"]
        target = report["risk_target"]
        graders = report["graders"]
        title = "Accepted error rate by grader"
    confidence = report.get("holdout_confidence")
    rate_key = "error_rate"
    if confidence is not None:
        title = f"{confidence * 100:.4g}% upper error bound by grader"
        rate_key = "risk_upper_bound"
    width, left, right, top = 690, 186, 540, 85
    axis = top + max(1, len(graders)) * 36
    marks = [label(20, 28, title, size=18, weight="650")]
    if threshold is None:
        marks.append(
            label(20, 56, "No candidate threshold met the target.", color=MUTED)
        )
        return canvas(width, 76, title, marks)
    marks.append(label(20, 49, f"Cutoff {threshold:.8g}", color=MUTED))
    observed = max([target or 0, *(entry[rate_key] or 0 for entry in graders.values())])
    maximum = min(1.0, ceil(max(0.25, 1.1 * observed) * 10) / 10)

    def x(value):
        return round(left + (right - left) * value / maximum, 2)

    marks.extend(
        [
            label(20, top - 13, "Grader", color=MUTED),
            label(width - 20, top - 13, "Errors / accepted", color=MUTED, anchor="end"),
        ]
    )
    for tick in (0, maximum / 2, maximum):
        marks.append(line(x(tick), top, x(tick), axis))
        marks.append(
            label(x(tick), axis + 19, f"{tick:.0%}", anchor="middle", color=MUTED)
        )
    for index, (name, counts) in enumerate(graders.items()):
        center = top + 36 * index + 15
        rate = counts[rate_key]
        display_name = name if len(name) <= 21 else name[:19] + "…"
        marks.append(
            f"<g><title>{escape(name)}</title>{label(20, center + 4, display_name)}</g>"
        )
        marks.append(
            label(
                width - 20,
                center + 4,
                f"{counts['errors']} / {counts['accepted']}",
                anchor="end",
            )
        )
        if rate is None:
            marks.append(label(left + 8, center + 4, "No accepted rows", color=MUTED))
            continue
        marks.append(line(left, center, right, center))
        marks.append(
            f'<circle cx="{x(rate)}" cy="{center}" r="5.5" fill="{BLUE}" stroke="white"/>'
        )
        marks.append(
            label(
                x(rate),
                center - 8,
                f"{rate:.2%}",
                size=12,
                anchor="middle",
                weight="600",
            )
        )
    if target is not None:
        marks.extend(
            [
                line(x(target), top - 3, x(target), axis, ORANGE, dashed=True),
                label(
                    x(target),
                    axis + 39,
                    f"Target {target * 100:.4g}%",
                    anchor="middle",
                    color=ORANGE,
                ),
            ]
        )
    return canvas(width, axis + (55 if target is not None else 32), title, marks)


def render_plot(report, curve=None):
    return (
        risk_coverage(report, curve or ())
        if report["command"] == "evaluate"
        else grader_comparison(report)
    )


def write_svg(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8", newline="\n")
