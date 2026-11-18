"""Report formats for the CLI and entry-point plugins."""

import json
from dataclasses import dataclass
from typing import Callable

from .html_report import render_html
from .plots import render_plot
from .registry import Registry
from .report import describe


@dataclass(frozen=True)
class Renderer:
    render: Callable
    format: str
    commands: tuple[str, ...]


class RenderReport(dict):
    def __init__(self, report, curve, curves):
        super().__init__(report)
        self.curve = curve
        self.curves = curves


ALL_REPORTS = (
    "evaluate",
    "audit",
    "audit-block",
    "select",
    "policy",
    "run",
    "plan",
    "two-stage-plan",
    "queue",
    "route",
    "scores",
    "monitor",
    "plugins",
)
PLOTTED_REPORTS = ("evaluate", "audit", "policy")
RENDERERS = Registry(
    "renderers",
    Renderer,
    {
        "text": Renderer(describe, "text", ALL_REPORTS),
        "json": Renderer(
            lambda report: json.dumps(report, indent=2, allow_nan=False),
            "json",
            ALL_REPORTS,
        ),
        "svg": Renderer(
            lambda report: render_plot(report, getattr(report, "curve", None)),
            "svg",
            PLOTTED_REPORTS,
        ),
        "html": Renderer(
            lambda report: render_html(
                report, getattr(report, "curve", None), getattr(report, "curves", None)
            ),
            "html",
            (*PLOTTED_REPORTS, "run"),
        ),
    },
)


def render_report(name, report, *, curve=None, curves=None):
    renderer = RENDERERS[name]
    if report["command"] not in renderer.commands:
        raise ValueError(
            f"renderer {name!r} does not support {report['command']!r} reports"
        )
    data = renderer.render(RenderReport(report, curve, curves))
    if not isinstance(data, str):
        raise ValueError(f"renderer {name!r} must return text")
    return data
