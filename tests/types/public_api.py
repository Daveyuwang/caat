"""Type-check a caller against the installed CAAT package."""

from pathlib import Path
from typing import Any

from typing_extensions import assert_type

from caat import GradedRow, audit, evaluate, load_csv, policy, select
from caat.metrics import Counts


def use_public_api(path: Path, paired: list[GradedRow]) -> None:
    rows = assert_type(load_csv(path, correct_columns=("correct",)), list[GradedRow])
    metrics, curve = evaluate(rows, threshold=0.5, extra_metrics=["ranking"])
    assert_type(metrics, dict[str, Any])
    assert_type(curve, list[tuple[float, Counts]])
    assert_type(curve[0][1].error_rate, float | None)
    assert_type(
        select(rows, graders=["correct"], thresholds=(0.5,), risk_target=0.1),
        dict[str, Any],
    )
    assert_type(
        policy(rows, graders=("correct",), risk_target=0.1, design_grid_size=20),
        dict[str, Any],
    )
    assert_type(
        audit(paired, graders=["original", "replacement"], threshold=0.5),
        dict[str, Any],
    )


def rejects_invalid_inputs(rows: list[GradedRow]) -> None:
    # Strict mypy rejects unused ignores if these parameters lose their types.
    load_csv(42)  # type: ignore[arg-type]
    evaluate(rows, threshold="0.5")  # type: ignore[arg-type]
    policy(rows, graders=("correct",), risk_target="0.1")  # type: ignore[arg-type]
    policy(rows, graders=("correct",), risk_target=0.1, design_grid_size="20")  # type: ignore[arg-type]
    select(rows, graders=["correct"], thresholds=0.5, risk_target=0.1)  # type: ignore[arg-type]
    audit(rows, graders=None, threshold=0.5)  # type: ignore[arg-type]
