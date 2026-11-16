from random import Random

from .audit import AuditRow, audit, validated_rows
from .evaluation import Counts, ranking_metrics, threshold_points
from .selection import design_thresholds, select_threshold


def split_rows(rows, seed):
    """Sort by ID, shuffle, and split about 20/40/40 into design, calibration, and held-out rows."""
    rows = sorted(rows, key=lambda row: row.id)
    Random(seed).shuffle(rows)
    first, second = len(rows) // 5, 3 * len(rows) // 5
    return rows[:first], rows[first:second], rows[second:]


def build_policy(
    rows, *, graders, check_graders=(), risk_target, failure_rate=0.05, rule="ltt-bonferroni", seed=42
) -> dict:
    """Select a cutoff on calibration rows and replay it unchanged on held-out rows.

    Each row holds one outcome per grader, selection graders first. The check
    graders take no part in selection and are reported on the held-out rows.
    """
    everyone = (*graders, *check_graders)
    rows = validated_rows(rows, everyone)
    if not graders:
        raise ValueError("provide at least one grader for selection")
    if len(rows) < 5:
        raise ValueError("at least five rows are required for the design, calibration, and held-out splits")
    design, calibration, heldout = split_rows(rows, seed)
    ranking = {
        name: ranking_metrics(threshold_points((row.score, not row.outcomes[index]) for row in calibration))
        for index, name in enumerate(everyone)
    }
    selection = select_threshold(
        (AuditRow(row.id, row.score, row.outcomes[:len(graders)]) for row in calibration),
        graders=graders, thresholds=design_thresholds(row.score for row in design),
        risk_target=risk_target, failure_rate=failure_rate, rule=rule,
    )
    threshold = selection["selected_threshold"]
    check = None
    if threshold is not None:
        accepted = [row for row in heldout if row.score >= threshold]
        results = {}
        for index, name in enumerate(everyone):
            counts = Counts(len(heldout), len(accepted), sum(not row.outcomes[index] for row in accepted))
            risk = counts.error_rate
            results[name] = {
                **counts.metrics(),
                "meets_target": risk <= risk_target if risk is not None else None,
            }
        check = {"graders": results}
        if len(everyone) > 1:
            check["accepted_comparisons"] = audit(
                heldout, graders=everyone, threshold=threshold, risk_target=risk_target
            )["accepted_comparisons"]
    return {
        "schema_version": 1,
        "split": {"seed": seed, "design": len(design), "calibration": len(calibration), "heldout": len(heldout)},
        "check_graders": list(check_graders),
        "ranking": ranking,
        "selection": selection,
        "heldout": check,
        "confirmed": check is not None and all(result["meets_target"] is True for result in check["graders"].values()),
    }


def percent(value):
    return f"{100 * value:.3g}%"


def describe(report) -> str:
    """Summarize a policy report in plain text."""
    split, selection = report["split"], report["selection"]
    lines = [
        f"Split with seed {split['seed']}: {split['design']} design, "
        f"{split['calibration']} calibration, and {split['heldout']} held-out rows."
    ]
    for name, ranking in report["ranking"].items():
        auroc = ranking["auroc"]
        if auroc is None:
            lines.append(f"Score ranking under {name}: undefined, every calibration row has the same outcome.")
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
    if threshold is None:
        lines.append(f"No candidate threshold met the {target} risk target under {graders} ({rule}).")
        return "\n".join(lines)
    lines.append(f"Selected threshold {threshold!r} for a {target} risk target under {graders} ({rule}).")
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
        "The held-out rows confirm the policy." if report["confirmed"]
        else "The held-out rows do not confirm the policy."
    )
    return "\n".join(lines)
