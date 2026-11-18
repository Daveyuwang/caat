"""Freeze the HellaSwag design, run the three handoffs and replay their results."""

import argparse
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from caat.finite import hypergeometric_cdf
from caat.plan import forecast_block_cost, forecast_verdict, sample_error_limits
from common import ROOT, read_csv, read_json, write_report
from collect_hellaswag import CHOICES, OUT, dataset_file
from study_batch_audit import command, write_json
from study_winogrande import drive, keyed

POLICY = ROOT / "results/anli_r2/policy.json"
BATCH = OUT / "predictions.csv"
ARGS = ("--threshold-report", POLICY, "--batch-predictions", BATCH)
RATES = (0.15, 0.25)
SCENARIOS = ("--error-scenarios", *RATES, "--power", 0.8)


def inputs(batch=BATCH, manifest_path=OUT / "manifest.csv"):
    manifest = keyed(read_csv(manifest_path))
    rows = keyed(read_csv(batch))
    if manifest.keys() != rows.keys():
        raise ValueError("scored IDs differ from the frozen sources")
    for uid, r in rows.items():
        scores = {c: float(r[c]) for c in CHOICES}
        choice = max(scores, key=scores.get)
        a, b = sorted(scores.values(), reverse=True)[:2]
        if (r["answer"], r["output_sha256"], float(r["gap"])) != (
            choice,
            manifest[uid][choice + "_sha256"],
            a - b,
        ):
            raise ValueError(f"{uid}: answer or score differs from the frozen logits")
    return rows, manifest


def draw(work, cap, first=None, *, args=ARGS):
    sample = work / "sample.csv"
    path = work / ("single.json" if first is None else "two-stage.json")
    extra = () if first is None else ("--interim-labels", first, *SCENARIOS)
    report = command(
        "plan",
        *args,
        "--label-budget",
        cap,
        *extra,
        "--sample-seed",
        42,
        "--sample-output",
        sample,
        "--output",
        path,
    )
    return sample, path, report


def freeze(work, *, args=ARGS):
    probe = command(
        "plan",
        "--forecast",
        *args,
        "--label-budget",
        500,
        "--expected-error",
        0.15,
        statuses=(0, 1),
    )
    n = probe["accepted_population"]
    base = dict(
        total=probe["total"],
        population=n,
        sample_seed=42,
        risk_target=0.2,
        one_sided_confidence=0.95,
        required_directional_power=0.8,
        request_cost=1,
    )
    for cap in range(1, min(500, n) + 1):
        limits = sample_error_limits(n, cap, 0.2, 0.95)
        if all(
            forecast_verdict(n, cap, r, limits)["verdict_probabilities"][v] >= 0.8
            for r, v in zip(RATES, ("meets", "exceeds"))
        ):
            break
    else:
        return dict(base, cap=None)
    candidates = []
    for first in sorted(set(range(50, cap, 25)) | {2 * cap // 3}):
        report = command(
            "plan",
            "--forecast",
            *args,
            "--label-budget",
            cap,
            "--interim-labels",
            first,
            *SCENARIOS,
            statuses=(0, 1),
        )
        if report["qualifies"]:
            candidates.append((max(s["cost"] for s in report["scenarios"]), first))
    draw(work, cap, args=args)
    result = dict(
        base,
        cap=cap,
        interim=None,
        single_stage=report["single_stage"],
    )
    if candidates:
        _, first = min(candidates)
        _, _, report = draw(work, cap, first, args=args)
        result.update(
            interim=first,
            stages=report["stages"],
            scenarios=report["scenarios"],
        )
    return result


def source_grader(rows, manifest):
    import pyarrow.dataset as ds

    data = ds.dataset(dataset_file(), format="parquet")

    def grade(issued):
        positions = [int(manifest[r["id"]]["source_row"]) for r in issued]
        labels = data.take(positions, columns=["label"]).column("label").to_pylist()
        out = []
        for r, y in zip(issued, labels, strict=True):
            pred = rows[r["id"]]
            if r["output_key"] != pred["output_sha256"] or int(y) not in range(4):
                raise ValueError("unexpected issued output or reference label")
            out.append(
                dict(
                    id=r["id"],
                    output_key=r["output_key"],
                    correct=int(pred["answer"] == CHOICES[int(y)]),
                )
            )
        return out

    return grade


def audit(work, design, grade):
    cap, first = design["cap"], design["interim"]
    arms = [
        ("single_stage", design["single_stage"]["block_size"], None),
        ("bulk", cap, None),
    ]
    if first is not None:
        arms.insert(0, ("two_stage", None, first))
    results = {}
    for name, block, checkpoint in arms:
        folder = work / name
        folder.mkdir()
        sample, plan, _ = draw(folder, cap, checkpoint)
        if read_csv(sample) != read_csv(OUT / "sample.csv"):
            raise ValueError("draw differs from the frozen sample")
        trace, stop = drive(
            BATCH,
            POLICY,
            plan,
            grade,
            folder,
            block_size=block,
            interrupt_first=checkpoint is not None,
        )
        if [x["id"] for x in trace] != [
            x["id"] for x in read_csv(sample)[: len(trace)]
        ]:
            raise ValueError("issued rows differ from the frozen order")
        result = stop["result"]
        results[name] = dict(
            labels=len(trace),
            requests=stop["requests"],
            errors=result["errors"],
            verdict=result["batch_verdict"],
        )
        if checkpoint is not None:
            results[name]["decision_stage"] = result["decision_stage"]
    return results


def comparisons(design):
    out = []
    for price in (0, 1, 5):
        report = command(
            "plan",
            "--forecast",
            *ARGS,
            "--label-budget",
            design["cap"],
            "--interim-labels",
            design["interim"],
            *SCENARIOS,
            "--request-cost",
            price,
        )
        old = report["single_stage"]
        baseline = max(s["cost"] for s in old["scenarios"])
        # Keep the frozen two-stage design even when the reporting price changes.
        new = max(
            s["expected_labels"] + price * s["expected_requests"]
            for s in design["scenarios"]
        )
        out.append(
            dict(
                request_cost=price,
                single_block_size=old["block_size"],
                single_cost=baseline,
                two_stage_cost=new,
                reduction=1 - new / baseline,
            )
        )
    return out


def census_cost(design, errors):
    """Expected issued work if the frozen design were redrawn from this census."""
    n, cap = design["population"], design["cap"]
    first = design["stages"][0]
    look = first["labels"]
    low, high = first["max_errors_to_meet"], first["min_errors_to_exceed"]
    pending = 1
    if low is not None:
        pending -= hypergeometric_cdf(n, errors, look, low)
    if high is not None:
        pending -= hypergeometric_cdf(n, n - errors, look, look - high)
    single = design["single_stage"]
    return dict(
        single_stage=forecast_block_cost(n, errors, cap, single, single["block_size"]),
        two_stage=dict(
            expected_labels=float(look + (cap - look) * pending),
            expected_requests=float(1 + pending),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("design", "audit", "census", "replay"))
    stage = parser.parse_args().stage
    rows, manifest = inputs()
    with TemporaryDirectory() as temp:
        work = Path(temp)
        if stage == "design":
            result = freeze(work)
            if result["cap"] is not None:
                copyfile(work / "sample.csv", OUT / "sample.csv")
            write_json(OUT / "design.json", result)
            print(result)
            return
        design = read_json(OUT / "design.json")
        if design["cap"] is None:
            print("Insufficient single-stage budget.")
            return
        if stage == "audit":
            result = audit(work, design, source_grader(rows, manifest))
            write_json(OUT / "audit.json", result)
            print(result)
        elif stage == "census":
            eligible = [
                dict(id=k, output_key=r["output_sha256"])
                for k, r in rows.items()
                if float(r["gap"]) >= 4.75
            ]
            if len(eligible) != design["population"]:
                raise ValueError("candidate population differs from census")
            grades = source_grader(rows, manifest)(eligible)
            write_report(OUT / "census.csv", grades)
            print(
                "Eligible errors",
                sum(1 - r["correct"] for r in grades),
                "/",
                len(grades),
            )
        else:
            if freeze(work) != design:
                raise ValueError("budget design changed")
            grades = keyed(read_csv(OUT / "census.csv"))
            if grades.keys() != {k for k, r in rows.items() if float(r["gap"]) >= 4.75}:
                raise ValueError("census IDs differ from eligible outputs")
            for uid, g in grades.items():
                if g["output_key"] != rows[uid]["output_sha256"] or g[
                    "correct"
                ] not in ("0", "1"):
                    raise ValueError("census key or grade changed")
            result = audit(
                work, design, lambda requested: [grades[r["id"]] for r in requested]
            )
            if result != read_json(OUT / "audit.json"):
                raise ValueError("handoff differs from recorded verdict")
            print(result)
            if design["interim"] is not None:
                print(comparisons(design))
                print(
                    census_cost(
                        design, sum(r["correct"] == "0" for r in grades.values())
                    )
                )
            print(
                "Eligible errors",
                sum(r["correct"] == "0" for r in grades.values()),
                "/",
                len(grades),
            )


if __name__ == "__main__":
    main()
