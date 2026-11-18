"""Draw, grade in blocks, independently complete, and replay the WinoGrande audit."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from caat.plan import forecast_block_cost

from collect_winogrande import OUT, dataset_file
from common import ROOT, read_csv, read_json, write_report
from study_batch_audit import R2, archived_policy, command, draw, measure, write_json

POLICY = R2 / "policy.json"
BATCH = OUT / "predictions.csv"
PLAN = OUT / "plan.json"
SAMPLE = OUT / "sample.csv"
GRADE_FIELDS = ["id", "output_key", "correct"]


def keyed(rows):
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate item ID")
    return by_id


def issued_rows(rows):
    return [{"id": r["id"], "output_key": r["output_sha256"]} for r in rows]


def verified_predictions():
    manifest = keyed(read_csv(OUT / "manifest.csv"))
    rows = keyed(read_csv(BATCH))
    if rows.keys() != manifest.keys():
        raise ValueError("prediction IDs differ from the frozen cohort")
    for uid, r in rows.items():
        a, b = float(r["A"]), float(r["B"])
        choice = "A" if a >= b else "B"
        if (choice, abs(a - b), manifest[uid][choice + "_sha256"]) != (
            r["answer"],
            float(r["gap"]),
            r["output_sha256"],
        ):
            raise ValueError(f"option logits or semantic output key differ for {uid}")
    return rows


def freeze_draw():
    verified_predictions()
    batch, policy, sample, plan = [
        p.relative_to(ROOT) for p in (BATCH, POLICY, SAMPLE, PLAN)
    ]
    result = draw(batch, policy, sample, plan)
    result["sample_path"] = sample.as_posix()
    result["input"].update(
        batch_predictions=batch.as_posix(), threshold_report=policy.as_posix()
    )
    write_json(PLAN, result)
    print("Eligible", result["accepted_population"], "drawn", result["accepted"])


def source_grader():
    import pyarrow.dataset as ds

    data = ds.dataset(dataset_file(), format="parquet")
    predictions = verified_predictions()
    manifest = keyed(read_csv(OUT / "manifest.csv"))

    def grade(issued):
        if not issued:
            return []
        ids = list(keyed(issued))
        for row in issued:
            if row["output_key"] != predictions[row["id"]]["output_sha256"]:
                raise ValueError("issued key differs from the saved output")
        positions = [int(manifest[uid]["source_row"]) for uid in ids]
        answers = data.take(positions, columns=["answer"]).column("answer").to_pylist()
        results = []
        for uid, value in zip(ids, answers, strict=True):
            choice = {"1": "A", "2": "B"}[value]
            r = predictions[uid]
            results.append(
                dict(
                    id=uid,
                    output_key=r["output_sha256"],
                    correct=int(choice == r["answer"]),
                )
            )
        return results

    return grade


def drive(batch, policy, plan, grader, work, *, block_size=10, interrupt_first=False):
    args = (
        "audit-block",
        "--handoff",
        work / "handoff",
        *(("--block-size", block_size) if block_size is not None else ()),
        "--grader",
        "dataset",
        "--batch-predictions",
        batch,
        "--threshold-report",
        policy,
        "--plan-report",
        plan,
    )
    start = perf_counter()
    report = command(*args)
    trace = []
    calls = 1
    while request := report["request"]:
        rows = read_csv(Path(request["path"]))
        if set(report["outstanding"]["dataset"]) != set(keyed(rows)):
            raise ValueError("grader received an unexpected row")
        grades = grader(rows)
        if {r["id"]: r["output_key"] for r in grades} != {
            r["id"]: r["output_key"] for r in rows
        } or len(grades) != len(rows):
            raise ValueError("grader did not return exactly the issued outputs")
        trace.extend(dict(block=report["requests"], **r) for r in grades)
        path = work / "receipt.csv"
        receipt = ("--request", request["id"], "--grader-file", f"dataset={path}")
        if interrupt_first and request["start"] == 0:
            split = len(grades) // 2
            write_report(path, grades[:split], GRADE_FIELDS)
            command(*args, *receipt)
            resumed = command(*args)
            if resumed["request"]["id"] != request["id"] or set(
                resumed["outstanding"]["dataset"]
            ) != set(keyed(grades[split:])):
                raise ValueError("restart lost the partial receipt")
            calls += 2
        write_report(path, grades, GRADE_FIELDS)
        report = command(*args, *receipt)
        calls += 1
        if request["start"] == 0:
            if command(*args, *receipt) != report:
                raise ValueError("retry changed the issued work")
            calls += 1
    stop = {
        k: report[k]
        for k in ("requests", "unique_requested", "requested_labels", "returned_labels")
    }
    stop.update(
        audit_calls=calls,
        grader_calls=report["requests"],
        wall_seconds=perf_counter() - start,
        result=report["graders"]["dataset"],
    )
    if (
        len(keyed(trace)) != stop["returned_labels"]
        or len(trace) != stop["unique_requested"]
    ):
        raise ValueError("receipt accounting disagrees with issued IDs")
    return trace, stop


def grade_blocks():
    grader = source_grader()
    with TemporaryDirectory() as temp:
        trace, stop = drive(BATCH, POLICY, PLAN, grader, Path(temp))
    write_report(OUT / "receipts.csv", trace, ["block", *GRADE_FIELDS])
    write_json(OUT / "stop.json", stop)
    print(stop)


def verify_full():
    stop = read_json(OUT / "stop.json")
    grades = source_grader()(read_csv(SAMPLE))
    path = OUT / "full-grades.csv"
    write_report(path, grades, GRADE_FIELDS)
    full = measure(BATCH, POLICY, SAMPLE, PLAN, path)
    if full["batch_verdict"] != stop["result"]["batch_verdict"]:
        raise ValueError("full-budget verdict differs from the stopped verdict")
    write_json(OUT / "full-result.json", full)
    print(full)


def census():
    full = read_json(OUT / "full-result.json")
    if full != measure(BATCH, POLICY, SAMPLE, PLAN, OUT / "full-grades.csv"):
        raise ValueError("verify the frozen full-draw decision before the census")
    rows = [r for r in verified_predictions().values() if float(r["gap"]) >= 4.75]
    grades = source_grader()(issued_rows(rows))
    write_report(OUT / "census.csv", grades, GRADE_FIELDS)
    print("Census", len(grades), sum(1 - r["correct"] for r in grades))


def replay(work):
    pred = verified_predictions()
    grades = keyed(read_csv(OUT / "census.csv"))
    eligible = {
        i: r["output_sha256"] for i, r in pred.items() if float(r["gap"]) >= 4.75
    }
    if eligible != {i: r["output_key"] for i, r in grades.items()}:
        raise ValueError("eligible ID or output key differs from census")
    sample, plan = work / "sample.csv", work / "plan.json"
    forecast = draw(BATCH, POLICY, sample, plan)
    if read_csv(sample) != read_csv(SAMPLE):
        raise ValueError("frozen draw changed")

    def grade(rows):
        return [grades[r["id"]] for r in rows]

    trace, stop = drive(BATCH, POLICY, PLAN, grade, work)
    frozen = read_json(OUT / "stop.json")
    if any(stop[k] != v for k, v in frozen.items() if k != "wall_seconds"):
        raise ValueError("recorded stopping decision changed")
    if read_csv(OUT / "receipts.csv") != [
        {k: str(v) for k, v in r.items()} for r in trace
    ]:
        raise ValueError("receipt history changed")
    if read_csv(OUT / "full-grades.csv") != grade(read_csv(SAMPLE)):
        raise ValueError("full draw grades differ from census")
    full = measure(BATCH, POLICY, SAMPLE, PLAN, OUT / "full-grades.csv")
    if (
        full != read_json(OUT / "full-result.json")
        or full["batch_verdict"] != stop["result"]["batch_verdict"]
    ):
        raise ValueError("full comparator changed")
    k = sum(r["correct"] == "0" for r in grades.values())
    print(
        "WinoGrande",
        stop["unique_requested"],
        full["batch_verdict"],
        "census",
        k,
        len(grades),
        "expected labels",
        forecast_block_cost(len(grades), k, forecast["accepted"], forecast, 10)[
            "expected_labels"
        ],
    )


def archives():
    for name, expected in [("anli", 90), ("anli_r2", 60), ("boolq", 100)]:
        with TemporaryDirectory() as temp:
            work = Path(temp)
            policy = POLICY
            if name == "anli_r2":
                batch = R2 / "predictions.csv"
                grades = keyed(read_csv(R2 / "census.csv"))
            else:
                policy = work / "policy.json"
                write_json(policy, archived_policy(name, work))
                batch = work / "holdout.csv"
                grades = {
                    r["id"]: dict(
                        id=r["id"], output_key=r["output_sha256"], correct=r["correct"]
                    )
                    for r in read_csv(batch)
                }
            sample, plan = work / "sample.csv", work / "plan.json"
            draw(batch, policy, sample, plan)

            def grade(rows):
                return [grades[r["id"]] for r in rows]

            _, stop = drive(batch, policy, plan, grade, work)
            labels = work / "full.csv"
            write_report(labels, grade(read_csv(sample)))
            full = measure(batch, policy, sample, plan, labels)
            if (
                stop["unique_requested"] != expected
                or stop["result"]["batch_verdict"] != full["batch_verdict"]
            ):
                raise ValueError(f"{name}: block replay differs from full sample")
            print(name, expected, full["batch_verdict"])


if __name__ == "__main__":
    stages = {
        "draw": freeze_draw,
        "grade-blocks": grade_blocks,
        "verify-full": verify_full,
        "census": census,
        "archives": archives,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*stages, "replay"])
    stage = parser.parse_args().stage
    if stage == "replay":
        with TemporaryDirectory() as temp:
            replay(Path(temp))
    else:
        stages[stage]()
