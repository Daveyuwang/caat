"""Freeze, grade and replay the prospective ANLI R2 fixed-batch audit."""

import argparse
import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from random import Random
from shutil import copyfile
from tempfile import TemporaryDirectory

from caat.cli import main as caat
from caat.workflow import run_workflow
from collect_anli import LABELS
from common import ROOT, read_csv, write_report
from prepare_anli import source
from prepare_boolq import digest

R2 = ROOT / "results/anli_r2"
BUDGET = 100
SEED = 42


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def command(*args, statuses=(0,)):
    with redirect_stdout(StringIO()) as out:
        try:
            code = caat([*map(str, args), "--json"])
        except SystemExit as exc:
            raise ValueError(f"{args[0]} rejected the study inputs") from exc
    if code not in statuses:
        raise ValueError(f"{args[0]} returned {code}")
    return json.loads(out.getvalue())


def archived_policy(name, work):
    """Rebuild the 0.6 fitted BoolQ or scalar ANLI source report."""
    archive = ROOT / "results" / name
    rows = sorted(read_csv(archive / "scores.csv"), key=lambda r: r["id"])
    cal_role = "calibration" if name == "anli" else "calibration_pool"
    cal = [r for r in rows if r["role"] == cal_role]
    if name == "boolq":
        Random(42).shuffle(cal)
        cal = cal[:480]
    held_role = "primary" if name == "anli" else "holdout"
    for stem, split in (
        ("calibration", cal),
        ("reference", [r for r in rows if r["role"] == "reference"]),
        ("holdout", [r for r in rows if r["role"] == held_role]),
    ):
        write_report(work / (stem + ".csv"), split)
    settings = 'correct_columns=["correct"]\noutput_key_column="output_sha256"\nrisk_target=0.2\nholdout_confidence=0.95\n'
    if name == "anli":
        settings += 'score_columns=["gap"]\n'
    settings += '[predictions]\ncalibration="calibration.csv"\nholdout="holdout.csv"\n[frozen]\nreference="reference.csv"\n'
    if name == "boolq":
        copyfile(archive / "scorer.json", work / "model.json")
        settings += 'scorer="model.json"\n'
    config = work / "run.toml"
    config.write_text(settings, encoding="utf-8")
    report = run_workflow(config)
    # The input tables can be regenerated in any directory.
    meta = report["input"]
    meta["config"] = config.name
    meta["predictions"] = {k: Path(v).name for k, v in meta["predictions"].items()}
    meta["html"] = Path(meta["html"]).name
    if name == "boolq":
        meta["frozen"]["scorer_path"] = "model.json"
    selection = report["policy"]["selection"]
    held = report["policy"]["heldout"]["graders"]["correct"]
    expected = (588, 154, False) if name == "anli" else (2666, 321, True)
    if (held["accepted"], held["errors"], report["policy"]["confirmed"]) != expected:
        raise ValueError("archived source audit changed")
    if name == "anli" and selection["selected_threshold"] != 4.75:
        raise ValueError("archived ANLI cutoff changed")
    return report


def draw(batch, policy, sample, plan):
    return command(
        "plan",
        "--batch-predictions",
        batch,
        "--threshold-report",
        policy,
        "--label-budget",
        BUDGET,
        "--sample-seed",
        SEED,
        "--sample-output",
        sample,
        "--output",
        plan,
    )


def measure(batch, policy, sample, plan, grades):
    report = command(
        "evaluate",
        sample,
        "--batch-predictions",
        batch,
        "--threshold-report",
        policy,
        "--plan-report",
        plan,
        "--grader-file",
        f"dataset={grades}",
        "--grader-output-key-column",
        "output_key",
        statuses=(0, 1),
    )
    counts = report["overall"]
    return {
        key: counts[key]
        for key in (
            "accepted",
            "errors",
            "risk_lower_errors",
            "risk_upper_errors",
            "risk_lower_bound",
            "risk_upper_bound",
            "batch_verdict",
        )
    }


def prepare(generation):
    generated = sorted(read_csv(generation / "generate.csv"), key=lambda r: r["id"])
    ids = [r["id"] for r in read_csv(R2 / "manifest.csv")]
    if sorted(ids) != [r["id"] for r in generated] or len(set(ids)) != len(ids):
        raise ValueError("generator output differs from frozen manifest")
    for row in generated:
        logits = [float(row[c]) for c in "ABC"]
        answer = LABELS[logits.index(max(logits))]
        if answer != row["answer"] or digest(answer) != row["output_sha256"]:
            raise ValueError(f"answer mismatch for {row['id']}")
        first, second, _ = sorted(logits, reverse=True)
        row["gap"] = first - second
    write_report(R2 / "predictions.csv", generated)
    copyfile(generation / "generate.json", R2 / "collection.json")
    with TemporaryDirectory() as temp:
        write_json(R2 / "policy.json", archived_policy("anli", Path(temp)))
    # Use checkout-relative paths in the portable plan.
    directory = R2.relative_to(ROOT)
    plan = draw(
        directory / "predictions.csv",
        directory / "policy.json",
        directory / "sample.csv",
        directory / "plan.json",
    )
    plan["sample_path"] = (directory / "sample.csv").as_posix()
    for field, name in (
        ("batch_predictions", "predictions.csv"),
        ("threshold_report", "policy.json"),
    ):
        plan["input"][field] = (directory / name).as_posix()
    write_json(R2 / "plan.json", plan)


def dataset_grades(rows):
    lookup = {r["id"]: r for r in rows}
    grades = source("test_r2", ["uid", "label"], ids=set(lookup))
    if sorted(r["uid"] for r in grades) != sorted(lookup):
        raise ValueError("dataset grade IDs differ from requested outputs")
    result = []
    for g in sorted(grades, key=lambda r: r["uid"]):
        r = lookup[g["uid"]]
        if g["label"] not in range(len(LABELS)):
            raise ValueError("invalid ANLI relation label")
        result.append(
            {
                "id": r["id"],
                "output_key": r["output_sha256"],
                "correct": int(r["answer"] == LABELS[g["label"]]),
            }
        )
    return result


def sample_verdict():
    return measure(
        *(
            R2 / f
            for f in (
                "predictions.csv",
                "policy.json",
                "sample.csv",
                "plan.json",
                "sample-grades.csv",
            )
        )
    )


def grade_sample():
    preds = {r["id"]: r for r in read_csv(R2 / "predictions.csv")}
    sampled = read_csv(R2 / "sample.csv")
    write_report(
        R2 / "sample-grades.csv", dataset_grades([preds[r["id"]] for r in sampled])
    )
    write_json(R2 / "sample-result.json", sample_verdict())


def census():
    if sample_verdict() != json.loads((R2 / "sample-result.json").read_bytes()):
        raise ValueError("record the sampled verdict before opening census grades")
    rows = [r for r in read_csv(R2 / "predictions.csv") if float(r["gap"]) >= 4.75]
    write_report(R2 / "census.csv", dataset_grades(rows))


def replay():
    batch = R2 / "predictions.csv"
    policy = R2 / "policy.json"
    with TemporaryDirectory() as temp:
        work = Path(temp)
        sample, plan = work / "sample.csv", work / "plan.json"
        fresh = draw(batch, policy, sample, plan)
        original = json.loads((R2 / "plan.json").read_bytes())
        if (
            read_csv(sample) != read_csv(R2 / "sample.csv")
            or fresh["input"]["accepted_population_sha256"]
            != original["input"]["accepted_population_sha256"]
        ):
            raise ValueError("frozen draw differs from the replay")
        with redirect_stderr(StringIO()) as err, redirect_stdout(StringIO()):
            try:
                caat(
                    [
                        "route",
                        str(batch),
                        "--threshold-report",
                        str(policy),
                        "--output",
                        str(work / "blocked.csv"),
                    ]
                )
            except SystemExit as exc:
                if (
                    exc.code != 2
                    or "requires a confirmed run report" not in err.getvalue()
                ):
                    raise
            else:
                raise ValueError("withheld report allowed routing")
        decisions = work / "comparison.csv"
        command(
            "route",
            batch,
            "--score-column",
            "gap",
            "--threshold",
            "4.75",
            "--output-key-column",
            "output_sha256",
            "--output",
            decisions,
        )
        command(
            "queue",
            decisions,
            "--prediction-filter",
            "decision=accept",
            "--output-key-column",
            "output_key",
            "--budget",
            BUDGET,
            "--seed",
            SEED,
            "--random-output",
            work / "comparison-sample.csv",
        )
        if read_csv(work / "comparison-sample.csv") != read_csv(sample):
            raise ValueError("manual 0.6 workflow returned a different sample")
        counted = {
            r["id"]: r["output_key"]
            for r in read_csv(decisions)
            if r["decision"] == "accept"
        }
        grades = read_csv(R2 / "census.csv")
        if len(grades) != len(counted) or counted != {
            r["id"]: r["output_key"] for r in grades
        }:
            raise ValueError("census IDs or output keys differ from the candidate set")
        grade_map = {r["id"]: r for r in grades}
        if any(grade_map[r["id"]] != r for r in read_csv(R2 / "sample-grades.csv")):
            raise ValueError("sample grades disagree with the census")
        observed = measure(batch, policy, sample, plan, R2 / "sample-grades.csv")
        if observed != json.loads((R2 / "sample-result.json").read_bytes()):
            raise ValueError("sampled verdict changed")
        print(
            f"ANLI R2: {observed}, census errors {sum(r['correct'] == '0' for r in grades)}/{len(grades)}"
        )
        boolq = work / "boolq.json"
        write_json(boolq, archived_policy("boolq", work))
        control = work / "holdout.csv"
        result = draw(control, boolq, sample, plan)
        rows = {r["id"]: r for r in read_csv(control)}
        baseline = {
            r["id"]
            for r in read_csv(ROOT / "results/boolq/accepted.csv")
            if r["arm"] == "fitted_reference_480"
        }
        expected = sorted(baseline)
        Random(SEED).shuffle(expected)
        if (
            result["accepted_population"] != len(baseline)
            or [r["id"] for r in read_csv(sample)] != expected[:BUDGET]
        ):
            raise ValueError("BoolQ draw differs from archived eligible IDs")
        labels = work / "grades.csv"
        write_report(
            labels,
            [
                dict(
                    id=i,
                    output_key=rows[i]["output_sha256"],
                    correct=rows[i]["correct"],
                )
                for i in expected[:BUDGET]
            ],
        )
        print(
            "BoolQ:", measure(control, boolq, sample, plan, labels), "census 321/2666"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_subparsers(dest="stage", required=True)
    choices.add_parser("prepare").add_argument("generation", type=Path)
    for stage in ("grade-sample", "census", "replay"):
        choices.add_parser(stage)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.generation)
    elif args.stage == "grade-sample":
        grade_sample()
    elif args.stage == "census":
        census()
    else:
        replay()
