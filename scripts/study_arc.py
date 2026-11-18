"""Freeze, issue and replay the ARC-Challenge audit on paired seed-42 draws."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from collect_arc import CHOICES, OUT, dataset_file
from common import ROOT, read_csv, read_json, write_report
from study_batch_audit import command, write_json
from study_hellaswag import draw, freeze, inputs
from study_letter_audit import sample_digest
from study_winogrande import drive, issued_rows, keyed

BATCH = OUT / "predictions.csv"
POLICY = ROOT / "results/anli_r2/policy.json"
ARGS = ("--threshold-report", POLICY, "--batch-predictions", BATCH)


def preview():
    report = command(
        "plan",
        *ARGS,
        "--forecast",
        "--budget-ceiling",
        500,
        "--error-scenarios",
        0.15,
        0.25,
        "--power",
        0.8,
        "--request-cost",
        1,
        statuses=(0, 1),
    )
    for key, path in (("threshold_report", POLICY), ("batch_predictions", BATCH)):
        report["input"][key] = path.relative_to(ROOT).as_posix()
    return report


def source_grader(predictions, manifest):
    import pyarrow.dataset as ds

    source = ds.dataset(dataset_file(), format="parquet")

    def grade(issued):
        positions = [int(manifest[r["id"]]["source_row"]) for r in issued]
        refs = source.take(
            positions, columns=["id", "choices", "answerKey"]
        ).to_pylist()
        grades = []
        for r, ref in zip(issued, refs, strict=True):
            prediction = predictions[r["id"]]
            if r["id"] != ref["id"] or r["output_key"] != prediction["output_sha256"]:
                raise ValueError("issued output does not match the pinned ARC row")
            answer = CHOICES[ref["choices"]["label"].index(ref["answerKey"])]
            grades.append(
                dict(
                    id=r["id"],
                    output_key=r["output_key"],
                    correct=int(answer == prediction["answer"]),
                )
            )
        return grades

    return grade


def plans(work, offers, prior):
    report = work / "offers.json"
    write_json(report, offers)
    arms = {}
    for name in offers["offers"]:
        folder = work / name
        folder.mkdir()
        sample, plan = folder / "sample.csv", folder / "plan.json"
        command(
            "plan",
            *ARGS,
            "--from-comparison",
            report,
            "--choose",
            name,
            "--sample-seed",
            42,
            "--sample-output",
            sample,
            "--output",
            plan,
        )
        arms[name] = sample, plan, None
    if prior["cap"] is not None:
        for name, first, block in (
            ("prior-two-stage", prior["interim"], None),
            ("prior-fixed", None, prior["single_stage"]["block_size"]),
        ):
            if first is None and block is None:
                continue
            folder = work / name
            folder.mkdir()
            sample, plan, _ = draw(folder, prior["cap"], first, args=ARGS)
            arms[name] = sample, plan, block
    return arms


def paired_draws(arms):
    samples = {name: read_csv(sample) for name, (sample, _, _) in arms.items()}
    longest = max(samples.values(), key=len, default=[])
    if any(rows != longest[: len(rows)] for rows in samples.values()):
        raise ValueError("audit arms do not share a single random order")
    return longest, {name: sample_digest(paths[0]) for name, paths in arms.items()}


def dispatch(arms, grade):
    audits, receipts = {}, {}
    for name, (sample, plan, block) in arms.items():
        trace, stop = drive(
            BATCH,
            POLICY,
            plan,
            grade,
            sample.parent,
            block_size=block,
            interrupt_first=True,
        )
        issued = read_csv(sample)[: len(trace)]
        if [r["id"] for r in trace] != [r["id"] for r in issued]:
            raise ValueError("issued rows differ from frozen order")
        for r in trace:
            row = dict(
                id=r["id"], output_key=r["output_key"], correct=int(r["correct"])
            )
            if receipts.setdefault(r["id"], row) != row:
                raise ValueError("inconsistent grade across arms")
        audits[name] = dict(
            labels=stop["returned_labels"],
            requests=stop["requests"],
            errors=stop["result"]["errors"],
            verdict=stop["result"]["batch_verdict"],
            request_sizes=[
                sum(r["block"] == i for r in trace)
                for i in range(1, stop["requests"] + 1)
            ],
        )
    return audits, list(receipts.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("design", "audit", "census", "replay"))
    stage = parser.parse_args().stage
    rows, manifest = inputs(BATCH, OUT / "cohort.csv")
    eligible = issued_rows([r for r in rows.values() if float(r["gap"]) >= 4.75])
    with TemporaryDirectory() as directory:
        work = Path(directory)
        if stage == "census":
            audit = read_json(OUT / "audit.json")
            receipts = keyed(read_csv(OUT / "receipts.csv"))
            if max(r["labels"] for r in audit.values()) != len(receipts):
                raise ValueError("complete paired handoffs before grading the census")
            grades = source_grader(rows, manifest)(eligible)
            write_report(OUT / "census.csv", grades)
            print(
                "Eligible errors:",
                sum(1 - r["correct"] for r in grades),
                "/",
                len(grades),
            )
            return
        offers_path, design_path = OUT / "offers.json", OUT / "design.json"
        if stage in ("design", "replay"):
            offers = preview()
            prior = freeze(work, args=ARGS)
        else:
            offers = read_json(offers_path)
            prior = read_json(design_path)["prior"]
        arms = plans(work, offers, prior)
        sample, digests = paired_draws(arms)
        design = dict(prior=prior, sample_sha256=digests)
        if stage == "design":
            write_json(offers_path, offers)
            write_json(design_path, design)
            write_report(OUT / "sample.csv", sample)
            return
        frozen = read_json(offers_path)
        if (
            design != read_json(design_path)
            or sample != read_csv(OUT / "sample.csv")
            or any(v != frozen.get(k) for k, v in offers.items() if k != "caat_version")
            or len(eligible) != offers["accepted_population"]
        ):
            raise ValueError("audit design differs from the frozen cohort or plan")
        if stage == "audit":
            audits, receipts = dispatch(arms, source_grader(rows, manifest))
            write_json(OUT / "audit.json", audits)
            write_report(OUT / "receipts.csv", receipts)
        else:
            receipts = keyed(read_csv(OUT / "receipts.csv"))
            audits, grades = dispatch(
                arms, lambda issued: [receipts[r["id"]] for r in issued]
            )
            if audits != read_json(OUT / "audit.json") or len(grades) != len(receipts):
                raise ValueError("replayed receipts differ from the frozen handoffs")
            census = keyed(read_csv(OUT / "census.csv"))
            if (
                {r["id"]: r["output_key"] for r in eligible}
                != {i: g["output_key"] for i, g in census.items()}
                or any(g["correct"] not in ("0", "1") for g in census.values())
                or any(census[i] != g for i, g in receipts.items())
            ):
                raise ValueError("eligible census differs from predictions or receipts")
            print(
                "Eligible errors:",
                sum(g["correct"] == "0" for g in census.values()),
                "/",
                len(census),
            )
        print(audits)


if __name__ == "__main__":
    main()
