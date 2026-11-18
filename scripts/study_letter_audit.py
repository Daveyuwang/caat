"""Freeze, audit and census the UCI Letter Recognition batch."""

import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from caat.plan import forecast_block_cost
from common import ROOT, fetch, read_csv, read_json, write_report
from study_batch_audit import command, measure, write_json
from study_winogrande import drive

OUT = ROOT / "results/letter_audit"
URL = "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip"
SOURCE_DIGEST = "2b89f3602cf768d3c8355267d2f13f2417809e101fc2b5ceee10db19a60de6e2"
BATCH = OUT / "predictions.csv"
POLICY = OUT / "policy.json"
BATCH_ARGS = ("--threshold-report", POLICY, "--batch-predictions", BATCH)
BUDGETS = (50, 100, 150, 300)


def source_rows(work):
    with ZipFile(fetch(URL, work / "letters.zip")) as archive:
        data = archive.read("letter-recognition.data")
    if sha256(data).hexdigest() != SOURCE_DIGEST:
        raise ValueError("UCI letter data differs from the frozen revision")
    return data.splitlines()


def draw(work, budget):
    sample, plan = work / "sample.csv", work / "plan.json"
    report = command(
        "plan",
        *BATCH_ARGS,
        "--label-budget",
        budget,
        "--sample-seed",
        42,
        "--sample-output",
        sample,
        "--output",
        plan,
    )
    return sample, plan, report


def sample_digest(path):
    # Hash logical lines so that a replay works with either checkout line ending.
    return sha256(path.read_text(encoding="utf-8").encode()).hexdigest()


def freeze(work, raw):
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    # The letter occupies the first byte. Only the training and policy rows
    # supply labels here. All 4,000 new rows supply features alone.
    features = [[int(v) for v in line[2:].split(b",")] for line in raw]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    with threadpool_limits(limits=1):
        model.fit(features[:12000], [line[:1].decode() for line in raw[:12000]])
        probabilities = model.predict_proba(features[12000:])
    fitted, batch = [], []
    for index, probs in enumerate(probabilities, 12000):
        letter = model.classes_[probs.argmax()]
        row = dict(id=str(index + 1), prediction=letter, score=f"{probs.max():.10f}")
        if index < 16000:
            fitted.append(dict(row, correct=int(letter == raw[index][:1].decode())))
        else:
            batch.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    fit = work / "policy-input.csv"
    write_report(fit, fitted)
    write_report(BATCH, batch)
    policy = command(
        "policy",
        fit,
        "--score-column",
        "score",
        "--correct-column",
        "correct",
        "--output-key-column",
        "prediction",
        "--track-fit-outputs",
        "--risk-target",
        ".05",
        "--seed",
        42,
        "--split-checks",
        0,
        statuses=(0, 1),
    )
    policy["input"]["path"] = fit.name
    write_json(POLICY, policy)
    scenarios = []
    for budget in BUDGETS:
        pair = {}
        for rate in ("0.02", "0.08"):
            f = command(
                "plan",
                "--forecast",
                *BATCH_ARGS,
                "--label-budget",
                budget,
                "--expected-error",
                rate,
                "--block-size",
                10,
            )
            pair[rate] = {
                "errors": f["expected_population_errors"],
                **f["verdict_probabilities"],
                **f["block_cost"],
            }
        scenarios.append({"budget": budget, **pair})
    chosen = next(
        (
            s["budget"]
            for s in scenarios
            if s["0.02"]["meets"] >= 0.8 and s["0.08"]["exceeds"] >= 0.8
        ),
        None,
    )
    samples = {}
    for budget in dict.fromkeys((chosen or 100, 100)):
        path, _, plan = draw(work, budget)
        samples[str(budget)] = sample_digest(path)
    write_json(
        OUT / "design.json",
        {
            "source_sha256": SOURCE_DIGEST,
            "sklearn": sklearn.__version__,
            "accepted_population": plan["accepted_population"],
            "accepted_population_sha256": plan["input"]["accepted_population_sha256"],
            "threshold_report_sha256": plan["input"]["threshold_report_sha256"],
            "scenarios": scenarios,
            "qualifying_budget": chosen,
            "sample_sha256": samples,
        },
    )


def frozen_draw(work, design, budget):
    sample, plan, report = draw(work, budget)
    if sample_digest(sample) != design["sample_sha256"][str(budget)] or any(
        report["input"][key] != design[key]
        for key in ("accepted_population_sha256", "threshold_report_sha256")
    ):
        raise ValueError(
            "sample, policy or eligible population differs from the freeze"
        )
    return sample, plan, report


def source_grader(raw, predictions):
    def grade(rows):
        grades = []
        for row in rows:
            uid = row["id"]
            prediction = predictions[uid]["prediction"]
            if row["output_key"] != prediction:
                raise ValueError(
                    f"{uid}: issued output differs from the saved prediction"
                )
            correct = prediction == raw[int(uid) - 1][:1].decode()
            grades.append(dict(id=uid, output_key=prediction, correct=int(correct)))
        return grades

    return grade


def audit(work, raw, design, predictions):
    budget = design["qualifying_budget"] or 100
    sample, plan, _ = frozen_draw(work, design, budget)
    grade = source_grader(raw, predictions)
    trace, stop = drive(BATCH, POLICY, plan, grade, work)
    issued = read_csv(sample)[: len(trace)]
    if [(r["id"], r["output_key"]) for r in trace] != [
        (r["id"], r["output_key"]) for r in issued
    ]:
        raise ValueError("issued order differs from frozen sample")
    results = {}
    for cap in dict.fromkeys((budget, 100)):
        sample, plan, _ = frozen_draw(work, design, cap)
        grades = work / "full-grades.csv"
        write_report(grades, grade(read_csv(sample)))
        results[str(cap)] = measure(BATCH, POLICY, sample, plan, grades)
    if results[str(budget)]["batch_verdict"] != stop["result"]["batch_verdict"]:
        raise ValueError("block and full-draw verdicts differ")
    write_json(
        OUT / "audit.json",
        {
            "labels": len(trace),
            "requests": stop["requests"],
            "block_verdict": stop["result"],
            "full_draws": results,
        },
    )


def census(work, raw, design, predictions):
    budget = design["qualifying_budget"] or 100
    _, _, plan = frozen_draw(work, design, budget)
    result = read_json(OUT / "audit.json")
    cutoff = plan["input"]["threshold"]
    rows = [
        dict(id=uid, output_key=r["prediction"])
        for uid, r in predictions.items()
        if float(r["score"]) >= cutoff
    ]
    if len(rows) != plan["accepted_population"]:
        raise ValueError("census differs from eligible population")
    errors = sum(1 - r["correct"] for r in source_grader(raw, predictions)(rows))
    cost = forecast_block_cost(len(rows), errors, plan["accepted"], plan, 10)
    print(
        f"{errors}/{len(rows)} eligible errors; expected {cost['expected_labels']:.2f} labels, "
        f"{cost['expected_requests']:.2f} requests; observed {result['labels']}, {result['requests']}"
    )
    write_json(OUT / "census.json", {"eligible": len(rows), "errors": errors, **cost})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "audit", "census"))
    args = parser.parse_args()
    with TemporaryDirectory() as directory:
        work = Path(directory)
        raw = source_rows(work)
        if args.stage == "freeze":
            freeze(work, raw)
        else:
            design = read_json(OUT / "design.json")
            predictions = {r["id"]: r for r in read_csv(BATCH)}
            if args.stage == "audit":
                audit(work, raw, design, predictions)
            else:
                census(work, raw, design, predictions)


if __name__ == "__main__":
    main()
