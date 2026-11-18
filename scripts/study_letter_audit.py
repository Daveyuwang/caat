"""Freeze, audit and census the UCI Letter Recognition batch."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from caat.plan import forecast_block_cost
from common import ROOT, fetch, read_csv, read_json, write_report
from study_batch_audit import command, measure, write_json
from study_winogrande import drive

OUT = ROOT / "results/letter_audit"
URL = "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip"
BATCH = OUT / "predictions.csv"
POLICY = OUT / "policy.json"
BATCH_ARGS = ("--threshold-report", POLICY, "--batch-predictions", BATCH)
BUDGETS = (50, 100, 150, 300)


def source_rows(work):
    with ZipFile(fetch(URL, work / "letters.zip")) as archive:
        data = archive.read("letter-recognition.data")
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
    write_json(
        OUT / "design.json",
        {
            "sklearn": sklearn.__version__,
            **budget_design(),
        },
    )


def budget_design():
    scenarios = []
    for budget in BUDGETS:
        pair = {}
        for rate in ("0.02", "0.08"):
            forecast = command(
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
                "errors": forecast["expected_population_errors"],
                **forecast["verdict_probabilities"],
                **forecast["block_cost"],
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
    return {
        "accepted_population": forecast["accepted_population"],
        "scenarios": scenarios,
        "qualifying_budget": chosen,
    }


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
    sample, plan, _ = draw(work, budget)
    grade = source_grader(raw, predictions)
    trace, stop = drive(BATCH, POLICY, plan, grade, work)
    issued = read_csv(sample)[: len(trace)]
    if [(r["id"], r["output_key"]) for r in trace] != [
        (r["id"], r["output_key"]) for r in issued
    ]:
        raise ValueError("issued order differs from frozen sample")
    results = {}
    for cap in dict.fromkeys((budget, 100)):
        sample, plan, _ = draw(work, cap)
        grades = work / "full-grades.csv"
        write_report(grades, grade(read_csv(sample)))
        results[str(cap)] = measure(BATCH, POLICY, sample, plan, grades)
    if results[str(budget)]["batch_verdict"] != stop["result"]["batch_verdict"]:
        raise ValueError("block and full-draw verdicts differ")
    return {
        "labels": len(trace),
        "requests": stop["requests"],
        "block_verdict": stop["result"],
        "full_draws": results,
    }


def census(work, raw, design, predictions):
    budget = design["qualifying_budget"] or 100
    _, _, plan = draw(work, budget)
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
    return {"eligible": len(rows), "errors": errors, **cost}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "audit", "census", "replay"))
    args = parser.parse_args()
    with TemporaryDirectory() as directory:
        work = Path(directory)
        raw = source_rows(work)
        if args.stage == "freeze":
            freeze(work, raw)
            return
        design = read_json(OUT / "design.json")
        predictions = {r["id"]: r for r in read_csv(BATCH)}
        if args.stage == "replay":
            fresh = budget_design()
            if any(v != design[k] for k, v in fresh.items()):
                raise ValueError("budget comparison differs from the frozen design")
        for stage in (audit, census):
            name = stage.__name__
            if args.stage in (name, "replay"):
                result = stage(work, raw, design, predictions)
                path = OUT / f"{name}.json"
                if args.stage != "replay":
                    write_json(path, result)
                elif result != read_json(path):
                    raise ValueError(f"{name} differs from the archived result")
                print(name, result)


if __name__ == "__main__":
    main()
