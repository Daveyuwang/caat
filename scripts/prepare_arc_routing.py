"""Join lm-eval ARC-Challenge choice scores to their original question IDs."""

import argparse
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from caat.lm_eval_log import argument_records, choice_scores, read_samples
from common import read_csv, write_report


def prepare(train_log, test_log, output):
    seen_ids = set()
    records = []
    with TemporaryDirectory() as directory:
        for split, source, expected in (
            ("train", train_log, 1119),
            ("heldout", test_log, 1172),
        ):
            path = Path(directory) / "choices.csv"
            choice_scores(source, path)
            rows = read_csv(path)
            if len(rows) != expected:
                raise ValueError(f"{split} needs all {expected} official questions")
            for row, sample in zip(rows, read_samples(source, None), strict=True):
                doc = sample["doc"]
                args = argument_records(sample)
                if (
                    str(sample["doc_id"]) != row["id"]
                    or row["prompt_hash"] != sample["prompt_hash"]
                    or doc["choices"]["label"].index(doc["answerKey"])
                    != int(sample["target"])
                    or args[0][0] != f"Question: {doc['question']}\nAnswer:"
                    or any(a[0] != args[0][0] for a in args)
                    or [a[1].strip() for a in args]
                    != [v.strip() for v in doc["choices"]["text"]]
                ):
                    raise ValueError(
                        "sample identity, answer choices or grade mismatch"
                    )
                question = doc["id"]
                if not question or question in seen_ids:
                    raise ValueError(f"duplicate ARC question ID {question}")
                seen_ids.add(question)
                records.append({**row, "id": question, "split": split})
    train = sorted((r for r in records if r["split"] == "train"), key=lambda r: r["id"])
    heldout = [r for r in records if r["split"] == "heldout"]
    prompts = {r["prompt_hash"] for r in heldout}
    unique = []
    for r in train:
        if r["prompt_hash"] not in prompts:
            unique.append(r)
            prompts.add(r["prompt_hash"])
    print(
        f"Retained {len(unique)} of {len(train)} training rows after prompt deduplication; {len(heldout)} test rows."
    )
    train = unique
    records = train + heldout
    Random(42).shuffle(train)
    for i, row in enumerate(train):
        row["split"] = "design" if i < len(train) // 3 else "calibration"
    output.parent.mkdir(exist_ok=True, parents=True)
    if output.resolve() in {train_log.resolve(), test_log.resolve()}:
        raise ValueError("output would overwrite an input log")
    write_report(output, records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_log", type=Path)
    parser.add_argument("test_log", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.train_log, args.test_log, args.output)
