"""Prepare aligned scores for prefixes of the archived ten samples."""

import argparse
from math import isclose
from pathlib import Path

from caat.collect import write_responses
from caat.features import FEATURES, compute_feature, read_response
from caat.table import read_external_scores

from common import ROOT, read_csv, read_json, write_report
from study_black_box import check_ids, normalize_answer, read_jsonl, score_rows
from study_external_scores import IDENTITY, SCORES, generation_digest
from study_model_families import read_partitions


BUDGETS = (2, 5, 10)
BASELINES = ("mean_token_logprob", "sequence_logprob", "shorter_response", "random")
AGREEMENT = ("sample_agreement", "normalized_sample_agreement")
SAMPLE_SCORES = (*AGREEMENT, "negative_semantic_entropy", *SCORES.values())
SOURCE_FOLDERS = {
    "qwen2_5_7b": "black_box_1k",
    "phi4_mini": "black_box_families/phi4_mini",
    "smollm2": "black_box_families/smollm2",
}
FROZEN = ROOT / "results/black_box_families"
REPLAY = ROOT / "replay/sample_budget"


def prefix_column(name, budget):
    return f"{name}_k{budget}"


def scorer_id(dataset, model, item_id):
    return f"{dataset}/{model}/{item_id}"


def check_archive_value(item, name, value, reference):
    if not isclose(value, float(reference), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Archived score changed for {item}, {name}")


def prepare(replay, output):
    records, timings = [], []
    prefixes = {budget: [] for budget in BUDGETS}
    for dataset, assignments in read_partitions(FROZEN / "split.csv").items():
        active = sorted(
            item for item, fold in assignments.items() if fold != "calibration"
        )
        for model, source in SOURCE_FOLDERS.items():
            folder = replay / source / dataset
            responses = read_jsonl(folder / "responses.jsonl")
            clusters = read_jsonl(folder / "clusters.jsonl")
            rebuilt = {row["id"]: row for row in score_rows(dataset, folder, 10)}
            original = {
                row["id"]: row
                for row in read_csv(FROZEN / model / f"{dataset}_scores.csv")
            }
            for data in (responses, clusters, rebuilt, original):
                check_ids(assignments, data)
            timing = read_json(folder / "timing.json")
            sample = timing["methods"]["sample"]
            if sample["calls"] != len(assignments) * 10:
                raise ValueError(
                    f"Unexpected archived sample count for {model}/{dataset}"
                )
            timings.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "samples_per_answer": 10,
                    "prompt_evaluations": sample["calls"],
                    "model_invocations": sample["batches"],
                    "sampling_seconds": sample["seconds"],
                }
            )
            for item in active:
                full = responses[item]
                labels = clusters[item]["clusters"]
                reference = original[item]
                key = scorer_id(dataset, model, item)
                row = {
                    "dataset": dataset,
                    "model": model,
                    "id": item,
                    "correct": rebuilt[item]["correct"],
                }
                for name in (*BASELINES, "correct"):
                    check_archive_value(key, name, rebuilt[item][name], reference[name])
                    row[name] = rebuilt[item][name]
                for budget in BUDGETS:
                    draws = full["samples"][:budget]
                    _, answer = read_response(
                        {**full, "samples": draws, "clusters": labels[:budget]}
                    )
                    scores = {
                        name: compute_feature(FEATURES[name], answer)
                        for name in AGREEMENT
                    }
                    scores["negative_semantic_entropy"] = -compute_feature(
                        FEATURES["semantic_entropy"], answer
                    )
                    for name, value in scores.items():
                        row[prefix_column(name, budget)] = value
                        if budget == 10:
                            check_archive_value(key, name, value, reference[name])
                    distinct = (
                        set(map(normalize_answer, draws)) - {""}
                        if dataset == "triviaqa"
                        else set()
                    )
                    row[prefix_column("nli_pairs", budget)] = len(distinct) * (
                        len(distinct) - 1
                    )
                    prefixes[budget].append(
                        {"id": key, "text": full["text"], "samples": draws}
                    )
                records.append(row)
    output.mkdir(parents=True, exist_ok=True)
    for budget, rows in prefixes.items():
        write_responses(output / f"prefix-{budget}.jsonl", rows)
    write_report(output / "builtin_scores.csv", records)
    write_report(output / "collection_cost.csv", timings)


def join(output):
    rows = read_csv(output / "builtin_scores.csv")
    cases = {(row["dataset"], row["model"]) for row in rows}
    original = {
        case: {
            row["id"]: row
            for row in read_csv(FROZEN / case[1] / f"{case[0]}_scores.csv")
        }
        for case in cases
    }
    by_key = {scorer_id(row["dataset"], row["model"], row["id"]): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("Duplicate question identity in the prefix scores")
    for budget in BUDGETS:
        prefixes = read_jsonl(output / f"prefix-{budget}.jsonl")
        check_ids(by_key, prefixes)
        digests = {key: generation_digest(row) for key, row in prefixes.items()}
        for provider, name in SCORES.items():
            scores = read_external_scores(
                output / f"prefix-{budget}-{provider}.csv", name, IDENTITY
            )
            check_ids({(key,): None for key in by_key}, scores)
            for key, row in by_key.items():
                digest, value = scores[(key,)]
                if digest != digests[key]:
                    raise ValueError(
                        f"{provider} scored a different sample prefix for {key}"
                    )
                if budget == 10:
                    reference = original[(row["dataset"], row["model"])][row["id"]][
                        name
                    ]
                    check_archive_value(key, name, value, reference)
                row[prefix_column(name, budget)] = value
    write_report(output / "prefix_scores.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "join"))
    parser.add_argument("--output-dir", type=Path, default=REPLAY)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(ROOT / "replay", args.output_dir)
    else:
        join(args.output_dir)


if __name__ == "__main__":
    main()
