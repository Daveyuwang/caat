"""Collect and replay the Qwen confidence benchmark on TriviaQA and MMLU-Pro."""

import argparse
import json
import re
from dataclasses import replace
from itertools import combinations
from math import isfinite
from pathlib import Path
from random import Random
from string import ascii_uppercase
from time import perf_counter

from caat.collect import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    TransformersBackend,
    collect_responses,
    read_prompts,
    write_responses,
)
from caat.features import FEATURES, compute_feature, read_response
from caat.metrics import calibration_metrics
from caat.scores import DEFAULT_COVERAGES, ScoredRow, compare_scores

from common import ROOT, fetch, read_csv, write_report
from prepare_mmlu_pro import DATASET_REVISION as MMLU_REVISION
from prepare_mmlu_pro import DATASET_URL as MMLU_URL
from prepare_triviaqa import DATASET_URL as TRIVIA_URL
from prepare_triviaqa import normalize_answer


SAMPLE_SIZE = 128
SEED = 42
SAMPLES = 5
TOKEN_LIMITS = {"triviaqa": 16, "mmlu_pro": 4}
GPU_MEMORY = "13GiB"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
NLI_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"
RESAMPLES = 2000
BOOTSTRAP_SEED = 20261115
BASELINE = "mean_token_logprob"
METHODS = (
    BASELINE,
    "sequence_logprob",
    "shorter_response",
    "random",
    "p_true",
    "verbalized_confidence",
    "sample_agreement",
    "normalized_sample_agreement",
    "negative_semantic_entropy",
)
CHOICE = re.compile(r"\s*\(?([A-J])\)?[.)]?\s*", re.IGNORECASE)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_jsonl(path):
    rows = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if not row["id"] or row["id"] in rows:
                raise ValueError(f"{path} has an empty or duplicate ID")
            rows[row["id"]] = row
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def check_ids(source, produced):
    if source.keys() != produced.keys():
        raise ValueError("Question and response IDs do not match")


def question_sample(table, id_column, count):
    rows = sorted(table.to_pylist(), key=lambda row: row[id_column])
    if len({row[id_column] for row in rows}) != len(rows):
        raise ValueError(f"Expected unique {id_column} values")
    return sorted(Random(SEED).sample(rows, count), key=lambda row: row[id_column])


def prepare(output, size):
    from pyarrow.parquet import read_table

    trivia = fetch(
        TRIVIA_URL, ROOT / "replay/triviaqa/sources/rc.web.nocontext-validation.parquet"
    )
    mmlu = fetch(
        MMLU_URL, ROOT / "replay/mmlu_pro/sources" / f"test-{MMLU_REVISION}.parquet"
    )
    short_rows = question_sample(
        read_table(trivia, columns=["question_id", "question", "answer"]),
        "question_id",
        size,
    )
    choice_rows = question_sample(
        read_table(
            mmlu,
            columns=[
                "question_id",
                "question",
                "options",
                "answer",
                "answer_index",
                "category",
            ],
        ),
        "question_id",
        size,
    )
    datasets = {"triviaqa": [], "mmlu_pro": []}
    for row in short_rows:
        aliases = sorted(
            {normalize_answer(alias) for alias in row["answer"]["normalized_aliases"]}
            - {""}
        )
        if not aliases:
            raise ValueError(f"TriviaQA question {row['question_id']} has no aliases")
        datasets["triviaqa"].append(
            {
                "id": row["question_id"],
                "question": row["question"],
                "prompt": f"{row['question']}\nReply with only the short answer.",
                "aliases": aliases,
            }
        )
    for row in choice_rows:
        letters = ascii_uppercase[: len(row["options"])]
        if not 2 <= len(letters) <= 10 or row["answer"] != letters[row["answer_index"]]:
            raise ValueError(
                f"MMLU-Pro question {row['question_id']} has inconsistent choices"
            )
        choices = "\n".join(
            f"{letter}. {option}"
            for letter, option in zip(letters, row["options"], strict=True)
        )
        datasets["mmlu_pro"].append(
            {
                "id": str(row["question_id"]),
                "question": row["question"],
                "prompt": f"{row['question']}\n{choices}\nReply with only the letter of the best answer.",
                "choices": letters,
                "answer": row["answer"],
                "category": row["category"],
            }
        )
    for name, rows in datasets.items():
        folder = output / name
        write_responses(folder / "questions.jsonl", rows)
        write_responses(
            folder / "prompts.jsonl",
            [{key: row[key] for key in ("id", "prompt")} for row in rows],
        )
        print(f"Prepared {len(rows)} {name} prompts", flush=True)


def collect(output, gpu_memory):
    loaded_at = perf_counter()
    backend = TransformersBackend(
        DEFAULT_MODEL,
        DEFAULT_REVISION,
        device="cuda",
        seed=SEED,
        local_files_only=True,
        gpu_memory=gpu_memory,
    )
    load_seconds = perf_counter() - loaded_at
    torch = backend.torch
    warmup_at = perf_counter()
    collect_responses(
        {"warmup": "What is the capital of Italy? Reply with only the short answer."},
        backend,
        max_new_tokens=TOKEN_LIMITS["triviaqa"],
        p_true=True,
        verbalized=True,
        samples=1,
    )
    torch.cuda.synchronize()
    warmup_seconds = perf_counter() - warmup_at
    run = {
        **backend.metadata,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": str(backend.model.config.dtype),
        "weight_memory_cap": gpu_memory,
        "offloaded_modules": [
            name
            for name, device in backend.model.hf_device_map.items()
            if device == "cpu"
        ],
        "model_load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
    }
    print(
        f"Loaded {run['model']} on {run['device']} in {load_seconds:.1f}s", flush=True
    )
    for name, token_limit in TOKEN_LIMITS.items():
        folder = output / name
        prompts = list(read_prompts(folder / "prompts.jsonl").items())
        torch.manual_seed(SEED)
        torch.cuda.reset_peak_memory_stats()
        all_records, totals = [], {}
        collection_start = perf_counter()
        for start in range(0, len(prompts), 16):
            records, timings = collect_responses(
                dict(prompts[start : start + 16]),
                backend,
                max_new_tokens=token_limit,
                p_true=True,
                verbalized=True,
                samples=SAMPLES,
            )
            all_records.extend(records)
            for method, timing in timings.items():
                total = totals.setdefault(method, {"calls": 0, "seconds": 0.0})
                total["calls"] += timing["calls"]
                total["seconds"] += timing["seconds"]
            print(
                f"{name} {len(all_records)}/{len(prompts)} in {perf_counter() - collection_start:.1f}s",
                flush=True,
            )
        torch.cuda.synchronize()
        timing = {
            **run,
            "prompt_count": len(prompts),
            "max_new_tokens": token_limit,
            "samples_per_prompt": SAMPLES,
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "collection_seconds": perf_counter() - collection_start,
            "methods": totals,
        }
        write_responses(folder / "responses.jsonl", all_records)
        write_json(folder / "timing.json", timing)


def option_match(text, choices):
    match = CHOICE.fullmatch(text)
    return match if match and match[1].upper() in choices else None


def choice_clusters(samples, choices):
    labels = []
    for answer in samples:
        match = option_match(answer, choices)
        labels.append(
            match[1].upper() if match else f"other:{normalize_answer(answer)}"
        )
    return labels


def nli_clusters(question, samples, model, tokenizer, torch, entailment):
    normalized = [normalize_answer(answer) for answer in samples]
    distinct = {}
    for key, answer in zip(normalized, samples, strict=True):
        if key:
            distinct.setdefault(key, f"Question: {question}\nAnswer: {answer.strip()}")
    pairs = list(combinations(distinct, 2))
    edges = set()
    if pairs:
        premises, hypotheses = [], []
        for first, second in pairs:
            premises.extend((distinct[first], distinct[second]))
            hypotheses.extend((distinct[second], distinct[first]))
        inputs = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=384,
            return_tensors="pt",
        )
        with torch.inference_mode():
            predictions = model(**inputs).logits.argmax(dim=-1).tolist()
        edges = {
            frozenset(pair)
            for pair, left, right in zip(
                pairs, predictions[::2], predictions[1::2], strict=True
            )
            if left == right == entailment
        }

    representatives = []
    memberships = []
    for answer in normalized:
        for index, representative in enumerate(representatives):
            if answer == representative or frozenset((answer, representative)) in edges:
                break
        else:
            index = len(representatives)
            representatives.append(answer)
        memberships.append(str(index))
    return memberships, int(bool(pairs)), 2 * len(pairs)


def cluster(output):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    start = perf_counter()
    options = {"revision": NLI_REVISION, "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL, **options)
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL, use_safetensors=True, **options
    ).eval()
    (entailment,) = [
        index
        for index, label in model.config.id2label.items()
        if label.lower() == "entailment"
    ]
    model_load_seconds = perf_counter() - start
    for dataset in TOKEN_LIMITS:
        folder = output / dataset
        questions = read_jsonl(folder / "questions.jsonl")
        responses = read_jsonl(folder / "responses.jsonl")
        check_ids(questions, responses)
        start = perf_counter()
        clusters, forwards, pairs = [], 0, 0
        for item_id, question in questions.items():
            samples = responses[item_id]["samples"]
            if dataset == "triviaqa":
                labels, calls, directed = nli_clusters(
                    question["question"], samples, model, tokenizer, torch, entailment
                )
                forwards += calls
                pairs += directed
            else:
                labels = choice_clusters(samples, question["choices"])
            clusters.append({"id": item_id, "clusters": labels})
        seconds = perf_counter() - start
        write_responses(folder / "clusters.jsonl", clusters)
        write_json(
            folder / "clustering.json",
            {
                "method": "bidirectional_nli"
                if dataset == "triviaqa"
                else "choice_identity",
                "model": NLI_MODEL if dataset == "triviaqa" else None,
                "revision": NLI_REVISION if dataset == "triviaqa" else None,
                "cpu_threads": torch.get_num_threads(),
                "nli_load_seconds": model_load_seconds
                if dataset == "triviaqa"
                else None,
                "nli_forward_calls": forwards,
                "nli_directional_pairs": pairs,
                "clustering_seconds": seconds,
            },
        )
        print(
            f"Clustered {len(clusters)} {dataset} responses in {seconds:.2f}s",
            flush=True,
        )


def score_rows(dataset, folder):
    questions = read_jsonl(folder / "questions.jsonl")
    responses = read_jsonl(folder / "responses.jsonl")
    clusters = read_jsonl(folder / "clusters.jsonl")
    check_ids(questions, responses)
    check_ids(questions, clusters)
    for item_id, question in questions.items():
        record = {**responses[item_id], "clusters": clusters[item_id]["clusters"]}
        if len(record["samples"]) != SAMPLES:
            raise ValueError(f"Expected {SAMPLES} samples for {item_id}")
        raw_answer = record["text"]
        if dataset == "triviaqa":
            correct = normalize_answer(raw_answer) in question["aliases"]
        else:
            match = option_match(raw_answer, question["choices"])
            correct = bool(match and match[1].upper() == question["answer"])
            if match:
                record["answer_span"] = list(match.span(1))
        _, response = read_response(record)
        scores = {
            method: compute_feature(FEATURES[method], response)
            for method in METHODS
            if method in FEATURES
        }
        scores["random"] = 0.0
        scores["negative_semantic_entropy"] = -compute_feature(
            FEATURES["semantic_entropy"], response
        )
        row = {
            "id": item_id,
            "correct": int(correct),
            **{method: scores[method] for method in METHODS},
        }
        if dataset == "mmlu_pro":
            row["answer_span_logprob"] = compute_feature(
                FEATURES["answer_span_logprob"], response
            )
        else:
            lexical = replace(
                response,
                clusters=tuple(normalize_answer(answer) for answer in response.samples),
            )
            row["negative_lexical_entropy"] = -compute_feature(
                FEATURES["semantic_entropy"], lexical
            )
        yield row


def comparison_rows(scores_dir):
    for dataset in TOKEN_LIMITS:
        rows = read_csv(scores_dir / f"{dataset}_scores.csv")
        extra = (
            "answer_span_logprob"
            if dataset == "mmlu_pro"
            else "negative_lexical_entropy"
        )
        methods = (*METHODS, extra)
        if len({row["id"] for row in rows}) != len(rows) or not all(
            row["id"] for row in rows
        ):
            raise ValueError(f"Expected unique nonempty IDs for {dataset}")
        available = {
            method: sum(bool(row[method]) for row in rows) for method in methods
        }
        scored = [
            ScoredRow(
                row["id"],
                tuple(float(row[method]) for method in methods),
                (row["correct"] == "1",),
            )
            for row in rows
            if all(row[method] for method in methods)
        ]
        for row in rows:
            if row["correct"] not in ("0", "1"):
                raise ValueError(f"Invalid correctness value for {row['id']}")
        result = compare_scores(
            scored,
            names=methods,
            graders=("correct",),
            baseline=BASELINE,
            coverages=DEFAULT_COVERAGES,
            split="all",
            bootstrap=RESAMPLES,
            seed=BOOTSTRAP_SEED,
        )["graders"]["correct"]["scores"]
        labels = [row.outcomes[0] for row in scored]
        for method in methods:
            ranking = result[method]
            calibration = {}
            if method in ("p_true", "verbalized_confidence"):
                index = methods.index(method)
                calibration = calibration_metrics(
                    [row.scores[index] for row in scored], labels
                )
            metrics = {
                "dataset": dataset,
                "method": method,
                "total_rows": len(rows),
                "scored_rows": available[method],
                "matched_rows": len(scored),
                "matched_correct": sum(labels),
                "auroc": ranking["auroc"],
                "normalized_aurc_gain": ranking["normalized_aurc_gain"],
                "brier_score": calibration.get("brier_score"),
                "ece": calibration.get("ece"),
            }
            for budget in ranking["budgets"]:
                suffix = str(round(budget["target_coverage"] * 100))
                metrics[f"risk_{suffix}"] = budget["error_rate"]
            at_75 = ranking["budgets"][-1]
            metrics["accepted_75"] = at_75["accepted"]
            metrics["delta_75_vs_mean"] = at_75["risk_delta"]
            metrics["delta_75_ci_low"], metrics["delta_75_ci_high"] = at_75[
                "risk_delta_95"
            ]
            yield {
                key: f"{value:.10f}" if isinstance(value, float) else value
                for key, value in metrics.items()
            }


def summarize(work, results):
    results.mkdir(parents=True, exist_ok=True)
    costs, metadata = [], {}
    for dataset in TOKEN_LIMITS:
        folder = work / dataset
        scored = list(score_rows(dataset, folder))
        write_report(results / f"{dataset}_scores.csv", scored)
        timing = json.loads((folder / "timing.json").read_text(encoding="utf-8"))
        clustering = json.loads(
            (folder / "clustering.json").read_text(encoding="utf-8")
        )
        count = len(scored)
        if timing["prompt_count"] != count or timing["samples_per_prompt"] != SAMPLES:
            raise ValueError(f"Call metadata differs from scored rows for {dataset}")
        for method, per_question in (
            ("greedy", 1),
            ("p_true", 1),
            ("verbalized_confidence", 1),
            ("sample", SAMPLES),
        ):
            observed = timing["methods"][method]
            if (
                observed["calls"] != count * per_question
                or not isfinite(observed["seconds"])
                or observed["seconds"] <= 0
            ):
                raise ValueError(f"Invalid collection cost for {dataset}/{method}")
            costs.append(
                {
                    "dataset": dataset,
                    "operation": method,
                    "qwen_calls": observed["calls"],
                    "nli_forward_calls": 0,
                    "nli_directional_pairs": 0,
                    "seconds": observed["seconds"],
                    "ms_per_question": 1000 * observed["seconds"] / count,
                }
            )
        seconds = clustering["clustering_seconds"]
        costs.append(
            {
                "dataset": dataset,
                "operation": clustering["method"],
                "qwen_calls": 0,
                "nli_forward_calls": clustering["nli_forward_calls"],
                "nli_directional_pairs": clustering["nli_directional_pairs"],
                "seconds": seconds,
                "ms_per_question": 1000 * seconds / count,
            }
        )
        metadata[dataset] = {
            "dataset_source": TRIVIA_URL if dataset == "triviaqa" else MMLU_URL,
            "split": "web validation" if dataset == "triviaqa" else "test",
            "greedy_correct": sum(row["correct"] for row in scored),
            "decoding": {
                "batch_size": 1,
                "sample_temperature": 0.7,
                "sample_top_p": 0.9,
                "sample_top_k": 0,
                "seed": SEED,
            },
            "collection": timing,
            "clustering": clustering,
        }
    write_report(results / "comparison.csv", list(comparison_rows(results)))
    write_report(results / "costs.csv", costs)
    write_json(results / "run.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "replay/black_box")
    subcommands = parser.add_subparsers(dest="operation", required=True)
    preparation = subcommands.add_parser(
        "prepare", help="Prepare frozen dataset samples."
    )
    preparation.add_argument("--size", type=int, default=SAMPLE_SIZE)
    collection = subcommands.add_parser(
        "collect", help="Collect saved responses on CUDA."
    )
    collection.add_argument("--gpu-memory", default=GPU_MEMORY)
    subcommands.add_parser("cluster", help="Cluster saved samples on CPU.")
    scoring = subcommands.add_parser(
        "summarize", help="Score saved answers and export results."
    )
    scoring.add_argument("--results", type=Path, default=ROOT / "results/black_box")
    replay = subcommands.add_parser(
        "replay", help="Recompute comparisons from saved scores."
    )
    replay.add_argument("--scores-dir", type=Path, default=ROOT / "results/black_box")
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare(args.work_dir, args.size)
    elif args.operation == "collect":
        collect(args.work_dir, args.gpu_memory)
    elif args.operation == "cluster":
        cluster(args.work_dir)
    elif args.operation == "summarize":
        summarize(args.work_dir, args.results)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_report(args.output, list(comparison_rows(args.scores_dir)))


if __name__ == "__main__":
    main()
