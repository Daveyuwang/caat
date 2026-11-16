"""Build a CAAT CSV from archived HELM MMLU-Pro completions and official answers."""

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from string import ascii_uppercase

from common import FULL_SCORE_FIELDS, fetch, full_scores, token_logprobs

DATASET_REVISION = "3373e0b32277875b8db2aa555a333b78a08477ea"
DATASET_URL = (
    f"https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/{DATASET_REVISION}"
    "/data/test-00000-of-00001.parquet"
)
HELM_BASE = (
    "https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output"
    "/runs/v1.0.0/mmlu_pro:subset=all,use_chain_of_thought=true,use_few_shot=false,"
    "model="
)
HELM_MODELS = {
    "qwen": "qwen_qwen2.5-7b-instruct-turbo",
    "llama": "meta_llama-3.1-8b-instruct-turbo",
}
SAMPLE_SIZE = 24


def extract_answer(text: str) -> str:
    # Use the ordered patterns in HELM's GPQAChainOfThoughtMetric (b1bc37e).
    for pattern in (
        r"answer is \(?([A-J])\)?",
        r"\.*\[aA\]nswer:\s*\(?([A-J])\)?",
        r"correct answer is [^A-Za-z]*([A-J])",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def tiger_answer(text):
    """TIGER-Lab's extraction at 26d41d0, with no random guess for missing answers."""
    match = re.search(r"answer is \(?([A-J])\)?", text)
    if not match:
        match = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1)
    final = re.findall(r"\b[A-J]\b", text)
    return final[-1] if final else ""


def answer_token_score(completion):
    text = completion["text"]
    match = re.search(r"(?i:\banswer\s*(?:is\b|:))\s*\(?([A-J])\b", text)
    tokens = completion["tokens"]
    if match is None or not "".join(token["text"] for token in tokens).startswith(text):
        return None
    position = match.start(1)
    end = 0
    for token in tokens:
        end += len(token["text"])
        if end > position:
            return token["logprob"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", choices=HELM_MODELS, default="qwen")
    parser.add_argument("--full", action="store_true", help="Grade every archived response.")
    args = parser.parse_args()
    output = args.output_dir or Path("replay") / ("mmlu_pro" if args.model == "qwen" else "mmlu_pro_llama")

    from pyarrow.parquet import read_table

    cache = output / "sources"
    cache.mkdir(parents=True, exist_ok=True)
    gold_file = cache / f"test-{DATASET_REVISION}.parquet"
    run_file = cache / f"{args.model}_scenario_state.json"
    run_url = HELM_BASE + HELM_MODELS[args.model] + "/scenario_state.json"
    for path, url in ((gold_file, DATASET_URL), (run_file, run_url)):
        fetch(url, path)

    columns = ["question_id", "question", "options", "answer", "answer_index", "category"]
    official = {row["question_id"]: row for row in read_table(gold_file, columns=columns).to_pylist()}
    with run_file.open(encoding="utf-8") as source:
        states = json.load(source)["request_states"]
    sample = states if args.full else states[:SAMPLE_SIZE]

    rows = []
    for state in sample:
        instance = state["instance"]
        question_id = int(instance["id"].removeprefix("id"))
        gold = official[question_id]
        options = dict(zip(ascii_uppercase, gold["options"]))
        reference = gold["answer"]
        helm_correct = [
            ref["output"]["text"] for ref in instance["references"] if "correct" in ref["tags"]
        ]
        if (
            instance["split"] != "test"
            or instance["input"]["text"] != gold["question"]
            or state["output_mapping"] != options
            or reference != ascii_uppercase[gold["answer_index"]]
            or helm_correct != [options[reference]]
        ):
            raise ValueError(f"Archived question or answer differs from official test row {question_id}")
        if not state["result"]["success"] or state["prompt_truncated"]:
            raise ValueError(f"Invalid saved generation for MMLU-Pro question {question_id}")
        completion, = state["result"]["completions"]
        scores = token_logprobs(completion, question_id)
        text = completion["text"]
        prediction = extract_answer(text)
        row = (question_id, prediction, reference, int(prediction == reference),
               mean(scores), gold["category"])
        if args.full:
            tiger = tiger_answer(text)
            row = (*row, *full_scores(completion, scores), tiger,
                   int(tiger == reference), answer_token_score(completion))
        rows.append(row)

    expected = 1000 if args.full else SAMPLE_SIZE
    if len({row[0] for row in rows}) != len(rows) or len(rows) != expected:
        raise ValueError(f"Expected {expected} distinct MMLU-Pro questions")
    path = output / ("predictions_full.csv" if args.full else "predictions.csv")
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        fields = ("id", "prediction", "reference", "correct", "mean_token_logprob", "group")
        if args.full:
            fields = (*fields, *FULL_SCORE_FIELDS, "tiger_prediction", "tiger_correct", "answer_token_logprob")
        writer.writerow(fields)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
