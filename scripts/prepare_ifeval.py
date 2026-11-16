"""Build a CAAT CSV from saved HELM responses graded with Google's IFEval."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from caat.features import (
    FULL_SCORE_FIELDS,
    SavedResponse,
    full_scores,
    mean_token_logprob,
    token_logprobs,
)

from common import fetch

GRADER_REVISION = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
GRADER_URL = (
    f"https://raw.githubusercontent.com/google-research/google-research/{GRADER_REVISION}"
    "/instruction_following_eval"
)
DATASET_URL = (
    "https://huggingface.co/datasets/google/IFEval/resolve/"
    "966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl"
)
HELM_BASE = (
    "https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output"
    "/runs/v1.0.0/ifeval:model="
)
HELM_MODELS = {
    "qwen": "qwen_qwen2.5-7b-instruct-turbo",
    "llama": "meta_llama-3.1-8b-instruct-turbo",
}
SAMPLE_SIZE = 24


def helm_strict_passes(run_url, path):
    path = fetch(f"{run_url}/per_instance_stats.json", path)
    outcomes = {}
    with path.open(encoding="utf-8") as source:
        for entry in json.load(source):
            for stat in entry["stats"]:
                if stat["name"]["name"] != "ifeval_strict_accuracy":
                    continue
                key = int(entry["instance_id"].removeprefix("id"))
                if key in outcomes or stat["count"] != 1 or not 0 <= stat["sum"] <= 1:
                    raise ValueError(f"Invalid published IFEval score for key {key}")
                outcomes[key] = int(stat["sum"] == 1)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", choices=HELM_MODELS, default="qwen")
    parser.add_argument(
        "--full", action="store_true", help="Grade every archived response."
    )
    args = parser.parse_args()
    output = args.output_dir or Path("replay") / (
        "ifeval" if args.model == "qwen" else "ifeval_llama"
    )
    run_url = HELM_BASE + HELM_MODELS[args.model]

    import nltk
    from langdetect import DetectorFactory

    cache = output / "sources"
    nltk_data = str(cache / "nltk_data")
    nltk.data.path.insert(0, nltk_data)
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        nltk.download(
            "punkt_tab", download_dir=nltk_data, quiet=True, raise_on_error=True
        )

    grader = cache / "google" / GRADER_REVISION / "instruction_following_eval"
    for filename in (
        "evaluation_lib.py",
        "instructions_registry.py",
        "instructions.py",
        "instructions_util.py",
    ):
        fetch(f"{GRADER_URL}/{filename}", grader / filename)
    (grader / "__init__.py").touch()
    sys.path.insert(0, str(grader.parent))
    from instruction_following_eval.evaluation_lib import (
        InputExample,
        test_instruction_following_strict,
    )

    dataset = fetch(DATASET_URL, cache / "ifeval_input_data.jsonl")
    with dataset.open(encoding="utf-8") as source:
        examples = {row["key"]: InputExample(**row) for row in map(json.loads, source)}
    with fetch(
        f"{run_url}/scenario_state.json", cache / f"{args.model}_scenario_state.json"
    ).open(encoding="utf-8") as source:
        states = json.load(source)["request_states"]
    sample = states if args.full else states[:SAMPLE_SIZE]

    random.seed(0)
    DetectorFactory.seed = 0
    rows = []
    for state in sample:
        instance = state["instance"]
        example = examples[int(instance["id"].removeprefix("id"))]
        prompt = instance["input"]["text"]
        instruction_kwargs = [
            {name: value for name, value in params.items() if value is not None}
            for params in instance["extra_data"]["instruction_kwargs"]
        ]
        if (
            instance["split"] != "test"
            or prompt != example.prompt.strip()
            or instance["extra_data"]["instruction_ids"] != example.instruction_id_list
            or instruction_kwargs != example.kwargs
        ):
            raise ValueError(
                f"Archived prompt or instructions differ for IFEval key {example.key}"
            )
        if state["prompt_truncated"] or not state["result"]["success"]:
            raise ValueError(f"Invalid saved generation for IFEval key {example.key}")
        (completion,) = state["result"]["completions"]
        graded = test_instruction_following_strict(
            example, {example.prompt: completion["text"]}
        )
        scores = token_logprobs(completion, example.key)
        response = SavedResponse(completion["text"], completion["tokens"], scores)
        row = (
            example.key,
            mean_token_logprob(response),
            int(graded.follow_all_instructions),
        )
        if args.full:
            needs_language = any(
                name.startswith(("change_case:english_", "language:"))
                for name in example.instruction_id_list
            )
            group = "language_or_case" if needs_language else "other"
            row = (*row, *full_scores(response), group)
        rows.append(row)

    expected = len(examples) if args.full else SAMPLE_SIZE
    if len({row[0] for row in rows}) != len(rows) or len(rows) != expected:
        raise ValueError(f"Expected {expected} distinct IFEval keys")
    fields = ("id", "mean_token_logprob", "strict_pass")
    if args.full:
        published = helm_strict_passes(
            run_url, cache / f"{args.model}_per_instance_stats.json"
        )
        if set(published) != examples.keys():
            raise ValueError(
                "Published IFEval scores do not match the pinned dataset keys"
            )
        fields = (*fields, *FULL_SCORE_FIELDS, "group", "helm_strict_pass")
        rows = [(*row, published[row[0]]) for row in rows]
    path = output / ("predictions_full.csv" if args.full else "predictions.csv")
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
