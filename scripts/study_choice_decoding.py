"""Measure option-likelihood decoding on saved multiple-choice prompts."""

import argparse
from pathlib import Path
from time import perf_counter

from caat.collect import TransformersBackend, batches

from common import write_report
from study_black_box import GPU_MEMORY, SEED, read_jsonl, write_json


def study(questions, model, revision, size):
    backend = TransformersBackend(
        model,
        revision,
        device="cuda",
        seed=SEED,
        local_files_only=True,
        gpu_memory=GPU_MEMORY,
    )
    rows = []
    start = perf_counter()
    calls = 0
    for group in batches(questions.values(), size):
        formatted, batch = backend.batch_inputs(
            [question["prompt"] for question in group]
        )
        prefixes = [
            [token for token, keep in zip(ids, mask, strict=True) if keep]
            for ids, mask in zip(
                batch["input_ids"].tolist(),
                batch["attention_mask"].tolist(),
                strict=True,
            )
        ]
        logits = backend.generate_tokens(batch, 1, tokens=True, use_cache=False).logits[
            0
        ]
        for i, (question, prompt, prefix) in enumerate(
            zip(group, formatted, prefixes, strict=True)
        ):
            labels = question["choices"]
            tokens = backend.choice_tokens(prompt, prefix, labels)
            chosen = labels[logits[i, tokens].argmax().item()]
            rows.append(
                {
                    "id": question["id"],
                    "choice": chosen,
                    "correct": int(chosen == question["answer"]),
                }
            )
        calls += 1
    backend.torch.cuda.synchronize()
    return rows, {
        **backend.metadata,
        "prompts": len(rows),
        "batch_size": size,
        "batches": calls,
        "seconds": perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.output.suffix != ".csv" or args.questions.resolve() in {
        args.output.resolve(),
        args.output.with_suffix(".json").resolve(),
    }:
        parser.error("provide a new CSV output path")
    questions = read_jsonl(args.questions)
    rows, metadata = study(questions, args.model, args.revision, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output, rows)
    write_json(args.output.with_suffix(".json"), metadata)


if __name__ == "__main__":
    main()
