"""Score saved benchmark generations with UQLM or LM-Polygraph and join the results."""

import argparse
import json
from hashlib import sha256
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from time import perf_counter

from caat.table import read_external_scores, write_rows

from common import read_csv
from study_black_box import check_ids, read_jsonl


SCORES = {"uqlm": "uqlm_exact_match", "lm-polygraph": "lm_polygraph_rouge_l"}
IDENTITY = ("id", "generation_sha256")


def generation_digest(response):
    payload = json.dumps([response["text"], response["samples"]], ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def score_responses(responses, provider, output):
    answers = [row["text"] for row in responses.values()]
    samples = [row["samples"] for row in responses.values()]
    if any(len(draws) < 2 for draws in samples):
        raise ValueError("each response needs at least two saved samples")
    if provider == "uqlm":
        import torch
        from uqlm import BlackBoxUQ

        start = perf_counter()
        frame = (
            BlackBoxUQ(scorers=["exact_match"], device=torch.device("cpu"))
            .score(
                responses=answers, sampled_responses=samples, show_progress_bars=False
            )
            .to_df()
        )
        if (
            frame["response"].tolist() != answers
            or frame["sampled_responses"].tolist() != samples
        ):
            raise ValueError("UQLM reordered or changed the saved generations")
        values = frame["exact_match"].tolist()
    else:
        from lm_polygraph.estimators import LexicalSimilarity

        start = perf_counter()
        values = (-LexicalSimilarity("rougeL")({"sample_texts": samples})).tolist()
    seconds = perf_counter() - start
    if len(values) != len(responses) or not all(isfinite(value) for value in values):
        raise ValueError(f"{provider} must return one finite score per answer")
    count = write_rows(
        output,
        (*IDENTITY, SCORES[provider]),
        (
            (item_id, generation_digest(response), score)
            for (item_id, response), score in zip(
                responses.items(), values, strict=True
            )
        ),
    )
    print(f"{provider} {version(provider)} scored {count} answers in {seconds:.3f}s.")


def merge_scores(
    responses, builtin_path, external_paths, output, *, keep_incomplete=False
):
    builtin = read_csv(builtin_path)
    indexed = {row["id"]: row for row in builtin}
    if len(indexed) != len(builtin):
        raise ValueError("duplicate question IDs in the built-in scores")
    check_ids(responses, indexed)
    fields = tuple(builtin[0])
    external = {
        SCORES[provider]: read_external_scores(path, SCORES[provider], IDENTITY)
        for provider, path in external_paths.items()
    }
    if set(fields).intersection(external):
        raise ValueError("external score columns already exist in the built-in scores")

    def rows():
        for row in builtin:
            item_id = row["id"]
            digest = generation_digest(responses[item_id])
            values = []
            for name, source in external.items():
                match = source.pop((item_id,), None)
                if match is None or match[0] != digest:
                    raise ValueError(f"{name} is missing or misaligned for {item_id!r}")
                values.append(match[1])
            if keep_incomplete or all(value != "" for value in row.values()):
                yield (*row.values(), *values)
        if any(external.values()):
            raise ValueError("an external score file has unexpected question IDs")

    count = write_rows(output, (*fields, *external), rows())
    print(f"Joined both scorers to {count} of {len(builtin)} score rows.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in (*SCORES, "join"):
        command = commands.add_parser(operation)
        command.add_argument("responses", type=Path)
        command.add_argument("--output", type=Path, required=True)
    join = commands.choices["join"]
    join.add_argument("--builtins", type=Path, required=True)
    join.add_argument("--uqlm", type=Path, required=True)
    join.add_argument("--lm-polygraph", type=Path, required=True)
    join.add_argument("--keep-incomplete", action="store_true")
    args = parser.parse_args()
    inputs = [args.responses]
    if args.operation == "join":
        external = {"uqlm": args.uqlm, "lm-polygraph": args.lm_polygraph}
        inputs.extend((args.builtins, *external.values()))
    if args.output.resolve() in {path.resolve() for path in inputs}:
        parser.error("output would overwrite an input file")
    try:
        responses = read_jsonl(args.responses)
        if args.operation == "join":
            merge_scores(
                responses,
                args.builtins,
                external,
                args.output,
                keep_incomplete=args.keep_incomplete,
            )
        else:
            score_responses(responses, args.operation, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
