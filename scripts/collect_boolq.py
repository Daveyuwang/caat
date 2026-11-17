"""Score fixed BoolQ answers and verify them with a second model on the same passage."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from common import ROOT, read_csv, write_report
from prepare_boolq import dataset_file, digest, normalized

MODELS = {
    "generate": (
        "Qwen/Qwen2.5-7B-Instruct",
        "a09a35458c702b33eeacc393d103063234e8bc28",
    ),
    "verify": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
}


def dataset(manifest):
    import pyarrow.parquet as pq

    sources = {
        s: pq.read_table(dataset_file(s), columns=["question", "passage"]).to_pylist()
        for s in ("train", "validation")
    }
    items = {}
    for row in read_csv(manifest):
        doc = sources[row["split"]][int(row["row"])]
        passage = normalized(doc["passage"])
        if (
            row["passage_sha256"] != digest(passage)
            or row["id"] != digest(passage + "\n" + normalized(doc["question"]))
            or row["id"] in items
        ):
            raise ValueError("BoolQ manifest does not identify the source text")
        items[row["id"]] = doc
    return items


def collect(stage, manifest, directory, batch_size):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, revision = MODELS[stage]
    directory.mkdir(parents=True, exist_ok=True)
    docs = dataset(manifest)
    previous = {}
    if stage == "verify":
        rows = read_csv(directory / "generate.csv")
        previous = {r["id"]: r for r in rows}
        if len(rows) != len(previous) or previous.keys() != docs.keys():
            raise ValueError("generation IDs must match the manifest")
    load_start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "12GiB", "cpu": "40GiB"},
    ).eval()
    names = ("Yes", "No") if stage == "generate" else ("True", "False")
    label_ids = [tokenizer.encode(n, add_special_tokens=False) for n in names]
    if any(len(v) != 1 for v in label_ids) or label_ids[0] == label_ids[1]:
        raise ValueError("expected distinct single-token choices")
    tokens = [v[0] for v in label_ids]
    prompts = []
    for i, doc in docs.items():
        question = f"Passage: {doc['passage']}\nQuestion: {doc['question']}\n"
        if stage == "generate":
            question += "Answer with exactly Yes or No."
        else:
            answer = previous[i]["answer"]
            if answer not in ("Yes", "No") or previous[i]["output_sha256"] != digest(
                answer
            ):
                raise ValueError("invalid saved answer")
            question += f"Proposed answer: {answer}\nReply with exactly True if the proposed answer is correct or False if it is incorrect."
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append((i, text))
    prompts.sort(key=lambda pair: len(pair[1]))
    load_seconds = perf_counter() - load_start
    start = perf_counter()
    results = []
    batches = token_count = 0
    logit_checks = 0
    with torch.inference_mode():
        for offset in range(0, len(prompts), batch_size):
            chunk = prompts[offset : offset + batch_size]
            batch = tokenizer(
                [s for _, s in chunk],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(model.device)
            logits = (
                model(**batch, logits_to_keep=1, use_cache=False).logits[:, -1].float()
            )
            chosen = logits[:, tokens]
            probs = chosen.softmax(dim=-1)
            log_z = logits.logsumexp(dim=-1)
            if batches % 20 == 0:
                direct = logits.log_softmax(dim=-1)[:, tokens]
                torch.testing.assert_close(
                    direct, chosen - log_z[:, None], atol=3e-6, rtol=1e-5
                )
                torch.testing.assert_close(
                    direct.softmax(dim=-1), probs, atol=1e-6, rtol=1e-5
                )
                logit_checks += len(chunk)
            for (i, text), pair, prob, z in zip(
                chunk, chosen.tolist(), probs.tolist(), log_z.tolist(), strict=True
            ):
                row = {
                    "id": i,
                    "prompt_sha256": digest(text),
                    "first_logit": pair[0],
                    "second_logit": pair[1],
                    "log_normalizer": z,
                }
                if stage == "generate":
                    answer = names[int(pair[1] > pair[0])]
                    row.update(
                        answer=answer,
                        output_sha256=digest(answer),
                        winner_logprob=max(pair) - z,
                        gap=abs(pair[0] - pair[1]),
                        confidence=max(prob),
                    )
                else:
                    row.update(
                        output_sha256=previous[i]["output_sha256"], p_true=prob[0]
                    )
                results.append(row)
            token_count += int(batch.attention_mask.sum())
            batches += 1
            if batches % 25 == 0:
                print(f"{stage}: {len(results)}/{len(prompts)}", flush=True)
    elapsed = perf_counter() - start
    write_report(directory / f"{stage}.csv", results)
    info = dict(
        model=model_id,
        revision=revision,
        device=torch.cuda.get_device_name(0),
        dtype="bfloat16",
        transformers=transformers.__version__,
        torch=torch.__version__,
        label_tokens=dict(zip(names, tokens)),
        logical_requests=len(results),
        batches=batches,
        input_tokens=token_count,
        load_seconds=load_seconds,
        wall_seconds=elapsed,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        checked_rows=logit_checks,
        batch_size=batch_size,
        teacher_forced_replays=0,
    )
    (directory / f"{stage}.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf8"
    )
    print(info, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=MODELS)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "results/boolq/manifest.csv"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "replay/boolq")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    collect(args.stage, args.manifest, args.output, args.batch_size)
