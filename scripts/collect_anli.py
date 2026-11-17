"""Collect ANLI label likelihoods and verify the original predicted relation."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from collect_boolq import MODELS
from common import ROOT, read_csv, write_report
from prepare_anli import SPLITS, source
from prepare_boolq import digest, normalized

LABELS = ("entailment", "neutral", "contradiction")


def inputs(stage, manifest, directory):
    ids = {row["id"]: row for row in read_csv(manifest)}
    previous = {}
    if stage == "verify":
        generated = read_csv(directory / "generate.csv")
        previous = {r["id"]: r for r in generated}
        if previous.keys() != ids.keys() or len(previous) != len(generated):
            raise ValueError("generator IDs differ from the manifest")
    semantics = LABELS[1:] + LABELS[:1] if stage == "rotate" else LABELS
    for split in SPLITS:
        for doc in source(split, ["uid", "premise", "hypothesis"]):
            uid = doc["uid"]
            if uid not in ids:
                continue
            row = ids[uid]
            if row["split"] != split or any(
                row[field + "_sha256"] != digest(normalized(doc[field]))
                for field in ("premise", "hypothesis")
            ):
                raise ValueError(f"invalid manifest row {uid}")
            if stage == "rotate" and row["role"] != "design":
                continue
            msg = f"Premise: {doc['premise']}\nHypothesis: {doc['hypothesis']}\n"
            if stage == "verify":
                label = previous[uid]["answer"]
                if (
                    label not in LABELS
                    or digest(label) != previous[uid]["output_sha256"]
                ):
                    raise ValueError(f"invalid saved answer {uid}")
                msg += f"Proposed relation: {label}\nReply with exactly True if the proposed relation is correct or False if it is incorrect."
            else:
                msg += "Does the premise entail the hypothesis, leave it neutral, or contradict it?\n"
                msg += "\n".join(
                    f"{code}: {label}"
                    for code, label in zip("ABC", semantics, strict=True)
                )
                msg += "\nReply with exactly A, B or C."
            yield uid, msg, previous.get(uid, {}).get("output_sha256", "")


def collect(stage, manifest, directory, batch_size):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name, rev = MODELS["verify" if stage == "verify" else "generate"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=rev, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=rev,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "12GiB", "cpu": "40GiB"},
    ).eval()
    options = ("True", "False") if stage == "verify" else tuple("ABC")
    encoded = [tokenizer.encode(v, add_special_tokens=False) for v in options]
    if any(len(v) != 1 for v in encoded) or len({v[0] for v in encoded}) != len(
        options
    ):
        raise ValueError("options must map to distinct single tokens")
    tokens = [v[0] for v in encoded]
    prompts = [
        (
            uid,
            tokenizer.apply_chat_template(
                [{"role": "user", "content": msg}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            key,
        )
        for uid, msg, key in inputs(stage, manifest, directory)
    ]
    prompts.sort(key=lambda row: len(row[1]))
    results = []
    count = checks = 0
    start = perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(prompts), batch_size):
            items = prompts[offset : offset + batch_size]
            batch = tokenizer(
                [r[1] for r in items],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(model.device)
            logits = (
                model(**batch, logits_to_keep=1, use_cache=False).logits[:, -1].float()
            )
            chosen = logits[:, tokens]
            z = logits.logsumexp(dim=-1)
            if offset // batch_size % 20 == 0:
                torch.testing.assert_close(
                    chosen - z[:, None],
                    logits.log_softmax(dim=-1)[:, tokens],
                    atol=3e-6,
                    rtol=1e-5,
                )
                checks += len(items)
            for (uid, text, key), values, normalizer in zip(
                items, chosen.tolist(), z.tolist(), strict=True
            ):
                row = {
                    "id": uid,
                    "prompt_sha256": digest(text),
                    "log_z": normalizer,
                    **dict(zip(options, values, strict=True)),
                }
                if stage != "verify":
                    idx = max(range(3), key=values.__getitem__)
                    label = LABELS[(idx + int(stage == "rotate")) % 3]
                    row.update(answer=label, output_sha256=digest(label))
                else:
                    row["output_sha256"] = key
                results.append(row)
            count += int(batch.attention_mask.sum())
    elapsed = perf_counter() - start
    if len(results) != (400 if stage == "rotate" else 3014) or len(
        {r["id"] for r in results}
    ) != len(results):
        raise ValueError("incomplete inference stage")
    directory.mkdir(parents=True, exist_ok=True)
    write_report(directory / (stage + ".csv"), results)
    cost = dict(
        model=model_name,
        revision=rev,
        label_tokens=dict(zip(options, tokens, strict=True)),
        logical_requests=len(results),
        batches=(len(results) + batch_size - 1) // batch_size,
        input_tokens=count,
        wall_seconds=elapsed,
        device=torch.cuda.get_device_name(0),
        checked_rows=checks,
        torch=torch.__version__,
        transformers=transformers.__version__,
    )
    (directory / (stage + ".json")).write_text(
        json.dumps(cost, indent=2) + "\n", encoding="utf8"
    )
    print(stage, cost, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("generate", "rotate", "verify"))
    p.add_argument("--manifest", type=Path, default=ROOT / "results/anli/manifest.csv")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch size must be positive")
    collect(args.stage, args.manifest, args.output, args.batch_size)
