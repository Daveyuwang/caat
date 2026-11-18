"""Score one-token answer choices with the pinned Qwen generator."""

import json
from time import perf_counter
from collect_boolq import MODELS
from common import write_report
from prepare_boolq import digest


def collect_choices(rows, choices, out, dataset_revision):
    from caat.collect import TransformersBackend
    import torch

    name, revision = MODELS["generate"]
    engine = TransformersBackend(
        name,
        revision,
        device="cuda",
        seed=42,
        local_files_only=False,
        gpu_memory="12GiB",
    )
    tokenizer = engine.tokenizer
    tokens = [tokenizer.encode(c, add_special_tokens=False) for c in choices]
    token_ids = list(range(32, 32 + len(choices)))
    if tokens != [[t] for t in token_ids] or engine.model.dtype != torch.bfloat16:
        raise ValueError(
            "collection requires the frozen choice tokens and BF16 weights"
        )
    formatted = sorted(
        [(m, engine.format_prompt(text)) for m, text in rows],
        key=lambda pair: len(pair[1]),
    )
    results = []
    total_tokens = 0
    start = perf_counter()
    with torch.inference_mode():
        for i in range(0, len(rows), 16):
            group = formatted[i : i + 16]
            inputs = tokenizer(
                [s for _, s in group],
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(engine.model.device)
            logits = (
                engine.model(**inputs, logits_to_keep=1, use_cache=False)
                .logits[:, -1, token_ids]
                .float()
                .tolist()
            )
            total_tokens += int(inputs.attention_mask.sum())
            for (m, text), values in zip(group, logits, strict=True):
                choice = choices[max(range(len(values)), key=values.__getitem__)]
                top, second = sorted(values, reverse=True)[:2]
                results.append(
                    dict(
                        id=m["id"],
                        prompt_sha256=digest(text),
                        **dict(zip(choices, values)),
                        answer=choice,
                        output_sha256=m[choice + "_sha256"],
                        gap=top - second,
                    )
                )
    elapsed = perf_counter() - start
    write_report(out / "predictions.csv", sorted(results, key=lambda r: r["id"]))
    summary = dict(
        **engine.metadata,
        dtype="bfloat16",
        logical_requests=len(rows),
        batches=(len(rows) + 15) // 16,
        input_tokens=total_tokens,
        wall_seconds=elapsed,
        dataset_revision=dataset_revision,
    )
    (out / "collection.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(summary)
