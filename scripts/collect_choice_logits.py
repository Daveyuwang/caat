"""Score one-token answer choices with the pinned Qwen generator."""

import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from collect_boolq import MODELS
from common import write_report
from prepare_boolq import digest


def streamed_qwen(name, revision, torch):
    """Read each BF16 tensor from its pinned shard as Accelerate needs it."""
    from accelerate import cpu_offload, init_empty_weights
    from accelerate.utils.offload import OffloadedWeightsLoader
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    folder = Path(
        snapshot_download(
            name, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt"]
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(folder, padding_side="left")
    config = AutoConfig.from_pretrained(folder)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    shards = json.loads((folder / "model.safetensors.index.json").read_bytes())[
        "weight_map"
    ]
    if set(shards) != set(model.state_dict()):
        raise ValueError("Qwen checkpoint parameters differ from the model")
    weights = OffloadedWeightsLoader(
        index={
            k: {"safetensors_file": str(folder / shard)} for k, shard in shards.items()
        }
    )
    cpu_offload(model, execution_device=torch.device("cuda:0"), state_dict=weights)
    meta = dict(
        model=name,
        revision=revision,
        device=torch.cuda.get_device_name(),
        seed=42,
        torch=version("torch"),
        transformers=version("transformers"),
    )
    return model.eval(), tokenizer, meta


def collect_choices(
    rows,
    choices,
    out,
    dataset_revision,
    *,
    batch_size=16,
    memory_gib=None,
    checkpoint=None,
):
    from caat.collect import TransformersBackend
    import torch

    name, revision = MODELS["generate"]
    if memory_gib is not None:
        size = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(memory_gib * 2**30 / size)
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model, tokenizer, metadata = streamed_qwen(name, revision, torch)
    else:
        engine = TransformersBackend(
            name,
            revision,
            device="cuda",
            seed=42,
            local_files_only=False,
            gpu_memory="12GiB",
        )
        model, tokenizer, metadata = engine.model, engine.tokenizer, engine.metadata
    tokens = [tokenizer.encode(c, add_special_tokens=False) for c in choices]
    token_ids = list(range(32, 32 + len(choices)))
    if tokens != [[t] for t in token_ids] or model.dtype != torch.bfloat16:
        raise ValueError(
            "collection requires the frozen choice tokens and BF16 weights"
        )
    formatted = sorted(
        [
            (
                m,
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
            )
            for m, text in rows
        ],
        key=lambda pair: len(pair[1]),
    )
    results = []
    total_tokens = 0
    elapsed = 0.0
    identity = dict(
        model=name,
        revision=revision,
        dataset_revision=dataset_revision,
        batch_size=batch_size,
        cohort=digest("|".join(m["id"] + digest(s) for m, s in formatted)),
    )
    if checkpoint:
        checkpoint.parent.mkdir(exist_ok=True, parents=True)
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_bytes())
            if saved["identity"] != identity:
                raise ValueError("checkpoint belongs to a different collection")
            results, total_tokens, elapsed = (
                saved["rows"],
                saved["input_tokens"],
                saved["wall_seconds"],
            )
    prior_seconds = elapsed
    start = perf_counter()
    with torch.inference_mode():
        for i in range(len(results), len(rows), batch_size):
            group = formatted[i : i + batch_size]
            inputs = tokenizer(
                [s for _, s in group],
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")
            logits = (
                model(**inputs, logits_to_keep=1, use_cache=False)
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
            elapsed = prior_seconds + perf_counter() - start
            if checkpoint:
                pending = Path(str(checkpoint) + ".tmp")
                pending.write_text(
                    json.dumps(
                        dict(
                            identity=identity,
                            rows=results,
                            input_tokens=total_tokens,
                            wall_seconds=elapsed,
                        )
                    ),
                    encoding="utf-8",
                )
                pending.replace(checkpoint)
    write_report(out / "predictions.csv", sorted(results, key=lambda r: r["id"]))
    summary = dict(
        **metadata,
        dtype="bfloat16",
        logical_requests=len(rows),
        batches=(len(rows) + batch_size - 1) // batch_size,
        input_tokens=total_tokens,
        wall_seconds=elapsed,
        dataset_revision=dataset_revision,
    )
    if memory_gib is not None:
        summary.update(
            batch_size=batch_size,
            allocator_limit_gib=memory_gib,
            peak_reserved_mib=torch.cuda.max_memory_reserved() // 2**20,
        )
    (out / "collection.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(summary)
    if checkpoint:
        checkpoint.unlink(missing_ok=True)
