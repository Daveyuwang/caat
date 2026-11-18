"""Freeze WinoGrande input families and score the two original choices on GPU."""

import argparse
import json
from time import perf_counter

from collect_boolq import MODELS
from common import ROOT, read_csv, write_report
from prepare_boolq import digest, normalized

REVISION = "01e74176c63542e6b0bcb004dcdea22d94fb67b5"
OUT = ROOT / "results/winogrande"
FIELDS = ["sentence", "option1", "option2"]


def dataset_file():
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        "allenai/winogrande",
        "winogrande_xl/validation-00000-of-00001.parquet",
        repo_type="dataset",
        revision=REVISION,
    )


def prompts():
    import pyarrow.parquet as pq

    raw = pq.read_table(dataset_file(), columns=FIELDS).to_pylist()
    if len(raw) != 1267:
        raise ValueError("unexpected validation split size")
    families = {}
    for i, r in enumerate(raw):
        choices = [r["option1"], r["option2"]]
        norm = [normalized(s) for s in choices]
        family = digest(json.dumps(sorted(norm), ensure_ascii=False))
        uid = "wino-" + digest(
            json.dumps([normalized(r["sentence"]), *norm], ensure_ascii=False)
        )
        meta = dict(
            id=uid,
            source_row=i,
            family_sha256=family,
            A_sha256=digest(choices[0]),
            B_sha256=digest(choices[1]),
        )
        msg = f"Sentence: {r['sentence']}\nWhich choice fills the blank?\nA: {choices[0]}\nB: {choices[1]}\nReply with exactly A or B."
        if family not in families or uid < families[family][0]["id"]:
            families[family] = (meta, msg)
    return sorted(families.values(), key=lambda pair: pair[0]["id"])


def collect(rows):
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
    tokens = [tokenizer.encode(c, add_special_tokens=False) for c in "AB"]
    if tokens != [[32], [33]] or engine.model.dtype != torch.bfloat16:
        raise ValueError("collection requires the frozen A/B tokens and BF16 weights")
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
                .logits[:, -1, [32, 33]]
                .float()
                .tolist()
            )
            total_tokens += int(inputs.attention_mask.sum())
            for (m, text), (a, b) in zip(group, logits, strict=True):
                choice = "A" if a >= b else "B"
                results.append(
                    dict(
                        id=m["id"],
                        prompt_sha256=digest(text),
                        A=a,
                        B=b,
                        answer=choice,
                        output_sha256=m[choice + "_sha256"],
                        gap=abs(a - b),
                    )
                )
    elapsed = perf_counter() - start
    write_report(OUT / "predictions.csv", sorted(results, key=lambda r: r["id"]))
    summary = dict(
        **engine.metadata,
        dtype="bfloat16",
        logical_requests=len(rows),
        batches=(len(rows) + 15) // 16,
        input_tokens=total_tokens,
        wall_seconds=elapsed,
        dataset_revision=REVISION,
    )
    (OUT / "collection.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "collect"])
    args = parser.parse_args()
    rows = prompts()
    manifest = [m for m, _ in rows]
    if args.stage == "freeze":
        OUT.mkdir(parents=True, exist_ok=True)
        write_report(OUT / "manifest.csv", manifest)
        print("Frozen families:", len(manifest))
    else:
        if read_csv(OUT / "manifest.csv") != [
            {k: str(v) for k, v in m.items()} for m in manifest
        ]:
            raise ValueError("cohort changed after freeze")
        collect(rows)
