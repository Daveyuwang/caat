"""Select one prompt per HellaSwag source and score the four original endings."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.artifacts import staged_outputs
from collect_choice_logits import collect_choices, formatted_prompts
from common import ROOT, read_csv, read_json, write_report
from prepare_boolq import digest

OUT = ROOT / "results/hellaswag"
REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
CHOICES = "ABCD"


def dataset_file():
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        "Rowan/hellaswag",
        "data/validation-00000-of-00001.parquet",
        repo_type="dataset",
        revision=REVISION,
    )


def prompts():
    import pyarrow.parquet as pq

    raw = pq.read_table(
        dataset_file(), columns=["ind", "source_id", "ctx", "endings"]
    ).to_pylist()
    families = {}
    for i, r in enumerate(raw):
        uid = digest(f"{r['source_id']}|{r['ind']}|{i}")
        family = digest(r["source_id"])
        endings = dict(zip(CHOICES, r["endings"], strict=True))
        meta = dict(
            id=uid,
            source_row=i,
            source_sha256=family,
            **{c + "_sha256": digest(v) for c, v in endings.items()},
        )
        prompt = (
            "Context: "
            + r["ctx"]
            + "\nWhich ending best continues the context?\n"
            + "\n".join(f"{c}: {v}" for c, v in endings.items())
            + "\nReply with exactly A, B, C or D."
        )
        if family not in families or uid < families[family][0]["id"]:
            families[family] = meta, prompt
    return [families[k] for k in sorted(families)[:1500]]


def confirm_gpu(rows):
    from transformers import AutoTokenizer

    path = OUT / "collection.json"
    meta = read_json(path)
    if "gpu_confirmation" in meta or not any(
        r["device"] == "cpu" for r in meta["scoring_runs"]
    ):
        return
    tokenizer = AutoTokenizer.from_pretrained(meta["model"], revision=meta["revision"])
    ordered = [m["id"] for m, _ in formatted_prompts(tokenizer, rows)]
    by_id = {m["id"]: (m, text) for m, text in rows}
    first, selected = 0, []
    for run in meta["scoring_runs"]:
        if run["device"] == "cpu":
            selected.extend(by_id[k] for k in ordered[first : run["end"]])
        first = run["end"]
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        collect_choices(
            selected,
            CHOICES,
            folder,
            REVISION,
            batch_size=8,
            memory_gib=2.25,
            checkpoint=ROOT / "replay" / "hellaswag-gpu-confirmation.json",
        )
        verified = read_csv(folder / "predictions.csv")
        timing = read_json(folder / "collection.json")
    predictions = OUT / "predictions.csv"
    scored = {r["id"]: r for r in read_csv(predictions)}
    changed_choices = changed_eligibility = 0
    max_delta = 0.0
    for new in verified:
        old = scored[new["id"]]
        if old["prompt_sha256"] != new["prompt_sha256"]:
            raise ValueError("GPU confirmation used a different prompt")
        changed_choices += old["answer"] != new["answer"]
        changed_eligibility += (float(old["gap"]) >= 4.75) != (
            float(new["gap"]) >= 4.75
        )
        max_delta = max(
            max_delta, *(abs(float(new[c]) - float(old[c])) for c in CHOICES)
        )
        scored[new["id"]] = new
    meta["gpu_confirmation"] = dict(
        rows=len(verified),
        ordered_ids_sha256=digest("\n".join(m["id"] for m, _ in selected)),
        changed_choices=changed_choices,
        changed_eligibility=changed_eligibility,
        max_logit_difference=max_delta,
        wall_seconds=timing["wall_seconds"],
        input_tokens=timing["input_tokens"],
        device=timing["device"],
    )
    with staged_outputs([predictions, path]) as draft:
        write_report(draft[predictions], list(scored.values()))
        draft[path].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(meta["gpu_confirmation"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "collect", "confirm-gpu"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--stop-after", type=int, help="pause after the batch containing this many rows"
    )
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    rows = prompts()
    manifest = [m for m, _ in rows]
    path = OUT / "manifest.csv"
    if args.stage == "freeze":
        OUT.mkdir(exist_ok=True, parents=True)
        write_report(path, manifest)
    else:
        if read_csv(path) != [{k: str(v) for k, v in m.items()} for m in manifest]:
            raise ValueError("HellaSwag cohort differs from the frozen manifest")
        if args.stage == "confirm-gpu":
            confirm_gpu(rows)
            raise SystemExit(0)
        collect_choices(
            rows,
            CHOICES,
            OUT,
            REVISION,
            batch_size=8,
            memory_gib=2.25,
            checkpoint=ROOT / "replay" / "hellaswag-collection.json",
            device=args.device,
            stop_after=args.stop_after,
        )
