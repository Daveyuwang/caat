"""Join saved HELM completions to WildBench and Omni-MATH judge outcomes."""

import argparse
import csv
import gzip
import json
import re
from math import isclose
from pathlib import Path
from shutil import copyfileobj
from statistics import mean
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from archived_scores import FULL_SCORE_FIELDS, full_scores, token_logprobs


BASE = "https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output"
SCENARIOS = {
    "wildbench": ("score", "wildbench_score_rescaled"),
    "omni_math": ("equivalence_judgement", "omni_math_accuracy"),
}
JUDGES = ("gpt", "llama", "claude")
INVENTORY = ("release", "scenario", "model", "suite", "run", "kept", "excluded", "status",
             "token_text_verified")
EXCLUSION = ("release", "scenario", "model", "id", "reason")
LITERAL_TERMINATORS = {"<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "</s>", "<|return|>"}
TERMINATORS = LITERAL_TERMINATORS | {"<|eot|>", "<|user|>"}
MATCHED_TOKEN_TEXT = {"literal", "terminal_marker", "byte_level", "final_channel", "byte_level_final_channel"}

# Reverse the standard GPT-2 byte-to-Unicode table used by ByteLevel tokenizers.
VISIBLE_BYTES = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
BYTE_LEVEL_DECODER = str.maketrans({
    chr(256 + index): chr(byte) for index, byte in enumerate(sorted(set(range(256)) - VISIBLE_BYTES))
})


def read_json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def fetch(url, path):
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"Accept-Encoding": "gzip"})
    with urlopen(request, timeout=90) as source:
        with NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            try:
                if source.headers.get("Content-Encoding") == "gzip":
                    with gzip.GzipFile(fileobj=source) as uncompressed:
                        copyfileobj(uncompressed, output, length=1 << 20)
                else:
                    copyfileobj(source, output, length=1 << 20)
            except BaseException:
                output.close()
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(path)
    return path


def indexed(rows, key):
    index = {}
    for row in rows:
        item = key(row)
        if item in index:
            raise ValueError(f"duplicate instance ID {item}")
        index[item] = row
    return index


def genuine_completions(states):
    for item, request in states.items():
        result = request.get("result") or {}
        completions = result.get("completions") or []
        if not result.get("success") or len(completions) != 1:
            yield item, None, None, "missing_or_multiple_completions"
            continue
        completion = completions[0]
        try:
            scores = token_logprobs(completion, item)
        except (KeyError, TypeError, ValueError):
            yield item, completion, None, "invalid_or_missing_token_logprobs"
            continue
        if not any(scores):
            yield item, completion, None, "all_zero_token_logprobs"
            continue
        yield item, completion, scores, None


def read_judges(display, scenario):
    suffix, metric = SCENARIOS[scenario]
    notes = (display.get("annotations") or {}).get(scenario) or {}
    values = tuple(notes.get(f"{judge}_{suffix}") for judge in JUDGES)
    if any(value is None for value in values):
        return None
    if scenario == "wildbench":
        if any(type(value) not in (int, float) or not 1 <= value <= 10 for value in values):
            return None
        recomputed = (mean(values) - 1) / 9
    else:
        if any(type(value) is not bool for value in values):
            return None
        recomputed = mean(values)
        values = tuple(int(value) for value in values)
    published = display["stats"].get(metric)
    if published is None or not isclose(published, recomputed, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{metric} differs for instance {display['instance_id']}")
    return (*values, published)


def same_response(rendered, text, terminators):
    return rendered == text or (rendered.startswith(text) and rendered[len(text):] in terminators)


def final_channel(rendered):
    if rendered.startswith("<|channel|>"):
        return rendered.partition("<|channel|>final<|message|>")[2]
    if rendered.lstrip().startswith("<think>"):
        return rendered.partition("</think>")[2]
    return ""


def token_alignment(completion):
    """Verify the visible response independently of the stored completion text."""
    text = completion["text"]
    rendered = "".join(token["text"] for token in completion["tokens"])
    if rendered == text:
        return 1, "literal"
    literal_match = same_response(rendered, text, LITERAL_TERMINATORS)
    if same_response(rendered, text, TERMINATORS):
        return int(literal_match), "terminal_marker"
    final = final_channel(rendered)
    if final and same_response(final.lstrip(), text, TERMINATORS):
        return 0, "final_channel"
    try:
        decoded = rendered.translate(BYTE_LEVEL_DECODER).encode("latin-1").decode("utf-8")
    except UnicodeError:
        decoded = rendered
    if same_response(decoded, text, TERMINATORS):
        return 0, "byte_level"
    final = final_channel(decoded)
    if final and same_response(final.lstrip(), text, TERMINATORS):
        return 0, "byte_level_final_channel"
    return 0, "replacement_character" if "\ufffd" in rendered else "different_text"


def prepare_run(cache, output, release, scenario, model, suite, run):
    stamp = {"release": release, "scenario": scenario, "model": model}
    name = quote(model, safe="-_.")
    path = output / scenario / f"{name}.csv"
    run_cache = cache / "runs" / suite / scenario / name
    run_url = f"{BASE}/runs/{quote(suite, safe='')}/{quote(run, safe='')}"
    display_url = f"{run_url}/display_predictions.json"
    display_cache = run_cache / "display_predictions.json"
    try:
        state = read_json(fetch(f"{run_url}/scenario_state.json", run_cache / "scenario_state.json"))
    except HTTPError as exc:
        if exc.code != 404:
            raise
        path.unlink(missing_ok=True)
        try:
            display = read_json(fetch(display_url, display_cache))
        except HTTPError as display_exc:
            if display_exc.code != 404:
                raise
            display = []
        shown = indexed(display, lambda row: row["instance_id"])
        exclusions = [{**stamp, "id": item, "reason": "unavailable_state"} for item in shown]
        return {**stamp, "suite": suite, "run": run, "kept": 0, "excluded": len(exclusions),
                "status": "unavailable_state", "token_text_verified": 0}, exclusions

    states = indexed(state["request_states"], lambda row: row["instance"]["id"])
    del state
    completions = genuine_completions(states)
    # Scan every response for non-placeholder log probabilities before requesting display data.
    if not any(scores is not None for _, _, scores, _ in completions):
        path.unlink(missing_ok=True)
        exclusions = [
            {**stamp, "id": item, "reason": reason}
            for item, _, _, reason in genuine_completions(states)
        ]
        return {**stamp, "suite": suite, "run": run, "kept": 0, "excluded": len(exclusions),
                "status": "no_real_logprobs", "token_text_verified": 0}, exclusions
    try:
        display = read_json(fetch(display_url, display_cache))
    except HTTPError as exc:
        if exc.code != 404:
            raise
        path.unlink(missing_ok=True)
        exclusions = [{**stamp, "id": item, "reason": "unavailable_display"} for item in states]
        return {**stamp, "suite": suite, "run": run, "kept": 0, "excluded": len(exclusions),
                "status": "unavailable_display", "token_text_verified": 0}, exclusions
    shown = indexed(display, lambda row: row["instance_id"])
    if shown.keys() != states.keys():
        raise ValueError(f"{run}: state and display instance IDs differ")

    suffix, metric = SCENARIOS[scenario]
    rows, exclusions = [], []
    verified = 0
    for item, completion, scores, reason in genuine_completions(states):
        judgment = read_judges(shown[item], scenario)
        if completion is not None and shown[item]["predicted_text"] != completion["text"]:
            raise ValueError(f"{run}: displayed response differs for {item}")
        if reason is None and judgment is None:
            reason = "incomplete_or_invalid_judge_scores"
        if reason is not None:
            exclusions.append({**stamp, "id": item, "reason": reason})
            continue
        literal_match, alignment = token_alignment(completion)
        verified += alignment in MATCHED_TOKEN_TEXT
        rows.append((item, mean(scores), *full_scores(completion, scores), literal_match, alignment, *judgment))
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow((
                "id", "mean_token_logprob", *FULL_SCORE_FIELDS, "token_text_matches", "token_alignment",
                *(f"{judge}_{suffix}" for judge in JUDGES), metric,
            ))
            writer.writerows(rows)
    else:
        path.unlink(missing_ok=True)
    return {**stamp, "suite": suite, "run": run, "kept": len(rows), "excluded": len(exclusions),
            "status": "complete" if rows else "no_complete_rows", "token_text_verified": verified}, exclusions


def update_table(path, fields, records, selection):
    previous = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as source:
            previous = [
                row for row in csv.DictReader(source)
                if (row["release"], row["scenario"], row["model"]) not in selection
            ]
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(
            previous + records, key=lambda r: (r["release"], r["scenario"], r["model"], r.get("id", ""))
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--release", default="v1.15.0")
    parser.add_argument("--output-dir", default=Path("replay/helm_judges"), type=Path)
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument("--model", action="append")
    models.add_argument("--all-with-logprobs", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"v\d+\.\d+\.\d+", args.release):
        parser.error("release must be a HELM version such as v1.15.0")
    inventory_path = args.output_dir / "inventory.csv"
    if inventory_path.is_file():
        with inventory_path.open(encoding="utf-8", newline="") as source:
            if any(row["release"] != args.release for row in csv.DictReader(source)):
                parser.error("use a separate --output-dir for each HELM release")
    cache = args.output_dir / "sources"
    url = f"{BASE}/releases/{args.release}"
    mapping = read_json(fetch(f"{url}/runs_to_run_suites.json", cache / f"runs_to_run_suites_{args.release}.json"))
    specs = read_json(fetch(f"{url}/run_specs.json", cache / f"run_specs_{args.release}.json"))
    runs = {}
    for spec in specs:
        run = spec["name"]
        if run.startswith(f"{args.scenario}:"):
            model = re.search(r"(?:^|[:,])model=([^,]+)", run).group(1)
            if model in runs:
                raise ValueError(f"multiple release runs for {args.scenario} and {model}")
            runs[model] = (mapping[run], run)
    chosen = sorted(set(args.model) if args.model else runs)
    if not chosen or set(chosen) - runs.keys():
        parser.error(f"model absent from {args.release} {args.scenario}: {sorted(set(chosen) - runs.keys())}")
    inventory, exclusions = [], []
    for model in chosen:
        row, excluded = prepare_run(cache, args.output_dir, args.release, args.scenario, model, *runs[model])
        inventory.append(row)
        exclusions.extend(excluded)
        print(f"{model}: {row['status']}, {row['kept']} kept, {row['excluded']} excluded", flush=True)
    selection = {(args.release, args.scenario, model) for model in chosen}
    update_table(inventory_path, INVENTORY, inventory, selection)
    update_table(args.output_dir / "exclusions.csv", EXCLUSION, exclusions, selection)
    return 0 if any(row["kept"] for row in inventory) else 1


if __name__ == "__main__":
    raise SystemExit(main())
