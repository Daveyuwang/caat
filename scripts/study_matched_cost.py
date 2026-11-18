"""Replay fixed score budgets on paired WildBench model outputs."""

import argparse
import json
from hashlib import sha256
from math import ceil, isfinite
from operator import itemgetter
from pathlib import Path
from random import Random

from caat.audit import percentile_interval
from caat.policy import split_rows
from caat.scores import _ScoreRanking

from common import JUDGES, ROOT, read_csv, read_json, write_report
from prepare_helm_judges import indexed, read_judges
from study_black_box import BOOTSTRAP_SEED, RESAMPLES


SMALL = "qwen_qwen2.5-7b-instruct-turbo"
LARGE = "qwen_qwen2.5-72b-instruct-turbo"
COVERAGES = (0.1, 0.25, 0.5)
SCORE = "mean_token_logprob"


def attach_inputs(cache, model):
    run = cache / "sources/runs/v1.0.0/wildbench" / model
    states = indexed(
        read_json(run / "scenario_state.json")["request_states"],
        lambda row: row["instance"]["id"],
    )
    displays = indexed(
        read_json(run / "display_predictions.json"), itemgetter("instance_id")
    )
    rows = indexed(read_csv(cache / "wildbench" / f"{model}.csv"), itemgetter("id"))
    for key, row in rows.items():
        state = states[key]
        completions = state["result"]["completions"]
        if len(completions) != 1 or not state["result"]["success"]:
            raise ValueError(f"{model} {key} needs one saved completion")
        text = completions[0]["text"]
        if (
            text != displays[key]["predicted_text"]
            or sha256(text.encode("utf-8")).hexdigest() != row["output_sha256"]
        ):
            raise ValueError(f"{model} completion differs for {key}")
        grades = tuple(float(row[f"{judge}_score"]) for judge in JUDGES)
        verdicts = read_judges(displays[key], "wildbench")
        if verdicts is None or verdicts[:-1] != grades:
            raise ValueError(f"{model} archived grades differ for {key}")
        prompt = json.dumps(
            state["instance"]["input"], ensure_ascii=False, sort_keys=True
        )
        row["input_sha256"] = sha256(prompt.encode("utf-8")).hexdigest()
    return rows


def matched_rows(cache):
    small, large = attach_inputs(cache, SMALL), attach_inputs(cache, LARGE)
    common = sorted(small.keys() & large.keys())
    rows = []
    for key in common:
        mini, full = small[key], large[key]
        if mini["input_sha256"] != full["input_sha256"]:
            raise ValueError(f"models saw different saved inputs for {key}")
        row = {
            "id": key,
            "input_sha256": mini["input_sha256"],
            "small_output_sha256": mini["output_sha256"],
            "large_output_sha256": full["output_sha256"],
            SCORE: float(mini[SCORE]),
        }
        for judge in JUDGES:
            row[f"{judge}_small_correct"] = int(float(mini[f"{judge}_score"]) >= 7)
            row[f"{judge}_large_correct"] = int(float(full[f"{judge}_score"]) >= 7)
        rows.append(row)
    for split, part in zip(
        ("design", "calibration", "heldout"),
        split_rows(rows, 42, id_key=itemgetter("id")),
        strict=True,
    ):
        for row in part:
            row["split"] = split
    return rows


def compare_judge(rows, judge, budgets):
    size = len(rows)
    scores = [float(row[SCORE]) for row in rows]
    if not all(isfinite(score) for score in scores):
        raise ValueError("the 7B scores must be finite")
    small_errors = [int(row[f"{judge}_small_correct"]) == 0 for row in rows]
    large_errors = [int(row[f"{judge}_large_correct"]) == 0 for row in rows]
    ranks = [
        _ScoreRanking.build(scores, flags) for flags in (small_errors, large_errors)
    ]

    def measure(weights):
        small_rate = (
            sum(w for w, error in zip(weights, small_errors, strict=True) if error)
            / size
        )
        large_rate = (
            sum(w for w, error in zip(weights, large_errors, strict=True) if error)
            / size
        )
        small_risks, large_risks = (
            ranking.risks(weights, budgets) for ranking in ranks
        )
        values = []
        for accepted, small_top, large_top in zip(
            budgets, small_risks, large_risks, strict=True
        ):
            share = accepted / size
            delta = share * (small_top - large_top)
            random_delta = share * (small_rate - large_rate)
            values.append(
                (
                    small_rate,
                    large_rate,
                    large_rate + delta,
                    large_rate + random_delta,
                    delta,
                    delta - random_delta,
                )
            )
        return values

    samples = [[[], []] for _ in budgets]
    rng = Random(BOOTSTRAP_SEED)
    for _ in range(RESAMPLES):
        weights = [0] * size
        for item in rng.choices(range(size), k=size):
            weights[item] += 1
        for (_, _, _, _, delta, random_delta), (against_large, against_random) in zip(
            measure(weights), samples, strict=True
        ):
            against_large.append(delta)
            against_random.append(random_delta)

    result = []
    for accepted, target, point, (large, random) in zip(
        budgets, COVERAGES, measure([1] * size), samples, strict=True
    ):
        small_rate, large_rate, routed, random_rate, delta, random_delta = point
        lower, upper = percentile_interval(large)
        random_lower, random_upper = percentile_interval(random)
        result.append(
            {
                "grader": judge,
                "heldout_rows": size,
                "coverage_target": target,
                "small_cost_per_large_cost": accepted / size,
                "accepted_small": accepted,
                "n_small_calls": size,
                "n_large_calls": size - accepted,
                "small_only_error": small_rate,
                "large_only_error": large_rate,
                "routed_error": routed,
                "random_error": random_rate,
                "delta_from_large": delta,
                "delta_large_ci_low": lower,
                "delta_large_ci_high": upper,
                "delta_from_random": random_delta,
                "delta_random_ci_low": random_lower,
                "delta_random_ci_high": random_upper,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "results/matched_cost/paired.csv"
    )
    parser.add_argument(
        "--prepare-from", type=Path, help="Rebuild pairs from a prepared HELM archive."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replay/matched_cost")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare_from:
        archive = matched_rows(args.prepare_from)
        write_report(args.output_dir / "paired.csv", archive)
    else:
        archive = read_csv(args.input)
    heldout = [row for row in archive if row["split"] == "heldout"]
    if not heldout:
        parser.error("the archive needs held-out rows")
    budgets = tuple(ceil(len(heldout) * fraction) for fraction in COVERAGES)
    results = [
        row for judge in JUDGES for row in compare_judge(heldout, judge, budgets)
    ]
    write_report(args.output_dir / "comparison.csv", results)


if __name__ == "__main__":
    main()
