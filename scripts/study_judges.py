"""Run the validation gate on the saved HELM judge archives."""

import argparse
import json
from math import ceil
from pathlib import Path
from urllib.parse import quote

import numpy as np

from caat.metrics import ranking_metrics, threshold_points
from common import JUDGES, MATCHED_TOKEN_TEXT, ROOT, SCENARIOS, read_csv, write_report


PAIRS = (("llama", "gpt"), ("claude", "gpt"), ("llama", "claude"))
COVERAGES = (0.25, 0.50, 0.75)
SEED = 20261115
BOOTSTRAPS = 2000
PERMUTATIONS = 10_000
MEAN = "mean_token_logprob"
CONTROLS = {
    "ifeval_qwen": ("ifeval", ("strict_pass", "helm_strict_pass")),
    "ifeval_llama": ("ifeval_llama", ("strict_pass", "helm_strict_pass")),
    "mmlu_pro_qwen": ("mmlu_pro", ("correct", "tiger_correct")),
    "mmlu_pro_llama": ("mmlu_pro_llama", ("correct", "tiger_correct")),
}


def scores_from(rows):
    scores = {
        name: np.array([float(row[name]) for row in rows])
        for name in (MEAN, "sequence_logprob", "low_token_logprob")
    }
    scores["shorter_response"] = -np.array([int(row["token_count"]) for row in rows])
    scores["negative_mean_token_logprob"] = -scores[MEAN]
    return scores


def permutation_distributions(values, accepted_counts):
    """Shuffle scores, preserving the number actually accepted at every cutoff."""
    n = len(values)
    rng = np.random.default_rng(SEED)
    whole = values.mean(axis=0)
    distributions = {
        count: np.empty((PERMUTATIONS, values.shape[1])) for count in accepted_counts
    }
    for start in range(0, PERMUTATIONS, 128):
        draws = min(128, PERMUTATIONS - start)
        indices = np.stack([rng.permutation(n) for _ in range(draws)])
        totals = np.cumsum(values[indices], axis=1, dtype=np.int32)
        for count, distribution in distributions.items():
            distribution[start : start + draws] = np.abs(
                totals[:, count - 1] / count - whole
            )
    for distribution in distributions.values():
        distribution.sort(axis=0)
    return distributions


def analyze(archive, scenario, scores, outcomes, pairs):
    # One column per judge pair and outcome cutoff keeps every resample paired.
    signed = np.array(
        [
            outcomes[cutoff][first].astype(np.int8) - outcomes[cutoff][second]
            for cutoff in outcomes
            for first, second in pairs
        ]
    ).T
    n = len(signed)
    whole = signed.mean(axis=0)
    ranks = [n - ceil(n * coverage) for coverage in COVERAGES]
    indices = np.random.default_rng(SEED).integers(0, n, size=(BOOTSTRAPS, n))
    sampled_diff = signed[indices]
    sampled_whole = sampled_diff.mean(axis=1)
    cells, discrimination = [], []
    for method, score in scores.items():
        auc_by_cutoff = {}
        for cutoff, labels in outcomes.items():
            auc_by_cutoff[cutoff] = {}
            for grader, passed in labels.items():
                points = threshold_points(
                    zip(score.tolist(), (~passed).tolist(), strict=True)
                )
                metrics = ranking_metrics(points)
                auc_by_cutoff[cutoff][grader] = metrics["auroc"]
                discrimination.append(
                    {
                        "scenario": scenario,
                        "model": archive,
                        "outcome_cutoff": cutoff,
                        "score": method,
                        "judge": grader,
                        "n": n,
                        "pass_rate": passed.mean(),
                        **metrics,
                    }
                )
        partitions = np.partition(score, ranks)
        sampled_score = score[indices]
        sampled_partitions = np.partition(sampled_score, ranks, axis=1)
        for coverage, rank in zip(COVERAGES, ranks, strict=True):
            threshold = partitions[rank]
            accepted = score >= threshold
            count = int(accepted.sum())
            risk_shift = signed[accepted].mean(axis=0)
            bootstrap_accepted = sampled_score >= sampled_partitions[:, rank, None]
            bootstrap_shift = (
                np.einsum(
                    "bi,bij->bj", bootstrap_accepted, sampled_diff, dtype=np.int32
                )
                / bootstrap_accepted.sum(axis=1)[:, None]
            )
            lower, upper = np.percentile(
                bootstrap_shift - sampled_whole, [2.5, 97.5], axis=0
            )
            for cut_index, (cutoff, labels) in enumerate(outcomes.items()):
                for pair_index, (first, second) in enumerate(pairs):
                    j = cut_index * len(pairs) + pair_index
                    cells.append(
                        (
                            j,
                            {
                                "scenario": scenario,
                                "model": archive,
                                "outcome_cutoff": cutoff,
                                "score": method,
                                "first_judge": first,
                                "second_judge": second,
                                "requested_coverage": coverage,
                                "coverage": count / n,
                                "threshold": threshold,
                                "total": n,
                                "accepted": count,
                                "first_risk": (~labels[first][accepted]).mean(),
                                "second_risk": (~labels[second][accepted]).mean(),
                                "first_judge_auroc": auc_by_cutoff[cutoff][first],
                                "second_judge_auroc": auc_by_cutoff[cutoff][second],
                                "whole_shift": whole[j],
                                "risk_shift": risk_shift[j],
                                "excess_shift": risk_shift[j] - whole[j],
                                "lower_95": lower[j],
                                "upper_95": upper[j],
                                "accepted_disagreement": (
                                    signed[accepted, j] != 0
                                ).mean(),
                                "overall_disagreement": (signed[:, j] != 0).mean(),
                            },
                        )
                    )
    permuted = permutation_distributions(signed, {row["accepted"] for _, row in cells})
    results = []
    for j, row in cells:
        null = permuted[row["accepted"]][:, j]
        extreme = PERMUTATIONS - np.searchsorted(null, abs(row["excess_shift"]) - 1e-14)
        results.append(
            {**row, "permutation_pvalue": (extreme + 1) / (PERMUTATIONS + 1)}
        )
    return results, discrimination


def analyze_helm(model, scenario, rows):
    suffix, _ = SCENARIOS[scenario]
    cutoffs = (6, 7, 8) if scenario == "wildbench" else (1,)
    outcomes = {
        cutoff: {
            judge: np.array(
                [float(row[f"{judge}_{suffix}"]) >= cutoff for row in rows], dtype=bool
            )
            for judge in JUDGES
        }
        for cutoff in cutoffs
    }
    return analyze(model, scenario, scores_from(rows), outcomes, PAIRS)


def evaluate_gate(cells, auc_rows):
    primary = [
        c
        for c in cells
        if c["scenario"] == "wildbench"
        and c["outcome_cutoff"] == 7
        and c["score"] == MEAN
        and c["requested_coverage"] == 0.50
    ]
    model_count = len({c["model"] for c in primary})
    if not model_count:
        raise ValueError("the gate needs at least one WildBench archive")
    null_cells = sum(bool(c["lower_95"] <= 0 <= c["upper_95"]) for c in primary)
    primary_auc = [
        row
        for row in auc_rows
        if row["scenario"] == "wildbench" and row["outcome_cutoff"] == 7
    ]
    reach = {
        score: len(
            {
                row["model"]
                for row in primary_auc
                if row["score"] == score
                and row["auroc"] is not None
                and row["auroc"] >= 0.60
            }
        )
        for score in sorted({row["score"] for row in primary_auc})
    }
    stop_excess = 2 * null_cells > len(primary)
    stop_discrimination = all(2 * count <= model_count for count in reach.values())
    return primary, {
        "decision": "stop" if stop_excess or stop_discrimination else "pass",
        "models": model_count,
        "cells": len(primary),
        "intervals_containing_zero": null_cells,
        "per_pair": {
            f"{first}_to_{second}": {
                "cells": sum(
                    c["first_judge"] == first and c["second_judge"] == second
                    for c in primary
                ),
                "intervals_containing_zero": sum(
                    bool(
                        c["first_judge"] == first
                        and c["second_judge"] == second
                        and c["lower_95"] <= 0 <= c["upper_95"]
                    )
                    for c in primary
                ),
            }
            for first, second in PAIRS
        },
        "models_with_auroc_at_least_point60_by_score": reach,
        "stop_excess": stop_excess,
        "stop_discrimination": stop_discrimination,
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "permutation_draws": PERMUTATIONS,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "replay" / "helm_judges"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "replay" / "judge_study"
    )
    parser.add_argument("--release", default="v1.15.0")
    parser.add_argument("--deterministic-controls", type=Path, metavar="REPLAY_DIR")
    args = parser.parse_args()
    cells, auc_rows, alignment = [], [], {}
    verified_cells, verified_auc = [], []
    for run in read_csv(args.input_dir / "inventory.csv"):
        if run["release"] != args.release or run["status"] != "complete":
            continue
        scenario, model = run["scenario"], run["model"]
        rows = sorted(
            read_csv(args.input_dir / scenario / f"{quote(model, safe='-_.')}.csv"),
            key=lambda row: row["id"],
        )
        if len(rows) != int(run["kept"]) or len({row["id"] for row in rows}) != len(
            rows
        ):
            raise ValueError(f"{scenario}/{model} does not match the inventory")
        case_cells, case_auc = analyze_helm(model, scenario, rows)
        cells.extend(case_cells)
        auc_rows.extend(case_auc)
        if scenario == "wildbench":
            alignment[model] = sum(
                int(row["token_text_matches"]) for row in rows
            ) / len(rows)
            verified = [
                row for row in rows if row["token_alignment"] in MATCHED_TOKEN_TEXT
            ]
            if len(verified) != int(run["token_text_verified"]):
                raise ValueError(
                    f"{scenario}/{model} token alignment does not match the inventory"
                )
            if verified:
                if len(verified) != len(rows):
                    case_cells, case_auc = analyze_helm(model, scenario, verified)
                verified_cells.extend(case_cells)
                verified_auc.extend(case_auc)
        print(f"Analyzed {scenario}/{model}: {len(rows)} rows", flush=True)
    if args.deterministic_controls is not None:
        for model, (directory, pair) in CONTROLS.items():
            rows = sorted(
                read_csv(
                    args.deterministic_controls / directory / "predictions_full.csv"
                ),
                key=lambda row: row["id"],
            )
            outcomes = {
                1: {
                    judge: np.array([row[judge] == "1" for row in rows])
                    for judge in pair
                }
            }
            case_cells, case_auc = analyze(
                model,
                "deterministic",
                {MEAN: scores_from(rows)[MEAN]},
                outcomes,
                (pair,),
            )
            cells.extend(case_cells)
            auc_rows.extend(case_auc)
    primary, gate = evaluate_gate(cells, auc_rows)
    for name, models in (
        ("without_two_row_models", {c["model"] for c in primary if c["total"] > 2}),
        (
            "at_least_90_percent_token_text_alignment",
            {m for m, value in alignment.items() if value >= 0.90},
        ),
    ):
        if models:
            _, sensitivity = evaluate_gate(
                [c for c in cells if c["model"] in models],
                [r for r in auc_rows if r["model"] in models],
            )
            gate.setdefault("sensitivity", {})[name] = sensitivity
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_report(args.output_dir / "cells.csv", cells)
    write_report(args.output_dir / "auroc.csv", auc_rows)
    write_report(args.output_dir / "gate_cells.csv", primary)
    if verified_cells:
        verified_primary, verified_gate = evaluate_gate(verified_cells, verified_auc)
        gate.setdefault("sensitivity", {})["reconstructed_rows"] = verified_gate
        write_report(args.output_dir / "reconstructed_gate_cells.csv", verified_primary)
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validation gate: {gate['decision']}", flush=True)


if __name__ == "__main__":
    main()
