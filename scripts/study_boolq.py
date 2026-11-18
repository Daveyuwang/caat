"""Audit the frozen BoolQ policy and compare the value of its verifier."""

import argparse
import json
from math import ceil, exp, isclose
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from caat.workflow import run_workflow

import numpy as np
from scipy.stats import beta, binom

from caat.metrics import ranking_metrics, threshold_points
from caat.scorer import fit_scorer
from caat.scores import ScoredRow, _ScoreRanking
from caat.select import design_thresholds
from common import ROOT, read_csv, write_report
from prepare_boolq import dataset_file, digest, normalized

FEATURES = ("winner_logprob", "gap", "p_true")
SEED = 20261117


def indexed(path):
    rows = read_csv(path)
    result = {r["id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate ID in {path}")
    return result


def assemble(logs, manifest_path, output):
    import pyarrow.parquet as pq

    manifest = indexed(manifest_path)
    generated = indexed(logs / "generate.csv")
    verified = indexed(logs / "verify.csv")
    if generated.keys() != manifest.keys() or verified.keys() != manifest.keys():
        raise ValueError("score IDs differ from the frozen manifest")
    gold = {
        s: pq.read_table(
            dataset_file(s), columns=["passage", "question", "answer"]
        ).to_pylist()
        for s in ("train", "validation")
    }
    rows = []
    for i, m in manifest.items():
        g, v = generated[i], verified[i]
        source = gold[m["split"]][int(m["row"])]
        first, second = float(g["first_logit"]), float(g["second_logit"])
        predicted = first >= second
        answer = "Yes" if predicted else "No"
        native = max(first, second) - float(g["log_normalizer"])
        gap = abs(first - second)
        p_true = 1 / (1 + exp(float(v["second_logit"]) - float(v["first_logit"])))
        if (
            g["answer"] != answer
            or g["output_sha256"] != digest(answer)
            or v["output_sha256"] != digest(answer)
            or digest(
                normalized(source["passage"]) + "\n" + normalized(source["question"])
            )
            != i
            or not isclose(native, float(g["winner_logprob"]), abs_tol=1e-5)
            or not isclose(gap, float(g["gap"]), abs_tol=1e-6)
            or not isclose(1 / (1 + exp(-gap)), float(g["confidence"]), abs_tol=1e-6)
            or not isclose(p_true, float(v["p_true"]), abs_tol=1e-6)
        ):
            raise ValueError(f"log or grade identity mismatch for {i}")
        rows.append(
            {
                "id": i,
                "role": m["role"],
                "output_sha256": g["output_sha256"],
                "answer": answer,
                "correct": ""
                if m["role"] == "reference"
                else int(predicted == source["answer"]),
                "generator_prompt_sha256": g["prompt_sha256"],
                "verifier_prompt_sha256": v["prompt_sha256"],
                "yes_logit": first,
                "no_logit": second,
                "generator_log_z": g["log_normalizer"],
                "true_logit": v["first_logit"],
                "false_logit": v["second_logit"],
                "winner_logprob": g["winner_logprob"],
                "gap": g["gap"],
                "confidence": g["confidence"],
                "p_true": v["p_true"],
            }
        )
    write_report(output, rows)


def score_quality(scores, holdout, wrong, best, native):
    ranks = {
        k: _ScoreRanking.build(s[holdout], wrong[holdout]) for k, s in scores.items()
    }
    n = len(holdout)
    count = ceil(n * 0.75)
    estimates = {k: r.risks([1] * n, (count,))[0] for k, r in ranks.items()}
    rng = np.random.default_rng(SEED)
    competitors = list(dict.fromkeys([best, native]))
    deltas = {k: [] for k in competitors}
    for _ in range(5000):
        weights = np.bincount(rng.integers(n, size=n), minlength=n).tolist()
        fitted = ranks["fitted"].risks(weights, (count,))[0]
        for k in competitors:
            deltas[k].append(fitted - ranks[k].risks(weights, (count,))[0])
    intervals = {}
    for k, draws in deltas.items():
        low, high = np.quantile(draws, [0.025, 0.975])
        intervals[k] = {
            "delta": estimates["fitted"] - estimates[k],
            "lower95": float(low),
            "upper95": float(high),
        }
    return {
        "count": count,
        "accepted_error": estimates,
        "comparisons": intervals,
        "design_selected_single": best,
        "design_selected_native": native,
    }


def evaluate_grid(values, grid, cal, holdout, wrong):
    chosen = values[cal, None] >= grid
    n = chosen.sum(axis=0)
    k = (chosen * wrong[cal, None]).sum(axis=0)
    passed = (n > 0) & (binom.cdf(k, n, 0.2) <= 0.05 / len(grid))
    cutoff = float(np.min(grid[passed])) if np.any(passed) else None
    mask = (
        values[holdout] >= cutoff
        if cutoff is not None
        else np.zeros(len(holdout), dtype=bool)
    )
    return cutoff, mask


def audit(mask, wrong, cutoff):
    n = int(mask.sum())
    k = int((mask & wrong).sum())
    upper = None if n == 0 else 1.0 if k == n else float(beta.ppf(0.95, k + 1, n - k))
    return {
        "cutoff": cutoff,
        "accepted": n,
        "errors": k,
        "upper95": upper,
        "confirmed": upper is not None and upper <= 0.2,
    }


def allocation(scores, reference, pool, holdout, wrong, ids, output):
    grids = {"numeric": np.array([i / 100 for i in range(50, 100, 5)])}
    arms = {}
    seeds = []
    for kind in ("confidence", "fitted", "p_true", "winner_logprob", "gap"):
        values = scores[kind]
        grids["reference"] = np.array(design_thresholds(values[reference]))
        for grid_name in (
            ("numeric", "reference") if kind == "confidence" else ("reference",)
        ):
            for budget in (320, 480):
                key = f"{kind}_{grid_name}_{budget}"
                flips = []
                coverage = []
                empty = []
                previous = None
                for seed in range(200):
                    draw = list(pool)
                    Random(seed).shuffle(draw)
                    cutoff, mask = evaluate_grid(
                        values, grids[grid_name], draw[:budget], holdout, wrong
                    )
                    coverage.append(float(mask.mean()))
                    empty.append(cutoff is None)
                    if seed % 2:
                        flips.append(float((previous != mask).mean()))
                    else:
                        previous = mask
                    if seed == 42:
                        primary = audit(mask, wrong[holdout], cutoff)
                        accepted_ids = [ids[j] for j in holdout[mask]]
                    seeds.append(
                        {
                            "arm": key,
                            "seed": seed,
                            "cutoff": cutoff,
                            "accepted": int(mask.sum()),
                        }
                    )
                arms[key] = {
                    "flips": flips,
                    "coverage": coverage,
                    "empty": empty,
                    "primary": primary,
                    "accepted_ids": accepted_ids,
                }
    write_report(output / "refits.csv", seeds)
    primary_rows = [
        {"arm": k, "id": i} for k, v in arms.items() for i in v.pop("accepted_ids")
    ]
    write_report(output / "accepted.csv", primary_rows, ["arm", "id"])
    small = arms["confidence_reference_320"]
    full = arms["confidence_numeric_480"]
    rng = np.random.default_rng(SEED)
    pairs = rng.integers(100, size=(5000, 100))
    noninferiority = {}
    for metric in ("flips", "empty", "coverage"):
        a = np.asarray(small[metric], dtype=float)
        b = np.asarray(full[metric], dtype=float)
        diffs = a - b if metric != "coverage" else b - a
        if len(diffs) == 200:
            diffs = diffs.reshape(-1, 2).mean(axis=1)
        noninferiority[metric] = {
            "difference": float(diffs.mean()),
            "upper95": float(np.quantile(diffs[pairs].mean(axis=1), 0.95)),
        }
    summaries = {
        k: {
            "mean_flip": float(np.mean(v["flips"])),
            "mean_coverage": float(np.mean(v["coverage"])),
            "no_cutoff": sum(v["empty"]),
            "primary": v["primary"],
        }
        for k, v in arms.items()
    }
    return summaries, noninferiority


def check_workflow(data, ref, pool, held, output, expected):
    draw = list(pool)
    Random(42).shuffle(draw)
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "model.json").write_bytes((output / "scorer.json").read_bytes())
        for name, indices in (
            ("reference", ref),
            ("calibration", draw[:480]),
            ("holdout", held),
        ):
            fields = ["id", "output_sha256", *FEATURES]
            if name != "reference":
                fields.append("correct")
            write_report(
                root / f"{name}.csv", [{k: data[i][k] for k in fields} for i in indices]
            )
        config = root / "run.toml"
        config.write_text(
            """correct_columns=["correct"]
output_key_column="output_sha256"
risk_target=0.2
json_output="policy.json"
[predictions]
calibration="calibration.csv"
holdout="holdout.csv"
[frozen]
reference="reference.csv"
scorer="model.json"
""",
            encoding="utf8",
        )
        policy = run_workflow(config, save_outputs=True)["policy"]
        actual = policy["heldout"]["graders"]["correct"]
        for name in ("accepted", "errors"):
            if actual[name] != expected[name]:
                raise ValueError(f"workflow and independent audit differ on {name}")
        if (
            policy["selection"]["selected_threshold"] != expected["cutoff"]
            or abs(actual["risk_upper_bound"] - expected["upper95"]) > 1e-10
        ):
            raise ValueError("workflow and independent frozen cutoff audit differ")


def study(table, output):
    data = sorted(read_csv(table), key=lambda r: r["id"])
    ids = [r["id"] for r in data]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate score IDs")
    roles = np.array([r["role"] for r in data])
    design, ref, pool, held = (
        np.flatnonzero(roles == r)
        for r in ("design", "reference", "calibration_pool", "holdout")
    )
    if [len(design), len(ref), len(pool), len(held)] != [400, 160, 960, 2938]:
        raise ValueError("unexpected frozen partition sizes")
    if any(
        r["correct"] not in (("",) if r["role"] == "reference" else ("0", "1"))
        for r in data
    ):
        raise ValueError("unexpected grades")
    wrong = np.array([r["correct"] == "0" for r in data])
    scores = {
        c: np.array([float(r[c]) for r in data]) for c in (*FEATURES, "confidence")
    }
    values = [tuple(scores[c][j] for c in FEATURES) for j in range(len(ids))]
    model = fit_scorer(
        FEATURES, [ScoredRow(ids[j], values[j], (bool(not wrong[j]),)) for j in design]
    )
    scores["fitted"] = np.array([model.score(v) for v in values])
    gains = {
        k: ranking_metrics(
            threshold_points(zip(scores[k][design], wrong[design], strict=True))
        )["normalized_aurc_gain"]
        for k in FEATURES
    }
    best = max(FEATURES, key=gains.get)
    native = max(FEATURES[:2], key=gains.get)
    output.mkdir(exist_ok=True, parents=True)
    arms, ni = allocation(scores, ref, pool, held, wrong, ids, output)
    quality = score_quality(scores, held, wrong, best, native)
    report = {
        "design_gain": gains,
        "quality": quality,
        "arms": arms,
        "label_savings": ni,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    (output / "scorer.json").write_text(
        json.dumps(model.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    check_workflow(
        data, ref, pool, held, output, arms["fitted_reference_480"]["primary"]
    )
    # Exploratory ablation motivated by the design-set imbalance between Yes and No answers.
    inputs = [
        (*v[:2], float(r["answer"] == "Yes")) for v, r in zip(values, data, strict=True)
    ]
    ablation = fit_scorer(
        (*FEATURES[:2], "answer_yes"),
        [ScoredRow(ids[j], inputs[j], (bool(not wrong[j]),)) for j in design],
    )
    compared = {
        "fitted": scores["fitted"],
        "native_and_answer": np.array([ablation.score(v) for v in inputs]),
    }
    exploratory = score_quality(
        compared, held, wrong, "native_and_answer", "native_and_answer"
    )
    (output / "ablation.json").write_text(
        json.dumps(exploratory, indent=2) + "\n", encoding="utf8"
    )
    print(
        json.dumps(
            {"quality": quality, "label_savings": ni, "exploratory": exploratory},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect-dir",
        type=Path,
        help="Join the completed model logs to the pinned grader",
    )
    parser.add_argument("--table", type=Path, default=ROOT / "results/boolq/scores.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "replay/boolq/replay")
    args = parser.parse_args()
    if args.collect_dir:
        assemble(args.collect_dir, ROOT / "results/boolq/manifest.csv", args.table)
    study(args.table, args.output)
