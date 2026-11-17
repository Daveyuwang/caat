"""Compare fitted CAAT and CalibRoute cutoffs on paired HELM archives."""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from common import ARCHIVES, ROOT, SEED, read_csv, write_report


CONFIDENCE = 0.95
FAMILY_WISE_FAILURE_RATE = 0.05
RULES = ("ltt-bonferroni", "ltt-holm", "calibroute-clopper-pearson")
PARTITIONS = ("design", "calibration", "holdout")
OMNI = "omni_math_qwen"


def call(*args, statuses=(0,)):
    result = subprocess.run(args, encoding="utf-8", capture_output=True)
    if result.returncode not in statuses:
        raise RuntimeError(f"{args}: status {result.returncode}\n{result.stderr}")
    return result


def load_inspect(directory):
    paths = {part: directory / f"{part}.csv" for part in PARTITIONS}
    splits = {}
    seen = set()
    for (part, path), count in zip(paths.items(), (200, 400, 400), strict=True):
        rows = read_csv(path)
        if len(rows) != count:
            raise ValueError(f"{path}: expected {count} rows")
        splits[part] = []
        for row in rows:
            item = row["id"]
            score = float(row["confidence"])
            digest = row["output_sha256"]
            primary, alternate = row["correct"], row["canonical_exact"]
            if (
                not item
                or item in seen
                or not 0 <= score <= 1
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or primary not in ("0", "1")
                or alternate not in ("0", "1")
            ):
                raise ValueError(f"{path}: invalid or reused sample {item!r}")
            seen.add(item)
            splits[part].append(
                {
                    "id": item,
                    "confidence": score,
                    "output_sha256": digest,
                    "correct": primary,
                    "alternate_correct": alternate,
                }
            )
    return splits, {
        "score": "external P(True)",
        "risk_target": 0.4,
        "counts": {part: len(rows) for part, rows in splits.items()},
        "file_sha256": {
            part: sha256(path.read_bytes()).hexdigest() for part, path in paths.items()
        },
    }


def load_case(name):
    if name == OMNI:
        path = ROOT / "replay/helm-import/omni.csv"
        total, target = 1000, 0.8
        graders = (
            "omni_math.gpt_equivalence_judgement",
            "omni_math.llama_equivalence_judgement",
        )
    else:
        directory, total, graders = ARCHIVES[name]
        path = ROOT / "replay" / directory / "predictions_full.csv"
        target = 0.4
    source = read_csv(path)
    ids = {row["id"] for row in source}
    if not all(ids) or len(ids) != len(source) or len(source) != total:
        raise ValueError(f"{path}: missing or duplicate sample IDs")
    records = {}
    raw_scores = {}
    for row in source:
        item = row["id"]
        score = float(row["mean_token_logprob"])
        digest = row["output_sha256"]
        outcomes = tuple(row[grader] for grader in graders)
        if (
            not math.isfinite(score)
            or score > 0
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or any(bit not in ("0", "1") for bit in outcomes)
        ):
            raise ValueError(f"{path}: invalid score, hash, or grade for {item}")
        raw_scores[item] = score
        records[item] = {
            "id": item,
            "confidence": math.exp(score),
            "output_sha256": digest,
            "correct": outcomes[0],
            "alternate_correct": outcomes[1],
        }
    if len(set(raw_scores.values())) != len(
        {row["confidence"] for row in records.values()}
    ):
        raise ValueError(f"{path}: exponentiation changed the score ties")
    if name == OMNI:
        splits = []
        previous_end = None
        for partition, stem in zip(
            PARTITIONS, ("design", "calibration", "heldout"), strict=True
        ):
            rows = read_csv(ROOT / "results/monitoring" / f"{stem}.csv")
            times = [int(row["request_timestamp"]) for row in rows]
            if (
                not times
                or times != sorted(times)
                or (previous_end is not None and times[0] <= previous_end)
            ):
                raise ValueError(f"invalid chronological {partition} boundary")
            previous_end = times[-1]
            for row in rows:
                item = row["id"]
                if row["correct"] != records[item]["correct"] or not math.isclose(
                    float(row["mean_token_logprob"]),
                    raw_scores[item],
                    abs_tol=1e-12,
                    rel_tol=0,
                ):
                    raise ValueError(
                        f"chronological replay disagrees with the archive for {item}"
                    )
            splits.append([row["id"] for row in rows])
    else:
        order = sorted(ids)
        Random(SEED).shuffle(order)
        first, second = total // 5, 3 * total // 5
        splits = (order[:first], order[first:second], order[second:])
    if (
        sum(map(len, splits)) != total
        or {item for part in splits for item in part} != ids
    ):
        raise ValueError(f"{name}: split coverage or overlap changed")
    partition_rows = {
        partition: [records[item] for item in items]
        for partition, items in zip(PARTITIONS, splits, strict=True)
    }
    metadata = {
        "archive": "HELM Capabilities v1.0.0",
        "prepared_source": path.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(path.read_bytes()).hexdigest(),
        "primary_column": graders[0],
        "alternate_column": graders[1],
        "risk_target": target,
        "split": "chronological" if name == OMNI else "seed-42",
        "counts": {name: len(rows) for name, rows in partition_rows.items()},
        "split_id_sha256": {
            name: sha256(json.dumps([row["id"] for row in rows]).encode()).hexdigest()
            for name, rows in partition_rows.items()
        },
    }
    return partition_rows, metadata


def exact_upper(errors, accepted):
    if not accepted:
        return None
    if errors == accepted:
        return 1.0
    coefficients = [math.comb(accepted, count) for count in range(errors + 1)]
    low, high = 0.0, 1.0
    for _ in range(60):
        point = (low + high) / 2
        cdf = math.fsum(
            coefficient * point**count * (1 - point) ** (accepted - count)
            for count, coefficient in enumerate(coefficients)
        )
        if cdf > 1 - CONFIDENCE:
            low = point
        else:
            high = point
    return high


def measure(rows, threshold, column, target):
    accepted = (
        []
        if threshold is None
        else [row for row in rows if row["confidence"] >= threshold]
    )
    count = len(accepted)
    errors = sum(row[column] == "0" for row in accepted)
    bound = exact_upper(errors, count)
    return {
        "total": len(rows),
        "accepted": count,
        "coverage": count / len(rows),
        "errors": errors,
        "error_rate": errors / count if count else None,
        "risk_upper_bound": bound,
        "meets_target": bound is not None and bound <= target,
    }


def verify_counts(label, reported, expected):
    for field, value in expected.items():
        actual = reported[field]
        if actual == value or (
            isinstance(actual, float)
            and isinstance(value, float)
            and math.isclose(actual, value, abs_tol=1e-12, rel_tol=0)
        ):
            continue
        raise ValueError(f"{label}: {field} is {actual!r}, expected {value!r}")


def compare_case(case, directory, caat, calibroute):
    partitions, meta = load_case(case)
    target = meta["risk_target"]
    paths = {name: directory / f"{name}.csv" for name in PARTITIONS}
    for name, rows in partitions.items():
        write_report(paths[name], rows)
    holdout = partitions["holdout"]
    alternate = directory / "alternate.csv"
    write_report(
        alternate, [{**row, "correct": row["alternate_correct"]} for row in holdout]
    )
    results = []
    raw_reports = {}
    for rule in RULES:
        fit_path = directory / f"{rule}.json"
        external = rule == "calibroute-clopper-pearson"
        if external:
            fit = call(
                calibroute,
                "fit",
                "--input",
                paths["calibration"],
                "--output",
                fit_path,
                "--max-risk",
                str(target),
                "--min-coverage",
                str(1 / len(partitions["calibration"])),
                "--risk-method",
                "clopper_pearson",
                "--confidence-level",
                str(CONFIDENCE),
                statuses=(0, 2),
            )
            if (
                fit.returncode == 2
                and "no validation operating point" not in fit.stderr
            ):
                raise RuntimeError(fit.stderr)
        else:
            fit = call(
                caat,
                "select",
                paths["calibration"],
                "--design",
                paths["design"],
                "--score-column",
                "confidence",
                "--correct-column",
                "correct",
                "--rule",
                rule,
                "--risk-target",
                str(target),
                "--failure-rate",
                str(FAMILY_WISE_FAILURE_RATE),
                "--track-fit-outputs",
                "--output-key-column",
                "output_sha256",
                "--output",
                fit_path,
                statuses=(0, 1),
            )
        fit_report = json.loads(fit_path.read_bytes()) if fit_path.exists() else None
        threshold = (
            fit_report["accept_threshold" if external else "selected_threshold"]
            if fit_report is not None
            else None
        )
        if (threshold is not None) != (fit.returncode == 0):
            raise ValueError(f"{case}/{rule}: inconsistent cutoff and fit status")
        primary = measure(holdout, threshold, "correct", target)
        alternate_counts = measure(holdout, threshold, "alternate_correct", target)
        raw_reports[rule] = {"fit": fit_report, "fit_status": fit.returncode}
        if threshold is not None and external:
            checks = {}
            for name, path, expected in (
                ("primary", paths["holdout"], primary),
                ("alternate", alternate, alternate_counts),
            ):
                report_path = directory / f"calibroute-{name}.json"
                validated = call(
                    calibroute,
                    "validate",
                    "--input",
                    path,
                    "--policy",
                    fit_path,
                    "--max-risk",
                    str(target),
                    "--confidence-level",
                    str(CONFIDENCE),
                    "--output",
                    report_path,
                    statuses=(0, 1),
                )
                found = json.loads(report_path.read_bytes())
                verify_counts(
                    f"{case}/{rule}/{name}",
                    {
                        **found,
                        "total": found["count"],
                        "error_rate": found["empirical_risk"],
                        "meets_target": found["passed"],
                    },
                    expected,
                )
                if found[
                    "accept_threshold"
                ] != threshold or validated.returncode != int(
                    not expected["meets_target"]
                ):
                    raise ValueError(
                        f"{case}/{rule}/{name}: unexpected frozen cutoff or exit status"
                    )
                checks[name] = found
            raw_reports[rule]["holdout"] = checks
        elif threshold is not None:
            report_path = directory / f"{rule}-audit.json"
            validated = call(
                caat,
                "audit",
                paths["holdout"],
                "--threshold-report",
                fit_path,
                "--correct-column",
                "correct",
                "--correct-column",
                "alternate_correct",
                "--holdout-confidence",
                str(CONFIDENCE),
                "--output",
                report_path,
                statuses=(0, 1),
            )
            found = json.loads(report_path.read_bytes())
            if (
                not found["input"]["fit_outputs_checked"]
                or found["threshold"] != threshold
            ):
                raise ValueError(
                    f"{case}/{rule}: frozen holdout audit failed its identity check"
                )
            verify_counts(
                f"{case}/{rule}/primary", found["graders"]["correct"], primary
            )
            verify_counts(
                f"{case}/{rule}/alternate",
                found["graders"]["alternate_correct"],
                alternate_counts,
            )
            if validated.returncode != int(
                not (primary["meets_target"] and alternate_counts["meets_target"])
            ):
                raise ValueError(f"{case}/{rule}: unexpected audit status")
            raw_reports[rule]["holdout"] = found
        results.append(
            {
                "case": case,
                "rule": rule,
                "target": target,
                "threshold": threshold,
                "holdout": primary["total"],
                "accepted": primary["accepted"],
                "coverage": primary["coverage"],
                "primary_errors": primary["errors"],
                "primary_risk": primary["error_rate"],
                "primary_upper_95": primary["risk_upper_bound"],
                "primary_met_target": primary["meets_target"],
                "alternate_errors": alternate_counts["errors"],
                "alternate_risk": alternate_counts["error_rate"],
                "alternate_upper_95": alternate_counts["risk_upper_bound"],
                "alternate_met_target": alternate_counts["meets_target"],
            }
        )
    return results, meta, raw_reports


def main():
    suffix = ".exe" if os.name == "nt" else ""
    directory = Path(sys.executable).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caat", default=str(directory / f"caat{suffix}"))
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "replay/decision-baselines"
    )
    args = parser.parse_args()
    calibroute_version = version("calibroute-ai")
    if calibroute_version != "0.4.0":
        parser.error("the replay requires calibroute-ai==0.4.0")
    calibroute = directory / f"calibroute{suffix}"
    metadata = {
        "caat": call(args.caat, "--version").stdout.strip(),
        "calibroute": calibroute_version,
        "score": "exp(mean_token_logprob)",
        "holdout_confidence": CONFIDENCE,
        "caat_fit": {
            "grid": "design-score deciles",
            "failure_rate": FAMILY_WISE_FAILURE_RATE,
        },
        "calibroute_fit": {
            "grid": "calibration scores",
            "risk_method": "clopper_pearson",
            "confidence_level": CONFIDENCE,
            "minimum_accepted": 1,
        },
        "cases": {},
    }
    rows, reports = [], {}
    with TemporaryDirectory(prefix="caat-decision-baselines-") as work:
        for case in (*ARCHIVES, OMNI):
            scratch = Path(work) / case
            scratch.mkdir()
            compared, metadata["cases"][case], reports[case] = compare_case(
                case, scratch, args.caat, calibroute
            )
            rows.extend(compared)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_report(args.output_dir / "comparison.csv", rows)
    for name, payload in (("protocol", metadata), ("replay", reports)):
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
