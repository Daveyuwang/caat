"""Replay the locked ANLI score comparison on CPU."""

import argparse
import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from math import ceil, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import beta, binom
from sklearn.model_selection import StratifiedKFold

from caat.cli import main
from caat.metrics import ranking_metrics, threshold_points
from caat.scorer import apply_scorer, fit_scorer
from caat.scores import ScoredRow, _ScoreRanking, compare_scores
from caat.workflow import run_workflow
from common import ROOT, read_csv, write_report

SEED = 20261118
NATIVE = ("winner_logprob", "gap")
ARMS = {
    "class_only": (("unit",), True),
    "gap": (("gap",), False),
    "native_pair": (NATIVE, False),
    "native_class": (NATIVE, True),
    "all_logits": (("A", "B", "C"), False),
}


def raw(row, columns):
    return tuple(1.0 if k == "unit" else float(row[k]) for k in columns)


def fit(arm, design, definitions):
    if arm == "gap":
        return None
    cols, cat = definitions[arm]
    return fit_scorer(
        cols,
        [
            ScoredRow(
                r["id"],
                raw(r, cols),
                (r["correct"] == "1",),
                predicted_class=r["answer"],
            )
            for r in design
        ],
        class_column="answer" if cat else None,
    )


def predict(model, row):
    if model is None:
        return float(row["gap"])
    return model.score(raw(row, model.columns), predicted_class=row["answer"])


def crossfit(arms, design, definitions):
    labels = [r["correct"] == "1" for r in design]
    scores = {k: np.zeros(len(design)) for k in arms}
    for train, test in StratifiedKFold(5, shuffle=True, random_state=42).split(
        design, labels
    ):
        for k in arms:
            model = fit(k, [design[i] for i in train], definitions)
            scores[k][test] = [predict(model, design[i]) for i in test]
    return scores


def contrasts(scores, wrong, comparisons):
    n = len(wrong)
    count = ceil(n * 0.75)
    rankings = {
        k: _ScoreRanking.build(scores[k], wrong)
        for k in dict.fromkeys(k for pair in comparisons for k in pair)
    }
    point = {k: r.risks([1] * n, [count])[0] for k, r in rankings.items()}
    rng = np.random.default_rng(SEED)
    draws = [[] for pair in comparisons]
    for _ in range(5000):
        weights = np.bincount(rng.integers(n, size=n), minlength=n).tolist()
        risks = {k: r.risks(weights, (count,))[0] for k, r in rankings.items()}
        for values, (a, b) in zip(draws, comparisons, strict=True):
            values.append(risks[a] - risks[b])
    result = {}
    for values, (a, b) in zip(draws, comparisons, strict=True):
        lo, hi = np.quantile(values, [0.025, 0.975])
        result[a + "_minus_" + b] = {
            "delta": point[a] - point[b],
            "lower95": float(lo),
            "upper95": float(hi),
        }
    return result


def top_weights(scores):
    count = ceil(len(scores) * 0.75)
    cutoff = sorted(scores, reverse=True)[count - 1]
    scores = np.asarray(scores)
    return (scores > cutoff).astype(float) + (scores == cutoff) * (
        (count - int(sum(scores > cutoff))) / int(sum(scores == cutoff))
    )


def lock_design(design):
    wrong = [r["correct"] == "0" for r in design]
    vals = crossfit(ARMS, design, ARMS)
    gain = {
        k: ranking_metrics(threshold_points(zip(s, wrong, strict=True)))[
            "normalized_aurc_gain"
        ]
        for k, s in vals.items()
    }
    best = max(gain, key=gain.get)
    definitions = {**ARMS}
    for name, base in (("paid_best", best), ("paid_class", "native_class")):
        cols, cat = ARMS[base]
        definitions[name] = ((*cols, "p_true"), cat)
    vals.update(crossfit(["paid_best"], design, definitions))
    primary = [("native_class", "native_pair"), ("paid_best", best)]
    precision = contrasts(vals, wrong, primary)
    for a, b in primary:
        key = a + "_minus_" + b
        mass_a, mass_b = top_weights(vals[a]), top_weights(vals[b])
        precision[key]["discordant_accept_mass"] = float(np.abs(mass_a - mass_b).sum())
        # Project the design bootstrap half-width to the primary cohort size.
        precision[key]["projected_half_width"] = (
            (precision[key]["upper95"] - precision[key]["lower95"])
            / 2
            * sqrt(len(design) / 846)
        )
    return {
        "cheap_arm": best,
        "oof_gain": gain,
        "design_contrasts": precision,
        "rotated_answer_changes": sum(
            r["rotated_answer"] != r["answer"] for r in design
        ),
        "design_rows": len(design),
    }, definitions


def distributions(rows, vals):
    names = tuple(vals)
    scored = [
        ScoredRow(
            r["id"],
            tuple(vals[k][i] for k in names),
            (r["correct"] == "1",),
            predicted_class=r["answer"],
        )
        for i, r in enumerate(rows)
    ]
    report = compare_scores(
        scored,
        names=names,
        graders=["gold"],
        split="all",
        coverages=[0.75],
        bootstrap=0,
    )["graders"]["gold"]
    entries = {**report["controls"], **report["scores"]}
    return {k: v["budgets"][0] for k, v in entries.items()}


def cli(cmd, path, *args, expected_failure=None):
    with redirect_stdout(StringIO()) as stream, redirect_stderr(StringIO()) as err:
        try:
            main([cmd, str(path), *map(str, args), "--json"])
        except SystemExit as exc:
            if (
                exc.code == 2
                and expected_failure
                and expected_failure in err.getvalue()
            ):
                return None
            raise ValueError(err.getvalue()) from exc
        if expected_failure:
            raise ValueError(f"{cmd} unexpectedly succeeded")
        return json.loads(stream.getvalue())


def deploy(parts, models, best, output):
    model = models[best]
    with TemporaryDirectory() as temp:
        root = Path(temp)
        cols = list(model.input_columns) if model else ["gap"]
        for role in ("reference", "calibration", "primary", "stress"):
            rows = [
                {
                    "id": r["id"],
                    "output_sha256": r["output_sha256"],
                    **{c: (1 if c == "unit" else r[c]) for c in cols},
                    **(
                        {"gold": r["correct"], "check": r["correct"]}
                        if role != "reference"
                        else {}
                    ),
                }
                for r in parts[role]
            ]
            write_report(root / (role + ".csv"), rows)
        config = 'correct_columns=["gold"]\noutput_key_column="output_sha256"\nrisk_target=0.2\nholdout_confidence=0.95\njson_output="run.json"\n'
        if model:
            (root / "model.json").write_text(
                json.dumps(model.to_dict()), encoding="utf8"
            )
        else:
            config += 'score_columns=["gap"]\n'
        config += '[predictions]\ncalibration="calibration.csv"\nholdout="primary.csv"\n[frozen]\nreference="reference.csv"\n'
        if model:
            config += 'scorer="model.json"\n'
        cfg = root / "workflow.toml"
        cfg.write_text(config, encoding="utf8")
        report = run_workflow(cfg, save_outputs=True)
        policy = report["policy"]
        ref = sorted((predict(model, r) for r in parts["reference"]), reverse=True)
        grid = sorted({ref[(len(ref) * i - 1) // 10] for i in range(1, 11)})
        cuts = []
        for t in grid:
            accepted = [r for r in parts["calibration"] if predict(model, r) >= t]
            if accepted and binom.cdf(
                sum(r["correct"] == "0" for r in accepted), len(accepted), 0.2
            ) <= 0.05 / len(grid):
                cuts.append(t)
        cutoff = min(cuts, default=None)
        if cutoff != policy["selection"]["selected_threshold"]:
            raise ValueError("production and reference calibration disagree")
        if model:
            (output / "scorer.json").write_text(
                json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf8"
            )
        audits = {}
        accepted_ids = []
        for role in ("primary", "stress"):
            rows = parts[role]
            ids = {
                r["id"]
                for r in rows
                if cutoff is not None and predict(model, r) >= cutoff
            }
            n = len(ids)
            errors = sum(r["correct"] == "0" and r["id"] in ids for r in rows)
            bound = (
                float(beta.ppf(0.95, errors + 1, n - errors))
                if n and errors < n
                else (1 if n else None)
            )
            audits[role] = dict(
                total=len(rows), accepted=n, errors=errors, upper95=bound
            )
            if role == "primary" and cutoff is not None:
                prod = policy["heldout"]["graders"]["gold"]
                if (prod["accepted"], prod["errors"]) != (n, errors) or (
                    n and abs(prod["risk_upper_bound"] - bound) > 1e-10
                ):
                    raise ValueError("production and reference audit disagree")
            if cutoff is not None:
                path = root / (role + ".csv")
                saved = ["--threshold-report", str(root / "run.json")]
                route_policy = saved
                if not policy["confirmed"]:
                    cli(
                        "route",
                        path,
                        *saved,
                        "--output",
                        root / "blocked.csv",
                        expected_failure="requires a confirmed run report",
                    )
                    # Replay the candidate separately to check the reference mask.
                    # The unconfirmed report itself cannot authorize routing.
                    if model:
                        scored = root / "explicit.csv"
                        apply_scorer(path, scored, model)
                        path = scored
                    route_policy = [
                        "--threshold",
                        str(cutoff),
                        "--score-column",
                        "fitted_score" if model else "gap",
                    ]
                cli("route", path, *route_policy, "--output", root / "decisions.csv")
                actual = {
                    r["id"]
                    for r in read_csv(root / "decisions.csv")
                    if r["decision"] == "accept"
                }
                if actual != ids:
                    raise ValueError("routing differs from independent cutoff")
                for expected in (True, False):
                    block = [
                        r
                        for r in read_csv(root / (role + ".csv"))
                        if (r["id"] in ids) == expected
                    ]
                    if not block:
                        continue
                    path = root / "subset.csv"
                    write_report(path, block)
                    for cmd in ("evaluate", "audit", "monitor"):
                        args = ["--correct-column", "gold"]
                        if cmd == "audit":
                            args += ["--correct-column", "check"]
                        got = cli(cmd, path, *saved, *args)
                        counts = (
                            got["graders"]["gold"] if cmd == "audit" else got["overall"]
                        )
                        if counts["accepted"] != len(block) * expected:
                            raise ValueError(f"{cmd} disagrees on accepted IDs")
            accepted_ids += [{"cohort": role, "id": i} for i in sorted(ids)]
        write_report(output / "accepted.csv", accepted_ids, ["cohort", "id"])
        return {
            "selected_score": best,
            "threshold": cutoff,
            "confirmed": policy["confirmed"],
            "audits": audits,
            "calibration": {
                "accepted": sum(
                    predict(model, r) >= cutoff for r in parts["calibration"]
                ),
                "errors": sum(
                    predict(model, r) >= cutoff and r["correct"] == "0"
                    for r in parts["calibration"]
                ),
            }
            if cutoff is not None
            else None,
        }


def study(table, output, design_only):
    rows = sorted(read_csv(table), key=lambda r: r["id"])
    parts = {
        role: [r for r in rows if r["role"] == role]
        for role in ("design", "reference", "calibration", "primary", "stress")
    }
    design = parts["design"]
    if len(design) != 400 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("unexpected design or duplicate score ID")
    locked, defs = lock_design(design)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "design.json"
    if design_only:
        if len(rows) != len(design):
            raise ValueError("design lock requires a design-only input")
        lock.write_text(json.dumps(locked, indent=2) + "\n", encoding="utf8")
        return
    stored = json.loads(lock.read_text(encoding="utf8"))
    if locked["cheap_arm"] != stored["cheap_arm"] or list(map(len, parts.values())) != [
        400,
        160,
        480,
        846,
        1128,
    ]:
        raise ValueError("score selection or partition differs from the locked design")
    best = locked["cheap_arm"]
    models = {k: fit(k, design, defs) for k in defs}
    result = {}
    for role in ("primary", "stress"):
        target = parts[role]
        vals = {k: [predict(m, r) for r in target] for k, m in models.items()}
        result[role] = {
            "budgets": distributions(target, vals),
            "contrasts": contrasts(
                vals,
                [r["correct"] == "0" for r in target],
                [
                    ("native_class", "native_pair"),
                    ("paid_best", best),
                    ("paid_class", "native_class"),
                ],
            ),
        }
    result["deployment"] = deploy(parts, models, best, output)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", type=Path, default=ROOT / "results/anli/scores.csv")
    p.add_argument("--output", type=Path, default=ROOT / "results/anli")
    p.add_argument("--design-only", action="store_true")
    args = p.parse_args()
    study(args.table, args.output, args.design_only)
