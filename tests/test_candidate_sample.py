import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from fractions import Fraction
from math import comb
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from caat.cli import main
from caat.plan import batch_verdict, sample_error_limits
from caat.metrics import Counts
from caat.rules import assess_accepted_risk
from caat.scorer import LinearScorer
from caat.table import output_fingerprint


class CandidateSampleTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.policy = root / "policy.json"
        self.batch = root / "batch.csv"
        self.plan = root / "plan.json"
        self.sample = root / "sample.csv"
        self.decisions = root / "decisions.csv"
        self.report = {
            "command": "run",
            "input": {
                "id_column": "uid",
                "review_missing_scores": True,
                "output_key_column": "key",
            },
            "selected_score": "value",
            "policy": {
                "confirmed": False,
                "selection": {"selected_threshold": 2, "risk_target": 0.2},
            },
        }
        self.rows = ["a,1,A", "b,2,A", "c,3,B", "d,4,?", "e,,B"]

    def write_inputs(self, rows):
        self.policy.write_text(json.dumps(self.report), encoding="utf-8")
        with self.batch.open("w", encoding="utf-8") as stream:
            stream.write("uid,value,class,key\n")
            for row in rows:
                stream.write(f"{row},key-{row[0]}\n")

    def plan_args(self, *extra, budget=2):
        return list(
            map(
                str,
                (
                    "plan",
                    "--batch-predictions",
                    self.batch,
                    "--threshold-report",
                    self.policy,
                    "--label-budget",
                    budget,
                    "--sample-output",
                    self.sample,
                    "--output",
                    self.plan,
                    *extra,
                ),
            )
        )

    def assert_cli_error(self, args, message):
        with (
            redirect_stderr(io.StringIO()) as err,
            self.assertRaises(SystemExit) as caught,
        ):
            main(args)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn(message, err.getvalue())

    def test_saved_scores_and_draw_match_hand_scored_candidates(self):
        for mode, eligible in (
            ("lower", {"a": -1, "b": -2}),
            ("fitted", {"b": 2, "c": 3, "d": 4}),
            ("categorical", {"a": 3, "b": 4, "c": 3}),
        ):
            with self.subTest(mode=mode):
                if mode == "lower":
                    self.report["input"]["negated_score_source"] = "value"
                    self.report["selected_score"] = "negative_value"
                    self.report["policy"]["selection"]["selected_threshold"] = -2
                else:
                    self.report["input"].pop("negated_score_source", None)
                    self.report["selected_score"] = "fitted_score"
                    self.report["policy"]["selection"]["selected_threshold"] = 2
                    categories = ("A", "B") if mode == "categorical" else ()
                    n = 1 + len(categories)
                    self.report["scorer"] = LinearScorer(
                        ("value",),
                        (1,) * n,
                        (0,) * n,
                        (1,) * n,
                        (1, 2, 0) if categories else (1,),
                        0,
                        0.1,
                        2,
                        "0" * 64,
                        "class" if categories else None,
                        categories,
                    ).to_dict()
                expected = sorted(eligible)
                records = "".join(
                    json.dumps(
                        [i, float(eligible[i]).hex(), "key-" + i], separators=(",", ":")
                    )
                    + "\n"
                    for i in expected
                )
                Random(7).shuffle(expected)
                for rows in (self.rows, reversed(self.rows)):
                    self.write_inputs(rows)
                    original = self.policy.read_bytes()
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(main(self.plan_args("--sample-seed", "7")), 0)
                    plan = json.loads(self.plan.read_bytes())
                    self.assertEqual(
                        plan["input"]["accepted_population_sha256"],
                        sha256(records.encode()).hexdigest(),
                    )
                    self.assertEqual(
                        (plan["total"], plan["review"], plan["accepted_population"]),
                        (5, 5 - len(eligible), len(eligible)),
                    )
                    self.assertEqual((plan["risk_target"], plan["accepted"]), (0.2, 2))
                    with self.sample.open(encoding="utf-8", newline="") as f:
                        sample = list(csv.DictReader(f))
                    self.assertEqual([r["id"] for r in sample], expected[:2])
                    for row in sample:
                        self.assertEqual(float(row["score"]), eligible[row["id"]])
                        self.assertEqual(row["output_key"], "key-" + row["id"])
                        self.assertNotIn("decision", row)
                    self.assertEqual(self.policy.read_bytes(), original)
                    self.assert_cli_error(
                        list(
                            map(
                                str,
                                (
                                    "route",
                                    self.batch,
                                    "--threshold-report",
                                    self.policy,
                                    "--output",
                                    self.decisions,
                                ),
                            )
                        ),
                        "requires a confirmed run report",
                    )
                    self.assertFalse(self.decisions.exists())

    def test_census_empty_population_and_invalid_inputs(self):
        self.write_inputs(self.rows)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.plan_args(budget=100)), 0)
            self.assertEqual(json.loads(self.plan.read_bytes())["accepted"], 3)
            self.assertEqual(main(self.plan_args("--prediction-filter", "uid=a")), 1)
        self.assertEqual(
            self.sample.read_text().splitlines(), ["id,score,group,output_key"]
        )
        plan = json.loads(self.plan.read_bytes())
        self.assertEqual((plan["total"], plan["review"], plan["accepted"]), (1, 1, 0))
        prior = self.sample.read_bytes(), self.plan.read_bytes()
        self.report["input"]["fit_outputs"] = {
            "key_column": "key",
            "sha256": [output_fingerprint("a", "key-a")],
        }
        self.write_inputs(self.rows)
        self.assert_cli_error(self.plan_args(), "reuses output from the fit")
        self.report["policy"]["selection"]["selected_threshold"] = None
        self.write_inputs(self.rows)
        self.assert_cli_error(self.plan_args(), "did not select a cutoff")
        self.assertEqual((self.sample.read_bytes(), self.plan.read_bytes()), prior)

    def test_graders_are_bound_to_the_draw_and_get_separate_verdicts(self):
        self.write_inputs(self.rows)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.plan_args(budget=100)), 0)
        with self.sample.open() as f:
            sample = list(csv.DictReader(f))
        labeled = self.sample.with_name("graded.csv")

        def write_labels(rows):
            with labeled.open("w", newline="") as f:
                w = csv.DictWriter(f, [*sample[0], "first", "second"])
                w.writeheader()
                w.writerows(dict(r, first=1, second=0) for r in rows)

        write_labels(reversed(sample))
        args = list(
            map(
                str,
                [
                    "audit",
                    labeled,
                    "--correct-column",
                    "first",
                    "--correct-column",
                    "second",
                    "--threshold-report",
                    self.policy,
                    "--plan-report",
                    self.plan,
                    "--batch-predictions",
                    self.batch,
                    "--json",
                ],
            )
        )
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(args), 1)
        report = json.loads(out.getvalue())
        self.assertEqual(report["graders"]["first"]["batch_verdict"], "meets")
        self.assertEqual(report["graders"]["second"]["batch_verdict"], "exceeds")
        self.assertEqual(report["min_errors_to_exceed"], 1)
        for bad in (
            sample[:-1],
            [dict(sample[0], output_key="other"), *sample[1:]],
            [dict(sample[0], score="55"), *sample[1:]],
        ):
            write_labels(bad)
            self.assert_cli_error(args, "do not match the fixed-budget draw")
        write_labels(sample)
        self.write_inputs([*self.rows, "f,5,A"])
        self.assert_cli_error(args, "batch population does not match")
        self.report["policy"]["confirmed"] = True
        self.write_inputs(self.rows)
        self.assert_cli_error(args, "threshold report or risk target")

    def test_integer_limits_match_enumerated_tails(self):
        alpha = Fraction(1, 20)
        for n in range(1, 9):
            for m in range(1, n + 1):
                for target in (0, 0.2, 0.5, 1):
                    allowed = int(Fraction(str(target)) * n)

                    def mass(k, x):
                        return Fraction(comb(k, x) * comb(n - k, m - x), comb(n, m))

                    verdicts = []
                    for errors in range(m + 1):
                        upper = max(
                            k
                            for k in range(n + 1)
                            if sum(mass(k, x) for x in range(errors + 1)) > alpha
                        )
                        lower = min(
                            k
                            for k in range(n + 1)
                            if sum(mass(k, x) for x in range(errors, m + 1)) > alpha
                        )
                        wanted = (
                            "meets"
                            if upper <= allowed
                            else ("exceeds" if lower > allowed else "inconclusive")
                        )
                        counts = assess_accepted_risk(
                            Counts(m, m, errors), target, 0.95, accepted_population=n
                        )
                        result = batch_verdict(counts, n, target, 0.95)
                        self.assertEqual(
                            (
                                counts["risk_upper_errors"],
                                result["risk_lower_errors"],
                                result["batch_verdict"],
                            ),
                            (upper, lower, wanted),
                        )
                        verdicts.append(wanted)
                    limits = sample_error_limits(n, m, target, 0.95)
                    self.assertEqual(
                        limits["max_errors_to_meet"],
                        max(
                            (x for x, v in enumerate(verdicts) if v == "meets"),
                            default=None,
                        ),
                    )
                    self.assertEqual(
                        limits["min_errors_to_exceed"],
                        next(
                            (x for x, v in enumerate(verdicts) if v == "exceeds"), None
                        ),
                    )
        self.assertEqual(
            sample_error_limits(588, 100, 0.2, 0.95),
            {"max_errors_to_meet": 13, "min_errors_to_exceed": 27},
        )
