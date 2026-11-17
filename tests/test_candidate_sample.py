import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from caat.cli import main
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
