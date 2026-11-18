import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from caat import GradedRow
from caat.cli import main
from caat.policy import split_rows


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ifeval_graders.csv"


class PolicyTests(unittest.TestCase):
    def run_policy(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["policy", *args])
        return exit_code, output.getvalue()

    def test_split_is_reproducible_and_complete(self):
        rows = [GradedRow(str(i), i / 10, (True,)) for i in range(10)]
        design, calibration, heldout = split_rows(rows, 42)
        self.assertEqual((len(design), len(calibration), len(heldout)), (2, 4, 4))
        self.assertEqual(
            sorted(design + calibration + heldout, key=lambda row: row.id), rows
        )
        self.assertEqual(split_rows(rows[::-1], 42), (design, calibration, heldout))

    def test_saved_policy_sends_unscored_outputs_to_review_in_every_check(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            predictions, cutoff, heldout, decisions = (
                root / name
                for name in (
                    "predictions.jsonl",
                    "policy.json",
                    "holdout.csv",
                    "decisions.csv",
                )
            )
            rows = [GradedRow(f"item-{i:02}", 0.4, (True,)) for i in range(20)]
            partitions = split_rows(rows, 42)
            unscored = {part[0].id for part in partitions}
            predictions.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": row.id,
                            "loss": None if row.id in unscored else row.score,
                            "key": f"answer-{row.id}",
                            "primary": row.id not in unscored,
                            "alternate": True,
                        }
                    )
                    for row in rows
                ),
                encoding="utf-8",
            )
            status, output = self.run_policy(
                str(predictions),
                "--score-column",
                "loss",
                "--direction",
                "lower",
                "--correct-column",
                "primary",
                "--check-column",
                "alternate",
                "--review-missing-scores",
                "--output-key-column",
                "key",
                "--track-fit-outputs",
                "--risk-target",
                "0.5",
                "--holdout-confidence",
                "0.95",
                "--calibrate",
                "platt",
                "--output",
                str(cutoff),
                "--holdout-output",
                str(heldout),
                "--json",
            )
            self.assertEqual(status, 0)
            policy = json.loads(output)
            self.assertEqual(
                policy["scored"], {"design": 3, "calibration": 7, "heldout": 7}
            )
            self.assertEqual(policy["selection"]["calibration_total"], 8)
            self.assertEqual(policy["selection"]["selected_threshold"], -0.4)
            self.assertEqual(policy["selection"]["candidates"][0]["coverage"], 7 / 8)
            self.assertEqual(policy["heldout"]["graders"]["primary"]["accepted"], 7)
            calibration = policy["probability_calibration"]["graders"]["primary"][
                "heldout_calibrated"
            ]
            self.assertEqual(
                sum(row["count"] for row in calibration["reliability_bins"]), 7
            )

            common = [str(heldout), "--threshold-report", str(cutoff), "--json"]
            artifacts = root / "evaluation"
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(
                    main(
                        [
                            "evaluate",
                            *common,
                            "--correct-column",
                            "primary",
                            "--output-dir",
                            str(artifacts),
                        ]
                    ),
                    0,
                )
            evaluation = json.loads(stdout.getvalue())
            self.assertEqual(evaluation["input"]["review_missing_scores"], True)
            self.assertEqual(
                evaluation["overall"], policy["heldout"]["graders"]["primary"]
            )
            with (artifacts / "risk_coverage.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                self.assertEqual(next(csv.DictReader(stream))["total"], "8")
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(
                    main(
                        [
                            "audit",
                            *common,
                            "--correct-column",
                            "primary",
                            "--correct-column",
                            "alternate",
                        ]
                    ),
                    0,
                )
            audit = json.loads(stdout.getvalue())
            self.assertEqual(
                audit["accepted_comparisons"]["alternate"]["overall_risk_change"],
                -1 / 8,
            )
            self.assertEqual(audit["scored"], 7)

            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(
                    main(["route", *common, "--output", str(decisions)]), 0
                )
            self.assertEqual(
                json.loads(stdout.getvalue())["overall"]["missing_scores"], 1
            )
            with decisions.open(newline="", encoding="utf-8") as stream:
                decision_rows = list(csv.DictReader(stream))
            self.assertEqual(
                [
                    (row["id"], row["score"])
                    for row in decision_rows
                    if row["decision"] == "review"
                ],
                [(partitions[2][0].id, "")],
            )

    def test_unscored_cohort_completes_without_a_cutoff(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "unscored.csv"
            source.write_text(
                "id,score,correct\n" + "".join(f"{i},,1\n" for i in range(5)),
                encoding="utf-8",
            )
            common = [
                str(source),
                "--correct-column",
                "correct",
                "--review-missing-scores",
                "--risk-target",
                "0.5",
                "--json",
            ]
            status, output = self.run_policy(*common, "--calibrate", "platt")
            self.assertEqual(status, 1)
            self.assertEqual(json.loads(output)["selection"]["candidates"], [])
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(
                    main(["evaluate", *common, "--threshold", "0.5", "--probability"]),
                    1,
                )
            measured = json.loads(stdout.getvalue())
            self.assertEqual(
                (measured["overall"]["total"], measured["overall"]["accepted"]), (5, 0)
            )
            self.assertIsNone(measured["brier_score"])
            self.assertEqual(measured["reliability_bins"], [])

    def test_fit_joins_grader_files_by_output_key_and_exports_joined_holdout(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, grades, cutoff, holdout = (
                root / name
                for name in (
                    "scores.jsonl",
                    "grades.jsonl",
                    "policy.json",
                    "holdout.csv",
                )
            )
            rows = [
                {
                    "id": f"case-{i}",
                    "score": 0.9,
                    "output_sha256": f"{i:064x}",
                    "check": 0,
                }
                for i in range(10)
            ]
            replies = [
                {"id": row["id"], "response_key": row["output_sha256"], "correct": True}
                for row in reversed(rows)
            ]
            stale = {"id": "case-0", "response_key": "old", "correct": False}
            source.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            grades.write_text(
                "\n".join(map(json.dumps, [stale, *replies])), encoding="utf-8"
            )
            args = [
                str(source),
                "--grader-file",
                f"primary={grades}",
                "--grader-output-key-column",
                "response_key",
                "--allow-extra-grades",
                "--track-fit-outputs",
                "--risk-target",
                "0.4",
                "--rule",
                "empirical",
                "--json",
            ]
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(["select", *args, "--thresholds", "0.9"]), 0)
            selection = json.loads(stdout.getvalue())
            self.assertEqual(selection["graders"], ["primary"])
            self.assertEqual(selection["selected_threshold"], 0.9)
            self.assertEqual(
                selection["input"]["fit_outputs"]["key_column"], "output_sha256"
            )

            policy_args = [
                *args,
                "--check-column",
                "check",
                "--output",
                str(cutoff),
                "--holdout-output",
                str(holdout),
            ]
            status, output = self.run_policy(*policy_args)
            self.assertEqual(status, 1)
            policy = json.loads(output)
            self.assertEqual(policy["selection"]["graders"], ["primary"])
            self.assertEqual(policy["selection"]["selected_threshold"], 0.9)
            heldout = policy["heldout"]["graders"]
            self.assertEqual(
                (heldout["primary"]["errors"], heldout["check"]["errors"]), (0, 4)
            )
            with holdout.open(newline="", encoding="utf-8") as stream:
                exported = list(csv.DictReader(stream))
            _, _, split = split_rows(
                [GradedRow(row["id"], 0.9, (True,)) for row in rows], 42
            )
            heldout_ids = {row.id for row in split}
            self.assertEqual(
                exported,
                [
                    {**{key: str(value) for key, value in row.items()}, "primary": "1"}
                    for row in rows
                    if row["id"] in heldout_ids
                ],
            )

            previous = (cutoff.read_bytes(), holdout.read_bytes())
            grades.write_text(
                "\n".join(map(json.dumps, [stale, *replies[:-1]])), encoding="utf-8"
            )
            with (
                redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit) as error,
            ):
                self.run_policy(*policy_args)
            self.assertEqual(error.exception.code, 2)
            self.assertIn(
                "primary: output key mismatch for 'case-0'", errors.getvalue()
            )
            self.assertEqual((cutoff.read_bytes(), holdout.read_bytes()), previous)

    def test_holdout_output_exports_the_selected_source_rows(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "predictions.jsonl"
            holdout = Path(directory) / "review" / "holdout.csv"
            rows = [
                {
                    "case_id": index,
                    "rating": index / 10,
                    "passed": True,
                    "eval_id": "evaluation",
                    "epoch": 1,
                    "output_sha256": f"{index:064x}",
                    "prompt": f" Question {index},\nsecond line ",
                }
                for index in range(9, -1, -1)
            ]
            skipped = {**rows[0], "rating": "unused", "eval_id": "other-evaluation"}
            source.write_text(
                "\n".join(json.dumps(row) for row in [skipped, *rows]), encoding="utf-8"
            )
            args = [
                str(source),
                "--id-column",
                "case_id",
                "--score-column",
                "rating",
                "--correct-column",
                "passed",
                "--risk-target",
                "0.4",
                "--rule",
                "empirical",
                "--seed",
                "17",
                "--adapter",
                "csv",
                "--prediction-filter",
                "eval_id=evaluation",
                "--holdout-output",
                str(holdout),
                "--design-grid-size",
                "1",
                "--json",
            ]
            status, output = self.run_policy(*args)
            self.assertEqual(status, 0)
            report = json.loads(output)
            self.assertEqual(
                report["input"]["prediction_filter"],
                {"column": "eval_id", "value": "evaluation"},
            )
            self.assertEqual(report["input"]["holdout_output"], str(holdout))
            self.assertEqual(report["input"]["design_grid_size"], 1)
            self.assertEqual(len(report["selection"]["candidates"]), 1)
            self.assertEqual(report["split"]["heldout"], 4)
            with holdout.open(encoding="utf-8", newline="") as handle:
                records = list(csv.DictReader(handle))
            expected = [
                {**{name: str(value) for name, value in row.items()}, "passed": "1"}
                for row in rows
                if row["case_id"] in {2, 4, 6, 8}
            ]
            self.assertEqual(records, expected)

            previous = holdout.read_bytes()
            with (
                self.assertRaises(SystemExit) as rejected,
                redirect_stderr(io.StringIO()) as errors,
            ):
                self.run_policy(*args, "--output", str(holdout))
            self.assertEqual(rejected.exception.code, 2)
            self.assertIn("different path", errors.getvalue())
            self.assertEqual(holdout.read_bytes(), previous)

    def test_check_grader_leaves_selection_unchanged(self):
        base = [
            str(EXAMPLE),
            "--score-column",
            "mean_token_logprob",
            "--risk-target",
            "0.4",
            "--correct-column",
            "strict_pass",
        ]
        exit_code, output = self.run_policy(
            *base, "--check-column", "helm_strict_pass", "--json"
        )
        report = json.loads(output)
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            report["split"],
            {"seed": 42, "design": 108, "calibration": 216, "heldout": 217},
        )
        self.assertEqual(report["selection"]["graders"], ["strict_pass"])
        self.assertEqual(
            report["selection"]["selected_threshold"], -0.43964836263206303
        )
        held = report["heldout"]["graders"]
        self.assertEqual(
            (held["strict_pass"]["accepted"], held["strict_pass"]["errors"]), (144, 33)
        )
        self.assertEqual(
            (held["helm_strict_pass"]["accepted"], held["helm_strict_pass"]["errors"]),
            (144, 45),
        )
        self.assertEqual(
            report["heldout"]["accepted_comparisons"]["helm_strict_pass"][
                "newly_failed"
            ],
            12,
        )

        exit_code, output = self.run_policy(
            *base, "--correct-column", "helm_strict_pass", "--json"
        )
        report = json.loads(output)
        self.assertEqual(report["selection"]["selected_threshold"], -0.3327795407672605)
        held = report["heldout"]["graders"]
        self.assertEqual([held[name]["errors"] for name in held], [19, 23])
        self.assertEqual(held["strict_pass"]["accepted"], 92)

        exit_code, summary = self.run_policy(
            *base,
            "--check-column",
            "helm_strict_pass",
            "--risk-target",
            "0.3",
            "--rule",
            "empirical",
        )
        self.assertEqual(exit_code, 1)
        self.assertIn(
            "helm_strict_pass (check only): accepted 144 of 217 with 45 errors "
            "(31.2%, 95% upper bound 38.2%). The observed rate exceeds the target.",
            summary,
        )
        self.assertIn("gives no statistical guarantee.", summary)
        self.assertTrue(summary.rstrip().endswith("Exit status 1."))

    def test_reversed_score_selects_nothing(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "graded.csv"
            rows = [f"{i},{0.1 if i % 2 else 0.9},{i % 2}" for i in range(100)]
            source.write_text(
                "id,score,correct\n" + "\n".join(rows) + "\n", encoding="utf-8"
            )
            exit_code, summary = self.run_policy(
                str(source), "--correct-column", "correct", "--risk-target", "0.3"
            )
            self.assertEqual(exit_code, 1)
            self.assertIn("AUROC 0.000, no better than random", summary)
            self.assertIn("No candidate threshold met the 30% risk target", summary)

            source.write_text(
                "id,score,correct\n1,0.9,1\n2,0.8,1\n3,0.7,0\n4,0.6,1\n",
                encoding="utf-8",
            )
            with (
                self.assertRaises(SystemExit) as caught,
                redirect_stderr(io.StringIO()) as errors,
            ):
                main(
                    [
                        "policy",
                        str(source),
                        "--correct-column",
                        "correct",
                        "--risk-target",
                        "0.3",
                    ]
                )
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("at least five rows", errors.getvalue())

    def test_holdout_bound_leaves_calibration_selection_unchanged(self):
        args = [
            str(EXAMPLE),
            "--score-column",
            "mean_token_logprob",
            "--correct-column",
            "strict_pass",
            "--check-column",
            "helm_strict_pass",
            "--risk-target",
            "0.4",
            "--json",
        ]
        empirical_status, empirical = self.run_policy(*args)
        bound_status, bounded = self.run_policy(*args, "--holdout-confidence", "0.999")
        empirical, bounded = json.loads(empirical), json.loads(bounded)
        self.assertEqual((empirical_status, bound_status), (0, 1))
        self.assertEqual(empirical["selection"], bounded["selection"])
        alternate = bounded["heldout"]["graders"]["helm_strict_pass"]
        self.assertEqual(alternate["error_rate"], 45 / 144)
        self.assertAlmostEqual(alternate["risk_upper_bound"], 0.44206182998747356)
        self.assertFalse(alternate["meets_target"])


if __name__ == "__main__":
    unittest.main()
