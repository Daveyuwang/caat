import csv
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from math import nextafter
from pathlib import Path
from tempfile import TemporaryDirectory

from caat import GradedRow, evaluate, load_csv
from caat.cli import main


class EvaluationTests(unittest.TestCase):
    def test_evaluate_matches_separate_grades_and_groups_by_reference(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()) as stdout:
            root = Path(directory)
            predictions, grades = root / "predictions.csv", root / "grades.jsonl"
            predictions.write_text(
                "answer_id,score,response_key,category\na,0.9,a-v2,short\nb,0.2,b-v1,long\n",
                encoding="utf-8",
            )
            grades.write_text(
                '{"id":"b","output_sha256":"b-v1","correct":false}\n'
                '{"id":"a","output_sha256":"a-v1","correct":false}\n'
                '{"id":"a","output_sha256":"a-v2","correct":true}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                main(
                    [
                        "evaluate",
                        str(predictions),
                        "--id-column",
                        "answer_id",
                        "--output-key-column",
                        "response_key",
                        "--grader-id-column",
                        "id",
                        "--grader-output-key-column",
                        "output_sha256",
                        "--allow-extra-grades",
                        "--grader-file",
                        f"reference={grades}",
                        "--reference-column",
                        "category",
                        "--threshold",
                        "0.8",
                        "--probability",
                        "--json",
                    ]
                ),
                0,
            )
            report = json.loads(stdout.getvalue())
            self.assertEqual(
                report["input"]["grader_files"], {"reference": str(grades)}
            )
            self.assertEqual(report["overall"]["accepted"], 1)
            self.assertEqual(report["overall"]["errors"], 0)
            self.assertEqual(report["by_reference"]["short"]["accepted"], 1)
            self.assertEqual(report["by_reference"]["long"]["accepted"], 0)
            self.assertAlmostEqual(report["brier_score"], 0.025)

    def test_custom_csv_with_tied_scores(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "ticket_triage.csv"
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            main(
                [
                    "evaluate",
                    str(example),
                    "--threshold",
                    "0.75",
                    "--probability",
                    "--id-column",
                    "ticket_id",
                    "--prediction-column",
                    "model_route",
                    "--reference-column",
                    "expected_route",
                    "--score-column",
                    "confidence",
                    "--group-column",
                    "inbox",
                    "--output-dir",
                    temp,
                ]
            )
            metrics = json.loads(
                (Path(temp) / "metrics.json").read_text(encoding="utf-8")
            )
            with (Path(temp) / "risk_coverage.csv").open(
                encoding="utf-8", newline=""
            ) as source:
                curve = list(csv.DictReader(source))

        self.assertEqual(
            metrics["overall"],
            {
                "total": 5,
                "accepted": 3,
                "errors": 1,
                "coverage": 0.6,
                "error_rate": 1 / 3,
            },
        )
        self.assertEqual(metrics["by_group"]["email"]["errors"], 1)
        self.assertEqual(metrics["by_reference"]["security"]["accepted"], 1)
        self.assertAlmostEqual(metrics["brier_score"], 0.201)
        self.assertEqual([int(row["accepted"]) for row in curve], [1, 3, 4, 5])

    def test_rank_score_with_no_accepted_rows(self):
        metrics, _ = evaluate([GradedRow("1", -2.0, (True,))], threshold=0.0)
        self.assertEqual(metrics["overall"]["coverage"], 0.0)
        self.assertIsNone(metrics["overall"]["error_rate"])
        self.assertIsNone(metrics["brier_score"])

    def test_observed_risk_compares_exact_error_counts_to_the_target(self):
        rows = [GradedRow(str(i), 0.9, (i != 0,)) for i in range(3)]
        for target, passes in ((1 / 3, False), (nextafter(1 / 3, 1.0), True)):
            with self.subTest(risk_target=target):
                metrics, _ = evaluate(rows, threshold=0.8, risk_target=target)
                self.assertEqual(metrics["overall"]["error_rate"], 1 / 3)
                self.assertIs(metrics["overall"]["meets_target"], passes)

    def test_fixed_population_census_reports_observed_risk(self):
        rows = [
            GradedRow("a", 0.9, (True,)),
            GradedRow("b", 0.9, (False,)),
            GradedRow("review", 0.1, (False,)),
        ]
        result, _ = evaluate(
            rows,
            threshold=0.8,
            risk_target=0.5,
            holdout_confidence=0.95,
            accepted_population=2,
        )
        self.assertEqual(result["accepted_population"], 2)
        self.assertEqual(result["overall"]["risk_upper_errors"], 1)
        self.assertEqual(result["overall"]["risk_upper_bound"], 0.5)
        self.assertTrue(result["overall"]["meets_target"])
        with self.assertRaisesRegex(ValueError, "exceeds the accepted population"):
            evaluate(
                rows, threshold=0.8, holdout_confidence=0.95, accepted_population=1
            )

    def test_fixed_population_compares_exact_error_counts_to_the_target(self):
        rows = [GradedRow(str(index), 0.9, (index != 0,)) for index in range(3)]
        bound = nextafter(1 / 3, 1.0)
        for target, passes in ((1 / 3, False), (bound, True)):
            with self.subTest(risk_target=target):
                result, _ = evaluate(
                    rows,
                    threshold=0.8,
                    risk_target=target,
                    holdout_confidence=0.95,
                    accepted_population=3,
                )
                overall = result["overall"]
                self.assertEqual(overall["risk_upper_errors"], 1)
                self.assertEqual(overall["risk_upper_bound"], bound)
                self.assertIs(overall["meets_target"], passes)

    def test_external_evaluator_results(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            predictions = root / "checks.csv"
            predictions.write_text(
                "id,score,strict_pass,group\n1,0.9,0,format\n2,0.8,1,format\n3,0.6,1,length\n",
                encoding="utf-8",
            )
            main(
                [
                    "evaluate",
                    str(predictions),
                    "--correct-column",
                    "strict_pass",
                    "--threshold",
                    "0.75",
                    "--output-dir",
                    temp,
                ]
            )
            metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["overall"]["accepted"], 2)
        self.assertEqual(metrics["overall"]["error_rate"], 0.5)
        self.assertEqual(metrics["by_group"]["format"]["errors"], 1)
        self.assertEqual(metrics["by_reference"], {})

    def test_external_result_takes_precedence_over_exact_match(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "graded.csv"
            source.write_text(
                "id,score,prediction,reference,g\n1,0.9,a,A,1\n", encoding="utf-8"
            )
            rows = load_csv(
                source, correct_columns=("g",), reference_column="reference"
            )
            metrics, _ = evaluate(rows, threshold=0.5)
        self.assertEqual(metrics["by_reference"]["A"]["errors"], 0)

    def test_frozen_holdout_risk_gate(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            path = Path(directory) / "holdout.csv"
            path.write_text(
                "id,score,correct\na,0.9,1\nb,0.8,0\nc,0.8,1\nd,0.5,0\n",
                encoding="utf-8",
            )
            output = path.with_suffix(".json")
            args = ["evaluate", str(path), "--correct-column", "correct"]
            args += ["--output", str(output), "--risk-target", "0.4"]
            self.assertEqual(main([*args, "--threshold", "0.8"]), 0)
            empirical = json.loads(output.read_text(encoding="utf-8"))["overall"]
            self.assertTrue(empirical["meets_target"])
            self.assertNotIn("risk_upper_bound", empirical)

            args += ["--holdout-confidence", "0.95"]
            self.assertEqual(main([*args, "--threshold", "0.8"]), 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["holdout_confidence"], 0.95)
            self.assertEqual(report["risk_target"], 0.4)
            self.assertEqual(report["overall"]["accepted"], 3)
            self.assertEqual(report["overall"]["error_rate"], 1 / 3)
            self.assertAlmostEqual(report["overall"]["risk_upper_bound"], 0.86464964)
            self.assertFalse(report["overall"]["meets_target"])

            self.assertEqual(main([*args, "--threshold", "1"]), 1)
            empty = json.loads(output.read_text(encoding="utf-8"))["overall"]
            self.assertIsNone(empty["risk_upper_bound"])
            self.assertIsNone(empty["meets_target"])

    def test_independent_holdout_uses_the_exact_bound_at_the_target(self):
        for labels, confidence, target, bound, passes in (
            (3, 0.5, 0.5, 0.5, True),
            (4, 0.6875, nextafter(0.5, 0), 0.5, False),
            (2, 1e-300, 1e-150, 1e-150, True),
        ):
            rows = [GradedRow(str(index), 0.9, (index > 0,)) for index in range(labels)]
            report, _ = evaluate(
                rows, threshold=0.8, risk_target=target, holdout_confidence=confidence
            )
            self.assertEqual(report["overall"]["risk_upper_bound"], bound)
            self.assertIs(report["overall"]["meets_target"], passes)

    def test_invalid_inputs(self):
        row = GradedRow("1", 0.9, (True,))
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            evaluate([row, row], threshold=0.5)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            evaluate([GradedRow("1", float("nan"), (True,))], threshold=0.5)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            evaluate([GradedRow("1", 2.0, (True,))], threshold=0.5, probability=True)
        with self.assertRaisesRegex(ValueError, "risk target"):
            evaluate([row], threshold=0.5, risk_target=-0.1)
        with self.assertRaisesRegex(ValueError, "holdout confidence"):
            evaluate([row], threshold=0.5, holdout_confidence=1)
