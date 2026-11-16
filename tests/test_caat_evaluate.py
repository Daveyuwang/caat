import csv
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from caat import GradedRow, evaluate, load_csv
from caat.cli import main


class EvaluationTests(unittest.TestCase):
    def test_custom_csv_with_tied_scores(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "ticket_triage.csv"
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            main([
                "evaluate", str(example), "--threshold", "0.75", "--probability",
                "--id-column", "ticket_id", "--prediction-column", "model_route",
                "--reference-column", "expected_route", "--score-column", "confidence",
                "--group-column", "inbox", "--output-dir", temp,
            ])
            metrics = json.loads((Path(temp) / "metrics.json").read_text(encoding="utf-8"))
            with (Path(temp) / "risk_coverage.csv").open(encoding="utf-8", newline="") as source:
                curve = list(csv.DictReader(source))

        self.assertEqual(metrics["overall"], {
            "total": 5, "accepted": 3, "errors": 1, "coverage": 0.6, "error_rate": 1 / 3,
        })
        self.assertEqual(metrics["by_group"]["email"]["errors"], 1)
        self.assertEqual(metrics["by_reference"]["security"]["accepted"], 1)
        self.assertAlmostEqual(metrics["brier_score"], 0.201)
        self.assertEqual([int(row["accepted"]) for row in curve], [1, 3, 4, 5])

    def test_rank_score_with_no_accepted_rows(self):
        metrics, _ = evaluate([GradedRow("1", -2.0, (True,))], threshold=0.0)
        self.assertEqual(metrics["overall"]["coverage"], 0.0)
        self.assertIsNone(metrics["overall"]["error_rate"])
        self.assertIsNone(metrics["brier_score"])

    def test_external_evaluator_results(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            predictions = root / "checks.csv"
            predictions.write_text(
                "id,score,strict_pass,group\n1,0.9,0,format\n2,0.8,1,format\n3,0.6,1,length\n",
                encoding="utf-8",
            )
            main(["evaluate", str(predictions), "--correct-column", "strict_pass",
                  "--threshold", "0.75", "--output-dir", temp])
            metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["overall"]["accepted"], 2)
        self.assertEqual(metrics["overall"]["error_rate"], 0.5)
        self.assertEqual(metrics["by_group"]["format"]["errors"], 1)
        self.assertEqual(metrics["by_reference"], {})

    def test_external_result_takes_precedence_over_exact_match(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "graded.csv"
            source.write_text("id,score,prediction,reference,g\n1,0.9,a,A,1\n", encoding="utf-8")
            rows = load_csv(source, correct_columns=("g",), reference_column="reference")
            metrics, _ = evaluate(rows, threshold=0.5)
        self.assertEqual(metrics["by_reference"]["A"]["errors"], 0)

    def test_invalid_inputs(self):
        row = GradedRow("1", 0.9, (True,))
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            evaluate([row, row], threshold=0.5)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            evaluate([GradedRow("1", float("nan"), (True,))], threshold=0.5)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            evaluate([GradedRow("1", 2.0, (True,))], threshold=0.5, probability=True)
