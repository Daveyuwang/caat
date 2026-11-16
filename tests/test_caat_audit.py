import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.cli import main


class AuditTests(unittest.TestCase):
    def run_audit(self, rows, threshold, *flags):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "graded.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["key", "conf", "primary", "alternate", "group"])
                writer.writerows(rows)
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main([
                    "audit", str(source), "--id-column", "key", "--score-column", "conf",
                    "--correct-column", "primary", "--correct-column", "alternate",
                    "--threshold", str(threshold), "--risk-target", "0.4", *flags,
                ])
        return exit_code, json.loads(output.getvalue())

    def test_fixed_set_and_target_crossing(self):
        exit_code, report = self.run_audit([
            ("d", 0.1, 1, 0, "other"),
            ("b", 0.8, 1, 0, "case"),
            ("a", 0.8, 1, 0, "case"),
            ("c", 0.9, 0, 1, "other"),
        ], 0.8, "--fail-on-target")
        self.assertEqual(exit_code, 1)
        primary = report["graders"]["primary"]
        self.assertEqual((primary["accepted"], primary["errors"]), (3, 1))
        self.assertTrue(primary["meets_target"])
        self.assertFalse(report["graders"]["alternate"]["meets_target"])
        self.assertEqual(report["graders"]["alternate"]["by_group"]["case"]["errors"], 2)
        self.assertEqual(report["accepted_comparisons"]["alternate"], {
            "newly_failed": 2, "newly_passed": 1, "risk_change": 1 / 3,
            "overall_risk_change": 1 / 2, "excess_risk_change": 1 / 3 - 1 / 2,
            "target_decision_changed": True,
        })
        self.assertEqual([(row["id"], row["accepted"]) for row in report["discordant"]],
                         [("a", True), ("b", True), ("c", True), ("d", False)])

    def test_zero_coverage_has_no_risk(self):
        exit_code, report = self.run_audit([("a", 0.1, 0, 0, "x")], 0.8, "--fail-on-target")
        self.assertEqual(exit_code, 1)
        self.assertIsNone(report["graders"]["primary"]["error_rate"])
        self.assertIsNone(report["accepted_comparisons"]["alternate"]["target_decision_changed"])

    def test_paired_bootstrap_for_all_and_zero_acceptance(self):
        rows = [("a", 0.7, 1, 0, "x"), ("b", 0.8, 1, 0, "x")]
        for threshold, expected in ((0.5, [0.0, 0.0]), (0.9, None)):
            _, report = self.run_audit(rows, threshold, "--bootstrap", "30", "--seed", "18")
            compared = report["accepted_comparisons"]["alternate"]
            self.assertEqual(compared["overall_risk_change"], 1.0)
            self.assertEqual(compared["bootstrap_95"]["overall_risk_change"], [1.0, 1.0])
            self.assertEqual(compared["bootstrap_95"]["excess_risk_change"], expected)
            self.assertEqual(report["bootstrap"], {"replicates": 30, "seed": 18})

    def test_duplicate_ids_and_missing_grades_are_errors(self):
        for rows, message in (
            ([("a", 0.7, 1, 1, "x"), ("a", 0.8, 0, 1, "x")], "duplicate id"),
            ([("a", 0.7, 1, "", "x")], "correctness must be 0 or 1"),
        ):
            with self.subTest(message=message), redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as raised:
                    self.run_audit(rows, 0.8)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, err.getvalue())

    def test_join_grader_exports_and_reject_stale_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, primary, alternate = (root / name for name in ("pred.csv", "first.csv", "second.csv"))
            predictions.write_text("id,score,output_sha256\na,0.9,a-out\nb,0.6,b-out\n", encoding="utf-8")
            primary.write_text("id,output_sha256,correct\nb,b-out,0\na,a-out,1\n", encoding="utf-8")
            header = "id,output_sha256,correct\n"
            alternate.write_text(header + "a,a-out,0\nb,b-out,0\n", encoding="utf-8")
            args = ["audit", str(predictions), "--threshold", "0.75", "--risk-target", "0.3",
                    "--grader", f"primary={primary}", "--grader", f"alternate={alternate}"]
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(args), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["accepted_comparisons"]["alternate"]["newly_failed"], 1)
            self.assertEqual(report["input"]["grader_files"]["alternate"], str(alternate))

            for contents, message in (
                ("a,earlier-output,0\nb,b-out,0\n", "output key mismatch"),
                ("a,a-out,0\n", "missing 1 ids"),
                ("a,a-out,0\nb,b-out,0\nb,b-out,0\n", "duplicate id"),
                ("a,a-out,0\nb,b-out,0\nc,c-out,0\n", "unexpected id"),
            ):
                alternate.write_text(header + contents, encoding="utf-8")
                with self.subTest(message=message), redirect_stderr(io.StringIO()) as err:
                    with self.assertRaises(SystemExit) as raised:
                        main(args)
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(message, err.getvalue())

    def test_select_run_from_mixed_prediction_csv(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mixed.csv"
            source.write_text(
                "id,score,output_sha256,tier,old,new\n"
                "same-id,0.95,other-output,other,0,0\n"
                "same-id,0.85,selected-output,selected,1,0\n",
                encoding="utf-8",
            )
            old, new = root / "old.csv", root / "new.csv"
            for path, bit in ((old, 1), (new, 0)):
                path.write_text(f"id,output_sha256,correct\nsame-id,selected-output,{bit}\n", encoding="utf-8")

            args = ["audit", str(source), "--threshold", "0.8"]
            for grader_args in (
                ["--correct-column", "old", "--correct-column", "new"],
                ["--grader", f"old={old}", "--grader", f"new={new}"],
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main([*args, *grader_args, "--prediction-filter", "tier=selected"]), 0)
                report = json.loads(output.getvalue())
                self.assertEqual(report["input"]["prediction_filter"], {"column": "tier", "value": "selected"})
                self.assertEqual(report["graders"]["old"]["total"], 1)
                self.assertEqual(report["graders"]["old"]["errors"], 0)
                self.assertEqual(report["graders"]["new"]["errors"], 1)

            with redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as raised:
                main([*args, "--correct-column", "old", "--correct-column", "new"])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("duplicate id", err.getvalue())
