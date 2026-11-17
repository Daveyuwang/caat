import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from math import ceil, nextafter
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from caat import GradedRow, audit
from caat.cli import main


class AuditTests(unittest.TestCase):
    def test_multiple_graders_match_direct_seeded_row_resampling(self):
        rows = [
            GradedRow("a", 0.8, (True, False, True)),
            GradedRow("b", 0.7, (False, True, True)),
            GradedRow("c", 0.6, (True, True, False)),
            GradedRow("d", 0.5, (False, False, True)),
        ]
        threshold, replicates, seed = 0.8, 100, 17
        report = audit(
            rows,
            graders=("primary", "one", "two"),
            threshold=threshold,
            bootstrap=replicates,
            seed=seed,
        )
        for index, name in enumerate(("one", "two"), start=1):
            rng = Random(seed)
            risk_values, whole_values, excess_values = [], [], []
            for _ in range(replicates):
                sampled = rng.choices(rows, k=len(rows))
                changes = [row.outcomes[0] - row.outcomes[index] for row in sampled]
                accepted = [
                    change
                    for row, change in zip(sampled, changes, strict=True)
                    if row.score >= threshold
                ]
                whole = sum(changes) / len(changes)
                whole_values.append(whole)
                if accepted:
                    risk = sum(accepted) / len(accepted)
                    risk_values.append(risk)
                    excess_values.append(risk - whole)

            compared = report["accepted_comparisons"][name]["bootstrap_95"]
            self.assertLess(len(risk_values), replicates)
            self.assertEqual(compared["replicates_with_acceptance"], len(risk_values))
            for key, values in zip(
                ("risk_change", "overall_risk_change", "excess_risk_change"),
                (risk_values, whole_values, excess_values),
                strict=True,
            ):
                ordered = sorted(values)
                self.assertEqual(
                    compared[key],
                    [ordered[ceil(p * len(ordered)) - 1] for p in (0.025, 0.975)],
                )

    def run_audit(self, rows, threshold, *flags):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "graded.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["key", "conf", "primary", "alternate", "group"])
                writer.writerows(rows)
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "audit",
                        str(source),
                        "--json",
                        "--id-column",
                        "key",
                        "--score-column",
                        "conf",
                        "--correct-column",
                        "primary",
                        "--correct-column",
                        "alternate",
                        "--threshold",
                        str(threshold),
                        "--risk-target",
                        "0.4",
                        *flags,
                    ]
                )
        return exit_code, json.loads(output.getvalue())

    def test_fixed_set_and_target_crossing(self):
        exit_code, report = self.run_audit(
            [
                ("d", 0.1, 1, 0, "other"),
                ("b", 0.8, 1, 0, "case"),
                ("a", 0.8, 1, 0, "case"),
                ("c", 0.9, 0, 1, "other"),
            ],
            0.8,
        )
        self.assertEqual(exit_code, 1)
        primary = report["graders"]["primary"]
        self.assertEqual((primary["accepted"], primary["errors"]), (3, 1))
        self.assertTrue(primary["meets_target"])
        self.assertFalse(report["graders"]["alternate"]["meets_target"])
        self.assertEqual(
            report["graders"]["alternate"]["by_group"]["case"]["errors"], 2
        )
        self.assertEqual(
            report["accepted_comparisons"]["alternate"],
            {
                "newly_failed": 2,
                "newly_passed": 1,
                "risk_change": 1 / 3,
                "overall_risk_change": 1 / 2,
                "excess_risk_change": 1 / 3 - 1 / 2,
                "target_decision_changed": True,
            },
        )
        self.assertEqual(
            [(row["id"], row["accepted"]) for row in report["discordant"]],
            [("a", True), ("b", True), ("c", True), ("d", False)],
        )

    def test_zero_coverage_has_no_risk(self):
        with TemporaryDirectory() as directory:
            exported = Path(directory) / "disagreements.csv"
            exit_code, report = self.run_audit(
                [("a", 0.1, 0, 0, "x")],
                0.8,
                "--holdout-confidence",
                "0.95",
                "--disagreements-output",
                str(exported),
            )
            self.assertEqual(
                exported.read_text(encoding="utf-8"),
                "id,score,decision,group,grade_primary,grade_alternate\n",
            )
        self.assertEqual(exit_code, 1)
        self.assertIsNone(report["graders"]["primary"]["error_rate"])
        self.assertIsNone(report["graders"]["primary"]["risk_upper_bound"])
        self.assertIsNone(
            report["accepted_comparisons"]["alternate"]["target_decision_changed"]
        )

    def test_holdout_bound_can_fail_when_observed_risk_passes(self):
        exit_code, report = self.run_audit(
            [(str(i), 0.9, 1, int(i >= 4), "x") for i in range(12)],
            0.8,
            "--holdout-confidence",
            "0.95",
        )
        self.assertEqual(exit_code, 1)
        self.assertEqual(report["holdout_confidence"], 0.95)
        self.assertTrue(report["graders"]["primary"]["meets_target"])
        alternate = report["graders"]["alternate"]
        self.assertEqual(alternate["error_rate"], 1 / 3)
        self.assertAlmostEqual(alternate["risk_upper_bound"], 0.6091376982290366)
        self.assertFalse(alternate["meets_target"])
        self.assertTrue(
            report["accepted_comparisons"]["alternate"]["target_decision_changed"]
        )

    def test_fixed_population_audit_checks_both_upper_error_counts(self):
        exit_code, report = self.run_audit(
            [(str(i), 0.9, int(i >= 8), int(i >= 16), "x") for i in range(36)],
            0.8,
            "--holdout-confidence",
            "0.95",
            "--accepted-population",
            "101",
        )
        self.assertEqual(exit_code, 1)
        self.assertEqual(report["accepted_population"], 101)
        primary, alternate = report["graders"].values()
        self.assertEqual(primary["risk_upper_errors"], 34)
        self.assertEqual(alternate["risk_upper_errors"], 57)
        self.assertEqual(primary["risk_upper_bound"], 34 / 101)
        self.assertEqual(alternate["risk_upper_bound"], nextafter(57 / 101, 1.0))
        self.assertTrue(primary["meets_target"])
        self.assertFalse(alternate["meets_target"])

    def test_paired_bootstrap_for_all_and_zero_acceptance(self):
        rows = [("a", 0.7, 1, 0, "x"), ("b", 0.8, 1, 0, "x")]
        for threshold, expected in ((0.5, [0.0, 0.0]), (0.9, None)):
            _, report = self.run_audit(
                rows, threshold, "--bootstrap", "30", "--seed", "18"
            )
            compared = report["accepted_comparisons"]["alternate"]
            self.assertEqual(compared["overall_risk_change"], 1.0)
            self.assertEqual(
                compared["bootstrap_95"]["overall_risk_change"], [1.0, 1.0]
            )
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
            predictions, primary, alternate = (
                root / name for name in ("pred.csv", "first.csv", "second.csv")
            )
            predictions.write_text(
                "id,score,output_sha256\nb,0.6,b-out\na,0.9,a-out\n", encoding="utf-8"
            )
            primary.write_text(
                "id,output_sha256,correct\nb,b-out,0\na,a-out,1\n", encoding="utf-8"
            )
            header = "id,output_sha256,correct\n"
            alternate.write_text(header + "a,a-out,0\nb,b-out,1\n", encoding="utf-8")
            disagreements = root / "disagreements.csv"
            args = [
                "audit",
                str(predictions),
                "--json",
                "--threshold",
                "0.75",
                "--risk-target",
                "0.3",
                "--grader-file",
                f"primary={primary}",
                "--grader-file",
                f"alternate={alternate}",
                "--disagreements-output",
                str(disagreements),
            ]
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(args), 1)
            report = json.loads(output.getvalue())
            self.assertEqual(
                report["accepted_comparisons"]["alternate"]["newly_failed"], 1
            )
            self.assertEqual(
                report["input"]["grader_files"]["alternate"], str(alternate)
            )
            self.assertEqual(
                report["input"]["disagreements_output"], str(disagreements)
            )
            expected = (
                "id,score,decision,group,output_key,grade_primary,grade_alternate\n"
                "a,0.9,accept,,a-out,1,0\nb,0.6,review,,b-out,0,1\n"
            )
            self.assertEqual(disagreements.read_text(encoding="utf-8"), expected)
            queued = root / "queue.csv"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "queue",
                            str(disagreements),
                            "--prediction-filter",
                            "decision=accept",
                            "--output-key-column",
                            "output_key",
                            "--correct-column",
                            "grade_primary",
                            "--budget",
                            "1",
                            "--output",
                            str(queued),
                        ]
                    ),
                    0,
                )
            self.assertEqual(
                queued.read_text(encoding="utf-8"),
                "id,score,group,output_key\na,0.9,,a-out\n",
            )

            for contents, message in (
                ("a,earlier-output,0\nb,b-out,0\n", "output key mismatch"),
                ("a,a-out,0\n", "missing 1 ids"),
                ("a,a-out,0\nb,b-out,0\nb,b-out,0\n", "duplicate id"),
                ("a,a-out,0\nb,b-out,0\nc,c-out,0\n", "unexpected id"),
            ):
                alternate.write_text(header + contents, encoding="utf-8")
                with (
                    self.subTest(message=message),
                    redirect_stderr(io.StringIO()) as err,
                ):
                    with self.assertRaises(SystemExit) as raised:
                        main(args)
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(message, err.getvalue())
                    self.assertEqual(
                        disagreements.read_text(encoding="utf-8"), expected
                    )

    def test_full_grader_exports_match_selected_output_keys(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, primary, alternate = (
                root / name for name in ("sample.csv", "full.csv", "mixed.jsonl")
            )
            predictions.write_text(
                "id,score,output_key\nb,0.6,b-out\na,0.9,a-out\n", encoding="utf-8"
            )
            primary.write_text(
                "grade_id,output_sha256,correct\n"
                "other,other-out,0\na,old-a-out,0\na,a-out,1\nb,b-out,1\n",
                encoding="utf-8",
            )
            alternate.write_text(
                '{"grade_id":"b","output_sha256":"b-out","correct":0}\n'
                '{"grade_id":"a","output_sha256":"a-out","correct":1}\n',
                encoding="utf-8",
            )
            args = [
                "audit",
                str(predictions),
                "--threshold",
                "0.6",
                "--output-key-column",
                "output_key",
                "--grader-id-column",
                "grade_id",
                "--grader-output-key-column",
                "output_sha256",
                "--allow-extra-grades",
                "--grader-file",
                f"primary={primary}",
                "--grader-file",
                f"alternate={alternate}",
                "--json",
            ]
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(args), 0)
            report = json.loads(stdout.getvalue())
            self.assertTrue(report["input"]["allow_extra_grades"])
            self.assertEqual(report["input"]["grader_id_column"], "grade_id")
            self.assertEqual(
                report["input"]["grader_output_key_column"], "output_sha256"
            )
            self.assertEqual(report["graders"]["primary"]["accepted"], 2)
            self.assertEqual(report["graders"]["primary"]["errors"], 0)
            self.assertEqual(report["graders"]["alternate"]["errors"], 1)

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
                path.write_text(
                    f"id,output_sha256,correct\nsame-id,selected-output,{bit}\n",
                    encoding="utf-8",
                )

            args = ["audit", str(source), "--json", "--threshold", "0.8"]
            for grader_args in (
                ["--correct-column", "old", "--correct-column", "new"],
                ["--grader-file", f"old={old}", "--grader-file", f"new={new}"],
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(
                        main(
                            [
                                *args,
                                *grader_args,
                                "--prediction-filter",
                                "tier=selected",
                            ]
                        ),
                        0,
                    )
                report = json.loads(output.getvalue())
                self.assertEqual(
                    report["input"]["prediction_filter"],
                    {"column": "tier", "value": "selected"},
                )
                self.assertEqual(report["graders"]["old"]["total"], 1)
                self.assertEqual(report["graders"]["old"]["errors"], 0)
                self.assertEqual(report["graders"]["new"]["errors"], 1)

            with (
                redirect_stderr(io.StringIO()) as err,
                self.assertRaises(SystemExit) as raised,
            ):
                main([*args, "--correct-column", "old", "--correct-column", "new"])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("duplicate id", err.getvalue())

    def test_malformed_grader_csv_cannot_change_an_audit_decision(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, reference, grader, report = (
                root / name
                for name in ("pred.csv", "ref.csv", "grader.csv", "report.json")
            )
            predictions.write_text(
                "id,score,output_sha256\na,0.9,out\n", encoding="utf-8"
            )
            reference.write_text(
                "id,output_sha256,correct\na,out,1\n", encoding="utf-8"
            )
            header = "id,output_sha256,annotation,correct\n\n"
            args = [
                "audit",
                str(predictions),
                "--threshold",
                "0.8",
                "--risk-target",
                "0.4",
                "--grader-file",
                f"reference={reference}",
                "--grader-file",
                f"alternate={grader}",
                "--output",
                str(report),
            ]
            for row, message in (
                ("a,out,label,1,0\n", "line 3 has 5 fields"),
                ("a,out,0\n", "line 3 has 3 fields"),
                ('a,out,"label,1,0\n', "CSV line 3"),
            ):
                with self.subTest(row=row), redirect_stderr(io.StringIO()) as err:
                    grader.write_text(header + row, encoding="utf-8")
                    with self.assertRaises(SystemExit) as raised:
                        main(args)
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(str(grader), err.getvalue())
                    self.assertIn(message, err.getvalue())
                    self.assertFalse(report.exists())

            grader.write_text(header + 'a,out,"label,1\nchecked",0\n', encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 1)
            saved = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(saved["graders"]["alternate"]["errors"], 1)

    def test_replay_a_saved_selection_at_full_precision(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            calibration = root / "cal.csv"
            holdout = root / "holdout.csv"
            selection = root / "cutoff.json"
            result = root / "checked.json"
            cutoff = "0.7981867790222168"
            header = "answer_id,confidence,score,old,new\n"
            calibration.write_text(header + f"cal,{cutoff},0,1,1\n", encoding="utf-8")
            holdout.write_text(
                header + f"a,{cutoff},0,1,0\nb,0.95,0,1,1\nc,0.7,1,1,1\n",
                encoding="utf-8",
            )
            select_args = [
                "select",
                str(calibration),
                "--thresholds",
                cutoff,
                "--id-column",
                "answer_id",
                "--score-column",
                "confidence",
                "--correct-column",
                "old",
                "--risk-target",
                "0.4",
                "--rule",
                "empirical",
                "--output",
                str(selection),
            ]
            self.assertEqual(main(select_args), 0)
            use_args = [
                str(holdout),
                "--threshold-report",
                str(selection),
                "--correct-column",
                "old",
                "--output",
                str(result),
            ]
            self.assertEqual(main(["evaluate", *use_args]), 0)
            report = json.loads(result.read_bytes())
            self.assertEqual(report["overall"]["accepted"], 2)
            self.assertEqual(report["input"]["score_column"], "confidence")
            self.assertEqual(main(["audit", *use_args, "--correct-column", "new"]), 1)
            report = json.loads(result.read_bytes())
            self.assertEqual(report["threshold"], float(cutoff))
            self.assertEqual(report["risk_target"], 0.4)
            self.assertEqual(report["input"]["threshold_report"], str(selection))
            self.assertEqual(report["graders"]["new"]["errors"], 1)

            original = selection.read_bytes()
            with (
                redirect_stderr(io.StringIO()) as error,
                self.assertRaises(SystemExit) as raised,
            ):
                main(["evaluate", *use_args, "--output", str(selection)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("output would overwrite an input file", error.getvalue())
            self.assertEqual(selection.read_bytes(), original)

            empty = json.loads(original)
            empty["selected_threshold"] = None
            selection.write_text(json.dumps(empty), encoding="utf-8")
            with (
                redirect_stderr(io.StringIO()) as error,
                self.assertRaises(SystemExit) as raised,
            ):
                main(["evaluate", *use_args])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("did not select a cutoff", error.getvalue())
