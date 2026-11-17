import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from fractions import Fraction
from math import nextafter
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.cli import main
from caat.monitor import monitor_risk
from caat.table import GradedRow


def inputs(outcomes, score=0.9):
    return [
        GradedRow(str(index), score, (outcome,))
        for index, outcome in enumerate(outcomes, 1)
    ]


def direct_mixture(outcomes, risk_target):
    risk = Fraction(str(risk_target))
    fractions = (Fraction(1, 10), Fraction(3, 10), Fraction(3, 5))
    peak = Fraction(1)
    for end in range(1, len(outcomes) + 1):
        evidence = Fraction(1, end + 1)
        for start in range(1, end + 1):
            correct = sum(outcomes[start - 1 : end])
            errors = end - start + 1 - correct
            for fraction in fractions:
                q = risk + fraction * (1 - risk)
                evidence += (
                    (q / risk) ** errors
                    * ((1 - q) / (1 - risk)) ** correct
                    / (3 * start * (start + 1))
                )
        peak = max(peak, evidence)
    return float(1 / peak)


class MonitorTests(unittest.TestCase):
    def test_recurrence_matches_the_exact_direct_mixture(self):
        target = 0.27
        outcomes = (False, False, True, True, False, False, False, True, False)
        report = monitor_risk(
            inputs(outcomes), grader="passed", threshold=0.9, risk_target=target
        )
        self.assertEqual(report["anytime_pvalue"], direct_mixture(outcomes, target))
        self.assertEqual(report["overall"]["accepted"], len(outcomes))

    def test_alert_compares_exact_evidence_at_the_test_level(self):
        rows = inputs([False])
        for risk, rate, pvalue, alerts in (
            (0.1, 0.4, 0.4, True),
            (0.1, nextafter(0.4, 0), 0.4, False),
            (0.4, nextafter(0.8, 0), 0.8, False),
        ):
            with self.subTest(risk=risk, rate=rate):
                report = monitor_risk(
                    rows,
                    grader="passed",
                    threshold=0.8,
                    risk_target=risk,
                    failure_rate=rate,
                )
                self.assertEqual(report["anytime_pvalue"], pvalue)
                self.assertEqual(report["first_alert"] is not None, alerts)

    def test_subnormal_rates_retain_counts_after_pvalue_rounds_to_zero(self):
        report = monitor_risk(
            inputs([False, False, True]),
            grader="passed",
            threshold=0.8,
            risk_target=5e-324,
            failure_rate=5e-324,
        )
        self.assertEqual(report["first_alert"]["row"], 2)
        self.assertEqual(report["anytime_pvalue"], 0.0)
        self.assertEqual(report["overall"]["accepted"], 3)
        self.assertEqual(report["overall"]["errors"], 2)

    def test_late_failure_and_recovery_keep_the_first_alert(self):
        history = inputs([True] * 300 + [False] * 40 + [True] * 300)
        ignored = GradedRow("rejected", 0.1, (False,))
        rows = [ignored, *history]
        report = monitor_risk(rows, grader="passed", threshold=0.8, risk_target=0.1)
        alert = report["first_alert"]
        self.assertLessEqual(report["anytime_pvalue"], 0.05)
        self.assertGreater(alert["accepted"], 300)
        self.assertLessEqual(alert["accepted"], 340)
        self.assertEqual(alert["row"], alert["accepted"] + 1)
        self.assertEqual(report["overall"]["errors"], 40)
        prefix = monitor_risk(
            rows[: alert["row"] - 1], grader="passed", threshold=0.8, risk_target=0.1
        )
        self.assertIsNone(prefix["first_alert"])

    def test_boundary_risks_and_unaccepted_rows(self):
        rows = inputs([True, False])
        no_rows = monitor_risk(rows, grader="pass", threshold=1, risk_target=0)
        self.assertIsNone(no_rows["first_alert"])
        self.assertIsNone(no_rows["overall"]["error_rate"])
        zero = monitor_risk(rows, grader="pass", threshold=0.9, risk_target=0)
        self.assertEqual(zero["first_alert"]["id"], "2")
        self.assertEqual(zero["anytime_pvalue"], 0.0)
        one = monitor_risk(rows, grader="pass", threshold=0.9, risk_target=1)
        self.assertIsNone(one["first_alert"])
        self.assertEqual(one["anytime_pvalue"], 1.0)

    def test_cli_keeps_alert_report_and_invalid_inputs_return_two(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "arrivals.jsonl"
            path.write_text(
                '{"id":"s1","score":0.9,"passed":false}\n', encoding="utf-8"
            )
            report = root / "monitor.json"
            args = [
                "monitor",
                str(path),
                "--correct-column",
                "passed",
                "--threshold",
                "0.9",
                "--risk-target",
                "0",
                "--output",
                str(report),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 1)
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result["command"], "monitor")
            self.assertEqual(result["first_alert"]["id"], "s1")
            for options in (
                ("--failure-rate", "1"),
                ("--min-coverage", "0.5"),
                ("--coverage-window", "0", "--min-coverage", "0.5"),
                ("--coverage-window", "1", "--min-coverage", "nan"),
            ):
                with (
                    self.subTest(options=options),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    main([*args, *options])
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(json.loads(report.read_text(encoding="utf-8")), result)

    def test_coverage_alert_checks_complete_windows_before_labels_arrive(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, labels, output = (
                root / name
                for name in ("predictions.csv", "labels.jsonl", "monitor.json")
            )
            labels.write_text("", encoding="utf-8")
            args = [
                "monitor",
                str(predictions),
                "--threshold",
                "0.8",
                "--risk-target",
                "0.1",
                "--grader-file",
                f"human={labels}",
                "--review-missing-scores",
                "--coverage-window",
                "50",
                "--min-coverage",
                "0.14",
                "--output",
                str(output),
            ]
            score_cells = ["0.9"] * 7 + ["0.1"] + [""] * 43
            for length, windows, status in ((49, 0, 0), (50, 1, 0), (51, 2, 1)):
                with (
                    self.subTest(length=length),
                    redirect_stdout(io.StringIO()) as stdout,
                ):
                    predictions.write_text(
                        "id,score,output_sha256\n"
                        + "".join(
                            f"s{i},{score},key-{i}\n"
                            for i, score in enumerate(score_cells[:length], 1)
                        ),
                        encoding="utf-8",
                    )
                    self.assertEqual(main(args), status)
                    report = json.loads(output.read_bytes())
                    self.assertEqual(report["overall"]["total"], 0)
                    self.assertIsNone(report["first_alert"])
                    coverage = report["coverage_check"]
                    self.assertEqual(coverage["windows_checked"], windows)
                    latest = coverage["latest_window"]
                    if status:
                        self.assertEqual(
                            coverage["first_alert"],
                            {
                                "id": "s51",
                                "start_row": 2,
                                "end_row": 51,
                                "accepted": 6,
                                "coverage": 0.12,
                                "missing_scores": 43,
                            },
                        )
                        self.assertEqual(
                            latest, {**coverage["first_alert"], "meets_minimum": False}
                        )
                        self.assertIn(
                            "Coverage alert at rows 2 to 51", stdout.getvalue()
                        )
                    else:
                        self.assertIsNone(coverage["first_alert"])
                        if windows:
                            self.assertEqual(
                                latest,
                                {
                                    "id": "s50",
                                    "start_row": 1,
                                    "end_row": 50,
                                    "accepted": 7,
                                    "coverage": 0.14,
                                    "missing_scores": 42,
                                    "meets_minimum": True,
                                },
                            )
                        else:
                            self.assertIsNone(latest)

    def test_monitor_coverage_without_labels_matches_unkeyed_routes(self):
        with (
            TemporaryDirectory() as directory,
            redirect_stdout(io.StringIO()) as stdout,
        ):
            root = Path(directory)
            scores, decisions, output = (
                root / name for name in ("scores.csv", "decisions.csv", "monitor.json")
            )
            scores.write_text("id,score\na,0.9\nb,\nc,0.9\nd,0.9\n", encoding="utf-8")
            cutoff = ("--threshold", "0.8")
            review = "--review-missing-scores"
            self.assertEqual(
                main(
                    ["route", str(scores), *cutoff, review, "--output", str(decisions)]
                ),
                0,
            )
            common = (
                *cutoff,
                "--coverage-window",
                "2",
                "--min-coverage",
                "0.75",
                "--output",
                str(output),
            )
            raw_args = ("monitor", str(scores), *common, review)
            self.assertEqual(main(raw_args), 1)
            raw = json.loads(output.read_bytes())
            self.assertEqual(
                main(["monitor", "--batch-decisions", str(decisions), *common]), 1
            )
            routed = json.loads(output.read_bytes())
            self.assertEqual(
                raw["overall"],
                {"total": 4, "accepted": 3, "coverage": 0.75, "missing_scores": 1},
            )
            self.assertEqual(routed["overall"], raw["overall"])
            self.assertEqual(routed["coverage_check"], raw["coverage_check"])
            self.assertEqual(raw["coverage_check"]["first_alert"]["id"], "b")
            self.assertEqual(
                raw["coverage_check"]["latest_window"],
                {
                    "id": "d",
                    "start_row": 3,
                    "end_row": 4,
                    "accepted": 2,
                    "coverage": 1.0,
                    "missing_scores": 0,
                    "meets_minimum": True,
                },
            )
            self.assertIsNone(routed["input"]["output_key_column"])
            self.assertNotIn("anytime_pvalue", raw)
            self.assertNotIn("first_alert", routed)
            self.assertIn("Accepted 3 of 4 rows", stdout.getvalue())
            self.assertIn(
                "Latest window at rows 3 to 4 accepted 2 of 2", stdout.getvalue()
            )
            self.assertNotIn("Anytime p-value", stdout.getvalue())

            for options in (("--risk-target", "0.1"), ("--failure-rate", "0.05")):
                with (
                    self.subTest(options=options),
                    redirect_stderr(io.StringIO()) as errors,
                    self.assertRaises(SystemExit) as rejected,
                ):
                    main([*raw_args, *options])
                self.assertEqual(rejected.exception.code, 2)
                self.assertIn(
                    "require --correct-column or --grader-file", errors.getvalue()
                )
                self.assertEqual(json.loads(output.read_bytes()), routed)

    def test_monitor_uses_the_saved_cutoff_with_its_own_alert_level(self):
        with TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            calibration = root / "calibration.csv"
            arrivals = root / "arrivals.csv"
            cutoff = root / "policy.json"
            output = root / "monitor.json"
            columns = "answer_id,confidence,correct\n"
            calibration.write_text(
                columns + "".join(f"fit-{i},0.9,1\n" for i in range(5)),
                encoding="utf-8",
            )
            arrivals.write_text(columns + "a,0.8,0\nb,0.9,0\nc,1,1\n", encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "policy",
                        str(calibration),
                        "--id-column",
                        "answer_id",
                        "--score-column",
                        "confidence",
                        "--correct-column",
                        "correct",
                        "--risk-target",
                        "0",
                        "--rule",
                        "empirical",
                        "--failure-rate",
                        "0.20",
                        "--holdout-confidence",
                        "0.99",
                        "--output",
                        str(cutoff),
                    ]
                ),
                1,
            )
            args = [
                "monitor",
                str(arrivals),
                "--correct-column",
                "correct",
                "--threshold-report",
                str(cutoff),
            ]
            self.assertEqual(main([*args, "--output", str(output)]), 1)
            report = json.loads(output.read_bytes())
            self.assertEqual(report["threshold"], 0.9)
            self.assertEqual(report["risk_target"], 0)
            self.assertEqual(report["failure_rate"], 0.05)
            self.assertEqual(report["input"]["threshold_report"], str(cutoff))
            self.assertEqual(report["input"]["score_column"], "confidence")
            self.assertEqual(
                report["first_alert"], {"id": "b", "row": 2, "accepted": 1, "errors": 1}
            )
            original = cutoff.read_bytes()
            with (
                redirect_stderr(io.StringIO()) as err,
                self.assertRaises(SystemExit) as raised,
            ):
                main([*args, "--output", str(cutoff)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("output would overwrite a monitoring input", err.getvalue())
            self.assertEqual(cutoff.read_bytes(), original)

    def test_monitor_checks_tracked_fit_outputs_before_alerting(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            calibration, arrivals, cutoff, output = (
                root / name
                for name in ("fit.csv", "arrivals.csv", "cutoff.json", "monitor.json")
            )
            header = "id,score,key,correct\n"
            calibration.write_text(header + "a,0.9,fit-output,1\n", encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "select",
                        str(calibration),
                        "--correct-column",
                        "correct",
                        "--thresholds",
                        "0.8",
                        "--risk-target",
                        "0",
                        "--rule",
                        "empirical",
                        "--output-key-column",
                        "key",
                        "--track-fit-outputs",
                        "--output",
                        str(cutoff),
                    ]
                ),
                0,
            )
            args = [
                "monitor",
                str(arrivals),
                "--threshold-report",
                str(cutoff),
                "--correct-column",
                "correct",
                "--output",
                str(output),
            ]
            arrivals.write_text(header + "a,0.9,new-output,0\n", encoding="utf-8")
            self.assertEqual(main(args), 1)
            original = output.read_bytes()
            report = json.loads(original)
            self.assertTrue(report["input"]["fit_outputs_checked"])
            self.assertEqual(report["input"]["output_key_column"], "key")
            self.assertEqual(report["first_alert"]["id"], "a")

            arrivals.write_text(header + "a,0.1,fit-output,0\n", encoding="utf-8")
            with (
                redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit) as rejected,
            ):
                main(args)
            self.assertEqual(rejected.exception.code, 2)
            self.assertIn(
                "monitoring stream reuses output from the fit", errors.getvalue()
            )
            self.assertEqual(output.read_bytes(), original)

    def test_monitor_waits_for_labels_in_prediction_order(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            predictions, labels, output = (
                root / name
                for name in ("predictions.csv", "labels.jsonl", "monitor.json")
            )
            predictions.write_text(
                "id,score,output_sha256\na,0.9,key-a\nr,0.1,key-r\nb,0.9,key-b\nc,0.9,key-c\n",
                encoding="utf-8",
            )
            args = [
                "monitor",
                str(predictions),
                "--threshold",
                "0.8",
                "--risk-target",
                "0",
                "--grader-file",
                f"human={labels}",
                "--output",
                str(output),
            ]
            labels.write_text("\n", encoding="utf-8")
            self.assertEqual(main(args), 0)
            initial = json.loads(output.read_bytes())
            self.assertEqual(initial["overall"]["total"], 0)
            self.assertIsNone(initial["overall"]["coverage"])
            self.assertEqual(initial["anytime_pvalue"], 1.0)
            self.assertEqual(
                initial["stream"]["first_unlabeled"], {"id": "a", "row": 1}
            )

            arriving = (
                '{"id":"c","output_sha256":"key-c","correct":false}\n'
                '{"id":"a","output_sha256":"key-a","correct":true}\n'
            )
            labels.write_text(arriving, encoding="utf-8")
            self.assertEqual(main(args), 0)
            partial = json.loads(output.read_bytes())
            self.assertEqual(partial["overall"]["total"], 2)
            self.assertEqual(partial["overall"]["accepted"], 1)
            self.assertEqual(
                partial["stream"],
                {
                    "total": 4,
                    "accepted": 3,
                    "coverage": 0.75,
                    "unlabeled_accepted": 1,
                    "first_unlabeled": {"id": "b", "row": 3},
                },
            )
            self.assertIsNone(partial["first_alert"])

            complete = arriving + '{"id":"b","output_sha256":"key-b","correct":true}\n'
            labels.write_text(complete, encoding="utf-8")
            self.assertEqual(main(args), 1)
            previous = output.read_bytes()
            final = json.loads(previous)
            self.assertEqual(
                final["first_alert"], {"id": "c", "row": 4, "accepted": 3, "errors": 1}
            )
            self.assertEqual(final["overall"]["coverage"], 0.75)
            self.assertIsNone(final["stream"]["first_unlabeled"])

            labels.write_text(complete.replace("key-c", "stale"), encoding="utf-8")
            with (
                redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit) as rejected,
            ):
                main(args)
            self.assertEqual(rejected.exception.code, 2)
            self.assertIn("human: output key mismatch for 'c'", errors.getvalue())
            self.assertEqual(output.read_bytes(), previous)

    def test_monitor_reads_routed_order_and_exact_grades_at_saved_cutoff(self):
        with TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            fit, predictions, cutoff, decisions, grades, output = (
                root / name
                for name in (
                    "fit.csv",
                    "new.csv",
                    "cutoff.json",
                    "decisions.csv",
                    "grades.csv",
                    "monitor.json",
                )
            )
            fit.write_text(
                "answer_id,entropy,key,correct\nfit,0.2,fit-key,1\n", encoding="utf-8"
            )
            predictions.write_text(
                "answer_id,entropy,key\nmissing,,k1\nfirst,0.1,k2\nreview,0.8,k3\nwaiting,0.2,k4\nlast,0.15,k5\n",
                encoding="utf-8",
            )
            self.assertEqual(
                main(
                    [
                        "select",
                        str(fit),
                        "--id-column",
                        "answer_id",
                        "--score-column",
                        "entropy",
                        "--output-key-column",
                        "key",
                        "--track-fit-outputs",
                        "--correct-column",
                        "correct",
                        "--direction",
                        "lower",
                        "--risk-target",
                        "0",
                        "--rule",
                        "empirical",
                        "--thresholds",
                        "0.2",
                        "--output",
                        str(cutoff),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "route",
                        str(predictions),
                        "--threshold-report",
                        str(cutoff),
                        "--review-missing-scores",
                        "--output",
                        str(decisions),
                    ]
                ),
                0,
            )
            saved = decisions.read_bytes()
            partial = "answer_id,key,correct\nlast,k5,0\nwaiting,old,0\nfirst,k2,1\n"
            grades.write_text(partial, encoding="utf-8")
            args = [
                "monitor",
                "--batch-decisions",
                str(decisions),
                "--threshold-report",
                str(cutoff),
                "--grader-file",
                f"human={grades}",
                "--allow-extra-grades",
                "--output",
                str(output),
            ]
            self.assertEqual(main(args), 0)
            waiting = json.loads(output.read_bytes())
            self.assertEqual(waiting["threshold"], -0.2)
            self.assertTrue(waiting["input"]["fit_outputs_checked"])
            self.assertEqual(waiting["input"]["grader_id_column"], "answer_id")
            self.assertEqual(waiting["input"]["grader_output_key_column"], "key")
            self.assertEqual(waiting["overall"]["total"], 3)
            self.assertEqual(waiting["overall"]["accepted"], 1)
            self.assertEqual(
                waiting["stream"],
                {
                    "total": 5,
                    "accepted": 3,
                    "coverage": 0.6,
                    "unlabeled_accepted": 1,
                    "first_unlabeled": {"id": "waiting", "row": 4},
                    "missing_scores": 1,
                },
            )
            self.assertIsNone(waiting["first_alert"])

            grades.write_text(partial + "waiting,k4,1\n", encoding="utf-8")
            self.assertEqual(main(args), 1)
            previous = output.read_bytes()
            complete = json.loads(previous)
            self.assertEqual(
                complete["first_alert"],
                {"id": "last", "row": 5, "accepted": 3, "errors": 1},
            )
            self.assertIsNone(complete["stream"]["first_unlabeled"])

            decisions.write_bytes(
                saved.replace(b"last,-0.15,accept", b"last,-0.6,accept")
            )
            with (
                redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit) as rejected,
            ):
                main(args)
            self.assertEqual(rejected.exception.code, 2)
            self.assertIn("decisions do not match the cutoff", errors.getvalue())
            self.assertEqual(output.read_bytes(), previous)

    def test_monitor_full_grader_export_waits_for_exact_output(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            predictions, labels, output = (
                root / name
                for name in ("predictions.csv", "grades.jsonl", "monitor.json")
            )
            predictions.write_text(
                "id,score,output_sha256,partition\n"
                "a,0.9,old-a,prior\n"
                "a,0.9,new-a,live\n"
                "b,0.9,key-b,live\n",
                encoding="utf-8",
            )
            labels.write_text(
                '{"grade_id":"b","graded_key":"key-b","correct":false}\n'
                '{"grade_id":"a","graded_key":"old-a","correct":false}\n'
                '{"grade_id":"outside","graded_key":"other","correct":true}\n',
                encoding="utf-8",
            )
            args = [
                "monitor",
                str(predictions),
                "--prediction-filter",
                "partition=live",
                "--threshold",
                "0.8",
                "--risk-target",
                "0",
                "--grader-file",
                f"human={labels}",
                "--grader-id-column",
                "grade_id",
                "--grader-output-key-column",
                "graded_key",
                "--allow-extra-grades",
                "--output",
                str(output),
            ]
            self.assertEqual(main(args), 0)
            partial = json.loads(output.read_bytes())
            self.assertEqual(partial["input"]["grader_id_column"], "grade_id")
            self.assertEqual(partial["overall"]["total"], 0)
            self.assertEqual(partial["stream"]["total"], 2)
            self.assertEqual(partial["stream"]["unlabeled_accepted"], 1)
            self.assertEqual(
                partial["stream"]["first_unlabeled"], {"id": "a", "row": 1}
            )
            self.assertIsNone(partial["first_alert"])

            with labels.open("a", encoding="utf-8") as target:
                target.write('{"grade_id":"a","graded_key":"new-a","correct":true}\n')
            self.assertEqual(main(args), 1)
            complete = json.loads(output.read_bytes())
            self.assertEqual(complete["overall"]["accepted"], 2)
            self.assertEqual(complete["overall"]["errors"], 1)
            self.assertEqual(
                complete["first_alert"],
                {"id": "b", "row": 2, "accepted": 2, "errors": 1},
            )
            self.assertIsNone(complete["stream"]["first_unlabeled"])
