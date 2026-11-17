import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from math import nextafter
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.cli import main


class RouteTests(unittest.TestCase):
    def test_missing_scores_stay_in_batch_and_outside_the_label_sample(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            decisions = root / "decisions.csv"
            sample = root / "sample.csv"
            records = [("missing", None), ("pass", 0.8), ("below", 0.2), ("above", 0.9)]
            predictions.write_text(
                "\n".join(
                    json.dumps({"id": item, "score": score, "key": f"key-{item}"})
                    for item, score in records
                ),
                encoding="utf-8",
            )
            route = [
                "route",
                str(predictions),
                "--threshold",
                "0.8",
                "--output-key-column",
                "key",
                "--output",
                str(decisions),
                "--json",
            ]
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main([*route, "--review-missing-scores"]), 0)
            self.assertEqual(
                json.loads(stdout.getvalue())["overall"],
                {
                    "total": 4,
                    "accepted": 2,
                    "review": 2,
                    "coverage": 0.5,
                    "missing_scores": 1,
                },
            )
            with decisions.open(newline="", encoding="utf-8") as source:
                routed = list(csv.DictReader(source))
            self.assertEqual(
                [
                    (row["id"], row["score"], row["decision"], row["output_key"])
                    for row in routed
                ],
                [
                    ("missing", "", "review", "key-missing"),
                    ("pass", "0.8", "accept", "key-pass"),
                    ("below", "0.2", "review", "key-below"),
                    ("above", "0.9", "accept", "key-above"),
                ],
            )
            plan = [
                "plan",
                "--risk-target",
                "0.4",
                "--expected-error",
                "0",
                "--batch-decisions",
                str(decisions),
                "--sample-output",
                str(sample),
                "--json",
            ]
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(plan), 0)
            planned = json.loads(stdout.getvalue())
            self.assertEqual(planned["accepted_population"], 2)
            with sample.open(newline="", encoding="utf-8") as source:
                drawn = list(csv.DictReader(source))
            self.assertEqual(len(drawn), planned["accepted"])
            self.assertTrue({row["id"] for row in drawn} <= {"above", "pass"})

            previous = decisions.read_bytes()
            with (
                redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit) as rejected,
            ):
                main(route)
            self.assertEqual(rejected.exception.code, 2)
            self.assertIn("invalid score at input line 1", errors.getvalue())
            self.assertEqual(decisions.read_bytes(), previous)

            predictions.write_text(
                predictions.read_text(encoding="utf-8").replace("0.9", '"nan"'),
                encoding="utf-8",
            )
            with (
                redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit) as rejected,
            ):
                main([*route, "--review-missing-scores"])
            self.assertEqual(rejected.exception.code, 2)
            self.assertIn("score for id 'above' must be finite", errors.getvalue())
            self.assertEqual(decisions.read_bytes(), previous)

            decisions.write_bytes(
                previous.replace(b"missing,,review,", b"missing,,accept,")
            )
            with (
                redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit) as rejected,
            ):
                main(plan)
            self.assertEqual(rejected.exception.code, 2)
            self.assertIn(
                "accepted prediction 'missing' needs a score", errors.getvalue()
            )

    def test_filter_selects_a_tier_before_reading_scores(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, decisions = root / "mixed.csv", root / "decisions.csv"
            source.write_text(
                "id,tier,score,key,correct\n"
                "a,helper,unscored,old-a,unknown\n"
                "a,encoder,0.9,new-a,0\n"
                "b,encoder,0.7,new-b,1\n",
                encoding="utf-8",
            )
            common = [
                str(source),
                "--threshold",
                "0.8",
                "--output-key-column",
                "key",
                "--prediction-filter",
                "tier=encoder",
                "--json",
            ]
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(
                    main(["route", *common, "--output", str(decisions)]), 0
                )
            routing = json.loads(stdout.getvalue())
            self.assertEqual(
                routing["input"]["prediction_filter"],
                {"column": "tier", "value": "encoder"},
            )
            self.assertEqual(routing["overall"]["total"], 2)
            self.assertEqual(routing["overall"]["accepted"], 1)
            with decisions.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(
                    [
                        (row["id"], row["decision"], row["output_key"])
                        for row in csv.DictReader(handle)
                    ],
                    [("a", "accept", "new-a"), ("b", "review", "new-b")],
                )

            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(
                    main(["evaluate", *common, "--correct-column", "correct"]), 0
                )
            evaluation = json.loads(stdout.getvalue())
            self.assertEqual(
                evaluation["input"]["prediction_filter"],
                routing["input"]["prediction_filter"],
            )
            self.assertEqual(evaluation["overall"]["total"], 2)
            self.assertEqual(evaluation["overall"]["errors"], 1)

    def test_unlabeled_jsonl_keeps_order_ties_and_output_keys(self):
        with (
            TemporaryDirectory() as directory,
            redirect_stdout(io.StringIO()) as stdout,
        ):
            root = Path(directory)
            source = root / "predictions.jsonl"
            output = root / "decisions.csv"
            source.write_text(
                '{"answer_id":"02","confidence":0.9,"group":"g","key":"out-2"}\n'
                '{"answer_id":"01","confidence":0.8,"group":null,"key":"out-1"}\n',
                encoding="utf-8",
            )
            args = [
                "route",
                str(source),
                "--threshold",
                "0.9",
                "--id-column",
                "answer_id",
                "--score-column",
                "confidence",
                "--output-key-column",
                "key",
                "--output",
                str(output),
                "--json",
            ]
            self.assertEqual(main(args), 0)
            self.assertEqual(
                json.loads(stdout.getvalue())["overall"],
                {
                    "total": 2,
                    "accepted": 1,
                    "review": 1,
                    "coverage": 0.5,
                },
            )
            with output.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(
                    list(csv.reader(handle)),
                    [
                        ["id", "score", "decision", "group", "output_key"],
                        ["02", "0.9", "accept", "g", "out-2"],
                        ["01", "0.8", "review", "", "out-1"],
                    ],
                )
            original = output.read_bytes()
            source.write_bytes(source.read_bytes() * 2)
            with (
                redirect_stderr(io.StringIO()) as err,
                self.assertRaises(SystemExit) as raised,
            ):
                main(args)
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("duplicate id", err.getvalue())
            self.assertEqual(output.read_bytes(), original)

    def test_route_reuses_selection_and_refuses_unconfirmed_policy(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            calibration, source, cutoff, output = (
                root / name
                for name in (
                    "calibration.csv",
                    "predictions.csv",
                    "cutoff.json",
                    "decisions.csv",
                )
            )
            calibration.write_text(
                "key,quality,answer_hash,correct\n"
                + "".join(f"fit-{i},0.9,hash-{i},1\n" for i in range(5)),
                encoding="utf-8",
            )
            source.write_text("key,quality\nnew,0.9\n", encoding="utf-8")
            fit = [
                str(calibration),
                "--id-column",
                "key",
                "--score-column",
                "quality",
                "--output-key-column",
                "answer_hash",
                "--correct-column",
                "correct",
                "--risk-target",
                "0.4",
                "--rule",
                "empirical",
                "--output",
                str(cutoff),
            ]
            args = [
                "route",
                str(source),
                "--threshold-report",
                str(cutoff),
                "--output",
                str(output),
            ]
            self.assertEqual(main(["select", *fit, "--thresholds", "0.9"]), 0)
            self.assertEqual(main(args), 0)
            with output.open(encoding="utf-8", newline="") as handle:
                decision = next(csv.DictReader(handle))
            self.assertEqual(decision["decision"], "accept")
            self.assertNotIn("output_key", decision)

            source.write_text(
                "key,quality,answer_hash,correct\nnew,0.9,new-hash,1\n",
                encoding="utf-8",
            )
            check = [
                "evaluate",
                str(source),
                "--threshold-report",
                str(cutoff),
                "--correct-column",
                "correct",
                "--holdout-confidence",
                "0.95",
                "--batch-decisions",
                str(output),
            ]
            self.assertEqual(main(check), 0)
            self.assertEqual(main(args), 0)
            with output.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(next(csv.DictReader(handle))["output_key"], "new-hash")
            self.assertEqual(main(check), 0)

            self.assertEqual(main(["policy", *fit, "--holdout-confidence", "0.95"]), 1)
            self.assertEqual(
                json.loads(cutoff.read_bytes())["selection"]["selected_threshold"], 0.9
            )
            original = output.read_bytes()
            with (
                redirect_stderr(io.StringIO()) as err,
                self.assertRaises(SystemExit) as raised,
            ):
                main(args)
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("routing requires a confirmed policy report", err.getvalue())
            self.assertEqual(output.read_bytes(), original)

    def test_saved_output_key_aligns_separate_grader_exports(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            cutoff, predictions, first, second = (
                root / name
                for name in ("cutoff.json", "new.csv", "first.csv", "second.csv")
            )
            cutoff.write_text(
                json.dumps(
                    {
                        "command": "select",
                        "input": {
                            "id_column": "id",
                            "score_column": "score",
                            "output_key_column": "answer_hash",
                        },
                        "selected_threshold": 0.8,
                        "risk_target": 0.4,
                    }
                ),
                encoding="utf-8",
            )
            predictions.write_text(
                "id,score,answer_hash\na,0.9,hash-a\nb,0.8,hash-b\n", encoding="utf-8"
            )
            first.write_text(
                "id,answer_hash,correct\na,hash-a,1\nb,hash-b,1\n", encoding="utf-8"
            )
            second.write_text(
                "id,answer_hash,correct\na,hash-a,1\nb,hash-b,0\n", encoding="utf-8"
            )
            common = [
                str(predictions),
                "--threshold-report",
                str(cutoff),
                "--grader-file",
                f"primary={first}",
                "--json",
            ]
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(
                    main(["audit", *common, "--grader-file", f"alternate={second}"]), 1
                )
            audit = json.loads(stdout.getvalue())
            self.assertEqual(audit["input"]["output_key_column"], "answer_hash")
            self.assertEqual(audit["graders"]["alternate"]["errors"], 1)

            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(["monitor", *common]), 0)
            monitor = json.loads(stdout.getvalue())
            self.assertEqual(monitor["input"]["output_key_column"], "answer_hash")
            self.assertEqual(monitor["overall"]["accepted"], 2)

    def test_fixed_batch_uses_the_route_count_and_checks_sample_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, decisions, sample, audit = (
                root / name
                for name in ("rows.csv", "decisions.csv", "sample.csv", "audit.json")
            )
            header = "id,score,output_key,primary,alternate\n"
            labels = "a,0.9,key-a,1,1\nb,0.8,key-b,1,0\n"
            predictions.write_text(
                header + labels + "c,0.76,key-c,0,0\nd,0.1,key-d,1,1\n",
                encoding="utf-8",
            )
            sample.write_text(header + labels, encoding="utf-8")
            route = [
                "route",
                str(predictions),
                "--threshold",
                "0.75",
                "--output-key-column",
                "output_key",
                "--output",
                str(decisions),
            ]
            batch = ["--batch-decisions", str(decisions)]
            check = [
                str(sample),
                "--threshold",
                "0.75",
                "--risk-target",
                "0.6",
                "--holdout-confidence",
                "0.95",
                "--output-key-column",
                "output_key",
                "--correct-column",
                "primary",
            ]
            args = [
                "audit",
                *check,
                *batch,
                "--correct-column",
                "alternate",
                "--output",
                str(audit),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(route), 0)
                self.assertEqual(main(args), 1)
            report = json.loads(audit.read_bytes())
            self.assertEqual(report["input"]["batch_decisions"], str(decisions))
            self.assertNotIn("sample_seed", report["input"])
            self.assertEqual(report["accepted_population"], 3)
            self.assertEqual(
                report["graders"]["alternate"]["risk_upper_bound"],
                nextafter(2 / 3, 1.0),
            )

            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(
                    main(["evaluate", *check, *batch, "--sample-seed", "42", "--json"]),
                    0,
                )
            evaluation = json.loads(stdout.getvalue())
            self.assertEqual(evaluation["overall"]["risk_upper_errors"], 1)
            self.assertEqual(evaluation["input"]["sample_seed"], 42)
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(
                    main(
                        [
                            "plan",
                            "--accepted",
                            "2",
                            "--expected-error",
                            "0",
                            *batch,
                            "--json",
                        ]
                    ),
                    0,
                )
            plan = json.loads(stdout.getvalue())
            self.assertEqual(plan["accepted_population"], 3)
            self.assertEqual(plan["input"]["batch_decisions"], str(decisions))

            original = audit.read_bytes()
            with (
                redirect_stderr(io.StringIO()) as err,
                self.assertRaises(SystemExit) as stopped,
            ):
                main([*args, "--sample-seed", "0"])
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("sample does not match the draw for seed 0", err.getvalue())
            self.assertEqual(audit.read_bytes(), original)

            with (
                redirect_stderr(io.StringIO()) as err,
                self.assertRaises(SystemExit) as stopped,
            ):
                main(["evaluate", *check, "--sample-seed", "42"])
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("--sample-seed requires --batch-decisions", err.getvalue())

            for changed, message in (
                (
                    labels.replace("a,0.9", "other,0.9"),
                    "missing from the batch decisions",
                ),
                (labels.replace("a,0.9", "a,0.91"), "score mismatch"),
                (labels.replace("key-a", "stale"), "output key mismatch"),
            ):
                with self.subTest(message=message):
                    sample.write_text(header + changed, encoding="utf-8")
                    with (
                        redirect_stderr(io.StringIO()) as err,
                        self.assertRaises(SystemExit) as stopped,
                    ):
                        main(args)
                    self.assertEqual(stopped.exception.code, 2)
                    self.assertIn(message, err.getvalue())
                    self.assertEqual(audit.read_bytes(), original)

            sample.write_text(header + labels, encoding="utf-8")
            moved_cutoff = ["0.8" if value == "0.75" else value for value in args]
            with (
                redirect_stderr(io.StringIO()) as err,
                self.assertRaises(SystemExit) as stopped,
            ):
                main(moved_cutoff)
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("decisions do not match the cutoff", err.getvalue())
            self.assertEqual(audit.read_bytes(), original)
