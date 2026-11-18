import csv
import io
import json
import unittest
from contextlib import redirect_stdout
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.cli import main
from caat.helm_log import export_run


def request(item_id, grade, *, trial=0, logprob=-0.25):
    completion = {"text": "4\n", "tokens": [{"text": "4\n", "logprob": logprob}]}
    return {
        "instance": {"id": item_id, "split": "test"},
        "train_trial_index": trial,
        "result": {"success": True, "completions": [completion]},
        "annotations": {"judge": {"passed": grade}},
    }


def stat(item_id, score, *, trial=0, name="accuracy"):
    return {
        "instance_id": item_id,
        "train_trial_index": trial,
        "stats": [{"name": {"name": name, "split": "test"}, "sum": score, "count": 1}],
    }


def save(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


class HelmTests(unittest.TestCase):
    def test_export_aligns_stats_and_annotations_for_audit(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            output = run / "grades.csv"
            save(
                run / "scenario_state.json",
                {"request_states": [request("first", False), request("second", True)]},
            )
            save(
                run / "per_instance_stats.json",
                [stat("second", 0.5), stat("first", 0, name="other"), stat("first", 1)],
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "helm",
                            str(run),
                            "--metric",
                            "accuracy",
                            "--annotation",
                            "judge.passed",
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
            with output.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(
                [(row["id"], row["accuracy"], row["judge.passed"]) for row in rows],
                [("first", "1", "0"), ("second", "0", "1")],
            )
            for row in rows:
                self.assertEqual(float(row["mean_token_logprob"]), -0.25)
                self.assertEqual(row["output_sha256"], sha256(b"4\n").hexdigest())
            report = run / "audit.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "audit",
                            str(output),
                            "--score-column",
                            "mean_token_logprob",
                            "--correct-column",
                            "accuracy",
                            "--correct-column",
                            "judge.passed",
                            "--threshold",
                            "-0.25",
                            "--risk-target",
                            "0.75",
                            "--output-key-column",
                            "output_sha256",
                            "--output",
                            str(report),
                        ]
                    ),
                    0,
                )
            graders = json.loads(report.read_text(encoding="utf-8"))["graders"]
            self.assertEqual(
                {name: values["errors"] for name, values in graders.items()},
                {"accuracy": 1, "judge.passed": 1},
            )

    def test_rejects_incompatible_grades_and_preserves_the_output(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            output = run / "grades.csv"
            output.write_bytes(b"preserved\n")
            save(run / "scenario_state.json", {"request_states": [request("a", True)]})
            cases = (
                ([stat("a", 1, trial=1)], "missing metric"),
                ([stat("a", 1), stat("b", 0)], "unexpected metric"),
                ([stat("a", 1), stat("a", 1)], "duplicate metric"),
                ([stat("a", 3)], "needs one score"),
            )
            for records, error in cases:
                with self.subTest(error=error):
                    save(run / "per_instance_stats.json", records)
                    with self.assertRaisesRegex(ValueError, error):
                        export_run(run, output, metrics=("accuracy",))
                    self.assertEqual(output.read_bytes(), b"preserved\n")

    def test_annotation_only_rejects_incomplete_records(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            state = run / "scenario_state.json"
            output = run / "grades.csv"
            opts = {"annotations": ("judge.passed",)}
            save(state, {"request_states": [request("first", True)]})
            self.assertEqual(export_run(run, output, **opts), 1)
            before = output.read_bytes()
            cases = (
                (
                    [request("first", True), request("first", False, trial=1)],
                    "duplicate HELM instance",
                ),
                ([request("first", None)], "needs a boolean annotation"),
                (
                    [request("first", True, logprob=0)],
                    "all-zero token log probabilities",
                ),
            )
            for records, error in cases:
                with self.subTest(error=error):
                    save(state, {"request_states": records})
                    with self.assertRaisesRegex(ValueError, error):
                        export_run(run, output, **opts)
                    self.assertEqual(output.read_bytes(), before)
