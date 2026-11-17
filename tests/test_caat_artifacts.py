import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from caat.cli import main


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
PREVIOUS_CSV = b"old,output\r\n"


class ArtifactTests(unittest.TestCase):
    def assert_output_error(self, *args):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as stopped:
                main([str(arg) for arg in args])
        self.assertEqual(stopped.exception.code, 2, stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        return stderr.getvalue()

    def test_unwritable_report_leaves_companion_outputs_intact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            blocked = root / "blocked"
            blocked.touch()
            cases = {
                "audit": (
                    root / "disagreements.csv",
                    [
                        EXAMPLES / "audit/predictions.csv",
                        "--grader-file",
                        f"reference={EXAMPLES / 'audit/reference.csv'}",
                        "--grader-file",
                        f"rubric={EXAMPLES / 'audit/rubric.csv'}",
                        "--output-key-column",
                        "output_key",
                        "--threshold",
                        "0.75",
                        "--risk-target",
                        "0.7",
                        "--disagreements-output",
                        root / "disagreements.csv",
                    ],
                ),
                "policy": (
                    root / "holdout.csv",
                    [
                        EXAMPLES / "ifeval_graders.csv",
                        "--score-column",
                        "mean_token_logprob",
                        "--correct-column",
                        "strict_pass",
                        "--check-column",
                        "helm_strict_pass",
                        "--risk-target",
                        "0.4",
                        "--holdout-output",
                        root / "holdout.csv",
                    ],
                ),
                "evaluate": (
                    root / "risk_coverage.csv",
                    [
                        EXAMPLES / "ifeval_strict.csv",
                        "--score-column",
                        "mean_token_logprob",
                        "--correct-column",
                        "strict_pass",
                        "--threshold",
                        "-0.35",
                        "--output-dir",
                        root,
                    ],
                ),
            }
            for command, (companion, args) in cases.items():
                with self.subTest(command=command):
                    companion.write_bytes(PREVIOUS_CSV)
                    self.assert_output_error(
                        command, *args, "--output", blocked / "report.json"
                    )
                    self.assertEqual(companion.read_bytes(), PREVIOUS_CSV)
                    self.assertFalse((root / "metrics.json").exists())
                    self.assertEqual(list(root.glob(".caat-*")), [])

    def test_queue_preserves_carriage_returns_in_csv(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "rows.jsonl"
            row = {
                "id": "sample\r1",
                "score": 0.9,
                "group": "group\r1",
                "output_key": 'a\rb\nc,"d"',
            }
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            ranked, sample = root / "ranked.csv", root / "sample.csv"
            args = [
                "queue",
                source,
                "--budget",
                "1",
                "--output-key-column",
                "output_key",
                "--output",
                ranked,
                "--random-output",
                sample,
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(arg) for arg in args]), 0)

            expected = {**row, "score": "0.9"}
            for path in (ranked, sample):
                with path.open(encoding="utf-8", newline="") as stream:
                    self.assertEqual(list(csv.DictReader(stream)), [expected])

    def test_queue_rolls_back_after_a_late_replacement_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            ranked, sample = root / "ranked.csv", root / "sample.csv"
            ranked.write_bytes(PREVIOUS_CSV)
            sample.write_bytes(PREVIOUS_CSV)
            replace = Path.replace

            def replace_output(path, target):
                if target == sample:
                    self.assertNotEqual(ranked.read_bytes(), PREVIOUS_CSV)
                    raise PermissionError("cannot replace sample.csv")
                return replace(path, target)

            with patch.object(Path, "replace", replace_output):
                error = self.assert_output_error(
                    "queue",
                    EXAMPLES / "audit/predictions.csv",
                    "--budget",
                    "3",
                    "--output",
                    ranked,
                    "--random-output",
                    sample,
                )

            self.assertIn("cannot replace sample.csv", error)
            self.assertEqual(ranked.read_bytes(), PREVIOUS_CSV)
            self.assertEqual(sample.read_bytes(), PREVIOUS_CSV)
            self.assertEqual(list(root.glob(".caat-*")), [])

    @unittest.skipUnless(
        find_spec("tomllib") or find_spec("tomli"), "Requires TOML support"
    )
    def test_workflow_rolls_back_after_a_late_replacement_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = root / "caat.toml"
            config.write_text(
                'predictions="rows.csv"\nscore_columns=["score"]\n'
                'correct_columns=["correct"]\nrisk_target=0.4\nrule="empirical"\n'
                'json_output="report.json"\nhtml="report.html"\n'
                'holdout_output="holdout.csv"\n',
                encoding="utf-8",
            )
            (root / "rows.csv").write_text(
                "id,score,correct\n" + "".join(f"{i},0.9,1\n" for i in range(20)),
                encoding="utf-8",
            )
            report, html, csv = (
                root / name for name in ("report.json", "report.html", "holdout.csv")
            )
            report.write_bytes(b"previous JSON")
            html.write_bytes(b"previous HTML")
            replace = Path.replace

            def replace_output(path, target):
                if target == report:
                    self.assertTrue(csv.is_file())
                    self.assertNotEqual(html.read_bytes(), b"previous HTML")
                    raise PermissionError("cannot replace report.json")
                return replace(path, target)

            with patch.object(Path, "replace", replace_output):
                error = self.assert_output_error("run", config)

            self.assertIn("cannot replace report.json", error)
            self.assertEqual(report.read_bytes(), b"previous JSON")
            self.assertEqual(html.read_bytes(), b"previous HTML")
            self.assertFalse(csv.exists())
            self.assertEqual(list(root.glob(".caat-*")), [])
