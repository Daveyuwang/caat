import json
import math
import shlex
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
CLI = [sys.executable, "-c", "from caat.cli import main; raise SystemExit(main())"]


def readme_commands():
    examples = {}
    for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("caat "):
            _, command, *arguments = shlex.split(line)
            if command in examples:
                raise ValueError(f"Multiple README examples for {command}")
            examples[command] = arguments
    return examples


def run_readme(command, arguments):
    return subprocess.run(
        [*CLI, command, *arguments], cwd=ROOT, check=True, capture_output=True, encoding="utf-8",
    ).stdout


def normalize_audit_paths(report):
    # The CLI reports the native file names passed to it.
    inputs = report["input"]
    inputs["path"] = Path(inputs["path"]).as_posix()
    inputs["grader_files"] = {name: Path(path).as_posix() for name, path in inputs["grader_files"].items()}
    return report


class ReadmeReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = readme_commands()

    def assert_golden_report(self, actual, filename):
        frozen = json.loads((GOLDEN / filename).read_text(encoding="utf-8"))
        if "selection" in actual:
            selection, reference = actual["selection"], frozen["selection"]
        else:
            selection, reference = actual, frozen
        if "candidates" in selection:
            for candidate, expected in zip(selection["candidates"], reference["candidates"], strict=True):
                # lgamma can round differently on different Python platforms.
                self.assertIs(type(candidate["pvalue"]), float)
                self.assertTrue(math.isclose(candidate["pvalue"], expected["pvalue"], rel_tol=1e-11, abs_tol=1e-15))
                candidate["pvalue"] = expected["pvalue"]
        self.assertEqual(json.dumps(actual, sort_keys=True, indent=2), json.dumps(frozen, sort_keys=True, indent=2))

    def test_readme_examples_have_golden_reports(self):
        self.assertEqual(set(self.examples), {"evaluate", "policy", "audit", "select"})

    def test_evaluate_example(self):
        with TemporaryDirectory() as directory:
            args = self.examples["evaluate"].copy()
            args[args.index("--output-dir") + 1] = directory
            run_readme("evaluate", args)
            output = Path(directory)
            report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assert_golden_report(report, "evaluate/metrics.json")
            self.assertEqual(
                (output / "risk_coverage.csv").read_text(encoding="utf-8"),
                (GOLDEN / "evaluate" / "risk_coverage.csv").read_text(encoding="utf-8"),
            )

    def test_audit_example(self):
        report = json.loads(run_readme("audit", self.examples["audit"]))
        self.assert_golden_report(normalize_audit_paths(report), "audit.json")

    def test_select_example(self):
        self.assert_golden_report(json.loads(run_readme("select", self.examples["select"])), "select.json")

    def test_policy_example(self):
        arguments = self.examples["policy"]
        self.assertEqual(run_readme("policy", arguments), (GOLDEN / "policy.txt").read_text(encoding="utf-8"))
        self.assert_golden_report(json.loads(run_readme("policy", [*arguments, "--json"])), "policy.json")
