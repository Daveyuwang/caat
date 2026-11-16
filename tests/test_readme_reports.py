import json
import math
import re
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
    result = subprocess.run([*CLI, command, *arguments], cwd=ROOT, capture_output=True, encoding="utf-8")
    expected = int(command == "audit")  # The alternate grader misses the example's target.
    if result.returncode != expected:
        raise ValueError(f"{command} returned {result.returncode}, expected {expected}: {result.stderr}")
    return result.stdout


def normalize_paths(report):
    # Normalize workspace paths for fixtures shared by Windows and Linux.
    def relative(value):
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(ROOT):
            path = path.relative_to(ROOT)
        return path.as_posix()

    inputs = report["input"]
    for name in ("path", "design_path"):
        if inputs.get(name):
            inputs[name] = relative(inputs[name])
    if "grader_files" in inputs:
        inputs["grader_files"] = {name: relative(path) for name, path in inputs["grader_files"].items()}
    return report


class ReadmeReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = readme_commands()

    def assert_golden_report(self, actual, filename):
        actual = normalize_paths(actual)
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
        report = json.loads(run_readme("audit", [*self.examples["audit"], "--json"]))
        self.assert_golden_report(report, "audit.json")

    def test_select_example(self):
        self.assert_golden_report(json.loads(run_readme("select", [*self.examples["select"], "--json"])), "select.json")

    def test_policy_example(self):
        arguments = self.examples["policy"]
        self.assertEqual(run_readme("policy", arguments), (GOLDEN / "policy.txt").read_text(encoding="utf-8"))
        self.assert_golden_report(json.loads(run_readme("policy", [*arguments, "--json"])), "policy.json")

    def test_output_file_for_each_command(self):
        with TemporaryDirectory() as directory:
            for command, args in self.examples.items():
                with self.subTest(command=command):
                    args = args.copy()
                    path = Path(directory) / f"{command}.json"
                    if command == "evaluate":
                        args[args.index("--output-dir") + 1] = directory
                    output = run_readme(command, [*args, "--output", str(path)])
                    self.assertFalse(output.lstrip().startswith("{"))
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(payload["command"], command)
                    self.assertEqual(payload["schema_version"], 2)

    def test_documented_fields_exist_in_real_reports(self):
        text = (ROOT / "docs" / "guide" / "reports.md").read_text(encoding="utf-8")
        fields = {
            name.lower(): re.findall(r"^\| `([^\x60]+)` \|", section, flags=re.MULTILINE)
            for name, section in re.findall(r"^## (\w+)\n(.*?)(?=^## |\Z)", text, flags=re.MULTILINE | re.DOTALL)
        }
        self.assertEqual(set(fields), {"common", *self.examples})
        with TemporaryDirectory() as directory:
            for command, args in self.examples.items():
                args = args.copy()
                if command == "evaluate":
                    del args[args.index("--output-dir"):args.index("--output-dir") + 2]
                if command == "audit":
                    predictions = Path(directory) / "predictions.csv"
                    header, *rows = (ROOT / args[0]).read_text(encoding="utf-8").splitlines()
                    predictions.write_text(header + ",batch\n" + "\n".join(row + ",current" for row in rows),
                                           encoding="utf-8")
                    args[0] = str(predictions)
                    args.extend(("--prediction-filter", "batch=current", "--bootstrap", "5"))
                report = json.loads(run_readme(command, [*args, "--json"]))
                for field in fields["common"] + fields[command]:
                    with self.subTest(command=command, field=field):
                        values = [report]
                        for key in field.replace("[]", ".[]").split("."):
                            if key == "[]":
                                values = [entry for value in values for entry in value]
                            elif key == "*":
                                values = [entry for value in values for entry in value.values()]
                            else:
                                values = [value[key] for value in values]
                            self.assertTrue(values)
