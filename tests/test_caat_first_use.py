"""Checks for the first-use fixes: label matching, whole-file counts, and routing columns."""

import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.cli import main


def run(*args):
    """Return the exit code, stdout, and stderr of one CLI call."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = main([str(arg) for arg in args])
        except SystemExit as stop:
            code = stop.code
    return code, out.getvalue(), err.getvalue()


class FirstUseTests(unittest.TestCase):
    def setUp(self):
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name).resolve()

    def table(self, name="graded.csv", *, reversed_score=False, group=False):
        # Even rows are right ("cat" == "cat") and carry the high score.
        path = self.root / name
        header = "id,score,guess,truth,ok" + (",group" if group else "")
        lines = [header]
        for i in range(100):
            right = i % 2 == 0
            score = 0.9 if right != reversed_score else 0.1
            line = f"{i:03d},{score},cat,{'cat' if right else 'dog'},{int(right)}"
            lines.append(line + (f",{'a' if i < 50 else 'b'}" if group else ""))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_label_matching_equals_a_correctness_column(self):
        table = self.table()
        labels = ["--prediction-column", "guess", "--reference-column", "truth"]
        for command in (
            ["policy", table, "--risk-target", "0.3"],
            ["select", table, "--thresholds", "0.9", "0.1", "--risk-target", "0.3"],
        ):
            with self.subTest(command=command[0]):
                code, matched, _ = run(*command, *labels, "--json")
                self.assertEqual(code, 0)
                _, graded, _ = run(*command, "--correct-column", "ok", "--json")
                matched, graded = json.loads(matched), json.loads(graded)
                self.assertEqual(matched["input"]["prediction_column"], "guess")
                self.assertEqual(matched["input"]["reference_column"], "truth")
                self.assertNotIn("prediction_column", graded["input"])
                # The reference column names the grader; the numbers are identical.
                text = json.dumps({k: v for k, v in matched.items() if k != "input"})
                self.assertEqual(
                    json.loads(text.replace('"truth"', '"ok"')),
                    {k: v for k, v in graded.items() if k != "input"},
                )

        code, matched, _ = run(
            "scores", table, "--score-column", "score", *labels, "--bootstrap", "0",
            "--split", "all", "--json",
        )  # fmt: skip
        self.assertEqual(code, 0)
        _, graded, _ = run(
            "scores", table, "--score-column", "score", "--correct-column", "ok",
            "--bootstrap", "0", "--split", "all", "--json",
        )  # fmt: skip
        matched, graded = json.loads(matched), json.loads(graded)
        self.assertEqual(matched["graders"]["truth"], graded["graders"]["ok"])

    def test_label_matching_rejects_ambiguous_or_missing_grades(self):
        table = self.table()
        base = ["policy", table, "--risk-target", "0.3"]
        code, _, error = run(
            *base, "--check-column", "ok", "--prediction-column", "guess"
        )
        self.assertEqual(code, 2)
        self.assertIn("--check-column needs --correct-column or --grader-file", error)
        code, _, error = run(
            *base, "--correct-column", "ok", "--prediction-column", "guess"
        )
        self.assertEqual(code, 2)
        self.assertIn(
            "use --correct-column for 0/1 grades or --prediction-column", error
        )
        code, _, error = run(*base)
        self.assertEqual(code, 2)
        self.assertIn("missing CSV columns: prediction, reference", error)
        self.assertIn("Name a 0/1 grade column with --correct-column", error)
        code, _, error = run(
            "scores", table, "--score-column", "score", "--correct-column", "ok",
            "--prediction-column", "guess",
        )  # fmt: skip
        self.assertEqual(code, 2)
        self.assertIn("--prediction-column applies to exact-match grading", error)

    def test_policy_reports_the_whole_table_next_to_the_holdout(self):
        code, summary, _ = run(
            "policy", self.table(), "--correct-column", "ok", "--risk-target", "0.3"
        )
        self.assertEqual(code, 0)
        # 50 of 100 rows score 0.9; design and calibration hold 20 + 40 rows.
        self.assertIn(
            "Applied to all 100 rows, the cutoff accepts 50 (50%). 60 of the 100 rows "
            "were used to choose the cutoff, so only the held-out rows give the error "
            "estimate.",
            summary,
        )
        _, report, _ = run(
            "policy",
            self.table(),
            "--correct-column",
            "ok",
            "--risk-target",
            "0.3",
            "--json",
        )
        report = json.loads(report)
        self.assertEqual(
            report["all_rows"],
            {"total": 100, "accepted": 50, "coverage": 0.5, "fitting": 60},
        )
        self.assertEqual(report["selection"]["selected_threshold"], 0.9)

    def test_no_cutoff_reports_the_lowest_error_reached(self):
        table = self.table(reversed_score=True)
        code, summary, _ = run(
            "policy", table, "--correct-column", "ok", "--risk-target", "0.3"
        )
        self.assertEqual(code, 1)
        _, report, _ = run(
            "policy", table, "--correct-column", "ok", "--risk-target", "0.3", "--json"
        )
        report = json.loads(report)
        self.assertNotIn("all_rows", report)
        # The 0.9 cutoff accepts only wrong rows. The 0.1 cutoff accepts every row.
        accept_all = report["selection"]["candidates"][-1]
        self.assertEqual((accept_all["threshold"], accept_all["coverage"]), (0.1, 1.0))
        rate = f"{100 * accept_all['worst_error_rate']:.3g}%"
        self.assertIn(
            f"The lowest calibration error among the candidates was {rate} at 100% "
            "coverage (threshold 0.1). Compare other scores with `caat scores`, or "
            "raise the target.",
            summary,
        )
        code, summary, _ = run(
            "select", table, "--correct-column", "ok", "--thresholds", "0.9", "0.1",
            "--risk-target", "0.3",
        )  # fmt: skip
        self.assertEqual(code, 1)
        self.assertIn("was 50% at 100% coverage (threshold 0.1)", summary)

    def test_route_keeps_named_columns_and_omits_an_absent_group(self):
        decisions = self.root / "decisions.csv"
        route = ["route", "--threshold", "0.5", "--output", decisions]
        code, _, _ = run(
            *route, self.table(), "--keep-column", "guess", "--keep-column", "truth"
        )
        self.assertEqual(code, 0)
        with decisions.open(encoding="utf-8", newline="") as source:
            rows = list(csv.reader(source))
        self.assertEqual(rows[0], ["id", "score", "decision", "guess", "truth"])
        self.assertEqual(rows[1], ["000", "0.9", "accept", "cat", "cat"])
        self.assertEqual(rows[2], ["001", "0.1", "review", "cat", "dog"])

        code, _, _ = run(*route, self.table("grouped.csv", group=True))
        self.assertEqual(code, 0)
        with decisions.open(encoding="utf-8", newline="") as source:
            self.assertEqual(
                next(csv.reader(source)), ["id", "score", "decision", "group"]
            )
        # A decisions file without a group column still plans a label sample.
        code, summary, _ = run(*route, self.table())
        code, summary, _ = run(
            "plan", "--batch-decisions", decisions, "--risk-target", "0.3",
            "--expected-error", "0.1",
        )  # fmt: skip
        self.assertEqual(code, 0, summary)

        code, _, error = run(*route, self.table(), "--keep-column", "score")
        self.assertEqual(code, 2)
        self.assertIn("--keep-column names must be distinct", error)


if __name__ == "__main__":
    unittest.main()
