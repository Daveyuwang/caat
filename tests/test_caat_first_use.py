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

    def table(
        self, name="graded.csv", *, reversed_score=False, group=False, noisy=False
    ):
        # Even rows are right ("cat" == "cat") and carry the high score.
        # With noisy, every tenth row is a confident error.
        path = self.root / name
        header = "id,score,twin,guess,truth,ok" + (",group" if group else "")
        lines = [header]
        for i in range(100):
            right = i % 2 == 0 and not (noisy and i % 10 == 0)
            score = 0.9 if (i % 2 == 0) != reversed_score else 0.1
            line = (
                f"{i:03d},{score},{score},cat,{'cat' if right else 'dog'},{int(right)}"
            )
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

    def test_no_cutoff_explains_a_rate_under_the_target(self):
        # Four right rows above the cutoff: 0% error, but 0.6**4 = 0.13 > 0.05.
        path = self.root / "few.csv"
        rows = [f"{i},{0.9 if i < 4 else 0.1},{int(i < 4)}" for i in range(8)]
        path.write_text("id,score,ok\n" + "\n".join(rows) + "\n", encoding="utf-8")
        command = [
            "select", path, "--correct-column", "ok", "--thresholds", "0.9",
            "--risk-target", "0.4",
        ]  # fmt: skip
        code, summary, _ = run(*command)
        self.assertEqual(code, 1)
        self.assertIn(
            "was 0% at 50% coverage (threshold 0.9). That rate is under the target, "
            "but 4 accepted rows are too few to show it with 95% confidence.",
            summary,
        )
        code, summary, _ = run(*command, "--rule", "empirical")
        self.assertEqual(code, 0)
        code, summary, _ = run(*command, "--rule", "ltt-fixed-sequence")
        self.assertEqual(code, 1)
        self.assertIn(
            "The strictest candidate (threshold 0.9, 4 accepted rows, 0% calibration "
            "error) did not pass, and the fixed-sequence rule stops at the first failure.",
            summary,
        )

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

    def test_policy_summary_says_why_the_result_holds(self):
        base = [
            "policy",
            self.table(),
            "--correct-column",
            "ok",
            "--risk-target",
            "0.3",
        ]
        _, summary, _ = run(*base)
        self.assertIn(
            "The rule keeps accepted error at or below 30% with 95% confidence on new "
            "rows like the calibration rows.",
            summary,
        )
        self.assertIn(
            "Score ranking under ok on the calibration rows: AUROC 1.000.", summary
        )
        _, report, _ = run(*base, "--json")
        held = json.loads(report)["heldout"]["graders"]["ok"]
        # With zero errors among k accepted rows the 95% bound is 1 - 0.05 ** (1 / k).
        self.assertEqual(held["errors"], 0)
        bound = 1 - 0.05 ** (1 / held["accepted"])
        self.assertAlmostEqual(held["upper_bound_95"], bound, places=9)
        self.assertIn(
            f"with 0 errors (0%, 95% upper bound {100 * bound:.3g}%). "
            "The observed rate meets the target.",
            summary,
        )

        _, report, _ = run(*base, "--holdout-confidence", "0.95", "--json")
        held = json.loads(report)["heldout"]["graders"]["ok"]
        self.assertNotIn("upper_bound_95", held)
        self.assertAlmostEqual(held["risk_upper_bound"], bound, places=9)

        _, summary, _ = run(*base, "--rule", "empirical")
        self.assertIn("gives no statistical guarantee.", summary)
        code, summary, _ = run(
            "policy", self.table(reversed_score=True), "--correct-column", "ok",
            "--risk-target", "0.3",
        )  # fmt: skip
        self.assertEqual(code, 1)
        self.assertTrue(
            summary.rstrip().endswith("Exit status 1: no cutoff was selected.")
        )

    def test_scores_names_the_split_and_calls_a_tie(self):
        code, summary, _ = run(
            "scores", self.table(), "--score-column", "score", "--score-column", "twin",
            "--correct-column", "ok", "--bootstrap", "50",
        )  # fmt: skip
        self.assertEqual(code, 0)
        self.assertIn(
            "Compared scores on 40 held-out rows. Baseline score: score.", summary
        )
        self.assertIn("chosen by AURC gain on 20 design rows.", summary)
        # The twin column equals the baseline, so every paired difference is zero.
        self.assertIn(
            "Tied with score: every 95% interval for the difference includes zero.",
            summary,
        )
        self.assertNotIn("(unknown)", summary)
        # Every wrong row outranks every right row, so the AUROC is 0.
        _, summary, _ = run(
            "scores", self.table(reversed_score=True), "--score-column", "score",
            "--correct-column", "ok", "--bootstrap", "0",
        )  # fmt: skip
        self.assertIn("  score. AUROC 0.000.", summary)
        self.assertIn(
            "AUROC well below 0.5: the score may point the wrong way. If smaller "
            "values mean more confidence, add --lower-is-better score.",
            summary,
        )

    def test_policy_explains_the_cutoff_it_did_not_choose(self):
        table = self.table()
        base = ["policy", table, "--correct-column", "ok", "--risk-target", "0.3"]
        _, summary, _ = run(*base)
        _, report, _ = run(*base, "--json")
        # The only looser cutoff accepts every calibration row, wrong ones included.
        accept_all = json.loads(report)["selection"]["candidates"][-1]
        wrong = accept_all["errors"]["ok"]
        self.assertEqual(accept_all["accepted"], 40)
        self.assertIn(
            "The next looser candidate (threshold 0.1, 100% of calibration rows) had "
            f"{100 * wrong / 40:.3g}% calibration error, above the target.",
            summary,
        )
        # Every split puts the 50 right rows above the 0.9 cutoff.
        self.assertIn(
            "Split check: 20 of 20 other seeds selected a cutoff. Those cutoffs "
            "accept 50 to 50 of the 100 rows (median 50).",
            summary,
        )
        _, report, _ = run(*base, "--split-checks", "3", "--json")
        check = json.loads(report)["split_sensitivity"]
        counts = []
        for seed in (43, 44, 45):
            _, other, _ = run(*base, "--seed", seed, "--split-checks", "0", "--json")
            other = json.loads(other)
            self.assertNotIn("split_sensitivity", other)
            counts.append(other["all_rows"]["accepted"])
        self.assertEqual(
            check,
            {
                "splits": 3,
                "selected": 3,
                "accepted_min": min(counts),
                "accepted_median": sorted(counts)[1],
                "accepted_max": max(counts),
            },
        )

    def test_external_scores_can_match_on_the_sample_id_alone(self):
        from caat.table import write_external_score_join

        scores = self.root / "scores.csv"
        scores.write_text("sample_id,confidence\nb,0.2\na,0.9\n", encoding="utf-8")
        identity = ("eval_id", "id", "epoch", "output_sha256")

        def join(**options):
            samples = (
                (("run", name, "1", f"hash-{name}"), (grade,))
                for name, grade in (("a", 1), ("b", 0))
            )
            return write_external_score_join(
                self.root / "log.eval",
                scores,
                self.root / "joined.csv",
                samples,
                identity_columns=identity,
                grade_columns=("ok",),
                score_column="confidence",
                **options,
            )

        self.assertEqual(join(score_id_column="sample_id"), 2)
        with (self.root / "joined.csv").open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(
            [
                (row["id"], row["output_sha256"], row["confidence"], row["ok"])
                for row in rows
            ],
            [("a", "hash-a", "0.9", "1"), ("b", "hash-b", "0.2", "0")],
        )
        with self.assertRaisesRegex(ValueError, "which has the identity columns"):
            join()

    def test_policy_says_when_the_bound_is_above_the_target(self):
        # A fifth of the confident rows are wrong: under a 30% target, but not
        # by enough for about 20 held-out accepted rows to rule out more.
        base = [
            "policy", self.table(noisy=True), "--correct-column", "ok",
            "--risk-target", "0.3", "--rule", "empirical",
        ]  # fmt: skip
        code, summary, _ = run(*base)
        _, report, _ = run(*base, "--json")
        held = json.loads(report)["heldout"]["graders"]["ok"]
        self.assertTrue(held["meets_target"])
        self.assertGreater(held["upper_bound_95"], 0.3)
        self.assertEqual(code, 0)
        self.assertIn(
            "The observed rate meets the target, but the upper bound is above it: "
            "these rows are too few to rule out a higher rate.",
            summary,
        )
        self.assertIn(
            "The held-out rows confirm the policy by the observed rate. "
            "Add --holdout-confidence 0.95 to make the upper bound decide.",
            summary,
        )

    def test_plan_takes_the_expected_error_from_a_policy_report(self):
        report = self.root / "policy.json"
        table = self.table(noisy=True)
        # Ten percent of the confident rows are wrong, so a 50% target leaves room.
        run(
            "policy",
            table,
            "--correct-column",
            "ok",
            "--risk-target",
            "0.5",
            "--output",
            report,
        )
        held = json.loads(report.read_text(encoding="utf-8"))["heldout"]["graders"][
            "ok"
        ]
        self.assertGreater(held["errors"], 0)
        code, plan, _ = run("plan", "--threshold-report", report, "--json")
        self.assertEqual(code, 0, plan)
        plan = json.loads(plan)
        self.assertEqual(plan["input"]["expected_error"], held["error_rate"])
        self.assertEqual(plan["input"]["expected_error_source"], "threshold_report")
        _, summary, _ = run("plan", "--threshold-report", report)
        self.assertIn("the held-out rate in the threshold report", summary)
        code, _, error = run("plan", "--risk-target", "0.3")
        self.assertEqual(code, 2)
        self.assertIn("give --expected-error", error)

    def test_calibration_summary_states_the_direction(self):
        for score, lean in (("0.9", "overconfident"), ("0.1", "underconfident")):
            path = self.root / f"flat-{lean}.csv"
            rows = [f"{i},{score},{i % 2}" for i in range(10)]
            path.write_text("id,score,ok\n" + "\n".join(rows) + "\n", encoding="utf-8")
            _, summary, _ = run(
                "evaluate", path, "--correct-column", "ok", "--threshold", "0.5",
                "--probability",
            )  # fmt: skip
            percent = "90%" if score == "0.9" else "10%"
            self.assertIn(
                f"Mean probability {percent} against 50% correct: {lean}.", summary
            )

    def test_inspect_grader_error_lists_the_scorers(self):
        from types import SimpleNamespace

        from caat.inspect_log import binary_grade

        sample = SimpleNamespace(
            id="q1",
            scores={
                "alias": SimpleNamespace(value="C"),
                "exact": SimpleNamespace(value="I"),
            },
        )
        self.assertEqual(binary_grade(sample, "alias"), 1)
        with self.assertRaisesRegex(ValueError, "Scorers in this sample: alias, exact"):
            binary_grade(sample, "primary")


if __name__ == "__main__":
    unittest.main()
