import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.audit import AuditRow
from caat.cli import main
from caat.policy import split_rows


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ifeval_graders.csv"


class PolicyTests(unittest.TestCase):
    def run_policy(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["policy", *args])
        return exit_code, output.getvalue()

    def test_split_is_reproducible_and_complete(self):
        rows = [AuditRow(str(i), i / 10, (True,)) for i in range(10)]
        design, calibration, heldout = split_rows(rows, 42)
        self.assertEqual((len(design), len(calibration), len(heldout)), (2, 4, 4))
        self.assertEqual(sorted(design + calibration + heldout, key=lambda row: row.id), rows)
        self.assertEqual(split_rows(rows[::-1], 42), (design, calibration, heldout))

    def test_check_grader_leaves_selection_unchanged(self):
        base = [str(EXAMPLE), "--score-column", "mean_token_logprob", "--risk-target", "0.4",
                "--correct-column", "strict_pass"]
        exit_code, output = self.run_policy(*base, "--check-column", "helm_strict_pass", "--json")
        report = json.loads(output)
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["split"], {"seed": 42, "design": 108, "calibration": 216, "heldout": 217})
        self.assertEqual(report["selection"]["graders"], ["strict_pass"])
        self.assertEqual(report["selection"]["selected_threshold"], -0.43964836263206303)
        held = report["heldout"]["graders"]
        self.assertEqual((held["strict_pass"]["accepted"], held["strict_pass"]["errors"]), (144, 33))
        self.assertEqual((held["helm_strict_pass"]["accepted"], held["helm_strict_pass"]["errors"]), (144, 45))
        self.assertEqual(report["heldout"]["accepted_comparisons"]["helm_strict_pass"]["newly_failed"], 12)

        exit_code, output = self.run_policy(*base, "--correct-column", "helm_strict_pass", "--json")
        report = json.loads(output)
        self.assertEqual(report["selection"]["selected_threshold"], -0.3327795407672605)
        held = report["heldout"]["graders"]
        self.assertEqual([held[name]["errors"] for name in held], [19, 23])
        self.assertEqual(held["strict_pass"]["accepted"], 92)

        exit_code, summary = self.run_policy(*base, "--check-column", "helm_strict_pass", "--risk-target", "0.3",
                                             "--rule", "empirical")
        self.assertEqual(exit_code, 1)
        self.assertIn("helm_strict_pass (check only): accepted 144 of 217 with 45 errors (31.2%), which exceeds",
                      summary)

    def test_reversed_score_selects_nothing(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "graded.csv"
            rows = [f"{i},{0.1 if i % 2 else 0.9},{i % 2}" for i in range(100)]
            source.write_text("id,score,correct\n" + "\n".join(rows) + "\n", encoding="utf-8")
            exit_code, summary = self.run_policy(str(source), "--correct-column", "correct", "--risk-target", "0.3")
            self.assertEqual(exit_code, 1)
            self.assertIn("AUROC 0.000, no better than random", summary)
            self.assertIn("No candidate threshold met the 30% risk target", summary)

            source.write_text("id,score,correct\n1,0.9,1\n2,0.8,1\n3,0.7,0\n4,0.6,1\n", encoding="utf-8")
            with self.assertRaises(SystemExit) as caught, redirect_stderr(io.StringIO()) as errors:
                main(["policy", str(source), "--correct-column", "correct", "--risk-target", "0.3"])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("at least five rows", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
