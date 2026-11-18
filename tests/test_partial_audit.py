import csv
import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.cli import main
from caat.metrics import Counts
from caat.partial import progress_report
from caat.plan import batch_verdict
from caat.rules import assess_accepted_risk

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "results/anli_r2"


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_grades(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, ["id", "output_key", "correct"])
        writer.writeheader()
        writer.writerows(rows)


class PartialAuditTests(unittest.TestCase):
    def test_prefix_verdict_matches_every_completion(self):
        for size in range(1, 7):
            for cap in range(1, size + 1):
                for target in (0, 0.2, 0.5, 1):
                    plan = dict(
                        accepted_population=size,
                        accepted=cap,
                        label_budget=cap,
                        risk_target=target,
                        holdout_confidence=0.95,
                    )
                    terminal = []
                    for errors in range(cap + 1):
                        full = assess_accepted_risk(
                            Counts(cap, cap, errors),
                            target,
                            0.95,
                            accepted_population=size,
                        )
                        terminal.append(
                            batch_verdict(full, size, target, 0.95)["batch_verdict"]
                        )
                    for t in range(cap + 1):
                        for bits in product((0, 1), repeat=t):
                            grades = {"g": dict(enumerate(bits))}
                            result = progress_report(plan, grades)["graders"]["g"]
                            errors = t - sum(bits)
                            possible = set(terminal[errors : errors + cap - t + 1])
                            self.assertEqual(
                                result["batch_verdict"],
                                possible.pop() if len(possible) == 1 else None,
                            )
                            self.assertEqual("risk_upper_bound" in result, t == cap)

    def test_empty_eligible_batch_and_full_text(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp)
            batch, sample, plan, grades = [
                p / name
                for name in ("batch.csv", "sample.csv", "plan.json", "grades.csv")
            ]
            batch.write_text("id,gap,output_sha256\nnew,0,untaken\n", encoding="utf-8")
            write_grades(grades, [])
            inputs = [
                "--batch-predictions",
                str(batch),
                "--threshold-report",
                str(R2 / "policy.json"),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "plan",
                            *inputs,
                            "--sample-output",
                            str(sample),
                            "--output",
                            str(plan),
                            "--label-budget",
                            "10",
                        ]
                    ),
                    1,
                )
            with redirect_stdout(io.StringIO()) as text:
                self.assertEqual(
                    main(
                        [
                            "evaluate",
                            str(sample),
                            *inputs,
                            "--partial",
                            "--plan-report",
                            str(plan),
                            "--grader-file",
                            f"g={grades}",
                        ]
                    ),
                    1,
                )
            self.assertIn("0 eligible outputs", text.getvalue())
            self.assertIn("Full-budget verdict: inconclusive", text.getvalue())
            self.assertNotIn("risk limits", text.getvalue())
            with redirect_stdout(io.StringIO()) as text:
                main(
                    [
                        "evaluate",
                        str(R2 / "sample.csv"),
                        "--partial",
                        "--threshold-report",
                        str(R2 / "policy.json"),
                        "--plan-report",
                        str(R2 / "plan.json"),
                        "--batch-predictions",
                        str(R2 / "predictions.csv"),
                        "--grader-file",
                        f"g={R2 / 'sample-grades.csv'}",
                    ]
                )
            self.assertIn("risk limits", text.getvalue())

    def test_r2_prefix_and_grade_identity(self):
        sample = read_rows(R2 / "sample.csv")
        lookup = {r["id"]: r for r in read_rows(R2 / "sample-grades.csv")}
        with TemporaryDirectory() as tmp:
            grades = Path(tmp) / "grades.csv"
            args = [
                "evaluate",
                str(R2 / "sample.csv"),
                "--partial",
                "--batch-predictions",
                str(R2 / "predictions.csv"),
                "--threshold-report",
                str(R2 / "policy.json"),
                "--plan-report",
                str(R2 / "plan.json"),
                "--grader-file",
                f"anli={grades}",
                "--json",
            ]
            for t, want in [(0, None), (58, None), (59, "exceeds"), (100, "exceeds")]:
                write_grades(grades, [lookup[r["id"]] for r in reversed(sample[:t])])
                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertEqual(main(args), 1)
                result = json.loads(stdout.getvalue())["graders"]["anli"]
                self.assertEqual(result["batch_verdict"], want)
                self.assertEqual(result["graded"], t)
                if t == 100:
                    self.assertEqual(result["risk_lower_errors"], 205)
            for bad in [
                [lookup[sample[1]["id"]]],
                [dict(lookup[sample[0]["id"]], output_key="changed")],
            ]:
                write_grades(grades, bad)
                with (
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as exc,
                ):
                    main(args)
                self.assertEqual(exc.exception.code, 2)
