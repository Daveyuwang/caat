from itertools import combinations
from math import comb
import csv
import io
import json
from pathlib import Path
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory
import unittest

from caat.cli import main
from caat.two_stage import design


class TwoStageTests(unittest.TestCase):
    def test_checkpoint_issuance_resumes_independent_graders(self):
        root = Path(__file__).resolve().parents[1] / "results/letter_audit"

        def call(*args):
            with redirect_stdout(io.StringIO()) as out:
                code = main([*map(str, args), "--json"])
            self.assertIn(code, (0, 1))
            return json.loads(out.getvalue())

        def rows(path):
            with Path(path).open(newline="") as f:
                return list(csv.DictReader(f))

        with TemporaryDirectory() as directory:
            tmp = Path(directory)
            source = (
                "--threshold-report",
                root / "policy.json",
                "--batch-predictions",
                root / "predictions.csv",
            )
            sample, plan = tmp / "sample.csv", tmp / "plan.json"
            call(
                "plan",
                *source,
                "--label-budget",
                300,
                "--interim-labels",
                200,
                "--error-scenarios",
                0.02,
                0.08,
                "--sample-output",
                sample,
                "--output",
                plan,
            )
            args = (
                "audit-block",
                *source,
                "--plan-report",
                plan,
                "--handoff",
                tmp / "handoff",
                "--grader",
                "low",
                "--grader",
                "high",
                "--grader",
                "middle",
            )
            first = call(*args)["request"]
            ordered = rows(sample)
            self.assertEqual(rows(first["path"]), ordered[:200])

            def send(request, name, grades):
                file = tmp / (name + ".csv")
                with file.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["id", "output_key", "correct"])
                    writer.writerows(
                        (r["id"], r["output_key"], bit) for r, bit in reversed(grades)
                    )
                return call(
                    *args, "--request", request["id"], "--grader-file", f"{name}={file}"
                )

            graded = [(r, int(i >= 8)) for i, r in enumerate(ordered)]
            send(first, "middle", graded[:19])
            self.assertEqual(call(*args)["request"]["id"], first["id"])
            send(first, "low", [(r, 1) for r in ordered[:200]])
            send(first, "high", [(r, 0) for r in ordered[:200]])
            second = send(first, "middle", graded[:200])["request"]
            self.assertEqual(second["graders"], ["middle"])
            self.assertEqual(rows(second["path"]), ordered[200:])
            self.assertEqual(call(*args)["request"]["id"], second["id"])
            stop = send(
                second,
                "middle",
                [(r, 0 if i < 4 else 1) for i, r in enumerate(ordered[200:])],
            )
            self.assertIsNone(stop["request"])
            self.assertEqual(
                [
                    (
                        stop["graders"][g]["decision_stage"],
                        stop["graders"][g]["batch_verdict"],
                    )
                    for g in ("low", "high", "middle")
                ],
                [(1, "meets"), (1, "exceeds"), (2, "inconclusive")],
            )

    def test_exhaustive_directional_risk_and_forecast(self):
        size, cap, first = 12, 9, 5
        report = design(size, cap, first, 0.3, 0.85, [0, 0.75], 0.6, 1)
        early, final = report["stages"]
        for errors in range(size + 1):
            counts = dict(meets=0, exceeds=0, inconclusive=0)
            issued = requests = 0
            for positions in combinations(range(size), errors):
                for stage in (early, final):
                    n = stage["labels"]
                    x = sum(i < n for i in positions)
                    meet = stage["max_errors_to_meet"]
                    exceed = stage["min_errors_to_exceed"]
                    if meet is not None and x <= meet:
                        verdict = "meets"
                        break
                    if exceed is not None and x >= exceed:
                        verdict = "exceeds"
                        break
                    verdict = "inconclusive"
                counts[verdict] += 1
                issued += n
                requests += 1 + (n == cap)
            total = comb(size, errors)
            self.assertLessEqual(
                counts["exceeds" if errors <= 3 else "meets"] / total, 0.15
            )
            for s in report["scenarios"]:
                if s["errors"] == errors:
                    for key, value in counts.items():
                        self.assertAlmostEqual(s[key], value / total)
                    self.assertAlmostEqual(s["expected_labels"], issued / total)
                    self.assertAlmostEqual(s["expected_requests"], requests / total)
