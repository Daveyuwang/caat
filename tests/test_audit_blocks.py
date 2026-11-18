import csv
import json
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.cli import main
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

R2 = Path(__file__).resolve().parents[1] / "results/anli_r2"


class BlockTests(unittest.TestCase):
    def test_concurrent_issue_and_recover_partial_receipts(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            args = [
                "audit-block",
                "--handoff",
                str(folder / "work"),
                "--plan-report",
                str(R2 / "plan.json"),
                "--threshold-report",
                str(R2 / "policy.json"),
                "--batch-predictions",
                str(R2 / "predictions.csv"),
                "--grader",
                "a",
                "--grader",
                "b",
                "--block-size",
                "10",
                "--json",
            ]

            def invoke(extra=()):
                with redirect_stdout(StringIO()) as out:
                    self.assertEqual(main([*args, *extra]), 0)
                return json.loads(out.getvalue())

            def start(_):
                cmd = [
                    sys.executable,
                    "-B",
                    "-c",
                    "from caat.cli import main; raise SystemExit(main())",
                    *args,
                ]
                return json.loads(subprocess.check_output(cmd, text=True))

            with ThreadPoolExecutor(3) as pool:
                reports = list(pool.map(start, range(3)))
            self.assertEqual(reports, [reports[0]] * 3)
            report = reports[0]
            first = report["request"]["id"]
            with (R2 / "sample-grades.csv").open(newline="") as stream:
                known = {r["id"]: r for r in csv.DictReader(stream)}

            def put(name, ids, token, *, flip=False, invalid=False):
                path = folder / "returned.csv"
                with path.open("w", newline="") as out:
                    w = csv.DictWriter(out, ["id", "output_key", "correct"])
                    w.writeheader()
                    for uid in ids:
                        r = known[uid]
                        bit = int(r["correct"]) if name == "b" else 0
                        w.writerow(dict(r, correct=1 - bit if flip else bit))
                extra = ["--request", token, "--grader-file", f"{name}={path}"]
                if invalid:
                    with (
                        redirect_stderr(StringIO()),
                        self.assertRaises(SystemExit) as e,
                    ):
                        invoke(extra)
                    self.assertEqual(e.exception.code, 2)
                    return
                return invoke(extra)

            while req := report["request"]:
                with Path(req["path"]).open(newline="") as stream:
                    ids = [r["id"] for r in csv.DictReader(stream)]
                if req["id"] == first:
                    partial = put("a", [ids[-1]], first)
                    self.assertEqual(partial["outstanding"]["a"], ids[:-1])
                    self.assertEqual(partial["requests"], 1)
                    self.assertEqual(invoke(), partial)
                    put("a", [ids[-1]], first, flip=True, invalid=True)
                elif req["start"] == 10:
                    put("a", [ids[0]], first, invalid=True)
                for name in req["graders"]:
                    report = put(name, reversed(ids), req["id"])
                self.assertEqual(report, invoke())
            self.assertEqual(
                (
                    report["unique_requested"],
                    report["requested_labels"],
                    report["returned_labels"],
                ),
                (60, 90, 90),
            )
            self.assertEqual(
                {name: r["graded"] for name, r in report["graders"].items()},
                {"a": 30, "b": 60},
            )
            original = args.index("10")
            args[original] = "11"
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                invoke()
