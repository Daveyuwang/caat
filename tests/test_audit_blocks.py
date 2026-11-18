import csv
import json
import sqlite3
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.cli import main
from contextlib import closing, redirect_stderr, redirect_stdout
from io import StringIO

R2 = Path(__file__).resolve().parents[1] / "results/anli_r2"


class BlockTests(unittest.TestCase):
    def test_block_reissues_the_selected_sample_fields(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            sample = folder / "draw.csv"
            policy = [
                "--threshold-report",
                str(R2 / "policy.json"),
                "--batch-predictions",
                str(R2 / "predictions.csv"),
            ]
            plan = folder / "plan.json"
            with redirect_stdout(StringIO()):
                self.assertEqual(
                    main(
                        [
                            "plan",
                            *policy,
                            "--label-budget",
                            "100",
                            "--sample-output",
                            str(sample),
                            "--sample-source",
                            str(R2 / "predictions.csv"),
                            "--sample-column",
                            "answer",
                            "--output",
                            str(plan),
                        ]
                    ),
                    0,
                )
            with sample.open(encoding="utf-8", newline="") as stream:
                expected = list(csv.DictReader(stream))[:10]
            args = [
                "audit-block",
                *policy,
                "--plan-report",
                str(plan),
                "--block-size",
                "10",
                "--grader",
                "anli",
                "--handoff",
                str(folder / "handoff"),
                "--json",
            ]
            with redirect_stdout(StringIO()) as out:
                self.assertEqual(main(args), 0)
            path = Path(json.loads(out.getvalue())["request"]["path"])
            with path.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream)), expected)
            path.unlink()
            sample.unlink()
            with redirect_stdout(StringIO()):
                self.assertEqual(main(args), 0)
            with path.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream)), expected)

    def test_upgrade_resumes_an_open_request_and_reads_the_saved_policy(self):
        legacy = Path(__file__).parent / "fixtures/upgrade"
        expected = json.loads((legacy / "expected.json").read_text(encoding="utf-8"))
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with closing(sqlite3.connect(directory / "audit.sqlite3")) as db:
                db.executescript((legacy / "handoff.sql").read_text(encoding="utf-8"))
            with redirect_stdout(StringIO()) as output:
                main(
                    [
                        "audit-block",
                        "--handoff",
                        tmp,
                        "--plan-report",
                        str(legacy / "plan.json"),
                        "--threshold-report",
                        str(R2 / "policy.json"),
                        "--batch-predictions",
                        str(R2 / "predictions.csv"),
                        "--block-size",
                        "10",
                        "--grader",
                        "anli",
                        "--json",
                    ]
                )
            report = json.loads(output.getvalue())
            self.assertEqual(report["request"]["id"], expected["request"])
            self.assertEqual(report["outstanding"], expected["outstanding"])
            self.assertEqual(report["returned_labels"], 1)

            decisions = directory / "decisions.csv"
            with redirect_stdout(StringIO()):
                main(
                    [
                        "route",
                        str(R2.parents[1] / "examples/ifeval_strict.csv"),
                        "--threshold-report",
                        str(legacy / "policy.json"),
                        "--output",
                        str(decisions),
                    ]
                )
            with decisions.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(
                    {r["id"]: r["decision"] for r in csv.DictReader(stream)},
                    expected["decisions"],
                )

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
            issued_path = Path(report["request"]["path"])
            initial_stat = issued_path.stat()
            self.assertEqual(invoke(), report)
            self.assertEqual(issued_path.stat().st_mtime_ns, initial_stat.st_mtime_ns)
            issued_path.unlink()
            self.assertEqual(invoke(), report)
            self.assertTrue(issued_path.is_file())
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
