import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.cli import main


class SelectedOfferTests(unittest.TestCase):
    def test_selected_schedule_survives_partial_receipt_and_retry(self):
        root = Path(__file__).resolve().parents[1] / "results"
        policy = root / "anli_r2/policy.json"
        full = root / "winogrande/predictions.csv"

        def call(*args):
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main([*map(str, args), "--json"]), 0)
            return json.loads(out.getvalue())

        for small, name in ((True, "smallest-cap"), (False, "lowest-cost")):
            with self.subTest(small=small), TemporaryDirectory() as directory:
                work = Path(directory)
                batch = full
                if small:
                    batch = work / "batch.csv"
                    with full.open(newline="") as f, batch.open("w", newline="") as out:
                        reader = csv.DictReader(f)
                        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
                        writer.writeheader()
                        writer.writerows(
                            [r for r in reader if float(r["gap"]) >= 4.75][:12]
                        )
                source = ("--threshold-report", policy, "--batch-predictions", batch)
                preview, plan, sample = (
                    work / n for n in ("offers.json", "plan.json", "sample.csv")
                )
                compared = call(
                    "plan",
                    *source,
                    "--forecast",
                    "--budget-ceiling",
                    500,
                    "--error-scenarios",
                    *(("0", "1") if small else (".15", ".25")),
                    "--output",
                    preview,
                )
                draw = (
                    "plan",
                    *source,
                    "--from-comparison",
                    preview,
                    "--choose",
                    name,
                    "--output",
                    plan,
                    "--sample-output",
                    sample,
                )
                saved = call(*draw)
                offer = compared["offers"][name]
                self.assertEqual(saved["accepted"], offer["cap"])
                if small:
                    self.assertEqual(saved["input"]["block_size"], offer["block_size"])
                    self.assertEqual(saved["command"], "fixed-block-plan")
                else:
                    self.assertEqual(saved["stages"], offer["stages"])
                issue = (
                    "audit-block",
                    *source,
                    "--plan-report",
                    plan,
                    "--grader",
                    "reference",
                    "--handoff",
                    work / "handoff",
                )
                request = call(*issue)["request"]
                with Path(request["path"]).open(newline="") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(
                    len(rows),
                    offer["block_size"] if small else offer["stages"][0]["labels"],
                )
                receipt = work / "grades.csv"
                submit = (
                    *issue,
                    "--request",
                    request["id"],
                    "--grader-file",
                    f"reference={receipt}",
                )
                for items in (rows[:1], rows):
                    with receipt.open("w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(("id", "output_key", "correct"))
                        w.writerows((r["id"], r["output_key"], 1) for r in items)
                    returned = call(*submit)
                    self.assertEqual(returned, call(*issue))
                    self.assertEqual(returned, call(*submit))
                self.assertIsNone(returned["request"])
                self.assertEqual(
                    returned["graders"]["reference"]["batch_verdict"], "meets"
                )
                if small:
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        call(*issue, "--block-size", 1)
                    compared["offers"][name]["cap"] -= 1
                    preview.write_text(json.dumps(compared))
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        call(*draw)
