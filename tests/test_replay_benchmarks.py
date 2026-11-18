import csv
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]


class ReplayTests(unittest.TestCase):
    def test_bundled_samples_keep_their_counts(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "replay_benchmarks.py"),
                    "--output-dir",
                    directory,
                ],
                check=True,
                capture_output=True,
            )
            with (output / "summary.csv").open(encoding="utf-8", newline="") as source:
                counts = {
                    row["benchmark"]: (int(row["accepted"]), int(row["errors"]))
                    for row in csv.DictReader(source)
                }
        self.assertEqual(
            counts, {"ifeval": (12, 4), "mmlu_pro": (12, 5), "triviaqa": (13, 0)}
        )


if __name__ == "__main__":
    unittest.main()
