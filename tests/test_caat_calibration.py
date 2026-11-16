import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

from caat import GradedRow, policy
from caat.calibrate import calibration_transform, fit_calibration
from caat.cli import main
from caat.metrics import calibration_metrics
from caat.policy import split_rows


class CalibrationTests(unittest.TestCase):
    def test_calibration_metrics_and_boundary_ties(self):
        probabilities = [0.0, 0.5, 0.5, 1.0]
        outcomes = [True, True, False, True]
        result = calibration_metrics(probabilities, outcomes, bins=2)
        self.assertEqual(result["brier_score"], 0.375)
        self.assertEqual(result["ece"], 0.25)
        self.assertEqual(result["ace"], 0.5)
        self.assertEqual(result["reliability_bins"][0]["count"], 1)
        self.assertEqual(result["reliability_bins"][1]["count"], 3)
        self.assertEqual(
            result,
            calibration_metrics(probabilities, [True, False, True, True], bins=2),
        )

    def test_calibrators_have_hand_checked_maps(self):
        def fit(method, scores, labels, **options):
            saved = fit_calibration(scores, labels, method=method, **options)
            return calibration_transform(json.loads(json.dumps(saved)))

        # Positive logit log(3) and a 3:1 observed rate give temperature 1.
        temperature = fit("temperature", [0.75] * 4, [0, 1, 1, 1])
        self.assertAlmostEqual(temperature(0.75), 0.75)
        self.assertAlmostEqual(temperature(0.25), 0.25)
        platt = fit("platt", [5, 5, 5, 5], [0, 1, 1, 1])
        self.assertAlmostEqual(platt(-100), 0.75, places=8)
        isotonic = fit("isotonic", [0, 1, 1, 2, 3], [0, 1, 1, 0, 1])
        for score, expected in ((-1, 0), (0.5, 1 / 3), (1, 2 / 3), (2, 2 / 3), (4, 1)):
            self.assertAlmostEqual(isotonic(score), expected)
        histogram = fit("histogram", [0, 1, 0.9], [0, 1, 1], bins=4)
        self.assertEqual([histogram(x) for x in (0, 0.5, 1)], [0, 2 / 3, 1])

    def test_policy_map_excludes_design_and_heldout_labels(self):
        rows = [GradedRow(str(i), i / 30, (i % 3 != 0,)) for i in range(30)]
        options = {
            "graders": ("correct",),
            "risk_target": 0.5,
            "calibrate": "histogram",
        }
        actual = policy(rows, **options)
        raw = policy(rows, graders=("correct",), risk_target=0.5)
        self.assertEqual(actual["selection"], raw["selection"])
        self.assertEqual(actual["heldout"], raw["heldout"])
        design, calibration, heldout = split_rows(rows, 42)
        flipped = [
            GradedRow(row.id, row.score, (not row.outcomes[0],))
            for row in (*design, *heldout)
        ]
        modified = policy([*calibration, *flipped], **options)
        self.assertEqual(
            actual["probability_calibration"]["graders"]["correct"]["map"],
            modified["probability_calibration"]["graders"]["correct"]["map"],
        )

    def test_calibrate_apply_evaluate_and_html(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            calibration = output / "calibration.csv"
            calibration.write_text(
                "id,score,correct\na,0,0\nb,0.7,1\n", encoding="utf-8"
            )
            heldout = output / "heldout.csv"
            heldout.write_text("id,score,correct\nc,0.35,1\nd,1,1\n", encoding="utf-8")
            mapping = output / "isotonic.json"
            mapped = output / "mapped.csv"
            reliability = output / "reliability.svg"
            html = output / "report.html"
            self.assertEqual(
                main(
                    [
                        "calibrate",
                        "fit",
                        str(calibration),
                        "--correct-column",
                        "correct",
                        "--method",
                        "isotonic",
                        "--output",
                        str(mapping),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "calibrate",
                        "apply",
                        str(heldout),
                        "--map",
                        str(mapping),
                        "--output",
                        str(mapped),
                    ]
                ),
                0,
            )
            with mapped.open(encoding="utf-8", newline="") as source:
                self.assertEqual(
                    [row["calibrated_probability"] for row in csv.DictReader(source)],
                    ["0.5", "1.0"],
                )
            self.assertEqual(
                main(
                    [
                        "evaluate",
                        str(mapped),
                        "--correct-column",
                        "correct",
                        "--score-column",
                        "calibrated_probability",
                        "--threshold",
                        "0.5",
                        "--probability",
                        "--reliability-plot",
                        str(reliability),
                        "--html",
                        str(html),
                    ]
                ),
                0,
            )
            self.assertEqual(
                len(ElementTree.parse(reliability).findall(".//{*}circle")), 2
            )
            report = html.read_text(encoding="utf-8")
            self.assertIn("Reliability diagram", report)
            self.assertIn("Brier 0.1250", report)
            self.assertIn('aria-labelledby="chart-2"', report)
