import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from caat import GradedRow, select
from caat.cli import main
from caat.metrics import Counts, ranking_metrics, threshold_points
from caat.rules import binomial_pvalue


class RankingTests(unittest.TestCase):
    def test_ties_and_single_class(self):
        rows = [(0.9, False), (0.8, True), (0.8, False), (0.1, True)]
        ranking = ranking_metrics(threshold_points(rows))
        correct = [score for score, error in rows if not error]
        wrong = [score for score, error in rows if error]
        pair_wins = sum(
            (high > low) + 0.5 * (high == low) for high in correct for low in wrong
        )
        self.assertEqual(ranking["auroc"], pair_wins / (len(correct) * len(wrong)))
        self.assertAlmostEqual(ranking["aurc"], 13 / 48)
        self.assertAlmostEqual(ranking["normalized_aurc_gain"], 11 / 14)
        self.assertEqual(
            ranking_metrics(threshold_points([(0.9, True), (0.8, True)])),
            {
                "aurc": 1.0,
                "auroc": None,
                "normalized_aurc_gain": None,
            },
        )


class SelectionTests(unittest.TestCase):
    def test_exact_binomial_tail_and_boundaries(self):
        self.assertAlmostEqual(
            binomial_pvalue(Counts(10, 5, 1), 0.25), 0.75**5 + 5 * 0.25 * 0.75**4
        )
        self.assertEqual(binomial_pvalue(Counts(10, 0, 0), 0.25), 1.0)
        self.assertEqual(binomial_pvalue(Counts(10, 5, 5), 1.0), 1.0)
        self.assertEqual(binomial_pvalue(Counts(10, 5, 4), 1.0), 0.0)

    def test_fixed_sequence_stops_at_first_failure(self):
        rows = [
            GradedRow(str(i), score, (True, correct))
            for i, (score, correct) in enumerate(
                [(0.9, True)] * 10 + [(0.8, False)] * 6 + [(0.7, True)] * 45
            )
        ]
        options = {
            "graders": ("first", "second"),
            "thresholds": [0.7, 0.9, 0.8],
            "risk_target": 0.3,
        }
        fixed = select(rows, rule="ltt-fixed-sequence", **options)
        self.assertEqual(fixed["selected_threshold"], 0.9)
        self.assertEqual(
            [row["passes"] for row in fixed["candidates"]], [True, False, None]
        )
        self.assertEqual(fixed["candidates"][1]["accepted"], 16)
        bonferroni = select(rows, rule="ltt-bonferroni", **options)
        self.assertEqual(bonferroni["selected_threshold"], 0.7)
        reversed_rows = [
            GradedRow(row.id, row.score, row.outcomes[::-1]) for row in rows
        ]
        reordered = select(
            reversed_rows,
            rule="ltt-fixed-sequence",
            **{**options, "graders": options["graders"][::-1]},
        )
        self.assertEqual(fixed["selected_threshold"], reordered["selected_threshold"])
        self.assertEqual(fixed["candidates"], reordered["candidates"])

    def test_cli_design_grid_and_no_selection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "cal.csv"
            calibration.write_text("id,conf,g\na,0.8,1\nb,0.2,0\n", encoding="utf-8")
            design = root / "design.csv"
            design.write_text("conf\n0.6\n0.9\n", encoding="utf-8")
            args = [
                "select",
                str(calibration),
                "--json",
                "--score-column",
                "conf",
                "--correct-column",
                "g",
                "--risk-target",
                "0.2",
                "--design",
                str(design),
            ]
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([*args, "--rule", "empirical"]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["selected_threshold"], 0.6)
            self.assertIsNone(report["guarantee"])
            self.assertEqual([row["accepted"] for row in report["candidates"]], [0, 1])
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(args), 1)
            self.assertIsNone(json.loads(output.getvalue())["selected_threshold"])
