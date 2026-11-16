import unittest

from caat.policy import split_rows
from caat.scores import ScoredRow, compare_scores


class ScoreTests(unittest.TestCase):
    def test_ties_reversed_scores_and_single_class(self):
        rows = [
            ScoredRow("a", (3, -3), (True, True), 1),
            ScoredRow("b", (2, -2), (False, True), 2),
            ScoredRow("c", (2, -2), (True, True), 2),
            ScoredRow("d", (1, -1), (False, True), 3),
        ]
        options = dict(
            names=("mean_token_logprob", "reversed"),
            graders=("gold", "all_true"),
            split="all",
            length_control=True,
            bootstrap=25,
            costs={"reversed": "external"},
        )
        result = compare_scores(rows, **options)
        self.assertEqual(result, compare_scores(reversed(rows), **options))
        candidates = result["graders"]["gold"]["scores"]
        raw, reverse = candidates.values()
        self.assertEqual(
            result["costs"],
            {"mean_token_logprob": "no-extra-call", "reversed": "external"},
        )
        self.assertAlmostEqual(raw["normalized_aurc_gain"], 11 / 14)
        self.assertEqual((raw["auroc"], reverse["auroc"]), (0.875, 0.125))
        self.assertTrue(reverse["direction_warning"])
        self.assertEqual(
            [item["error_rate"] for item in raw["budgets"]], [0, 0.25, 1 / 3]
        )
        self.assertEqual(raw["budgets"][1]["risk_delta_95"], [0, 0])
        self.assertEqual(reverse["budgets"][1]["risk_delta"], 0.5)
        self.assertIsNone(result["graders"]["all_true"]["scores"]["reversed"]["auroc"])
        self.assertEqual(
            result["graders"]["gold"]["controls"]["random"]["budgets"][1]["error_rate"],
            0.5,
        )
        self.assertEqual(
            result["graders"]["gold"]["controls"]["length_only"]["budgets"][1][
                "error_rate"
            ],
            0.25,
        )

    def test_recommendation_uses_design_labels_only(self):
        rows = [
            ScoredRow(str(i), (float(i % 2), -float(i % 2)), (bool(i % 2),))
            for i in range(50)
        ]
        options = dict(names=("candidate", "reverse"), graders=("gold",), bootstrap=0)
        original = compare_scores(rows, **options)
        design, calibration, heldout = split_rows(rows, 42)
        edited = compare_scores(
            [
                *design,
                *(
                    ScoredRow(row.id, row.scores, (not row.outcomes[0],))
                    for row in calibration + heldout
                ),
            ],
            **options,
        )
        self.assertEqual(original["recommended_score"], "candidate")
        self.assertEqual(original["recommended_score"], edited["recommended_score"])
        self.assertEqual(
            original["design_normalized_aurc_gain"],
            edited["design_normalized_aurc_gain"],
        )
        self.assertNotEqual(original["graders"], edited["graders"])
