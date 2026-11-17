import unittest
from math import comb, fsum
from statistics import mean

from hypothesis import given, settings, strategies as st

from caat.metrics import (
    Counts,
    calibration_metrics,
    ranking_metrics,
    risk_at_count,
    threshold_points,
)
from caat.rules import binomial_pvalue, binomial_upper_bound


class MetricProperties(unittest.TestCase):
    @settings(max_examples=100, derandomize=True, deadline=None)
    @given(
        st.lists(st.tuples(st.integers(-3, 3), st.booleans()), min_size=1, max_size=30)
    )
    def test_ranking_and_every_budget_match_direct_tie_counts(self, rows):
        points = threshold_points(rows)
        self.assertEqual(points, threshold_points(reversed(rows)))
        for threshold, counts in points:
            accepted = [is_error for score, is_error in rows if score >= threshold]
            self.assertEqual(counts, Counts(len(rows), len(accepted), sum(accepted)))

        expected_risks = []
        scores = sorted((score for score, _ in rows), reverse=True)
        for count, boundary in enumerate(scores, 1):
            above = [is_error for score, is_error in rows if score > boundary]
            tied = [is_error for score, is_error in rows if score == boundary]
            expected = (sum(above) + (count - len(above)) * mean(tied)) / count
            expected_risks.append(expected)
            self.assertAlmostEqual(risk_at_count(points, count), expected, places=13)

        summary = ranking_metrics(points)
        self.assertAlmostEqual(summary["aurc"], mean(expected_risks), places=13)
        correct = [score for score, is_error in rows if not is_error]
        wrong = [score for score, is_error in rows if is_error]
        if correct and wrong:
            auc = mean(
                (good > bad) + 0.5 * (good == bad) for good in correct for bad in wrong
            )
            self.assertAlmostEqual(summary["auroc"], auc, places=13)
        else:
            self.assertIsNone(summary["auroc"])

    @settings(max_examples=100, derandomize=True, deadline=None)
    @given(
        st.lists(st.tuples(st.integers(0, 10), st.booleans()), min_size=1, max_size=40),
        st.integers(1, 15),
    )
    def test_calibration_bins_conserve_rows_and_reproduce_ece(self, rows, bins):
        ticks, outcomes = zip(*rows, strict=True)
        probabilities = [tick / 10 for tick in ticks]
        report = calibration_metrics(probabilities, outcomes, bins=bins)
        expected_brier = sum((tick - 10 * y) ** 2 for tick, y in rows) / (
            100 * len(rows)
        )
        self.assertAlmostEqual(report["brier_score"], expected_brier, places=13)
        reliability = report["reliability_bins"]
        self.assertEqual(len(reliability), bins)
        self.assertEqual(sum(group["count"] for group in reliability), len(rows))
        expected_ece = fsum(
            group["count"] * abs(group["mean_probability"] - group["accuracy"])
            for group in reliability
            if group["count"]
        ) / len(rows)
        self.assertAlmostEqual(report["ece"], expected_ece, places=13)
        reversed_report = calibration_metrics(
            reversed(probabilities), reversed(outcomes), bins=bins
        )
        self.assertEqual(report, reversed_report)

    @settings(max_examples=100, derandomize=True, deadline=None)
    @given(st.lists(st.booleans(), min_size=1, max_size=60), st.integers(0, 100))
    def test_exact_binomial_tail_and_bound_match_direct_summation(
        self, errors, percent
    ):
        n, k, p = len(errors), sum(errors), percent / 100
        counts = Counts(n, n, k)

        def cdf(probability):
            return fsum(
                comb(n, i) * probability**i * (1 - probability) ** (n - i)
                for i in range(k + 1)
            )

        self.assertAlmostEqual(binomial_pvalue(counts, p), cdf(p), places=12)
        upper = binomial_upper_bound(counts, 0.95)
        if k == n:
            self.assertEqual(upper, 1.0)
        else:
            self.assertAlmostEqual(cdf(upper), 0.05, places=12)
