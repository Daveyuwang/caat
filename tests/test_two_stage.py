from itertools import combinations
from math import comb
import unittest

from caat.two_stage import design


class TwoStageTests(unittest.TestCase):
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
