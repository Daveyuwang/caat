from itertools import combinations
import unittest

from caat.budget import compare_budgets
from caat.plan import sample_error_limits
from caat.two_stage import checkpoint_design


class BudgetTests(unittest.TestCase):
    def test_small_search_against_enumerated_draws(self):
        size = 10
        ranks = []
        draws = [list(combinations(range(size), errors)) for errors in (1, 6)]

        def assess(cap, stop):
            costs, requests = [], []
            for samples, direction in zip(draws, ("meets", "exceeds")):
                outcomes = [stop(sample) for sample in samples]
                if sum(v == direction for v, _, _ in outcomes) < 0.7 * len(samples):
                    return
                costs.append(sum(n + r for _, n, r in outcomes) / len(samples))
                requests.append(sum(r for _, _, r in outcomes) / len(samples))
            ranks.append((cap, max(costs), max(requests)))

        def verdict(limits, errors):
            low, high = limits["max_errors_to_meet"], limits["min_errors_to_exceed"]
            if low is not None and errors <= low:
                return "meets"
            if high is not None and errors >= high:
                return "exceeds"
            return "inconclusive"

        for cap in range(1, size + 1):
            limits = sample_error_limits(size, cap, 0.3, 0.85)
            for block in range(1, cap + 1):

                def fixed(sample):
                    for r, start in enumerate(range(0, cap, block), 1):
                        n = min(start + block, cap)
                        x = sum(i < n for i in sample)
                        v = verdict(limits, x)
                        if v == verdict(limits, x + cap - n):
                            return v, n, r

                assess(cap, fixed)
            for first in range(1, cap):
                design = checkpoint_design(
                    size, cap, first, 0.3, 0.85, [0.1, 0.6], 0.7, 1
                )

                def staged(sample):
                    for r, stage in enumerate(design["stages"], 1):
                        n = stage["labels"]
                        v = verdict(stage, sum(i < n for i in sample))
                        if v != "inconclusive" or r == 2:
                            return v, n, r

                assess(cap, staged)
        result = compare_budgets(size, 15, 0.3, 0.85, [0.1, 0.6], 0.7, 1)
        self.assertEqual(result["caps_tested"], list(range(1, 11)))
        for key, rank in (
            ("smallest-cap", min(ranks)),
            ("lowest-cost", min(ranks, key=lambda x: (x[1], x[0], x[2]))),
        ):
            offer = result["offers"][key]
            self.assertEqual(offer["cap"], rank[0])
            self.assertAlmostEqual(offer["worst_cost"], rank[1])
            self.assertAlmostEqual(
                max(s["expected_requests"] for s in offer["scenarios"]), rank[2]
            )
        failed = compare_budgets(size, 1, 0.3, 0.95, [0.1, 0.6], 0.8, 1)
        self.assertEqual((failed["caps_tested"], failed["offers"]), ([1], {}))

    def test_archived_population_costs(self):
        for size, target, rates, base, best in (
            (1276, 0.2, [0.15, 0.25], (321, 292.90), (350, 283.16)),
            (1458, 0.05, [0.02, 0.08], (300, 268.30), (325, 261.46)),
            (667, 0.2, [0.15, 0.25], (261, 237.69), (275, 225.94)),
        ):
            report = compare_budgets(size, 500, target, 0.95, rates, 0.8, 1)
            self.assertEqual(report["minimum_single_stage_cap"], base[0])
            for key, (cap, cost) in zip(("smallest-cap", "lowest-cost"), (base, best)):
                offer = report["offers"][key]
                self.assertEqual(offer["cap"], cap)
                self.assertAlmostEqual(offer["worst_cost"], cost, delta=0.01)
