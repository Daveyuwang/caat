import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from math import nextafter
from pathlib import Path
from tempfile import TemporaryDirectory

from caat import GradedRow, select
from caat.cli import main
from caat.metrics import Counts, ranking_metrics, threshold_points
from caat.rules import binomial_pvalue
from caat.select import design_thresholds


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
    def test_design_grid_size_uses_distinct_quantiles(self):
        scores = [0.9] * 5 + [0.8, 0.4, 0.1]
        self.assertEqual(design_thresholds(scores, grid_size=4), [0.9, 0.8, 0.1])
        self.assertEqual(
            design_thresholds(scores, grid_size=10**50), [0.9, 0.8, 0.4, 0.1]
        )
        for invalid in (0, -2, True, 4.0):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                design_thresholds(scores, grid_size=invalid)

    def test_holm_can_accept_a_later_cutoff_than_bonferroni(self):
        rows = [
            GradedRow(str(i), score, (correct,))
            for i, (score, correct) in enumerate(
                [(0.9, True)] * 12
                + [(0.8, True)] * 8
                + [(0.8, False)] * 2
                + [(0.7, False)] * 10
            )
        ]
        options = {
            "graders": ("grader",),
            "thresholds": (0.9, 0.8, 0.7),
            "risk_target": 0.3,
        }
        self.assertEqual(
            select(rows, rule="ltt-bonferroni", **options)["selected_threshold"], 0.9
        )
        report = select(rows, rule="ltt-holm", **options)
        self.assertEqual(report["selected_threshold"], 0.8)
        self.assertEqual(
            [row["passes"] for row in report["candidates"]], [True, True, False]
        )
        self.assertEqual(report["guarantee"]["confidence"], 0.95)
        self.assertIsNone(report["per_candidate_level"])

    def test_exact_binomial_tail_and_boundaries(self):
        self.assertAlmostEqual(
            binomial_pvalue(Counts(10, 5, 1), 0.25), 0.75**5 + 5 * 0.25 * 0.75**4
        )
        self.assertEqual(binomial_pvalue(Counts(10, 0, 0), 0.25), 1.0)
        self.assertEqual(binomial_pvalue(Counts(10, 5, 5), 1.0), 1.0)
        self.assertEqual(binomial_pvalue(Counts(10, 5, 4), 1.0), 0.0)
        self.assertEqual(binomial_pvalue(Counts(1074, 1074, 0), 0.5), 5e-324)
        self.assertEqual(binomial_pvalue(Counts(1075, 1075, 0), 0.5), 0.0)

    def test_empirical_rule_compares_the_worst_exact_error_ratio(self):
        rows = [GradedRow(str(i), 0.9, (True, i != 0)) for i in range(3)]
        for target, passes in ((1 / 3, False), (nextafter(1 / 3, 1.0), True)):
            with self.subTest(risk_target=target):
                report = select(
                    rows,
                    graders=("primary", "alternate"),
                    thresholds=(1.0, 0.8),
                    risk_target=target,
                    rule="empirical",
                )
                self.assertEqual(
                    [candidate["passes"] for candidate in report["candidates"]],
                    [False, passes],
                )
                self.assertEqual(report["candidates"][1]["worst_error_rate"], 1 / 3)
                self.assertEqual(report["selected_threshold"], 0.8 if passes else None)

    def test_statistical_rules_use_the_exact_binomial_boundary(self):
        for rule in ("ltt-bonferroni", "ltt-fixed-sequence", "ltt-holm"):
            for labels, level, probability, passes in (
                (3, 0.5, 0.5, True),
                (4, nextafter(0.3125, 0), 0.3125, False),
            ):
                with self.subTest(rule=rule, labels=labels):
                    rows = [
                        GradedRow(str(index), 0.9, (index != 0,))
                        for index in range(labels)
                    ]
                    report = select(
                        rows,
                        graders=("grader",),
                        thresholds=(0.8,),
                        risk_target=0.5,
                        failure_rate=level,
                        rule=rule,
                    )
                    self.assertEqual(report["candidates"][0]["pvalue"], probability)
                    self.assertIs(report["candidates"][0]["passes"], passes)
                    self.assertEqual(
                        report["selected_threshold"], 0.8 if passes else None
                    )

    def test_holm_resolves_candidates_on_opposite_sides_of_a_rounded_tie(self):
        rows = [
            GradedRow(str(index), score, (correct,))
            for index, (score, correct) in enumerate(
                [(0.9, True), (0.9, True), (0.9, False), (0.8, True), (0.8, False)]
            )
        ]
        options = {
            "graders": ("grader",),
            "thresholds": (0.9, 0.8),
            "risk_target": 0.5000000000000001,
            "failure_rate": 0.9999999999999997,
        }
        # The exact tails straddle 0.49999999999999985 and round to the same float.
        bonferroni = select(rows, rule="ltt-bonferroni", **options)
        self.assertEqual(
            [candidate["pvalue"] for candidate in bonferroni["candidates"]],
            [bonferroni["per_candidate_level"]] * 2,
        )
        self.assertEqual(
            [candidate["passes"] for candidate in bonferroni["candidates"]],
            [False, True],
        )
        holm = select(rows, rule="ltt-holm", **options)
        self.assertEqual(
            [candidate["passes"] for candidate in holm["candidates"]], [True, True]
        )

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
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(
                    main([*args, "--rule", "empirical", "--design-grid-size", "1"]), 0
                )
            result = json.loads(output.getvalue())
            self.assertEqual(result["input"]["design_grid_size"], 1)
            self.assertEqual([row["threshold"] for row in result["candidates"]], [0.6])
            design.write_text(
                "conf,tier\njunk,other\n0.6,main\n0.9,main\n", encoding="utf-8"
            )
            calibration.write_text(
                "id,conf,g,tier\na,junk,x,other\na,0.8,1,main\nb,0.2,0,main\n",
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(
                    main(
                        [
                            *args,
                            "--rule",
                            "empirical",
                            "--design-grid-size",
                            "1",
                            "--prediction-filter",
                            "tier=main",
                        ]
                    ),
                    0,
                )
            filtered = json.loads(output.getvalue())
            self.assertEqual(filtered["candidates"], result["candidates"])
            self.assertEqual(
                filtered["input"]["prediction_filter"],
                {"column": "tier", "value": "main"},
            )
            with (
                redirect_stderr(io.StringIO()) as error,
                self.assertRaises(SystemExit) as rejected,
            ):
                main([*args[:-2], "--thresholds", "0.6", "--design-grid-size", "1"])
            self.assertEqual(rejected.exception.code, 2)
            self.assertIn("--design-grid-size requires --design", error.getvalue())

    def test_design_checks_reuse_after_filtering_without_fit_tracking(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration.csv"
            design = root / "design.jsonl"
            output = root / "cutoff.json"
            calibration.write_text(
                "sample,score,output_key,correct,tier\na,0.9,old,1,main\nb,0.4,b,0,main\n",
                encoding="utf-8",
            )
            source = (
                '{"sample":"a","score":null,"output_key":"old","tier":"other"}\n'
                '{"sample":"a","score":0.7,"output_key":"new","tier":"main"}\n'
            )
            design.write_text(source, encoding="utf-8")
            args = [
                "select",
                str(calibration),
                "--design",
                str(design),
                "--id-column",
                "sample",
                "--correct-column",
                "correct",
                "--prediction-filter",
                "tier=main",
                "--risk-target",
                "0.2",
                "--rule",
                "empirical",
                "--output",
                str(output),
            ]
            keyed = [*args, "--output-key-column", "output_key"]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(keyed), 0)
            saved = output.read_bytes()
            self.assertEqual(json.loads(saved)["selected_threshold"], 0.7)

            for data, flags in (
                (source, args),
                (source.replace('"new"', '"old"'), keyed),
            ):
                design.write_text(data, encoding="utf-8")
                with (
                    redirect_stderr(io.StringIO()) as error,
                    self.assertRaises(SystemExit) as rejected,
                ):
                    main(flags)
                self.assertEqual(rejected.exception.code, 2)
                self.assertIn(
                    "design and calibration reuse sample ID 'a'", error.getvalue()
                )
                self.assertEqual(output.read_bytes(), saved)

    def test_lower_direction_uses_original_units_for_design_and_explicit_cutoffs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration, design = root / "calibration.csv", root / "design.csv"
            calibration.write_text(
                "id,loss,correct\na,0.3,1\nb,0.9,0\n", encoding="utf-8"
            )
            design.write_text("loss\n0.4\n0.8\n", encoding="utf-8")
            args = [
                "select",
                str(calibration),
                "--score-column",
                "loss",
                "--direction",
                "lower",
                "--correct-column",
                "correct",
                "--risk-target",
                "0.2",
                "--rule",
                "empirical",
                "--json",
            ]
            results = []
            for grid in (("--design", str(design)), ("--thresholds", "0.4", "0.8")):
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main([*args, *grid]), 0)
                report = json.loads(output.getvalue())
                self.assertEqual(report["selected_threshold"], -0.8)
                self.assertEqual(report["input"]["score_column"], "negative_loss")
                self.assertEqual(report["input"]["negated_score_source"], "loss")
                self.assertEqual(
                    [candidate["threshold"] for candidate in report["candidates"]],
                    [-0.4, -0.8],
                )
                results.append(report["candidates"])
            self.assertEqual(results[0], results[1])

    def test_saved_fit_identities_check_independent_holdout_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration, design, holdout, cutoff = (
                root / name
                for name in ("cal.csv", "design.csv", "test.csv", "cutoff.json")
            )
            calibration.write_text(
                "id,score,key,g\nc,0.9,calibration,1\n", encoding="utf-8"
            )
            design.write_text("id,score,key\nd,0.8,design\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "select",
                            str(calibration),
                            "--design",
                            str(design),
                            "--correct-column",
                            "g",
                            "--risk-target",
                            "0.9",
                            "--rule",
                            "empirical",
                            "--output-key-column",
                            "key",
                            "--track-fit-outputs",
                            "--output",
                            str(cutoff),
                        ]
                    ),
                    0,
                )
            saved = json.loads(cutoff.read_bytes())
            self.assertEqual(len(saved["input"]["fit_outputs"]["sha256"]), 2)
            calibration.unlink()
            design.unlink()

            header = "id,score,key,g,alternate\n"
            independent = "z,0.9,new-z,1,1\n"
            args = [
                str(holdout),
                "--threshold-report",
                str(cutoff),
                "--correct-column",
                "g",
                "--holdout-confidence",
                "0.95",
                "--json",
            ]
            holdout.write_text(
                header + "c,0.9,new-output,1,1\n" + independent, encoding="utf-8"
            )
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["evaluate", *args]), 0)
            result = json.loads(output.getvalue())
            self.assertTrue(result["input"]["fit_outputs_checked"])

            for reused in ("c,0.1,calibration,0,0\n", "d,0.1,design,0,0\n"):
                with self.subTest(row=reused):
                    holdout.write_text(header + independent + reused, encoding="utf-8")
                    with (
                        redirect_stderr(io.StringIO()) as errors,
                        self.assertRaises(SystemExit) as rejected,
                    ):
                        main(["audit", *args, "--correct-column", "alternate"])
                    self.assertEqual(rejected.exception.code, 2)
                    self.assertIn(
                        "holdout reuses output from the fit", errors.getvalue()
                    )
