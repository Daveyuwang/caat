import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from math import ceil, nextafter
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from caat.audit import percentile_interval
from caat.cli import main
from caat.metrics import risk_at_count, threshold_points
from caat.policy import split_rows
from caat.scores import ScoredRow, compare_scores, paired_risk_intervals


class ScoreTests(unittest.TestCase):
    def test_compare_matches_separate_grades_before_ranking(self):
        with (
            TemporaryDirectory() as directory,
            redirect_stdout(io.StringIO()) as stdout,
        ):
            root = Path(directory)
            scores, primary, alternate = (
                root / name for name in ("scores.csv", "primary.csv", "alternate.jsonl")
            )
            scores.write_text(
                "answer_id,confidence,loss,answer_key,tier\n"
                "a,unused,unused,old-a,old\na,0.9,0.1,new-a,live\n"
                "b,0.8,0.2,b,live\nc,0.3,,c,live\n",
                encoding="utf-8",
            )
            primary.write_text(
                "id,graded_key,correct\nc,c,0\nb,b,1\na,old-a,0\na,new-a,1\n",
                encoding="utf-8",
            )
            alternate.write_text(
                '{"id":"a","graded_key":"new-a","correct":false}\n'
                '{"id":"b","graded_key":"b","correct":true}\n'
                '{"id":"c","graded_key":"c","correct":false}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                main(
                    [
                        "scores",
                        str(scores),
                        "--id-column",
                        "answer_id",
                        "--output-key-column",
                        "answer_key",
                        "--grader-id-column",
                        "id",
                        "--grader-output-key-column",
                        "graded_key",
                        "--grader-file",
                        f"primary={primary}",
                        "--grader-file",
                        f"alternate={alternate}",
                        "--allow-extra-grades",
                        "--prediction-filter",
                        "tier=live",
                        "--review-missing-scores",
                        "--score-column",
                        "confidence",
                        "--score-column",
                        "loss",
                        "--lower-is-better",
                        "loss",
                        "--split",
                        "all",
                        "--coverage",
                        "0.5",
                        "--bootstrap",
                        "0",
                        "--json",
                    ]
                ),
                0,
            )
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["split"]["reported"], 3)
            self.assertEqual(
                result["input"]["grader_files"],
                {"primary": str(primary), "alternate": str(alternate)},
            )
            for name, risk in (("primary", 0.0), ("alternate", 0.5)):
                loss = result["graders"][name]["scores"]["loss"]
                self.assertEqual(loss["scored"], 2)
                self.assertEqual(loss["budgets"][0]["error_rate"], risk)

    def test_partial_scores_use_full_budgets_and_only_joint_feasible_draws(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "scores.jsonl"
            loss = (None, 0.4, 0.1, None)
            confidence = (0.7, 0.4, 0.3, 0.0)
            errors = (False, True, False, True)
            source.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": str(i),
                            "loss": loss[i],
                            "confidence": confidence[i],
                            "empty": None,
                            "correct": not errors[i],
                        }
                    )
                    for i in range(4)
                ),
                encoding="utf-8",
            )
            args = [
                "scores",
                str(source),
                "--score-column",
                "loss",
                "--score-column",
                "confidence",
                "--score-column",
                "empty",
                "--baseline",
                "confidence",
                "--lower-is-better",
                "loss",
                "--correct-column",
                "correct",
                "--split",
                "all",
                "--coverage",
                "0.5",
                "--coverage",
                "0.75",
                "--bootstrap",
                "31",
                "--seed",
                "53",
                "--json",
            ]
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)

            differences = []
            rng = Random(53)
            for _ in range(31):
                draw = rng.choices(range(4), k=4)
                available = [(-loss[i], errors[i]) for i in draw if loss[i] is not None]
                if len(available) >= 2:
                    baseline = threshold_points(
                        (confidence[i], errors[i]) for i in draw
                    )
                    differences.append(
                        risk_at_count(threshold_points(available), 2)
                        - risk_at_count(baseline, 2)
                    )
            self.assertLess(len(differences), 31)
            for workers in (1, 3):
                with (
                    self.subTest(workers=workers),
                    redirect_stdout(io.StringIO()) as stdout,
                ):
                    self.assertEqual(
                        main(
                            [
                                *args,
                                "--review-missing-scores",
                                "--workers",
                                str(workers),
                            ]
                        ),
                        0,
                    )
                    report = json.loads(stdout.getvalue())
                    self.assertEqual(report["split"]["reported"], 4)
                    candidates = report["graders"]["correct"]["scores"]
                    partial = candidates["loss"]
                    self.assertEqual(
                        (partial["scored"], partial["maximum_coverage"]), (2, 0.5)
                    )
                    self.assertEqual(partial["auroc"], 1.0)
                    half, three_quarters = partial["budgets"]
                    self.assertEqual(
                        (half["accepted"], half["error_rate"], half["risk_delta"]),
                        (2, 0.5, 0.0),
                    )
                    self.assertEqual(half["risk_delta_paired_draws"], len(differences))
                    self.assertEqual(
                        half["risk_delta_95"], percentile_interval(differences)
                    )
                    for key in (
                        "accepted",
                        "coverage",
                        "error_rate",
                        "risk_delta",
                        "risk_delta_95",
                    ):
                        self.assertIsNone(three_quarters[key])
                    self.assertEqual(candidates["empty"]["scored"], 0)
                    self.assertIsNone(candidates["empty"]["aurc"])

    def test_decimal_coverage_uses_the_same_row_count_for_risk_and_bootstrap(self):
        rows = [
            ScoredRow(f"{i:02}", (i, 1.0 if i < 7 else None), (i >= 7,))
            for i in range(50)
        ]
        result = compare_scores(
            rows,
            names=("full", "partial"),
            graders=("correct",),
            split="all",
            coverages=(0.14, nextafter(0.14, 1.0)),
            review_missing_scores=True,
            bootstrap=60,
            seed=53,
        )
        scores = result["graders"]["correct"]["scores"]
        self.assertEqual(
            [entry["accepted"] for entry in scores["full"]["budgets"]], [7, 8]
        )
        exact, above = scores["partial"]["budgets"]
        self.assertEqual(
            (exact["accepted"], exact["coverage"], exact["error_rate"]), (7, 0.14, 1.0)
        )
        self.assertIsNone(above["accepted"])
        self.assertIsNone(above["risk_delta_95"])

        rng = Random(53)
        available = [
            sum(i < 7 for i in rng.choices(range(50), k=50)) for _ in range(60)
        ]
        expected = [sum(size >= target for size in available) for target in (7, 8)]
        self.assertGreater(expected[0], expected[1])
        self.assertEqual(
            [entry["risk_delta_paired_draws"] for entry in (exact, above)], expected
        )

    def test_cli_lower_score_uses_the_original_baseline_and_cost_names(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "scores.csv"
            source.write_text(
                "id,loss,confidence,correct\na,0.1,0.1,1\nb,0.2,0.2,1\nc,0.9,0.8,0\n",
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()) as output:
                status = main(
                    [
                        "scores",
                        str(source),
                        "--score-column",
                        "loss",
                        "--score-column",
                        "confidence",
                        "--lower-is-better",
                        "loss",
                        "--baseline",
                        "loss",
                        "--score-cost",
                        "loss=external",
                        "--correct-column",
                        "correct",
                        "--split",
                        "all",
                        "--coverage",
                        "0.5",
                        "--bootstrap",
                        "0",
                        "--json",
                    ]
                )
            self.assertEqual(status, 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["baseline"], "loss")
            self.assertEqual(report["input"]["lower_is_better"], ["loss"])
            self.assertEqual(report["costs"]["loss"], "external")
            scores = report["graders"]["correct"]["scores"]
            self.assertEqual(scores["loss"]["budgets"][0]["error_rate"], 0)
            self.assertEqual(scores["confidence"]["budgets"][0]["error_rate"], 0.5)

    def test_rank_reuse_matches_resorting_seeded_draws(self):
        scores = (
            (5, 4, 4, 3, 1, 1, 0, -2),
            (0, 0, -1, 4, 3, 3, 0, 1),
            (1,) * 8,
        )
        errors = (True, False, True, False, True, False, False, True)
        coverages = (0.01, 0.32, 0.51, 1.0)
        targets = [ceil(len(errors) * c) for c in coverages]
        reference = 1
        replicates = 31
        rng = Random(53)
        differences = [[[] for _ in targets] for _ in scores]
        for _ in range(replicates):
            draw = rng.choices(range(len(errors)), k=len(errors))
            risks = []
            for column in scores:
                points = threshold_points((column[i], errors[i]) for i in draw)
                risks.append([risk_at_count(points, target) for target in targets])
            for column_deltas, column_risks in zip(differences, risks, strict=True):
                for deltas, risk, baseline in zip(
                    column_deltas, column_risks, risks[reference], strict=True
                ):
                    deltas.append(risk - baseline)
        expected = [
            [percentile_interval(deltas) for deltas in column] for column in differences
        ]
        for workers in (1, 3):
            with self.subTest(workers=workers):
                self.assertEqual(
                    paired_risk_intervals(
                        scores,
                        errors,
                        reference,
                        coverages,
                        replicates,
                        53,
                        workers=workers,
                    ),
                    expected,
                )

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

        partial = compare_scores(
            [
                ScoredRow(
                    row.id,
                    (row.scores[0], None if row == design[0] else row.scores[0]),
                    row.outcomes,
                )
                for row in rows
            ],
            **{**options, "names": ("candidate", "sometimes")},
            review_missing_scores=True,
        )
        self.assertEqual(partial["recommended_score"], "candidate")
        self.assertEqual(
            partial["design_scored"],
            {"candidate": len(design), "sometimes": len(design) - 1},
        )
        self.assertIsNone(partial["design_normalized_aurc_gain"]["sometimes"])
