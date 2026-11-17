import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from math import comb, nextafter
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.cli import main
from caat.finite import (
    compare_hypergeometric_cdf,
    hypergeometric_cdf,
    hypergeometric_cdf_float,
)
from caat.plan import plan_labels


class LabelPlanTests(unittest.TestCase):
    def test_routing_and_sample_source_reuse_tracked_identity_columns(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            fit, cutoff, predictions, batch, plan, sample = (
                root / name
                for name in (
                    "fit.csv",
                    "cutoff.json",
                    "predictions.jsonl",
                    "decisions.csv",
                    "plan.json",
                    "sample.csv",
                )
            )
            fit.write_text(
                "answer_id,confidence,answer_key,past\nfit,0.8,fit-key,1\n",
                encoding="utf-8",
            )
            predictions.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "answer_id": item,
                            "confidence": 0.9,
                            "answer_key": f"key-{item}",
                            "completion": f"answer {item}",
                        }
                    )
                    for item in ("a", "b", "c")
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                main(
                    [
                        "select",
                        str(fit),
                        "--id-column",
                        "answer_id",
                        "--score-column",
                        "confidence",
                        "--correct-column",
                        "past",
                        "--output-key-column",
                        "answer_key",
                        "--track-fit-outputs",
                        "--thresholds",
                        "0.8",
                        "--rule",
                        "empirical",
                        "--risk-target",
                        "0.5",
                        "--output",
                        str(cutoff),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "route",
                        str(predictions),
                        "--threshold-report",
                        str(cutoff),
                        "--output",
                        str(batch),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "plan",
                        "--threshold-report",
                        str(cutoff),
                        "--expected-error",
                        "0",
                        "--batch-decisions",
                        str(batch),
                        "--sample-output",
                        str(sample),
                        "--sample-source",
                        str(predictions),
                        "--sample-column",
                        "completion",
                        "--output",
                        str(plan),
                    ]
                ),
                0,
            )
            metadata = json.loads(plan.read_bytes())["input"]
            self.assertEqual(metadata["sample_source_id_column"], "answer_id")
            self.assertEqual(metadata["sample_source_output_key_column"], "answer_key")
            with sample.open(encoding="utf-8", newline="") as source:
                self.assertEqual(
                    [
                        (row["id"], row["output_key"], row["completion"])
                        for row in csv.DictReader(source)
                    ],
                    [("b", "key-b", "answer b"), ("a", "key-a", "answer a")],
                )

    def test_sample_source_selects_exact_output_versions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            batch, source, plan, sample = (
                root / name
                for name in ("batch.csv", "source.jsonl", "plan.json", "sample.csv")
            )
            batch.write_text(
                "id,score,decision,output_key\na,0.88,accept,key-a\n"
                "b,0.83,accept,key-b\nc,0.80,accept,key-c\n",
                encoding="utf-8",
            )
            exact = {"sample_id": "a", "answer_key": "key-a", "completion": "one\rtwo"}
            other = {"sample_id": "b", "answer_key": "key-b", "completion": "answer b"}
            stale = {**exact, "answer_key": "old-a", "completion": "old answer"}
            source.write_text(
                "\n".join(map(json.dumps, (stale, exact, other))), encoding="utf-8"
            )
            args = [
                "plan",
                "--risk-target",
                "0.5",
                "--expected-error",
                "0",
                "--batch-decisions",
                str(batch),
                "--sample-output",
                str(sample),
                "--output",
                str(plan),
                "--sample-source",
                str(source),
                "--id-column",
                "sample_id",
                "--output-key-column",
                "answer_key",
                "--sample-column",
                "completion",
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            with sample.open(encoding="utf-8", newline="") as content:
                reader = csv.DictReader(content)
                self.assertEqual(
                    reader.fieldnames,
                    ["id", "score", "group", "output_key", "completion"],
                )
                self.assertEqual(
                    [
                        (row["id"], row["output_key"], row["completion"])
                        for row in reader
                    ],
                    [("b", "key-b", "answer b"), ("a", "key-a", "one\rtwo")],
                )
            previous = plan.read_bytes(), sample.read_bytes()
            for records, message in (
                (
                    (stale, other),
                    "no source row matching the output key for sampled ID 'a'",
                ),
                ((exact, other, other), "multiple source rows for sampled ID 'b'"),
            ):
                with self.subTest(message=message):
                    source.write_text(
                        "\n".join(map(json.dumps, records)), encoding="utf-8"
                    )
                    with (
                        redirect_stderr(io.StringIO()) as stderr,
                        self.assertRaises(SystemExit) as stopped,
                    ):
                        main(args)
                    self.assertEqual(stopped.exception.code, 2)
                    self.assertIn(message, stderr.getvalue())
                    self.assertEqual((plan.read_bytes(), sample.read_bytes()), previous)

    def test_plan_exports_and_verifies_a_seeded_sample(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            batch, plan, sample, labeled, audit = (
                root / name
                for name in (
                    "batch.csv",
                    "plan.json",
                    "sample.csv",
                    "labels.csv",
                    "audit.json",
                )
            )
            batch.write_text(
                "id,score,decision,group,output_key\n"
                "c,0.80,accept,tier,key-c\na,0.88,accept,first,key-a\n"
                "b,0.83,accept,second,key-b\nd,0.40,review,third,key-d\n",
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "plan",
                            "--risk-target",
                            "0.5",
                            "--expected-error",
                            "0",
                            "--batch-decisions",
                            str(batch),
                            "--sample-output",
                            str(sample),
                            "--output",
                            str(plan),
                        ]
                    ),
                    0,
                )
            report = json.loads(plan.read_bytes())
            self.assertEqual(report["input"]["sample_seed"], 42)
            self.assertEqual(report["sample_path"], str(sample))
            with sample.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(
                    list(csv.reader(stream)),
                    [
                        ["id", "score", "group", "output_key"],
                        ["b", "0.83", "second", "key-b"],
                        ["a", "0.88", "first", "key-a"],
                    ],
                )
            fields = "id,score,output_key,correct\n"
            labeled.write_text(
                fields + "a,0.88,key-a,1\nb,0.83,key-b,1\n", encoding="utf-8"
            )
            args = [
                "evaluate",
                str(labeled),
                "--threshold",
                "0.75",
                "--correct-column",
                "correct",
                "--batch-decisions",
                str(batch),
                "--plan-report",
                str(plan),
                "--output",
                str(audit),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            original = audit.read_bytes()
            self.assertEqual(json.loads(original)["input"]["sample_seed"], 42)
            labeled.write_text(
                fields + "b,0.83,key-b,1\nc,0.80,key-c,1\n", encoding="utf-8"
            )
            with (
                redirect_stderr(io.StringIO()) as stderr,
                self.assertRaises(SystemExit) as stopped,
            ):
                main(args)
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("does not match the draw for seed 42", stderr.getvalue())
            self.assertEqual(audit.read_bytes(), original)

    def test_planned_score_uses_the_cutoff_scale_even_if_source_used_score(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fit, cutoff, batch, plan, sample, copied, raw = (
                root / name
                for name in (
                    "fit.csv",
                    "cutoff.json",
                    "batch.csv",
                    "plan.json",
                    "sample.csv",
                    "annotated.csv",
                    "raw.csv",
                )
            )
            fit.write_text(
                "id,score,output_key,correct\nfit,0.05,k-fit,1\n", encoding="utf-8"
            )
            batch.write_text(
                "id,score,decision,output_key\na,-0.1,accept,k-a\nb,-0.2,accept,k-b\nc,-0.3,accept,k-c\n",
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "select",
                            str(fit),
                            "--correct-column",
                            "correct",
                            "--direction",
                            "lower",
                            "--output-key-column",
                            "output_key",
                            "--track-fit-outputs",
                            "--thresholds",
                            "0.5",
                            "--risk-target",
                            "0.5",
                            "--rule",
                            "empirical",
                            "--output",
                            str(cutoff),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "plan",
                            "--threshold-report",
                            str(cutoff),
                            "--batch-decisions",
                            str(batch),
                            "--expected-error",
                            "0",
                            "--output",
                            str(plan),
                            "--sample-output",
                            str(sample),
                        ]
                    ),
                    0,
                )
            lines = sample.read_text(encoding="utf-8").splitlines()
            labeled = (
                "\n".join([lines[0] + ",correct", *(line + ",1" for line in lines[1:])])
                + "\n"
            )
            sample.write_text(labeled, encoding="utf-8")
            copied.write_text(labeled, encoding="utf-8")
            raw.write_text(labeled.replace(",-0.", ",0."), encoding="utf-8")
            reports = []
            for path in (sample, copied, raw):
                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertEqual(
                        main(
                            [
                                "evaluate",
                                str(path),
                                "--threshold-report",
                                str(cutoff),
                                "--batch-decisions",
                                str(batch),
                                "--plan-report",
                                str(plan),
                                "--correct-column",
                                "correct",
                                "--json",
                            ]
                        ),
                        0,
                    )
                reports.append(json.loads(stdout.getvalue()))
            oriented, copied_report, original = reports
            self.assertEqual(oriented["overall"], original["overall"])
            self.assertEqual(copied_report["overall"], oriented["overall"])
            self.assertEqual(oriented["overall"]["accepted"], 2)
            self.assertTrue(oriented["input"]["fit_outputs_checked"])
            self.assertEqual(original["input"]["negated_score_source"], "score")
            self.assertNotIn("negated_score_source", oriented["input"])
            self.assertNotIn("negated_score_source", copied_report["input"])

    def test_plan_binds_all_accepted_scores_and_keys(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            batch, plan, sample = (
                root / name for name in ("decisions.csv", "plan.json", "sample.csv")
            )
            records = (
                "id,score,decision,output_key\n"
                "a,0.88,accept,key-a\nb,0.83,accept,key-b\nc,0.80,accept,key-c\n"
                "d,0.40,review,key-d\n"
            )
            batch.write_text(records, encoding="utf-8")
            sample.write_text(
                "id,score,output_key,correct\na,0.88,key-a,1\nb,0.83,key-b,1\n",
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "plan",
                            "--risk-target",
                            "0.5",
                            "--expected-error",
                            "0",
                            "--batch-decisions",
                            str(batch),
                            "--output",
                            str(plan),
                        ]
                    ),
                    0,
                )
            payload = json.loads(plan.read_bytes())
            self.assertEqual(payload["accepted"], 2)
            self.assertRegex(
                payload["input"]["accepted_population_sha256"], r"^[0-9a-f]{64}$"
            )
            args = [
                "evaluate",
                str(sample),
                "--threshold",
                "0.75",
                "--correct-column",
                "correct",
                "--output-key-column",
                "output_key",
                "--plan-report",
                str(plan),
                "--json",
            ]
            batch_args = ["--batch-decisions", str(batch)]
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main([*args, *batch_args]), 0)
            self.assertTrue(
                json.loads(stdout.getvalue())["input"]["accepted_population_checked"]
            )
            for changed in (
                records.replace("key-c", "other"),
                records.replace("c,0.80", "c,0.81"),
            ):
                with self.subTest(changed=changed):
                    batch.write_text(changed, encoding="utf-8")
                    with (
                        redirect_stderr(io.StringIO()) as stderr,
                        self.assertRaises(SystemExit) as stopped,
                    ):
                        main([*args, *batch_args])
                    self.assertEqual(stopped.exception.code, 2)
                    self.assertIn(
                        "batch accepted outputs do not match the label plan",
                        stderr.getvalue(),
                    )
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(args), 0)
            self.assertNotIn(
                "accepted_population_checked", json.loads(stdout.getvalue())["input"]
            )

    def test_evaluate_uses_a_saved_label_plan(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            labels = root / "sample.csv"
            labels.write_text("id,score,correct\na,0.9,1\nb,0.8,1\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "plan",
                            "--accepted",
                            "2",
                            "--expected-error",
                            "0",
                            "--accepted-population",
                            "3",
                            "--holdout-confidence",
                            "0.90",
                            "--output",
                            str(plan),
                        ]
                    ),
                    0,
                )
            args = [
                "evaluate",
                str(labels),
                "--threshold",
                "0.75",
                "--plan-report",
                str(plan),
                "--correct-column",
                "correct",
                "--json",
            ]
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(args), 0)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["accepted_population"], 3)
            self.assertEqual(result["risk_target"], nextafter(1 / 3, 1.0))
            self.assertEqual(
                result["overall"]["risk_upper_bound"], result["risk_target"]
            )
            self.assertEqual(result["input"]["planned_labels"], 2)
            original = plan.read_bytes()
            for flags, message in (
                (
                    ["--risk-target", "0.5"],
                    "--risk-target does not match the label plan",
                ),
                (
                    ["--holdout-confidence", "0.95"],
                    "--holdout-confidence does not match the label plan",
                ),
                (
                    ["--accepted-population", "4"],
                    "accepted population does not match the label plan",
                ),
                (["--output", str(plan)], "would overwrite an input file"),
            ):
                with self.subTest(flags=flags):
                    with (
                        redirect_stderr(io.StringIO()) as stderr,
                        self.assertRaises(SystemExit) as stopped,
                    ):
                        main([*args, *flags])
                    self.assertEqual(stopped.exception.code, 2)
                    self.assertIn(message, stderr.getvalue())
            self.assertEqual(plan.read_bytes(), original)

    def test_minimum_labels_match_exact_binomial_reference(self):
        for assumed, count, errors, probability in (
            (0.05, 179, 11, 0.812940653347),
            (0.07, 549, 43, 0.803776194929),
        ):
            with self.subTest(expected_error=assumed):
                plan = plan_labels(risk_target=0.1, expected_error=assumed)
                self.assertEqual(plan["accepted"], count)
                self.assertEqual(plan["max_errors"], errors)
                self.assertAlmostEqual(plan["certification_probability"], probability)
                before = plan_labels(
                    risk_target=0.1, expected_error=assumed, max_accepted=count - 1
                )
                self.assertIsNone(before["accepted"])

    def test_minimum_binomial_plan_includes_the_exact_test_boundary(self):
        options = dict(risk_target=0.5, expected_error=0, holdout_confidence=0.875)
        plan = plan_labels(**options)
        self.assertEqual((plan["accepted"], plan["max_errors"]), (3, 0))
        self.assertEqual(plan["certification_probability"], 1.0)
        self.assertIsNone(plan_labels(max_accepted=2, **options)["accepted"])

    def test_fixed_budget_inverts_the_holdout_bound(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(
                ["plan", "--accepted", "179", "--expected-error", "0.05", "--json"]
            )
        result = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(result["max_errors"], 11)
        self.assertAlmostEqual(result["risk_target"], 0.0996685894533419)
        self.assertGreaterEqual(result["certification_probability"], 0.8)

    def test_cli_reports_infeasible_budget(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(
                [
                    "plan",
                    "--risk-target",
                    "0.1",
                    "--expected-error",
                    "0.07",
                    "--max-accepted",
                    "500",
                    "--json",
                ]
            )
        self.assertEqual(status, 1)
        self.assertIsNone(json.loads(output.getvalue())["accepted"])

    def test_fixed_population_plan_matches_exact_sample_counts(self):
        options = {"accepted_population": 101, "expected_error": 0.2}
        plan = plan_labels(risk_target=0.4, **options)
        self.assertEqual(plan["expected_population_errors"], 21)
        self.assertEqual((plan["accepted"], plan["max_errors"]), (28, 7))
        self.assertAlmostEqual(plan["certification_probability"], 0.8220609221253428)
        self.assertIsNone(
            plan_labels(risk_target=0.4, max_accepted=27, **options)["accepted"]
        )
        self.assertLess(hypergeometric_cdf(101, 21, 29, 7), Fraction(4, 5))
        self.assertGreater(hypergeometric_cdf(101, 41, 29, 8), Fraction(1, 20))
        self.assertEqual(plan_labels(risk_target=0.4, max_accepted=29, **options), plan)
        fixed = plan_labels(accepted=28, **options)
        self.assertEqual(fixed["risk_target"], 39 / 101)
        self.assertEqual(fixed["max_errors"], 7)
        census = plan_labels(accepted=101, **options)
        self.assertEqual(census["risk_target"], nextafter(21 / 101, 1.0))
        self.assertEqual(census["certification_probability"], 1.0)

    def test_fixed_population_plans_use_exact_decimal_rates(self):
        options = {"accepted_population": 3, "expected_error": 0}
        below = plan_labels(risk_target=1 / 3, **options)
        self.assertEqual((below["accepted"], below["max_errors"]), (3, 0))
        self.assertIsNone(
            plan_labels(risk_target=1 / 3, max_accepted=2, **options)["accepted"]
        )
        above = plan_labels(risk_target=nextafter(1 / 3, 1.0), **options)
        self.assertEqual((above["accepted"], above["max_errors"]), (2, 0))

        census = plan_labels(accepted_population=6, accepted=6, expected_error=5 / 6)
        self.assertEqual(census["expected_population_errors"], 6)
        self.assertEqual((census["risk_target"], census["max_errors"]), (1.0, 6))

    def test_fixed_population_includes_the_exact_test_boundary(self):
        plan = plan_labels(accepted_population=20, expected_error=0, accepted=1)
        self.assertEqual(plan["risk_target"], 18 / 20)
        self.assertEqual(plan["max_errors"], 0)
        with self.assertRaisesRegex(ValueError, "exceeds the accepted population"):
            plan_labels(accepted_population=20, expected_error=0, accepted=21)

    def test_fast_hypergeometric_comparison_matches_exact_tails(self):
        for size, errors, labels, observed in (
            (20, 19, 1, 0),
            (14, 7, 7, 3),
            (20, 3, 20, 2),
            (20, 3, 20, 3),
            (200, 40, 190, 37),
            (32_000, 5_000, 4_800, 720),
            (32_000, 5_000, 4_800, 0),
        ):
            probability = hypergeometric_cdf(size, errors, labels, observed)
            self.assertEqual(
                hypergeometric_cdf_float(size, errors, labels, observed),
                float(probability),
            )
            for level in (Fraction(1, 20), Fraction(4, 5), probability):
                self.assertEqual(
                    compare_hypergeometric_cdf(size, errors, labels, observed, level),
                    (probability > level) - (probability < level),
                    (size, errors, labels, observed, level),
                )

    def test_fixed_population_search_matches_all_small_budgets(self):
        def exact_minimum(size, expected, allowed, confidence, power, limit):
            alpha = 1 - Fraction(str(confidence))
            wanted = Fraction(str(power))
            for labels in range(1, limit + 1):
                draws = comb(size, labels)

                def cumulative(errors, observed):
                    return sum(
                        comb(errors, count) * comb(size - errors, labels - count)
                        for count in range(observed + 1)
                    )

                max_errors = labels
                if allowed < size:
                    max_errors = max(
                        (
                            count
                            for count in range(labels + 1)
                            if cumulative(allowed + 1, count) <= alpha * draws
                        ),
                        default=-1,
                    )
                probability = Fraction(cumulative(expected, max_errors), draws)
                if probability >= wanted:
                    return labels, max_errors, float(probability)
            return None, None, None

        for size in (2, 5, 9):
            for expected in range(size + 1):
                for allowed in range(size + 1):
                    for confidence, power in ((0.95, 0.8), (0.8, 0.1), (0.5, 0.5)):
                        for limit in (size // 2, size):
                            plan = plan_labels(
                                accepted_population=size,
                                expected_error=max(0.0, (expected - 0.5) / size),
                                risk_target=min(1.0, (allowed + 0.5) / size),
                                holdout_confidence=confidence,
                                power=power,
                                max_accepted=limit,
                            )
                            self.assertEqual(
                                (
                                    plan["accepted"],
                                    plan["max_errors"],
                                    plan["certification_probability"],
                                ),
                                exact_minimum(
                                    size, expected, allowed, confidence, power, limit
                                ),
                                (size, expected, allowed, confidence, power, limit),
                            )

    def test_cli_reuses_saved_policy_and_checks_batch_cutoff(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, cutoff, decisions, output = (
                root / name
                for name in ("fit.csv", "policy.json", "batch.csv", "plan.json")
            )
            source.write_text(
                "id,score,correct\n" + "".join(f"fit-{i},0.8,1\n" for i in range(20)),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "policy",
                            str(source),
                            "--rule",
                            "empirical",
                            "--correct-column",
                            "correct",
                            "--risk-target",
                            "0.4",
                            "--holdout-confidence",
                            "0.90",
                            "--output",
                            str(cutoff),
                        ]
                    ),
                    0,
                )
            batch = "id,score,decision\na,0.9,accept\nb,0.75,review\nc,0.8,accept\n"
            decisions.write_text(batch, encoding="utf-8")
            args = [
                "plan",
                "--threshold-report",
                str(cutoff),
                "--expected-error",
                "0",
                "--batch-decisions",
                str(decisions),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([*args, "--output", str(output)]), 0)
            saved = output.read_bytes()
            plan = json.loads(saved)
            self.assertEqual(plan["accepted_population"], 2)
            self.assertEqual(plan["risk_target"], 0.4)
            self.assertEqual(plan["input"]["holdout_confidence"], 0.90)
            self.assertEqual(plan["input"]["threshold"], 0.8)
            previous_cutoff = cutoff.read_bytes()
            decisions.write_text(
                batch.replace("b,0.75,review", "b,0.75,accept"), encoding="utf-8"
            )
            for path, message in (
                (output, "decisions do not match the cutoff"),
                (cutoff, "output would overwrite the threshold report"),
            ):
                with self.subTest(path=path):
                    with (
                        self.assertRaises(SystemExit) as stopped,
                        redirect_stderr(io.StringIO()) as stderr,
                    ):
                        main([*args, "--output", str(path)])
                    self.assertEqual(stopped.exception.code, 2)
                    self.assertIn(message, stderr.getvalue())
            self.assertEqual(output.read_bytes(), saved)
            self.assertEqual(cutoff.read_bytes(), previous_cutoff)
