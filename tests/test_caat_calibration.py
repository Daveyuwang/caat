import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
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
            apply = [
                "calibrate",
                "apply",
                str(heldout),
                "--map",
                str(mapping),
                "--output",
                str(mapped),
            ]
            self.assertEqual(main(apply), 0)
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

            previous = mapped.read_bytes()
            heldout.write_text(
                "id,score,correct\nc,0.35,1\nd,invalid,1\n", encoding="utf-8"
            )
            with (
                redirect_stderr(io.StringIO()) as stderr,
                self.assertRaises(SystemExit) as stopped,
            ):
                main(apply)
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("invalid score at input line 3", stderr.getvalue())
            self.assertEqual(mapped.read_bytes(), previous)

    def test_fit_and_apply_share_the_missing_score_rule(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            table, mapping, holdout, mapped = (
                root / name
                for name in (
                    "calibration.csv",
                    "map.json",
                    "holdout.jsonl",
                    "mapped.csv",
                )
            )
            table.write_text(
                "id,confidence,correct\na,0,0\nb,,1\nc,0.7,1\n", encoding="utf-8"
            )
            fit = [
                "calibrate",
                "fit",
                str(table),
                "--score-column",
                "confidence",
                "--correct-column",
                "correct",
                "--method",
                "isotonic",
                "--review-missing-scores",
                "--output",
                str(mapping),
            ]
            holdout.write_text(
                '{"id":"01","confidence":0.35}\n{"id":"02","confidence":null}\n',
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(fit), 0)
                self.assertEqual(
                    main(
                        [
                            "calibrate",
                            "apply",
                            str(holdout),
                            "--map",
                            str(mapping),
                            "--output",
                            str(mapped),
                        ]
                    ),
                    0,
                )
            original = mapping.read_bytes()
            payload = json.loads(original)
            self.assertEqual(payload["calibration_total"], 3)
            self.assertEqual(payload["scored"], 2)
            self.assertEqual(
                payload["input"],
                {"score_column": "confidence", "review_missing_scores": True},
            )
            self.assertEqual(calibration_transform(payload)(0.35), 0.5)
            with mapped.open(encoding="utf-8", newline="") as source:
                self.assertEqual(
                    [
                        (row["id"], row["calibrated_probability"])
                        for row in csv.DictReader(source)
                    ],
                    [("01", "0.5"), ("02", "")],
                )
            table.write_text("id,confidence,correct\na,,0\nb,,1\n", encoding="utf-8")
            with (
                redirect_stderr(io.StringIO()) as stderr,
                self.assertRaises(SystemExit) as stopped,
            ):
                main(fit)
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn(
                "calibration requires at least one scored row", stderr.getvalue()
            )
            self.assertEqual(mapping.read_bytes(), original)

    def test_fit_and_apply_share_the_lower_score_direction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "calibration.csv"
            mapping = root / "map.json"
            holdout = root / "holdout.jsonl"
            mapped = root / "mapped.csv"
            table.write_text("id,loss,correct\na,1,1\nb,3,0\nc,,1\n", encoding="utf-8")
            raw = (
                '{"id":"low","loss":1}\n{"id":"high","loss":3}\n'
                '{"id":"missing","loss":null}\n'
            )
            fit = [
                "calibrate",
                "fit",
                str(table),
                "--score-column",
                "loss",
                "--direction",
                "lower",
                "--correct-column",
                "correct",
                "--method",
                "isotonic",
                "--review-missing-scores",
                "--output",
                str(mapping),
            ]
            apply = [
                "calibrate",
                "apply",
                str(holdout),
                "--map",
                str(mapping),
                "--output",
                str(mapped),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(fit), 0)
            original = mapping.read_bytes()
            payload = json.loads(original)
            self.assertEqual(payload["input"]["negated_score_source"], "loss")
            self.assertEqual(payload["scored"], 2)
            for contents, override in (
                (raw, ()),
                (
                    raw.replace('"loss"', '"uncertainty"'),
                    ("--raw-score-column", "uncertainty"),
                ),
            ):
                holdout.write_text(contents, encoding="utf-8")
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main([*apply, *override]), 0)
                with mapped.open(encoding="utf-8", newline="") as source:
                    probabilities = {
                        row["id"]: row["calibrated_probability"]
                        for row in csv.DictReader(source)
                    }
                self.assertEqual(
                    probabilities, {"low": "1.0", "high": "0.0", "missing": ""}
                )
            with (
                redirect_stderr(io.StringIO()) as stderr,
                self.assertRaises(SystemExit) as stopped,
            ):
                main([*fit, "--input-scale", "probability"])
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("requires score-scale calibration", stderr.getvalue())
            self.assertEqual(mapping.read_bytes(), original)

    def test_calibration_filters_before_validating_scores_and_grades(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            table, mapping, holdout, mapped = (
                root / name
                for name in ("mixed.csv", "map.json", "mixed.jsonl", "mapped.csv")
            )
            table.write_text(
                "id,confidence,correct,tier\na,0,0,encoder\na,unused,unused,other\n"
                "b,0.7,1,encoder\nc,,1,encoder\n",
                encoding="utf-8",
            )
            holdout.write_text(
                '{"id":"d","tier":"encoder","confidence":0.35}\n'
                '{"id":"d","tier":"other","confidence":"unused"}\n'
                '{"id":"e","tier":"encoder","confidence":null}\n',
                encoding="utf-8",
            )
            fit = [
                "calibrate",
                "fit",
                str(table),
                "--score-column",
                "confidence",
                "--correct-column",
                "correct",
                "--method",
                "isotonic",
                "--review-missing-scores",
                "--output",
                str(mapping),
            ]
            apply = [
                "calibrate",
                "apply",
                str(holdout),
                "--map",
                str(mapping),
                "--output",
                str(mapped),
            ]
            choose = ("--prediction-filter", "tier=encoder")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([*fit, *choose]), 0)
                self.assertEqual(main([*apply, *choose]), 0)
            payload = json.loads(mapping.read_bytes())
            self.assertEqual(
                payload["input"]["prediction_filter"],
                {"column": "tier", "value": "encoder"},
            )
            self.assertEqual((payload["calibration_total"], payload["scored"]), (3, 2))
            with mapped.open(encoding="utf-8", newline="") as source:
                self.assertEqual(
                    [
                        (row["id"], row["tier"], row["calibrated_probability"])
                        for row in csv.DictReader(source)
                    ],
                    [("d", "encoder", "0.5"), ("e", "encoder", "")],
                )
            for command, output in ((fit, mapping), (apply, mapped)):
                previous = output.read_bytes()
                with (
                    self.subTest(operation=command[1]),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as stopped,
                ):
                    main([*command, "--prediction-filter", "tier=absent"])
                self.assertEqual(stopped.exception.code, 2)
                self.assertEqual(output.read_bytes(), previous)

    def test_fit_joins_one_grader_to_the_selected_calibration_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scores = root / "scores.csv"
            grades = root / "grades.jsonl"
            mapping = root / "map.json"
            scores.write_text(
                "answer_id,loss,answer_key,tier\n"
                "a,unused,old-a,other\na,1,new-a,main\nb,3,b,main\nc,,c,main\n",
                encoding="utf-8",
            )
            matching = '{"id":"a","graded_key":"new-a","correct":true}\n'
            grade_rows = (
                '{"id":"c","graded_key":"c","correct":true}\n'
                '{"id":"b","graded_key":"b","correct":false}\n'
                '{"id":"a","graded_key":"old-a","correct":false}\n'
            )
            grades.write_text(grade_rows + matching, encoding="utf-8")
            fit = [
                "calibrate",
                "fit",
                str(scores),
                "--id-column",
                "answer_id",
                "--score-column",
                "loss",
                "--output-key-column",
                "answer_key",
                "--grader-file",
                f"primary={grades}",
                "--grader-id-column",
                "id",
                "--grader-output-key-column",
                "graded_key",
                "--allow-extra-grades",
                "--direction",
                "lower",
                "--method",
                "isotonic",
                "--review-missing-scores",
                "--prediction-filter",
                "tier=main",
                "--output",
                str(mapping),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(fit), 0)
            saved = mapping.read_bytes()
            payload = json.loads(saved)
            self.assertEqual(payload["input"]["grader_files"], {"primary": str(grades)})
            self.assertEqual(payload["input"]["negated_score_source"], "loss")
            self.assertEqual((payload["calibration_total"], payload["scored"]), (3, 2))
            transform = calibration_transform(payload)
            self.assertEqual((transform(-1.0), transform(-3.0)), (1.0, 0.0))

            grades.write_text(grade_rows, encoding="utf-8")
            with (
                redirect_stderr(io.StringIO()) as error,
                self.assertRaises(SystemExit) as stopped,
            ):
                main(fit)
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("output key mismatch for 'a'", error.getvalue())
            self.assertEqual(mapping.read_bytes(), saved)

    def test_checked_application_rejects_fitted_pairs_and_keeps_previous_output(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            calibration = root / "calibration.csv"
            holdout = root / "holdout.csv"
            mapping = root / "map.json"
            output = root / "probabilities.csv"
            calibration.write_text(
                "id,output_sha256,score,correct\na,a-v1,0.9,1\nb,b-v1,0.3,0\nc,c-v1,,1\n",
                encoding="utf-8",
            )
            fit = [
                "calibrate",
                "fit",
                str(calibration),
                "--method",
                "isotonic",
                "--correct-column",
                "correct",
                "--review-missing-scores",
                "--track-fit-outputs",
                "--output",
                str(mapping),
            ]
            self.assertEqual(main(fit), 0)
            report = json.loads(mapping.read_bytes())
            self.assertEqual(len(report["input"]["fit_outputs"]["sha256"]), 2)
            independent = "id,output_sha256,score\na,a-v2,0.7\nc,c-v1,\n"
            holdout.write_text(independent, encoding="utf-8")
            apply = [
                "calibrate",
                "apply",
                str(holdout),
                "--map",
                str(mapping),
                "--check-holdout",
                "--output",
                str(output),
            ]
            self.assertEqual(main(apply), 0)
            saved = output.read_bytes()
            holdout.write_text(independent + "b,b-v1,0.3\n", encoding="utf-8")
            with (
                redirect_stderr(io.StringIO()) as error,
                self.assertRaises(SystemExit) as rejected,
            ):
                main(apply)
            self.assertEqual(rejected.exception.code, 2)
            self.assertIn(
                "holdout reuses output from the fit for sample 'b'", error.getvalue()
            )
            self.assertEqual(output.read_bytes(), saved)

    def test_apply_uses_the_report_score_and_explicit_grader(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            labeled = root / "graded.csv"
            labeled.write_text(
                "id,raw_confidence,primary,alternate\n"
                + "".join(f"{index},{index},1,0\n" for index in range(10)),
                encoding="utf-8",
            )
            report = root / "policy.json"
            with redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "policy",
                        str(labeled),
                        "--score-column",
                        "raw_confidence",
                        "--correct-column",
                        "primary",
                        "--correct-column",
                        "alternate",
                        "--risk-target",
                        "0.40",
                        "--calibrate",
                        "isotonic",
                        "--output",
                        str(report),
                    ]
                )
            self.assertEqual(status, 1)
            unlabeled = root / "new.jsonl"
            unlabeled.write_text('{"id":"01","raw_confidence":4}\n', encoding="utf-8")
            mapped = root / "mapped.csv"
            apply = [
                "calibrate",
                "apply",
                str(unlabeled),
                "--map",
                str(report),
                "--output",
                str(mapped),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([*apply, "--grader", "alternate"]), 0)
            with mapped.open(encoding="utf-8", newline="") as content:
                self.assertEqual(
                    list(csv.DictReader(content)),
                    [
                        {
                            "id": "01",
                            "raw_confidence": "4",
                            "calibrated_probability": "0.0",
                        }
                    ],
                )
            original = mapped.read_bytes()
            with (
                redirect_stderr(io.StringIO()) as stderr,
                self.assertRaises(SystemExit) as stopped,
            ):
                main(apply)
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn(
                "select a map with --grader (primary, alternate)", stderr.getvalue()
            )
            self.assertEqual(mapped.read_bytes(), original)
