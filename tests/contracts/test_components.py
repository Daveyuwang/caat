import json
import unittest
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from caat import GradedRow, select
from caat.adapters import ADAPTERS
from caat.calibrate import CALIBRATORS, calibration_transform, fit_calibration
from caat.features import FEATURES, compute_feature, read_response
from caat.metrics import METRICS
from caat.plugins import catalog
from caat.renderers import RENDERERS, render_report
from caat.rules import RULES
from caat.scores import COSTS
from caat.table import validated_rows


ROOT = Path(__file__).resolve().parents[2]


class ComponentContracts(unittest.TestCase):
    def test_features(self):
        _, response = read_response(
            {
                "id": "fixture",
                "text": "A.",
                "answer_span": [0, 1],
                "tokens": [
                    {"text": "A", "logprob": -0.3},
                    {"text": ".", "logprob": -0.1},
                ],
                "p_true": 0.8,
                "verbalized_confidence": "75%",
                "samples": ["A", "A.", "B"],
                "clusters": ["a", "a", "b"],
                "sample_logprobs": [-0.4, -0.5, -0.8],
            }
        )
        for name, component in FEATURES.items():
            with self.subTest(feature=name):
                self.assertIn(component.direction, ("higher", "lower"))
                self.assertIn(component.cost, COSTS)
                actual = compute_feature(component, response)
                self.assertIsNotNone(actual)
                self.assertTrue(isfinite(actual))
                self.assertEqual(actual, compute_feature(component, response))

    def test_calibrators(self):
        for name, component in CALIBRATORS.items():
            with self.subTest(calibrator=name):

                def fit():
                    model = fit_calibration(
                        [0.1, 0.2, 0.4, 0.8, 0.9], [0, 1, 0, 1, 1], method=name
                    )
                    return json.dumps(model, allow_nan=False, sort_keys=True)

                saved = fit()
                self.assertEqual(saved, fit())
                model = calibration_transform(json.loads(saved))
                grid = [i / 20 for i in range(21)]
                values = list(map(model, grid))
                self.assertTrue(all(0 <= value <= 1 for value in values))
                self.assertEqual(values, list(map(model, grid)))
                if component.monotone:
                    self.assertEqual(values, sorted(values))

    def test_metrics(self):
        rows = [GradedRow(str(i), i / 4, (i % 2 == 0,)) for i in range(5)]
        for name, component in METRICS.items():
            with self.subTest(metric=name):
                self.assertIn(component.direction, ("higher", "lower", "none", "mixed"))
                self.assertIs(type(component.requires_probability), bool)
                result = component.compute(rows)
                self.assertIsInstance(result, (int, float, dict, list, tuple))
                json.dumps(result, allow_nan=False)
                self.assertEqual(result, component.compute(rows))

    def test_rules(self):
        rows = [
            GradedRow(str(i), 1.0 if i < 40 else 0.0, (i < 40,)) for i in range(100)
        ]
        for name in RULES:
            with self.subTest(rule=name):
                result = select(
                    rows,
                    graders=("grade",),
                    thresholds=(2.0, 1.0, 0.0),
                    risk_target=0.2,
                    rule=name,
                )
                for item in result["candidates"]:
                    if item["passes"]:
                        self.assertGreater(item["accepted"], 0)
                        self.assertLessEqual(
                            item["worst_error_rate"], result["risk_target"]
                        )
                        if result["per_candidate_level"] is not None:
                            self.assertLessEqual(
                                item["pvalue"], result["per_candidate_level"]
                            )
                self.assertIn(result["selected_threshold"], (None, 1.0))
                self.assertEqual(
                    result,
                    select(
                        rows,
                        graders=("grade",),
                        thresholds=(2.0, 1.0, 0.0),
                        risk_target=0.2,
                        rule=name,
                    ),
                )

    def test_adapters(self):
        with TemporaryDirectory() as directory:
            for name, component in ADAPTERS.items():
                with self.subTest(adapter=name):
                    path, options = (
                        ROOT / "examples/ticket_triage.csv",
                        {
                            "id_column": "ticket_id",
                            "score_column": "confidence",
                            "prediction_column": "model_route",
                            "reference_column": "expected_route",
                        },
                    )
                    if component.fixture is not None:
                        fixture_dir = Path(directory) / name
                        fixture_dir.mkdir()
                        path, options = component.fixture(fixture_dir)
                    grader_names = options.get("correct_columns") or ("correct",)
                    rows = validated_rows(component.read(path, options), grader_names)
                    self.assertEqual(rows, list(component.read(path, options)))
                    self.assertTrue(
                        component.source_format and component.source_version
                    )

    def test_renderers(self):
        golden = ROOT / "tests/golden"
        reports = [
            json.loads((golden / file).read_text(encoding="utf-8"))
            for file in (
                "evaluate/metrics.json",
                "audit.json",
                "select.json",
                "policy.json",
            )
        ]
        for name, component in RENDERERS.items():
            for report in reports:
                if report["command"] in component.commands:
                    with self.subTest(renderer=name, command=report["command"]):
                        data = render_report(name, report)
                        self.assertTrue(data.strip())
                        self.assertEqual(data, render_report(name, report))

    def test_catalog(self):
        result = catalog()
        self.assertEqual(
            set(result),
            {"features", "calibrators", "metrics", "rules", "adapters", "renderers"},
        )
        self.assertTrue(all(entries for entries in result.values()))
