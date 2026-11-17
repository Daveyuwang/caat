import csv
import json
import unittest
from contextlib import redirect_stdout
from importlib.util import find_spec
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.cli import main
from caat.policy import split_rows
from caat.scorer import LinearScorer, fit_scorer
from caat.scores import ScoredRow
from caat.workflow import run_workflow


@unittest.skipUnless(find_spec("sklearn"), "requires the scorer extra")
class FittedScorerTests(unittest.TestCase):
    def test_python_reference_and_roundtrip(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        x = np.array([[i % 7, (i * 13) % 17] for i in range(100)], dtype=float)
        y = x[:, 0] - x[:, 1] / 4 > 1
        rows = [
            ScoredRow(str(i), tuple(v), (bool(b),))
            for i, (v, b) in enumerate(zip(x, y, strict=True))
        ]
        model = fit_scorer(["a", "b"], rows, lower_is_better=["b"])
        loaded = LinearScorer.from_dict(json.loads(json.dumps(model.to_dict())))
        reference = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)
        ).fit(x * [1, -1], y)
        np.testing.assert_allclose(
            [loaded.score(v) for v in x],
            reference.decision_function(x * [1, -1]),
            atol=1e-8,
            rtol=0,
        )
        self.assertIn(
            fit_scorer(["a", "b"], rows, regularization="auto").regularization,
            (0.01, 0.1, 1.0),
        )
        self.assertIsNone(loaded.score((1, None)))
        with self.assertRaises(ValueError):
            LinearScorer.from_dict({**model.to_dict(), "scales": [0, 1]})

    def test_workflow_and_portable_routing(self):
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            data = (
                Path(__file__).resolve().parents[1]
                / "examples"
                / "digit_classifier.jsonl"
            )
            rows = [json.loads(line) for line in data.read_text().splitlines()]
            design, calibration, holdout = split_rows(
                rows, 42, id_key=lambda r: r["image_id"]
            )
            for name, part in (
                ("design", design),
                ("calibration", calibration),
                ("holdout", holdout),
            ):
                (root / f"{name}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in part)
                )
            config = root / "workflow.toml"
            config.write_text("""id_column="image_id"
score_columns=["margin","top_prob"]
prediction_column="predicted"
reference_column="actual"
risk_target=0.02
refit_checks=4
bootstrap=0
review_missing_scores=true
lower_is_better=["margin"]
json_output="report.json"
holdout_output="holdout.csv"
html="report.html"
[scorer]
regularization=0.1
[predictions]
design="design.jsonl"
calibration="calibration.jsonl"
holdout="holdout.jsonl"
""")
            report = run_workflow(config, save_outputs=True)
            self.assertTrue(report["policy"]["confirmed"])
            self.assertIn(
                "mean_changed_fraction", report["policy"]["refit_disagreement"]
            )
            self.assertIn(
                "fitted_score", (root / "holdout.csv").read_text().splitlines()[0]
            )
            # Reverse the feature order and drop both labels on the new batch.
            batch = root / "new.csv"
            batch.write_text(
                "top_prob,image_id,margin\n0.99,a,0.98\n0.4,b,0.1\n0.8,c,\n"
            )
            saved = root / "model.json"

            def cli(*args):
                return main([str(arg) for arg in args])

            self.assertEqual(
                cli(
                    "scorer",
                    "fit",
                    root / "design.jsonl",
                    "--id-column",
                    "image_id",
                    "--prediction-column",
                    "predicted",
                    "--reference-column",
                    "actual",
                    "--score-column",
                    "margin",
                    "--score-column",
                    "top_prob",
                    "--lower-is-better",
                    "margin",
                    "--output",
                    saved,
                ),
                0,
            )
            self.assertEqual(
                json.loads(saved.read_text()), json.loads(json.dumps(report["scorer"]))
            )
            self.assertEqual(
                cli(
                    "scorer",
                    "apply",
                    batch,
                    "--scorer",
                    saved,
                    "--review-missing-scores",
                    "--output",
                    root / "scored.csv",
                ),
                0,
            )
            for filename, input_path, override in (
                ("raw.csv", batch, ()),
                (
                    "explicit.csv",
                    root / "scored.csv",
                    ("--score-column", "fitted_score"),
                ),
            ):
                self.assertEqual(
                    cli(
                        "route",
                        input_path,
                        "--threshold-report",
                        root / "report.json",
                        *override,
                        "--output",
                        root / filename,
                    ),
                    0,
                )
            self.assertEqual(
                (root / "raw.csv").read_bytes(), (root / "explicit.csv").read_bytes()
            )
            with (root / "raw.csv").open() as handle:
                decisions = list(csv.DictReader(handle))
            self.assertEqual(
                [r["decision"] for r in decisions], ["accept", "review", "review"]
            )
            for row in holdout:
                row["actual"] = -1
            (root / "holdout.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in holdout)
            )
            altered = run_workflow(config, save_outputs=True)
            self.assertEqual(altered["scorer"], report["scorer"])
            self.assertEqual(
                altered["policy"]["selection"], report["policy"]["selection"]
            )
            self.assertEqual(
                altered["policy"]["refit_disagreement"],
                report["policy"]["refit_disagreement"],
            )
            self.assertFalse(altered["policy"]["confirmed"])
            with self.assertRaises(SystemExit):
                cli(
                    "route",
                    batch,
                    "--threshold-report",
                    root / "report.json",
                    "--output",
                    root / "denied.csv",
                )
