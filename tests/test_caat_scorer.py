import csv
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from importlib.util import find_spec
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.cli import main
from caat.policy import split_rows
from caat.scorer import LinearScorer, fit_scorer
from caat.scores import ScoredRow
from caat.workflow import run_workflow


class SavedPolicyTests(unittest.TestCase):
    def test_categorical_report_handoff(self):
        model = LinearScorer(
            ("x",),
            (1, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
            (0, 1, -1),
            0,
            0.1,
            12,
            "a" * 64,
            "label",
            ("yes", "no"),
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "run.json"
            report.write_text(
                json.dumps(
                    {
                        "command": "run",
                        "input": {"id_column": "id", "review_missing_scores": True},
                        "selected_score": "fitted_score",
                        "scorer": model.to_dict(),
                        "policy": {
                            "confirmed": True,
                            "selection": {
                                "selected_threshold": 0.5,
                                "risk_target": 0.4,
                            },
                        },
                    }
                )
            )
            raw = root / "new.csv"
            raw.write_text(
                "id,x,label,grade,other\na,0,yes,1,1\nb,0,no,1,1\nc,0,unknown,0,0\nd,0,,0,0\n"
            )
            for cmd, flags in (
                ("route", ["--output", str(root / "decisions.csv")]),
                ("evaluate", ["--correct-column", "grade"]),
                ("audit", ["--correct-column", "grade", "--correct-column", "other"]),
                ("monitor", ["--correct-column", "grade"]),
            ):
                with self.subTest(cmd=cmd), redirect_stdout(StringIO()) as out:
                    main(
                        [
                            cmd,
                            str(raw),
                            "--threshold-report",
                            str(report),
                            "--json",
                            *flags,
                        ]
                    )
                    result = json.loads(out.getvalue())
                    if cmd == "route":
                        with (root / "decisions.csv").open() as saved:
                            self.assertEqual(
                                [r["decision"] for r in csv.DictReader(saved)],
                                ["accept", "review", "review", "review"],
                            )
                    else:
                        counts = (
                            result["graders"]["grade"]
                            if cmd == "audit"
                            else result["overall"]
                        )
                        self.assertEqual((counts["accepted"], counts["total"]), (1, 4))

    def test_same_decisions_from_features_and_precomputed_scores(self):
        scorer = LinearScorer(
            ("margin", "loss"), (1, -1), (0, 0), (1, 1), (1, 1), 0, 1, 10, "f" * 64
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "policy.json"
            report.write_text(
                json.dumps(
                    {
                        "command": "run",
                        "input": {"id_column": "item", "review_missing_scores": True},
                        "selected_score": "fitted_score",
                        "scorer": scorer.to_dict(),
                        "policy": {
                            "confirmed": True,
                            "selection": {
                                "selected_threshold": 0.5,
                                "risk_target": 0.4,
                            },
                        },
                    }
                )
            )
            raw = root / "predictions.csv"
            raw.write_text(
                "loss,item,margin,fitted_score,correct,other,output_sha256\n"
                "0.25,a,0.75,-99,1,1,ka\n0.5,b,0.4,99,0,0,kb\n0,c,,99,0,0,kc\n"
            )
            labels = root / "labels.csv"
            labels.write_text("item,output_sha256,correct\na,ka,1\nb,kb,0\nc,kc,0\n")
            for extra, accepted in (([], 1), (["--score-column", "fitted_score"], 2)):
                for cmd, flags in (
                    ("route", ["--output", str(root / "decisions.csv")]),
                    ("evaluate", ["--correct-column", "correct"]),
                    ("evaluate", ["--grader-file", f"first={labels}"]),
                    (
                        "audit",
                        ["--correct-column", "correct", "--correct-column", "other"],
                    ),
                    ("monitor", ["--grader-file", f"first={labels}"]),
                ):
                    with (
                        self.subTest(cmd=cmd, flags=flags, extra=extra),
                        redirect_stdout(StringIO()) as out,
                    ):
                        main(
                            [
                                cmd,
                                str(raw),
                                "--threshold-report",
                                str(report),
                                "--json",
                                *flags,
                                *extra,
                            ]
                        )
                        result = json.loads(out.getvalue())
                        if cmd == "audit":
                            counts = result["graders"]["correct"]
                        else:
                            counts = result["overall"]
                        self.assertEqual(counts["accepted"], accepted)
                        if cmd == "route":
                            with (root / "decisions.csv").open() as handle:
                                ids = [
                                    r["id"]
                                    for r in csv.DictReader(handle)
                                    if r["decision"] == "accept"
                                ]
                            self.assertEqual(ids, ["a"] if not extra else ["b", "c"])
            raw.write_text("item,fitted_score,correct\na,1,1\n")
            with redirect_stderr(StringIO()) as err, self.assertRaises(SystemExit):
                main(
                    [
                        "evaluate",
                        str(raw),
                        "--correct-column",
                        "correct",
                        "--threshold-report",
                        str(report),
                    ]
                )
            self.assertIn("--score-column", err.getvalue())


@unittest.skipUnless(find_spec("sklearn"), "requires the scorer extra")
class FittedScorerTests(unittest.TestCase):
    def test_class_vocabulary_matches_reference_and_roundtrips(self):
        import numpy as np
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import LogisticRegression

        xs = np.arange(60) % 7
        labels = np.array(["red", "green", "blue"] * 20)
        ys = (xs > 3) | (labels == "red")
        rows = [
            ScoredRow(str(i), (float(x),), (bool(y),), predicted_class=c)
            for i, (x, c, y) in enumerate(zip(xs, labels, ys, strict=True))
        ]
        model = fit_scorer(["margin"], rows, class_column="prediction")
        replay = LinearScorer.from_dict(json.loads(json.dumps(model.to_dict())))
        onehot = OneHotEncoder(sparse_output=False).fit_transform(labels[:, None])
        design = np.column_stack((xs, onehot))
        expected = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)
        ).fit(design, ys)
        np.testing.assert_allclose(
            [replay.score(r.scores, predicted_class=r.predicted_class) for r in rows],
            expected.decision_function(design),
            atol=1e-10,
        )
        self.assertEqual(replay.input_columns, ("margin", "prediction"))
        for label in ("", "yellow"):
            with self.assertRaisesRegex(ValueError, "missing or unseen"):
                replay.score_row({"margin": "1", "prediction": label}, 2, False)
            self.assertIsNone(
                replay.score_row({"margin": "1", "prediction": label}, 2, True)
            )

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
