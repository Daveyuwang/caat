import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from caat import GradedRow, select
from caat.scorer import LinearScorer
from caat.select import design_thresholds
from caat.table import output_fingerprint
from caat.workflow import run_workflow


class FrozenWorkflowTests(unittest.TestCase):
    def test_frozen_grid_and_portable_score(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "run.toml"
            model = LinearScorer(("cost",), (-1,), (0,), (1,), (1,), 0, 1, 10, "f" * 64)
            (root / "model.json").write_text(json.dumps(model.to_dict()))
            (root / "ref.csv").write_text("id,cost\nr1,0.1\nr2,0.5\nr3,0.9\n")
            data = "id,cost,correct\n" + "".join(
                f"c{i},{v},{int(v < 0.5)}\n" for i, v in enumerate([0.1, 0.9] * 40)
            )
            (root / "cal.csv").write_text(data)
            (root / "hold.csv").write_text(data.replace("\nc", "\nh") + "missing,,0\n")
            common = """correct_columns=["correct"]
risk_target=0.25
review_missing_scores=true
holdout_output="holdout.csv"
json_output="report.json"
[predictions]
calibration="cal.csv"
holdout="hold.csv"
"""
            for saved in (False, True):
                for reference in (False, True):
                    extra = 'scorer="model.json"\n' if saved else ""
                    source = (
                        ""
                        if saved
                        else 'score_columns=["cost"]\nlower_is_better=["cost"]\n'
                    )
                    grid = (
                        'reference="ref.csv"'
                        if reference
                        else ("thresholds=[-0.5]" if saved else "thresholds=[0.5]")
                    )
                    cfg = source + common + "[frozen]\n" + extra + grid
                    config.write_text(cfg)
                    report = run_workflow(config, save_outputs=True)
                    policy = report["policy"]
                    thresholds = (
                        design_thresholds([-0.1, -0.5, -0.9]) if reference else [-0.5]
                    )
                    expected = select(
                        [
                            GradedRow(str(i), -v, (v < 0.5,))
                            for i, v in enumerate([0.1, 0.9] * 40)
                        ],
                        thresholds=thresholds,
                        graders=["correct"],
                        risk_target=0.25,
                    )
                    self.assertEqual(policy["selection"], expected)
                    self.assertTrue(policy["confirmed"])
                    self.assertEqual(
                        policy["heldout"]["graders"]["correct"]["accepted"], 40
                    )
                    self.assertEqual(policy["split"]["heldout"], 81)
                    self.assertEqual(
                        report["input"]["frozen"]["reference_rows"],
                        3 if reference else 0,
                    )
                    with (root / "holdout.csv").open() as out:
                        self.assertEqual(len(list(csv.DictReader(out))), 81)
                    before = (root / "hold.csv").read_text()
                    (root / "hold.csv").write_text(before.replace(",1\n", ",0\n"))
                    altered = run_workflow(config)
                    self.assertEqual(altered["policy"]["selection"], expected)
                    self.assertFalse(altered["policy"]["confirmed"])
                    (root / "hold.csv").write_text(before)
            (root / "ref.csv").write_text("id,cost\nc0,0.1\n")
            with self.assertRaisesRegex(ValueError, "reuses sample ID"):
                run_workflow(config)
            config.write_text('score_selection="joint-calibration"\n' + cfg)
            with self.assertRaisesRegex(ValueError, "fixed score"):
                run_workflow(config)

    def test_tracked_model_rejects_training_overlap_and_carries_identities(self):
        model = LinearScorer(("x",), (1,), (0,), (1,), (1,), 0, 1, 10, "f" * 64)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            old = output_fingerprint("old", "answer")
            (root / "model.json").write_text(
                json.dumps(
                    {
                        "scorer": model.to_dict(),
                        "input": {
                            "fit_outputs": {"key_column": "hash", "sha256": [old]}
                        },
                    }
                )
            )
            cfg = root / "a.toml"
            cfg.write_text("""correct_columns=["correct"]
risk_target=0.5
[predictions]
calibration="cal.csv"
holdout="h.csv"
[frozen]
scorer="model.json"
thresholds=[0.5]
""")
            (root / "cal.csv").write_text(
                "id,hash,x,correct\n" + "".join(f"c{i},k{i},1,1\n" for i in range(20))
            )
            (root / "h.csv").write_text("id,hash,x,correct\nold,answer,1,1\n")
            with self.assertRaisesRegex(ValueError, "reuses output"):
                run_workflow(cfg)
            (root / "h.csv").write_text("id,hash,x,correct\nnew,another,1,1\n")
            result = run_workflow(cfg)
            self.assertIn(old, result["input"]["fit_outputs"]["sha256"])
            self.assertEqual(
                result["input"]["frozen"]["training_output_check"], "tracked"
            )
