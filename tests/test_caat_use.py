import csv
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from caat import GradedRow
from caat.cli import main
from caat.queue import review_queue


class FakeBackend:
    metadata = {"model": "fake"}

    def generate(self, prompt, max_new_tokens, *, sampled=False, tokens=False):
        if tokens:
            return "London", [{"text": "London", "logprob": -0.5}]
        if sampled:
            return "London", []
        return "80%", []

    def p_true(self, prompt):
        return 0.7


class UseTests(unittest.TestCase):
    def test_queue_budget_and_quotas_for_both_orders(self):
        rows = [
            GradedRow("a", 0.1, (False,), "first"),
            GradedRow("b", 0.3, (False,), "first"),
            GradedRow("c", 0.4, (True,), "other"),
            GradedRow("d", 0.9, (True,), "other"),
        ]
        queue, random, report = review_queue(
            reversed(rows), budget=2, quotas={"first": 1}, graders=("grade",)
        )
        self.assertEqual([row.id for row in queue], ["a", "c"])
        self.assertEqual(report["graders"]["grade"]["errors_found"], 1)
        self.assertEqual(report["random_baseline"]["by_group"].get("first"), 1)
        self.assertEqual(len(random), 2)
        reverse, _, report = review_queue(
            rows, budget=2, direction="lower", graders=("grade",)
        )
        self.assertEqual([row.id for row in reverse], ["d", "c"])
        self.assertEqual(report["graders"]["grade"]["random_expected_errors"], 1)
        with self.assertRaisesRegex(ValueError, "cannot fill"):
            review_queue(rows, budget=4, quotas={"first": 1}, graders=("grade",))

    def test_queue_and_saved_response_pipeline(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.jsonl"
            prompts.write_text(
                '{"id":"qa1","prompt":"Capital of the UK?"}\n', encoding="utf-8"
            )
            responses, meta, features, queue = [
                root / file
                for file in ("response.jsonl", "meta.json", "scores.csv", "queue.csv")
            ]
            with (
                patch("caat.cli.TransformersBackend", return_value=FakeBackend()),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            "collect",
                            str(prompts),
                            "--output",
                            str(responses),
                            "--metadata",
                            str(meta),
                            "--verbalized",
                            "--p-true",
                            "--samples",
                            "2",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "features",
                            str(responses),
                            "--output",
                            str(features),
                            "--feature",
                            "p_true",
                            "--feature",
                            "verbalized_confidence",
                            "--feature",
                            "sample_agreement",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "queue",
                            str(features),
                            "--score-column",
                            "p_true",
                            "--budget",
                            "1",
                            "--output",
                            str(queue),
                        ]
                    ),
                    0,
                )
            timings = json.loads(meta.read_text(encoding="utf-8"))["methods"]
            self.assertEqual(
                {key: record["calls"] for key, record in timings.items()},
                {
                    "greedy": 1,
                    "p_true": 1,
                    "verbalized_confidence": 1,
                    "sample": 2,
                },
            )
            self.assertEqual(
                features.read_text(encoding="utf-8").splitlines()[1], "qa1,0.7,0.8,1.0"
            )
            with queue.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(
                    list(csv.DictReader(stream)),
                    [{"id": "qa1", "score": "0.7", "group": ""}],
                )
