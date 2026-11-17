import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from caat import GradedRow
from caat.cli import main
from caat.collect import collect_responses
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


class BatchBackend:
    def __init__(self):
        self.draws = {}

    def generate(self, prompt, max_new_tokens, *, sampled=False, tokens=False):
        if tokens:
            return prompt.upper(), [{"text": prompt.upper(), "logprob": -0.5}]
        if sampled:
            self.draws[prompt] = self.draws.get(prompt, 0) + 1
            return f"{prompt} {self.draws[prompt]}", []
        question = prompt.splitlines()[1]
        return ("bad confidence" if question == "ccc" else "60%"), []

    def generate_batch(self, prompts, max_new_tokens, **options):
        return [self.generate(prompt, max_new_tokens, **options) for prompt in prompts]

    def p_true(self, prompt):
        return len(prompt.splitlines()[4]) / 10

    def p_true_batch(self, prompts):
        return [self.p_true(prompt) for prompt in prompts]


class UseTests(unittest.TestCase):
    def test_collect_batches_keep_sample_and_confidence_rows_aligned(self):
        prompts = {"one": "a", "two": "bb", "three": "ccc"}
        options = {"p_true": True, "verbalized": True, "samples": 3}
        serial, _ = collect_responses(prompts, BatchBackend(), **options)
        batched, timings = collect_responses(
            prompts, BatchBackend(), batch_size=2, **options
        )
        self.assertEqual(batched, serial)
        same, serial_greedy_timings = collect_responses(
            prompts, BatchBackend(), batch_size=2, serial_greedy=True, **options
        )
        self.assertEqual(same, serial)
        self.assertEqual(serial_greedy_timings["greedy"]["batches"], 3)
        self.assertEqual(serial_greedy_timings["sample"]["batches"], 5)
        self.assertEqual([row["p_true"] for row in batched], [0.1, 0.2, 0.3])
        self.assertEqual(batched[1]["samples"], ["bb 1", "bb 2", "bb 3"])
        self.assertNotIn("verbalized_confidence", batched[2])
        self.assertEqual(
            {name: (item["calls"], item["batches"]) for name, item in timings.items()},
            {
                "greedy": (3, 2),
                "p_true": (3, 2),
                "verbalized_confidence": (3, 2),
                "sample": (9, 5),
            },
        )

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

    def test_sample_only_uses_the_routed_accept_group_and_keeps_keys(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "predictions.csv"
            routed = root / "decisions.csv"
            sample = root / "sample.csv"
            source.write_text(
                "id,score,key\nv,0.87,out-v\nr,0.7,out-r\nk,0.81,out-k\nt,0.95,out-t\n",
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "route",
                            str(source),
                            "--threshold",
                            "0.8",
                            "--output-key-column",
                            "key",
                            "--output",
                            str(routed),
                        ]
                    ),
                    0,
                )
            args = [
                "queue",
                str(routed),
                "--prediction-filter",
                "decision=accept",
                "--output-key-column",
                "output_key",
                "--random-output",
                str(sample),
                "--json",
            ]
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main([*args, "--budget", "2"]), 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(
                summary["input"]["prediction_filter"],
                {"column": "decision", "value": "accept"},
            )
            self.assertEqual(summary["sample_path"], str(sample))
            self.assertEqual((summary["total"], summary["budget"]), (3, 2))
            with sample.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(
                    list(csv.DictReader(stream)),
                    [
                        {
                            "id": "t",
                            "score": "0.95",
                            "group": "",
                            "output_key": "out-t",
                        },
                        {
                            "id": "k",
                            "score": "0.81",
                            "group": "",
                            "output_key": "out-k",
                        },
                    ],
                )
            original = sample.read_bytes()
            with (
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                main([*args, "--budget", "4"])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(sample.read_bytes(), original)
