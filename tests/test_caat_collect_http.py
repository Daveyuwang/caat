import csv
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from hashlib import sha256
from math import exp
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError

from caat.cli import main
from caat.collect_http import ChatCompletionsBackend


def reply(text, tokens=None):
    choice = {"message": {"content": text}, "finish_reason": "stop"}
    if tokens is not None:
        choice["logprobs"] = {"content": tokens}
    return io.BytesIO(json.dumps({"choices": [choice]}).encode())


class ChatCollectionTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        self.prompts = root / "prompts.jsonl"
        self.prompts.write_text('{"id":"a","prompt":"A question"}\n', encoding="utf-8")
        self.responses = root / "responses.jsonl"
        self.metadata = root / "metadata.json"
        self.scores = root / "scores.csv"
        self.base_url = "http://127.0.0.1:18743/v1"

    def collect(self, *options):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(
                    [
                        "collect",
                        str(self.prompts),
                        "--base-url",
                        self.base_url,
                        "--model",
                        "smoke",
                        "--output",
                        str(self.responses),
                        "--metadata",
                        str(self.metadata),
                        *options,
                    ]
                ),
                0,
            )
        return (
            [
                json.loads(line)
                for line in self.responses.read_text(encoding="utf-8").splitlines()
            ],
            json.loads(self.metadata.read_text(encoding="utf-8")),
        )

    def test_replays_api_token_probabilities_and_samples(self):
        tokens = [
            {"token": "caf", "logprob": -0.8, "bytes": [99, 97, 102]},
            {"token": "�", "logprob": -1.2, "bytes": [195]},
            {"token": "�", "logprob": -0.4, "bytes": [169]},
            {
                "token": "<stop>",
                "logprob": -9999.0,
                "bytes": [60, 115, 116, 111, 112, 62],
            },
        ]
        check = [
            {
                "token": "A",
                "logprob": -0.2,
                "top_logprobs": [{"token": "B", "logprob": -1.2}],
            }
        ]
        with (
            patch.dict(os.environ, {"CAAT_TEST_API_KEY": "test-token"}),
            patch(
                "caat.collect_http.urlopen",
                side_effect=[
                    reply("café", tokens),
                    reply("A", check),
                    reply("café"),
                    reply("café"),
                ],
            ) as request,
        ):
            rows, metadata = self.collect(
                "--api-key-env", "CAAT_TEST_API_KEY", "--p-true", "--samples", "2"
            )
        row = rows[0]
        self.assertEqual("".join(token["text"] for token in row["tokens"]), row["text"])
        self.assertAlmostEqual(row["p_true"], 1 / (1 + exp(-1)))
        calls = [call.args[0] for call in request.call_args_list]
        self.assertTrue(
            all(call.full_url == self.base_url + "/chat/completions" for call in calls)
        )
        self.assertTrue(
            all(
                call.get_header("Authorization") == "Bearer test-token"
                for call in calls
            )
        )
        self.assertNotIn("test-token", json.dumps(metadata))
        payloads = [json.loads(call.data) for call in calls]
        self.assertEqual(payloads[1]["top_logprobs"], 20)
        self.assertNotEqual(payloads[2]["seed"], payloads[3]["seed"])
        self.assertEqual(metadata["http_requests"], 4)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(
                    [
                        "features",
                        str(self.responses),
                        "--output",
                        str(self.scores),
                        "--include-output-key",
                        "--feature",
                        "mean_token_logprob",
                        "--feature",
                        "normalized_sample_agreement",
                    ]
                ),
                0,
            )
        with self.scores.open(newline="", encoding="utf-8") as source:
            scores = next(csv.DictReader(source))
        self.assertAlmostEqual(float(scores["mean_token_logprob"]), -0.8)
        self.assertEqual(float(scores["normalized_sample_agreement"]), 1)
        self.assertEqual(
            scores["output_sha256"], sha256("café".encode("utf-8")).hexdigest()
        )

    def test_retries_without_logprobs_and_keeps_text_signals(self):
        with self.prompts.open("a", encoding="utf-8") as source:
            source.write('{"id":"b","prompt":"Another question"}\n')
        error = HTTPError(
            self.base_url,
            400,
            "Bad Request",
            None,
            io.BytesIO(
                b'{"error":{"code":"unsupported_parameter","param":"logprobs"}}'
            ),
        )
        with patch(
            "caat.collect_http.urlopen",
            side_effect=[
                error,
                reply("a"),
                reply("a"),
                reply("b"),
                reply("b"),
            ],
        ) as request:
            rows, metadata = self.collect("--p-true", "--samples", "1")
        self.assertEqual(
            rows,
            [
                {"id": "a", "text": "a", "samples": ["a"]},
                {"id": "b", "text": "b", "samples": ["b"]},
            ],
        )
        self.assertEqual(metadata["http_requests"], 5)
        self.assertEqual(metadata["probability_rows"], {"tokens": 0, "p_true": 0})
        payloads = [json.loads(call.args[0].data) for call in request.call_args_list]
        self.assertTrue(payloads[0]["logprobs"])
        self.assertTrue(all("logprobs" not in payload for payload in payloads[1:]))

    def test_p_true_requires_both_candidate_probabilities(self):
        backend = ChatCompletionsBackend(self.base_url, "smoke", seed=42)
        check = [
            {
                "token": "A",
                "logprob": -0.2,
                "top_logprobs": [{"token": "B", "logprob": -9999.0}],
            }
        ]
        with patch("caat.collect_http.urlopen", return_value=reply("A", check)):
            self.assertIsNone(backend.p_true("review"))

    def test_leaves_unrelated_http_error_visible(self):
        backend = ChatCompletionsBackend(self.base_url, "smoke", seed=42)
        error = HTTPError(
            self.base_url,
            400,
            "Bad Request",
            None,
            io.BytesIO(b'{"error":{"message":"prompt is too long"}}'),
        )
        with (
            patch("caat.collect_http.urlopen", side_effect=error) as request,
            self.assertRaisesRegex(ValueError, "HTTP 400"),
        ):
            backend.generate("a", 32, tokens=True)
        self.assertEqual(request.call_count, 1)
