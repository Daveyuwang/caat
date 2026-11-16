import unittest
from dataclasses import replace

from caat.features import FEATURES, read_response


class FeatureTests(unittest.TestCase):
    def test_registered_scores_and_answer_offsets(self):
        _, response = read_response({
            "id": "example",
            "text": "One two.",
            "tokens": [
                {"text": "One", "logprob": -1.0},
                {"text": " two", "logprob": -2.0},
                {"text": ".", "logprob": -3.0},
            ],
            "answer_span": [3, 7],
        })
        scores = {name: feature.compute(response) for name, feature in FEATURES.items()}
        self.assertEqual(scores, {
            "mean_token_logprob": -2.0, "sequence_logprob": -6.0, "low_token_logprob": -3.0,
            "token_count": 3, "shorter_response": -3, "negative_mean_token_logprob": 2.0,
            "answer_span_logprob": -2.0,
        })
        self.assertIsNone(FEATURES["answer_span_logprob"].compute(replace(response, text="Unaligned")))

    def test_malformed_span_and_log_probability(self):
        record = {"id": "example", "text": "X", "tokens": [{"text": "X", "logprob": -1.0}]}
        with self.assertRaisesRegex(ValueError, "half-open"):
            read_response({**record, "answer_span": [0, 2]})
        record["tokens"][0]["logprob"] = 1.0
        with self.assertRaisesRegex(ValueError, "Invalid token"):
            read_response(record)
