import unittest
from dataclasses import replace
from math import log

from caat.features import (
    DEFAULT_FEATURES,
    FEATURES,
    compute_feature,
    parse_confidence,
    read_response,
)


class FeatureTests(unittest.TestCase):
    def test_registered_scores_and_answer_offsets(self):
        _, response = read_response(
            {
                "id": "example",
                "text": "One two.",
                "tokens": [
                    {"text": "One", "logprob": -1.0},
                    {"text": " two", "logprob": -2.0},
                    {"text": ".", "logprob": -3.0},
                ],
                "answer_span": [3, 7],
            }
        )
        scores = {
            name: compute_feature(FEATURES[name], response) for name in DEFAULT_FEATURES
        }
        self.assertEqual(
            scores,
            {
                "mean_token_logprob": -2.0,
                "sequence_logprob": -6.0,
                "low_token_logprob": -3.0,
                "token_count": 3,
                "shorter_response": -3,
                "negative_mean_token_logprob": 2.0,
                "answer_span_logprob": -2.0,
            },
        )
        self.assertIsNone(
            FEATURES["answer_span_logprob"].compute(replace(response, text="Unaligned"))
        )

    def test_malformed_span_and_log_probability(self):
        record = {
            "id": "example",
            "text": "X",
            "tokens": [{"text": "X", "logprob": -1.0}],
        }
        with self.assertRaisesRegex(ValueError, "half-open"):
            read_response({**record, "answer_span": [0, 2]})
        record["tokens"][0]["logprob"] = 1.0
        with self.assertRaisesRegex(ValueError, "Invalid token"):
            read_response(record)

    def test_black_box_scores_and_costs(self):
        _, response = read_response(
            {
                "id": "a",
                "text": "London",
                "p_true": 0.9,
                "verbalized_confidence": "Confidence: 72.5%",
                "samples": ["London", "LONDON.", "Paris", "London"],
                "clusters": ["uk", "uk", "fr", "uk"],
            }
        )
        self.assertEqual(compute_feature(FEATURES["p_true"], response), 0.9)
        self.assertEqual(
            compute_feature(FEATURES["verbalized_confidence"], response), 0.725
        )
        self.assertEqual(compute_feature(FEATURES["sample_agreement"], response), 0.5)
        self.assertEqual(
            compute_feature(FEATURES["normalized_sample_agreement"], response), 0.75
        )
        entropy = FEATURES["semantic_entropy"]
        self.assertEqual((entropy.direction, entropy.cost), ("lower", "n-samples"))
        expected = -0.75 * log(0.75) - 0.25 * log(0.25)
        self.assertAlmostEqual(compute_feature(entropy, response), expected)
        weighted = replace(
            response, sample_logprobs=(log(0.1), log(0.1), log(0.3), log(0.1))
        )
        self.assertAlmostEqual(compute_feature(entropy, weighted), log(2))
        self.assertEqual(parse_confidence("Very high"), 0.9)
        with self.assertRaises(ValueError):
            parse_confidence("Somewhere between 75% and 90%")
        with self.assertRaisesRegex(ValueError, "identify each sample"):
            read_response(
                {"id": "a", "text": "", "samples": ["A", "B"], "clusters": ["a"]}
            )
