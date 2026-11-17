import csv
import io
import json
import unittest
from contextlib import redirect_stdout
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.cli import main
from caat.lm_eval_log import choice_scores, join_scores


def sample(doc_id, *, sample_filter="strict", completion="A\n"):
    prompt = f"Question {doc_id}"
    return {
        "doc_id": doc_id,
        "filter": sample_filter,
        "prompt_hash": sha256(prompt.encode()).hexdigest(),
        "arguments": {"gen_args_0": {"arg_0": prompt, "arg_1": {}}},
        "resps": [[completion]],
        "filtered_resps": ["A"],
        "exact_match": float(doc_id == 0),
    }


def write_samples(path, *rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


class LMEvalLogTests(unittest.TestCase):
    def test_join_verifies_raw_output_keys_after_selecting_a_filter(self):
        with TemporaryDirectory() as directory:
            log, exported, external, joined = (
                Path(directory) / name
                for name in ("samples.jsonl", "export.csv", "scores.csv", "joined.csv")
            )
            write_samples(log, sample(0), sample(1), sample(0, sample_filter="loose"))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "lm-eval",
                            "export",
                            str(log),
                            "--filter",
                            "strict",
                            "--output",
                            str(exported),
                        ]
                    ),
                    0,
                )
            rows = csv_rows(exported)
            self.assertEqual(rows[0]["completion"], "A\n")
            self.assertEqual(rows[0]["output_sha256"], sha256(b"A\n").hexdigest())

            def write_scores():
                with external.open("w", encoding="utf-8", newline="") as target:
                    columns = (*rows[0], "confidence")
                    writer = csv.DictWriter(target, columns)
                    writer.writeheader()
                    for row in reversed(rows):
                        writer.writerow(
                            {**row, "confidence": 0.75 - 0.5 * int(row["id"])}
                        )

            write_scores()
            options = {"graders": ("exact_match",), "score_column": "confidence"}
            with self.assertRaisesRegex(ValueError, "several filters"):
                join_scores(log, external, joined, **options)
            self.assertEqual(
                join_scores(log, external, joined, sample_filter="strict", **options), 2
            )
            self.assertEqual(
                [
                    (r["id"], r["confidence"], r["exact_match"])
                    for r in csv_rows(joined)
                ],
                [("0", "0.75", "1"), ("1", "0.25", "0")],
            )
            self.assertEqual(
                join_scores(
                    log,
                    external,
                    joined,
                    sample_filter="strict",
                    lower_is_better=("confidence",),
                    **options,
                ),
                2,
            )
            self.assertEqual(
                [row["negative_confidence"] for row in csv_rows(joined)],
                ["-0.75", "-0.25"],
            )
            expected = joined.read_bytes()
            rows[1]["output_sha256"] = "different"
            write_scores()
            with self.assertRaisesRegex(ValueError, "output key mismatch"):
                join_scores(log, external, joined, sample_filter="strict", **options)
            self.assertEqual(joined.read_bytes(), expected)

    def test_choice_likelihoods_and_accuracy_are_aligned(self):
        with TemporaryDirectory() as directory:
            log = Path(directory) / "samples.jsonl"
            output = Path(directory) / "choices.csv"
            row = sample(0)
            row["arguments"] = {
                "gen_args_0": {"arg_0": "Question 0", "arg_1": " red"},
                "gen_args_1": {"arg_0": "Question 0", "arg_1": " blue"},
            }
            row.update(
                filtered_resps=[["-2.5", "False"], ["-1", "True"]],
                target="1",
                acc=1.0,
                acc_norm=0.0,
            )
            write_samples(log, row)
            self.assertEqual(choice_scores(log, output), 1)
            result = csv_rows(output)[0]
            self.assertEqual(result["output_sha256"], sha256(b" blue").hexdigest())
            self.assertEqual(result["max_loglikelihood"], "-1.0")
            self.assertEqual(result["loglikelihood_margin"], "1.5")
            self.assertEqual(result["acc"], "1")
            write_samples(log, row, row)
            with self.assertRaisesRegex(ValueError, "duplicate doc_id"):
                choice_scores(log, output)
            write_samples(log, {**row, "acc": 0.0})
            with self.assertRaisesRegex(ValueError, "acc disagrees"):
                choice_scores(log, output)
