import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.cli import main
from caat.eee_log import IDENTITY, export_generations, join_scores


def record(item_id, correct, **meta):
    return {
        "schema_version": "instance_level_eval_0.2.2",
        "evaluation_id": "saved-run",
        "model_id": "example/model",
        "evaluation_name": "math",
        "sample_id": item_id,
        "interaction_type": "single_turn",
        "input": {"raw": "2 + 2", "formatted": "Question: 2 + 2"},
        "output": {"raw": ["4\n"]},
        "evaluation": {"is_correct": correct, "score": 0.8},
        **meta,
    }


def save_samples(path, *samples):
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in samples), encoding="utf-8"
    )


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


class EveryEvalEverTests(unittest.TestCase):
    def test_named_score_files_require_the_same_saved_outputs(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            log, exported, first, second, joined = (
                folder / name
                for name in (
                    "samples.jsonl",
                    "export.csv",
                    "a.jsonl",
                    "b.jsonl",
                    "joined.csv",
                )
            )
            save_samples(log, record("a", True), record("b", False))
            export_generations(log, exported)
            keys = [
                {name: row[name] for name in IDENTITY} for row in read_rows(exported)
            ]
            save_samples(first, *({**key, "confidence": 0.8} for key in keys))
            uncertainty = [
                {**key, "entropy": index + 1} for index, key in enumerate(keys)
            ]
            save_samples(second, *reversed(uncertainty))
            args = [
                "eee",
                "join",
                str(log),
                "--score-file",
                f"confidence={first}",
                "--score-file",
                f"entropy={second}",
                "--lower-is-better",
                "entropy",
                "--output",
                str(joined),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            self.assertEqual(
                [
                    (
                        row["id"],
                        row["confidence"],
                        row["negative_entropy"],
                        row["correct"],
                    )
                    for row in read_rows(joined)
                ],
                [("a", "0.8", "-1.0", "1"), ("b", "0.8", "-2.0", "0")],
            )
            previous = joined.read_bytes()
            for records, message in (
                (uncertainty[:1], "entropy: missing score for sample 'b'"),
                (
                    [*uncertainty, {**uncertainty[0], "id": "extra"}],
                    "entropy: score file has an unexpected sample identity",
                ),
            ):
                with self.subTest(message=message):
                    save_samples(second, *records)
                    with (
                        redirect_stderr(io.StringIO()) as errors,
                        self.assertRaises(SystemExit) as stopped,
                    ):
                        main(args)
                    self.assertEqual(stopped.exception.code, 2)
                    self.assertIn(message, errors.getvalue())
                    self.assertEqual(joined.read_bytes(), previous)

            args.append("--review-missing-scores")
            save_samples(second, uncertainty[0])
            unscored = {**keys[1], "confidence": None}
            for records in ((), (unscored,)):
                with self.subTest(records=records), redirect_stdout(io.StringIO()):
                    save_samples(first, *records)
                    self.assertEqual(main(args), 0)
                    self.assertEqual(
                        [
                            (
                                row["id"],
                                row["confidence"],
                                row["negative_entropy"],
                                row["correct"],
                            )
                            for row in read_rows(joined)
                        ],
                        [("a", "", "-1.0", "1"), ("b", "", "", "0")],
                    )

            previous = joined.read_bytes()
            save_samples(first, {**unscored, "output_sha256": "stale"})
            with (
                redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit) as stopped,
            ):
                main(args)
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("output key mismatch for sample 'b'", errors.getvalue())
            self.assertEqual(joined.read_bytes(), previous)
            log.rename(folder / "released.jsonl")

    def test_legacy_log_joins_external_scores_by_id_and_exact_output(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            log, exported, external, joined = (
                folder / name
                for name in ("samples.jsonl", "export.csv", "scores.csv", "joined.csv")
            )
            save_samples(
                log,
                record("s1", True),
                record("s2", False, input={"raw": "2 + 2", "formatted": None}),
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(["eee", "export", str(log), "--output", str(exported)]), 0
                )
            rows = read_rows(exported)
            self.assertEqual(
                [row["input"] for row in rows], ["Question: 2 + 2", "2 + 2"]
            )
            self.assertEqual(rows[0]["completion"], "4\n")

            def write_scores(digest=None):
                with external.open("w", newline="", encoding="utf-8") as output:
                    writer = csv.writer(output)
                    writer.writerow((*IDENTITY, "confidence"))
                    for row, confidence in zip(
                        reversed(rows), (0.25, 0.75), strict=True
                    ):
                        writer.writerow(
                            (
                                *[row[name] for name in IDENTITY[:-1]],
                                digest or row["output_sha256"],
                                confidence,
                            )
                        )

            write_scores()
            self.assertEqual(join_scores(log, external, joined), 2)
            self.assertEqual(
                [
                    (row["id"], row["confidence"], row["correct"])
                    for row in read_rows(joined)
                ],
                [("s1", "0.75", "1"), ("s2", "0.25", "0")],
            )
            self.assertEqual(
                join_scores(log, external, joined, lower_is_better=("confidence",)), 2
            )
            self.assertEqual(
                [row["negative_confidence"] for row in read_rows(joined)],
                ["-0.75", "-0.25"],
            )
            before = joined.read_bytes()
            write_scores(digest="revised-output")
            with self.assertRaisesRegex(ValueError, "output key mismatch"):
                join_scores(log, external, joined)
            self.assertEqual(joined.read_bytes(), before)

    def test_result_selection_and_ambiguous_generations(self):
        with TemporaryDirectory() as directory:
            log = Path(directory) / "samples.jsonl"
            output = Path(directory) / "export.csv"
            first = record(
                "s1", True, schema_version="0.3.0", evaluation_result_id="raw-match"
            )
            second = record(
                "s1",
                False,
                schema_version="0.3.0",
                evaluation_result_id="normalized-match",
            )
            save_samples(log, first, second)
            with self.assertRaisesRegex(ValueError, "several results"):
                export_generations(log, output)
            self.assertFalse(output.exists())
            self.assertEqual(
                export_generations(log, output, result="normalized-match"), 1
            )
            self.assertEqual(read_rows(output)[0]["result"], "normalized-match")
            save_samples(log, {**first, "output": {"raw": ["4", "four"]}})
            with self.assertRaisesRegex(ValueError, "needs one saved text response"):
                export_generations(log, output)
