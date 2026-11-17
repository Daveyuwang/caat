import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from caat.calibrate import apply_csv, fit_calibration
from caat.cli import main
from caat.table import GradedRow, join_grader_files, load_csv


class JsonlTests(unittest.TestCase):
    def test_orient_adds_signed_columns_for_the_selected_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "scores.jsonl", root / "oriented.csv"
            rows = [
                {
                    "id": "01",
                    "perplexity": 1.5,
                    "loss": 0.2,
                    "tier": "keep",
                    "text": "a, b\n",
                },
                {
                    "text": "second",
                    "tier": "keep",
                    "loss": -0.25,
                    "perplexity": 2,
                    "id": "02",
                },
                {
                    "id": "03",
                    "perplexity": "unused",
                    "loss": 0.3,
                    "tier": "skip",
                    "text": "other",
                },
            ]
            source.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            args = [
                "orient",
                str(source),
                "--lower-is-better",
                "perplexity",
                "--lower-is-better",
                "loss",
                "--prediction-filter",
                "tier=keep",
                "--output",
                str(output),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            with output.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(
                    list(csv.reader(stream)),
                    [
                        [
                            "id",
                            "perplexity",
                            "loss",
                            "tier",
                            "text",
                            "negative_perplexity",
                            "negative_loss",
                        ],
                        ["01", "1.5", "0.2", "keep", "a, b\n", "-1.5", "-0.2"],
                        ["02", "2", "-0.25", "keep", "second", "-2.0", "0.25"],
                    ],
                )
            previous = output.read_bytes()
            rows[1]["perplexity"] = None
            source.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            with (
                redirect_stderr(io.StringIO()) as stderr,
                self.assertRaises(SystemExit) as stopped,
            ):
                main(args)
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("invalid score at input line 2", stderr.getvalue())
            self.assertEqual(output.read_bytes(), previous)

            args.append("--review-missing-scores")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            with output.open(newline="", encoding="utf-8") as stream:
                first, second = list(csv.DictReader(stream))
                self.assertEqual(first["negative_perplexity"], "-1.5")
                self.assertEqual(
                    (
                        second["id"],
                        second["perplexity"],
                        second["negative_perplexity"],
                        second["negative_loss"],
                    ),
                    ("02", "", "", "0.25"),
                )
            previous = output.read_bytes()
            rows[1]["perplexity"] = "nan"
            for records, message in (
                (rows, "score 'perplexity' must be finite at input line 2"),
                (
                    [{**row, "negative_perplexity": 0} for row in rows],
                    "derived score columns already exist: negative_perplexity",
                ),
            ):
                with self.subTest(message=message):
                    source.write_text(
                        "\n".join(map(json.dumps, records)), encoding="utf-8"
                    )
                    with (
                        redirect_stderr(io.StringIO()) as stderr,
                        self.assertRaises(SystemExit) as stopped,
                    ):
                        main(args)
                    self.assertEqual(stopped.exception.code, 2)
                    self.assertIn(message, stderr.getvalue())
                    self.assertEqual(output.read_bytes(), previous)

    def test_scalar_values_keep_csv_grades_and_identity(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "answers.JSONL"
            path.write_text(
                '{"id":"001","score":0.25,"good":true,"group":"à","key":"a"}\n'
                '{"key":"b","id":2,"score":"0.5","good":0,"group":null}\n',
                encoding="utf-8-sig",
            )
            self.assertEqual(
                load_csv(path, correct_columns=("good",), output_key_column="key"),
                [
                    GradedRow("001", 0.25, (True,), "à", "a"),
                    GradedRow("2", 0.5, (False,), None, "b"),
                ],
            )

    def test_jsonl_shape_errors_include_the_physical_line(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "answers.jsonl"
            good = '{"id":"a","score":0.3,"correct":true}\n\n'
            cases = (
                ('["b",0.4,false]', "expected an object"),
                ('{"id":"a","id":"b","score":0.4,"correct":false}', "duplicate field"),
                ('{"id":"b","score":0.4}', "differs from the first object's fields"),
                ('{"id":"b","score":[0.4],"correct":false}', "needs a scalar"),
                ('{"id":"b","score":NaN,"correct":false}', "invalid JSON number"),
                ('{"id":"b","score":0.4,"correct":"yes"}', "0 or 1 at input line 3"),
            )
            for record, error in cases:
                with self.subTest(record=record):
                    path.write_text(good + record + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error) as raised:
                        load_csv(path, correct_columns=("correct",))
                    self.assertIn("line 3", str(raised.exception))

    def test_mixed_grader_files_align_numeric_ids_and_keep_output_keys(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            reference = root / "reference.csv"
            review = root / "review.jsonl"
            predictions.write_text(
                '{"id":2,"key":"k2","score":0.5}\n{"id":1,"key":"k1","score":0.7}\n',
                encoding="utf-8",
            )
            reference.write_text("id,key,correct\n1,k1,1\n2,k2,0\n", encoding="utf-8")
            review.write_text(
                '{"id":1,"key":"k1","correct":false}\n'
                '{"id":2,"key":"k2","correct":true}\n',
                encoding="utf-8",
            )
            options = {
                "id_column": "id",
                "score_column": "score",
                "output_key_column": "key",
            }
            graders = [("reference", reference), ("review", review)]
            rows, group = join_grader_files(predictions, graders, **options)
            self.assertIsNone(group)
            self.assertEqual(
                rows,
                [
                    GradedRow("2", 0.5, (False, True), output_key="k2"),
                    GradedRow("1", 0.7, (True, False), output_key="k1"),
                ],
            )
            review.write_text(
                review.read_text(encoding="utf-8").replace("k1", "stale"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "output key mismatch"):
                join_grader_files(predictions, graders, **options)

    def test_apply_map_keeps_jsonl_columns_in_the_csv_export(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "scores.jsonl"
            output = Path(directory) / "mapped.csv"
            rows = [
                {"id": str(index), "confidence": score, "ok": bool(index)}
                for index, score in enumerate((0.25, 0.75))
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            mapping = fit_calibration([0.25, 0.75], [False, True], method="isotonic")
            self.assertEqual(apply_csv(source, output, mapping, "confidence", "p"), 2)
            with output.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(
                    list(csv.DictReader(stream)),
                    [
                        {"id": "0", "confidence": "0.25", "ok": "0", "p": "0.0"},
                        {"id": "1", "confidence": "0.75", "ok": "1", "p": "1.0"},
                    ],
                )
