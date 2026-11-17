import csv
import gc
import io
import json
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

from caat.cli import main
from caat.inspect_log import IDENTITY


@unittest.skipUnless(find_spec("inspect_ai"), "Inspect extra required")
class InspectImportTests(unittest.TestCase):
    def setUp(self):
        from inspect_ai.log import (
            EvalConfig,
            EvalDataset,
            EvalLog,
            EvalSample,
            EvalSpec,
            write_eval_log,
        )
        from inspect_ai.model import ChatMessageUser, ModelOutput
        from inspect_ai.scorer import Score

        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.log = self.root / "example.eval"
        self.scoring_input = self.root / "scoring-input.csv"
        self.scores = self.root / "scores.csv"
        self.output = self.root / "graded.csv"
        log = EvalLog(
            status="success",
            eval=EvalSpec(
                created="2026-11-16T00:00:00+00:00",
                task="test-import",
                model="mockllm/test-import",
                eval_id="test-import",
                dataset=EvalDataset(samples=2),
                config=EvalConfig(epochs=2),
            ),
            samples=[
                EvalSample(
                    id=item_id,
                    epoch=epoch,
                    input=[ChatMessageUser(content="question")]
                    if item_id == "b"
                    else "question",
                    target="reference",
                    output=ModelOutput.from_content(
                        "mockllm/test-import", "answer, text\nline"
                    ),
                    scores={"primary": Score(value="C"), "alternate": Score(value=0)},
                )
                for epoch in (1, 2)
                for item_id in ("a", "b")
            ],
        )
        write_eval_log(log, self.log)
        self.assertEqual(
            self.run_cli("export", str(self.log), "--output", str(self.scoring_input)),
            0,
        )

    def run_cli(self, *args):
        with redirect_stdout(io.StringIO()):
            return main(["inspect", *args])

    def read_csv(self, path):
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def write_scores(self, rows):
        with self.scores.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, (*IDENTITY, "confidence"), extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(
                {**row, "confidence": confidence} for row, confidence in rows
            )

    def join(self):
        return self.run_cli(
            "join",
            str(self.log),
            "--scores",
            str(self.scores),
            "--output",
            str(self.output),
            "--grader",
            "primary",
            "--grader",
            "alternate",
        )

    def test_native_log_exports_one_epoch_and_matches_reordered_scores(self):
        first, second = self.read_csv(self.scoring_input)
        self.assertEqual({first["epoch"], second["epoch"]}, {"1"})
        self.assertEqual(first["completion"], "answer, text\nline")
        self.assertEqual(json.loads(second["input"])[0]["content"], "question")
        self.assertNotIn("target", first)
        self.write_scores([(second, 0.25), (first, 0.75)])
        self.assertEqual(self.join(), 0)
        rows = self.read_csv(self.output)
        self.assertEqual(
            [(row["id"], row["confidence"]) for row in rows],
            [("a", "0.75"), ("b", "0.25")],
        )
        self.assertTrue(
            all(row["primary"] == "1" and row["alternate"] == "0" for row in rows)
        )

    def test_indexed_eval_releases_the_archive_after_export(self):
        from inspect_ai.log import read_eval_log, write_eval_log

        indexed = self.root / "indexed.eval"
        with self.log.open("rb") as source:
            log = read_eval_log(source)
        self.assertEqual(log.eval.dataset.sample_ids, ["a", "b"])
        write_eval_log(log, indexed)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", ResourceWarning)
            self.assertEqual(
                self.run_cli("export", str(indexed), "--output", str(self.output)), 0
            )
            indexed.rename(self.root / "moved.eval")
            gc.collect()
        self.assertEqual(self.output.read_bytes(), self.scoring_input.read_bytes())
        leaks = [
            str(warning.message)
            for warning in recorded
            if issubclass(warning.category, ResourceWarning)
            and indexed.name in str(warning.message)
        ]
        self.assertEqual(leaks, [])

    def test_export_includes_samples_added_at_runtime(self):
        from inspect_ai.log import read_eval_log, write_eval_log

        with self.log.open("rb") as source:
            log = read_eval_log(source)
        log.eval.dataset.sample_ids = ["b"]
        for suffix in (".eval", ".json"):
            with self.subTest(suffix=suffix):
                path = self.log.with_suffix(suffix)
                write_eval_log(log, path)
                self.assertEqual(
                    self.run_cli("export", str(path), "--output", str(self.output)), 0
                )
                self.assertEqual(
                    [row["id"] for row in self.read_csv(self.output)], ["b", "a"]
                )

    def test_json_grades_are_reloaded_after_replacing_the_log(self):
        from inspect_ai.log import read_eval_log, write_eval_log
        from inspect_ai.scorer import Score

        with self.log.open("rb") as source:
            log = read_eval_log(source)
        path = self.root / "run.json"
        write_eval_log(log, path)
        command = (
            "grades",
            str(path),
            "--predictions",
            str(self.scoring_input),
            "--grader",
            "primary",
            "--output",
            str(self.output),
        )
        self.assertEqual(self.run_cli(*command), 0)
        original = self.read_csv(self.output)
        self.assertEqual([row["correct"] for row in original], ["1", "1"])

        log.samples[0].scores["primary"] = Score(value="I")
        replacement = self.root / "replacement.json"
        write_eval_log(log, replacement)
        replacement.replace(path)
        self.assertEqual(self.run_cli(*command), 0)
        self.assertEqual(
            self.read_csv(self.output), [{**original[0], "correct": "0"}, original[1]]
        )

    def test_json_checks_declared_samples_in_the_selected_epoch(self):
        from inspect_ai.log import read_eval_log, write_eval_log

        with self.log.open("rb") as source:
            log = read_eval_log(source)
        log.samples = [
            sample for sample in log.samples if (sample.id, sample.epoch) != ("b", 1)
        ]
        path = self.root / "sparse.json"
        write_eval_log(log, path)

        command = ("export", str(path), "--output", str(self.output))
        self.output.write_bytes(b"previous export\n")
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            self.run_cli(*command)
        self.assertEqual(error.exception.code, 2)
        self.assertIn("sample 'b' is missing from epoch 1", stderr.getvalue())
        self.assertEqual(self.output.read_bytes(), b"previous export\n")

        log.eval.dataset.sample_ids = None
        write_eval_log(log, path)
        self.assertEqual(self.run_cli(*command), 0)
        self.assertEqual([row["id"] for row in self.read_csv(self.output)], ["a"])

    def test_missing_indexed_sample_preserves_the_output(self):
        from inspect_ai.log import read_eval_log, write_eval_log

        indexed = self.root / "indexed.eval"
        with self.log.open("rb") as source:
            write_eval_log(read_eval_log(source), indexed)
        incomplete = self.root / "missing.eval"
        missing = "samples/b_epoch_1.json"
        with ZipFile(indexed) as source, ZipFile(incomplete, "w") as target:
            self.assertIn(missing, source.namelist())
            for entry in source.infolist():
                if entry.filename != missing:
                    target.writestr(entry, source.read(entry))
        self.output.write_bytes(b"previous export\n")
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            self.run_cli("export", str(incomplete), "--output", str(self.output))
        self.assertEqual(error.exception.code, 2)
        self.assertIn("could not read Inspect log", stderr.getvalue())
        self.assertEqual(self.output.read_bytes(), b"previous export\n")

    def test_attached_prompt_exports_to_csv_and_jsonl_and_joins(self):
        from inspect_ai.log import read_eval_log, write_eval_log
        from inspect_ai.model import ChatMessageUser, ContentImage

        with self.log.open("rb") as source:
            log = read_eval_log(source)
        image = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/gQ0AAAAASUVORK5CYII="
        )
        log.samples[0].input = [
            ChatMessageUser(content=[ContentImage(image="attachment://picture")])
        ]
        log.samples[0].attachments = {"picture": image}
        write_eval_log(log, self.log)

        jsonl = self.root / "scoring-input.jsonl"
        for output in (self.scoring_input, jsonl):
            self.assertEqual(
                self.run_cli("export", str(self.log), "--output", str(output)), 0
            )
        records = [
            json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(records, self.read_csv(self.scoring_input))
        self.assertEqual(
            json.loads(records[0]["input"])[0]["content"][0]["image"], image
        )
        self.assertEqual(records[0]["completion"], "answer, text\nline")

        self.scores = self.root / "scores.jsonl"
        self.scores.write_text(
            "".join(json.dumps({**row, "confidence": 0.75}) + "\n" for row in records),
            encoding="utf-8",
        )
        self.assertEqual(self.join(), 0)
        joined = self.read_csv(self.output)
        self.assertEqual(
            [row["output_sha256"] for row in joined],
            [row["output_sha256"] for row in records],
        )

    def test_invalid_scores_keep_the_previous_output(self):
        first, second = self.read_csv(self.scoring_input)
        self.output.write_text("previous output\n", encoding="utf-8")
        for rows, message in (
            ([(second, 0.25)], "missing score"),
            ([(first, 0.75), (first, 0.75)], "duplicate score identity"),
            ([(first, float("nan")), (second, 0.25)], "must be finite"),
            (
                [({**first, "output_sha256": "0" * 64}, 0.75), (second, 0.25)],
                "output key mismatch",
            ),
            (
                [(first, 0.75), (second, 0.25), ({**first, "id": "c"}, 0.2)],
                "unexpected sample identity",
            ),
        ):
            with (
                self.subTest(message=message),
                redirect_stderr(io.StringIO()) as errors,
            ):
                self.write_scores(rows)
                with self.assertRaises(SystemExit) as rejected:
                    self.join()
                self.assertEqual(rejected.exception.code, 2)
                self.assertIn(message, errors.getvalue())
                self.assertEqual(
                    self.output.read_text(encoding="utf-8"), "previous output\n"
                )

    def test_join_aligns_each_named_score_file(self):
        first, second = self.read_csv(self.scoring_input)
        self.write_scores([(second, 0.25), (first, 0.75)])
        lengths = self.root / "lengths.jsonl"
        records = [
            {**{name: row[name] for name in IDENTITY}, "token_count": count}
            for row, count in ((first, 5), (second, 2))
        ]
        args = (
            "join",
            str(self.log),
            "--score-file",
            f"confidence={self.scores}",
            "--score-file",
            f"token_count={lengths}",
            "--lower-is-better",
            "token_count",
            "--grader",
            "primary",
            "--output",
            str(self.output),
        )
        lengths.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
        self.assertEqual(self.run_cli(*args), 0)
        rows = self.read_csv(self.output)
        self.assertEqual(
            list(rows[0]),
            [*IDENTITY, "confidence", "token_count", "negative_token_count", "primary"],
        )
        self.assertEqual(
            [
                (
                    row["id"],
                    row["confidence"],
                    row["token_count"],
                    row["negative_token_count"],
                )
                for row in rows
            ],
            [("a", "0.75", "5.0", "-5.0"), ("b", "0.25", "2.0", "-2.0")],
        )

        previous = self.output.read_bytes()
        for extra, message in (
            (
                ("--lower-is-better", "missing"),
                "needs an external score column: missing",
            ),
            (
                ("--score-file", f"negative_token_count={lengths}"),
                "provide distinct grader, score, and identity column names",
            ),
        ):
            with (
                redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit) as rejected,
            ):
                self.run_cli(*args, *extra)
            self.assertEqual(rejected.exception.code, 2)
            self.assertIn(message, errors.getvalue())
            self.assertEqual(self.output.read_bytes(), previous)

        records[1]["output_sha256"] = "0" * 64
        lengths.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
        with (
            self.assertRaises(SystemExit) as rejected,
            redirect_stderr(io.StringIO()) as errors,
        ):
            self.run_cli(*args)
        self.assertEqual(rejected.exception.code, 2)
        self.assertIn("token_count: output key mismatch", errors.getvalue())
        self.assertEqual(self.output.read_bytes(), previous)

    def test_export_grades_checks_the_holdout_identity(self):
        from inspect_ai.log import (
            read_eval_log,
            read_eval_log_samples_by_id,
            write_eval_log,
        )

        first, second = self.read_csv(self.scoring_input)
        self.write_scores([(first, 0.75), (second, 0.25)])
        self.assertEqual(self.join(), 0)
        heldout = self.read_csv(self.output)[1]
        predictions = self.root / "holdout.jsonl"
        grades = self.root / "grades.csv"

        def export(row, log):
            predictions.write_text(json.dumps(row) + "\n", encoding="utf-8")
            return self.run_cli(
                "grades",
                str(log),
                "--predictions",
                str(predictions),
                "--grader",
                "alternate",
                "--output",
                str(grades),
            )

        self.assertEqual(export(heldout, self.log), 0)
        self.assertEqual(
            self.read_csv(grades),
            [
                {
                    "id": heldout["id"],
                    "output_sha256": heldout["output_sha256"],
                    "correct": "0",
                }
            ],
        )
        previous = grades.read_bytes()
        indexed = self.root / "indexed.eval"
        with self.log.open("rb") as source:
            write_eval_log(read_eval_log(source), indexed)
        with patch(
            "inspect_ai.log.read_eval_log_samples_by_id",
            wraps=read_eval_log_samples_by_id,
        ) as read:
            self.assertEqual(export(heldout, indexed), 0)
        self.assertEqual(read.call_count, 1)
        self.assertEqual(read.call_args.args[1], [(heldout["id"], 1)])
        self.assertEqual(grades.read_bytes(), previous)
        for column, value, error in (
            ("output_sha256", "0" * 64, "Inspect identity mismatch"),
            ("eval_id", "another-run", "Inspect identity mismatch"),
            ("epoch", "2", "Inspect identity mismatch"),
            ("id", "unknown", "missing 1 prediction IDs"),
        ):
            with (
                self.subTest(column=column),
                redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit) as rejected,
            ):
                export({**heldout, column: value}, indexed)
            self.assertEqual(rejected.exception.code, 2)
            self.assertIn(error, errors.getvalue())
            self.assertEqual(grades.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
