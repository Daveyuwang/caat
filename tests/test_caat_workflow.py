import csv
import json
import shutil
import subprocess
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@unittest.skipUnless(
    find_spec("tomllib") or find_spec("tomli"),
    "Requires the workflow extra on Python 3.10",
)
class WorkflowTests(unittest.TestCase):
    def setUp(self):
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name).resolve()
        self.config_dir = self.root / "evaluation data"
        self.config_dir.mkdir()
        self.config = self.config_dir / "ifeval.toml"
        self.predictions = self.config_dir / "ifeval_graders.csv"
        shutil.copyfile(EXAMPLES / self.config.name, self.config)
        shutil.copyfile(EXAMPLES / self.predictions.name, self.predictions)

    def caat(self, *arguments, cwd=None):
        return subprocess.run(
            [
                sys.executable,
                "-c",
                "from caat.cli import main; raise SystemExit(main())",
                *arguments,
            ],
            cwd=cwd or self.root,
            capture_output=True,
            encoding="utf-8",
        )

    def replay(self, status):
        result = self.caat("run", str(self.config), "--json")
        self.assertEqual(result.returncode, status, result.stderr)
        return json.loads(result.stdout)

    def split_heldout(self):
        with self.predictions.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            names = reader.fieldnames
            rows = list(reader)
        order = sorted(rows, key=lambda row: row["id"])
        Random(42).shuffle(order)
        return names, rows, order[3 * len(order) // 5 :]

    def test_ifeval_config_replays_the_policy_from_another_directory(self):
        report = self.replay(0)
        result = report["policy"]
        self.assertEqual(report["selected_score"], "sequence_logprob")
        self.assertEqual(
            result["split"],
            {"seed": 42, "design": 108, "calibration": 216, "heldout": 217},
        )
        cutoff = result["selection"]["selected_threshold"]
        self.assertEqual(cutoff, -24.3777582636898)
        checks = list(result["heldout"]["graders"].values())
        self.assertEqual(
            [(row["accepted"], row["errors"]) for row in checks], [(55, 11), (55, 15)]
        )
        self.assertTrue(all(row["risk_upper_bound"] <= 0.4 for row in checks))
        metrics = result["probability_calibration"]["graders"]["strict_pass"][
            "heldout_calibrated"
        ]
        self.assertAlmostEqual(metrics["brier_score"], 0.19453547401399615)
        _, _, heldout = self.split_heldout()
        waiting = sorted(
            (float(row["sequence_logprob"]), row["id"])
            for row in heldout
            if float(row["sequence_logprob"]) < cutoff
        )
        queue = report["review_queue"]
        self.assertEqual(queue["eligible"], len(waiting))
        self.assertEqual(
            queue["entries"],
            [{"id": key, "score": value} for value, key in waiting[:20]],
        )
        html = Path(report["input"]["html"])
        self.assertEqual(html, self.root / "replay/workflow/ifeval.html")
        body = html.read_text(encoding="utf-8")
        for section in (
            "Score choice",
            "Probability calibration",
            "Acceptance cutoff",
            "Held-out check",
            "Review queue",
        ):
            self.assertIn(f"<h2>{section}</h2>", body)
        self.assertIn("55 of 217", body)
        self.assertIn("Queued 20 of 162", body)

    def test_saved_cutoff_reads_lower_is_better_columns_and_signed_overrides(self):
        self.predictions.write_text(
            "id,loss,correct\n"
            + "".join(f"sample-{index},{index},1\n" for index in range(60)),
            encoding="utf-8",
        )
        self.config.write_text(
            'predictions = "ifeval_graders.csv"\n'
            'score_columns = ["loss"]\nlower_is_better = ["loss"]\n'
            'correct_columns = ["correct"]\nrisk_target = 0.4\n'
            'bootstrap = 0\njson_output = "policy.json"\n',
            encoding="utf-8",
        )
        report = self.replay(0)
        self.assertEqual(report["selected_score"], "negative_loss")
        self.assertEqual(report["input"]["negated_score_source"], "loss")
        policy = self.config_dir / "policy.json"
        raw, first, second = (
            self.root / name for name in ("raw.csv", "first.csv", "second.csv")
        )
        raw.write_text("id,loss\nlow,1\nhigh,100\n", encoding="utf-8")
        route = self.caat(
            "route", str(raw), "--threshold-report", str(policy), "--output", str(first)
        )
        self.assertEqual(route.returncode, 0, route.stderr)
        with first.open(encoding="utf-8", newline="") as stream:
            self.assertEqual(
                [(row["score"], row["decision"]) for row in csv.DictReader(stream)],
                [("-1.0", "accept"), ("-100.0", "review")],
            )
        converted = self.caat(
            "route",
            str(first),
            "--threshold-report",
            str(policy),
            "--score-column",
            "score",
            "--output",
            str(second),
        )
        self.assertEqual(converted.returncode, 0, converted.stderr)
        original = first.read_bytes()
        self.assertEqual(second.read_bytes(), original)
        renamed = self.root / "renamed.csv"
        renamed.write_text("id,uncertainty\nlow,1\nhigh,100\n", encoding="utf-8")
        reused = self.caat(
            "route",
            str(renamed),
            "--threshold-report",
            str(policy),
            "--raw-score-column",
            "uncertainty",
            "--output",
            str(second),
        )
        self.assertEqual(reused.returncode, 0, reused.stderr)
        self.assertEqual(second.read_bytes(), original)
        report["input"]["negated_score_source"] = "wrong"
        policy.write_text(json.dumps(report), encoding="utf-8")
        stale = self.caat(
            "route", str(raw), "--threshold-report", str(policy), "--output", str(first)
        )
        self.assertEqual(stale.returncode, 2)
        self.assertIn("invalid saved score direction", stale.stderr)
        self.assertEqual(first.read_bytes(), original)

    def test_saved_workflow_applies_the_review_rule_to_blank_scores(self):
        for name, size in (("design", 5), ("calibration", 8), ("holdout", 8)):
            (self.config_dir / f"{name}.csv").write_text(
                f"id,loss,absent,correct\n{name}-0,,,0\n"
                + "".join(f"{name}-{index},0.4,,1\n" for index in range(1, size)),
                encoding="utf-8",
            )
        self.config.write_text(
            'score_columns = ["loss", "absent"]\nlower_is_better = ["loss"]\n'
            'correct_columns = ["correct"]\nrisk_target = 0.5\n'
            'score_selection = "joint-calibration"\nreview_missing_scores = true\n'
            'bootstrap = 0\njson_output = "workflow.json"\nholdout_output = "export.csv"\n'
            '[predictions]\ndesign = "design.csv"\ncalibration = "calibration.csv"\n'
            'holdout = "holdout.csv"\n',
            encoding="utf-8",
        )
        report = self.replay(0)
        policy = report["policy"]
        self.assertIs(report["input"]["review_missing_scores"], True)
        self.assertEqual(
            policy["scored"], {"design": 4, "calibration": 7, "heldout": 7}
        )
        selection = policy["selection"]
        self.assertEqual(selection["selected_threshold"], -0.4)
        self.assertEqual(selection["candidate_family_size"], 1)
        self.assertEqual(selection["candidate_thresholds_by_score"]["absent"], [])
        self.assertEqual(policy["heldout"]["graders"]["correct"]["coverage"], 7 / 8)
        self.assertEqual(
            report["review_queue"]["entries"], [{"id": "holdout-0", "score": None}]
        )

        saved = self.config_dir / "workflow.json"
        holdout = self.config_dir / "export.csv"
        decisions = self.root / "decisions.csv"
        routed = self.caat(
            "route",
            str(holdout),
            "--threshold-report",
            str(saved),
            "--output",
            str(decisions),
        )
        self.assertEqual(routed.returncode, 0, routed.stderr)
        queued = self.root / "queued.csv"
        queue = self.caat(
            "queue",
            str(decisions),
            "--prediction-filter",
            "decision=review",
            "--review-missing-scores",
            "--budget",
            "1",
            "--output",
            str(queued),
        )
        self.assertEqual(queue.returncode, 0, queue.stderr)
        with queued.open(encoding="utf-8", newline="") as stream:
            self.assertEqual(
                list(csv.DictReader(stream)),
                [{"id": "holdout-0", "score": "", "group": ""}],
            )

        probabilities = self.root / "probabilities.csv"
        calibrated = self.caat(
            "calibrate",
            "apply",
            str(holdout),
            "--map",
            str(saved),
            "--output",
            str(probabilities),
        )
        self.assertEqual(calibrated.returncode, 0, calibrated.stderr)
        with probabilities.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(
            [row["id"] for row in rows], [f"holdout-{index}" for index in range(8)]
        )
        self.assertEqual(rows[0]["calibrated_probability"], "")
        self.assertTrue(
            all(0 <= float(row["calibrated_probability"]) <= 1 for row in rows[1:])
        )

    def test_workflow_renders_when_a_split_has_no_scores(self):
        self.config.write_text(
            'predictions = "ifeval_graders.csv"\nscore_columns = ["score"]\n'
            'correct_columns = ["correct"]\nrisk_target = 0.5\n'
            'score_selection = "joint-calibration"\nreview_missing_scores = true\n'
            "bootstrap = 0\n",
            encoding="utf-8",
        )
        ids = [f"sample-{index:02}" for index in range(40)]
        order = ids.copy()
        Random(42).shuffle(order)
        fitting = set(order[:24])
        for fit_map in (False, True):
            with self.subTest(fit_map=fit_map):
                self.predictions.write_text(
                    "id,score,correct\n"
                    + "".join(
                        f"{key},{'0.5' if fit_map and key in fitting else ''},1\n"
                        for key in ids
                    ),
                    encoding="utf-8",
                )
                report = self.replay(1)
                policy = report["policy"]
                body = Path(report["input"]["html"]).read_text(encoding="utf-8")
                self.assertEqual(policy["scored"]["heldout"], 0)
                self.assertEqual(report["review_queue"]["eligible"], 16)
                self.assertIn("<td>Unscored</td>", body)
                if fit_map:
                    estimates = policy["probability_calibration"]["graders"]["correct"]
                    self.assertIsNone(estimates["heldout_calibrated"]["brier_score"])
                    self.assertIn("<td>Undefined</td>", body)
                else:
                    self.assertEqual(policy["selection"]["candidate_family_size"], 0)
                    self.assertNotIn("probability_calibration", policy)
                    self.assertIn("No scored calibration rows", body)

    def test_heldout_grades_cannot_change_score_choice_map_or_cutoff(self):
        original = self.replay(0)
        names, rows, heldout = self.split_heldout()
        for row in heldout:
            row["strict_pass"] = "0"
            row["helm_strict_pass"] = "0"
        with self.predictions.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, names)
            writer.writeheader()
            writer.writerows(rows)
        altered = self.replay(1)
        self.assertEqual(altered["selected_score"], original["selected_score"])
        self.assertEqual(
            altered["policy"]["selection"], original["policy"]["selection"]
        )
        for report in (original, altered):
            self.assertEqual(
                report["policy"]["probability_calibration"]["threshold_scale"], "raw"
            )
        before = original["policy"]["probability_calibration"]["graders"]["strict_pass"]
        after = altered["policy"]["probability_calibration"]["graders"]["strict_pass"]
        self.assertEqual(after["map"], before["map"])
        self.assertEqual(altered["review_queue"]["eligible"], 217)
        self.assertIn(
            "Held-out check failed.",
            Path(altered["input"]["html"]).read_text(encoding="utf-8"),
        )

    def test_joint_calibration_fits_one_family_and_reuses_the_winning_score(self):
        with self.config.open("a", encoding="utf-8") as handle:
            handle.write(
                '\nscore_selection = "joint-calibration"\n'
                'json_output = "joint.json"\nholdout_output = "holdout.csv"\n'
            )
        report = self.replay(0)
        self.assertEqual(report["input"]["score_selection"], "joint-calibration")
        self.assertEqual(
            report["score_comparison"]["recommended_score"], "sequence_logprob"
        )
        self.assertEqual(report["selected_score"], "mean_token_logprob")
        selection = report["policy"]["selection"]
        self.assertEqual(selection["candidate_family_size"], 20)
        self.assertEqual(selection["per_candidate_level"], 0.05 / 20)
        self.assertEqual(
            {
                name: len(grid)
                for name, grid in selection["candidate_thresholds_by_score"].items()
            },
            {"mean_token_logprob": 10, "sequence_logprob": 10},
        )
        self.assertEqual(selection["selected_threshold"], -0.3683515132637169)
        checks = report["policy"]["heldout"]["graders"]
        self.assertEqual(
            [(row["accepted"], row["errors"]) for row in checks.values()],
            [(108, 22), (108, 28)],
        )
        self.assertIn(
            "20 score-cutoff pairs",
            Path(report["input"]["html"]).read_text(encoding="utf-8"),
        )

        routed = self.caat(
            "route",
            str(self.config_dir / "holdout.csv"),
            "--threshold-report",
            str(self.config_dir / "joint.json"),
            "--output",
            str(self.root / "decisions.csv"),
            "--json",
        )
        self.assertEqual(routed.returncode, 0, routed.stderr)
        self.assertEqual(json.loads(routed.stdout)["overall"]["accepted"], 108)

        names, rows, heldout = self.split_heldout()
        for row in heldout:
            row["strict_pass"] = row["helm_strict_pass"] = "0"
        with self.predictions.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, names)
            writer.writeheader()
            writer.writerows(rows)
        changed = self.replay(1)
        self.assertEqual(changed["selected_score"], report["selected_score"])
        self.assertEqual(changed["policy"]["selection"], selection)
        self.assertEqual(changed["review_queue"]["eligible"], 217)

    def test_joint_holm_reports_step_down_decisions_for_all_scores(self):
        self.config.write_text(
            self.config.read_text(encoding="utf-8").replace(
                'rule = "ltt-bonferroni"', 'rule = "ltt-holm"'
            )
            + '\nscore_selection = "joint-calibration"\n',
            encoding="utf-8",
        )
        report = self.replay(0)
        selection = report["policy"]["selection"]
        self.assertEqual(selection["rule"], "ltt-holm")
        self.assertEqual(selection["candidate_family_size"], 20)
        self.assertIsNone(selection["per_candidate_level"])
        candidates = [
            row for rows in selection["candidates_by_score"].values() for row in rows
        ]
        self.assertEqual(sum(row["tested"] for row in candidates), 6)
        self.assertEqual(sum(row["passes"] is True for row in candidates), 5)
        self.assertTrue(
            all(row["tested"] is (row["passes"] is not None) for row in candidates)
        )
        self.assertEqual(report["selected_score"], "mean_token_logprob")
        self.assertEqual(selection["selected_threshold"], -0.43964836263206303)
        for counts in report["policy"]["heldout"]["graders"].values():
            self.assertEqual(counts["accepted"], 144)
            self.assertTrue(counts["meets_target"])
        html = Path(report["input"]["html"]).read_text(encoding="utf-8")
        self.assertIn("Holm", html)
        self.assertIn("Untested", html)

    def test_default_config_empty_queue_and_error_preserves_report(self):
        config = self.root / "caat.toml"
        config.write_text(
            'predictions="rows.csv"\nscore_columns=["confidence"]\n'
            'correct_columns=["correct"]\nrisk_target=0.4\nrule="empirical"\n',
            encoding="utf-8",
        )
        (self.root / "rows.csv").write_text(
            "id,confidence,correct\n" + "".join(f"{i},0.9,1\n" for i in range(20)),
            encoding="utf-8",
        )
        result = self.caat("run", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["selected_score"], "confidence")
        self.assertEqual(
            payload["review_queue"],
            {"eligible": 0, "requested_budget": 10, "entries": []},
        )
        html = self.root / "caat-report.html"
        original = html.read_bytes()
        with config.open("a", encoding="utf-8") as target:
            target.write("invalid_setting = 2\n")
        result = self.caat("run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown workflow setting: invalid_setting", result.stderr)
        self.assertEqual(html.read_bytes(), original)

    def test_config_saves_the_holdout_and_its_report(self):
        names, rows, heldout = self.split_heldout()
        source = self.config_dir / "scored rows.jsonl"
        names = [*names, "tier"]
        rows = [{**row, "tier": "main"} for row in rows]
        skipped = {**rows[0], "tier": "other", "sequence_logprob": "unused"}
        source.write_text(
            "\n".join(map(json.dumps, [skipped, *rows])), encoding="utf-8"
        )
        options = self.config.read_text(encoding="utf-8").replace(
            "ifeval_graders.csv", source.name
        )
        options += (
            '\nprediction_filter = "tier=main"\n'
            'json_output = "artifacts/cutoff.json"\n'
            'holdout_output = "artifacts/heldout.csv"\n'
        )
        self.config.write_text(options, encoding="utf-8")

        completed = self.caat("run", str(self.config))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = self.config_dir / "artifacts" / "cutoff.json"
        export = self.config_dir / "artifacts" / "heldout.csv"
        payload = json.loads(report.read_bytes())
        self.assertEqual(
            payload["input"]["prediction_filter"], {"column": "tier", "value": "main"}
        )
        self.assertEqual(Path(payload["input"]["holdout_output"]), export)
        self.assertEqual(Path(payload["input"]["json_output"]), report)
        ids = {row["id"] for row in heldout}
        with export.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, names)
            self.assertEqual(list(reader), [row for row in rows if row["id"] in ids])

        checkpoint = export.read_bytes()
        self.config.write_text(
            options.replace("risk_target = 0.40", "risk_target = 0.0"), encoding="utf-8"
        )
        missed = self.replay(1)
        self.assertEqual(missed, json.loads(report.read_bytes()))
        self.assertIsNone(missed["policy"]["selection"]["selected_threshold"])
        self.assertEqual(export.read_bytes(), checkpoint)

        before = {
            path: path.read_bytes()
            for path in (source, report, export, Path(missed["input"]["html"]))
        }
        self.config.write_text(
            options.replace("artifacts/heldout.csv", source.name), encoding="utf-8"
        )
        rejected = self.caat("run", str(self.config))
        self.assertEqual(rejected.returncode, 2)
        self.assertIn(
            "holdout_output would overwrite another workflow file", rejected.stderr
        )
        self.assertEqual({path: path.read_bytes() for path in before}, before)

        blocked = self.config_dir / "blocked"
        blocked.write_text("already a file", encoding="utf-8")
        export.write_bytes(b"previous holdout\n")
        self.config.write_text(
            options.replace(
                "../replay/workflow/ifeval.html", "blocked/diagnostic.html"
            ),
            encoding="utf-8",
        )
        rejected = self.caat("run", str(self.config))
        self.assertEqual(rejected.returncode, 2)
        self.assertEqual(export.read_bytes(), b"previous holdout\n")
        self.assertEqual(report.read_bytes(), before[report])

    def test_provided_splits_preserve_order_and_keep_holdout_grades_out_of_fitting(
        self,
    ):
        config = (
            'score_columns = ["baseline", "quality"]\n'
            'correct_columns = ["correct"]\ncheck_columns = ["alternate"]\n'
            'output_key_column = "answer_key"\nrisk_target = 0.4\nbootstrap = 0\n'
            "design_grid_size = 2\n"
            'json_output = "saved.json"\nholdout_output = "export.csv"\n'
            '[predictions]\ndesign = "design.csv"\n'
            'calibration = "calibration.jsonl"\nholdout = "holdout.csv"\n'
        )
        self.config.write_text(config, encoding="utf-8")
        (self.config_dir / "design.csv").write_text(
            "id,baseline,quality,correct,answer_key\n"
            "design-a,0.9,0.1,0,answer-a\ndesign-b,0.8,0.2,0,answer-b\n"
            "design-c,0.2,0.8,1,answer-c\ndesign-d,0.1,0.9,1,answer-d\n",
            encoding="utf-8",
        )
        calibration = (
            {
                "id": f"calibration-{index}",
                "baseline": 0.2,
                "quality": 0.9,
                "correct": 1,
                "answer_key": f"calibrated-answer-{index}",
            }
            for index in range(12)
        )
        (self.config_dir / "calibration.jsonl").write_text(
            "\n".join(map(json.dumps, calibration)), encoding="utf-8"
        )
        holdout = self.config_dir / "holdout.csv"
        source = "id,baseline,quality,correct,alternate,answer_key\n" + "".join(
            f"holdout-{index},0.2,0.9,1,1,checked-answer-{index}\n"
            for index in (6, 1, 2, 5, 3, 4)
        )
        holdout.write_text(source, encoding="utf-8")

        first = self.replay(0)
        self.assertEqual(first["selected_score"], "quality")
        self.assertEqual(first["input"]["design_grid_size"], 2)
        self.assertEqual(len(first["policy"]["selection"]["candidates"]), 2)
        self.assertEqual(
            first["policy"]["split"],
            {"seed": None, "design": 4, "calibration": 12, "heldout": 6},
        )
        self.assertEqual(
            first["score_comparison"]["split"],
            {
                "seed": None,
                "comparison": "heldout",
                "design": 4,
                "calibration": 12,
                "reported": 6,
            },
        )
        self.assertEqual(Path(first["input"]["predictions"]["holdout"]), holdout)
        self.assertEqual(set(first["policy"]["ranking"]), {"correct"})
        self.assertEqual(len(first["input"]["fit_outputs"]["sha256"]), 16)
        self.assertEqual(
            first["policy"]["heldout"]["graders"]["alternate"]["accepted"], 6
        )
        exported = self.config_dir / "export.csv"
        self.assertEqual(exported.read_text(encoding="utf-8"), source)

        failed_grades = source.replace(",1,1,", ",0,0,")
        holdout.write_text(failed_grades, encoding="utf-8")
        second = self.replay(1)
        self.assertEqual(second["selected_score"], first["selected_score"])
        self.assertEqual(second["policy"]["selection"], first["policy"]["selection"])
        maps = [
            report["policy"]["probability_calibration"]["graders"]["correct"]["map"]
            for report in (first, second)
        ]
        self.assertEqual(*maps)
        self.assertEqual(exported.read_text(encoding="utf-8"), failed_grades)

        reports = (
            exported,
            Path(second["input"]["html"]),
            self.config_dir / "saved.json",
        )
        previous = [path.read_bytes() for path in reports]
        holdout.write_text(source.replace("holdout-6", "design-b"), encoding="utf-8")
        reused = self.caat("run", str(self.config))
        self.assertEqual(reused.returncode, 2)
        self.assertIn("predictions.holdout reuses sample ID 'design-b'", reused.stderr)

        self.config.write_text(
            config.replace('"export.csv"', '"holdout.csv"'), encoding="utf-8"
        )
        overwritten = self.caat("run", str(self.config))
        self.assertEqual(overwritten.returncode, 2)
        self.assertIn(
            "holdout_output would overwrite another workflow file", overwritten.stderr
        )
        self.assertEqual([path.read_bytes() for path in reports], previous)

    def test_workflow_reads_fitting_grades_for_joint_calibration(self):
        config = self.config.read_text(encoding="utf-8") + (
            '\nscore_selection = "joint-calibration"\nholdout_output = "holdout.csv"\n'
        )
        self.config.write_text(config, encoding="utf-8")
        baseline = self.replay(0)
        names, rows, heldout = self.split_heldout()
        raw_names = [name for name in names if name != "strict_pass"]
        keys = {row["id"]: f"answer-{row['id']}" for row in rows}
        with self.predictions.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, [*raw_names, "output_sha256"])
            writer.writeheader()
            writer.writerows(
                dict(
                    {name: row[name] for name in raw_names},
                    output_sha256=keys[row["id"]],
                )
                for row in rows
            )
        labels = self.config_dir / "fitting-grades.jsonl"
        with labels.open("w", encoding="utf-8") as target:
            for row in reversed(rows):
                target.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "output_sha256": keys[row["id"]],
                            "correct": int(row["strict_pass"]),
                        }
                    )
                    + "\n"
                )
        self.config.write_text(
            config.replace(
                'correct_columns = ["strict_pass"]',
                'grader_files = {strict_pass = "fitting-grades.jsonl"}',
            ),
            encoding="utf-8",
        )
        joined = self.replay(0)
        self.assertEqual(joined["input"]["grader_files"], {"strict_pass": str(labels)})
        for key in ("selected_score", "score_comparison", "policy", "review_queue"):
            self.assertEqual(joined[key], baseline[key])
        export = self.config_dir / "holdout.csv"
        with export.open(encoding="utf-8", newline="") as source:
            observed = {row["id"]: row["strict_pass"] for row in csv.DictReader(source)}
        self.assertEqual(observed, {row["id"]: row["strict_pass"] for row in heldout})

    def test_workflow_joins_full_grader_file_only_to_the_holdout(self):
        baseline = self.replay(0)
        names, rows, holdout = self.split_heldout()
        raw_columns = [name for name in names if name != "helm_strict_pass"]
        predictions = [
            {
                **{name: row[name] for name in raw_columns},
                "output_sha256": f"out-{row['id']}",
            }
            for row in rows
        ]
        with self.predictions.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, predictions[0].keys())
            writer.writeheader()
            writer.writerows(predictions)
        grades = [
            {
                "grade_id": row["id"],
                "graded_key": f"out-{row['id']}",
                "correct": row["helm_strict_pass"],
            }
            for row in reversed(rows)
        ]
        stale_id = holdout[0]["id"]
        old_grade = {"grade_id": stale_id, "graded_key": "old-output", "correct": "0"}
        labels = self.config_dir / "grades.jsonl"
        labels.write_text(
            "\n".join(map(json.dumps, [old_grade, *grades])), encoding="utf-8"
        )
        config = self.config.read_text(encoding="utf-8").replace(
            'check_columns = ["helm_strict_pass"]',
            'holdout_grader_files = {helm_strict_pass = "grades.jsonl"}\n'
            'grader_id_column = "grade_id"\n'
            'grader_output_key_column = "graded_key"',
        )
        config += '\njson_output = "workflow.json"\nholdout_output = "heldout.csv"\n'
        self.config.write_text(config, encoding="utf-8")
        joined = self.replay(0)
        self.assertEqual(joined["input"]["grader_id_column"], "grade_id")
        self.assertEqual(joined["policy"]["selection"], baseline["policy"]["selection"])
        self.assertEqual(joined["policy"]["heldout"], baseline["policy"]["heldout"])
        self.assertEqual(set(joined["policy"]["ranking"]), {"strict_pass"})

        export = self.config_dir / "heldout.csv"
        grades_by_id = {row["grade_id"]: row["correct"] for row in grades}
        heldout_ids = {row["id"] for row in holdout}
        with export.open(newline="", encoding="utf-8") as source:
            self.assertEqual(
                list(csv.DictReader(source)),
                [
                    dict(row, helm_strict_pass=grades_by_id[row["id"]])
                    for row in predictions
                    if row["id"] in heldout_ids
                ],
            )

        outputs = (
            export,
            Path(joined["input"]["json_output"]),
            Path(joined["input"]["html"]),
        )
        previous = [path.read_bytes() for path in outputs]
        labels.write_text(
            "\n".join(
                map(
                    json.dumps,
                    [
                        old_grade,
                        *(row for row in grades if row["grade_id"] != stale_id),
                    ],
                )
            ),
            encoding="utf-8",
        )
        rejected = self.caat("run", str(self.config))
        self.assertEqual(rejected.returncode, 2)
        self.assertIn(
            f"helm_strict_pass: output key mismatch for {stale_id!r}", rejected.stderr
        )
        self.assertEqual([path.read_bytes() for path in outputs], previous)
