# Import Every Eval Ever samples

Use a per-sample JSONL file with one text response and a boolean `evaluation.is_correct` per question. Export the samples for scoring. [Datastore](https://huggingface.co/datasets/evaleval/EEE_datastore)

```bash
caat eee export samples.jsonl --output replay/to-score.csv
```

Return a `confidence` column with the original `evaluation_id`, `result`, `id` and `output_sha256`. Join these scores to the saved grades.

```bash
caat eee join samples.jsonl --scores replay/scored.csv --output replay/graded.csv
caat policy replay/graded.csv --score-column confidence --correct-column correct --risk-target 0.10
```

Pass the same `--result NAME` to export and join when a file has several results. Repeat `--score-file COLUMN=FILE` to compare scores. Use `--review-missing-scores` to keep unscored rows for review.
