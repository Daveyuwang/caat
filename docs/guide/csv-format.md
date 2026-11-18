# Prediction tables

Supply one row per prediction in UTF-8 CSV or JSONL. The ID must be unique. The default score column is `score`. Larger finite scores indicate greater confidence. For exact-match grading, supply the predicted and reference labels. For an existing grade, supply a 0/1 correctness column.

```csv
id,score,correct,group
001,0.92,1,math
002,0.34,0,history
```

## JSON Lines

Use one flat object per line with consistent fields. Use strings for IDs with leading zeros and `null` for an empty optional value.

```jsonl
{"id":"001","score":0.92,"correct":true,"group":"math"}
{"id":"002","score":0.34,"correct":false,"group":null}
```

```bash
caat evaluate predictions.jsonl --correct-column correct --threshold 0.8
```

## Separate grader exports

Give each grader a name and match its labels to the prediction IDs and output keys. An output key identifies a particular saved answer, including when the same question has multiple outputs. Use a stable key such as a run-specific response ID or the output text.

```bash
caat audit examples/audit/predictions.csv --output-key-column output_key --grader-file reference=examples/audit/reference.csv --grader-file rubric=examples/audit/rubric.csv --threshold 0.75
```

Use `--grader-id-column` or `--grader-output-key-column` for different headers. Use `--allow-extra-grades` when the grader export covers a larger cohort.

## Use lower-is-better scores

Set `--direction lower` when fitting a cutoff on perplexity or another lower-is-better score. The saved report records the original direction, so a new batch can supply the original values. [Workflow settings](workflow.md#settings)
