# Run a review policy

Run the bundled IFEval workflow from a checkout. Python 3.10 needs the workflow extra for TOML support.

```bash
python -m pip install ".[workflow]"
caat run examples/ifeval.toml
```

Open `replay/workflow/ifeval.html` to see score rankings, held-out results and the review queue. [Example configuration](../../examples/ifeval.toml) · [Runnable notebook](../../examples/ifeval_review.ipynb)

## Run on your own data

Paths resolve from the TOML file. With one predictions file, CAAT splits the rows by ID into design, calibration and holdout. You can instead supply these three files under `[predictions]`. [Separate-split example](../../examples/omni_math.toml)

```toml
predictions = "predictions.csv"
score_columns = ["token_confidence", "judge_confidence"]
correct_columns = ["primary_pass"]
check_columns = ["alternate_pass"]
risk_target = 0.20
queue_budget = 25
html = "review.html"
json_output = "workflow.json"
holdout_output = "holdout.csv"
```

Design rows choose the score. Calibration rows select a cutoff. Every required grader then checks that cutoff on the holdout. If the policy passes, the review queue prioritizes low-scoring rows below the cutoff. For a withheld policy, the queue can include any held-out row.

## Settings

| Setting | Default or requirement |
| --- | --- |
| `predictions` | Required CSV or JSONL path, or a table of paths for `design`, `calibration`, and `holdout`. |
| `score_columns` | Required list of score columns. The first is also the comparison baseline. |
| `lower_is_better` | Optional list of score columns to negate before ranking, fitting, and checking cutoffs. |
| `correct_columns` | List of inline grader columns for score choice and cutoff selection. Provide this or `grader_files`. |
| `grader_files` | Named CSV or JSONL grader files for score comparison, fitting, and holdout checks. Provide this or `correct_columns`. |
| `risk_target` | Required upper limit on accepted errors, between 0 and 1. |
| `id_column` | `id`. |
| `output_key_column` | Optional. Save design and calibration identities for later holdout checks. |
| `prediction_filter` | Optional `"COLUMN=VALUE"` filter. It selects rows before CAAT splits the file or reads each supplied partition. |
| `review_missing_scores` | `false`. Set `true` to keep unscored outputs in the cohort and send them to review. |
| `check_columns` | Empty by default. Each named grader checks the frozen cutoff on the holdout. |
| `holdout_grader_files` | Optional table of names and CSV or JSONL paths for additional holdout checks. |
| `grader_id_column` | Optional ID column in the separate grader files. Defaults to `id_column`. |
| `grader_output_key_column` | Optional key column in the separate grader files. Defaults to the prediction key column. |
| `score_selection` | `design-aurc`. Set `joint-calibration` to test all score-cutoff pairs on calibration rows with `ltt-bonferroni` or `ltt-holm`. |
| `rule` | `ltt-bonferroni`. |
| `failure_rate` | `0.05` for cutoff selection. |
| `design_grid_size` | `10` design quantiles. Use a positive integer. Duplicate cutoffs count once. |
| `holdout_confidence` | `0.95` for the one-sided upper error bound. |
| `calibrate` | `platt`. |
| `calibration_input` | The calibrator's default input scale. Set `score` or `probability` as appropriate. |
| `seed` | `42` for splits generated from a single file and for the paired score bootstrap. |
| `bootstrap` | `500` paired resamples for score comparisons. Set `0` to omit intervals. |
| `queue_budget` | `10` rows. |
| `html` | `caat-report.html`. |
| `json_output` | Optional JSON report for automation and cutoff reuse. |
| `holdout_output` | Optional CSV with the run's holdout rows and all original columns in input order. |

## Fit a confidence scorer and route the next batch

Training needs the scorer extra. The saved report includes the fitted coefficients, so the base CLI can score a new batch.

```bash
python -m pip install ".[scorer,workflow]"
caat run examples/fitted_classifier.toml
caat route examples/fitted_batch.csv --threshold-report replay/fitted-classifier/workflow.json --output replay/decisions.csv
caat evaluate examples/fitted_batch_graded.csv --threshold-report replay/fitted-classifier/workflow.json --correct-column correct
caat audit examples/fitted_batch_graded.csv --threshold-report replay/fitted-classifier/workflow.json --correct-column correct --correct-column second_grade
caat monitor examples/fitted_batch_graded.csv --threshold-report replay/fitted-classifier/workflow.json --correct-column correct
```

The policy accepts `new-a` and sends `new-b` to review. The one accepted grade gives insufficient evidence for a 2% target, so evaluation and audit return status 1. [Class-aware scoring example](classifier.md#hand-off-a-class-aware-policy)
