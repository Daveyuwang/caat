# Run a review policy from one file

On Python 3.10, install the workflow extra from the checkout.

```bash
python -m pip install ".[workflow]"
```

Run the bundled IFEval policy.

```bash
caat run examples/ifeval.toml
```

Open `replay/workflow/ifeval.html` for the score comparison, fitted probability
map metrics, cutoff candidates, held-out checks, and ranked review queue.
`caat run` without a path looks for `caat.toml` in the current directory.
Python 3.11 and later use the standard-library TOML reader. The `workflow`
extra supplies the reader on Python 3.10.
[Example configuration](../../examples/ifeval.toml) · [Python TOML reader](https://docs.python.org/3/library/tomllib.html)

Follow the held-out grader check and review queue in the
[IFEval tutorial notebook](../../examples/ifeval_review.ipynb).

## Run on your own data

Use one CSV or JSONL file with unique IDs, finite confidence scores, and
binary grader columns. Paths resolve from the directory containing the
TOML file.
[Prediction tables](csv-format.md)

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

The workflow ranks scores by normalized AURC gain on the design split
under the selection graders. The first score is the fallback when all
design gains are undefined or nonpositive. CAAT fits the probability map
and tests the design-derived cutoffs on calibration rows. The cutoff uses
the selected ranking score. The held-out check reuses that cutoff and
includes the additional graders in `check_columns`.
[Score choice at fixed risk targets](../research/benchmarks.md#choosing-a-score-for-a-risk-target)

With a confirmed cutoff, CAAT queues the lowest-scored rows below it.
When no cutoff qualifies or the held-out check fails, all held-out rows
are eligible for review. The queue contains up to the requested budget.

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

### Review unscored outputs

```toml
review_missing_scores = true
```

Keep every score column in the input. Use empty CSV cells or JSONL `null`
for unavailable scores. Each cutoff uses the full split as its coverage
denominator and can accept only scored outputs. Choose the first score
in the configuration as the fallback for `design-aurc`. That method
recommends scores that cover the full design split. With `joint-calibration`,
each score's available design values form its grid, and a score with no
design values contributes zero cutoffs.

The queue prioritizes unscored outputs by ID before scored outputs. The
JSON report records unscored queue entries as `null` scores. Saved cutoff
commands inherit the review rule from the report. Probability maps use
scored calibration rows, and `calibrate apply --map` leaves the probability
blank for unscored outputs. [Saved cutoffs](reports.md#reuse-a-saved-cutoff)

### Use a lower-is-better score

Name the original columns in `score_columns` and list the lower-is-better
columns in `lower_is_better`.

```toml
score_columns = ["perplexity", "token_logprob"]
lower_is_better = ["perplexity"]
```

The report names the oriented score `negative_perplexity`. Its cutoff
and default Platt map use those signed scores. The holdout export carries
the original columns. Save the workflow JSON with `json_output`.

`route`, `evaluate`, `audit`, and `monitor` use the saved orientation
from `--threshold-report`. `calibrate apply` also uses it from `--map`.
Set `--raw-score-column COLUMN` if the original score has a new name.
Use `--score-column score` for the already oriented `score` from a
decision export. [Use a saved cutoff](reports.md#reuse-a-saved-cutoff)

## Choose the score and cutoff together

Add this setting to `examples/ifeval.toml` or to your own workflow.

```toml
score_selection = "joint-calibration"
```

CAAT forms one grid per score using the design rows. On calibration rows,
`ltt-bonferroni` divides `failure_rate` by the total number of distinct
score-cutoff pairs. With `rule = "ltt-holm"`, CAAT orders the entire family
by p-value and stops at the first failure. At rank `i` out of `m` pairs it
uses `failure_rate / (m - i + 1)`. [Holm, 1979](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf)

CAAT selects the qualifying pair with the most accepted calibration rows.
Ties favor the smaller p-value, then the configured score order and
descending cutoffs. The holdout checks the selected pair under the
graders in `correct_columns` and `check_columns`.

The JSON report records the joint family size and the grids in
`policy.selection.candidate_family_size` and
`policy.selection.candidate_thresholds_by_score`. It records the p-values
and decisions for every pair in `policy.selection.candidates_by_score`.
The HTML candidate table shows the selected score's calibration results.
With Holm, pairs after the first failure are marked untested.
If zero cutoffs qualify, the report uses the design choice for score metrics
and sends all held-out rows to the review queue's eligible pool.
[Fitting and holdout comparison](../research/benchmarks.md#choosing-a-score-for-a-risk-target)

## Use existing splits

Supply three files to replay a time split or a shared benchmark split.
Put the `[predictions]` table after the other settings. Each split must
contain rows with IDs unique across the three files. Include the score
columns and `correct_columns` in every split. The `check_columns` are
required only in the holdout. With `output_key_column` set, each file
also needs that column for checking output identity. Each file also needs
the column named in `prediction_filter` when you set a filter.

```toml
score_columns = ["mean_token_logprob"]
correct_columns = ["correct"]
risk_target = 0.8
json_output = "workflow.json"
holdout_output = "exported-holdout.csv"

[predictions]
design = "design.csv"
calibration = "calibration.jsonl"
holdout = "holdout.csv"
```

The report records the three paths under `input.predictions` and sets
`policy.split.seed` to `null`. The paired bootstrap still uses `seed`.
The holdout export preserves the row order of the supplied holdout file.

```bash
caat run examples/omni_math.toml
```

The Omni-MATH example reuses the archived time split of 200 design,
400 calibration, and 400 holdout rows. The policy accepts 398 of 400
held-out answers with 278 errors and a 73.64% one-sided 95% upper error
bound. [Example](../../examples/omni_math.toml) · [Archive replay](monitor.md#chronological-omni-math-replay)

## Run with separate grader exports

Use `grader_files` for the graders that supply outcomes to score comparison,
cutoff fitting, and holdout verification. Each export needs the prediction
ID, output key, and binary `correct` grade. CAAT matches ID and output key
for every selected row in the design, calibration, and holdout partitions.

```toml
predictions = "inspect-scores.csv"
score_columns = ["confidence", "negative_token_count"]
grader_files = {original = "original-grades.csv"}
holdout_grader_files = {canonical = "canonical-grades.jsonl"}
risk_target = 0.40
json_output = "workflow.json"
holdout_output = "checked-holdout.csv"
```

Grade paths are relative to the configuration. The prediction key column
defaults to `output_sha256`. Use `grader_id_column` and
`grader_output_key_column` for different headers in the grade files.
The holdout CSV includes the matched grade columns.
[Separate grader exports](csv-format.md#separate-grader-exports)

## Check an external grader on the holdout

Add each new grader and its export path to `holdout_grader_files`.
The grader exports need the prediction ID column, the output key, and
a binary `correct` field. With these exports, `output_key_column` defaults
to `output_sha256`. Set `grader_id_column` or `grader_output_key_column`
when the grader files use different headers from the prediction table.
[Export grades from an Inspect log](inspect.md#check-a-rescored-log-at-a-frozen-cutoff)

```toml
predictions = "scored-predictions.csv"
score_columns = ["confidence"]
correct_columns = ["original"]
risk_target = 0.4
holdout_output = "checked-holdout.csv"
holdout_grader_files = {canonical = "rescored-grades.jsonl"}
```

CAAT joins these grades after computing or loading the split. Each held-out
output needs exactly one matching grade per file, using its ID and output key.
The files may cover a larger batch or contain older outputs for the same IDs.
The matched grades check the frozen cutoff. Score selection and fitting use
the design and calibration labels in `correct_columns`.

The holdout CSV appends the matched grade columns after the original columns.
Give each file a grader name distinct from the existing column names. The
workflow report lists these files under `input.holdout_grader_files`.

## Use in automation

Use `json_output` and `holdout_output` to save two companion files from
one run. For a report with a selected cutoff, pass these to `evaluate`,
`audit`, or the Inspect rescoring workflow. Both files are available for
completed runs with status 0 or 1.
[Saving a cutoff](reports.md#reuse-a-saved-cutoff) · [Inspect rescoring](inspect.md#check-a-rescored-log-at-a-frozen-cutoff)

Pass the workflow JSON as `--map` to `caat calibrate apply` to reuse its
fitted probability map on a new scored batch.
[Apply a saved probability map](calibration.md#apply-a-map-from-a-policy-report)

Add `--json` to print `selected_score`, the full `score_comparison`,
the `policy` report, and `review_queue`. The queue contains `eligible`,
`requested_budget`, and ordered `entries` with an `id` and raw `score`.
Exit code 0 means every grader passed the held-out check. Exit code 1
means the policy did not qualify and the report was written. Exit code 2
signals an invalid input or an I/O error. [Policy report fields](reports.md#policy)

## Fit a confidence scorer and route the next batch

Install the scorer extra to fit a standardized logistic regression. On Python 3.10 also install the workflow extra. The fitted parameters travel in the saved report. The base package can score and route a new batch from that report.

```bash
python -m pip install ".[scorer,workflow]"
caat run examples/fitted_classifier.toml
caat route new-digits.csv --threshold-report replay/fitted-classifier/workflow.json --output decisions.csv
```

The example uses the predicted and actual digit as its correctness label. A new batch needs only `image_id`, `top_prob` and `margin`. At the example's 2% target the fitted cutoff accepted 321 of 432 held-out predictions with one error. The one-sided 95% upper error bound was 1.47%. The scoring inputs and coefficients are available in the report's `scorer` field. [Classifier configuration](../../examples/fitted_classifier.toml)

Set `prediction_column` and `reference_column` together for exact matching, or supply `correct_columns` or `grader_files`. The first selection grader supplies the binary training outcome. A `[scorer]` table opts into the fitted score. Its `regularization` setting is the inverse L2 penalty C, default 0.1. Use `"auto"` to choose from 0.01, 0.1 and 1 using stratified design folds and mean normalized AURC gain. With fewer than two examples of either class, the fit uses C = 0.1. A single-class design yields a constant score. The logistic decision score ranks answers. The existing probability calibration step estimates correctness probabilities. With `score_selection = "joint-calibration"`, all raw and fitted score/cutoff pairs share the same multiple-testing correction.

With a fitted scorer, `refit_checks` defaults to 20. The workflow reshuffles only the original design and calibration rows, fits the scorer and cutoff again, and compares decisions on the same holdout IDs for each successive pair of fits. Every unqualified fit sends all rows to review. Supply an even number of checks, or zero to skip them. The report records the changed fraction, coverage and count of unqualified fits. Single-score workflows can request the same check explicitly.

To use a scorer independently, supply labeled design rows to `fit`. The artifact records feature names, signs, standardization, weights, intercept, training count and a digest of sorted design IDs. Keep design rows separate from calibration and audit rows. Use `--lower-is-better NAME` for a lower-is-better feature. With `--review-missing-scores`, fit uses complete design rows and apply leaves incomplete rows unscored.

```bash
caat scorer fit design.csv --score-column top_prob --score-column margin --correct-column correct --output scorer.json
caat scorer apply new-digits.csv --scorer scorer.json --output scored.csv
```

The Python equivalents are `fit_scorer`, `read_scorer` and `apply_scorer` in `caat.scorer`. Standalone apply also accepts a workflow JSON and writes a CSV with the extra `fitted_score` column. With the workflow report, `route`, `evaluate`, `audit` and `monitor` compute the same fitted score from the original features, including when the table contains a stale derived column. Routing requires a confirmed report. To use previously computed scores, pass `--score-column fitted_score`, or `--id-column id --score-column score` on a routed export. [Scorer benchmark](../research/routing-stability.md)

### Use a categorical prediction in a fitted scorer

Set `class_column = "predicted"` in the `[scorer]` table to learn an indicator for each class seen on complete design rows. The standalone equivalent is `caat scorer fit --class-column predicted` together with the numeric features and grading options. The prediction can also supply the predicted side of exact-match grading. Keep the reference label out of the features.

The saved version-2 scorer stores the design vocabulary, indicator order, scaling and weights. Give the same raw class column and numeric features to new batches. A missing or unseen value raises an error by default. With `review_missing_scores = true` in the workflow or `--review-missing-scores` on the CLI, incomplete training rows are omitted and unscorable predictions stay in the review population. Numeric-only scorers retain version 1. Both formats run in the base installation.

## Calibrate a frozen score

Bring labeled calibration and holdout files with disjoint IDs. For one existing scalar score, supply its column and predeclare cutoffs in the original column's units. A lower-is-better scalar uses the same `lower_is_better` setting as other workflows.

```toml
score_columns = ["confidence"]
correct_columns = ["correct"]
risk_target = 0.2
json_output = "policy.json"

[predictions]
calibration = "calibration.csv"
holdout = "holdout.csv"

[frozen]
thresholds = [0.7, 0.8, 0.9]
```

For rank-based cutoffs replace `thresholds` with `reference = "unlabeled.csv"`. CAAT builds the quantiles on that file using its IDs and scores. All calibration labels are available for testing the frozen cutoffs. To reuse a saved model, omit `score_columns` and set `scorer = "scorer.json"` in the frozen table. Its feature names, directions and weights travel with the new policy. Explicit cutoffs in that mode are on the saved scorer's decision scale.

The report records cutoffs on the acceptance scale, reference size, label counts, and the saved scorer's file digest. CAAT checks IDs between input partitions. For a scorer from a workflow with tracked output identities, it also checks all inputs against those identities and retains them for subsequent audits. A standalone 0.4 scorer records only a digest of the full training list. Its training overlap is reported as unavailable. Use an independent calibration cohort for such a model.
