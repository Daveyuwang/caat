# CSV and JSONL input

Start from saved outputs with one row per unique item. Use UTF-8 CSV
with a header row or a UTF-8 `.jsonl` file. `--id-column` and
`--score-column` default to `id` and `score`. Larger finite scores rank
ahead of smaller ones. The threshold keeps every row whose score is
at least that value.

In CSV, quote fields containing commas or line breaks. Keep an empty field
for each empty cell so every record has the same number of fields as the
header. CAAT reports the file and line for malformed records.

## JSON Lines

Write one flat object per line and use the same field names in each
object. String IDs preserve leading zeros. Numbers keep their written
form when used as IDs or output keys. CAAT reads `true` and `false` as
`1` and `0`. Use `null` for an empty optional value, such as `group`.

```jsonl
{"id":"001","score":0.92,"correct":true,"group":"math"}
{"id":"002","score":0.34,"correct":false,"group":null}
```

```bash
caat evaluate predictions.jsonl --correct-column correct --threshold 0.8
```

`evaluate`, `scores`, `select`, `policy`, `audit`, `monitor`, `calibrate`, `queue`,
`orient`, and `run` accept both formats. Separate design and grader exports and
external score files for `inspect join`, `lm-eval join`, and `eee join`
can use either format. Calibration and queue exports are CSV.

## Use lower-is-better scores

Use `--direction lower` with `caat select` or `caat policy` for a score
such as perplexity. The report records the signed cutoff and the source
column. New batches can supply the original scores.

```bash
caat policy graded.csv --score-column perplexity --direction lower --correct-column correct --risk-target 0.40 --holdout-confidence 0.95 --output policy.json
caat route new-predictions.jsonl --threshold-report policy.json --output decisions.csv
```

For a standalone table with signed scores, run `caat orient` with a
`--lower-is-better COLUMN` option for each score to negate. The CSV
contains the original fields and the derived `negative_COLUMN` fields.
Add `--review-missing-scores` to carry empty CSV scores or JSONL `null`
scores through as blank cells in the derived columns. Use the same
option when comparing or fitting with those columns.

[Apply a saved cutoff](queue.md#route-a-scored-batch) ·
[Calibrate lower-is-better scores](calibration.md#fit-on-lower-is-better-scores) ·
[Save score direction in a TOML workflow](workflow.md#use-a-lower-is-better-score)

## Grades in the prediction file

Provide a 0/1 outcome for each grader. Repeat `--correct-column` in the
order you want CAAT to use. `evaluate` uses one grader.

Without a 0/1 column, `evaluate`, `policy`, `select`, and `scores` grade by
exact match between `--prediction-column` and `--reference-column`, which
default to `prediction` and `reference`. The match is one grader, named
after the reference column in reports. In `evaluate`, supply
`--reference-column` with an explicit outcome column to group results by
reference label.
`policy --check-column` adds graders for the held-out comparison.

`--group-column` identifies groups for reporting. CAAT automatically
uses a column named `group` when present. `--output-key-column` declares
the saved response identity when the file includes one.

## Select a run from a combined export

Use `--prediction-filter COLUMN=VALUE` with `scores`, `select`, `policy`,
`route`, `evaluate`, `audit`, `orient`, `queue`, `calibrate fit`, or
`calibrate apply` to read one run or tier. CAAT selects matching rows
before checking IDs, scores, and grades. With
`select --design`, the filter also selects rows in the design file.
Analysis reports record the filter in `input.prediction_filter`. In a TOML workflow,
set `prediction_filter = "tier=encoder"`.
[Fit a cutoff on a mixed export](ci-audit.md#fit-a-cutoff-on-the-encoder-tier) · [Calibrate one tier](calibration.md#calibrate-one-tier-from-a-combined-export) · [TOML settings](workflow.md#run-on-your-own-data)

## Check a policy on your own data

This example uses `answer_id`, `confidence`, and `passed` as the input
fields. Set the filename, field names, and accepted-error target for
your data.

```bash
caat policy predictions.jsonl --id-column answer_id --score-column confidence --correct-column passed --risk-target 0.40 --holdout-confidence 0.95 --output policy.json
```

Read `heldout.graders.passed` for the accepted count, observed error
rate, and one-sided error bound. The report uses your correctness field
as the grader name. If calibration finds no qualifying cutoff, `heldout`
is null and the command exits with status 1. [Policy report fields](reports.md#policy)

## Separate grader exports

Each grader file needs the prediction ID, the output key, and a `correct`
column with 0/1 grades. The files can put rows in different orders and use
CSV and JSONL in the same command. Repeat `--grader-file NAME=FILE` with
`scores`, `select`, `policy`, or `audit`. `calibrate fit` and `evaluate` take one
grader file each.
The default output key column is `output_sha256`. Set
`--output-key-column` to change it in the predictions. Set
`--grader-id-column` or `--grader-output-key-column` when the grader files
use different headers from the prediction table.

Select the intended run with `--prediction-filter COLUMN=VALUE` when the
prediction file contains multiple runs. By default, each grader file must
cover the selected IDs exactly. Add `--allow-extra-grades` to use complete
grader exports for a sample or filtered run. Each selected ID and output
key must have one matching grade in each export. Missing matches and
duplicate matches return status 2.

Use `policy` to split a scored batch and fit from its separate grades.

```bash
caat policy scores.csv --score-column confidence --grader-file primary=grades.csv --risk-target 0.40 --holdout-confidence 0.95 --track-fit-outputs --output policy.json --holdout-output holdout.csv
```

For example, if `scores.csv` uses `answer_id` and `grades.csv` uses `id`,
add `--id-column answer_id --grader-id-column id` to the command.

The holdout CSV appends `primary` to the original prediction columns.
Give each file a grader name absent from the prediction headers when
exporting a holdout. `--check-column` can add an inline grader to the holdout
check. For `select`, supply labeled calibration rows and a separate `--design`
table or explicit `--thresholds`. The design table needs the output key when
tracking fit outputs.
[Use archived grades for a sample](queue.md#use-full-grader-exports) · [Fit a probability map](calibration.md#fit-from-a-separate-grader-export) · [CI audit example](ci-audit.md)

## Threshold selection data

Supply `select --thresholds` from an independent source or use
`select --design` to take score deciles from a separate file. The design
file needs the score column. If it also includes the configured ID column,
CAAT checks for reuse across the design and calibration files after any row
filter. When both records have a configured output key, CAAT matches the
ID and output key together. The calibration file needs scores and grader
outcomes. Save the selection with `--output cutoff.json`. Apply
that cutoff unchanged to a separate holdout with `evaluate
--threshold-report cutoff.json --holdout-confidence 0.95`.
[Saved cutoffs and holdout reports](reports.md#reuse-a-saved-cutoff)
