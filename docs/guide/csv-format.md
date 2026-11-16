# CSV input

Start from saved outputs with one row per unique item. Use UTF-8 CSV
with a header row. `--id-column` and `--score-column` default to `id`
and `score`. Larger finite scores rank ahead of smaller ones. The
threshold keeps every row whose score is at least that value.

## Grades in the prediction CSV

Provide a 0/1 outcome for each grader. Repeat `--correct-column` in the
order you want CAAT to use. `evaluate` uses one grader. Its exact-match
mode uses `prediction` and `reference` labels. Supply `--reference-column`
with an explicit outcome column to group results by reference label.
`policy --check-column` adds graders for the held-out comparison.

`--group-column` identifies groups for reporting. CAAT automatically
uses a column named `group` when present. `--output-key-column` declares
the saved response identity when the CSV includes one.

## Separate grader exports

Each grader file needs the same ID and output identity columns as the
prediction file plus a `correct` column with 0/1 grades. The files can put
rows in different orders. Run `audit --grader-file NAME=CSV` for each
grader. The default identity column is `output_sha256`. Specify another
with `--output-key-column`. If the prediction export contains multiple
runs, select the intended run with `--prediction-filter COLUMN=VALUE`.
Each grader file must cover the selected IDs exactly. [CI audit example](ci-audit.md)

## Threshold selection data

Supply `select --thresholds` from an independent source or use
`select --design` to take score deciles from a separate file. The design
file needs the score column. The calibration file needs the scores and
grader outcomes. Apply the selected threshold to held-out rows.
[Report fields](reports.md)
