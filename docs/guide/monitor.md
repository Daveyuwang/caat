# Monitor a frozen policy as labels arrive

Use `--threshold-report` with a saved `select`, `policy`, or `run` report
to read the frozen cutoff, column names, and risk target. For a numeric
cutoff, provide `--threshold`. To run the sequential test with a numeric
cutoff, also set `--risk-target`. The test uses `--failure-rate`, which
defaults to `0.05`.
[Saving a cutoff](reports.md#reuse-a-saved-cutoff)

For a tracked cutoff, keep each monitoring row's original ID and output
key. `monitor` inherits the key column and checks for reuse of design or
calibration rows before computing evidence. Use `--output-key-column`
if you renamed the column. A matching fit pair returns input error 2.
[Track fit outputs](reports.md#track-outputs-used-to-choose-the-cutoff)

Use `--correct-column` for inline labels or a
[separate label file](#read-arriving-labels-from-a-separate-file). Keep
predictions in the order set before the outcomes are known. Append new
rows for each check. The monitor updates after each accepted row and
records the first alert when the evidence reaches the alert level. Rejected
rows contribute to coverage counts.

Use CSV or JSONL with a unique `id` and a finite `score`. Inline grades
use a binary correctness column. `--output FILE` saves the report. Exit code 0 means
no alert, 1 means an alert, and 2 means invalid input.
[Report fields](reports.md#monitor)

## Alert when coverage falls

Set `--coverage-window` and `--min-coverage` together to check a minimum
acceptance rate in each full window of consecutive rows. For example,
a floor of 0.80 over 50 rows requires at least 40 accepts in each window.
Windows advance by one row in prediction order. Unscored review rows
contribute to the window size.

```bash
caat monitor results/monitoring/heldout.csv --threshold-report results/monitoring/selection.json --correct-column correct --coverage-window 50 --min-coverage 0.80 --json
```

The coverage check uses the full stream, including rows awaiting labels.
It works with `--batch-decisions` as well. An incomplete first window
shows zero windows checked. The first window below the floor returns
exit code 1 and is recorded in `coverage_check.first_alert`, even after
later windows recover. Read `coverage_check.latest_window.meets_minimum`
to check the most recent full window. That window also reports its row
range, coverage, and missing-score count. The sequential test reports its
risk alert in `first_alert` and uses its own `--failure-rate`.
[Report fields](reports.md#monitor)

### Check coverage before collecting labels

Supply the cutoff, window, and floor to monitor scores alone. Add
`--review-missing-scores` if the score table includes blank values.

```bash
caat monitor predictions.csv --threshold 0.8 --coverage-window 50 --min-coverage 0.80 --output monitor.json
```

Use `--batch-decisions decisions.csv` in place of the prediction table
to check recorded routing. The coverage-only report gives the full batch
counts and its first coverage alert. Supply `--correct-column` or
`--grader-file` to add the sequential error test and its risk fields.

## Sequential test

For an accepted row, let `r` be the risk target and let `q` be an
alternative error probability above it. The likelihood ratio multiplies
by `q/r` after an error and by `(1-q)/(1-r)` after a correct answer.
CAAT mixes three alternatives using `q = r + f(1-r)` for fractions
`0.1`, `0.3`, and `0.6`. It starts new likelihood ratios at each accepted
row `t` with weight `1/[t(t+1)]`. The remaining weight stays at one
until its start. The mixture keeps evidence from early and late parts
of the stream.

The null requires that the conditional probability of error on each
accepted row, given the previous observations, be at most `r`. Under
that null the mixture is a nonnegative supermartingale. Ville's
inequality bounds the probability of any alert by `--failure-rate`,
even when repeatedly checking the same growing stream. The JSON
`anytime_pvalue` is the reciprocal of the largest mixture value seen,
including its initial value of one. CAAT bounds each update with
integers and evaluates exact fractions when the bounds cannot decide
an alert or the rounded p-value. Use a label collection order and
sampling rule fixed before observing the labels. When all accepted
rows are correct by definition of a zero-risk null, any error triggers
an alert. A risk target of one cannot trigger this test.
[Waudby-Smith and Ramdas, 2024](https://doi.org/10.1093/jrsssb/qkad009)

## Chronological Omni-MATH replay

The replay uses the 1,000 saved Qwen2.5-7B-Instruct-Turbo completions from
HELM Capabilities v1.0.0 and the GPT equivalence judgments. Sort by saved
request timestamp and then instance ID. Cut the timeline near 20% and
60% of records, assigning requests with the same boundary timestamp to
the later partition. Derive mean-token-logprob deciles from the first
partition. Select a cutoff on the second using LTT–Bonferroni with an
80% risk target and a 5% failure rate. Freeze it and monitor the remaining
rows with a 5% alert level. Keep the original labels for that replay.
For a separate alert check, set the accepted outcomes to errors in the
last half of the monitoring rows. [Run provenance](../research/judge-study.md#archived-inputs)

The time split had 200 design, 400 calibration, and 400 monitoring rows
from the same archived run. The frozen cutoff was `-0.22340339817772675`.
It accepted 398 of 400 later responses with 278 errors (69.85%). The
stream generated no alert. In the altered replay, changing 58 labels
raised the final accepted error rate to 84.42% and triggered the first
alert at row 290. The injected errors start at row 201. The altered
stream tests the alert path using the same scores and acceptance decisions.
[Replay summary](../../results/monitoring/summary.json) ·
[Selection report](../../results/monitoring/selection.json)

```bash
caat monitor results/monitoring/heldout.csv --threshold-report results/monitoring/selection.json --correct-column correct --json
```

To recompute the cutoff, write a fresh report and pass that path to
`--threshold-report`.

```bash
caat select results/monitoring/calibration.csv --design results/monitoring/design.csv --score-column mean_token_logprob --correct-column correct --risk-target 0.8 --output replay/monitor/selection.json
```

[Design scores](../../results/monitoring/design.csv) ·
[Calibration grades](../../results/monitoring/calibration.csv) ·
[Chronological holdout](../../results/monitoring/heldout.csv)

## Read arriving labels from a separate file

Keep the chronological table of scored predictions, including rejected
rows. Supply a separate CSV or JSONL with the same IDs and output keys and
a `correct` field for each label. `monitor` uses the saved key column when
it is present in the scored table and defaults to `output_sha256` for
separate labels. Override it with `--output-key-column`.
Before labels arrive, use a header-only CSV or empty JSONL.

```bash
caat monitor predictions.csv --threshold-report cutoff.json --grader-file human=labels.csv --output monitor.json
```

Add labels in any order. The test processes predictions up to the first
accepted row still awaiting its label, including the rejected rows before
it. Later labels enter the test after that gap is filled. `overall`
describes the checked rows, and `stream.coverage` measures acceptance on
the full prediction table.
[Report fields](reports.md#monitor)

### Monitor a cohort with a full grader export

Use `--prediction-filter` to select rows from a shared prediction table.
Monitoring follows their order in that table. With `--allow-extra-grades`,
each label must match both the selected ID and its output key. If the export
has only an older version of an accepted output, monitoring waits at that
row for its exact grade. Duplicate grades for the same selected output
raise an input error. Set `--grader-id-column` or
`--grader-output-key-column` when the grader file uses a different header.

```bash
caat monitor predictions.csv --prediction-filter partition=monitoring --threshold-report cutoff.json --grader-file human=all-labels.jsonl --grader-output-key-column graded_key --allow-extra-grades --output monitor.json
```

## Monitor a routed batch

Use the keyed decision file from `caat route` to monitor recorded decisions
in batch order. The monitor checks the scored decisions against the cutoff.
Review rows with blank scores contribute to coverage and require no label
for the sequential test. `stream.missing_scores` counts those rows.

```bash
caat route predictions.csv --threshold-report cutoff.json --output-key-column output_sha256 --review-missing-scores --output decisions.csv
caat monitor --batch-decisions decisions.csv --threshold-report cutoff.json --grader-file human=labels.csv --output monitor.json
```

The decision file uses `id`, `score`, and `decision` as written by `route`.
Retain its `output_key` column for grader joins and cutoffs that track
fit outputs. The grader file uses `id` and `output_key` by default. A
saved cutoff supplies the original identity headers when recorded. Set
`--grader-id-column` or `--grader-output-key-column` to override them.
[Route a scored batch](queue.md#route-a-scored-batch)
