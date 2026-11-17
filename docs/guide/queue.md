# Queue predictions for review

```bash
caat queue predictions.csv --score-column confidence --budget 50 --output replay/queue.csv --random-output replay/random_queue.csv
```

The queue accepts CSV or JSONL with unique IDs and finite scores. It orders rows
from lowest confidence to highest, breaking score ties by ID. With
`--direction lower`, high scores enter the queue first, which is useful
with uncertainty scores such as semantic entropy.

Add `--review-missing-scores` to keep empty score cells or JSONL `null`
scores. The queue orders those rows by ID ahead of finite scores in
either direction. Use `--prediction-filter decision=review` when queuing
from a routing CSV.

The output is an ordered CSV with `id`, `score`, and `group` columns.
Use `--id-column`, `--score-column`, and `--group-column` to specify
different input headers. Repeat `--quota GROUP=COUNT` to place an upper
limit on selected rows from each named group. Both the score-ordered queue
and the random queue fill the same budget under the same limits.

Repeat `--correct-column NAME` to replay the queue against saved outcomes.
The report counts errors found under each grader for both ordering
methods. Without quotas, it also gives the random queue's expected number
of errors. `--seed` controls the random ordering and defaults to 42.
`--json` prints the full report.
[Course benchmark](../research/benchmarks.md#review-queues)

## Route a scored batch

Use `caat route` to apply a frozen cutoff to a CSV or JSONL table of scored
predictions. The input needs unique IDs and finite scores. Each row at or
above the cutoff receives `accept`. Lower scores receive `review`.
Use `--prediction-filter COLUMN=VALUE` to choose one run or tier before
checking IDs. [Multi-tier export example](ci-audit.md#route-and-evaluate-the-same-tier)

```bash
caat route examples/audit/predictions.csv --threshold 0.75 --output-key-column output_key --output replay/decisions.csv --json
```

The example accepts three tickets and marks two for review. The CSV keeps
input order and has `id`, `score`, and `decision` columns, plus `group` when
the input has a group column. The `id` column holds the values of your ID
column under the name `id`, which the later commands read. Add
`--keep-column NAME` for each input column to copy into the file, such as
your labels.
CAAT writes decisions as it reads the batch and replaces the destination
after all rows pass validation.
`--output-key-column` adds an `output_key` column with the supplied identity
values. With `--json`, stdout contains the accepted count, review count,
coverage, cutoff, and path to the decision file.

Pass `--threshold-report cutoff.json` to reuse a selection or confirmed
policy at full precision. When the scored batch has the output key column
saved in the report, routing includes its values in the decision file.
`--output-key-column` selects a different header.
For `policy` and `run` reports, routing requires
the report's holdout confirmation. A valid batch returns 0. An invalid
input, including an unconfirmed policy report, returns 2.
[Saved cutoffs](reports.md#reuse-a-saved-cutoff)

Add `--review-missing-scores` if the batch includes empty score cells or
JSONL scores set to `null`. CAAT keeps those rows in order with a blank
`score` and a `review` decision. They count toward the batch total for
coverage. The JSON summary includes `overall.missing_scores`. Keep the
score field in every JSONL object. Nonempty scores must be finite numbers.
[Monitor a routed batch](monitor.md#monitor-a-routed-batch)

## Sample accepted rows for labeling

Run `caat plan` to size a sample from the accepted rows of a frozen policy.
At a 40% risk target and 20% assumed error rate, the default 95%
confidence and 80% power call for 36 labels. Route a new batch with the
same cutoff, then export a uniform random sample without quotas. Choose
the budget and seed before collecting grades.

```bash
caat plan --threshold-report cutoff.json --expected-error 0.20 --json
caat route new-predictions.csv --threshold-report cutoff.json --output-key-column output_sha256 --output replay/new-decisions.csv
caat queue replay/new-decisions.csv --prediction-filter decision=accept --budget 36 --seed 42 --output-key-column output_key --random-output replay/label-sample.csv --json
```

Use the plan's `accepted` count for `--budget`. The sample keeps `id`,
`score`, `group`, and `output_key`. `--random-output` can write the sample
on its own. Use a new batch drawn from the target population for an
independent check. After labeling, supply both graders' files with `id`,
`output_key`, and `correct`, and check the frozen cutoff.

```bash
caat audit replay/label-sample.csv --threshold-report cutoff.json --id-column id --score-column score --output-key-column output_key --grader-file primary=primary-labels.csv --grader-file alternate=alternate-labels.csv --holdout-confidence 0.95 --output replay/label-audit.json
```

[Label planning](reports.md#plan) · [Separate grader exports](csv-format.md#separate-grader-exports)

## Check the error rate of a fixed batch

Route the batch with its frozen cutoff and keep the complete decision file.
Pass that file and the cutoff report to `caat plan` to check the decisions
and size the label sample. `--sample-output` draws a uniform sample without
replacement from the accepted rows. The plan records the seed, which
defaults to 42. Draw the sample before collecting grades. In the Inspect
replay, `N` was 101. At a 40%
target, a 20% assumed error rate, 95% confidence, and 80% requested power,
the fixed-batch plan called for 28 labels and at most 7 observed errors.

```bash
caat route new-predictions.csv --threshold-report cutoff.json --output-key-column output_sha256 --output replay/new-decisions.csv
caat plan --threshold-report cutoff.json --batch-decisions replay/new-decisions.csv --expected-error 0.20 --sample-output replay/label-sample.csv --output replay/label-plan.json
caat audit replay/label-sample.csv --threshold-report cutoff.json --grader-file primary=primary-labels.csv --grader-file alternate=alternate-labels.csv --batch-decisions replay/new-decisions.csv --plan-report replay/label-plan.json --output replay/batch-audit.json
```

This check estimates risk among the accepted outputs of the fixed batch.
Review rows with a blank score are outside the accepted population.
The audit checks the sample's IDs, scores, and output keys against the
batch decisions. It checks that the frozen cutoff reproduces the recorded
decisions. With the saved plan, CAAT reads `id`, `score`, and `output_key`
from the generated sample. Its `score` is already on the cutoff's scale.
Use `evaluate --correct-column` if the labels are in the sample table.
Pass the random queue's seed as `--sample-seed` to reconstruct
the draw if you used `caat queue` and the plan has no recorded seed. A plan
saved with `--sample-output` supplies the seed automatically. `--plan-report` also
supplies the risk target and confidence and checks the label count and
the accepted IDs, scores, and output keys against the saved plan.
Use the same cutoff, batch, and sample when comparing graders.
[Fixed-batch report fields](reports.md#fixed-accepted-batch) · [Inspect replay](inspect.md)

### Include the original responses

Use `--sample-source` with the original response table and repeat
`--sample-column` for each field the grader needs. Route with output keys
so CAAT can match every sampled ID to its exact output key. An Inspect
scoring export has `input` and `completion` fields.
[Export an Inspect log](inspect.md)

```bash
caat plan --threshold-report cutoff.json --batch-decisions replay/new-decisions.csv --expected-error 0.20 --sample-output replay/label-sample.csv --sample-source scoring-input.csv --sample-column input --sample-column completion --output replay/label-plan.json
```

The source can be CSV or JSONL with `id` and `output_sha256` by default.
For any default header absent from the table, CAAT uses the ID or output
key column from `--threshold-report` when available. Use `--id-column`
and `--output-key-column` to set the source headers explicitly. CAAT appends
each requested field in argument order. A missing
or duplicate match returns status 2. The plan and sample are published
together after the join succeeds. Use the same labeled-sample audit with
the resulting CSV.

### Use full grader exports

The planned sample uses `output_key`, and Inspect grade exports use
`output_sha256`. Use the original grade files with the saved sample and
plan by adding `--grader-output-key-column output_sha256` and
`--allow-extra-grades` to the audit. CAAT checks that every sampled output
has exactly one grade in each file.
[Separate grader exports](csv-format.md#separate-grader-exports)
