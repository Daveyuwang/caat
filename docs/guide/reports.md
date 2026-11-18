# CLI reports

[Terms used in the summaries](glossary.md)

The four review commands print a plain-text summary by default. `--json` prints the
full report to stdout. `--output FILE` writes that report as JSON whether
or not you use `--json`. A valid report can accompany exit code 1. Exit
code 2 signals invalid input. `evaluate --output-dir DIR` also writes
`metrics.json` and `risk_coverage.csv`. Use `--plot FILE.svg` with
`evaluate` for a risk–coverage curve or with `audit` and `policy` for
per-grader risk at the selected cutoff. Add `--html FILE.html` to any of
these three commands for an offline report with embedded SVGs. For a
standalone reliability diagram, use `evaluate --probability --reliability-plot FILE.svg`.
Use `--render NAME --render-output FILE` for a registered report renderer.
[Plugins](plugins.md)

Each rate is a fraction between 0 and 1. Risk has value `null` when the
policy accepts no rows. In the tables, `*` stands for a key in a map and
`[]` stands for an element in an array.

## Common

The `input.*` entries in this table apply to prediction tables.

| Field | Meaning |
| --- | --- |
| `schema_version` | Report schema version, currently 2. |
| `command` | CLI command used to produce the report. |
| `caat_version` | Installed CAAT version. |
| `input.path` | Prediction file path as supplied to the CLI. |
| `input.id_column` | Row identity column. |
| `input.score_column` | Score column. Larger values enter the policy first. |
| `input.group_column` | Explicit or detected group column, or null. |
| `input.output_key_column` | Column identifying the saved response, or null. |
| `input.correct_columns` | Ordered grade columns used for evaluation or selection. Separate grader files have their own map. |

### Reuse a saved cutoff

Use the JSON from `select`, `policy`, or `run` as `--threshold-report`
in `evaluate`, `audit`, `monitor`, or `route`. The command reads the cutoff at
full precision and uses the report's ID and score columns. `evaluate`,
`audit`, and `monitor` also read the risk target. `evaluate` and `audit`
read a saved holdout confidence. Explicit column and risk options take
precedence. Use separate holdout or monitoring rows with scores computed
in the same way as those used to choose the cutoff.
[Reuse score direction from a workflow](workflow.md#use-a-lower-is-better-score)

Use `--raw-score-column COLUMN` when the original score has a new column
name. CAAT reads that column using the saved direction. Use
`--score-column` for values already on the scale of the saved cutoff.

CAAT uses the saved output key column when it is present in a prediction
table. Use `--output-key-column` to read a different column. A batch
decision file without output keys can still be checked by ID and score.

```bash
caat select calibration.csv --design design.csv --score-column confidence --correct-column primary --risk-target 0.40 --output cutoff.json
caat audit holdout.csv --correct-column primary --correct-column alternate --threshold-report cutoff.json --holdout-confidence 0.95 --output audit.json
```

Use `evaluate` with one grade column. Set `json_output` in the TOML
configuration to save a `run` report, or print it with `--json`.
The new report records its source in `input.threshold_report`. A source
report with no selected cutoff returns input error 2.
`route` uses the cutoff and score column to write a decision per row and
requires confirmation for `policy` and `run` reports.
[Routing a batch](queue.md#route-a-scored-batch)

### Keep unscored outputs for review

Use `--review-missing-scores` with `select` or `policy` to keep empty
CSV scores and JSONL nulls in the population. These rows always go to
review. `policy` splits the full cohort before taking design quantiles
from its scored rows. Coverage uses the full cohort for each split.
Risk tests use the accepted rows and their errors. A design cohort with
no scores yields an empty cutoff grid.

```bash
caat policy predictions.csv --correct-column primary --risk-target 0.40 --holdout-confidence 0.95 --review-missing-scores --output cutoff.json
caat route new-predictions.csv --threshold-report cutoff.json --output decisions.csv
```

The saved cutoff carries the same review rule into `evaluate`, `audit`,
`monitor`, and `route`. With a numeric cutoff, pass the option directly.
Ranking metrics and probability calibration use scored rows. The
holdout report's coverage and grader comparison retain the full cohort.
[Compare scores with missing values](scores.md#compare-scores-with-missing-values)

### Track outputs used to choose the cutoff

Add `--track-fit-outputs` to `select` or `policy` to save SHA-256 digests
of the design and calibration rows' `(ID, output key)` pairs. Set
`--output-key-column` for inline grades. With `--grader-file`, CAAT uses
the column from the grader join, which defaults to `output_sha256`.
A separate `--design` table needs both identity columns. For `run`, set
`output_key_column` in the TOML configuration.

```bash
caat select calibration.csv --design design.csv --score-column confidence --correct-column primary --risk-target 0.40 --output-key-column output_sha256 --track-fit-outputs --output cutoff.json
caat audit holdout.csv --correct-column primary --correct-column alternate --threshold-report cutoff.json --holdout-confidence 0.95 --output audit.json
```

With a tracked report and holdout confidence, `evaluate` and `audit`
inherit the key column. `monitor` applies the same check to a tracked
cutoff before its sequential test. Preserve the ID and key values when
reordering or regrading inputs. A matching fit pair returns input error 2.
A completed overlap check records `input.fit_outputs_checked`. The
statistical tests retain their sampling assumptions.

## Evaluate

Apply a frozen cutoff to an independent holdout with one grader.

```bash
caat evaluate holdout.csv --correct-column correct --threshold 0.8 --risk-target 0.10 --holdout-confidence 0.95 --output holdout.json
```

Use `--grader-file NAME=FILE` to read one separate CSV or JSONL grader
export. CAAT matches each prediction by ID and output key and records the
file under `input.grader_files`. Use `--reference-column` to group results
by a reference label in the prediction table.
[Separate grader exports](csv-format.md#separate-grader-exports)

With a risk target, the command exits 0 if the check passes and 1 if
the target is missed or the cutoff accepts zero rows. The default check
compares the exact observed error ratio with the decimal target.
`--holdout-confidence` uses the one-sided
exact binomial upper bound and assumes independent holdout observations
from the target population. The reported bound is the lowest representable
rate whose exact lower-tail probability is at or below the test level.
A target equal to the bound passes.

| Field | Meaning |
| --- | --- |
| `threshold` | Scores at or above this cutoff are accepted. |
| `input.prediction_column` | Predicted label column, or null. |
| `input.reference_column` | Reference label column, or null. |
| `overall.total` | Number of rows. |
| `overall.accepted` | Number of accepted rows. |
| `overall.errors` | Number of accepted rows graded incorrect. |
| `overall.coverage` | Accepted count divided by total. |
| `overall.error_rate` | Accepted errors divided by accepted count. |
| `risk_target` | Maximum accepted-set error rate, present with `--risk-target`. |
| `holdout_confidence` | One-sided confidence level, present with `--holdout-confidence`. |
| `overall.risk_upper_bound` | One-sided upper error bound, or null for zero accepted rows. Present with `--holdout-confidence`. |
| `overall.meets_target` | Whether the risk check passed, or null when no target or risk is available. Present with either risk option. |
| `by_reference` | The same counts and rates by reference label. |
| `by_group` | The same counts and rates by group. |
| `brier_score` | Brier loss when scores are supplied as probabilities. |
| `ece` | Expected calibration error from fixed-width bins. Present with `--probability`. |
| `ace` | Mean absolute calibration gap over nonempty adaptive bins. Present with `--probability`. |
| `reliability_bins` | Fixed-width probability intervals with row counts, mean probability, and observed correctness. Present with `--probability`. |
| `auroc` | Area under the ROC curve for the correctness labels. |
| `aurc` | Mean risk over prefixes of the score ranking, averaging over ties. |
| `normalized_aurc_gain` | AURC improvement on random order as a fraction of the oracle improvement. |

## Audit

| Field | Meaning |
| --- | --- |
| `threshold` | Frozen acceptance cutoff. |
| `risk_target` | Risk limit for deciding success, or null. |
| `holdout_confidence` | Requested one-sided confidence level, present with `--holdout-confidence`. |
| `primary_grader` | First grader, used for directional comparisons. |
| `graders.*.meets_target` | Whether that grader meets the limit, or null. Each grader also has overall counts and per-group counts. |
| `graders.*.risk_upper_bound` | One-sided exact binomial upper bound on accepted error, present with `--holdout-confidence`. |
| `input.grader_files` | Separate export paths by grader, present when supplied. |
| `input.disagreements_output` | Review CSV path, present with `--disagreements-output`. |
| `input.prediction_filter.column` | Prediction filter column, present when supplied. |
| `input.prediction_filter.value` | Exact value selected by the filter. |
| `accepted_comparisons.*.newly_failed` | Accepted rows that pass the primary grader and fail this grader. |
| `accepted_comparisons.*.newly_passed` | Accepted rows that fail the primary grader and pass this grader. |
| `accepted_comparisons.*.risk_change` | This grader's accepted-set risk minus the primary risk. |
| `accepted_comparisons.*.overall_risk_change` | The same difference using all rows. |
| `accepted_comparisons.*.excess_risk_change` | Accepted-set change minus overall change. |
| `accepted_comparisons.*.target_decision_changed` | Whether the target decision differs from the primary grader's. |
| `accepted_comparisons.*.bootstrap_95` | Paired 95% intervals for the three risk shifts and the count of replicates with acceptance, present when requested. |
| `bootstrap.replicates` | Requested number of bootstrap replicates, present when requested. |
| `bootstrap.seed` | Bootstrap seed. |
| `discordant[].id` | ID of a row on which graders differ. |
| `discordant[].group` | Group for that row, or null. |
| `discordant[].accepted` | Whether the policy accepts the row. |
| `discordant[].outcomes` | Boolean outcomes by grader. |

## Monitor

`monitor` applies a frozen cutoff to an ordered stream. A grade source
adds `risk_target`, `grader`, and the sequential test fields. An alert
returns exit code 1 and retains the JSON report. A growing input stream
keeps its first alert.
[Monitoring procedure](monitor.md)

| Field | Meaning |
| --- | --- |
| `failure_rate` | Probability limit for any alert under the sequential test's null. |
| `anytime_pvalue` | Reciprocal of the highest test evidence seen, including the starting value of one. |
| `first_alert` | `id`, one-based input `row`, and cumulative `accepted` and `errors` when the threshold first crossed, or null. |
| `coverage_check` | Window settings, first coverage alert, and latest full window when a floor is set. |

With `--coverage-window` and `--min-coverage`, `coverage_check` records
`window_size`, `min_coverage`, and `windows_checked` up to the first
coverage alert or the end of the stream. Its `first_alert` records the
one-based `start_row` and `end_row`, the last row's `id`, `accepted`,
`coverage`, and `missing_scores` in the first window below the floor.
It is null while no window has fallen below the floor.
`latest_window` reports the same fields for the final full window plus
`meets_minimum`. It is null before the first full window. The exit status
still reflects any alert recorded over the stream.
[Coverage monitoring](monitor.md#alert-when-coverage-falls)

With both coverage options and no grade source, `overall` records `total`,
`accepted`, `coverage`, and `missing_scores` for the full prediction table
or routing export. Add a grade source for the error counts, anytime p-value,
and sequential risk alert. [Run coverage alone](monitor.md#check-coverage-before-collecting-labels)

With `--grader-file`, `overall` summarizes the checked prefix, including
its rejected rows. Its coverage is null for an empty prefix. `stream`
contains full-stream `total`, `accepted`, and `coverage`, plus the count
`unlabeled_accepted` and the `id` and one-based `row` of the
`first_unlabeled` accepted output. The latter is null when all accepted
rows have labels.
[Separate label files](monitor.md#read-arriving-labels-from-a-separate-file)

With `--batch-decisions`, `input.batch_decisions` records the decision file.
`stream.missing_scores` counts unscored review rows in the full stream.
With inline grades and `--review-missing-scores`, `overall.missing_scores`
records this count.
[Monitor a routed batch](monitor.md#monitor-a-routed-batch)

## Plan

`caat plan` sizes an independent check of one frozen acceptance policy.

To compare budgets before requesting labels, give `--forecast` a saved policy,
unlabeled predictions, a label budget and an explicit assumed error rate.

```bash
caat plan --forecast --threshold-report results/anli_r2/policy.json --batch-predictions results/winogrande/predictions.csv --label-budget 100 --expected-error 0.15
```

Here 667 rows qualify. A 15% scenario rounds up to 101 population errors.
Drawing 100 labels gives a 31.65% chance of supporting the 20% target.
With 300 labels that chance rises to 90.61%. Run another scenario with
`--expected-error 0.25` to see the chance of detecting excess error.
The JSON field `verdict_probabilities` gives the probabilities of `meets`,
`exceeds` and `inconclusive` for a uniform draw without replacement.
`expected_population_errors` records the rounded count. A budget larger
than the eligible population uses the whole population. An empty population
gives an inconclusive verdict and exit status 1. To issue a sample, replace
`--forecast --expected-error RATE` with `--sample-output sample.csv`.

Add `--block-size 10` to compare sending ten rows at a time with sending
the full sample in one request. For one grader, `block_cost.expected_labels`
and `expected_requests` average over random draws, including the final short
block. A block is issued only while at least two full-budget outcomes remain
possible. `bulk_labels` and `bulk_requests` count the work of sending the
whole sample at once. At an assumed 215 errors among 667 eligible rows
(`--expected-error 0.3223`), blocks of ten average 86.84 labels and 8.68
requests for a budget of 100. Changing the block size preserves the three
verdict probabilities. Multiple graders can stop at different times.

With `--batch-predictions` and `--label-budget`, the plan keeps the saved
risk target and samples from raw predictions. `total` counts selected input
rows, `accepted_population` counts eligible rows, `review` counts the rest
and `accepted` is the number sampled. The input metadata records the batch
path, requested label cap, cutoff, confidence, seed, and the digest of the
eligible IDs, oriented scores and output keys. This sampling mode produces
no certification probability. [Draw a fixed-budget sample](queue.md#draw-an-audit-sample-after-a-policy-misses-its-target)

For a nonempty draw, `max_errors_to_meet` and `min_errors_to_exceed` give
the inclusive thresholds for the observed sample error count. A null limit
means that direction cannot be established at the chosen budget. The plan
records the source report's canonical JSON SHA-256. A fixed-budget
`evaluate` or `audit` replays the draw on the supplied raw batch and
compares every sampled ID, oriented score and output key.

Each assessment reports `batch_verdict` and the exact lower and upper
population error counts (`risk_lower_errors`, `risk_upper_errors`) and
rates (`risk_lower_bound`, `risk_upper_bound`). Each limit is one-sided
at the declared confidence, which must exceed 50% in this mode. For a
population of N eligible outputs, the target permits floor(N × target)
errors. An upper limit at or below that integer supports the target. A lower
limit above it establishes excess. All other outcomes are inconclusive.
The existing `meets_target` field records whether the upper limit supports
the target.
Each required label comes from an independently sampled accepted row.
The command computes certification power from the exact binomial count
of errors at the assumed rate. Check the collected labels with the same
`--holdout-confidence` in `caat evaluate` for one grader or `caat audit`
for paired graders.
[Sample a routed batch for labeling](queue.md#sample-accepted-rows-for-labeling)

```bash
caat plan --risk-target 0.10 --expected-error 0.05 --holdout-confidence 0.95 --power 0.80
caat plan --accepted 179 --expected-error 0.05 --holdout-confidence 0.95 --power 0.80 --json
```

The first command finds the minimum number of accepted labels that meets
the requested power. The second finds the smallest risk target that the
specified budget can certify with that power. A search with no qualifying
label count up to `--max-accepted` returns exit code 1. Invalid inputs
return 2.

The minimum-budget search can rule out smaller budgets using a power
bound for a larger sample. A uniformly drawn subset of the larger sample
reproduces a smaller sample. CAAT checks the remaining budgets in order
because the power of the test can fall between budgets. The search
includes equality at the test level and at the requested power.

Use `--threshold-report FILE` to find the minimum labels for the risk target
in a saved `select`, `policy`, or `run` report. The planner also uses the
report's holdout confidence when present. Otherwise it defaults to 0.95.
Specify `--holdout-confidence` to use a different level. With a `policy` or
`run` report whose held-out check accepted rows, `--expected-error` is
optional and defaults to the held-out error rate of the first grader. The
report then records `input.expected_error_source` as `threshold_report`.

| Field | Meaning |
| --- | --- |
| `input.expected_error` | Assumed true error probability among accepted outputs. |
| `input.holdout_confidence` | Confidence level for the one-sided exact upper bound. |
| `input.threshold_report` and `input.threshold` | Source report path and full-precision cutoff when planning from a saved threshold. |
| `input.power` | Required chance of certification at the assumed error probability. |
| `risk_target` | Requested or computed accepted error limit. |
| `accepted` | Required or supplied number of accepted labels, or null if none qualifies. |
| `max_errors` | Greatest observed error count that certifies the target at that label count, or null. |
| `certification_probability` | Probability of observing at most `max_errors`, or null. |

### Fixed accepted batch

Use `--batch-decisions FILE` to count the accepted predictions in a complete
`caat route` export. Use `--accepted-population N` to enter the count directly.
With `--threshold-report`, the planner checks scored decisions against
the saved cutoff before sizing the sample. A review row can have a blank
score.
When planning from a decision file, CAAT saves a digest of the accepted
IDs, scores, and output keys as `input.accepted_population_sha256`. The
digest uses the parsed values and is independent of row order.
The plan multiplies the decimal `expected_error` by `N` and rounds up
to an integer. It reports this count as `expected_population_errors`
and uses the exact hypergeometric distribution for a uniform sample
without replacement. The label count is at most `N`. With `--accepted`,
CAAT reports the smallest representable risk target at or above the
certified error-count ratio.
The fixed-budget planner and audits use integer bounds to compare the tail
probability with the test level. If the bounds touch or straddle that level,
CAAT evaluates the exact tail. For a reported probability, the two bounds
must round to the same number to skip the exact calculation.
[Hypergeometric distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html)

Add `--sample-output FILE.csv` to draw and save the planned number of
accepted rows. The output keeps IDs, scores, groups, and available output
keys. The plan records the seed in `input.sample_seed` and the CSV path
in `sample_path`. Use `--sample-seed` to set the seed or to record it for a
separate draw from `--batch-decisions`.
[Sampling walkthrough](queue.md#check-the-error-rate-of-a-fixed-batch)

Pass the same batch decisions or accepted count and `--holdout-confidence`
to `evaluate` or `audit` on the labeled sample at the frozen cutoff.
With a decision file, both commands align the sample by ID, score, and any
output key. The report records the file in `input.batch_decisions`.
For a draw made with `caat queue --random-output` on accepted rows without
quotas, pass the same seed to `--sample-seed`. CAAT uses the number of
labeled rows to replay the draw and checks that the sample contains those
IDs in any order. A mismatch returns status 2. A successful match records
the seed in `input.sample_seed`.

Pass `--plan-report FILE` to use a saved `caat plan` report with either
command. The plan supplies the risk target and holdout confidence. It also
supplies the accepted population for a fixed batch when you omit both
population options. With a stored sample seed, supply `--batch-decisions`
to replay the same draw automatically. If you supply a saved setting
explicitly, it must match the plan. A stored cutoff must match the cutoff
you apply. CAAT checks the number of labeled accepted rows and records it as
`input.planned_labels` with the source path in `input.plan_report`.

For a sample written by `plan --sample-output`, the saved plan lets
both commands use the sample's `id`, oriented `score`, and optional
`output_key` without column flags. A labeled copy can keep those fields.
If the copy includes all the cutoff report's original input columns, CAAT
uses those columns by default. When the original lower-is-better score
and the routed score both use `score`, CAAT checks the copied sample's
values against the batch decisions and requires a single scale for all
rows. Explicit column options select the input fields when needed.

Use `--batch-decisions` to check the routed batch and sample alignment.
With a saved population digest, CAAT also checks all accepted records
against the plan and sets `input.accepted_population_checked` on a match.

The report includes `accepted_population` and each checked grader's
`risk_upper_errors`, the largest number of errors in the accepted batch
consistent with the one-sided test. CAAT compares that count divided by
`N` with the decimal risk target and rounds `risk_upper_bound` up when
the ratio requires rounding. Coverage and observed errors describe the
labeled input. A census gives the exact batch error count. Paired
`--bootstrap` intervals require an independent holdout.
[Fixed-batch walkthrough](queue.md#check-the-error-rate-of-a-fixed-batch)

## Select

`select` and `policy` accept `--direction lower` for lower-is-better scores.
Pass explicit `select --thresholds` in the original score units. The
report stores the signed cutoff and the original column as
`input.negated_score_source`. Commands reading `--threshold-report` use
the saved direction with the original score column.
[Lower-is-better input and routing](csv-format.md#use-lower-is-better-scores)

| Field | Meaning |
| --- | --- |
| `input.design_path` | Separate file used to choose candidate thresholds, or null. |
| `rule` | Threshold selection rule. |
| `graders` | Graders used in selection, in input order. |
| `risk_target` | Maximum accepted-set risk. |
| `failure_rate` | Error budget for a statistical selection rule. |
| `per_candidate_level` | Common significance level, when the rule uses one. |
| `guarantee` | Confidence and sampling assumptions for a certified selection, or null. |
| `calibration_total` | Number of calibration rows. |
| `selected_threshold` | Most permissive qualifying threshold, or null. |
| `candidates[].threshold` | Candidate cutoff, ordered from strictest to most permissive. |
| `candidates[].accepted` | Number accepted at the candidate cutoff. |
| `candidates[].coverage` | Calibration share accepted at this cutoff. |
| `candidates[].errors` | Accepted error counts by grader. |
| `candidates[].worst_error_rate` | Largest accepted error rate among the graders, or null. |
| `candidates[].pvalue` | Largest one-sided binomial p-value among the graders. |
| `candidates[].tested` | Whether the rule tested this threshold. |
| `candidates[].passes` | Rule decision for this threshold, or null when untested. |

The three LTT rules compare exact binomial tails with their significance
levels. The JSON p-values and levels are rounded. If Holm stops among
equal reported p-values, it marks the first remaining cutoff in threshold
order as failed.

The empirical rule compares each grader's exact observed error ratio
with the decimal risk target. The report displays the largest ratio as
`worst_error_rate`.

Repeat `--correct-column` to require each grader's accepted error rate to
meet the target. If your task treats an output as wrong whenever any grader
fails it, create a column with `1` only where every grader passes and use
that column for selection and validation.
[Paired calibration comparison](../research/benchmarks.md#fitting-with-several-graders)

## Policy

| Field | Meaning |
| --- | --- |
| `input.check_columns` | Held-out-only grade columns. |
| `input.holdout_output` | Exported holdout CSV path, present with `--holdout-output`. |
| `split.seed` | Reproducible split seed. |
| `split.design` | Number of design rows. |
| `split.calibration` | Number of calibration rows. |
| `split.heldout` | Number of held-out rows. |
| `check_graders` | Held-out-only grader names. |
| `ranking.*.auroc` | Calibration AUROC by grader. The same entry includes AURC and normalized AURC gain. |
| `selection` | Core selection report. The CLI envelope appears at the top level. |
| `holdout_confidence` | Requested one-sided confidence level, present with `--holdout-confidence`. |
| `heldout.graders.*.meets_target` | Whether a grader meets the target on the holdout. Each grader also has held-out counts and rates. |
| `heldout.graders.*.risk_upper_bound` | One-sided exact binomial upper bound on accepted error, present with `--holdout-confidence`. |
| `heldout.accepted_comparisons` | Directional shifts when two or more graders are listed. |
| `all_rows.accepted` | Rows at or above the selected cutoff in the whole input, with `total` and `coverage`. Present when a cutoff is selected. |
| `all_rows.fitting` | Design and calibration rows. They chose the cutoff, so the held-out rows give the error estimate. |
| `split_sensitivity.splits` | Number of other seeds refitted, set with `--split-checks`. Absent with `--split-checks 0`. |
| `split_sensitivity.selected` | How many of those splits selected a cutoff. |
| `split_sensitivity.accepted_min` | Fewest rows of the whole input that one of those cutoffs accepts. The entry also has `accepted_median` and `accepted_max`. |
| `confirmed` | True if a cutoff is selected and all graders meet the selected holdout check. |

`heldout` is null when selection finds no cutoff. Without
`--holdout-confidence`, each held-out grader also has `upper_bound_95`, a
one-sided 95% exact upper bound shown next to the observed rate. It does
not change `meets_target`.

Set `--design-grid-size N` on `caat select --design` or `caat policy`
to choose the number of evenly spaced design quantiles. The default is
10. Ties contribute one candidate per distinct cutoff. Reports record
the supplied value as `input.design_grid_size` and list the actual
cutoffs under `candidates` or `selection.candidates`.
[Design grid replay](../research/benchmarks.md#design-quantile-count)

Use `--holdout-output holdout.csv` with the built-in CSV or JSONL reader
to save the holdout rows in source order with all input columns. The
export is available for both policy exit statuses 0 and 1.
[Reuse the holdout from an Inspect run](inspect.md#check-a-rescored-log-at-a-frozen-cutoff)

By default, `audit` and `policy` compare the observed error rate to the target.
Use `--holdout-confidence 0.95` to compare each grader's one-sided
Clopper–Pearson upper bound to the target. This check assumes independent,
identically distributed labeled rows from the target population and a
cutoff fixed before looking at those rows. The reported bounds are
per-grader. `policy` still selects its threshold using the calibration
rule and its own `--failure-rate`. A bound is null when no rows are accepted.

An explicit `--adapter NAME` adds `input.adapter`. Metrics selected with
`evaluate --metric NAME` appear under `extra_metrics`, keyed by name.

When `--calibrate METHOD` is used, `probability_calibration.graders` contains
one fitted `map` per selection grader. Each entry reports
`heldout_calibrated` and, for probability inputs, `heldout_uncalibrated`.
These field values use the same Brier, ECE, ACE, and reliability-bin
definitions as `evaluate`. The policy cutoff stays on the raw score scale.
[Calibration workflow](calibration.md)
