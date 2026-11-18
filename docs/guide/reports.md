# CLI reports

## Common

Use `--json` for stdout or `--output FILE` to save a report. Status 1 can accompany a valid report when the requested target or power is missed. Status 2 means invalid input.

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

Pass a `select`, `policy` or `run` JSON as `--threshold-report` to reuse the full-precision cutoff and score direction. A saved fitted scorer can read the original feature columns. Set `--review-missing-scores` to send unscored rows to review. Use `--track-fit-outputs` to record which outputs were used for fitting and detect reuse on a holdout.

## Evaluate

Add `--output-dir DIR` for `metrics.json` and `risk_coverage.csv`. Use `--probability` for calibration metrics. With a fixed batch and label plan, `batch_verdict` is `meets`, `exceeds` or `inconclusive`.

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

The first grader is the reference. Each comparison uses the same predictions and cutoff. Add `--disagreements-output FILE` for the affected IDs.

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

Reports the first sequential error alert and, when configured, breaches of the coverage floor. [Monitoring](monitor.md)

| Field | Meaning |
| --- | --- |
| `failure_rate` | Probability limit for any alert under the sequential test's null. |
| `anytime_pvalue` | Reciprocal of the highest test evidence seen, including the starting value of one. |
| `first_alert` | `id`, one-based input `row`, and cumulative `accepted` and `errors` when the threshold first crossed, or null. |
| `coverage_check` | Window settings, first coverage alert, and latest full window when a floor is set. |

## Plan

A forecast describes operating characteristics before any labels are collected. `--label-budget` fixes the cap. Add `--interim-labels` for a two-stage design or `--budget-ceiling` to compare designs that meet the specified power under both error scenarios. The `smallest-cap` and `lowest-cost` offers report their caps, schedules and scenario costs. A cost is the expected labels plus the request price times expected requests. [Draw and dispatch an offer](queue.md#issue-audit-labels-in-blocks)

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

| Field | Meaning |
| --- | --- |
| `accepted_population` | Number of eligible outputs in this batch. |
| `max_errors_to_meet` and `min_errors_to_exceed` | Inclusive sample-error boundaries for the two directional verdicts. |
| `verdict_probabilities` | Chances of meeting the target, exceeding it or remaining inconclusive under the assumed error count. |
| `caps_tested` and `offers` | Budgets tested and the qualifying schedules from a budget comparison. |

## Select

A null selected threshold means no candidate qualified for the requested target. Empirical selection compares observed error. The Learn then Test rules account for testing multiple candidates.

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

## Policy

The report combines the calibration selection and independent held-out checks. A confirmed policy can route new predictions. [How the stages fit together](../research/method.md)

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
