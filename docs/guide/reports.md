# CLI reports

All commands print a plain-text summary by default. `--json` prints the
full report to stdout. `--output FILE` writes that report as JSON whether
or not you use `--json`. A valid report can accompany exit code 1. Exit
code 2 signals invalid input. `evaluate --output-dir DIR` also writes
`metrics.json` and `risk_coverage.csv`.

Each rate is a fraction between 0 and 1. Risk has value `null` when the
policy accepts no rows. In the tables, `*` stands for a key in a map and
`[]` stands for an element in an array.

## Common

| Field | Meaning |
| --- | --- |
| `schema_version` | Report schema version, currently 2. |
| `command` | CLI command used to produce the report. |
| `caat_version` | Installed CAAT version. |
| `input.path` | Prediction CSV path as supplied to the CLI. |
| `input.id_column` | Row identity column. |
| `input.score_column` | Score column. Larger values enter the policy first. |
| `input.group_column` | Explicit or detected group column, or null. |
| `input.output_key_column` | Column identifying the saved response, or null. |
| `input.correct_columns` | Ordered grade columns used for evaluation or selection. Separate grader files have their own map. |

## Evaluate

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
| `by_reference` | The same counts and rates by reference label. |
| `by_group` | The same counts and rates by group. |
| `brier_score` | Brier loss when scores are supplied as probabilities. |
| `auroc` | Area under the ROC curve for the correctness labels. |
| `aurc` | Mean risk over prefixes of the score ranking, averaging over ties. |
| `normalized_aurc_gain` | AURC improvement on random order as a fraction of the oracle improvement. |

## Audit

| Field | Meaning |
| --- | --- |
| `threshold` | Frozen acceptance cutoff. |
| `risk_target` | Risk limit for deciding success, or null. |
| `primary_grader` | First grader, used for directional comparisons. |
| `graders.*.meets_target` | Whether that grader meets the limit, or null. Each grader also has overall counts and per-group counts. |
| `input.grader_files` | Separate export paths by grader, present when supplied. |
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

## Select

| Field | Meaning |
| --- | --- |
| `input.design_path` | CSV used to choose candidate thresholds, or null. |
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

| Field | Meaning |
| --- | --- |
| `input.check_columns` | Held-out-only grade columns. |
| `split.seed` | Reproducible split seed. |
| `split.design` | Number of design rows. |
| `split.calibration` | Number of calibration rows. |
| `split.heldout` | Number of held-out rows. |
| `check_graders` | Held-out-only grader names. |
| `ranking.*.auroc` | Calibration AUROC by grader. The same entry includes AURC and normalized AURC gain. |
| `selection` | Core selection report. The CLI envelope appears at the top level. |
| `heldout.graders.*.meets_target` | Whether a grader's held-out risk meets the target. Each grader also has held-out counts and rates. |
| `heldout.accepted_comparisons` | Directional shifts when two or more graders are listed. |
| `confirmed` | True if a cutoff is selected and all graders meet the empirical target on held-out rows. |

`heldout` is null when selection finds no cutoff.
