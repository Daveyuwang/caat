# Changelog

## Unreleased

- Draw a reproducible, fixed-budget audit sample from new predictions with
  `caat plan --batch-predictions --label-budget`. A withheld policy keeps its
  routing status.
- Check the completed fixed-budget draw against the original batch and report
  whether its error rate meets or exceeds the saved target, or the evidence is
  inconclusive.

## 0.6.0

- Show expected accepts, accepted error, coverage and review volume by predicted class at the same global budget with `caat scores --class-column`.
- Fit a portable scorer using a categorical prediction and numeric features. Replay it on raw rows in a base installation. Numeric-only artifacts retain their existing format.
- Replay the fresh ANLI score comparison and independent cutoff audit. On the primary cohort the categorical feature gained 0.58 accepted-error points at 75% coverage. The design-selected cutoff missed its 20% risk target. [ANLI results](docs/research/anli-results.md)

## 0.5.0

- Route, evaluate, audit and monitor a saved fitted policy using its named features. Explicit score-column overrides still accept precomputed scores.
- Calibrate one frozen score with preset cutoffs or unlabeled reference predictions and independent labeled calibration and holdout sets.
- Replay the BoolQ verifier and label-budget comparisons on CPU. At 75% coverage the verifier combination improved on the best single score. An exploratory answer-aware native score had the lowest accepted error.

## 0.4.0rc1

- Fit standardized logistic confidence scorers on design data and reuse their JSON parameters on unlabeled batches. Fitting uses the optional scorer extra.
- Use predicted/reference labels in TOML workflows, save the fitted scorer alongside the cutoff, and route original feature columns from the saved report.
- Measure decision changes on fixed holdout rows when the design and calibration data are reassigned.

## 0.3.0

- Refit `caat policy` on the next 20 seeds and report how many rows those cutoffs would accept, with `--split-checks` to change or skip it.
- Name the next looser candidate in the `caat policy` summary and say why it was not selected.
- Say when a held-out error rate meets the target while its 95% upper bound does not.
- Match Inspect scores on the sample ID alone with `caat inspect join --scores-id-column`.
- Warn when a score's AUROC is well below 0.5, and give one verdict when the recommended score ties with the baseline.
- Print the queue path from `caat queue`, and the input file and score column in HTML reports.
- State the test each candidate had to pass above the HTML candidate table, and label its error columns.
- State the selection rule's guarantee in the `caat policy` summary, show a 95% upper bound next to the held-out error rate, and name the exit status when it is 1.
- When no cutoff qualifies although a candidate's error is under the target, say that too few rows were accepted to show it.
- Name the split behind each number in `caat scores`, and call two scores tied when every interval for their difference includes zero.
- Default `caat plan --expected-error` to the held-out rate in a `policy` or `run` report.
- Say whether probabilities are overconfident or underconfident in calibration summaries.
- List the saved scorers when `caat inspect join` gets an unknown grader, and point to the export when a score file lacks the identity columns.
- Add a glossary of report terms and two walkthroughs: a classifier's predictions, and model labels for human review.
- Compare the four selection rules over 200 splits of six archives and keep `ltt-bonferroni` as the default.
- Grade by exact match between `--prediction-column` and `--reference-column` in `policy`, `select`, and `scores`, as `evaluate` already does.
- Report the accepted count for the whole input next to the held-out check in `caat policy`.
- When no cutoff qualifies, report the lowest calibration error the candidates reached and its coverage.
- Copy input columns into routed decisions with `caat route --keep-column`, and omit the `group` column when the input has none.
- Build release wheels from normalized source archives with a fixed timestamp and pinned build backend.
- Export saved Inspect completions and align external confidence scores by sample identity and output hash.
- Restore retained Inspect prompt attachments in scoring exports and write JSONL when requested.
- Read indexed Inspect archives in batches and retain samples added during evaluation.
- Read updated grades and runtime samples from Inspect JSON logs, including when a log is replaced at the same path.
- Read only the requested sample records when exporting grades from an indexed Inspect archive.
- Join several named score columns from separate CSV or JSONL files for an Inspect evaluation.
- Extract aligned grader exports from original and rescored Inspect logs with `caat inspect grades`.
- Check a frozen policy on an independent holdout with a one-sided exact binomial bound in `evaluate`, `policy`, and `audit`.
- Track the outputs used to choose a cutoff and reject reuse in later statistical holdout checks and monitoring.
- Read arriving monitor labels from a separate table and report coverage on the complete scored stream.
- Export the exact policy holdout and its input columns with `caat policy --holdout-output`.
- Save the cutoff report and exact holdout from the same TOML workflow with `json_output` and `holdout_output`.
- Reuse a cutoff from a saved `select`, `policy`, or `run` JSON report in `evaluate`, `audit`, and `monitor`.
- Write per-prediction `accept` and `review` decisions at a frozen cutoff with `caat route`.
- Export a seeded label sample from a routed cohort with its output keys.
- Speed up paired score comparisons on large saved datasets.
- Batch local model collection with `--batch-size` and keep primary answer decoding per prompt with `--serial-greedy`. Report per-prompt evaluations and model invocations separately.
- Plan independent holdout label budgets and certifiable error targets with `caat plan`.
- Import multiple-choice likelihoods and align external scores to saved lm-eval generations with `caat lm-eval`.
- Run score choice, probability calibration, cutoff selection, holdout checks, and review queue generation from one TOML file with `caat run`.
- Reuse existing design, calibration, and holdout files in a TOML workflow, with alternate grades supplied on the holdout.
- Align external confidence scores to graded Every Eval Ever sample files with `caat eee`.
- Collect responses and confidence from OpenAI-compatible endpoints with `caat collect --base-url`. Keep text-based signals when probability fields are unavailable.
- Add an executable IFEval tutorial notebook for cutoff selection, held-out grader checks, and review queues.
- Replay score comparisons across Qwen2.5, Phi-4-mini, and SmolLM2 on shared TriviaQA and MMLU-Pro questions.
- Build searchable documentation with the bundled examples and benchmark tables in CI.
- Run `caat audit` as a reusable GitHub Action with a numeric cutoff or saved cutoff report. A missed risk target fails the check.
- Export grader disagreements for review with the sample ID, output key, score, cutoff decision, and grades. The audit action can publish the CSV path.
- Document how to fit and compare a combined confidence score, with held-out replays on TriviaQA and MMLU-Pro.
- Read token confidence and grader outcomes from saved HELM run directories with `caat helm`.
- Accept JSONL prediction tables, grader files, and external confidence scores alongside CSV.
- Reject malformed CSV records before aligning scores or grades and report their file and line number.
- Monitor the accepted risk of a frozen policy as labels arrive, with an anytime-valid alert and a chronological benchmark replay.
- Ship type hints for the public Python API and check ranking, calibration, and binomial calculations with property-based tests.
- Replay a WildBench 7B/72B cascade against large-only and random routing at matched modeled cost.
- Compare zero, two, five, and ten additional generations on the frozen three-family benchmark.
- Check native Inspect logs in CI with the minimum and latest compatible SDK versions.
- Resolve the Inspect extra's dependencies on Python 3.10 and use SDK versions that close log file streams.
- Compare fitted CAAT and CalibRoute cutoffs on five HELM archives with independent holdout and paired grader checks.
- Verify Bonferroni and Holm threshold selection against MAPIE on shared candidate cutoffs.
- Measure the effects of the candidate grid and multiple-testing correction in the CalibRoute comparison.
- Set the number of design quantiles for `caat select`, `caat policy`, and the TOML workflow.
- Prepare related report files together and restore previous outputs if a replacement fails.
- Plan and check labels sampled without replacement from a fixed batch of accepted predictions.
- Check a sampled fixed batch in CI with the audit action's `accepted-population` input.
- Read the accepted population from `--batch-decisions` and verify labeled samples against the routed outputs.
- Filter a mixed prediction export by column value in `caat evaluate` and `caat route`.
- Replay cutoff fitting with primary, per-grader, and either-grader errors on six archives.
- Compare design score choice and jointly tested score-cutoff pairs at fixed risk targets.
- Enable joint score and cutoff fitting in TOML with `score_selection = "joint-calibration"` and one Bonferroni correction.
- Preserve carriage returns in queue CSV fields and restore previous queues if either output fails.
- Verify a fixed-batch label sample against the seeded random queue in `evaluate`, `audit`, and the audit action.
- Filter mixed prediction exports before score comparison, cutoff fitting, and TOML workflows.
- Plan labels from a saved cutoff report and check routed batch decisions against that cutoff.
- Add higher-is-better copies of external uncertainty scores when joining Inspect, lm-eval, and Every Eval Ever logs.
- Join several named score files to the same lm-eval or Every Eval Ever samples in one command.
- Use a saved label plan to check the sample size and risk settings in `evaluate`, `audit`, and the audit action.
- Reuse seeded audit bootstrap draws for all graders and aggregate repeated outcomes.
- Match the accepted outputs in a decision file against the saved label plan, including outputs outside the sample.
- Add `caat orient` to prepare lower-is-better columns in prediction tables for comparison, fitting, and routing.
- Apply probability maps directly from saved `policy` and `run` reports with `caat calibrate apply`.
- Stream `calibrate apply` output and replace the destination after a complete write.
- Export a seeded label sample with `caat plan --sample-output` and verify its draw from the saved plan during audits.
- Stream `route` decisions to reduce memory use on large batches.
- Include fields from the original outputs in a planned label sample by matching IDs and output keys.
- Audit selected outputs against full grader exports and read grader keys from a separate column name.
- Set `lower_is_better` in a TOML workflow and reuse its saved score direction with raw input scores.
- Use `ltt-holm` for joint score and cutoff selection and report decisions for the full candidate family.
- Compare lower-is-better scores in `scores` and fit reusable cutoffs on their original values in `select` and `policy`.
- Monitor selected predictions with full grader exports, matching arriving labels by ID and output key.
- Read separate grader exports for holdout checks in TOML workflows and include the joined labels in the holdout CSV.
- Reuse a saved cutoff's output key column for routing, holdout checks, monitoring, and label sample sources.
- Export exact response hashes with `caat features` and fit cutoffs from separate grader files in `select` and `policy`.
- Route predictions with missing scores to review on request and keep them in the coverage denominator.
- Monitor keyed routing decisions in their original order as labels arrive, including review rows without scores.
- Compare scores with missing values at shared coverage budgets and report how many paired bootstrap draws reach each budget.
- Fit and check acceptance policies that send unscored outputs to review, and reuse that rule from saved cutoff reports.
- Extend missing-score review to TOML workflows and queues, and preserve unscored rows when applying a probability map.
- Keep unscored answers in partial external score joins for Inspect, lm-eval, and Every Eval Ever.
- Alert on low acceptance in rolling coverage windows, including while monitor labels are pending.
- Monitor coverage windows directly from unlabeled scores or routing decisions.
- Allow partially scored inputs in GitHub Action audits with numeric cutoffs.
- Preserve blank scores when `caat orient` exports lower-is-better columns for partial inputs.
- Fit standalone probability maps on scored calibration rows and carry the missing-score rule into application.
- Use exact decimal row budgets for score comparisons and their paired bootstrap.
- Fit and apply probability maps to selected tiers in combined prediction exports.
- Fit standalone maps directly on lower-is-better scores and reuse their saved direction on new batches.
- Read separate grader ID headers when fitting, auditing, and monitoring, including in TOML workflows and the audit action.
- Check for reused sample IDs or output identities when reading a separate cutoff design table.
- Read renamed raw scores with the saved direction when reusing cutoffs and probability maps.
- Fit a standalone probability map from saved scores and a separate grader export matched by ID and output key.
- Evaluate saved scores against one separate grader export, including holdout risk and probability calibration metrics.
- Compare multiple confidence scores against separate grader exports matched by ID and output key.
- Run a TOML workflow with separate keyed grader exports for score comparison, fitting, and holdout checks.
- Track the outputs used to fit a calibration map and check for reuse when applying it to a holdout.
- Report the latest coverage window alongside a monitor's first coverage alert.
- Read the columns of a planned label sample directly in `evaluate` and `audit`, including scores already oriented by routing.
- Match the score scale of copied label samples against their batch decisions when the original and routed scores share a header.
- Find minimum label budgets for large fixed batches faster by carrying exact sample counts between budgets.
- Speed up fixed-budget planning and fixed-batch audits with bounded integer comparisons of hypergeometric tails.
- Rule out infeasible ranges of fixed-batch label budgets using a power bound, then check the remaining budgets in order.
- Use exact binomial comparisons and power bounds when planning label budgets for independent samples.
- Invert the exact binomial tail for independent holdout bounds and fixed-budget label plans, including equality at the risk target.
- Reduce binomial tail evaluations when computing independent holdout bounds.
- Compare exact binomial tails at the Bonferroni, fixed-sequence, and Holm test levels.
- Compare fixed-batch error counts with decimal targets using exact ratios and round reported risk bounds up.
- Compare observed error ratios exactly in empirical selection and evaluation.
- Resolve sequential monitor alerts and p-value rounding with exact mixture comparisons.

## 0.2.0

- Fit and apply temperature, Platt, isotonic, and histogram probability maps.
- Report Brier score, ECE, ACE, and reliability bins on saved predictions.
- Export reliability SVGs and self-contained HTML evaluation, audit, and policy reports.
- Queue saved predictions for review under a budget, with group limits and a random baseline.
- Score saved confidence checks, answer samples, and semantic clusters. Collect model responses with the optional local Transformers backend.
- Compare candidate scores against random and length controls with paired intervals and a design-only recommendation.
- Keep only the IDs, outcomes, and scores needed to replay the course comparisons.
- Discover installed feature, calibrator, metric, rule, adapter, and renderer plugins through Python entry points. List them with `caat plugins`.
- Benchmark Qwen2.5-7B confidence scores on TriviaQA and MMLU-Pro, with held-out comparisons and measured calls and latency.

## 0.1.0

- Add the Python API and CLI for evaluating, auditing, and selecting review policies.
- Standardize JSON reports and exit codes for use in CI and automation.
- Compute scores from saved token log probabilities and export SVG reports.
- Include CPU benchmark replays, published-baseline cross-checks, and an audit of public routing predictions.
- Support empirical, Bonferroni, fixed-sequence, and Holm threshold selection.
