# Compare tools for confidence and review policies

CAAT takes saved predictions, confidence scores, and grades. It selects an
acceptance cutoff, checks coverage and error on a separate holdout, and
audits the frozen cutoff when the grader changes.

| Tool | Use it for | Documentation |
| --- | --- | --- |
| UQLM | Score LLM answers with built-in uncertainty estimators or tuned ensembles. Use its calibration tools to convert scores into correctness probabilities. | [Scorers and ensembles](https://cvs-health.github.io/uqlm/latest/_notebooks/examples/ensemble_tuning_demo.html) · [Calibration](https://cvs-health.github.io/uqlm/latest/_notebooks/examples/score_calibration_demo.html) |
| LM-Polygraph | Run uncertainty estimators that use text, token scores, or model internals, and compare uncertainty quality with metrics such as AURC. | [Estimator library](https://github.com/IINemo/lm-polygraph) · [Risk–coverage metric](https://lm-polygraph.readthedocs.io/en/latest/api/lm_polygraph.ue_metrics.risk_cov_curve.html) |
| MAPIE | Fit conformal predictors and Learn then Test risk controllers in Python. Its LLM judge example uses two thresholds to classify answers or abstain. | [Precision control](https://mapie.readthedocs.io/en/stable/generated/risk_control/1-quickstart/plot_risk_control_binary_classification/) · [LLM judge abstention](https://mapie.readthedocs.io/en/latest/generated/risk_control/2-advanced-analysis/plot_risk_control_llm_as_a_judge/) |
| CalibRoute | Fit a confidence-based policy, validate it on an independent holdout, and route a new batch to accept, review, or abstain. | [Fit, validate, and route](https://github.com/Garyouki/calibroute#quick-start) |
| CAAT | Compare saved confidence scores, select and independently check a cutoff, and trace grader disagreements. Route a scored batch and audit a random sample of its accepted outputs. | [Inspect workflow](guide/inspect.md) · [Fixed-batch audit](guide/queue.md#check-the-error-rate-of-a-fixed-batch) · [CI audit](guide/ci-audit.md) |

CalibRoute and CAAT both offer a one-sided exact check of an independent
holdout at a frozen cutoff. Their fitting rules differ. CalibRoute can
search calibration thresholds using empirical risk or pointwise exact
bounds. CAAT also provides Learn then Test rules that account for testing
multiple cutoffs in a prespecified grid.
[CalibRoute's holdout scope](https://github.com/Garyouki/calibroute#input-format) · [CAAT selection reports](guide/reports.md#select)

The HELM replay compares fitted cutoffs on five archives using shared
scores, splits, and targets. It reports acceptance and holdout error
under two graders. The shared-grid diagnostic measures the effects of
the correction and the candidate grid on acceptance.
[Fitted cutoff comparison](research/benchmarks.md#fitted-cutoffs-versus-calibroute) · [Shared-grid diagnostic](research/benchmarks.md#candidate-grid-and-multiple-testing-correction)

On the same HELM archives, both CAAT and MAPIE qualify the same
cutoffs with Bonferroni and Holm when given a shared design grid.
[Shared cutoff comparison](research/benchmarks.md#shared-cutoffs-with-mapie)

With scores from another tool, attach one finite value per saved answer.
Orient each score so larger values favor acceptance. Keep the answer ID
and output hash so graders and score exports can be joined to the same
answer. Use `caat scores` to compare the signals and `caat policy` to
select and check a cutoff.
[Score comparison](guide/scores.md)
