# Changelog

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
