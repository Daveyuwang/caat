# CAAT roadmap

## 1.0.0

Take a saved batch from scoring through a routing decision, then use the same policy to audit the next labeled batch. When a cutoff misses its target, a labeler can still draw a fixed-budget audit sample and stop once every possible completion gives the same verdict. [Policy walkthrough](../README.md#reuse-one-fitted-policy-from-routing-through-audit) · [Audit handoff](guide/queue.md#issue-audit-labels-in-blocks)

For 1.0, freeze the documented CLI flags, exit statuses, Python imports, plugin contracts and report fields. [Public interface](../CONTRIBUTING.md#public-interface) Replay existing saved policies from the installed wheel and verify the distribution from a clean checkout. Keep benchmark data, protocols and generated figures linked from the [benchmark guide](research/benchmarks.md).

## After 1.0

A priority for research is cutoff stability. A class-aware score changed accepted error by 0.58 percentage points on ANLI R1 at 75% coverage, and the interval included zero. A follow-up should compare scores at matched inference cost and test the selected cutoff on a fresh batch. [ANLI results](research/anli-results.md)

For integrations, start with a real prediction table or evaluation log and the decision its owner needs to make. [Contributor guide](../CONTRIBUTING.md) · [Plugin contracts](guide/plugins.md)
