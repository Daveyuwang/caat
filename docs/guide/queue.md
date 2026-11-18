# Review queues and audit samples

A review queue prioritizes low-confidence predictions for correction. A random audit sample estimates error among predictions that meet a cutoff.

## Route a scored batch

```bash
caat route examples/audit/predictions.csv --threshold 0.75 --output-key-column output_key --output replay/decisions.csv --json
```

The decision CSV contains each ID, score and `accept` or `review`. Use `--keep-column` to include a source field. The following command queues up to 50 flagged predictions.

```bash
caat queue replay/decisions.csv --prediction-filter decision=review --budget 50 --output replay/queue.csv
```

## Draw an audit sample after a policy misses its target

The ANLI policy has a selected cutoff. Its held-out check withheld routing. Draw 100 eligible outputs on R2 and use the saved grades to check the batch.

```bash
caat plan --threshold-report results/anli_r2/policy.json --batch-predictions results/anli_r2/predictions.csv --label-budget 100 --sample-seed 42 --sample-output replay/r2-sample.csv --output replay/r2-plan.json
caat evaluate replay/r2-sample.csv --grader-file anli=results/anli_r2/sample-grades.csv --grader-output-key-column output_key --threshold-report results/anli_r2/policy.json --batch-predictions results/anli_r2/predictions.csv --plan-report replay/r2-plan.json
```

The draw contains 43 errors. Its one-sided 95% lower bound of 35.34% exceeds the 20% target (exit 1). For an already routed population, use `--batch-decisions` to draw from its accepted rows.

## Issue audit labels in blocks

Preview candidate budgets on the UCI Letters batch. Select an offer, save the draw and issue the first request.

```bash
caat plan --forecast --threshold-report results/letter_audit/policy.json --batch-predictions results/letter_audit/predictions.csv --budget-ceiling 500 --error-scenarios .02 .08 --power .8 --request-cost 1 --output replay/offers.json
caat plan --threshold-report results/letter_audit/policy.json --batch-predictions results/letter_audit/predictions.csv --from-comparison replay/offers.json --choose lowest-cost --sample-output replay/letters.csv --output replay/letters-plan.json
caat audit-block --threshold-report results/letter_audit/policy.json --batch-predictions results/letter_audit/predictions.csv --plan-report replay/letters-plan.json --handoff replay/letters --grader dataset
```

Choose `smallest-cap` to reserve fewer labels. Use `--sample-source` and `--sample-column` at draw time to include a question or model answer in the requests.

Give the issued CSV to the named grader. A receipt contains `id`, `output_key` and binary `correct`. Repeat the handoff command with `--request REQUEST_ID --grader-file dataset=receipt.csv`. Partial receipts keep the same request open. Repeating the command resumes from the saved handoff directory. A null `request` marks completion.

## Stop when the fixed-budget verdict is determined

Use `caat evaluate --partial` with the original plan, batch and returned grades. The report gives the range of final error counts still possible. If all of them imply the same verdict, grading can stop. Two-stage designs instead use the boundaries at their planned checkpoints. [Planning report](reports.md#plan) · [Observed stopping times](../research/benchmarks.md#batch-audits)
