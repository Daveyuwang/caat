# Queue model labels for human review

The [Jigsaw example](../../results/course/jigsaw_lr_predictions.csv) contains 3,000 model labels paired with human toxicity labels. Compare the model’s stated confidence with a fitted trust score, then select a cutoff for a 5% error target.

```bash
caat scores results/course/jigsaw_lr_predictions.csv --id-column example_id --score-column llm_confidence_numeric --score-column trust_score_calibrated --prediction-column llm_label --reference-column gold_label
caat policy results/course/jigsaw_lr_predictions.csv --id-column example_id --score-column trust_score_calibrated --prediction-column llm_label --reference-column gold_label --risk-target 0.05 --output replay/jigsaw-policy.json
```

Use the saved policy to route the predictions and queue up to 200 for review.

```bash
caat route results/course/jigsaw_lr_predictions.csv --threshold-report replay/jigsaw-policy.json --keep-column llm_label --output replay/jigsaw-decisions.csv
caat queue replay/jigsaw-decisions.csv --prediction-filter decision=review --budget 200 --output replay/jigsaw-queue.csv
```

The cutoff keeps 2,066 model labels and flags 934. The queue contains the 200 lowest-scoring flagged rows. To estimate error among the labels the policy would keep on another batch, draw a [random audit sample](queue.md).
