# Start from a classifier's predictions

The [digit example](../../examples/digit_classifier.jsonl) contains 1,079 predictions with a predicted digit, a reference label and two scores. Compare the scores at the same review budget.

```bash
caat scores examples/digit_classifier.jsonl --id-column image_id --score-column top_prob --score-column margin --prediction-column predicted --reference-column actual
```

At 75% coverage, both scores give 0.309% error on the held-out rows. Fit a policy with a 2% risk target and apply it to the file.

```bash
caat policy examples/digit_classifier.jsonl --id-column image_id --score-column top_prob --prediction-column predicted --reference-column actual --risk-target 0.02 --calibrate temperature --output replay/digits-policy.json --html replay/digits.html
caat route examples/digit_classifier.jsonl --threshold-report replay/digits-policy.json --keep-column predicted --output replay/digits-decisions.csv
```

The cutoff accepts 321 of 432 held-out predictions with one error. Applied to the whole file it accepts 795 rows and sends 284 to review. The [example generator](../../scripts/prepare_digit_example.py) fits the classifier on a separate training split.

## Hand off a class-aware policy

A fitted scorer can use the predicted class along with numeric inputs. Run the example from a checkout.

```bash
python -m pip install ".[scorer,workflow]"
caat run examples/class_aware_digits.toml
caat route examples/class_batch.csv --threshold-report replay/class-aware/policy.json --output replay/class-decisions.csv
caat evaluate examples/class_batch_graded.csv --threshold-report replay/class-aware/policy.json --correct-column correct
caat audit examples/class_batch_graded.csv --threshold-report replay/class-aware/policy.json --correct-column correct --correct-column second_grade
caat monitor examples/class_batch_graded.csv --threshold-report replay/class-aware/policy.json --correct-column correct
```

The scorer accepts `new-a` and sends `new-b` to review. An unseen or missing class also goes to review under this configuration. [Training configuration](../../examples/class_aware_digits.toml)
