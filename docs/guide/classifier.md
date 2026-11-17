# Start from a classifier's predictions

This page goes from a file of predictions to a list of rows to accept and
rows to review. It uses [`examples/digit_classifier.jsonl`](../../examples/digit_classifier.jsonl),
the 1,079 test predictions of a small digit classifier. Each line has the
image ID, the predicted and true digit, the top class probability, and the
margin between the top two classes.

```json
{"image_id": "img_0058", "predicted": 6, "actual": 6, "top_prob": 0.960963, "margin": 0.945132}
```

Your own file needs the same three things: an ID, one or more confidence
columns, and either a 0/1 grade column or a predicted and a true label.
Run the commands from the CAAT checkout after `python -m pip install .`.

## 1. Pick a confidence score

```bash
caat scores examples/digit_classifier.jsonl --id-column image_id --score-column top_prob --score-column margin --prediction-column predicted --reference-column actual
```

```text
Compared scores on 432 held-out rows. Baseline score: top_prob.
The 215 design rows favor margin by AURC gain, but it is tied with top_prob on the held-out rows. Use either, and prefer the one that is cheaper to compute.
actual:
  control random. AUROC 0.500. 25% coverage: 6.02% error, 50% coverage: 6.02% error, 75% coverage: 6.02% error.
  top_prob. AUROC 0.954. 25% coverage: 0% error, 50% coverage: 0% error, 75% coverage: 0.309% error.
  margin. AUROC 0.949. 25% coverage: 0% error, 50% coverage: 0% error, 75% coverage: 0.309% error. Tied with top_prob: every 95% interval for the difference includes zero.
```

The classifier is wrong on 6.02% of the held-out rows. Accepting the most
confident 75% by either score brings that to 0.309%. The design rows
lean toward `margin`, but on the held-out rows the two scores are tied, so
the second line says to use either. The rest of this page uses `top_prob`. An AUROC well below 0.5 means the score points the wrong way,
and the output says so.
[Terms](glossary.md#comparing-scores) · [Score comparison guide](scores.md)

## 2. Choose and check a cutoff

Say that at most 2 in 100 accepted predictions may be wrong.

```bash
caat policy examples/digit_classifier.jsonl --id-column image_id --score-column top_prob --prediction-column predicted --reference-column actual --risk-target 0.02 --calibrate temperature --output replay/digits-policy.json --html replay/digits.html
```

```text
Split with seed 42: 215 design, 432 calibration, and 432 held-out rows.
Score ranking under actual on the calibration rows: AUROC 0.954.
Probability calibration uses temperature. The policy threshold uses the raw score.
Held-out raw probabilities under actual. Brier 0.0557. ECE 11.9%. ACE 11.7%. Mean probability 82.3% against 94% correct: underconfident.
Held-out calibrated probabilities under actual. Brier 0.0429. ECE 2.98%. ACE 3.31%. Mean probability 91% against 94% correct: underconfident.
Selected threshold 0.769127 for a 2% risk target under actual (ltt-bonferroni).
The rule keeps accepted error at or below 2% with 95% confidence on new rows like the calibration rows.
The next looser candidate (threshold 0.649585, 84% of calibration rows) had 0.275% calibration error, under the target but not by enough: its p-value 0.00549 had to be at or below 0.005, the 5% failure rate shared by 10 candidates.
Held-out rows under actual: accepted 321 of 432 with 1 errors (0.312%, 95% upper bound 1.47%). The observed rate meets the target.
Applied to all 1079 rows, the cutoff accepts 795 (73.7%). 647 of the 1079 rows were used to choose the cutoff, so only the held-out rows give the error estimate.
Split check: 19 of 20 other seeds selected a cutoff and 1 selected none. Those cutoffs accept 651 to 903 of the 1079 rows (median 754). The count depends on which rows fall in calibration. Fix the seed before looking, and report that run.
The held-out rows confirm the policy.
```

Read it in this order.

- **The cutoff.** Accept a prediction when `top_prob` is at least 0.769127.
- **Why it holds.** The cutoff was chosen on 432 calibration rows by a rule
  that keeps accepted error at or below 2% with 95% confidence.
- **Why not a looser one.** The next candidate would accept 84% of rows.
  Its error was also low, but the evidence fell just short of the level
  the rule requires. This is why the result sits well under the target.
- **The check.** On 432 rows the cutoff never saw, it accepted 321 and one
  was wrong. The 95% upper bound of 1.47% is under the 2% target.
- **The workload.** In the whole file the cutoff accepts 795 rows, so 284
  go to review.
- **How stable it is.** With other random splits the cutoff would accept
  between 651 and 903 rows. A file of about a thousand rows leaves this
  much play. Quote the run whose seed you fixed first.
  [Settings that move the result](glossary.md#settings-that-move-the-result)

`--calibrate temperature` is optional. It reports whether `top_prob` can be
read as a probability. Here the raw value is underconfident: it averages
82.3% on rows that are 94% correct. The cutoff does not depend on this
step. `platt` and `isotonic` treat the input as a score, so they print the
raw line only with `--calibration-input probability`. The exit status is 0 when the held-out rows confirm the policy and 1
when they do not or when no cutoff qualifies.
[Terms](glossary.md) · [Report fields](reports.md#policy)

## 3. Apply the cutoff

```bash
caat route examples/digit_classifier.jsonl --threshold-report replay/digits-policy.json --keep-column predicted --output replay/digits-decisions.csv
```

```text
Accepted 795 of 1079 rows at threshold 0.769127. Marked 284 for review. Wrote replay/digits-decisions.csv.
```

The decisions file has one row per prediction with `id`, `score`,
`decision`, and the columns named with `--keep-column`. The saved report
supplies the cutoff and the column names. Use the same command on new
predictions from the same classifier.
[Route a scored batch](queue.md#route-a-scored-batch)

## 4. Plan a check on new data

The guarantee covers new rows that look like the calibration rows. To
confirm the cutoff on a later batch, label a sample of its accepted rows.

```bash
caat plan --threshold-report replay/digits-policy.json
```

```text
Label 236 accepted rows for the 2% risk target. Up to 1 errors certify it at 95% confidence. The chance of certification is 83.2% at the assumed error rate of 0.312%, the held-out rate in the threshold report. The label count grows quickly as the assumed rate nears the target.
```

[Sample accepted rows for labeling](queue.md#sample-accepted-rows-for-labeling) ·
[Monitor a frozen policy](monitor.md)

## Where the example comes from

`python scripts/prepare_digit_example.py` rewrites the file. It needs
scikit-learn. The script trains a deliberately weak logistic regression on
40% of scikit-learn's bundled digits data and saves its predictions on the
other 60%, so the file contains errors to route.
[Source](../../scripts/prepare_digit_example.py)

## Hand off a class-aware policy

The digits workflow fits a scorer on two numeric features and the predicted digit. Its saved JSON includes the design vocabulary. A base installation can use that report and the raw batch without the training data or scikit-learn.

```bash
python -m pip install ".[scorer,workflow]"
caat run examples/class_aware_digits.toml
caat route examples/class_batch.csv --threshold-report replay/class-aware/policy.json --output replay/class-decisions.csv
caat evaluate examples/class_batch_graded.csv --threshold-report replay/class-aware/policy.json --correct-column correct
caat audit examples/class_batch_graded.csv --threshold-report replay/class-aware/policy.json --correct-column correct --correct-column second_grade
caat monitor examples/class_batch_graded.csv --threshold-report replay/class-aware/policy.json --correct-column correct
```

The policy accepts `new-a` and sends `new-b` to review on its numeric evidence. The batch also includes an unseen class (`new-c`) and a missing prediction (`new-d`). With `review_missing_scores = true`, both go to review with an empty score. Evaluation and audit exit 1 because the one accepted grade cannot confirm a 2% target. Monitoring exits 0. [Training configuration](../../examples/class_aware_digits.toml) · [Unlabeled batch](../../examples/class_batch.csv) · [Graded batch](../../examples/class_batch_graded.csv)
