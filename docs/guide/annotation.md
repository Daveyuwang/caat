# Decide which model labels need human review

You have labels from a model, a confidence value for each, and human labels
on a sample. This page turns them into three things: a cutoff above which
the model's label is kept, a review list for the rest, and the number of
human labels needed to check the cutoff later.

The example is [`results/course/jigsaw_lr_predictions.csv`](../../results/course/jigsaw_lr_predictions.csv):
3,000 Jigsaw toxicity comments labeled by Qwen, each with a human label.

| Column | Meaning |
| --- | --- |
| `example_id` | Comment ID |
| `llm_label` | The model's label |
| `gold_label` | The human label |
| `llm_confidence_numeric` | The confidence the model stated for its label |
| `trust_score_calibrated` | A trained scorer's probability that the model's label is right |

Run the commands from the CAAT checkout after `python -m pip install .`.

## 1. Check which confidence value is worth using

```bash
caat scores results/course/jigsaw_lr_predictions.csv --id-column example_id --score-column llm_confidence_numeric --score-column trust_score_calibrated --prediction-column llm_label --reference-column gold_label
```

```text
Compared scores on 1200 held-out rows. Baseline score: llm_confidence_numeric.
Design-only recommendation: trust_score_calibrated, chosen by AURC gain on 600 design rows.
gold_label:
  control random. AUROC 0.500. 25% coverage: 22.6% error, 50% coverage: 22.6% error, 75% coverage: 22.6% error.
  llm_confidence_numeric. AUROC 0.652. 25% coverage: 16% error, 50% coverage: 16% error, 75% coverage: 16% error.
  trust_score_calibrated. AUROC 0.978. 25% coverage: 0% error, 50% coverage: 0% error, 75% coverage: 1.47% error.
```

The model's labels disagree with the human labels on 22.6% of rows. Its
stated confidence barely separates right from wrong: keeping the most
confident rows still leaves 16% error. The trained scorer does separate
them. Keeping its top 75% leaves 1.47% error.
[Terms](glossary.md#comparing-scores)

## 2. Choose the cutoff

Allow at most 5 wrong labels in 100 kept labels.

```bash
caat policy results/course/jigsaw_lr_predictions.csv --id-column example_id --score-column trust_score_calibrated --prediction-column llm_label --reference-column gold_label --risk-target 0.05 --output replay/jigsaw-policy.json
```

```text
Split with seed 42: 600 design, 1200 calibration, and 1200 held-out rows.
Score ranking under gold_label on the calibration rows: AUROC 0.969.
Selected threshold 0.9285714285714286 for a 5% risk target under gold_label (ltt-bonferroni).
The rule keeps accepted error at or below 5% with 95% confidence on new rows like the calibration rows.
The next looser candidate (threshold 0.12626995645863573, 100% of calibration rows) had 23.3% calibration error, above the target.
Held-out rows under gold_label: accepted 828 of 1200 with 4 errors (0.483%, 95% upper bound 1.1%). The observed rate meets the target.
Applied to all 3000 rows, the cutoff accepts 2066 (68.9%). 1800 of the 3000 rows were used to choose the cutoff, so only the held-out rows give the error estimate.
Split check: 20 of 20 other seeds selected a cutoff, accepting 2016 to 2321 of the 3000 rows (median 2178). The count depends on which rows fall in calibration. Fix the seed before looking, and report that run.
The held-out rows confirm the policy.
```

Keep the model's label when the score is at least 0.9286. On the 1,200
held-out rows that rule kept 828 labels and 4 disagreed with the human
label. About 69% of comments need no human review. Quote the two numbers
for what they are: 2,066 of 3,000 is the workload, and 4 errors in 828 is
the error check. Other random splits would keep between 2,016 and 2,321.
[How to read each line](classifier.md#2-choose-and-check-a-cutoff) ·
[Terms](glossary.md)

## 3. Write the keep and review lists

```bash
caat route results/course/jigsaw_lr_predictions.csv --threshold-report replay/jigsaw-policy.json --keep-column llm_label --output replay/jigsaw-decisions.csv
```

```text
Accepted 2066 of 3000 rows at threshold 0.9285714285714286. Marked 934 for review. Wrote replay/jigsaw-decisions.csv.
```

Each row of the decisions file has `id`, `score`, `decision`, and
`llm_label`. To hand reviewers a fixed number of rows, queue the review
rows with the least confident first.

```bash
caat queue replay/jigsaw-decisions.csv --prediction-filter decision=review --budget 200 --output replay/jigsaw-queue.csv
```

```text
Queued 200 of 934 rows in order of increasing confidence.
Wrote the queue to replay/jigsaw-queue.csv.
```

[Queue guide](queue.md)

## 4. Plan the labels for the next batch

The cutoff was checked on this batch. On a later batch, label a sample of
the kept rows to confirm the error is still under the target.

```bash
caat plan --threshold-report replay/jigsaw-policy.json
```

```text
Label 93 accepted rows for the 5% risk target. Up to 1 errors certify it at 95% confidence. The chance of certification is 92.5% at the assumed error rate of 0.483%, the held-out rate in the threshold report. The label count grows quickly as the assumed rate nears the target.
```

Ninety-three human labels on kept rows are enough. If at most one
disagrees with the model, the 5% target is certified for that batch.
[Sample accepted rows for labeling](queue.md#sample-accepted-rows-for-labeling) ·
[Monitor a frozen policy](monitor.md)

## With your own annotations

Use a CSV or JSONL file with an ID, the model's label, the human label,
and a confidence column. `caat policy` needs a human label on every row it
reads. If only a sample is labeled, put the labeled rows in their own file,
fit on that file with `--output policy.json`, and run `caat route` on the
full file with `--threshold-report policy.json`.

If the stated confidence is the only score and it ranks poorly, as it does
here, a second signal usually helps: a small classifier's probability of
the model's label, agreement across repeated samples, or token log
probabilities. Compare candidates with `caat scores` before choosing a
cutoff.

A score must rise with the chance that the model's label is right. A
classifier's probability of one fixed class does not: it is high for that
class and low for the other, whichever label the model gave. `caat scores`
reports an AUROC well below 0.5 for such a column. Convert it to the
probability of the label the model chose before using it. For a binary
task with `p_toxic` and a 0/1 `model_label`:

```python
score = p_toxic if model_label == 1 else 1 - p_toxic
``` [Score definitions](features.md) ·
[Where the trained scorer comes from](../research/method.md)
