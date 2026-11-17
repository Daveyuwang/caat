# Combine confidence scores on the A3 archives

This replay uses the saved TriviaQA and MMLU-Pro scores for Qwen2.5-7B,
Phi-4-mini, and SmolLM2-1.7B. Each model and task keeps the same 200 design
rows and 400 held-out rows. The combination uses the complete primary
score columns from A3, apart from the constant random control. The
calibration fold is reserved. [A3 benchmark](model-families.md) ·
[Frozen split](../../results/black_box_families/split.csv)

## Protocol

For each model and task, fit L2 logistic regression with either
standardized raw scores or standardized uniform quantiles. The quantile
transform uses the full training partition and its size as the number of
quantiles. Test `C` values of 0.01, 0.1, and 1.0. Fit all preprocessing and
weights inside each training fold. The combination with the highest mean
validation normalized AURC gain wins. Use stratified folds, up to five,
with seed 42. Refit the selected configuration on all design rows and
export its decision score. The strongest single comparator is the column
with the highest normalized AURC gain on all design rows, as in A3.
[Logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) ·
[Quantile transform](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)

Measure expected error among the 300 accepted answers at 75% coverage.
Report each family and the equal-weight mean across families for each
task. The paired bootstrap resamples questions with the same draw across
all three families and recomputes the ranked budget. Use 2,000 draws and
seed 20261115. The core inclusion check requires a task-level error
delta below zero with its 95% interval entirely below zero on both tasks.

## Held-out results

| Task | Strongest design-selected score | Combined score | Difference, percentage points (95% interval) |
| --- | ---: | ---: | ---: |
| TriviaQA | 57.44% | 57.44% | 0.00 (−0.56 to +0.44) |
| MMLU-Pro | 79.00% | 78.89% | −0.11 (−0.56 to +0.67) |

At 75% coverage, the MMLU-Pro difference amounts to one fewer expected
error across 900 accepted answers. Both intervals include zero. Use the
combination recipe below to test selected columns on your own holdout.
Fitting and applying the combination uses saved scores and makes zero
model calls. Score collection costs depend on the input columns.
[Task results](../../results/combined_scores/task_results.csv) ·
[Per-model results](../../results/combined_scores/model_results.csv) ·
[Design selections](../../results/combined_scores/selection.json)

## Replay the comparison

Run from the repository root in a Python 3.12 environment. The script
writes the fitted choices, held-out scores, and comparisons to
`replay/combined_scores`. [Replay script](../../scripts/study_combined_scores.py)

```bash
python -m pip install . numpy==1.26.4 scipy==1.17.1 scikit-learn==1.9.1
python scripts/study_combined_scores.py
```

The [saved Qwen TriviaQA scores](../../results/combined_scores/qwen2_5_7b/triviaqa_heldout.csv)
contain the design-selected single score and the combined score for the
same rows. Compare them with the base CAAT installation.

```bash
caat scores results/combined_scores/qwen2_5_7b/triviaqa_heldout.csv --score-column single_score --score-column combined_score --correct-column correct --baseline single_score --split all --coverage 0.75 --bootstrap 2000 --seed 20261115
```

## Combine your own score columns

Prepare separate `design.csv` and `holdout.csv` files with `id`, binary
`correct`, and the same finite score columns. Choose the input columns,
design baseline, and regularization using the design rows. This example
uses a fixed `C` of 0.1. Run it with the Python packages above.

```python
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def score_matrix(rows, columns):
    return [[float(row[name]) for name in columns] for row in rows]


columns = ("sequence_logprob", "normalized_sample_agreement")
design = read_rows("design.csv")
holdout = read_rows("holdout.csv")
model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, l1_ratio=0.0))
model.fit(score_matrix(design, columns), [int(row["correct"]) for row in design])
scores = model.decision_function(score_matrix(holdout, columns))

fields = ("id", "correct", *columns, "combined_score")
with open("holdout_combined.csv", "w", encoding="utf-8", newline="") as target:
    writer = csv.DictWriter(target, fieldnames=fields)
    writer.writeheader()
    for row, score in zip(holdout, scores, strict=True):
        row = {name: row[name] for name in fields[:-1]}
        writer.writerow({**row, "combined_score": float(score)})
```

Set `--baseline` to the single column you selected on design. This
command compares the exported scores on the frozen holdout.

```bash
caat scores holdout_combined.csv --score-column sequence_logprob --score-column normalized_sample_agreement --score-column combined_score --correct-column correct --baseline sequence_logprob --split all --coverage 0.75
```
