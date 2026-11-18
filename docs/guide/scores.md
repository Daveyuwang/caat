# Compare confidence scores

Compare scores on the same predictions, grader and review budget. The IFEval example has two log-probability scores and two independent grading implementations.

```bash
caat scores examples/ifeval_graders.csv --score-column mean_token_logprob --score-column sequence_logprob --correct-column strict_pass --correct-column helm_strict_pass
```

The report gives AUROC, area under the risk–coverage curve and accepted error at 25%, 50% and 75% coverage. Change the budgets with `--coverage`. Ties at the budget boundary receive their expected share under uniform selection.

| Option | Use |
| --- | --- |
| `--baseline COLUMN` | Compare each score with a reference on the same rows. The first score is the default. |
| `--lower-is-better COLUMN` | Reverse the direction of a score such as perplexity. |
| `--class-column COLUMN` | Report coverage, accepted error and review counts by predicted class at the same global budget. |
| `--review-missing-scores` | Include unscored predictions in review volume. |
| `--grader-file NAME=PATH` | Join grades from a separate export. |

Use design rows to choose among scores and held-out rows to report performance. [Score definitions](features.md) · [Prediction tables](csv-format.md)
