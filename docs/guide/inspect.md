# Import an Inspect log

Install the Inspect extra and export one epoch from a saved `.eval` or `.json` log.

```bash
python -m pip install '.[inspect]'
caat inspect export run.eval --epoch 1 --output scoring-input.csv
```

Score the exported completions and keep `eval_id`, `id`, `epoch` and `output_sha256` with each confidence value. Use JSONL for long or multimodal inputs. Join the scores to named graders from the log.

```bash
caat inspect join run.eval --epoch 1 --scores scored.csv --grader primary --grader alternate --output graded.csv
caat policy graded.csv --score-column confidence --correct-column primary --check-column alternate --risk-target 0.20 --holdout-confidence 0.95 --output policy.json
```

Repeat `--score-file COLUMN=FILE` to compare several scores. Use `--scores-id-column` for a score file keyed only by sample ID when the selected log and epoch identify each response uniquely.

## Keep answers with missing scores

Set `--review-missing-scores` on both the join and policy commands to send unscored outputs to review.

## Check a rescored log at a frozen cutoff

Export grades with `caat inspect grades LOG --predictions holdout.csv --grader NAME --output grades.csv`. Use the original and rescored exports in the [grader audit](ci-audit.md).
