# Import lm-evaluation-harness samples

Enable `--log_samples` in lm-eval, then use the per-task `samples_*.jsonl` file.

## Use multiple-choice log-likelihoods

Compare the highest option log-likelihood and its margin over the runner-up. The grade is the raw-likelihood winner’s `acc`.

```bash
caat lm-eval choices samples_arc_easy.jsonl --output replay/choices.csv
caat scores replay/choices.csv --score-column loglikelihood_margin --score-column max_loglikelihood --correct-column acc --baseline max_loglikelihood
```

## Join a score for each saved generation

Export the saved completions, score them and return the identity columns with each score. Use the same filter for both commands.

```bash
caat lm-eval export samples_task.jsonl --filter strict-match --output replay/to-score.csv
caat lm-eval join samples_task.jsonl --filter strict-match --scores replay/scored.csv --grader exact_match --output replay/graded.csv
caat policy replay/graded.csv --score-column confidence --correct-column exact_match --risk-target 0.4
```

Repeat `--score-file COLUMN=FILE` to join several scores. Use `--lower-is-better COLUMN` for perplexity or another reversed score. [Input format](csv-format.md)
