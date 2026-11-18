# Use saved lm-evaluation-harness samples

Run lm-eval with `--log_samples` and pass one per-task `samples_*.jsonl`
file to `caat lm-eval`. The standard-library reader was checked against
lm-eval 0.4.13 output and the public MMLU-ProX generation archive.
[lm-eval logging options](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)
· [MMLU-ProX logs](https://huggingface.co/datasets/gililior/mmlu-prox-eval-predictions/tree/7ed529332cc75dfa3700ee35cbfb802d54f779a3/mmlu_prox_en/Qwen__Qwen3.5-2B)

## Use multiple-choice log-likelihoods

```bash
caat lm-eval choices samples_arc_easy.jsonl --output replay/choices.csv
caat scores replay/choices.csv --score-column loglikelihood_margin --score-column max_loglikelihood --correct-column acc --baseline max_loglikelihood
```

The scores are the winning option's raw log-likelihood and its lead over
the runner-up. The reader verifies that the saved binary `acc` outcome
matches the raw likelihood winner and target index. `acc_norm` uses a
different choice rule in lm-eval. Each exported row includes the document
ID, filter, prompt hash, and the SHA-256 of the winning continuation.

## Join a score for each saved generation

```bash
caat lm-eval export samples_task.jsonl --filter strict-match --output replay/to-score.csv
caat lm-eval join samples_task.jsonl --filter strict-match --scores replay/scored.csv --grader exact_match --output replay/graded.csv
caat policy replay/graded.csv --score-column confidence --correct-column exact_match --risk-target 0.4
```

Select a filter from the log when it has more than one. The export
supports one raw text response per document. Score the `input` and
`completion` from the export and write `id`, `filter`, `prompt_hash`,
`output_sha256`, and `confidence` to `scored.csv`. The join checks these
keys, requires one finite score per saved answer, and adds the requested
binary metrics from that filter. Use `--score-column` for a different
confidence column name. The joined CSV is also accepted by `caat scores`
and `caat evaluate`.

For incomplete score exports, use `--review-missing-scores` on `join`
and `policy`. The join keeps each selected document and checks the
identity of every record in the score file. [Missing scores](inspect.md#keep-answers-with-missing-scores)

For a perplexity score column, use `--score-column perplexity
--lower-is-better perplexity` on `join`. This keeps the original score
and adds `negative_perplexity` for comparison and cutoff fitting.

To compare several scores on the same generations, repeat `--score-file`
with a column name and file for each. Every source must cover the selected
documents and match their output hashes. The grade columns come from the
sample log.

```bash
caat lm-eval join samples_task.jsonl --filter strict-match --score-file confidence=replay/scored.csv --score-file perplexity=replay/perplexity.jsonl --lower-is-better perplexity --grader exact_match --output replay/compared.csv
caat scores replay/compared.csv --score-column confidence --score-column negative_perplexity --correct-column exact_match
```
