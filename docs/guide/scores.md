# Compare confidence scores on saved outcomes

```bash
python scripts/prepare_ifeval.py --full
caat scores replay/ifeval/predictions_full.csv --score-column mean_token_logprob --score-column sequence_logprob --length-column token_count --correct-column strict_pass --correct-column helm_strict_pass
```

Supply one row per saved response with a unique ID and one or more finite
score columns. Read correctness with `--correct-column` or `--grader-file`.
Higher scores indicate higher confidence by default. Repeat `--lower-is-better COLUMN`
for scores such as perplexity. The report uses the original column names,
including for `--baseline` and `--score-cost`, and includes AUROC, AURC,
normalized AURC gain, and accepted-set error at 25%, 50%, and 75%
coverage. Repeat `--coverage FRACTION` for other budgets. CAAT multiplies
the decimal coverage by the cohort size and rounds up to a row count.
A 28% budget on 400 rows requests 112 rows. Risk averages over score
ties at the budget boundary.

Random ordering uses the full set's error rate at every budget. With
`--length-column NAME`, a second control accepts shorter outputs first.
Each candidate's error difference from `--baseline NAME` comes with a
paired 95% bootstrap interval. The first score is the default baseline.
The bootstrap resamples complete rows and evaluates each score on the same
draw. CAAT orders scores once and uses the count of each row in the draw to
compute risk at each budget. Large CLI runs use up to eight processes by
default. Use `--workers 1` to run on one process. The seeded draws and
reported intervals are the same for every worker count.
`--bootstrap 0` skips intervals.

Cost classes describe collection work per prediction. Built-in feature
names carry their recorded class. Other columns default to `unknown`.
Repeat `--score-cost NAME=CLASS` to choose `no-extra-call`,
`one-extra-call`, `n-samples`, `external`, or `unknown`.

The default uses the same 20/40/40 ID split as `caat policy`. CAAT
recommends the candidate with the highest positive mean normalized AURC
gain on the design rows. The mean uses graders with both correct and
incorrect outcomes. The comparison reports risk on held-out rows. Keep
the score selected on design rows and use the same seed for policy
selection. Use `--direction lower` when fitting a cutoff on a score that
was compared with `--lower-is-better`. Use `--split all` to compare a
separate, already frozen evaluation CSV. This mode omits the recommendation.

Use `--json` for the full report, including per-grader controls, coverage
budgets, bootstrap intervals, and the design recommendation.
[Archived comparisons](../research/benchmarks.md#confidence-scores)

## Compare with separate grader exports

Use `--grader-file NAME=FILE` for each grader's CSV or JSONL export. CAAT
matches the ID and output key in every score row to the grader's binary
`correct` value. The default key column is `output_sha256`.

```bash
caat scores scores.csv --score-column mean_token_logprob --score-column uqlm_exact_match --score-column lm_polygraph_rouge_l --grader-file reference=grades.csv --grader-file alternate=grades.jsonl --allow-extra-grades
```

Add `--allow-extra-grades` for full grading exports when comparing a
selected cohort. Set `--grader-id-column` or `--grader-output-key-column`
if the grader headers differ from the prediction headers.
[Separate grader exports](csv-format.md#separate-grader-exports)

## Compare scores with missing values

Add `--review-missing-scores` when a score has empty CSV cells or JSONL
nulls. Each candidate can accept its scored rows. A coverage budget of
`c` still requests `ceil(c * n)` rows from the full cohort of size `n`.
The report marks unreachable budgets and leaves their risk estimates null.
`scored` and `maximum_coverage` show what each candidate can reach.

AUROC and AURC describe the rows with a score. The design recommendation
compares candidates that score every design row. `design_scored` gives
the counts used to check eligibility.

The bootstrap resamples the full cohort, including rows with missing
scores. Each interval uses the draws where both the candidate and
baseline reach the requested budget. `risk_delta_paired_draws` counts
those draws. Risk differences and intervals require both scores to
reach the budget in the reported cohort.
[Route a batch with missing scores](queue.md#route-a-scored-batch)

To combine several score columns, fit the weights on design rows and
export one score for each held-out row. Compare it as another
`--score-column`. [Combination recipe and replay](../research/combined-scores.md)
