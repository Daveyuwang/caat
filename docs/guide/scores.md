# Compare confidence scores on saved outcomes

```bash
python scripts/prepare_ifeval.py --full
caat scores replay/ifeval/predictions_full.csv --score-column mean_token_logprob --score-column sequence_logprob --length-column token_count --correct-column strict_pass --correct-column helm_strict_pass
```

Supply one row per saved response with a unique ID, one or more finite
score columns, and one or more binary correctness columns. Higher scores
should indicate higher confidence. The report includes AUROC, AURC,
normalized AURC gain, and accepted-set error at 25%, 50%, and 75%
coverage. Repeat `--coverage FRACTION` for other budgets. The budget
rounds up to a row count and averages over ties at its boundary.

Random ordering uses the full set's error rate at every budget. With
`--length-column NAME`, a second control accepts shorter outputs first.
Each candidate's error difference from `--baseline NAME` comes with a
paired 95% bootstrap interval. The first score is the default baseline.
The bootstrap resamples complete rows and recomputes both rankings on the
same draw. `--bootstrap 0` skips intervals.

Cost classes describe collection work per prediction. Built-in feature
names carry their recorded class. Other columns default to `unknown`.
Repeat `--score-cost NAME=CLASS` to choose `no-extra-call`,
`one-extra-call`, `n-samples`, `external`, or `unknown`.

The default uses the same 20/40/40 ID split as `caat policy`. CAAT
recommends the candidate with the highest positive mean normalized AURC
gain on the design rows. The mean uses graders with both correct and
incorrect outcomes. The comparison reports risk on held-out rows. Keep
the score selected on design rows and use the same seed for policy
selection. Use `--split all` to compare a separate, already frozen
evaluation CSV. This mode omits the recommendation.

Use `--json` for the full report, including per-grader controls, coverage
budgets, bootstrap intervals, and the design recommendation.
[Archived comparisons](../research/benchmarks.md#confidence-scores)
