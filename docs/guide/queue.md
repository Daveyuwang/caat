# Queue predictions for review

```bash
caat queue predictions.csv --score-column confidence --budget 50 --output replay/queue.csv --random-output replay/random_queue.csv
```

The queue accepts a CSV with unique IDs and finite scores. It orders rows
from lowest confidence to highest, breaking score ties by ID. With
`--direction lower`, high scores enter the queue first, which is useful
with uncertainty scores such as semantic entropy.

The output is an ordered CSV with `id`, `score`, and `group` columns.
Use `--id-column`, `--score-column`, and `--group-column` to specify
different input headers. Repeat `--quota GROUP=COUNT` to place an upper
limit on selected rows from each named group. Both the score-ordered queue
and the random queue fill the same budget under the same limits.

Repeat `--correct-column NAME` to replay the queue against saved outcomes.
The report counts errors found under each grader for both ordering
methods. Without quotas, it also gives the random queue's expected number
of errors. `--seed` controls the random ordering and defaults to 42.
`--json` prints the full report.
[Course benchmark](../research/benchmarks.md#review-queues)
