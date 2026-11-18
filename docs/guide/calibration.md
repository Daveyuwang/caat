# Calibrate correctness probabilities

Fit a map using labeled calibration rows. Apply the saved JSON map to a
separate CSV or JSONL file and evaluate its probabilities on held-out labels.

```bash
caat calibrate fit calibration.csv --score-column confidence --correct-column correct --method isotonic --output replay/map.json
caat calibrate apply heldout.csv --score-column confidence --map replay/map.json --output replay/heldout_calibrated.csv
caat evaluate replay/heldout_calibrated.csv --score-column calibrated_probability --correct-column correct --threshold 0.75 --probability --html replay/calibration.html
```

The input to `fit` has a unique ID, score, and binary correctness for each
row. Use `--id-column` for a different ID header. The `apply` command appends
`calibrated_probability` to the input columns. Set `--probability-column`
to choose a different name. CAAT writes the CSV row by row and replaces
the destination after a complete write.

| Method | Input and fit |
| --- | --- |
| `temperature` | A probability by default. Fits a positive inverse temperature on binary correctness logits by minimizing log loss. Set `--input-scale score` to supply binary correctness logits directly. |
| `platt` | A real-valued score by default. Fits a logistic map with an unpenalized intercept and unit L2 penalty on the slope after dividing scores by their maximum absolute calibration value. A single-class fit uses the Laplace-smoothed base rate. |
| `isotonic` | A real-valued score by default. Fits a nondecreasing least-squares map, interpolates between adjacent scores, and clips to the fitted endpoints. |
| `histogram` | A probability. Fits observed correctness in fixed-width bins, with the calibration base rate for empty bins. Use `--bins N` to set the number of bins. |

For temperature and Platt scaling, `--input-scale probability` converts
the input to logits and clips probabilities of 0 and 1 before conversion.
Use a binary correctness logit rather than a next-token log probability
for temperature scaling on the score scale.

Use `caat policy --calibrate METHOD` to fit one map per selection grader
on the calibration split and check the probabilities on held-out rows.
The design grid, risk test, and held-out acceptances all use the original
scores. Keeping this grid on the original scale preserves its independence
from the labels used in the risk test. The calibration map may introduce
ties, and Platt or histogram fitting can also change the rank order.

## Fit from a separate grader export

Use `--grader-file NAME=FILE` to fit from a saved score table and one
grader's CSV or JSONL export. CAAT matches the ID and output key. Each
selected row needs one binary `correct` grade for that output.

```bash
caat calibrate fit calibration-scores.csv --score-column confidence --grader-file reference=all-grades.csv --allow-extra-grades --method platt --input-scale probability --output replay/map.json
caat calibrate apply heldout-scores.csv --map replay/map.json --output replay/heldout_calibrated.csv
caat evaluate replay/heldout_calibrated.csv --score-column calibrated_probability --grader-file reference=all-grades.csv --allow-extra-grades --threshold 0.75 --probability --html replay/calibration.html
```

The example uses a full grader export. `--allow-extra-grades` matches
the IDs and output keys in each split. The map records the score column,
grader file, identity columns, and calibration row counts. Set
`--grader-id-column` or `--grader-output-key-column` for different
headers in the grader file. [Separate grader exports](csv-format.md#separate-grader-exports)

## Fit on lower-is-better scores

Use `--direction lower` with a score-scale method to fit directly on
perplexity or another score where smaller values imply greater confidence.
The map records the raw score column and the sign change, which `apply`
uses on the new batch.

```bash
caat calibrate fit calibration.csv --score-column perplexity --direction lower --correct-column correct --method isotonic --output replay/map.json
caat calibrate apply heldout.csv --map replay/map.json --output replay/heldout_calibrated.csv
```

Use `--raw-score-column COLUMN` on `apply` if the original score column
has a new name. If the batch already has a signed score, set
`--score-column` to use that column as supplied.

## Fit on partially scored data

Add `--review-missing-scores` to fit a standalone map on scored calibration
rows. Each row still needs a unique ID and binary grade. At least one row
must have a score. The map records `calibration_total`, `scored`, and the
score column name.

```bash
caat calibrate fit calibration.csv --score-column confidence --correct-column correct --method platt --review-missing-scores --output replay/map.json
caat calibrate apply new-scores.jsonl --map replay/map.json --output replay/probabilities.csv
```

`apply` reads the score column and missing-score rule from the saved map.
The CSV keeps every row and uses blank probability cells for missing
scores. `--raw-score-column` reads a renamed column with the same rule.

## Calibrate one tier from a combined export

Use `--prediction-filter COLUMN=VALUE` on `fit` and `apply` to select a
model tier or run. CAAT selects rows before validating their IDs,
scores, or grades. The fitted map records the calibration filter,
score column, and selected row counts.

```bash
caat calibrate fit mixed-calibration.csv --prediction-filter tier=encoder --id-column complaint_id --score-column confidence --correct-column correct --method platt --input-scale probability --output replay/tier-map.json
caat calibrate apply mixed-holdout.jsonl --prediction-filter tier=encoder --map replay/tier-map.json --output replay/encoder-probabilities.csv
```

Set the filter on the batch that `apply` reads. A new file already split
into one tier can use the saved map directly.

## Check for reused outputs in a holdout

Use `fit --track-fit-outputs` to save the ID and output key fingerprints
of the scored fitting rows. The default key column is `output_sha256`.
Use `apply --check-holdout` to compare each selected row against those
identities before publishing its probabilities.

```bash
caat calibrate fit calibration.csv --score-column confidence --correct-column correct --method platt --track-fit-outputs --output replay/map.json
caat calibrate apply heldout.csv --map replay/map.json --check-holdout --output replay/heldout_calibrated.csv
```

The check also works with a `policy` or `run` report that tracks its
design and calibration outputs. If identity headers differ in the new
file, set `--id-column` or `--output-key-column` on `apply`. A reused
output returns status 2 and preserves the previous CSV.
[Track fit outputs in policy reports](reports.md#track-outputs-used-to-choose-the-cutoff)

## Apply a map from a policy report

Pass the JSON report from `caat policy --calibrate` or `caat run` as
`--map`. The report supplies the fitted map, the score column, and any
saved orientation. The new batch only needs scores for this step.

```bash
caat calibrate apply new-scores.jsonl --map replay/policy.json --output replay/probabilities.csv
```

Use `--grader NAME` when the report contains maps for multiple graders.
Set `--raw-score-column` if the original score has a different column name in
the new batch. [Save a workflow report](workflow.md#use-in-automation)

If the report sends unscored outputs to review, application keeps those
rows in order and leaves their probability cells blank. Set
`--review-missing-scores` on `apply` to allow blanks for other saved maps.

## Read the report

- Brier score averages squared probability error over all rows.
- ECE averages the absolute gap between mean predicted probability and
  observed correctness in each fixed-width bin, weighted by row count.
- ACE averages those absolute gaps equally over nonempty adaptive bins.
  It assigns tied probabilities as one block using the block's midpoint
  rank. Large ties can leave bins empty.
- Reliability diagrams plot each occupied fixed-width bin at its mean
  probability and observed correctness. The diagonal marks perfect
  agreement. Bin counts appear in the report and on each SVG point.

`caat evaluate --probability --bins N` sets the number of bins for ECE and
ACE. It defaults to 10. Brier, ECE, and ACE all have lower-is-better scales.

[Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html) ·
[Adaptive calibration error](https://arxiv.org/abs/1904.01685) ·
[Archived calibration replay](../research/benchmarks.md#correctness-calibration)
