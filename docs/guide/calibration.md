# Calibrate correctness probabilities

Fit a map using labeled calibration rows. Apply the saved JSON map to a
separate CSV and evaluate its probabilities on held-out labels.

```bash
caat calibrate fit calibration.csv --score-column confidence --correct-column correct --method isotonic --output replay/map.json
caat calibrate apply heldout.csv --score-column confidence --map replay/map.json --output replay/heldout_calibrated.csv
caat evaluate replay/heldout_calibrated.csv --score-column calibrated_probability --correct-column correct --threshold 0.75 --probability --html replay/calibration.html
```

The input to `fit` has a unique ID, score, and binary correctness for each
row. Use `--id-column` for a different ID header. The `apply` command appends
`calibrated_probability` to the input columns. Set `--probability-column`
to choose a different name.

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
