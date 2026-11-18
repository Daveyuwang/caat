# Choosing an audit budget for letter recognition

## Frozen protocol

Use David J. Slate's UCI Letter Recognition data in source order. The
SHA-256 of `letter-recognition.data` is
`2b89f3602cf768d3c8355267d2f13f2417809e101fc2b5ceee10db19a60de6e2`.
The first 12,000 rows train a StandardScaler and multinomial logistic
regression (L-BFGS, C=1, 1,000 iterations maximum). The next 4,000 rows
fit and check a CAAT policy using the largest class probability, rounded
to ten decimal places, and a 5% accepted error target. Use the default
design, calibration and holdout split, seed 42, and single-sided 95%
checks. The final 4,000 rows form the new batch. Use the source row number
as the ID and the predicted letter as its output key.
[UCI data and original split](https://archive.ics.uci.edu/dataset/59/letter+recognition)

Before grading the new batch, freeze its predictions, policy, eligible
population and seeded draw. At assumed error rates of 2% and 8%, compare
50, 100, 150 and 300 labels. Select the first budget with at least an 80%
chance of supporting the target in the first scenario and detecting excess
in the second. If none qualifies, record that outcome and run the 100-label
control. Keep the seed-42, 100-label draw as a comparator in either case.

Compare ten-at-a-time delivery with bulk at the chosen cap. Report the
break-even cost of an extra dispatch in units of one label under both
assumptions. Issue the blocks to one source-label grader. Check the returned
IDs and letters against the frozen draw. Complete the same draw after the
block verdict, run the 100-label control, then open the remaining eligible
grades for the census. Compare realized labels and requests to the forecasts
under both assumptions and under the observed census count.
