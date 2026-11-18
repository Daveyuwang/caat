# Choosing an audit budget for letter recognition

The forecast selected **300 labels** from four candidate budgets before
grading any of the 4,000 new predictions. Of these, 1,458 qualified at
the saved cutoff of 0.8526697034. The independent policy holdout had
18 errors in 602 eligible rows. [Saved policy](../../results/letter_audit/policy.json)

| Labels | Chance of meeting the 5% target at 2% error | Chance of detecting excess at 8% error |
| ---: | ---: | ---: |
| 50 | 0.00% | 20.71% |
| 100 | 37.80% | 27.62% |
| 150 | 62.71% | 42.80% |
| 300 | 93.02% | 80.15% |

The 300-label draw had 12 errors. At most 9 would support the target and
at least 21 would establish excess. The verdict was **inconclusive** after
30 requests of ten labels each, matching the bulk verdict. The 100-label
control had two errors and was also inconclusive. The subsequent census
found 47 errors among 1,458 eligible rows (3.22%). At this error count,
51.12% of uniform 300-label draws give an inconclusive verdict.
[Frozen design](../../results/letter_audit/design.json) ·
[Sample outcomes](../../results/letter_audit/audit.json) ·
[Census](../../results/letter_audit/census.json)

| 300-label scenario | Expected labels in blocks of ten | Expected requests | Cost of one extra request at break-even, in labels |
| --- | ---: | ---: | ---: |
| 2% assumed before grading | 299.66 | 29.97 | 0.012 |
| 8% assumed before grading | 256.98 | 25.70 | 1.74 |
| Observed census | 298.61 | 29.86 | 0.048 |

Bulk delivery takes 300 labels and one request. If an extra dispatch
costs two labels, bulk costs less under both original assumptions. At
one label per dispatch, bulk costs less under the 2% scenario and blocks
cost less under the 8% scenario. The observed draw used the full budget.

Run the base-package replay from an installed checkout. It fetches the
pinned data, checks both seeded sample identities and compares the budget
grid, issued grades, full verdicts and eligible census to the saved results.

```bash
python scripts/study_letter_audit.py replay
```

To repeat fitting, use scikit-learn 1.9.1 and run the stages `freeze`,
`audit` and `census` in that order. The protocol was recorded at
`4348e08`, the predictions and budget at `1b61ee2` and the sample
verdicts at `88d5959` before the census.

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
