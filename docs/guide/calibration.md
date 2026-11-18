# Calibrate a confidence score

Fit a mapping from score to probability of correctness using labeled calibration rows. Apply it to separate predictions and evaluate those probabilities against the same grader.

```bash
caat calibrate fit calibration.csv --score-column confidence --correct-column correct --method isotonic --output replay/map.json
caat calibrate apply heldout.csv --score-column confidence --map replay/map.json --output replay/heldout_calibrated.csv
caat evaluate replay/heldout_calibrated.csv --score-column calibrated_probability --correct-column correct --threshold 0.75 --probability --html replay/calibration.html
```

| Setting | Use |
| --- | --- |
| `--method platt` | Fit a logistic map. Set `--input-scale probability` for probability inputs. |
| `--method isotonic` | Fit a nondecreasing step function. |
| `--direction lower` | Orient a score such as perplexity so lower values mean greater confidence. |
| `--grader-file NAME=PATH` | Fit from a separate grader export keyed to the predictions. |
| `--track-fit-outputs` | Record fit identities. Use `--check-holdout` on apply to detect reused outputs. |

Read the Brier score and reliability bins on the held-out evaluation. Each occupied bin compares the mean calibrated probability with the fraction correct. Decisions can use the original score and cutoff. [Reliability diagram](../figures/calibration-example.svg) · [Report fields](reports.md#evaluate)
