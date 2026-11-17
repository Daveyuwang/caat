# A fixed-budget audit on ANLI R2

## Protocol

Use Qwen2.5-7B-Instruct at revision `a09a35458c702b33eeacc393d103063234e8bc28`
with the 0.6 ANLI prompt, token mapping, BF16 settings, and top-two logit gap.
Keep the cutoff at 4.75, the risk target at 20%, and each one-sided confidence
level at 95%. The original calibration and R1 holdout produced a withheld policy.
[Original protocol](anli-protocol.md) · [R1 results](anli-results.md)

Use test R2 at dataset revision `8e4813d81f46d313dac7892e1c28076917cfcdf9`.
Exclude every normalized premise in the 0.6 manifest. Keep one hypothesis per
remaining premise using the original hash and UID tie break. Freeze the IDs
before inference. Score the three original option tokens once per row.
[ANLI source](https://huggingface.co/datasets/facebook/anli/tree/8e4813d81f46d313dac7892e1c28076917cfcdf9)

Draw up to 100 eligible outputs with seed 42, sorted by ID before shuffling.
Use the dataset's relation label as the grader, keyed to ID and the saved
semantic answer. Record the sample and decision boundaries before opening
the sampled grades. Record the sampled verdict before opening the remaining
eligible grades. Compare the result with the census, and reproduce the draw
by composing the 0.6 scoring, numeric cutoff and random-queue commands.
Run the same 100-label check on the published BoolQ fitted policy as a
known passing census. Each sampled verdict stands even if it is inconclusive.
