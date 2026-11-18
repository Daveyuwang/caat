# A fixed-budget audit on ANLI R2

The sample found enough errors to withhold the ANLI cutoff on this batch.
It took 100 labels to establish excess risk among 580 eligible outputs.
The separately opened census found 236 errors (40.69%). On the archived
BoolQ control, the same budget supported the 20% target.

| Batch | Eligible / total | Remaining for review at cutoff | Sample errors / labels | One-sided 95% lower limit | One-sided 95% upper limit | Sample verdict | Census |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ANLI R2 | 580 / 879 | 299 | 43 / 100 | 35.34% | 50.86% | Exceeds | 236 / 580 (40.69%) |
| BoolQ control | 2,666 / 2,938 | 272 | 12 / 100 | 7.16% | 18.57% | Meets | 321 / 2,666 (12.04%) |

Both verdicts describe their fixed eligible populations. The R2 boundary
was at most 13 observed errors to support the target or at least 27 to
establish excess. The protocol and cohort were frozen before inference.
The draw was recorded before opening its grades, and the verdict was recorded
before opening the remaining 480 eligible grades.
The development history records the protocol at `3beaea7`, the draw at `19fa4e1` and the verdict at `4e404b2`. [Frozen cohort](../../results/anli_r2/manifest.csv) · [Sampled outputs](../../results/anli_r2/sample.csv) · [Sample verdict](../../results/anli_r2/sample-result.json)

The RTX 5080 scored 879 original prompts in 55 batches (149,208 input
tokens, 37.66 seconds excluding model loading). Planning and auditing then
ran on CPU. The two-command 0.6 comparison used an explicit numeric cutoff
to create a decision table and a random queue. A single 0.7 planning call
reproduced its 580 eligible IDs and all 100 sample IDs, scores and output
keys while keeping the report withheld.
[Collection](../../results/anli_r2/collection.json) · [Sample plan](../../results/anli_r2/plan.json) · [Census grades](../../results/anli_r2/census.csv)

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

## Reproduce the handoff

Run the [two bundled CLI commands](../../README.md#audit-a-withheld-policy-on-a-new-batch)
for the R2 draw and verdict. To replay the manual comparison and BoolQ control
from a checkout, run the following command with the workflow extra installed.

```bash
python scripts/study_batch_audit.py replay
```

To reproduce collection, use the [0.6 collection environment](anli-results.md)
and execute the stages in order from the checkout root. `grade-sample` reads
grades for the frozen draw. `census` verifies the recorded sampled verdict
before requesting all eligible grades.

```bash
python scripts/prepare_anli.py results/anli_r2/manifest.csv --r2-exclude results/anli/manifest.csv
python scripts/collect_anli.py generate --manifest results/anli_r2/manifest.csv --output replay/anli-r2
python scripts/study_batch_audit.py prepare replay/anli-r2
python scripts/study_batch_audit.py grade-sample
python scripts/study_batch_audit.py census
```
