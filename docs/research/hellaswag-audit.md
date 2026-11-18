# Two-stage audit on HellaSwag

The saved ANLI cutoff selected 1,276 of 1,500 Qwen2.5-7B-Instruct
predictions. At a common cap of 321 labels, two checkpoints reduced the
higher expected cost across the two design scenarios by **8.00%**, from
318.37 to 292.90 label-equivalents. The optimized CAAT 1.1 baseline issued
blocks of 314. Each request cost one label-equivalent in this comparison.
The two-stage workflow from milestone 1.2 ships in the 1.1 series supplement.
[Frozen design](../../results/hellaswag/design.json) · [Installation](https://github.com/Daveyuwang/caat/releases/tag/v1.1.0)

| Seed 42, same ordered draw | Labels issued | Requests | Errors returned | Verdict |
| --- | ---: | ---: | ---: | --- |
| Bulk | 321 | 1 | 46 | Meets |
| Best fixed block at the design price | 314 | 1 | 46 | Meets |
| Two checkpoints | 321 | 2 | 46 | Meets |

After 125 receipts, the grader restarted with the same request ID and
the other 125 rows outstanding. The first 250 labels contained 40 errors.
The checkpoint required at most
36 to establish that the batch met the 20% target, or at least 64 to
establish excess. The grader therefore received the remaining 71 rows.
At 321 labels, 46 errors cleared the final boundary of 53. The fixed-block
audit could settle after 314 because even seven further errors would leave
it at 53. The saved policy still withholds routing.
[Issued outcomes](../../results/hellaswag/audit.json) · [Shared draw](../../results/hellaswag/sample.csv)

| Assumed eligible error | Correct verdict probability, fixed / two-stage | Expected labels, fixed / two-stage | Expected requests, fixed / two-stage |
| --- | ---: | ---: | ---: |
| 15% (192 errors) | 82.63% / 82.81% | 316.95 / 291.32 | 1.42 / 1.58 |
| 25% (319 errors) | 80.37% / 80.55% | 315.59 / 290.32 | 1.23 / 1.57 |

Both designs cleared the 80% power requirement. An independent calculation
gave 3.91% false passes at 256 population errors and 4.99% false excess
verdicts at 255, within the two separate 5% limits. Keeping the chosen
checkpoints and reoptimizing the fixed-block baseline, expected cost fell
6.81% with free requests and 7.61% at five label-equivalents per request.

The subsequent census found **150 errors in 1,276 eligible rows (11.76%)**.
At that error count, a new uniform draw would average 254.42 labels and
1.06 requests at the frozen checkpoints, versus 314.20 and 1.03 for the
frozen fixed block. The chance of needing the second checkpoint was 6.22%.
[Eligible census](../../results/hellaswag/census.csv)

## Frozen protocol

Use Qwen2.5-7B-Instruct at `a09a35458c702b33eeacc393d103063234e8bc28`
in BF16. Take at most 1,500 validation sources in SHA-256 order, choosing
one row per source by its hashed source, example index and file position.
Keep the four endings in source order and score the single A/B/C/D tokens
in one forward pass. Apply the saved ANLI candidate gap cutoff of 4.75
to the difference between the two highest logits. Keep its 20% error
target and both 95% one-sided limits. Pin the dataset to
`218ec52e09a7e7462a5400043bb9a69a41d06b76`. Collection reads only IDs,
contexts and endings. [HellaSwag dataset](https://huggingface.co/datasets/Rowan/hellaswag/tree/218ec52e09a7e7462a5400043bb9a69a41d06b76)

Before opening validation answers, choose the smallest single-stage cap
up to 500 that reaches 80% directional power at both 15% and 25% error.
Compare interim counts from 50 in steps of 25 below the cap, plus the
integer part of two thirds of the cap. Keep the qualifying checkpoint with
the lowest worst-scenario cost at one label per request, breaking ties by
the smaller interim count. If there is no qualifying design, record that
outcome. Freeze the eligible cohort and the shared seed-42 draw. Compare
bulk, the best constant 1.1 block size and the two-stage rule at this cap.
For all three record issued labels, requests and verdict. Open the eligible
census last. Also report costs at zero and five labels per request. The
milestone threshold is a 5% reduction in worst-scenario expected cost
against the best 1.1 block size, with the same power requirement and cap.

For the directional check, condition on the first 250 labels' error count
and sum over the cases that continue (37 through 63). The 71 remaining
draws follow the hypergeometric law on the remaining 1,026 rows. Add the
probability of crossing the same direction at the first checkpoint.

At the census error count, the paired two-stage draw costs less than the
314-row block on 93.78% of draws, ties on 2.13%, and costs more on 4.09%
at the design price. For the last case the first 250 must contain 37 to
63 errors and the fixed block must settle at 314. Condition on the first
count, then draw the next 64 from the remaining 1,026. The fixed block
settles at cumulative counts of 46 or fewer, 54 to 67, or 75 or more.

The first pass used 378,774 input tokens. While the desktop was in use,
176 rows ran on CPU. Rescoring those rows in their original batches on
the RTX 5080 changed two choices and two cutoff decisions. All published
logits come from the GPU. The cohort was committed before collection, the
design before grading, and the issued outcomes before the census.
[Collection record](../../results/hellaswag/collection.json) ·
[Cohort freeze](https://github.com/Daveyuwang/caat/commit/5ef24cf0cbc24e83a6869ee318badd8b8cd6a56f) ·
[Design freeze](https://github.com/Daveyuwang/caat/commit/17701c5add6cb80820bf2fbdda2b43ea865bd4a3) ·
[Audit before census](https://github.com/Daveyuwang/caat/commit/215563b8685d5f22a989b5df9ca1530878db26b6)

## Reproduce the audit

From a checkout, the base install replays the saved logits and reference
grades on CPU. It redraws the sample, restarts a partially returned first
request and compares the three verdicts.

```bash
python -m pip install .
python scripts/study_hellaswag.py replay
```

To repeat collection, install the models extra, PyArrow and a CUDA build
of PyTorch. Run `scripts/collect_hellaswag.py freeze` and then `collect`.
The collector streams the pinned BF16 weights from disk, caps PyTorch's
GPU allocator at 2.25 GiB and saves progress after each batch of eight.
Repeating `collect` resumes from the last completed batch. Use `--device cpu`
to continue with two CPU threads. The collection report records each device
and its last row in scoring order. Use `--stop-after` to repeat a segment.
After a mixed-device run, use `scripts/collect_hellaswag.py confirm-gpu`
to reproduce the single-device predictions used here.
Run `scripts/study_hellaswag.py` with `design`, `audit` and `census` in
that order. The dataset supplies the reference ending for each issued row.
