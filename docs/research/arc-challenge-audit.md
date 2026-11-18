# Choosing an ARC-Challenge audit under a 500-label ceiling

Reserving 26 more labels lowered the higher of two expected grading costs by **9.02%**, from 285.15 to 259.43 label-equivalents. Both offers cleared 80% power in each design scenario. On the frozen draw the lower-cost offer stopped after 200 labels, 50 fewer than the prior two-stage workflow. [Saved offers](../../results/arc_challenge/offers.json) · [Observed handoffs](../../results/arc_challenge/audit.json)

| Design | First request / cap | Power at 15% / 25% error | Higher expected cost | Actual labels / requests | Sample errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smallest tested cap | 250 / 299 | 80.05% / 80.36% | 285.15 | 250 / 1 | 12 |
| Lowest expected cost | 200 / 325 | 82.19% / 80.38% | 259.43 | 200 / 1 | 11 |
| Prior two-stage workflow | 250 / 299 | 80.05% / 80.36% | 285.15 | 250 / 1 | 12 |
| Best fixed block at the prior cap | 292 / 299 | 80.00% / 80.36% | 296.71 | 292 / 1 | 14 |

The planning scenarios assume 163 or 271 wrong predictions among 1,081 eligible outputs. Power is the chance of concluding `meets` in the lower scenario or `exceeds` in the higher one. Cost counts one label per grade and one per request. At 216 population errors, just inside the 20% target, the larger cap costs more in expected labels, 317.22 versus 298.67. The 9.02% reduction applies to the two prespecified scenarios. [Forecasts and boundaries](../../results/arc_challenge/offers.json) · [Prior workflow](../../results/arc_challenge/design.json)

## Collection and grading

We kept all 1,165 four-option questions among the 1,172 distinct ARC-Challenge test IDs. Qwen2.5-7B-Instruct scored A through D in source order using BF16 next-token logits. The saved ANLI R2 policy selected the 1,081 predictions with a top-two gap of at least 4.75. One RTX 5080 processed 73 batches in 41 seconds of scoring time using the 12 GiB GPU placement setting. Model loading is outside that timing. The grader mapped each gold label to its position in the original choices, including the 21 rows with numeric labels. [Pinned dataset](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453/ARC-Challenge) · [Collection metadata](../../results/arc_challenge/collection.json)

The cohort and prompt were committed before inference. The four audit arms used prefixes of one seed-42 order. We saved the offers and draw before opening answers, read the requested positions for each arm, submitted a partial receipt, restarted the handoff and retried its completed receipt. All four arms reached `meets` in one request each. The subsequent census found 75 errors among the 1,081 eligible predictions (6.94%). The batch verdict leaves the saved policy withheld. [Frozen protocol](https://github.com/Daveyuwang/caat/blob/e824894/results/arc_challenge/protocol.md) · [Frozen offers and draw](https://github.com/Daveyuwang/caat/commit/b71540b) · [Receipts before census](https://github.com/Daveyuwang/caat/commit/05d1330) · [Census](../../results/arc_challenge/census.csv)

## Replay

From an installed checkout, run the base package on CPU.

```bash
python scripts/study_arc.py replay
```

The replay checks the predictions against the saved logits, regenerates both offers and the prior designs, verifies the paired sample, and reproduces each request count and verdict from the saved receipts. It then checks the eligible census against the scored rows.
