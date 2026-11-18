# Choosing an audit budget on ARC-Challenge

The saved ANLI policy selected 1,081 of 1,165 four-option ARC-Challenge test predictions from Qwen2.5-7B-Instruct. CAAT compared two plans under a ceiling of 500 labels. Both targeted 80% power at assumed batch error rates of 15% and 25%, around the 20% risk limit.

| Plan | First checkpoint | Label cap | Higher expected cost | Labels used | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smallest qualifying cap | 250 | 299 | 285.15 | 250 | 12 |
| Lowest expected cost | 200 | 325 | 259.43 | 200 | 11 |

Cost counts one unit per label and one per request. Reserving 26 more labels lowered the higher of the two scenario costs by 9.02%. Both plans reached `meets` in one request on the same seed-42 sampling order. A census later found 75 errors among the 1,081 eligible rows (6.94%). [Saved offers](../../results/arc_challenge/offers.json) · [Observed handoffs](../../results/arc_challenge/audit.json)

## Replay

```bash
python scripts/study_arc.py replay
```

The replay uses the saved scores and receipts. The original run scored A through D in source order and used a top-two logit gap of at least 4.75. Numeric answer keys were mapped to choice positions. [Collection settings](../../results/arc_challenge/collection.json) · [Study protocol](https://github.com/Daveyuwang/caat/blob/main/results/arc_challenge/protocol.md)
