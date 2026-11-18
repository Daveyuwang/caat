# Replay a model cascade at matched cost

This replay keeps the highest-scored answers from Qwen2.5-7B-Instruct-Turbo
and uses the saved Qwen2.5-72B-Instruct-Turbo answer for each remaining
question. The two runs come from the same WildBench v2 test archive in
HELM Capabilities v1.15.0, suite v1.0.0. The join matches instance IDs,
checks the saved prompt for both models, and verifies each answer hash
and its archived grades. Each judge passes a response at score 7 or
above. GPT is the primary judge, with Llama and Claude as checks.
[Archive provenance](judge-study.md#archived-inputs) ·
[7B run](https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/runs/v1.0.0/wildbench%3Asubset%3Dv2%2Cmodel%3Dqwen_qwen2.5-7b-instruct-turbo/display_predictions.json) ·
[72B run](https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/runs/v1.0.0/wildbench%3Asubset%3Dv2%2Cmodel%3Dqwen_qwen2.5-72b-instruct-turbo/display_predictions.json)

## Protocol

Use the 7B model's mean token log probability, the primary gate score in
the judge study. Divide the matched records with CAAT's seed-42 split and
report the holdout only. Fix the 7B acceptance budgets at about 10%, 25%,
and 50% of holdout questions, rounding accepted counts up. Rank by the
saved score and average over ties at the budget boundary. Two baselines
answer all questions with 72B or accept a uniformly random set of 7B
answers at the same cascade budget. Compare on the same instances and
grades. Use 2,000 paired question bootstrap draws with seed 20261115 and
recompute the budget boundary in each draw.

Normalize the cost of one 72B response to one unit. At a 7B acceptance
share `c`, the cascade spends `s + (1-c)` units per question, with `s` the
ratio of 7B to 72B request costs. Set `s = c` to compare at equal spend in
each scenario. The costs model generation requests at fixed unit prices.
The replay uses both saved answers, so machine billing, token volumes,
and response latency would need a deployment measurement.
[FrugalGPT cascade study](https://arxiv.org/abs/2305.05176)

## Held-out comparison

The join retained 997 questions. The split contains 199 design, 399
calibration, and 399 held-out questions. The GPT judge counted 56 errors
in the 72B-only holdout and 116 with 7B alone. For the matched-cost
comparisons, positive differences indicate more errors with the cascade.

| 7B accepted | 72B fallback calls | 72B only | Random cascade | Score-ranked cascade | Difference from 72B, percentage points (95% interval) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40/399 | 359 | 14.04% | 15.54% | 16.79% | +2.76 (+1.00 to +4.51) |
| 100/399 | 299 | 14.04% | 17.80% | 20.55% | +6.52 (+4.26 to +9.02) |
| 200/399 | 199 | 14.04% | 21.57% | 25.06% | +11.03 (+7.77 to +14.29) |

Each cascade makes 399 calls to 7B. At the middle budget the GPT judge
counted 26 additional errors with ranked acceptance, compared with 72B
alone. Ranked acceptance added 2.75 percentage points of error over
random acceptance (95% interval +0.93 to +4.94). Llama and Claude also
recorded more errors than 72B alone at all three budgets.
[All grader comparisons](../../results/matched_cost/comparison.csv) ·
[Paired scores and split](../../results/matched_cost/paired.csv)

## Reproduce the comparison

From the checkout, run the script on the bundled paired table.
[Replay script](../../scripts/study_matched_cost.py)

```bash
python -m pip install .
python scripts/study_matched_cost.py
```

To regenerate the pairs, prepare the two HELM archives and use
`--prepare-from`. This matches the source inputs and saved responses
before the comparison. [Archive preparation](judge-study.md#reproduce-the-gate)

```bash
python scripts/prepare_helm_judges.py --scenario wildbench --release v1.15.0 --model qwen_qwen2.5-7b-instruct-turbo
python scripts/prepare_helm_judges.py --scenario wildbench --release v1.15.0 --model qwen_qwen2.5-72b-instruct-turbo
python scripts/study_matched_cost.py --prepare-from replay/helm_judges
```
