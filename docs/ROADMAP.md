# CAAT roadmap

## Budget comparison (milestone 1.3, released in 1.1.2)

CAAT compares audit plans under one label budget and issues the plan you choose. On ARC-Challenge, raising the cap from 299 to 325 reduced the higher of two expected grading costs by 9.02%. The frozen draw used 200 labels instead of 250. [Budget comparison](guide/reports.md#plan) · [ARC-Challenge audit](research/arc-challenge-audit.md) · [Milestone](https://github.com/Daveyuwang/caat/milestone/2)

The retrospective design checks used the same 500-label ceiling and one label per request.

| Archived population | Prior checkpoints | Lowest-cost checkpoints tested | Higher expected cost, prior → selected |
| --- | ---: | ---: | ---: |
| HellaSwag | 250 / 321 | 200 / 350 | 292.90 → 283.16 |
| UCI Letters | 200 / 300 | 175 / 325 | 268.30 → 261.46 |
| WinoGrande | 175 / 261 | 175 / 275 | 237.69 → 225.94 |

Score-stratified auditing is the next alternative. The two equal-sized
score halves on HellaSwag contain 125 and 25 errors, compared with 122 and
114 on ANLI R2. Before adding strata, demonstrate lower decision cost under
the same directional guarantees. Arbitrary-time stopping is a separate
option for teams whose labels arrive continuously. [Stratified prediction-powered inference](https://arxiv.org/abs/2406.04291) · [Finite-population confidence sequences](https://arxiv.org/abs/2006.04347)

## 1.1 series

### Two-stage audit supplement

Plan a two-stage audit on an unlabeled batch, dispatch work at each checkpoint
and resume a partial receipt. On 1,500 HellaSwag predictions, the planner's
higher expected cost across the two prespecified scenarios fell 8.00% versus
the best 1.1 fixed block. The observed draw took 321 labels versus 314.
[HellaSwag study](research/hellaswag-audit.md) · [Completed milestone](https://github.com/Daveyuwang/caat/milestone/1) · [Original supplement](https://github.com/Daveyuwang/caat/releases/tag/v1.1.0)

### 1.1.0

Preview verdict probabilities and grading costs on an unlabeled batch,
then draw the audit sample at the chosen budget. The UCI letter study
selected 300 labels and used all 30 blocks before reaching an inconclusive
verdict. [Budget forecast](guide/reports.md#plan) · [Letter audit](research/letter-audit.md)

## 1.0.0

Take a saved batch from scoring through a routing decision, then use the same policy to audit the next labeled batch. When a cutoff misses its target, a labeler can still draw a fixed-budget audit sample and stop once every possible completion gives the same verdict. [Policy walkthrough](../README.md#reuse-one-fitted-policy-from-routing-through-audit) · [Audit handoff](guide/queue.md#issue-audit-labels-in-blocks)

[1.x compatibility](../CONTRIBUTING.md#public-interface) · [Release notes](../CHANGELOG.md)

## After 1.0

A priority for research is cutoff stability. A class-aware score changed accepted error by 0.58 percentage points on ANLI R1 at 75% coverage, and the interval included zero. A follow-up should compare scores at matched inference cost and test the selected cutoff on a fresh batch. [ANLI results](research/anli-results.md)

For integrations, start with a real prediction table or evaluation log and the decision its owner needs to make. [Contributor guide](../CONTRIBUTING.md) · [Plugin contracts](guide/plugins.md)
