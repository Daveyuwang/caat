# CAAT roadmap

## 1.3 milestone proposal

A label lead should be able to enter a hard budget and see two executable
offers. One minimizes the qualifying cap in the search. The other
minimizes expected cost under that ceiling. The two-stage HellaSwag study needed a custom
Python loop to choose its cap and first checkpoint. CAAT already knows how
to grade either design. [1.3 milestone](https://github.com/Daveyuwang/caat/milestone/2) · [Current selection loop](../scripts/study_hellaswag.py)

### What changes when the cap can vary?

The pilot used only eligible population sizes to select plans. It priced one
request as one label, required 80% directional power in both design scenarios,
and bounded each false direction at 5%. Cost is the larger expected number
of labels plus requests in the two scenarios.

| Archived population | Prior checkpoints | Lowest-cost checkpoints tested under 500 | Expected cost, prior → candidate | Extra labels to reserve |
| --- | ---: | ---: | ---: | ---: |
| HellaSwag | 250 / 321 | 200 / 350 | 292.90 → 283.16 (3.33% lower) | 29 |
| UCI Letters | 200 / 300 | 175 / 325 | 268.30 → 261.46 (2.55% lower) | 25 |
| WinoGrande | 175 / 261 | 175 / 275 | 237.69 → 225.94 (4.94% lower) | 14 |

To reproduce the grid, try final caps from 200 through 500 in steps of
25 plus the prior cap. For each, try first checkpoints from 50 in steps
of 25 below the cap, plus two thirds of the cap rounded down. Use the
five existing error allocations and the study's original target and
scenarios (5% with 2%/8% for Letters, 20% with 15%/25% for the other
two). A SciPy conditional hypergeometric calculation selected the grid
winners; CAAT's integer calculation reproduced their boundaries and costs.
These are retrospective design comparisons. [HellaSwag design](research/hellaswag-audit.md) · [Letter study](research/letter-audit.md)

The larger cap can cost more near the target. At 255 errors in HellaSwag's
1,276 eligible outputs, the 350-label design expects 343.59 labels and the
321-label design expects 319.79. Minimum cap and minimum expected sample
size are distinct design objectives. NIST gives the average sample number
calculation. The R package AcceptanceSampling evaluates double-sampling
plans and searches single-sample plans. CAAT's contribution is to take a
chosen plan through the same policy, draw and resumable grading journal.
[NIST double sampling](https://itl.nist.gov/div898/handbook/pmc/section2/pmc24.htm) · [AcceptanceSampling manual](https://cran.r-project.org/web/packages/AcceptanceSampling/refman/AcceptanceSampling.html)

### Delivery in three PRs

1. **Compare feasible plans in one preview.** Give `caat plan` a ceiling,
   two scenario rates, target power and a request price. Compare fixed
   blocks and two checkpoints on a documented finite grid. Include the
   original design among the candidates. Return the smallest tested cap
   and the lowest higher-scenario cost, with the extra reserve, power,
   expected requests and near-target workload for each. For equal costs
   prefer the smaller cap, then fewer requests. Report infeasibility with
   the searched range. Test on a small population by exhaustive enumeration
   and reproduce the three table rows within 0.01 cost units.
2. **Issue the plan the owner chose.** Let the owner select either offer
   and freeze its mode, blocks or checkpoints, thresholds, batch and draw
   through the existing handoff. Preview and execution must agree even
   after a partial receipt and retry. Replay both a fixed-block winner
   and a two-stage winner, including continued use of saved 1.1 plans.
3. **Run a fresh ARC-Challenge audit.** Use the 1,165 four-option questions
   among the 1,172 distinct test IDs at the pinned revision. Map source
   option labels by position, including the 21 numeric-label questions.
   Freeze Qwen2.5-7B-Instruct, its prompt, option order, gap cutoff 4.75 and
   the eligible population before reading answers. Reuse the collector's
   2.25 GiB GPU allocator limit. Use seed 42, ceiling
   500, a 20% target, 15%/25% scenarios, 80% power and one label per request.
   Freeze the offers and compare them on prefixes of one random ordering
   with the original minimum-single-cap workflow and its best fixed block.
   Record actual labels, requests and verdicts before opening the eligible
   census. A cost-improvement claim requires at least 2% lower modeled cost
   than the original two-stage workflow at the common ceiling, both power
   constraints and separate reporting of each plan's maximum. Publish a tie,
   loss or infeasible budget as observed. The software acceptance criterion
   is reproduction from the base wheel of the selected plan and its receipts.
   [AI2 ARC](https://allenai.org/data/arc) · [Pinned test data](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453/ARC-Challenge)

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
[HellaSwag study](research/hellaswag-audit.md) · [Completed milestone](https://github.com/Daveyuwang/caat/milestone/1) · [Install the 1.1 supplement](https://github.com/Daveyuwang/caat/releases/tag/v1.1.0)

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
