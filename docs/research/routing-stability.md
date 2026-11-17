# Routing stability and fitted confidence scores

A refit changes a decision when one cutoff accepts an answer and the other sends the same answer to review. This replay measures that fraction on fixed evaluation IDs. Every fit uses disjoint design and calibration rows. The held-out labels enter only the audit.

## Protocol

Use the saved three-family TriviaQA and MMLU-Pro partitions. At a fixed collection budget of one primary generation, compare mean token log probability, summed token log probability and negative token count. Select the single signal by design normalized AURC gain. The fitted score uses the same three features with design standardization and L2 logistic regression at C = 0.1. A one-class design yields a constant score. No additional generation calls are needed. Use a 40% error target for these archives and the two motivating cases. Accept zero rows when no cutoff qualifies.

The reference grid uses ten evenly spaced design ranks. The size control uses three. The shortlist uses 25 seeded subsamples without replacement, each containing 80% of design rows. For each subsample, take the largest *actual* accepted fraction among its ten ranks whose empirical error meets the target, including all score ties. Record zero when none qualifies. Convert the interpolated 25th, 50th and 75th percentiles of those fractions to ceiling ranks in the full design scores, dropping zeros and duplicate thresholds. Bonferroni selection tests the frozen candidates at a 5% family-wise failure rate.

Use the archived design/calibration assignment for the primary held-out audit. For stability, shuffle the sorted design and calibration IDs together using seeds 0 through 199. Compare decisions from seed pairs (0, 1), (2, 3) and so on. The sample-budget check takes 80% of each shuffled partition. The holdout is fixed for both checks. For the two motivating cases use the CAAT seed-42 holdout. The Inspect fixture contains the 1,000 Qwen3.5-4B TriviaQA outputs scored with Qwen3.5-2B P(True) and the alias grader. [Inspect workflow](../guide/inspect.md)

## ARC-Challenge confirmation

The protocol below was set before reading ARC test outcomes.

Freeze a 20% error target, the same three cutoff procedures, C = 0.1, and two score columns, raw winning-option log likelihood and its margin over the runner-up. Collect zero-shot lm-eval samples from Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`, using the ARC-Challenge prompt, native raw `acc` grader and dataset revision `210d026faf9955653af8916fad021475a3f00453`. Use one third of the shuffled official training questions for design and the remainder for calibration (seed 42). Keep the official test split intact and check question IDs and prompt hashes before auditing. All methods share the same option likelihoods. [ARC data](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453/ARC-Challenge)
## Archive results

| Case | Grid | Decisions changed | Mean coverage | Refits without a cutoff |
| --- | --- | ---: | ---: | ---: |
| Inspect TriviaQA | Ten quantiles | 8.0% | 32.1% | 4/200 |
| Inspect TriviaQA | Shortlist | 21.4% | 15.3% | 128/200 |
| IFEval | Ten quantiles | 22.3% | 81.4% | 6/200 |
| IFEval | Shortlist | 12.1% | 91.7% | 12/200 |

The shortlist frequently proposed cutoffs that calibration could not certify on Inspect. The ten-quantile grid remains the default. At 75% coverage, fitting three token features changed equal-weight mean accepted error by −0.11 percentage points on TriviaQA (paired 95% interval −0.56 to +0.33) and 0.00 on MMLU-Pro (−0.22 to 0.00). Each cell reused the same saved primary generation and added zero model calls. [All routing cells](../../results/routing_stability/stability.csv) · [Frozen policy audits and fit times](../../results/routing_stability/audits.csv) · [Matched subsamples](../../results/routing_stability/stability_80pct.csv) · [Paired score comparison](../../results/routing_stability/quality.csv)

The Inspect fixture uses the saved sample IDs, scores and alias grades from Inspect run `YipEsdeYRYzRQR8cXG3yVn`. It sampled 1,000 questions with seed 20261116 from TriviaQA Web validation, generated with Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, and scored the logged question and completion with Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. The confidence is the normalized next-token probability of True among True and False. [Replay input](../../results/routing_stability/inspect_scores.csv) · [Dataset revision](https://huggingface.co/datasets/mandarjoshi/trivia_qa/tree/0f7faf33a3908546c6fd5b73a660e0f8ff173c2f/rc.web.nocontext)

From a checkout with Python 3.12, run the following commands. Each case saves the ordered holdout IDs, design and calibration IDs for each fit, and a decision bit per holdout ID (1 means accept). Seed −1 denotes the original assignment.

```bash
python -m pip install . scikit-learn==1.7.2 numpy scipy
python scripts/study_routing_stability.py
python scripts/study_routing_stability.py --fit-fraction 0.8 --output replay/routing-subsamples
```
