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

On the original split, each model/task cell used 400 held-out answers. The counts below sum over the three model families. Every qualifying cutoff also met the one-sided 95% holdout check at the 40% target.

| Score | Cutoff grid | TriviaQA accepted / errors | MMLU-Pro accepted / errors |
| --- | --- | ---: | ---: |
| Single | Ten quantiles | 302 / 75 | 63 / 12 |
| Fitted | Ten quantiles | 299 / 72 | 63 / 12 |
| Single | Shortlist | 0 / 0 | 0 / 0 |
| Fitted | Shortlist | 0 / 0 | 0 / 0 |

## What moved the cutoff

On the Inspect case, varying design subsamples with calibration fixed changed 2.4% of decisions for the ten-quantile grid. Varying calibration subsamples with design fixed changed 12.1%. The corresponding IFEval rates were 4.6% and 34.8%. Both checks sampled 80% of each original partition. A reduced shortlist can remove every candidate that calibration has enough observations to certify.

As a follow-up, we tried ten fixed thresholds, 0.05 through 0.95 in steps of 0.10 on the probability scale. For IFEval we took their logarithms to match the log score. Both raw scorers were already fixed, so each fit could use its entire label budget for calibration. On the same 80% subsamples and evaluation IDs, decision changes fell from 14.3% to 6.4% on Inspect and from 36.2% to 17.2% on IFEval. Mean coverage rose from 26.3% to 35.6% and from 65.6% to 87.6%. These are exploratory results on the two cases that motivated the follow-up. A user with a prespecified score scale can use this approach with `caat select --thresholds` and a separate holdout. A learned score still needs separate design data. [Diagnostic results](../../results/routing_stability/diagnosis.csv)

## ARC-Challenge results and replay

The RTX 5080 scored 9,163 option likelihoods for 1,119 training and 1,172 test questions in 219.5 seconds of lm-eval evaluation time. The run used lm-eval 0.4.13, BF16, batch size 16 and a 12 GiB GPU weight cap. CAAT reconstructed the native raw-likelihood winners and grades. Six training prompts also occurred in the test set and one further training prompt repeated. Removing those rows before fitting left 370 design and 742 calibration questions. The complete test set contains 1,170 distinct prompt hashes. The paired intervals resample question IDs.

At the locked 20% target, every original-split cell sent all 1,172 answers to review. At 75% coverage the design-selected single margin accepted 879 answers at 40.16% error. The fitted score had 41.87% error at the same budget. The paired difference was +1.71 percentage points (95% interval −0.16 to +3.33). The scorer ships as an opt-in choice. [ARC audits](../../results/routing_stability/arc_audits.csv) · [Refits](../../results/routing_stability/arc_stability.csv) · [Score comparison](../../results/routing_stability/arc_quality.csv)

Replay the verified feature table on CPU. The diagnosis script also verifies 232 candidate decisions against independent SciPy binomial tails and MAPIE's Bonferroni correction.

```bash
python scripts/study_routing_stability.py --arc results/routing_stability/arc_scores.csv --output replay/arc04/replay
python -m pip install mapie==1.6.0
python scripts/study_routing_diagnosis.py
```

To regenerate the input with lm-eval 0.4.13 and its Hugging Face backend, run the pinned tasks below. Supply the resulting training and test sample log paths to the preparation script. It checks the native answer keys, prompt hashes, option strings and question IDs before assigning splits. [Prepared feature table](../../results/routing_stability/arc_scores.csv)

```bash
lm-eval run --model hf --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,revision=a09a35458c702b33eeacc393d103063234e8bc28,dtype=bfloat16,parallelize=True,max_memory_per_gpu=12GiB,max_cpu_memory=40GiB --include_path scripts/arc_challenge --tasks caat_arc_train,caat_arc_test --num_fewshot 0 --batch_size 16 --device cuda --log_samples --output_path replay/arc04/logs
python scripts/prepare_arc_routing.py TRAIN_SAMPLES.jsonl TEST_SAMPLES.jsonl --output replay/arc04/arc_scores.csv
```
