# Frozen policies on BoolQ

The RTX 5080 scored 4,458 distinct passages with Qwen2.5-7B-Instruct and checked each proposed answer with Qwen3.5-2B. This includes 2,938 evaluation passages. The generator took 220.46 seconds for 793,340 input tokens. Verification added 56.81 seconds for 791,529 input tokens. Each stage used 279 forward batches of up to 16 requests. These are elapsed scoring times with the GPU and CPU offload, excluding loading. Model log probabilities were checked against full-vocabulary normalization for 224 rows per stage. All saved choices, scores and response hashes were independently reconstructed during the join. [Frozen protocol](boolq-protocol.md) · [Collection measurements](../../results/boolq/collection.json)

## Which inputs improved the ranking?

At 75% coverage (2,204 of 2,938), the combination of verifier probability and two native likelihood features had 7.17% accepted error. The design-selected single, the native likelihood gap, had 9.52%. The difference was −2.35 percentage points with a paired 95% interval of −3.14 to −1.50. This meets the prespecified two-point gain and the joint upper-bound criterion. P(True) alone had 15.91% error. The native gap was also the selected zero-extra-call single, so the two planned contrasts coincide. [Score comparison and primary audits](../../results/boolq/summary.json)

An exploratory check used the answer itself as a cheap feature. In the 400 design examples there were 9 errors among 199 Yes answers and 51 among 201 No answers. Adding an indicator for Yes to the two native likelihood features yielded 5.94% held-out error at the same coverage. The verifier combination had 1.23 points higher error, with a paired interval of +0.50 to +1.95. This check was added after the primary audit. The data support trying an answer feature before paying for a verifier on this task. Its advantage needs a fresh cohort. [Exploratory comparison](../../results/boolq/ablation.json)

## Where did calibration labels help?

| Native confidence grid | Calibration labels | Decision changes | Mean coverage | Fits without a cutoff |
| --- | ---: | ---: | ---: | ---: |
| Fixed values | 320 | 27.11% | 80.43% | 32/200 |
| Reference quantiles | 320 | 5.24% | 90.46% | 0/200 |
| Fixed values | 480 | 3.41% | 97.27% | 1/200 |
| Reference quantiles | 480 | 4.73% | 94.31% | 0/200 |

Reference quantiles passed the equal-cost stability gate with 320 labels. At 480 labels the numeric grid was both more stable and covered more rows. The 320-label reference grid lost 6.81 points of mean coverage against the 480-label numeric baseline (one-sided upper limit 7.67), exceeding the two-point margin. The corresponding upper limit for extra flips was 3.35 points. The study supports keeping both grid choices available. It does not establish a saving of 160 labels at equivalent performance.

On the primary 480-label calibration draw, the numeric policy accepted 2,835 answers with 425 errors (95% upper bound 16.14%). The reference-quantile native policy accepted 2,629 with 344 errors (14.22%). The fitted scorer's reference-quantile policy accepted 2,666 with 321 errors (13.13%). Each passed the frozen 20% target. The TOML workflow reproduced the fitted cutoff, counts and bound from the independent calculation. [Refits](../../results/boolq/refits.csv) · [Primary accepted IDs](../../results/boolq/accepted.csv)

## Reproduce

Run the CPU replay from the checkout. It uses the bundled score table, trains on the designated design rows and loads the fitted scorer into a temporary workflow for an independent cutoff check.

```bash
python -m pip install ".[scorer,workflow]"
python scripts/study_boolq.py
```

To regenerate the score table, use the pinned model revisions in the collector and a CUDA environment with transformers supporting Qwen3.5, PyTorch, Accelerate, PyArrow and huggingface-hub. The collection JSON records the tested versions. The first stage saves the proposed answer and its hash. The second reads that saved answer. The assembly step reads the pinned BoolQ grades.

```bash
python scripts/prepare_boolq.py replay/boolq/manifest.csv
python scripts/collect_boolq.py generate --manifest replay/boolq/manifest.csv
python scripts/collect_boolq.py verify --manifest replay/boolq/manifest.csv
python scripts/study_boolq.py --collect-dir replay/boolq --table replay/boolq/scores.csv
```
