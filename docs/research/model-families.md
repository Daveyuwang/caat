# Confidence scores across three model families

The benchmark compares confidence scores for Qwen2.5-7B-Instruct,
Phi-4-mini-Instruct, and SmolLM2-1.7B-Instruct on the same 1,000 TriviaQA
and 1,000 MMLU-Pro questions. It reuses the expanded Qwen run. The two
additional models collect one greedy answer and ten samples per prompt on
the RTX 5080. The question text, grading rules, token limits, and sampling
settings follow the expanded benchmark. [Benchmark protocol](black-box-benchmark.md#inputs-and-grading)

| Model | Checkpoint |
| --- | --- |
| Qwen2.5-7B-Instruct | [a09a354](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28) |
| Phi-4-mini-Instruct | [cfbefac](https://huggingface.co/microsoft/Phi-4-mini-instruct/tree/cfbefacb99257ffa30c83adab238a50856ac3083) |
| SmolLM2-1.7B-Instruct | [31b70e2](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674) |

## Comparison

Each task uses the same frozen assignment of 200 design, 400 calibration,
and 400 held-out questions for all three models. The whole-sample table
describes the fixed score rankings. For the held-out comparison, each
model and task chooses its score using normalized AURC gain on design
questions. The identity of the score stays fixed while the holdout measures
ranking at coverage budgets of 25%, 50%, and 75%. [Frozen splits](../../results/black_box_families/split.csv)

The primary comparison uses questions with all primary scores available
across the three models. Verbalized confidence and option-span log
probability use their own shared subsets and compare against mean token log
probability on the same questions. The CSV records available and matched
row counts for each score. A budget that cuts through tied scores uses
expected risk under uniform tie-breaking. Each interval pairs a score with
the mean token log probability baseline on the same questions. Intervals
use 2,000 question resamples and a per-comparison 95% level.

UQLM 0.6.8 and LM-Polygraph 0.7.0 score the same saved generations as CAAT.
Their results join by question ID and a hash of the greedy response and
ordered samples. The external scorers make zero additional generation
calls. [External score methods](black-box-benchmark.md#expanded-comparison)

## Held-out results at 75% coverage

Each primary comparison accepts 300 of 400 held-out answers. The score
chosen on design is fixed for each row in the table. Delta intervals are
in percentage points relative to mean token log probability.

| Task | Model | Design-selected score | Selected error | Mean logprob error | Delta and 95% interval |
| --- | --- | --- | ---: | ---: | ---: |
| TriviaQA | Qwen2.5-7B | Summed token logprob | 44.00% | 45.00% | −1.00 [−2.67, +0.67] |
| TriviaQA | Phi-4-mini | Summed token logprob | 50.00% | 51.00% | −1.00 [−3.00, 0.00] |
| TriviaQA | SmolLM2-1.7B | Summed token logprob | 78.33% | 82.33% | −4.00 [−6.00, −1.67] |
| MMLU-Pro | Qwen2.5-7B | Mean token logprob | 56.33% | 56.33% | 0.00 [0.00, 0.00] |
| MMLU-Pro | Phi-4-mini | Shorter response | 97.67% | 98.67% | −1.00 [−2.33, 0.00] |
| MMLU-Pro | SmolLM2-1.7B | Summed token logprob | 83.00% | 83.67% | −0.67 [−2.33, 0.00] |

Summed token log probability was selected on TriviaQA for every family
and lowered observed held-out error by 1 to 4 percentage points. The
SmolLM2 interval excludes zero. For its TriviaQA answers, UQLM exact match
also lowers error by 2.89 points [−5.05, −0.52], and semantic entropy lowers
it by 3.08 points [−5.00, −1.00]. LM-Polygraph ROUGE-L changes it by
−0.67 points [−3.00, +1.67]. MMLU-Pro selects a different score for each
family.

Verbalized confidence has 111 shared held-out rows on TriviaQA and 172
on MMLU-Pro. Option-span log probability has 13 shared held-out rows.
The CSV gives the baseline risk on those exact rows. [Complete comparison](../../results/black_box_families/comparison.csv)

## MMLU-Pro decoding check

The four-token greedy protocol asks for a single option letter. Phi-4-mini
often starts an explanation and meets that format on 33 of 1,000 questions.
The decoding check takes the highest next-token logit among the valid
option letters on the same prompt and grades the chosen letter against the
same key.

| Model | Strict-format rows | Greedy correct / 1,000 | Option likelihood correct / 1,000 | Greedy correct / 400 held out | Option likelihood correct / 400 held out |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7B | 999 | 410 | 410 | 153 | 153 |
| Phi-4-mini | 33 | 22 | 371 | 7 | 154 |
| SmolLM2-1.7B | 946 | 146 | 154 | 57 | 62 |

These counts describe the two decoding protocols on the same questions.
The confidence comparisons above rank the original greedy outputs.
[Option-likelihood runner](../../scripts/study_choice_decoding.py) ·
[Phi-4-mini option predictions](../../results/black_box_families/phi4_mini/choice_decode.csv)

## Collection and scoring cost

Each task and model used 13,000 prompt evaluations across greedy answers,
P(True), verbalized checks, and sampling. TriviaQA used 205 batched model
invocations and MMLU-Pro used 814. Timings are measured at the recorded
settings. The Qwen run offloaded some weights to CPU.

| Model | TriviaQA collection | MMLU-Pro collection | TriviaQA NLI directional pairs | TriviaQA NLI clustering |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-7B | 572.25 s | 914.69 s | 9,338 | 112.54 s |
| Phi-4-mini | 126.90 s | 374.35 s | 25,848 | 561.57 s |
| SmolLM2-1.7B | 85.28 s | 214.80 s | 27,722 | 465.64 s |

The NLI runs use the pinned DeBERTa checkpoint and 8, 4, and 6 CPU threads
respectively. They measure the extra scoring cost after generation. The
option-likelihood check used one next-token evaluation per question in
63 batches per model. The archived run metadata records versions, device,
batch sizes, calls, and timing by stage.
[Qwen run](../../results/black_box_families/qwen2_5_7b/run.json) ·
[Phi run](../../results/black_box_families/phi4_mini/run.json) ·
[SmolLM2 run](../../results/black_box_families/smollm2/run.json)

## Replay from saved scores

Install the checkout with `python -m pip install .` and run from the root.

```bash
python scripts/study_model_families.py results/black_box_families/qwen2_5_7b results/black_box_families/phi4_mini results/black_box_families/smollm2 --split results/black_box_families/split.csv --output replay/family-comparison.csv
```

The replay uses the archived score and grade tables and needs no model
weights. Match the output with the archived comparison.
[Comparison CSV](../../results/black_box_families/comparison.csv)
