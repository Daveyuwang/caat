# Benchmarks

## Replay the bundled samples

From an installed checkout, run the three 24-row examples. Results go to `replay/benchmarks/summary.csv`, with a risk–coverage curve and a metrics report for each dataset.

```bash
python scripts/replay_benchmarks.py
```

| Sample | Score | Cutoff | Accepted | Accepted errors |
| --- | --- | ---: | ---: | ---: |
| [IFEval](../../examples/ifeval_strict.csv) | Mean token log probability | −0.35 | 12 / 24 | 4 |
| [MMLU-Pro](../../examples/mmlu_pro.csv) | Mean token log probability | −0.30 | 12 / 24 | 5 |
| [TriviaQA](../../examples/triviaqa.csv) | Archived response score | 0.50 | 13 / 24 | 0 |

The IFEval and MMLU-Pro examples use Qwen2.5-7B-Instruct-Turbo responses from HELM Capabilities v1.0.0. IFEval requires every instruction to pass. MMLU-Pro grades the extracted choice against the answer key. TriviaQA uses Xia’s Llama 8B DPO predictions on the Web validation split and normalized exact match against the answer aliases. [Datasets, model runs and graders](https://github.com/Daveyuwang/caat/blob/v1.1.2/docs/research/benchmarks.md#sources-and-grading)

## Policy selection

On 217 held-out IFEval responses, one cutoff accepted 144. Google’s strict grader counted 33 errors (22.9%) and HELM counted 45 (31.2%) on those same predictions. A 30% target falls between the two estimates. [Run the example](../../README.md#try-a-review-policy) · [Method](method.md)

## Batch audits

Each audit samples predictions at or above a saved cutoff. The verdict concerns the eligible population in that batch.

| Batch | Eligible predictions | Labels used | Result |
| --- | ---: | ---: | --- |
| [ANLI R2](../../results/anli_r2/sample-result.json) | 580 | 100 | 43 errors. Exceeds the 20% target. |
| [WinoGrande](../../results/winogrande/stop.json) | 667 | 90 of 100 | The minimum possible final error count reached 28. The boundary for excess was 27. |
| [UCI Letters](../../results/letter_audit/audit.json) | 1,458 | 300 | 12 errors. Inconclusive at the 5% target. |
| [HellaSwag](../../results/hellaswag/audit.json) | 1,276 | 321 | Two checkpoints used 250 and 71 labels. The final 46 errors met the 20% target. |
| [ARC-Challenge](arc-challenge-audit.md) | 1,081 | 200 | The lowest-cost plan met the 20% target after one request. The smallest-cap plan used 250 labels. |

![Range of possible final error counts in the WinoGrande audit after each block of grades.](../figures/winogrande-audit.svg)

Run `python scripts/study_winogrande.py replay`, `python scripts/study_hellaswag.py replay` or `python scripts/study_arc.py replay` on the bundled predictions. Run `python scripts/study_batch_audit.py replay` for ANLI R2 and its BoolQ control. The UCI replay, `python scripts/study_letter_audit.py replay`, downloads public reference labels. [Audit handoff](../guide/queue.md) · [Archived research, comparisons and reproduction scripts](https://github.com/Daveyuwang/caat/tree/v1.1.2/docs/research)
